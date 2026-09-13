#!/usr/bin/env python3
"""Native GPU verification probe for compact memory training and inference."""
from __future__ import annotations
import argparse, copy, json, math, os, sys, threading, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fabri_moss import compact_training
from fabri_moss.compact_cache import NativeCompactMemoryCacheAdapter, validate_compact_memory
from fabri_moss.compact_memory import CompactMemoryConfig
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.compact_training import NativeCompactMemorySequencePolicy
from fabri_moss.native_cache import NativeCacheConfig
from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.runtime import assert_native_fa2, compute_file_sha256, load_native_checkpoint
from fabri_moss.train_native import classify_parameter
from fabri_moss.verify_memory_training import select_layout_segments


def write_report(out_dir: Path, data: Dict[str, Any]) -> None:
    tmp = out_dir / "report.json.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(out_dir / "report.json")


def inspect_model_keys_params(policy: nn.Module) -> Tuple[List[str], Dict[str, int], int]:
    keys = list(policy.state_dict().keys())
    counts = {g: 0 for g in ("vision", "projector", "llm", "head")}
    total = 0
    for name, p in policy.named_parameters():
        counts[classify_parameter(name)] += p.numel()
        total += p.numel()
    return keys, counts, total


def run_train_mode(args: argparse.Namespace, out_dir: Path, dev: torch.device, report: Dict[str, Any]) -> None:
    policy, _, norm_stats, src_meta = load_native_checkpoint(
        fabri_root=args.fabri_root, checkpoint_path=args.checkpoint, vlm_path=args.vlm_path,
        device=args.device, trainable=True,
    )
    assert_native_fa2(policy)
    keys, counts, total = inspect_model_keys_params(policy)
    for n, p in policy.named_parameters():
        if p.dtype != torch.float32 or not p.requires_grad:
            raise AssertionError(f"Param {n} must be FP32 requires_grad=True, got {p.dtype}, {p.requires_grad}")
    seq_policy = NativeCompactMemorySequencePolicy(
        policy=policy, shallow_layer=6, use_timestamps=True, gradient_checkpointing=True,
        compact_config=CompactMemoryConfig(),
    ).train()
    report.update({
        "actual_backend": src_meta.get("actual_backend"), "source_checkpoint_sha256": src_meta.get("checkpoint_sha256"),
        "parent_checkpoint_sha256": src_meta["checkpoint_sha256"],
        "step": src_meta.get("step"), "model_keys": keys, "param_counts": counts, "total_params": total,
        "device": str(dev), "dtype": "float32",
    })
    write_report(out_dir, report)

    ds = NativeTrainingDataset(
        root=args.data_root, norm_stats=norm_stats, history_frames=16, target_frames=8,
        split="train", augmentation=False, stream_protocol="compact_memory_replay_v1", seed=4042,
    )
    ds.set_epoch(args.epoch)
    selected = select_layout_segments(ds)
    report["selected_segments_notes"] = "v1.2 layout score used for selection, not compact exact heaviest"

    opt = torch.optim.AdamW(policy.parameters(), lr=1e-6)
    orig_render = compact_training.render_compact_memory
    cases = ("dense", "current_only", "memory", "memory_context")

    for cname in cases:
        seg_idx, _ = selected[cname]
        raw_sample = ds[seg_idx]
        batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in raw_sample.items()}
        rendered_target_lens: List[int] = []
        rendered_block_lens: List[int] = []

        def tracked_render(*r_args: Any, **r_kwargs: Any) -> Any:
            res = orig_render(*r_args, **r_kwargs)
            t_len = int(res.inputs_embeds.shape[1])
            c_len = int(t_len - res.current_start)
            rendered_target_lens.append(t_len)
            rendered_block_lens.append(c_len)
            return res

        compact_training.render_compact_memory = tracked_render
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        try:
            out = seq_policy.forward(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise AssertionError(f"Case {cname} loss not finite: {loss.item()}")
            loss.backward()
            torch.cuda.synchronize(dev)
            fwd_bwd_sec = time.perf_counter() - t0

            for n, p in policy.named_parameters():
                if p.dtype != torch.float32 or not p.requires_grad:
                    raise AssertionError(f"Case {cname} param {n} violated FP32/requires_grad")
                if p.grad is not None and (p.grad.dtype != torch.float32 or not torch.isfinite(p.grad).all()):
                    raise AssertionError(f"Case {cname} param {n} grad not finite FP32")

            grp_norms = {g: 0.0 for g in ("vision", "projector", "llm", "head")}
            rep_p: Dict[str, Tuple[str, nn.Parameter, torch.Tensor]] = {}
            for n, p in policy.named_parameters():
                if p.grad is not None:
                    g = classify_parameter(n)
                    gn = float(p.grad.norm().item())
                    grp_norms[g] = max(grp_norms[g], gn)
                    if g not in rep_p and gn > 0.0:
                        rep_p[g] = (n, p, p.detach().clone())
            for g in grp_norms:
                if grp_norms[g] <= 0.0 or not math.isfinite(grp_norms[g]):
                    raise AssertionError(f"Case {cname} group {g} grad norm is zero")

            t_opt = time.perf_counter()
            opt.step()
            torch.cuda.synchronize(dev)
            opt_sec = time.perf_counter() - t_opt

            deltas: Dict[str, float] = {}
            for g in ("vision", "projector", "llm", "head"):
                pn, p_ref, old_val = rep_p[g]
                d = float((p_ref.detach() - old_val).abs().max().item())
                if d <= 0.0 or not math.isfinite(d):
                    raise AssertionError(f"Case {cname} param {pn} delta not positive: {d}")
                deltas[g] = d

            for n, p in policy.named_parameters():
                if p.grad is not None:
                    st = opt.state.get(p, {})
                    for m in ("exp_avg", "exp_avg_sq"):
                        if m not in st or st[m].dtype != torch.float32 or not torch.isfinite(st[m]).all():
                            raise AssertionError(f"Case {cname} param {n} moment {m} not finite FP32")

            if raw_sample.get("memory_replay", False):
                if len(rendered_block_lens) != len(raw_sample["target_indices"]):
                    raise AssertionError("Missing target materialization in compact probe")
                if any(bl != 1024 for bl in rendered_block_lens):
                    raise AssertionError(f"Case {cname} current blocks not 1024: {rendered_block_lens}")
                if not all(isinstance(x, int) and type(x) is int for x in rendered_target_lens + rendered_block_lens):
                    raise AssertionError(f"Case {cname} rendered lengths must be only ints")

            peak_mib = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
            report["cases"][cname] = {
                "segment_index": seg_idx, "observations": len(raw_sample["frame_ids"]),
                "targets": len(raw_sample["target_indices"]),
                "fwd_bwd_sec": fwd_bwd_sec, "optimizer_sec": opt_sec, "total_sec": fwd_bwd_sec + opt_sec,
                "peak_gpu_mib": peak_mib, "loss": float(loss.item()), "grad_norms": grp_norms,
                "max_deltas": deltas, "rendered_target_lens": rendered_target_lens,
                "rendered_block_lens": rendered_block_lens,
            }
            write_report(out_dir, report)
            print(f"TRAIN {cname}: loss={loss.item():.6f}, peak={peak_mib:.1f} MiB", flush=True)
            del out, loss, rep_p, old_val
        finally:
            compact_training.render_compact_memory = orig_render
            opt.zero_grad(set_to_none=True)
            del batch, raw_sample


@torch.no_grad()
def run_inference_mode(args: argparse.Namespace, out_dir: Path, dev: torch.device, report: Dict[str, Any]) -> None:
    policy, _, norm_stats, src_meta = load_native_checkpoint(
        fabri_root=args.fabri_root, checkpoint_path=args.checkpoint, vlm_path=args.vlm_path,
        device=args.device, trainable=False,
    )
    assert_native_fa2(policy)
    policy.eval()
    keys, counts, total = inspect_model_keys_params(policy)
    report.update({
        "actual_backend": src_meta.get("actual_backend"), "source_checkpoint_sha256": src_meta.get("checkpoint_sha256"),
        "parent_checkpoint_sha256": src_meta["checkpoint_sha256"],
        "step": src_meta.get("step"), "model_keys": keys, "param_counts": counts, "total_params": total,
        "device": str(dev), "dtype": "bfloat16_vlm_fp32_head",
    })
    write_report(out_dir, report)

    ds = NativeTrainingDataset(
        root=args.data_root, norm_stats=norm_stats, history_frames=16, target_frames=8,
        split="train", augmentation=False, stream_protocol="compact_memory_replay_v1", seed=4042,
    )
    ds.set_epoch(args.epoch)
    selected = select_layout_segments(ds)
    seg_idx, _ = selected["memory"]
    raw_sample = ds[seg_idx]

    # Select last 24 observations to ensure final observation coincides with sample's final target
    N = len(raw_sample["frame_ids"])
    if N < 24:
        raise AssertionError(f"Sample has N={N} < 24 frames")
    start_obs = N - 24
    obs_slice = range(start_obs, N)
    frame_ids = [raw_sample["frame_ids"][i] for i in obs_slice]
    obs_times = [raw_sample["observation_times"][i] for i in obs_slice]
    images_24 = [raw_sample["images_window"][i] for i in obs_slice]
    prompt = raw_sample["prompt"]

    cache_cfg = NativeCacheConfig(max_frames=16, shallow_layer=6, use_timestamps=True)
    compact_cfg = CompactMemoryConfig()
    ad_sync = NativeCompactMemoryCacheAdapter(policy, config=cache_cfg, compact_config=compact_cfg, background_rebuild=False)
    ad_async = NativeCompactMemoryCacheAdapter(policy, config=cache_cfg, compact_config=compact_cfg, background_rebuild=True)

    worker_stream_verified = False
    worker_stream_errors: List[str] = []
    main_th = threading.get_ident()
    orig_exec = ad_async._execute_native_layers

    def hooked_exec(*e_args: Any, **e_kwargs: Any) -> Any:
        nonlocal worker_stream_verified
        if threading.get_ident() != main_th:
            act_s = torch.cuda.current_stream(dev)
            if ad_async._cuda_stream is not None and act_s != ad_async._cuda_stream:
                worker_stream_errors.append(f"Worker stream {act_s} != {ad_async._cuda_stream}")
            worker_stream_verified = True
        return orig_exec(*e_args, **e_kwargs)

    ad_async._execute_native_layers = hooked_exec
    snapshots = ([0, 8], [8, 16], [16, 24])

    try:
        # Encode all 24 frames identically
        b_sync = [ad_sync.encode_frame(images_24[i], frame_ids[i], prompt, observation_time=obs_times[i]) for i in range(24)]
        b_async = [ad_async.encode_frame(images_24[i], frame_ids[i], prompt, observation_time=obs_times[i]) for i in range(24)]

        st_s, st_a = None, None
        cases_res: Dict[str, Any] = {}
        for s_idx, (s_start, s_end) in enumerate(snapshots):
            chunk_s, chunk_a = b_sync[s_start:s_end], b_async[s_start:s_end]
            torch.cuda.reset_peak_memory_stats(dev)
            t0_s = time.perf_counter()
            prev_s = st_s
            d_s, sh_s, st_s = ad_sync.read_blocks(chunk_s, prompt=prompt, previous=st_s)
            validate_compact_memory(st_s, previous=prev_s, snapshot=chunk_s, prompt=prompt)
            if st_s.pending is not None:
                st_s.pending.future.result()
            torch.cuda.synchronize(dev)
            dur_s = time.perf_counter() - t0_s
            peak_s = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

            torch.cuda.reset_peak_memory_stats(dev)
            t0_a = time.perf_counter()
            prev_a = st_a
            d_a, sh_a, st_a = ad_async.read_blocks(chunk_a, prompt=prompt, previous=st_a)
            validate_compact_memory(st_a, previous=prev_a, snapshot=chunk_a, prompt=prompt)
            if st_a.pending is not None:
                st_a.pending.future.result()
            torch.cuda.synchronize(dev)
            dur_a = time.perf_counter() - t0_a
            peak_a = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

            for tag, tens in (("deep_s", d_s), ("deep_a", d_a), ("shallow_s", sh_s), ("shallow_a", sh_a)):
                if tens.shape != (1, 1024, 1024) or not torch.isfinite(tens).all():
                    raise AssertionError(f"Snapshot {s_idx} {tag} invalid: shape {tens.shape}")

            if not torch.equal(d_s, d_a):
                diff_d = (d_s - d_a).abs().max().item()
                raise AssertionError(f"Snapshot {s_idx} deep features mismatch! max diff={diff_d}")
            if not torch.equal(sh_s, sh_a):
                diff_sh = (sh_s - sh_a).abs().max().item()
                raise AssertionError(f"Snapshot {s_idx} shallow features mismatch! max diff={diff_sh}")

            for l_idx, ((k_s, v_s), (k_a, v_a)) in enumerate(zip(st_s.layer_kv, st_a.layer_kv)):
                if not torch.equal(k_s, k_a) or not torch.equal(v_s, v_a):
                    raise AssertionError(f"Snapshot {s_idx} layer {l_idx} KV mismatch between sync and async")

            cases_res[f"snapshot_{s_idx}"] = {
                "sync_sec": dur_s, "async_sec": dur_a, "sync_peak_mib": peak_s, "async_peak_mib": peak_a,
                "deep_shape": list(d_s.shape), "shallow_shape": list(sh_s.shape),
                "state_physical_tokens": int(st_s.attention_mask.shape[1]), "state_kv_nbytes": int(st_s.kv_nbytes),
            }
            report["cases"] = cases_res
            write_report(out_dir, report)

        if st_a.pending is not None:
            _ = st_a.pending.future.result()

        if len(worker_stream_errors) > 0 or not worker_stream_verified:
            raise AssertionError(f"Worker stream check failed! errors={worker_stream_errors}, verified={worker_stream_verified}")
        report["worker_stream_verified"] = True

        # Anchors exact check: 8th and 16th frames (indices 7 and 15)
        for anchor_idx, frame_pos in enumerate((7, 15)):
            target_fid = frame_ids[frame_pos]
            final_memory = st_s.pending.future.result().memory if st_s.pending is not None else st_s.memory
            matching = [a for a in final_memory.anchors if a.frame_id == target_fid]
            if len(matching) != 1:
                raise AssertionError(f"Anchor for frame {target_fid} not found in anchors")
            a_tokens = matching[0].visual_tokens
            if a_tokens.shape != (1, 256, 1024):
                raise AssertionError(f"Anchor visual tokens shape {a_tokens.shape} != (1, 256, 1024)")
            if not torch.equal(a_tokens, b_sync[frame_pos].visual_tokens):
                raise AssertionError(f"Anchor visual tokens content mismatch for frame {target_fid}")
        report["frame_ids"] = frame_ids
        report["observation_times"] = obs_times
        report["anchors_verified"] = "8th and 16th frames preserved full 256 visual tokens with torch.equal"

        # Action sampling verification on final snapshot
        st_dev = raw_sample["state"][-1:].to(dev)
        st_mask_dev = raw_sample["state_mask"][-1:].to(dev)
        act_mask_dev = raw_sample["action_mask"][-1:].to(dev)

        if policy.action_head.config.num_inference_timesteps != 50:
            raise AssertionError("Original 50-step action sampling required")
        torch.manual_seed(4042)
        act_sync = policy.action_head.sample(d_s, state=st_dev, state_mask=st_mask_dev, action_mask=act_mask_dev, shallow_tokens=sh_s)
        torch.manual_seed(4042)
        act_async = policy.action_head.sample(d_a, state=st_dev, state_mask=st_mask_dev, action_mask=act_mask_dev, shallow_tokens=sh_a)

        if not torch.isfinite(act_sync).all() or not torch.isfinite(act_async).all():
            raise AssertionError("Action sampling generated non-finite actions")
        if not torch.equal(act_sync, act_async):
            d_act = (act_sync - act_async).abs().max().item()
            raise AssertionError(f"Action sampling mismatch between sync and async! max diff={d_act}")
        report["action_sampling"] = {
            "shape": list(act_sync.shape), "exact_equal": True, "note": "Measured latency only, no speedup multiple claimed",
        }
        write_report(out_dir, report)
    finally:
        ad_sync.close()
        ad_async.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Compact Memory Verification Probe")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output-dir", required=True, help="Target empty output directory")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Verification mode")
    parser.add_argument("--device", default="cuda:0", help="Target CUDA device")
    parser.add_argument("--epoch", type=int, default=5, help="Dataset epoch (default: 5)")
    parser.add_argument("--fabri-root", default="/root/FabriVLA", help="Path to FabriVLA repo")
    parser.add_argument("--vlm-path", default="/root/models/InternVL3_5-1B", help="Path to InternVL model")
    parser.add_argument("--data-root", default="/root/evo1_metaworld_dataset", help="Path to dataset root")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available to run verify_compact_native.py")
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise ValueError(f"Target device must be CUDA, got {args.device}")

    torch.cuda.set_device(dev)
    torch.set_num_threads(2)
    if args.epoch < 0:
        raise ValueError("epoch must be non-negative")
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Output directory {out_dir} exists and is not empty.")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "status": "in_progress", "mode": args.mode, "device": args.device,
        "probe_sha256": compute_file_sha256(Path(__file__).resolve()),
        "compact_protocol_contract": get_compact_protocol_contract(),
        "cases": {}, "utility_claim": False, "speedup_claim": False,
        "probe_weights_saved": False,
        "inference_precision_note": "BF16 VLM is the existing deployment cast of FP32 parent weights; not FP32 parity",
    }
    write_report(out_dir, report)

    try:
        if args.mode == "train":
            run_train_mode(args, out_dir, dev, report)
        else:
            run_inference_mode(args, out_dir, dev, report)
        report["status"] = "PASSED"
        write_report(out_dir, report)
        print("Verification PASSED.")
    except Exception as exc:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        write_report(out_dir, report)
        raise exc


if __name__ == "__main__":
    main()
