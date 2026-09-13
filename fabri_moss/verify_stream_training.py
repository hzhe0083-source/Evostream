"""GPU verification probe for stream replay multi-rate sequence training across 4 layouts.

Verifies:
1. CUDA availability and FlashAttention-2 assertion.
2. Load policy via runtime.load_native_checkpoint (trainable=True).
3. Wrap in NativeSequencePolicy (shallow=6, use_timestamps=True, checkpoint=True).
4. Assert all policy parameters FP32 on target CUDA device with requires_grad=True,
   and verify zero added parameters in wrapper (wrapper params == policy params).
5. Dataset scan over first 512 segments with stream_replay_v1 on epoch 1, selecting
   the segment with maximum summed group observations in each of the 4 layout categories:
   - dense
   - current_only
   - stream stride 2
   - stream stride 4
6. Create AdamW optimizer with create_native_optimizer_and_scheduler (FP32, warmup=0, total_steps=4).
7. Execute each of the 4 cases independently:
   - zero_grad(set_to_none=True)
   - forward pass under autocast (action_head outside autocast handled by policy)
   - capture extract_feature output with retain_grad in try/finally (assert 1 tensor per sample)
   - backward pass
   - check loss is finite
   - check all 4 parameter groups (vision, projector, llm, head) have grad norm > 0 and finite
   - check first historical pool frame visual grad is positive when history frames exist
   - check representative parameter max delta > 0 after step()
   - check all optimizer moment states (exp_avg, exp_avg_sq) are finite FP32
   - record timing, peak memory, loss, gradients, deltas
8. Write output_dir/report.json after all cases pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.native_training import NativeSequencePolicy
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint
from fabri_moss.stream_protocol import compute_stream_replay_v1_layout
from fabri_moss.train_native import classify_parameter, create_native_optimizer_and_scheduler


def classify_layout_category(layout_info: Dict[str, Any]) -> Optional[str]:
    """Identify category: 'dense', 'current_only', 'stream_stride2', 'stream_stride4'."""
    sl = layout_info.get("stream_layout", {})
    mode = sl.get("mode")
    if mode == "dense":
        return "dense"
    elif mode == "current_only":
        return "current_only"
    elif mode == "stream":
        stride = sl.get("query_stride")
        if stride == 2:
            return "stream_stride2"
        elif stride == 4:
            return "stream_stride4"
    return None


def select_four_layout_segments(dataset: NativeTrainingDataset, max_scan: int = 512) -> Dict[str, int]:
    """Scan segment metadata without image decode, pick max summed group observations per category."""
    n_scan = min(len(dataset), max_scan)
    best_segments: Dict[str, Tuple[int, int]] = {}  # category -> (seg_idx, sum_group_obs)

    for seg_idx in range(n_scan):
        ep_idx, start_r, end_r = dataset.segments[seg_idx]
        ep = dataset._episode_by_id[ep_idx]
        df = dataset._get_episode_dataframe(ep)
        all_timestamps, _ = dataset._validate_and_get_timestamps(df, ep_idx)

        layout = compute_stream_replay_v1_layout(
            seed=dataset.seed,
            epoch=dataset.epoch,
            ep_idx=ep_idx,
            target_start_row=start_r,
            target_end_row=end_r,
            all_timestamps=all_timestamps,
            history_frames=dataset.history_frames,
            split=dataset.split,
        )
        cat = classify_layout_category(layout)
        if cat is None:
            continue

        sum_obs = sum(len(g["observation_indices"]) for g in layout["replay_groups"])
        if cat not in best_segments or sum_obs > best_segments[cat][1]:
            best_segments[cat] = (seg_idx, sum_obs)

    required_cats = ("dense", "current_only", "stream_stride2", "stream_stride4")
    missing = [c for c in required_cats if c not in best_segments]
    if missing:
        raise RuntimeError(f"Failed to find segments for categories {missing} in first {n_scan} segments.")

    return {c: best_segments[c][0] for c in required_cats}


def run_probe(args: argparse.Namespace) -> None:
    """Execute full GPU verification probe across 4 stream replay layouts."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verify_stream_training.py, but CUDA is unavailable.")

    target_device = torch.device(args.device)
    if target_device.type != "cuda":
        raise ValueError(f"Target device must be CUDA, got {args.device!r}")

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Output directory {out_dir} exists and is not empty. Must be new/empty.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Checking FlashAttention-2 and loading native checkpoint on {args.device}...")
    torch.cuda.set_device(target_device)
    policy, checkpoint_config, norm_stats, source_metadata = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm_path,
        device=args.device,
        arm_key=args.arm_key,
        trainable=True,
    )

    fa2_diagnostics = assert_native_fa2(policy)

    # Check all policy parameters: FP32, requires_grad=True, on target device
    policy_params = list(policy.parameters())
    for name, p in policy.named_parameters():
        if p.dtype != torch.float32:
            raise AssertionError(f"Parameter {name} is {p.dtype}, expected torch.float32")
        if not p.requires_grad:
            raise AssertionError(f"Parameter {name} has requires_grad=False")
        if p.device.type != target_device.type or (target_device.index is not None and p.device.index != target_device.index):
            raise AssertionError(f"Parameter {name} on {p.device}, expected {target_device}")

    # Wrap in NativeSequencePolicy
    seq_policy = NativeSequencePolicy(
        policy=policy,
        shallow_layer=6,
        use_timestamps=True,
        gradient_checkpointing=True,
    )
    seq_policy.train()

    # Zero added parameters check
    wrapper_params = list(seq_policy.parameters())
    if len(wrapper_params) != len(policy_params) or {id(p) for p in wrapper_params} != {id(p) for p in policy_params}:
        raise AssertionError("NativeSequencePolicy added new parameters or parameter IDs do not match policy!")

    print("[2/5] Initializing dataset and scanning first 512 segments for 4 layout categories...")
    dataset = NativeTrainingDataset(
        root=args.data_root,
        norm_stats=norm_stats,
        history_frames=16,
        target_frames=8,
        split="train",
        max_episodes=args.max_episodes,
        augmentation=False,
        stream_protocol="stream_replay_v1",
    )
    dataset.set_epoch(1)

    selected_seg_indices = select_four_layout_segments(dataset, max_scan=512)
    print(f"Selected segment indices for 4 layouts: {selected_seg_indices}")

    print("[3/5] Creating optimizer (warmup_steps=0, total_steps=4)...")
    optimizer, _ = create_native_optimizer_and_scheduler(policy, warmup_steps=0, total_steps=4)

    embedder_model = getattr(getattr(policy, "embedder", None), "model", None)
    if embedder_model is None:
        raise AttributeError("policy.embedder.model is required for feature hook")

    report: Dict[str, Any] = {
        "status": "in_progress",
        "device": str(target_device),
        "source_metadata": source_metadata,
        "fa2": fa2_diagnostics,
        "cases": {},
    }

    print("[4/5] Executing independent forward/backward/step probe across 4 layouts...")
    cases_order = ("dense", "current_only", "stream_stride2", "stream_stride4")

    for case_idx, case_name in enumerate(cases_order):
        seg_idx = selected_seg_indices[case_name]
        raw_sample = dataset[seg_idx]

        # Move tensors to target device
        sample_gpu: Dict[str, Any] = {}
        for k, v in raw_sample.items():
            if isinstance(v, torch.Tensor):
                sample_gpu[k] = v.to(target_device)
            else:
                sample_gpu[k] = v

        obs_len = len(sample_gpu["images_window"])
        target_indices = sample_gpu["target_indices"]
        M = len(target_indices)
        replay_groups = sample_gpu.get("replay_groups", [])
        stream_layout = sample_gpu.get("stream_layout", {})

        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
        optimizer.zero_grad(set_to_none=True)

        # Hook extract_feature
        orig_extract_feature = embedder_model.extract_feature
        captured_vit_tensors: List[torch.Tensor] = []

        def wrapped_extract_feature(*w_args: Any, **w_kwargs: Any) -> Any:
            out = orig_extract_feature(*w_args, **w_kwargs)
            if isinstance(out, torch.Tensor):
                out.retain_grad()
                captured_vit_tensors.append(out)
            return out

        embedder_model.extract_feature = wrapped_extract_feature

        try:
            t0 = time.perf_counter()
            out_dict = seq_policy.forward(sample_gpu)
            loss = out_dict["loss"]
            if not torch.isfinite(loss):
                raise AssertionError(f"Case {case_name} produced non-finite loss: {loss.item()}")

            loss.backward()
            torch.cuda.synchronize(target_device)
            step_time_sec = time.perf_counter() - t0
        finally:
            embedder_model.extract_feature = orig_extract_feature

        if len(captured_vit_tensors) != 1:
            raise AssertionError(f"Expected exactly 1 visual extraction call, got {len(captured_vit_tensors)}")

        # Check parameter group gradients
        group_norms: Dict[str, float] = {}
        rep_params: Dict[str, Tuple[str, nn.Parameter, torch.Tensor]] = {}

        for name, p in policy.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise AssertionError(f"Parameter {name} has non-finite gradients!")
            if p.grad.dtype != torch.float32:
                raise AssertionError(f"Parameter {name} grad is not float32!")

            grp = classify_parameter(name)
            g_norm = float(p.grad.norm().item())
            if grp not in group_norms:
                group_norms[grp] = 0.0
            group_norms[grp] = max(group_norms[grp], g_norm)

            if grp not in rep_params and g_norm > 0.0:
                rep_params[grp] = (name, p, p.detach().clone())

        for req_grp in ("vision", "projector", "llm", "head"):
            if req_grp not in group_norms or group_norms[req_grp] <= 0.0:
                raise AssertionError(f"Group {req_grp} has zero or missing gradient norm in {case_name}")

        # Check earliest historical pool frame visual gradient
        history_frames_exist = (obs_len > M) or any(r not in set(target_indices) for r in range(obs_len))
        vit_grad = captured_vit_tensors[0].grad
        if vit_grad is None:
            raise AssertionError(f"Failed to capture visual embedding grad in {case_name}")

        pool_first_grad_norm = float(vit_grad[0].norm().item())
        if history_frames_exist and 0 not in set(target_indices):
            if pool_first_grad_norm <= 0.0 or not math.isfinite(pool_first_grad_norm):
                raise AssertionError(f"Historical frame 0 has zero or non-finite gradient in {case_name}")

        # Optimizer step and check parameter update deltas
        torch.cuda.synchronize(target_device)
        optimizer_start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(target_device)
        optimizer_secs = time.perf_counter() - optimizer_start

        # Check optimizer state moments
        for p in policy.parameters():
            if p.grad is not None:
                st = optimizer.state.get(p, {})
                for moment_key in ("exp_avg", "exp_avg_sq"):
                    if moment_key in st:
                        m_tensor = st[moment_key]
                        if m_tensor.dtype != torch.float32 or not torch.isfinite(m_tensor).all():
                            raise AssertionError(f"Optimizer moment {moment_key} not finite FP32")

        max_deltas: Dict[str, float] = {}
        for grp in ("vision", "projector", "llm", "head"):
            p_name, param, old_val = rep_params[grp]
            delta = float((param.detach() - old_val).abs().max().item())
            if delta <= 0.0 or not math.isfinite(delta):
                raise AssertionError(f"Parameter {p_name} in group {grp} failed to update! delta={delta}")
            max_deltas[grp] = delta

        peak_mib = torch.cuda.max_memory_allocated(target_device) / (1024 * 1024)

        report["cases"][case_name] = {
            "segment_index": seg_idx,
            "target_count": M,
            "total_images": obs_len,
            "num_replay_groups": len(replay_groups),
            "group_frame_counts": [len(g["observation_indices"]) for g in replay_groups],
            "group_spans_seconds": [g.get("span_seconds", 0.0) for g in stream_layout.get("groups", [])],
            "loss": float(loss.item()),
            "group_max_grad_norms": group_norms,
            "frame_0_grad_norm": pool_first_grad_norm,
            "representative_max_deltas": max_deltas,
            "forward_backward_seconds": step_time_sec,
            "optimizer_step_seconds": optimizer_secs,
            "peak_memory_mib": peak_mib,
        }
        print(f"Case [{case_idx + 1}/4] '{case_name}' PASSED - loss={loss.item():.4f}, peak={peak_mib:.1f}MiB, time={step_time_sec:.2f}s")

    print("[5/5] All 4 cases PASSED. Saving verification report...")
    report["status"] = "PASSED"
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {report_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify stream replay sequence training on GPU across 4 layouts.")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to FabriVLA repository")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Native checkpoint path")
    parser.add_argument("--vlm-path", type=str, default="/root/models/InternVL3_5-1B", help="VLM backbone path")
    parser.add_argument("--data-root", type=str, default="/root/evo1_metaworld_dataset", help="MetaWorld dataset root")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device")
    parser.add_argument("--output-dir", type=str, required=True, help="Empty output directory for verification artifacts")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Robot arm key")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional max episodes limit for dataset loading")

    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
