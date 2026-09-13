#!/usr/bin/env python3
"""Integration and acceptance verification script for compact intermediate memory and temporal RoPE.

Evaluates:
1. Deterministic seed, real tiny Qwen3 model config inspection, float32 precision,
   explicit eager attention backend, zero added parameters.
2. Single-frame parity between old NativeMemoryCacheAdapter and new NativeCompactMemoryCacheAdapter
   (intermediate_grid=None, strength=0.0) -> exact deep/shallow/KV match;
   temporal strength=1.0 non-zero time approximate match (tol <= 1e-5).
3. 24-frame multi-decision sequence parity across sync and async adapters with
   snapshots ending at decisions [3, 8, 17, 23], sparse frames/times, intermediate_grid=1 (4->1 pooled),
   no upsampling, anchors preserve full 4 visual tokens, and eval no_grad deep/shallow alignment
   with NativeCompactMemorySequencePolicy.
4. Backward pass on 2 targets, finite loss, non-zero gradients on 4 parameter groups
   (vision, projector, LLM, action head), and zero gradient on retired cold frames.
5. Wall-clock timing for sync vs async synthetic workloads, CUDA synchronization and peak memory
   tracking when running on CUDA, with explicit contract provenance fields.
6. JSON report written to output directory, PASSED status only when all assertions succeed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any, Dict, List, Sequence, Tuple


def _embedder_preprocess_images_bound(self: Any, images: Sequence[Any]) -> Tuple[Any, List[int]]:
    """MethodType-bound preprocess ensuring CPU-created tensors match model device and dtype."""
    device = getattr(self, "device", next(self.model.parameters()).device)
    dtype = next(self.model.parameters()).dtype
    pixel_values, tiles = self._preprocess_images_on_cpu(images)
    return pixel_values.to(device=device, dtype=dtype), tiles


def _embedder_prepare_and_fuse_embeddings_bound(
    self: Any,
    prompt: str,
    vit_embeds: Any,
    image_mask: Any,
    num_tiles_list: List[int],
) -> Tuple[Any, Any]:
    """MethodType-bound prepare and fuse calling embedder batch fusion."""
    return self._prepare_batch_and_fuse_embeddings(
        prompts=[prompt],
        vit_embeds_batch=[vit_embeds],
        image_masks=[image_mask],
        batch_num_tiles_list=[num_tiles_list],
    )


def _patch_embedder_for_single_frame(policy: Any) -> None:
    """Equip test fixture embedder with single-frame encode methods via MethodType."""
    embedder = policy.embedder
    embedder._preprocess_images = types.MethodType(_embedder_preprocess_images_bound, embedder)
    embedder._prepare_and_fuse_embeddings = types.MethodType(
        _embedder_prepare_and_fuse_embeddings_bound, embedder
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify compact intermediate memory and temporal RoPE on real tiny Qwen3 policy."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run verification on: 'cpu' (default) or 'cuda:0'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save report.json. Must not be non-empty.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    if out_path.exists():
        if not out_path.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {out_path}")
        if any(out_path.iterdir()):
            raise ValueError(f"Output directory {out_path} must be empty.")
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from fabri_moss.compact_cache import NativeCompactMemoryCacheAdapter, validate_compact_memory
        from fabri_moss.compact_memory import CompactMemoryConfig
        from fabri_moss.compact_training import NativeCompactMemorySequencePolicy
        from fabri_moss.memory_cache import NativeMemoryCacheAdapter
        from fabri_moss.native_cache import NativeCacheConfig
        from fabri_moss.periodic_memory import PeriodicMemoryConfig
        from fabri_moss.temporal_rope import TemporalRoPEConfig
        from fabri_moss.tests.test_memory_training import MockTokenizerForMemory
        from fabri_moss.tests.test_native_training import make_sample, make_tiny_training_policy
    except ImportError as e:
        sys.stderr.write(f"Missing dependency or package module: {e}\n")
        return 1

    device_str = args.device
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was specified but torch.cuda is not available.")

    report: Dict[str, Any] = {
        "status": "FAILED",
        "device": device_str,
        "attention_backend": "eager",
        "synthetic_tiny": True,
        "real_fabrivla_weights_tested": False,
        "utility_claim": False,
    }

    adapters_to_close: List[Any] = []
    verification_passed = False

    try:
        # 1. Deterministic seed, real tiny Qwen3 model config inspection, float32, eager, param counts
        seed = 42
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        base_seq_policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
        lm_model = base_seq_policy.policy.embedder.model.language_model
        core = getattr(lm_model, "model", lm_model)
        actual_layers = len(core.layers)
        attn_impl = getattr(core.config, "_attn_implementation", "eager")
        assert attn_impl == "eager", f"Expected eager attention, got {attn_impl}"

        base_seq_policy = base_seq_policy.to(device=device, dtype=torch.float32)
        policy_template = base_seq_policy.policy
        policy_template.embedder.device = device
        policy_template.embedder.tokenizer = MockTokenizerForMemory(policy_template.embedder.tokenizer)
        _patch_embedder_for_single_frame(policy_template)

        for p in policy_template.parameters():
            assert p.dtype == torch.float32, f"Parameter {p} is not float32"
            assert p.device == device, f"Parameter {p} not on {device}"

        cache_cfg = NativeCacheConfig(shallow_layer=1, use_timestamps=True)
        mem_cfg = PeriodicMemoryConfig(
            recent_frames=4,
            consolidate_every=4,
            memory_slots=2,
            spatial_grid=1,
            protect_decision_frames=True,
        )
        compact_cfg_s0 = CompactMemoryConfig(
            memory=mem_cfg,
            intermediate_grid=None,
            temporal=TemporalRoPEConfig(strength=0.0),
        )

        fresh_p_adapter = copy.deepcopy(policy_template)
        fresh_p_adapter.embedder.device = device
        test_adapter = NativeCompactMemoryCacheAdapter(
            fresh_p_adapter,
            config=cache_cfg,
            compact_config=compact_cfg_s0,
        )
        adapters_to_close.append(test_adapter)

        assert len(list(test_adapter.parameters())) == len(list(policy_template.parameters()))

        fresh_p_seq = copy.deepcopy(policy_template)
        fresh_p_seq.embedder.device = device
        test_seq_policy = NativeCompactMemorySequencePolicy(
            fresh_p_seq,
            shallow_layer=1,
            compact_config=compact_cfg_s0,
            gradient_checkpointing=False,
        )
        assert len(list(test_seq_policy.parameters())) == len(list(policy_template.parameters()))

        report["model_inspection"] = {
            "num_layers": actual_layers,
            "hidden_size": core.config.hidden_size,
            "num_heads": core.config.num_attention_heads,
            "attention_implementation": attn_impl,
            "total_parameters": sum(p.numel() for p in policy_template.parameters()),
        }

        # 2. Single-frame deep/shallow/KV equality and temporal RoPE tol
        prompt = "pick up cube"
        gen = torch.Generator(device="cpu").manual_seed(101)
        img_single_cpu = torch.randn(4, 16, generator=gen)
        img_single = img_single_cpu.to(device=device, dtype=torch.float32)

        p_old = copy.deepcopy(policy_template)
        p_old.embedder.device = device
        old_adapter = NativeMemoryCacheAdapter(
            p_old,
            config=cache_cfg,
            memory_config=mem_cfg,
        )
        adapters_to_close.append(old_adapter)

        p_new_s0 = copy.deepcopy(policy_template)
        p_new_s0.embedder.device = device
        new_adapter_s0 = NativeCompactMemoryCacheAdapter(
            p_new_s0,
            config=cache_cfg,
            compact_config=compact_cfg_s0,
            background_rebuild=False,
        )
        adapters_to_close.append(new_adapter_s0)

        b_old = old_adapter.encode_frame([img_single], frame_id=0, prompt=prompt, observation_time=0.0)
        b_new_s0 = new_adapter_s0.encode_frame([img_single], frame_id=0, prompt=prompt, observation_time=0.0)

        d_old, sh_old, state_old = old_adapter.read_blocks([b_old], prompt=prompt)
        d_new_s0, sh_new_s0, state_new_s0 = new_adapter_s0.read_blocks([b_new_s0], prompt=prompt)

        deep_diff_s0 = (d_old - d_new_s0).abs().max().item()
        shallow_diff_s0 = (sh_old - sh_new_s0).abs().max().item()
        assert deep_diff_s0 == 0.0, f"Single-frame deep mismatch with s=0: {deep_diff_s0}"
        assert shallow_diff_s0 == 0.0, f"Single-frame shallow mismatch with s=0: {shallow_diff_s0}"
        for idx, ((k_o, v_o), (k_n, v_n)) in enumerate(zip(state_old.layer_kv, state_new_s0.layer_kv)):
            assert (k_o - k_n).abs().max().item() == 0.0, f"Layer {idx} K mismatch with s=0"
            assert (v_o - v_n).abs().max().item() == 0.0, f"Layer {idx} V mismatch with s=0"

        compact_cfg_s1 = CompactMemoryConfig(
            memory=mem_cfg,
            intermediate_grid=None,
            temporal=TemporalRoPEConfig(strength=1.0),
        )
        p_new_s1 = copy.deepcopy(policy_template)
        p_new_s1.embedder.device = device
        new_adapter_s1 = NativeCompactMemoryCacheAdapter(
            p_new_s1,
            config=cache_cfg,
            compact_config=compact_cfg_s1,
            background_rebuild=False,
        )
        adapters_to_close.append(new_adapter_s1)

        b_new_s1 = new_adapter_s1.encode_frame([img_single], frame_id=0, prompt=prompt, observation_time=2.5)
        b_new_s0_time = new_adapter_s0.encode_frame([img_single], frame_id=0, prompt=prompt, observation_time=2.5)

        d_new_s1, sh_new_s1, state_new_s1 = new_adapter_s1.read_blocks([b_new_s1], prompt=prompt)
        d_new_s0_t, sh_new_s0_t, _ = new_adapter_s0.read_blocks([b_new_s0_time], prompt=prompt)

        tol = 1e-5
        deep_diff_s1 = (d_new_s1 - d_new_s0_t).abs().max().item()
        shallow_diff_s1 = (sh_new_s1 - sh_new_s0_t).abs().max().item()
        assert deep_diff_s1 <= tol, f"Single-frame deep temporal diff {deep_diff_s1} exceeds tol {tol}"
        assert shallow_diff_s1 <= tol, f"Single-frame shallow temporal diff {shallow_diff_s1} exceeds tol {tol}"

        report["single_frame_verification"] = {
            "strength_0_deep_max_diff": deep_diff_s0,
            "strength_0_shallow_max_diff": shallow_diff_s0,
            "strength_1_deep_diff_vs_s0": deep_diff_s1,
            "strength_1_shallow_diff_vs_s0": shallow_diff_s1,
            "tolerance": tol,
        }

        # 3. 24-frame multi-decision sequence parity across sync and async adapters
        N = 24
        decision_indices = [3, 8, 17, 23]
        target_indices = [17, 23]
        snapshots = [
            list(range(0, 4)),
            list(range(4, 9)),
            list(range(9, 18)),
            list(range(18, 24)),
        ]

        frame_ids = [10 + 3 * i for i in range(N)]
        observation_times = [round(0.0 + 0.15 * i + 0.02 * (i % 3), 4) for i in range(N)]

        compact_cfg_grid1 = CompactMemoryConfig(
            memory=mem_cfg,
            intermediate_grid=1,  # 2x2=4 tokens -> 1x1=1 token, strictly no upsampling
            temporal=TemporalRoPEConfig(strength=1.0),
        )
        compact_cfg_nocompact = CompactMemoryConfig(
            memory=mem_cfg,
            intermediate_grid=None,
            temporal=TemporalRoPEConfig(strength=1.0),
        )

        sample = make_sample(N=N, target_indices=target_indices, horizon=2, action_dim=4)
        sample["frame_ids"] = frame_ids
        sample["observation_times"] = observation_times
        sample["memory_replay"] = True
        sample["decision_indices"] = decision_indices
        sample["images_window"] = [img.to(device=device, dtype=torch.float32) for img in sample["images_window"]]
        sample["actions"] = sample["actions"].to(device=device, dtype=torch.float32)
        sample["action_mask"] = sample["action_mask"].to(device=device)

        fresh_policy_sync = copy.deepcopy(policy_template)
        fresh_policy_sync.embedder.device = device
        fresh_policy_async = copy.deepcopy(policy_template)
        fresh_policy_async.embedder.device = device
        fresh_policy_nocompact = copy.deepcopy(policy_template)
        fresh_policy_nocompact.embedder.device = device
        fresh_policy_seq = copy.deepcopy(policy_template)
        fresh_policy_seq.embedder.device = device

        adapter_sync = NativeCompactMemoryCacheAdapter(
            fresh_policy_sync,
            config=cache_cfg,
            compact_config=compact_cfg_grid1,
            background_rebuild=False,
        )
        adapters_to_close.append(adapter_sync)

        adapter_async = NativeCompactMemoryCacheAdapter(
            fresh_policy_async,
            config=cache_cfg,
            compact_config=compact_cfg_grid1,
            background_rebuild=True,
        )
        adapters_to_close.append(adapter_async)

        adapter_nocompact = NativeCompactMemoryCacheAdapter(
            fresh_policy_nocompact,
            config=cache_cfg,
            compact_config=compact_cfg_nocompact,
            background_rebuild=False,
        )
        adapters_to_close.append(adapter_nocompact)

        # Hook CUDA worker stream and source inputs check if running on CUDA
        cuda_worker_stream_verified = False
        cuda_stream_errors: List[str] = []
        if device.type == "cuda":
            orig_exec = adapter_async._execute_native_layers
            main_thread_id = threading.get_ident()

            def _hooked_exec(*exec_args: Any, **exec_kwargs: Any) -> Any:
                nonlocal cuda_worker_stream_verified
                cur_thread = threading.get_ident()
                if cur_thread != main_thread_id:
                    expected_st = adapter_async._cuda_stream
                    actual_st = torch.cuda.current_stream(device)
                    if expected_st is not None and actual_st != expected_st:
                        cuda_stream_errors.append(f"Worker stream mismatch: {actual_st} vs {expected_st}")
                    inp = exec_kwargs.get("inputs_embeds")
                    if inp is not None and inp.device != device:
                        cuda_stream_errors.append(f"Worker input device {inp.device} != {device}")
                    cuda_worker_stream_verified = True
                return orig_exec(*exec_args, **exec_kwargs)

            adapter_async._execute_native_layers = _hooked_exec  # type: ignore[assignment]

        prompt_seq = sample["prompt"]
        sync_state = None
        async_state = None
        nocompact_state = None
        sync_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
        async_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
        snapshot_metrics: List[Dict[str, Any]] = []

        # 1. Independent baseline run for nocompact adapter (isolated timing)
        nocompact_stats: List[Dict[str, int]] = []
        for snap in snapshots:
            blocks_nc = [
                adapter_nocompact.encode_frame(
                    images=[sample["images_window"][i]],
                    frame_id=sample["frame_ids"][i],
                    prompt=prompt_seq,
                    observation_time=sample["observation_times"][i],
                )
                for i in snap
            ]
            _, _, nocompact_state = adapter_nocompact.read_blocks(
                blocks_nc, prompt=prompt_seq, previous=nocompact_state
            )
            nocompact_stats.append({
                "physical_tokens": int(nocompact_state.attention_mask.shape[1]),
                "kv_nbytes": int(nocompact_state.kv_nbytes),
                "nbytes": int(nocompact_state.nbytes),
            })

        # 2. Timed loop for sync adapter
        sync_stats: List[Dict[str, Any]] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0_sync = time.perf_counter()

        for snap_idx, snap in enumerate(snapshots):
            blocks_s = [
                adapter_sync.encode_frame(
                    images=[sample["images_window"][i]],
                    frame_id=sample["frame_ids"][i],
                    prompt=prompt_seq,
                    observation_time=sample["observation_times"][i],
                )
                for i in snap
            ]
            prev_s = sync_state
            d_s, sh_s, sync_state = adapter_sync.read_blocks(blocks_s, prompt=prompt_seq, previous=sync_state)
            validate_compact_memory(sync_state, previous=prev_s, snapshot=blocks_s, prompt=prompt_seq)
            sync_features.append((d_s, sh_s))

            sync_stats.append({
                "physical_tokens": int(sync_state.attention_mask.shape[1]),
                "kv_nbytes": int(sync_state.kv_nbytes),
                "nbytes": int(sync_state.nbytes),
                "current_block_length": int(blocks_s[-1].inputs_embeds.shape[1]),
                "token_times": sync_state.token_times.clone(),
                "attention_mask": sync_state.attention_mask.clone(),
                "layer_kv": tuple((k.clone(), v.clone()) for k, v in sync_state.layer_kv),
            })

        # Final rebuild resolution included in sync latency
        if sync_state.pending is not None:
            _ = sync_state.pending.future.result()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_sync_total = time.perf_counter() - t0_sync
        sync_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

        # 3. Timed loop for async adapter
        async_stats: List[Dict[str, Any]] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0_async = time.perf_counter()

        for snap_idx, snap in enumerate(snapshots):
            blocks_a = [
                adapter_async.encode_frame(
                    images=[sample["images_window"][i]],
                    frame_id=sample["frame_ids"][i],
                    prompt=prompt_seq,
                    observation_time=sample["observation_times"][i],
                )
                for i in snap
            ]
            prev_a = async_state
            d_a, sh_a, async_state = adapter_async.read_blocks(blocks_a, prompt=prompt_seq, previous=async_state)
            validate_compact_memory(async_state, previous=prev_a, snapshot=blocks_a, prompt=prompt_seq)
            async_features.append((d_a, sh_a))

            async_stats.append({
                "physical_tokens": int(async_state.attention_mask.shape[1]),
                "kv_nbytes": int(async_state.kv_nbytes),
                "nbytes": int(async_state.nbytes),
                "current_block_length": int(blocks_a[-1].inputs_embeds.shape[1]),
                "token_times": async_state.token_times.clone(),
                "attention_mask": async_state.attention_mask.clone(),
                "layer_kv": tuple((k.clone(), v.clone()) for k, v in async_state.layer_kv),
            })

        # Explicit final rebuild resolution included in async latency to avoid false speedup claims
        if async_state.pending is not None:
            _ = async_state.pending.future.result()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_async_total = time.perf_counter() - t0_async
        async_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

        if device.type == "cuda":
            assert len(cuda_stream_errors) == 0, f"CUDA stream errors: {cuda_stream_errors}"
            assert cuda_worker_stream_verified, "CUDA worker stream hook was never triggered"

        # 4. Compare per-snapshot statistics, verify exact parity and length reduction
        for snap_idx, snap in enumerate(snapshots):
            s_stat = sync_stats[snap_idx]
            a_stat = async_stats[snap_idx]
            nc_stat = nocompact_stats[snap_idx]

            # Parity check between sync and async snapshots
            assert s_stat["physical_tokens"] == a_stat["physical_tokens"], (
                f"Snapshot {snap_idx} length mismatch: sync={s_stat['physical_tokens']} vs async={a_stat['physical_tokens']}"
            )
            assert s_stat["kv_nbytes"] == a_stat["kv_nbytes"], (
                f"Snapshot {snap_idx} KV nbytes mismatch: sync={s_stat['kv_nbytes']} vs async={a_stat['kv_nbytes']}"
            )
            assert s_stat["current_block_length"] == a_stat["current_block_length"], (
                f"Snapshot {snap_idx} current block length mismatch"
            )
            assert torch.equal(s_stat["attention_mask"], a_stat["attention_mask"]), (
                f"Snapshot {snap_idx} attention_mask mismatch between sync and async"
            )
            assert torch.equal(s_stat["token_times"], a_stat["token_times"]), (
                f"Snapshot {snap_idx} token_times mismatch between sync and async"
            )
            for l_idx, ((k_s, v_s), (k_a, v_a)) in enumerate(zip(s_stat["layer_kv"], a_stat["layer_kv"])):
                assert torch.allclose(k_s, k_a, atol=1e-5), f"Snapshot {snap_idx} layer {l_idx} K mismatch between sync and async"
                assert torch.allclose(v_s, v_a, atol=1e-5), f"Snapshot {snap_idx} layer {l_idx} V mismatch between sync and async"

            # Verify actual reduction in physical length compared to nocompact baseline
            compact_len = s_stat["physical_tokens"]
            nocompact_len = nc_stat["physical_tokens"]
            assert compact_len < nocompact_len, (
                f"Snapshot {snap_idx} expected compact length {compact_len} < nocompact length {nocompact_len}"
            )

            snapshot_metrics.append({
                "snapshot_idx": snap_idx,
                "frames": snap,
                "sync_physical_tokens": compact_len,
                "async_physical_tokens": a_stat["physical_tokens"],
                "sync_kv_bytes": s_stat["kv_nbytes"],
                "async_kv_bytes": a_stat["kv_nbytes"],
                "sync_total_nbytes": s_stat["nbytes"],
                "current_block_length": s_stat["current_block_length"],
                "nocompact_physical_tokens": nocompact_len,
                "length_reduction_pct": round(100.0 * (1.0 - compact_len / nocompact_len), 2),
            })

        # Verify sync vs async parity and token lengths
        for snap_idx in range(len(snapshots)):
            d_s, sh_s = sync_features[snap_idx]
            d_a, sh_a = async_features[snap_idx]
            assert torch.allclose(d_s, d_a, atol=1e-5), f"Sync vs async deep diff at snapshot {snap_idx}"
            assert torch.allclose(sh_s, sh_a, atol=1e-5), f"Sync vs async shallow diff at snapshot {snap_idx}"

        # Deep validation of anchors: shape, full 4 visual tokens, and content matching original decision frame
        assert len(sync_state.memory.anchors) > 0, "Expected non-empty anchors in consolidated state"
        for a in sync_state.memory.anchors:
            assert a.visual_tokens.shape == (1, 4, core.config.hidden_size), (
                f"Anchor visual_tokens shape mismatch: {a.visual_tokens.shape}"
            )
            orig_frame_local_idx = frame_ids.index(a.frame_id)
            orig_img = sample["images_window"][orig_frame_local_idx]
            orig_block = adapter_sync.encode_frame(
                images=[orig_img],
                frame_id=a.frame_id,
                prompt=prompt_seq,
                observation_time=a.observation_time,
            )
            assert torch.equal(a.visual_tokens, orig_block.visual_tokens), (
                f"Anchor frame {a.frame_id} content does not match original decision frame visual tokens"
            )

        # Alignment with NativeCompactMemorySequencePolicy
        # CPU runs full strict FP32 parity against cache adapter.
        # On GPU with BF16 autocast training path, cache FP32 vs training autocast operates at different precision.
        seq_policy = NativeCompactMemorySequencePolicy(
            fresh_policy_seq,
            shallow_layer=1,
            compact_config=compact_cfg_grid1,
            gradient_checkpointing=False,
        )
        seq_policy.eval()
        with torch.no_grad():
            deep_seq, shallow_seq = seq_policy.features(sample)

        deep_target_adapters = torch.cat([sync_features[2][0], sync_features[3][0]], dim=0)
        shallow_target_adapters = torch.cat([sync_features[2][1], sync_features[3][1]], dim=0)

        seq_deep_diff = (deep_seq - deep_target_adapters).abs().max().item()
        seq_shallow_diff = (shallow_seq - shallow_target_adapters).abs().max().item()

        if device.type == "cpu":
            assert seq_deep_diff <= 1e-5, f"Sequence policy vs adapter deep diff on CPU: {seq_deep_diff}"
            assert seq_shallow_diff <= 1e-5, f"Sequence policy vs adapter shallow diff on CPU: {seq_shallow_diff}"
            training_cache_parity_status = "STRICT_FP32_PASSED"
        else:
            # GPU autocast compute path note
            training_cache_parity_status = "MEASURED_DIFFERENT_PRECISION_NOT_ACCEPTED_AS_PARITY"

        report["multi_frame_verification"] = {
            "num_frames": N,
            "frame_ids": frame_ids,
            "observation_times": observation_times,
            "decision_indices": decision_indices,
            "target_indices": target_indices,
            "intermediate_grid": 1,
            "sync_total_seconds_including_rebuild": t_sync_total,
            "async_total_seconds_including_rebuild": t_async_total,
            "cuda_peak_mb_sync": sync_peak_mb,
            "cuda_peak_mb_async": async_peak_mb,
            "cuda_worker_stream_verified": cuda_worker_stream_verified if device.type == "cuda" else None,
            "seq_policy_deep_diff": seq_deep_diff,
            "seq_policy_shallow_diff": seq_shallow_diff,
            "training_cache_parity_status": training_cache_parity_status,
            "snapshots_metrics": snapshot_metrics,
        }

        # 4. Backward pass on 2 targets, finite loss, non-zero gradients on 4 groups + cold frame 0
        train_policy_fresh = copy.deepcopy(policy_template)
        train_policy_fresh.embedder.device = device
        train_seq_policy = NativeCompactMemorySequencePolicy(
            train_policy_fresh,
            shallow_layer=1,
            compact_config=compact_cfg_grid1,
            gradient_checkpointing=False,
        )
        train_seq_policy.train()

        sample_train = copy.deepcopy(sample)
        img0 = sample_train["images_window"][0].clone().detach().requires_grad_(True)
        img20 = sample_train["images_window"][20].clone().detach().requires_grad_(True)
        sample_train["images_window"][0] = img0
        sample_train["images_window"][20] = img20

        train_seq_policy.zero_grad(set_to_none=True)
        train_out = train_seq_policy(sample_train)
        loss = train_out["loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()

        if img0.grad is not None:
            assert torch.all(img0.grad == 0.0), f"Cold frame 0 received non-zero grad: {img0.grad.norm().item()}"

        assert img20.grad is not None and torch.isfinite(img20.grad).all()
        assert img20.grad.norm().item() > 0.0, "Recent frame 20 did not receive gradient"

        # Check that all existing parameter gradients are finite FP32
        for name, p_var in train_seq_policy.named_parameters():
            if p_var.grad is not None:
                assert p_var.grad.dtype == torch.float32, f"Grad for {name} is not FP32: {p_var.grad.dtype}"
                assert torch.isfinite(p_var.grad).all(), f"Grad for {name} contains non-finite values"

        # 4 original model parameter groups must have finite non-zero gradients:
        # 1. ViT encoder
        vit_w = train_seq_policy.policy.embedder.model.vision_model.encoder.conv.weight
        assert vit_w.grad is not None and vit_w.grad.norm().item() > 0.0, "ViT encoder missing grad"
        # 2. Projector
        proj_w = train_seq_policy.policy.embedder.model.projector.proj.weight
        assert proj_w.grad is not None and proj_w.grad.norm().item() > 0.0, "Projector missing grad"
        # 3. LLM layers
        llm_w = train_seq_policy.policy.embedder.model.language_model.model.layers[0].self_attn.q_proj.weight
        assert llm_w.grad is not None and llm_w.grad.norm().item() > 0.0, "LLM layers missing grad"
        # 4. Action head
        act_w = train_seq_policy.policy.action_head.proj.weight
        assert act_w.grad is not None and act_w.grad.norm().item() > 0.0, "Action head missing grad"

        report["gradient_verification"] = {
            "loss": float(loss.item()),
            "vit_grad_norm": float(vit_w.grad.norm().item()),
            "proj_grad_norm": float(proj_w.grad.norm().item()),
            "llm_grad_norm": float(llm_w.grad.norm().item()),
            "action_head_grad_norm": float(act_w.grad.norm().item()),
            "all_existing_grads_finite_fp32": True,
            "cold_frame_grad_zero": True,
            "hot_frame_grad_norm": float(img20.grad.norm().item()),
        }

        # Everything passed
        verification_passed = True
        report["status"] = "PASSED"

    finally:
        # Close adapters safely without swallowing errors that would mask failure
        close_exceptions: List[Exception] = []
        for adapter in adapters_to_close:
            if hasattr(adapter, "close") and callable(adapter.close):
                try:
                    adapter.close()
                except Exception as ex:
                    close_exceptions.append(ex)

        if close_exceptions:
            report["status"] = "FAILED"
            report["close_errors"] = [str(ex) for ex in close_exceptions]
            print(f"Errors occurred during adapter close: {close_exceptions}", file=sys.stderr)

        report_file = out_path / "report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(f"Compact memory verification finished. Status: {report['status']}")
    print(f"Saved report to: {out_path / 'report.json'}")
    return 0 if (verification_passed and report["status"] == "PASSED") else 1


if __name__ == "__main__":
    sys.exit(main())
