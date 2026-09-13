"""GPU verification probe for memory-augmented sequence training across up to 4 representative layouts.

Verifies:
1. CUDA availability and FlashAttention-2 assertion.
2. Load policy via runtime.load_native_checkpoint (trainable=True).
3. Wrap in NativeMemorySequencePolicy (shallow=6, default memory config, checkpoint=True).
4. Assert all policy parameters FP32 on target CUDA device with requires_grad=True,
   and verify zero added parameters in wrapper (wrapper params == policy params).
5. Dataset scan across all episodes/segments with memory_replay_v1 on epoch 1, selecting
   up to 4 representative samples without decoding images during scan:
   - dense (prioritizing target count 8 and dense observations 16 if available)
   - current_only (prioritizing target count 8 if available)
   - memory (maximum observations N > 72 to test cold prefix)
   - memory_context (maximum structural token score:
     recent_frames_count * 1024 + retired_anchor_count * 256 + retired_summary_bin_count * 16,
     explicitly excluding variable text headers, under target count 8)
6. Create temporary AdamW optimizer (FP32, lr=1e-6, total parameters) shared across all cases
   without claiming independent weight baselines. Updated probe weights are never saved.
7. Execute each case:
   - optimizer.zero_grad(set_to_none=True)
   - hook embedder_model.extract_feature:
     * cold frames (requires_grad=False): count calls, do not retain grad
     * hot frames (requires_grad=True): count calls, retain_grad on first hot tensor and verify non-zero grad
   - hook memory_training.materialize_memory:
     * record actual materialized token lengths and current block lengths (Python ints only, no tensors)
   - hook memory_training.advance_memory:
     * record intermediate states and target bank sizes
   - forward pass + backward pass
   - verify finite loss
   - verify 4 parameter groups (vision, projector, llm, head) have finite grad norm > 0
   - verify representative parameter delta > 0 after optimizer.step()
   - verify optimizer moments (exp_avg, exp_avg_sq) are present and finite FP32
   - memory replay cases verify cold/hot calls, consolidations, merges, exact conservation, and visual shapes
   - clean up per-case CUDA tensor references without invoking empty_cache
   - write in-progress report to output_dir/report.json after each case completes
8. Finalize report with status PASSED once all cases succeed. On failure, prior completed cases
   remain saved on disk and exceptions propagate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from fabri_moss import memory_training
from fabri_moss.memory_protocol import compute_memory_replay_layout, get_memory_protocol_contract
from fabri_moss.memory_training import NativeMemorySequencePolicy
from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.periodic_memory import PeriodicMemoryConfig, PeriodicMemoryState
from fabri_moss.runtime import assert_native_fa2, compute_file_sha256, load_native_checkpoint
from fabri_moss.train_native import classify_parameter


def compute_structural_token_score(calendar_queries: List[Dict[str, Any]]) -> int:
    """Compute maximum structural token score across calendar queries for a memory replay sample.

    Score formula:
        recent_frames_count * 1024 + retired_anchor_count * 256 + retired_summary_bin_count * 16

    Note: This measures structural visual token context pressure across the memory sequence.
    Variable text header tokens are explicitly excluded rather than falsely claiming
    an exact token count.
    """
    if not calendar_queries:
        return 0
    max_score = 0
    for q in calendar_queries:
        rf = int(q.get("recent_frames_count", 0))
        ra = int(q.get("retired_anchor_count", 0))
        rb = int(q.get("retired_summary_bin_count", 0))
        score = rf * 1024 + ra * 256 + rb * 16
        if score > max_score:
            max_score = score
    return max_score


def select_layout_segments(
    dataset: NativeTrainingDataset,
) -> Dict[str, Tuple[int, Dict[str, Any]]]:
    """Scan segment metadata without image decode, finding up to 4 representative samples:
    - dense: prioritized with target count == 8 and dense observations == 16 if available
    - current_only: prioritized with target count == 8 if available
    - memory: maximum observation count N (> 72) to test cold prefix
    - memory_context: maximum structural token score under target count == 8
    """
    best_dense: Optional[Tuple[int, Dict[str, Any]]] = None
    best_dense_priority: Tuple[int, int, int] = (-1, -1, -1)

    best_current_only: Optional[Tuple[int, Dict[str, Any]]] = None
    best_current_priority: Tuple[int, int] = (-1, -1)

    best_memory: Optional[Tuple[int, Dict[str, Any]]] = None
    max_mem_obs = -1

    best_context: Optional[Tuple[int, Dict[str, Any]]] = None
    best_context_priority: Tuple[int, int] = (-1, -1)

    # Episode dataframe timestamp and frame_id cache to avoid repeated disk reads across segments in same episode
    ep_data_cache: Dict[int, Tuple[List[float], List[int]]] = {}

    for seg_idx, (ep_idx, start_r, end_r) in enumerate(dataset.segments):
        target_count = end_r - start_r
        if ep_idx not in ep_data_cache:
            ep = dataset._episode_by_id[ep_idx]
            df = dataset._get_episode_dataframe(ep)
            all_timestamps, _ = dataset._validate_and_get_timestamps(df, ep_idx)
            all_frame_ids = [int(f) for f in df["frame_index"].values]
            ep_data_cache[ep_idx] = (all_timestamps, all_frame_ids)
        else:
            all_timestamps, all_frame_ids = ep_data_cache[ep_idx]

        layout = compute_memory_replay_layout(
            seed=dataset.seed,
            epoch=dataset.epoch,
            ep_idx=ep_idx,
            target_start_row=start_r,
            target_end_row=end_r,
            all_timestamps=all_timestamps,
            history_frames=dataset.history_frames,
            split=dataset.split,
            all_frame_ids=all_frame_ids,
        )
        stream_layout = layout["stream_layout"]
        mode = stream_layout["mode"]
        num_obs = stream_layout["num_observations"]

        if mode == "dense":
            prio = (
                1 if (target_count == 8 and num_obs == 16) else 0,
                1 if (target_count == 8) else 0,
                num_obs,
            )
            if best_dense is None or prio > best_dense_priority:
                best_dense = (seg_idx, layout)
                best_dense_priority = prio

        elif mode == "current_only":
            prio_curr = (
                1 if (target_count == 8) else 0,
                target_count,
            )
            if best_current_only is None or prio_curr > best_current_priority:
                best_current_only = (seg_idx, layout)
                best_current_priority = prio_curr

        elif mode == "memory":
            cal_queries = stream_layout.get("calendar_queries", [])
            sample_struct_score = compute_structural_token_score(cal_queries)
            stream_layout["structural_token_score"] = sample_struct_score

            # Track max observations N for testing cold prefix
            if num_obs > max_mem_obs:
                max_mem_obs = num_obs
                best_memory = (seg_idx, layout)

            # Track max structural token score under target count 8
            prio_ctx = (
                1 if (target_count == 8) else 0,
                sample_struct_score,
            )
            if best_context is None or prio_ctx > best_context_priority:
                best_context = (seg_idx, layout)
                best_context_priority = prio_ctx

    if best_dense is None:
        raise RuntimeError("Failed to find any 'dense' segment during dataset scan.")
    if best_current_only is None:
        raise RuntimeError("Failed to find any 'current_only' segment during dataset scan.")
    if best_memory is None:
        raise RuntimeError("Failed to find any 'memory' segment during dataset scan.")
    if max_mem_obs <= 72:
        raise RuntimeError(
            f"Maximum memory observations found was {max_mem_obs} <= 72. Need > 72 to stress memory sequence."
        )
    if best_context is None or best_context_priority[0] == 0:
        raise RuntimeError("Failed to find any 'memory_context' segment with target count 8 during dataset scan.")

    return {
        "dense": best_dense,
        "current_only": best_current_only,
        "memory": best_memory,
        "memory_context": best_context,
    }


def run_probe(args: argparse.Namespace) -> None:
    """Execute GPU verification probe for memory sequence training across representative layouts."""
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verify_memory_training.py, but CUDA is unavailable.")

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
    num_policy_params = sum(p.numel() for p in policy_params)
    for name, p in policy.named_parameters():
        if p.dtype != torch.float32:
            raise AssertionError(f"Parameter {name} is {p.dtype}, expected torch.float32")
        if not p.requires_grad:
            raise AssertionError(f"Parameter {name} has requires_grad=False")
        if p.device.type != target_device.type or (target_device.index is not None and p.device.index != target_device.index):
            raise AssertionError(f"Parameter {name} on {p.device}, expected {target_device}")

    # Wrap in NativeMemorySequencePolicy
    mem_config = PeriodicMemoryConfig()
    mem_policy = NativeMemorySequencePolicy(
        policy=policy,
        shallow_layer=6,
        use_timestamps=True,
        gradient_checkpointing=True,
        memory_config=mem_config,
    )
    mem_policy.train()

    # Zero added parameters check
    wrapper_params = list(mem_policy.parameters())
    num_wrapper_params = sum(p.numel() for p in wrapper_params)
    if len(wrapper_params) != len(policy_params) or {id(p) for p in wrapper_params} != {id(p) for p in policy_params}:
        raise AssertionError("NativeMemorySequencePolicy added new parameters or parameter IDs do not match policy!")
    has_no_new_params = (len(wrapper_params) == len(policy_params) and num_wrapper_params == num_policy_params)

    # Compute source code SHA256 of verify_memory_training.py and core modules
    probe_source_sha = compute_file_sha256(Path(__file__).resolve())
    memory_protocol_sha = compute_file_sha256(REPO_ROOT / "fabri_moss" / "memory_protocol.py")
    memory_training_sha = compute_file_sha256(REPO_ROOT / "fabri_moss" / "memory_training.py")
    periodic_memory_sha = compute_file_sha256(REPO_ROOT / "fabri_moss" / "periodic_memory.py")

    protocol_contract = get_memory_protocol_contract()

    print("[2/5] Initializing dataset and scanning metadata for up to 4 layout cases...")
    dataset = NativeTrainingDataset(
        root=args.data_root,
        norm_stats=norm_stats,
        history_frames=16,
        target_frames=8,
        split="train",
        max_episodes=args.max_episodes,
        augmentation=False,
        stream_protocol="memory_replay_v1",
    )
    dataset.set_epoch(1)

    selected_cases = select_layout_segments(dataset)
    for c_name, (s_idx, s_layout) in selected_cases.items():
        n_obs = s_layout["stream_layout"]["num_observations"]
        s_score = s_layout["stream_layout"].get("structural_token_score", 0)
        print(f"Selected {c_name}: seg_idx={s_idx}, N={n_obs}, structural_score={s_score}")

    same_sample_mem_ctx = (selected_cases["memory"][0] == selected_cases["memory_context"][0])
    if same_sample_mem_ctx:
        print(f"Note: memory and memory_context selected the same segment index ({selected_cases['memory'][0]}).")

    print("[3/5] Creating AdamW optimizer (all parameters, FP32, lr=1e-6)...")
    # 4 cases sequentially run on a shared loaded checkpoint and optimizer step without reset
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

    embedder_model = getattr(getattr(policy, "embedder", None), "model", None)
    if embedder_model is None:
        raise AttributeError("policy.embedder.model is required for feature hook")

    report: Dict[str, Any] = {
        "status": "in_progress",
        "device": str(target_device),
        "source_metadata": source_metadata,
        "probe_source_sha256": probe_source_sha,
        "module_sha256": {
            "memory_protocol.py": memory_protocol_sha,
            "memory_training.py": memory_training_sha,
            "periodic_memory.py": periodic_memory_sha,
        },
        "protocol_contract": protocol_contract,
        "parameter_counts": {
            "policy_parameters": num_policy_params,
            "wrapper_parameters": num_wrapper_params,
            "no_new_params": has_no_new_params,
        },
        "fa2": fa2_diagnostics,
        "notes": {
            "shared_weights": "Cases share initial 93k checkpoint and optimizer state across probe runs; probe weights are not saved.",
            "structural_token_score": "Counts recent * 1024 + anchors * 256 + summary_bins * 16; explicitly excludes variable text header tokens.",
        },
        "cases": {},
    }

    print("[4/5] Executing forward/backward/step probe across representative layouts...")
    cases_order = ("dense", "current_only", "memory", "memory_context")

    for case_idx, case_name in enumerate(cases_order):
        seg_idx, layout_dict = selected_cases[case_name]
        stream_layout = layout_dict["stream_layout"]
        structural_score = int(stream_layout.get("structural_token_score", 0))

        # Decode actual frames only for selected sample
        raw_sample = dataset[seg_idx]

        # Extract true N and M from raw_sample
        N = len(raw_sample["frame_ids"])
        M = len(raw_sample["target_indices"])
        sample_memory_replay = bool(raw_sample.get("memory_replay", False))

        # Invariant checks per case
        if case_name == "memory":
            if N <= 72:
                raise AssertionError(f"MaxN memory case expected N > 72, got {N}")
        elif case_name == "memory_context":
            if not sample_memory_replay:
                raise AssertionError(f"Case memory_context expected sample_memory_replay=True, got {sample_memory_replay}")
            if M != 8:
                raise AssertionError(f"Case memory_context expected target count M=8, got {M}")

        # Move tensors to GPU
        sample_gpu: Dict[str, Any] = {}
        for k, v in raw_sample.items():
            if isinstance(v, torch.Tensor):
                sample_gpu[k] = v.to(target_device)
            else:
                sample_gpu[k] = v

        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
        optimizer.zero_grad(set_to_none=True)

        # Hook extract_feature to count cold vs hot calls and retain grad on hot
        orig_extract_feature = embedder_model.extract_feature
        cold_calls = 0
        hot_calls = 0
        retained_hot_tokens: List[torch.Tensor] = []

        def wrapped_extract_feature(*w_args: Any, **w_kwargs: Any) -> Any:
            nonlocal cold_calls, hot_calls
            out = orig_extract_feature(*w_args, **w_kwargs)
            if torch.is_grad_enabled():
                hot_calls += 1
                if isinstance(out, torch.Tensor) and len(retained_hot_tokens) == 0:
                    out.retain_grad()
                    retained_hot_tokens.append(out)
            else:
                cold_calls += 1
            return out

        embedder_model.extract_feature = wrapped_extract_feature

        # Intercept memory_training.advance_memory to record actual states without pinning graph
        last_state_ref: List[PeriodicMemoryState] = []
        target_bank_sizes: List[Dict[str, int]] = []
        orig_advance_memory = memory_training.advance_memory

        def tracking_advance_memory(
            st: PeriodicMemoryState,
            fr: Any,
            cfg: Any,
        ) -> PeriodicMemoryState:
            new_st = orig_advance_memory(st, fr, cfg)
            if last_state_ref:
                last_state_ref[0] = new_st
            else:
                last_state_ref.append(new_st)
            # Check if this frame index is a target
            curr_idx = new_st.frame_count - 1
            if curr_idx in raw_sample["target_indices"]:
                target_bank_sizes.append({
                    "frame_count": new_st.frame_count,
                    "anchors": len(new_st.anchors),
                    "entries": len(new_st.entries),
                    "recent": len(new_st.recent),
                    "consolidations": new_st.consolidations,
                    "merges": new_st.merges,
                })
            return new_st

        memory_training.advance_memory = tracking_advance_memory

        # Hook memory_training.materialize_memory to record actual input token lengths and current block lengths
        orig_materialize_memory = memory_training.materialize_memory
        mat_token_counts: List[int] = []
        mat_block_lengths: List[int] = []

        def tracking_materialize_memory(*m_args: Any, **m_kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, int]:
            res = orig_materialize_memory(*m_args, **m_kwargs)
            mat_embeds, mat_mask, current_start = res
            t_len = int(mat_embeds.shape[1])
            c_len = int(t_len - current_start)
            mat_token_counts.append(t_len)
            mat_block_lengths.append(c_len)
            return res

        memory_training.materialize_memory = tracking_materialize_memory

        try:
            t0 = time.perf_counter()
            out_dict = mem_policy.forward(sample_gpu)
            loss = out_dict["loss"]
            if not torch.isfinite(loss):
                raise AssertionError(f"Case {case_name} produced non-finite loss: {loss.item()}")

            loss.backward()
            torch.cuda.synchronize(target_device)
            forward_backward_sec = time.perf_counter() - t0
        finally:
            embedder_model.extract_feature = orig_extract_feature
            memory_training.advance_memory = orig_advance_memory
            memory_training.materialize_memory = orig_materialize_memory

        print(
            f"Case [{case_idx + 1}/{len(cases_order)}] '{case_name}': "
            f"N={N}, M={M}, cold_calls={cold_calls}, hot_calls={hot_calls}"
        )

        # Invariant checks based on sample_memory_replay
        if hot_calls == 0:
            raise AssertionError(f"Case {case_name} expected hot calls > 0, got {hot_calls}")

        if case_name == "memory":
            if cold_calls == 0:
                raise AssertionError(f"MaxN memory case expected cold calls > 0, got {cold_calls}")

        if len(retained_hot_tokens) == 0 or retained_hot_tokens[0].grad is None:
            raise AssertionError(f"Case {case_name} failed to capture hot visual token gradient")
        hot_grad_norm = float(retained_hot_tokens[0].grad.norm().item())
        if hot_grad_norm <= 0.0 or not math.isfinite(hot_grad_norm):
            raise AssertionError(f"Case {case_name} hot visual token grad is zero or non-finite: {hot_grad_norm}")

        # Check 4 parameter group gradients
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

        # Optimizer step and parameter delta check
        torch.cuda.synchronize(target_device)
        t_opt = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(target_device)
        optimizer_sec = time.perf_counter() - t_opt

        # Check optimizer state moments: must fail if missing for active parameters
        for name, p in policy.named_parameters():
            if p.grad is not None:
                st = optimizer.state.get(p)
                if st is None:
                    raise AssertionError(f"Optimizer state missing for parameter {name} with gradient")
                for moment_key in ("exp_avg", "exp_avg_sq"):
                    if moment_key not in st:
                        raise AssertionError(f"Optimizer moment {moment_key} missing for parameter {name}")
                    m_tensor = st[moment_key]
                    if m_tensor.dtype != torch.float32 or not torch.isfinite(m_tensor).all():
                        raise AssertionError(f"Optimizer moment {moment_key} for parameter {name} not finite FP32")

        max_deltas: Dict[str, float] = {}
        for grp in ("vision", "projector", "llm", "head"):
            p_name, param, old_val = rep_params[grp]
            delta = float((param.detach() - old_val).abs().max().item())
            if delta <= 0.0 or not math.isfinite(delta):
                raise AssertionError(f"Parameter {p_name} in group {grp} failed to update! delta={delta}")
            max_deltas[grp] = delta

        peak_mib = torch.cuda.max_memory_allocated(target_device) / (1024 * 1024)

        if sample_memory_replay:
            if len(last_state_ref) != 1 or len(mat_token_counts) != M:
                raise AssertionError("Memory probe did not observe every target materialization")
            if mat_block_lengths != [policy.embedder.max_text_length] * M:
                raise AssertionError("Memory probe changed the current frame token block")

        # Inspect memory state counters and entries whenever sample_memory_replay is True
        if sample_memory_replay:
            final_st = last_state_ref[0]
            actual_consolidations = final_st.consolidations
            actual_merges = final_st.merges
            actual_anchors_count = len(final_st.anchors)
            actual_anchor_ids = [a.frame_id for a in final_st.anchors]
            actual_entry_ranges_counts = [
                {
                    "start_frame_id": e.start_frame_id,
                    "end_frame_id": e.end_frame_id,
                    "count": e.count,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                }
                for e in final_st.entries
            ]
            actual_recent_ids = [rf.frame_id for rf in final_st.recent]
            sum_counts = sum(e.count for e in final_st.entries)

            # Verification 1: exact accounting: sumcounts + len(anchors) + len(recent) == N
            total_accounted = sum_counts + actual_anchors_count + len(actual_recent_ids)
            if total_accounted != N:
                raise AssertionError(
                    f"Memory accounting mismatch: sum(counts)={sum_counts} + anchors={actual_anchors_count} + "
                    f"recent={len(actual_recent_ids)} = {total_accounted} != N={N}"
                )

            decision_ids = {raw_sample["frame_ids"][i] for i in raw_sample["decision_indices"]}
            expected_retired_ids = sorted(decision_ids - set(actual_recent_ids))
            if actual_anchor_ids != expected_retired_ids:
                raise AssertionError(f"Protected decision frames mismatch: {actual_anchor_ids} != {expected_retired_ids}")

            # Verification 3: inspect visual token shapes directly from real tensors
            anchor_token_shapes = [list(a.visual_tokens.shape) for a in final_st.anchors]
            entry_token_shapes = [list(e.visual_tokens.shape) for e in final_st.entries]
            recent_token_shapes = [list(rf.visual_tokens.shape) for rf in final_st.recent]

            if actual_anchors_count > 0:
                p_tokens = final_st.anchors[0].visual_tokens.shape[1]
                sqrt_p = int(math.isqrt(p_tokens))
                if sqrt_p * sqrt_p != p_tokens:
                    raise AssertionError(f"Anchor visual tokens P={p_tokens} is not a square shape")

            memory_stats: Optional[Dict[str, Any]] = {
                "has_memory": True,
                "consolidations": actual_consolidations,
                "merges": actual_merges,
                "anchors_count": actual_anchors_count,
                "anchor_frame_ids": actual_anchor_ids,
                "anchor_token_shapes": anchor_token_shapes,
                "entry_ranges_counts": actual_entry_ranges_counts,
                "entry_token_shapes": entry_token_shapes,
                "recent_frame_ids": actual_recent_ids,
                "recent_token_shapes": recent_token_shapes,
                "sum_entry_counts": sum_counts,
                "target_bank_sizes": target_bank_sizes,
            }
        else:
            memory_stats = {
                "has_memory": False,
                "consolidations": 0,
                "merges": 0,
                "anchors_count": 0,
                "anchor_frame_ids": [],
                "anchor_token_shapes": [],
                "entry_ranges_counts": [],
                "entry_token_shapes": [],
                "recent_frame_ids": [],
                "recent_token_shapes": [],
                "sum_entry_counts": 0,
                "target_bank_sizes": [],
            }

        # Observation times and decision IDs
        obs_times = raw_sample.get("observation_times", [])
        actual_obs_span = float(obs_times[-1] - obs_times[0]) if len(obs_times) > 1 else 0.0
        decision_frame_ids = (
            [int(raw_sample["frame_ids"][i]) for i in raw_sample["decision_indices"]]
            if "decision_indices" in raw_sample
            else []
        )

        case_report: Dict[str, Any] = {
            "layout_index": seg_idx,
            "observations_n": N,
            "target_count": M,
            "actual_time_span_seconds": actual_obs_span,
            "actual_observation_time_span_seconds": actual_obs_span,
            "structural_token_score": structural_score,
            "actual_materialized_token_lengths": mat_token_counts,
            "max_materialized_token_length": max(mat_token_counts, default=0),
            "current_block_lengths": mat_block_lengths,
            "decision_frame_ids": decision_frame_ids,
            "cold_feature_calls": cold_calls,
            "hot_feature_calls": hot_calls,
            "hot_visual_grad_norm": hot_grad_norm,
            "loss": float(loss.item()),
            "group_max_grad_norms": group_norms,
            "representative_max_deltas": max_deltas,
            "forward_backward_seconds": forward_backward_sec,
            "optimizer_step_seconds": optimizer_sec,
            "peak_memory_mib": peak_mib,
            "memory_stats": memory_stats,
        }
        if case_name == "memory_context" and same_sample_mem_ctx:
            case_report["same_sample_as"] = "memory"

        report["cases"][case_name] = case_report

        # Write in-progress report immediately to preserve passed cases on subsequent failures
        report_path = out_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(
            f"Case [{case_idx + 1}/{len(cases_order)}] '{case_name}' PASSED - N={N}, M={M}, loss={loss.item():.4f}, "
            f"peak={peak_mib:.1f}MiB, consolidations={memory_stats['consolidations']}, merges={memory_stats['merges']}, "
            f"anchors={memory_stats['anchors_count']}"
        )

        # Clear references to CUDA tensors to avoid cross-case memory retention
        del out_dict, loss, retained_hot_tokens, last_state_ref, sample_gpu, raw_sample, rep_params, old_val
        if sample_memory_replay:
            del final_st
        optimizer.zero_grad(set_to_none=True)

    print("[5/5] All cases PASSED. Saving final verification report...")
    report["status"] = "PASSED"
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {report_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify memory-augmented sequence training on GPU across representative layouts."
    )
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to FabriVLA repository")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Native checkpoint path")
    parser.add_argument("--vlm-path", type=str, default="/root/models/InternVL3_5-1B", help="VLM backbone path")
    parser.add_argument("--data-root", type=str, default="/root/evo1_metaworld_dataset", help="MetaWorld dataset root")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device")
    parser.add_argument("--output-dir", type=str, required=True, help="Empty output directory for verification artifacts")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Robot arm key")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional max episodes limit for dataset loading")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
