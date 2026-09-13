"""Real-GPU bounded delta visual memory verification and acceptance probe.

Verifies bounded delta visual memory mechanisms on real GPU with real FabriVLA weights:
1. Direct evaluation of two consecutive 5-frame batches:
   - Evaluates batch 1 [0..4] from None -> s1, clones s1.matrices values to verify immutability.
   - Evaluates batch 2 [5..9] from s1 -> s2.
   - Fresh replay of the exact two batches from None -> s2_fresh; asserts parity with online:
     maxdiff(deep, shallow, matrices) <= 1e-5.
   - State invariants: frame_count == 10, last_frame_id == obs[9].frame_id, nbytes == 2097152 (2 MiB).
   - KV projection counting: each batch projects exactly once per frame (wrap simple counters).
   - Difference check: batch 2 from None vs from history produces non-zero difference when
     synthetic open gates are active (attn_gate=0.1, mlp_gate=0.1) without adapter.
   - Matrices S_old == clone, FrameKV references not retained in result/state.
2. Real asynchronous pipeline verification:
   - AsyncVisualPlanner(stateful=True, max_frames=5, max_pending=8).
   - Submit batch 1 (frames 0..4), wait_ready=5, request_plan #1, submit batch 2 (frames 5..9).
   - wait_plan #1, wait_ready=5, request_plan #2, wait_plan #2.
   - Verifies callbacks maintain memory (frame_count=10, memory_bytes=2MiB).
   - Verifies deep/shallow match direct two-batch execution within 1e-4.
   - Actions finite, shape (1, 50, 24), and zero-padded for dims >= 4.
   - Records wall-clock event intervals and concurrency.
   - Verifies PlanResult.computation.next_memory is None, and reset clears stats/memory.
3. Backward probe (without adapter):
   - In model.train() mode, computes loss = deep.square().mean() + shallow.square().mean().
   - Backward pass: verifies K/V projections, write_logits, memory_gate, readout have finite,
     non-zero gradients in every layer while frozen base policy gradients are None.
   - Reports gradient norms without modifying weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import dataclasses
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from fabri_moss.async_pipeline import (
    AsyncVisualPlanner,
    EncodedFrame,
    Observation,
    PlanComputation,
    PlanResult,
    make_moss_callbacks,
)
from fabri_moss.core import FrameKV, MossConfig, MossInternVL
from fabri_moss.data import MetaWorldWindows
from fabri_moss.runtime import assert_native_fa2, compute_file_sha256, load_native_checkpoint
from fabri_moss.train import initialize_adapter_weights


DEFAULT_VERIFY_SEED = 4042


def compute_interval_overlap(
    interval_a: Tuple[float, float],
    interval_b: Tuple[float, float],
) -> float:
    """Compute length of time overlap between two [start, finish] intervals."""
    start = max(interval_a[0], interval_b[0])
    finish = min(interval_a[1], interval_b[1])
    return max(0.0, finish - start)


def compute_tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """Compute exact numerical difference metrics between two tensors."""
    a_f = a.float().detach().cpu()
    b_f = b.float().detach().cpu()
    diff = (a_f - b_f).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())
    norm_scale = float(torch.sqrt((b_f ** 2).mean()).item() + 1e-8)
    rel_rmse = float(rmse / norm_scale)
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "allclose_1e5": bool(max_diff <= 1e-5),
        "allclose_1e4": bool(max_diff <= 1e-4),
        "allclose_1e3": bool(max_diff <= 1e-3),
    }


def find_episode_samples(
    dataset: MetaWorldWindows,
    target_episode_id: int = 148,
    required_frames: int = 10,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Locate the first consecutive required_frames of target_episode_id using dataset.anchors metadata.

    Filters dataset.anchors by episode metadata to find the exact row-ordered indices for the target
    episode, and calls dataset[idx] exactly required_frames times.
    Raises ValueError if target_episode_id is not found or has fewer than required_frames.
    Never iterates over the entire dataset or falls back to other episodes.
    """
    if not hasattr(dataset, "anchors"):
        raise AttributeError("Dataset does not have 'anchors' metadata attribute")

    matching_indices: List[int] = []
    for idx, anchor in enumerate(dataset.anchors):
        ep_dict, row = anchor
        ep_idx = int(ep_dict.get("episode_index", ep_dict.get("episode_id", -1)))
        if ep_idx == target_episode_id:
            matching_indices.append(idx)
            if len(matching_indices) == required_frames:
                break

    if len(matching_indices) < required_frames:
        raise ValueError(
            f"Episode {target_episode_id} not found or has insufficient frames in dataset.anchors: "
            f"found {len(matching_indices)} frames, required {required_frames}. "
            f"Fallback to other episodes is strictly forbidden."
        )

    chosen_samples = [dataset[i] for i in matching_indices]
    fids = [int(s["frame_ids"][-1]) for s in chosen_samples]
    expected_fids = list(range(fids[0], fids[0] + required_frames))
    if fids != expected_fids:
        raise ValueError(
            f"Episode {target_episode_id} frames are not consecutive: got {fids}, expected {expected_fids}"
        )

    return target_episode_id, chosen_samples


def convert_samples_to_observations(
    samples: Sequence[Dict[str, Any]],
) -> List[Observation]:
    """Convert dataset samples to strict CPU Observation objects with monotonic capture timestamps."""
    observations: List[Observation] = []
    for s in samples:
        img_item = s["images_window"][-1]
        t_cap = time.monotonic()

        state_tensor = s["state"].clone().float()
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        state_mask_tensor = s["state_mask"].clone().bool()
        if state_mask_tensor.ndim == 1:
            state_mask_tensor = state_mask_tensor.unsqueeze(0)

        action_mask_tensor = s["action_mask"].clone().bool()
        if action_mask_tensor.ndim == 2:
            action_mask_tensor = action_mask_tensor[0:1, :]
        elif action_mask_tensor.ndim == 1:
            action_mask_tensor = action_mask_tensor.unsqueeze(0)

        obs = Observation(
            frame_id=int(s["frame_ids"][-1]),
            capture_time=t_cap,
            images=list(img_item),
            state=state_tensor,
            state_mask=state_mask_tensor,
            action_mask=action_mask_tensor,
        )
        observations.append(obs)
    return observations


@contextmanager
def count_wrapped_calls(model: MossInternVL) -> Iterator[Dict[str, int]]:
    """Context manager wrapping encode_image and project_frame to record exact execution counts.

    Guarantees restoration of original methods via try/finally even if an exception occurs.
    """
    counts = {"encode_image": 0, "project_frame": 0}
    orig_encode = model.encode_image
    orig_project = model.project_frame

    def counting_encode(*args: Any, **kwargs: Any) -> torch.Tensor:
        counts["encode_image"] += 1
        return orig_encode(*args, **kwargs)

    def counting_project(*args: Any, **kwargs: Any) -> FrameKV:
        counts["project_frame"] += 1
        return orig_project(*args, **kwargs)

    model.encode_image = counting_encode  # type: ignore
    model.project_frame = counting_project  # type: ignore

    try:
        yield counts
    finally:
        model.encode_image = orig_encode  # type: ignore
        model.project_frame = orig_project  # type: ignore


# ============================================================================
# Section 1: Direct Two-Batch Evaluation & Parity Verification
# ============================================================================


def verify_direct_batches(
    model: MossInternVL,
    observations: List[Observation],
    prompt: str,
    synthetic_open_gate: bool,
) -> Dict[str, Any]:
    """Execute direct two-batch delta verification (batch 1: 0..4, batch 2: 5..9)."""
    model.eval()

    obs_batch1 = observations[:5]
    obs_batch2 = observations[5:10]

    with count_wrapped_calls(model) as counts:
        with torch.no_grad():
            # Project batch 1 and batch 2 (10 frames total)
            frames_batch1 = [
                model.project_frame(model.encode_image(list(obs.images)), frame_id=obs.frame_id)
                for obs in obs_batch1
            ]
            frames_batch2 = [
                model.project_frame(model.encode_image(list(obs.images)), frame_id=obs.frame_id)
                for obs in obs_batch2
            ]

            pre_query_counts = dict(counts)
            # After 10 frame projections, counts must be 10 each
            counts_after_initial_10 = (
                pre_query_counts["encode_image"] == 10 and pre_query_counts["project_frame"] == 10
            )

            # 1. read_delta(batch1) -> deep1, shallow1, s1
            deep1, shallow1, s1 = model.read_delta(frames_batch1, prompt, previous=None)

            # Clone s1.matrices values to verify later that s1 was not modified in-place
            s1_cloned_matrices = tuple(m.clone() for m in s1.matrices)

            # 2. read_delta(batch2, previous=s1) -> deep2, shallow2, s2
            deep2, shallow2, s2 = model.read_delta(frames_batch2, prompt, previous=s1)

            counts_after_online_queries = dict(counts)
            # read_delta calls must not trigger encode_image or project_frame
            queries_did_not_increment_counts = (
                counts_after_online_queries["encode_image"] == pre_query_counts["encode_image"]
                and counts_after_online_queries["project_frame"] == pre_query_counts["project_frame"]
            )

            # Verify immutability: s1 matrices must exactly match s1_cloned_matrices
            s1_unchanged = True
            for m_curr, m_cloned in zip(s1.matrices, s1_cloned_matrices):
                if not torch.equal(m_curr, m_cloned):
                    s1_unchanged = False
                    break

            # 3. Fresh replay of identical 2 batches from None
            fresh_frames1 = [
                model.project_frame(model.encode_image(list(obs.images)), frame_id=obs.frame_id)
                for obs in obs_batch1
            ]
            fresh_frames2 = [
                model.project_frame(model.encode_image(list(obs.images)), frame_id=obs.frame_id)
                for obs in obs_batch2
            ]

            deep1_fresh, shallow1_fresh, s1_fresh = model.read_delta(
                fresh_frames1, prompt, previous=None
            )
            deep2_fresh, shallow2_fresh, s2_fresh = model.read_delta(
                fresh_frames2, prompt, previous=s1_fresh
            )

            # 4. Same batch2 evaluated from None (isolated, no history)
            deep2_from_none, shallow2_from_none, s2_from_none = model.read_delta(
                fresh_frames2, prompt, previous=None
            )

        total_counts = dict(counts)
        # Total counts across initial 10 frames + fresh 10 frames = exactly 20 each
        counts_total_correct = (
            total_counts["encode_image"] == 20 and total_counts["project_frame"] == 20
        )

    # Numerical parity checks
    deep1_diff = compute_tensor_diff(deep1, deep1_fresh)
    shallow1_diff = compute_tensor_diff(shallow1, shallow1_fresh)
    deep2_diff = compute_tensor_diff(deep2, deep2_fresh)
    shallow2_diff = compute_tensor_diff(shallow2, shallow2_fresh)

    state_matrices_maxdiff = 0.0
    for m_on, m_fr in zip(s2.matrices, s2_fresh.matrices):
        diff_m = float((m_on.float() - m_fr.float()).abs().max().item())
        state_matrices_maxdiff = max(state_matrices_maxdiff, diff_m)

    parity_pass = bool(
        deep1_diff["allclose_1e5"]
        and shallow1_diff["allclose_1e5"]
        and deep2_diff["allclose_1e5"]
        and shallow2_diff["allclose_1e5"]
        and state_matrices_maxdiff <= 1e-5
    )

    # State architecture and invariants check
    state_frame_count_correct = (s2.frame_count == 10)
    state_last_id_correct = (s2.last_frame_id == obs_batch2[-1].frame_id)
    state_nbytes = s2.nbytes
    expected_nbytes = sum(
        1 * block.num_kv_heads * block.head_dim * block.head_dim * 4
        for block in model.cross_blocks.values()
    )
    state_nbytes_correct = (state_nbytes == expected_nbytes)

    # Memory state retains no FrameKV or image references
    state_clean = not hasattr(s2, "frames") and not hasattr(s2, "images")

    # History impact check: compare batch2 with history vs batch2 from None
    diff_vs_none_deep = compute_tensor_diff(deep2, deep2_from_none)
    diff_vs_none_shallow = compute_tensor_diff(shallow2, shallow2_from_none)
    history_impact_observed = (
        diff_vs_none_deep["max_diff"] > 0.0 or diff_vs_none_shallow["max_diff"] > 0.0
    )

    if synthetic_open_gate and not history_impact_observed:
        history_difference_pass = False
    else:
        history_difference_pass = True

    overall_direct_pass = (
        parity_pass
        and s1_unchanged
        and state_frame_count_correct
        and state_last_id_correct
        and state_nbytes_correct
        and state_clean
        and history_difference_pass
        and counts_after_initial_10
        and queries_did_not_increment_counts
        and counts_total_correct
    )

    return {
        "overall_passed": bool(overall_direct_pass),
        "parity_pass": bool(parity_pass),
        "s1_immutability_preserved": bool(s1_unchanged),
        "state_frame_count": s2.frame_count,
        "state_frame_count_correct": bool(state_frame_count_correct),
        "state_last_frame_id": s2.last_frame_id,
        "state_last_id_correct": bool(state_last_id_correct),
        "state_nbytes": state_nbytes,
        "expected_nbytes": expected_nbytes,
        "state_nbytes_correct": bool(state_nbytes_correct),
        "history_impact_observed": bool(history_impact_observed),
        "diff_vs_none_deep_max": diff_vs_none_deep["max_diff"],
        "diff_vs_none_shallow_max": diff_vs_none_shallow["max_diff"],
        "counts": total_counts,
        "pre_query_counts": pre_query_counts,
        "counts_after_initial_10": bool(counts_after_initial_10),
        "queries_did_not_increment_counts": bool(queries_did_not_increment_counts),
        "counts_total_correct": bool(counts_total_correct),
        "deep1_diff": deep1_diff,
        "shallow1_diff": shallow1_diff,
        "deep2_diff": deep2_diff,
        "shallow2_diff": shallow2_diff,
        "state_matrices_maxdiff": state_matrices_maxdiff,
        "batch1_outputs": {
            "deep": deep1.detach().cpu(),
            "shallow": shallow1.detach().cpu(),
        },
        "batch2_outputs": {
            "deep": deep2.detach().cpu(),
            "shallow": shallow2.detach().cpu(),
        },
    }


# ============================================================================
# Section 2: Real Asynchronous Pipeline Verification
# ============================================================================


def verify_async_delta_pipeline(
    model: MossInternVL,
    observations: List[Observation],
    prompt: str,
    direct_batch2_outputs: Dict[str, Any],
    wait_timeout: float = 120.0,
) -> Dict[str, Any]:
    """Execute asynchronous decoupled replay verification with stateful delta memory."""
    model.eval()

    encode_intervals: List[Dict[str, Any]] = []
    plan_intervals: List[Dict[str, Any]] = []

    base_encode_fn, base_plan_fn, base_val_fn = make_moss_callbacks(model)

    def tracking_encode(obs: Observation) -> Any:
        th_id = threading.get_ident()
        t0 = time.monotonic()
        payload = base_encode_fn(obs)
        t1 = time.monotonic()
        encode_intervals.append(
            {
                "frame_id": obs.frame_id,
                "thread_id": th_id,
                "start": t0,
                "finish": t1,
                "duration": t1 - t0,
            }
        )
        return payload

    def tracking_plan(
        frames: Tuple[EncodedFrame, ...], p: str, prev_mem: Optional[Any] = None
    ) -> PlanComputation:
        th_id = threading.get_ident()
        t0 = time.monotonic()
        res = base_plan_fn(frames, p, prev_mem)
        t1 = time.monotonic()
        plan_intervals.append(
            {
                "source_frame_id": frames[-1].observation.frame_id,
                "frame_ids": [f.observation.frame_id for f in frames],
                "thread_id": th_id,
                "start": t0,
                "finish": t1,
                "duration": t1 - t0,
            }
        )
        return res

    plan_1: Optional[PlanResult] = None
    plan_2: Optional[PlanResult] = None
    stats_after_plan_1: Dict[str, Any] = {}
    stats_after_plan_2: Dict[str, Any] = {}
    stats_after_reset: Dict[str, Any] = {}
    stats_after_close: Dict[str, Any] = {}

    expected_p1_ids = [obs.frame_id for obs in observations[:5]]
    expected_p2_ids = [obs.frame_id for obs in observations[5:10]]

    with AsyncVisualPlanner(
        encode=tracking_encode,
        plan=tracking_plan,
        max_frames=5,
        max_pending=8,
        validate=base_val_fn,
        stateful=True,
    ) as planner:
        planner.reset(episode_id="verify_delta_ep", prompt=prompt)

        # 1. Submit first 5 frames [0..4]
        for f_idx in range(5):
            planner.submit(observations[f_idx])

        # Await ready = 5
        if not planner.wait_ready(min_frames=5, timeout=wait_timeout):
            raise TimeoutError("Timed out waiting for initial 5 frames ready in async planner")

        # 2. Request plan #1
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan #1 in async planner")

        # 3. Submit next 5 frames [5..9]
        for f_idx in range(5, 10):
            planner.submit(observations[f_idx])

        # 4. Wait for plan #1 completion
        plan_1 = planner.wait_plan(timeout=wait_timeout)
        if plan_1 is None:
            raise TimeoutError("Timed out waiting for plan #1 in async planner")

        stats_after_plan_1 = copy.deepcopy(planner.stats())

        # 5. Wait for frames [5..9] ready
        if not planner.wait_ready(min_frames=5, timeout=wait_timeout):
            raise TimeoutError("Timed out waiting for second batch of 5 frames ready")

        # 6. Request plan #2
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan #2 in async planner")

        # 7. Wait for plan #2 completion
        plan_2 = planner.wait_plan(timeout=wait_timeout)
        if plan_2 is None:
            raise TimeoutError("Timed out waiting for plan #2 in async planner")

        stats_after_plan_2 = copy.deepcopy(planner.stats())

        # 8. Genuine reset verification inside context: reset for new episode and verify clearing
        planner.reset(episode_id="verify_delta_ep_reset", prompt=prompt)
        stats_after_reset = copy.deepcopy(planner.stats())

    # 9. Verify stats after close
    stats_after_close = copy.deepcopy(planner.stats())

    # Invariant and metric checks
    assert plan_1 is not None and plan_2 is not None

    p1_fids = [f.observation.frame_id for f in plan_1.frames]
    p2_fids = [f.observation.frame_id for f in plan_2.frames]

    p1_cutoff_correct = (plan_1.source_frame_id == expected_p1_ids[-1]) and (p1_fids == expected_p1_ids)
    p2_cutoff_correct = (plan_2.source_frame_id == expected_p2_ids[-1]) and (p2_fids == expected_p2_ids)
    batches_disjoint = bool(set(p1_fids).isdisjoint(set(p2_fids)))

    # Verify memory invariants from stats
    mem_count_p1 = stats_after_plan_1.get("memory_frame_count")
    mem_bytes_p1 = stats_after_plan_1.get("memory_bytes")
    mem_count_p2 = stats_after_plan_2.get("memory_frame_count")
    mem_bytes_p2 = stats_after_plan_2.get("memory_bytes")

    expected_bytes = sum(
        1 * block.num_kv_heads * block.head_dim * block.head_dim * 4
        for block in model.cross_blocks.values()
    )
    memory_invariants_pass = (
        mem_count_p1 == 5
        and mem_bytes_p1 == expected_bytes
        and mem_count_p2 == 10
        and mem_bytes_p2 == expected_bytes
    )

    # Reset verification: memory count and memory bytes must be strictly zero
    reset_verified = (
        stats_after_reset.get("memory_frame_count") == 0
        and stats_after_reset.get("memory_bytes") == 0
        and stats_after_close.get("memory_frame_count") == 0
        and stats_after_close.get("memory_bytes") == 0
    )

    # Verify next_memory is stripped in public PlanComputation
    p1_next_mem_none = (plan_1.computation.next_memory is None)
    p2_next_mem_none = (plan_2.computation.next_memory is None)

    # Parity check between async plan_2 cached outputs and direct batch 2 outputs
    async_deep2 = plan_2.computation.deep
    async_shallow2 = plan_2.computation.shallow
    direct_deep2 = direct_batch2_outputs["deep"]
    direct_shallow2 = direct_batch2_outputs["shallow"]

    deep_diff = (
        compute_tensor_diff(async_deep2, direct_deep2)
        if async_deep2 is not None
        else {"allclose_1e4": False, "max_diff": float("nan")}
    )
    shallow_diff = (
        compute_tensor_diff(async_shallow2, direct_shallow2)
        if async_shallow2 is not None
        else {"allclose_1e4": False, "max_diff": float("nan")}
    )

    async_parity_pass = bool(deep_diff.get("allclose_1e4", False) and shallow_diff.get("allclose_1e4", False))

    # Action validity and padding check: shape strictly (1, 50, 24)
    actions_p1 = plan_1.computation.actions
    actions_p2 = plan_2.computation.actions

    actions_finite = bool(
        torch.isfinite(actions_p1).all().item() and torch.isfinite(actions_p2).all().item()
    )
    actions_shape_valid = (
        actions_p1.ndim == 3
        and actions_p1.shape == (1, 50, 24)
        and actions_p2.ndim == 3
        and actions_p2.shape == (1, 50, 24)
    )

    # Actions padded with zeros for index >= 4 (dimensions 4..23)
    padded_zeros = bool(
        (actions_p1[:, :, 4:] == 0).all().item() and (actions_p2[:, :, 4:] == 0).all().item()
    )

    # Host-side thread concurrency quantification (host event overlap, not GPU kernel execution)
    p1_interval = (plan_1.started, plan_1.finished)
    concurrent_encodes: List[Dict[str, Any]] = []
    total_overlap_seconds = 0.0
    for enc in encode_intervals:
        if enc["frame_id"] in expected_p2_ids:
            enc_interval = (enc["start"], enc["finish"])
            overlap = compute_interval_overlap(p1_interval, enc_interval)
            if overlap > 0.0:
                total_overlap_seconds += overlap
                concurrent_encodes.append(
                    {
                        "frame_id": enc["frame_id"],
                        "overlap_seconds": overlap,
                        "encode_interval": enc_interval,
                        "plan_interval": p1_interval,
                    }
                )
    concurrency_observed = bool(total_overlap_seconds > 0.0)

    overall_async_pass = (
        p1_cutoff_correct
        and p2_cutoff_correct
        and batches_disjoint
        and memory_invariants_pass
        and reset_verified
        and p1_next_mem_none
        and p2_next_mem_none
        and async_parity_pass
        and actions_finite
        and actions_shape_valid
        and padded_zeros
    )

    return {
        "overall_passed": bool(overall_async_pass),
        "p1_cutoff_correct": bool(p1_cutoff_correct),
        "p2_cutoff_correct": bool(p2_cutoff_correct),
        "batches_disjoint": bool(batches_disjoint),
        "memory_invariants_pass": bool(memory_invariants_pass),
        "reset_verified": bool(reset_verified),
        "memory_frame_count_after_p1": mem_count_p1,
        "memory_frame_count_after_p2": mem_count_p2,
        "memory_bytes_p1": mem_bytes_p1,
        "memory_bytes_p2": mem_bytes_p2,
        "p1_next_mem_none": bool(p1_next_mem_none),
        "p2_next_mem_none": bool(p2_next_mem_none),
        "async_parity_pass": bool(async_parity_pass),
        "deep_diff": deep_diff,
        "shallow_diff": shallow_diff,
        "actions_finite": bool(actions_finite),
        "actions_shape_valid": bool(actions_shape_valid),
        "actions_padded_zero": bool(padded_zeros),
        "host_concurrency_observed": bool(concurrency_observed),
        "total_host_overlap_seconds": float(total_overlap_seconds),
        "concurrent_encodes": concurrent_encodes,
        "encode_intervals": encode_intervals,
        "plan_intervals": plan_intervals,
        "stats_after_plan_1": stats_after_plan_1,
        "stats_after_plan_2": stats_after_plan_2,
        "stats_after_reset": stats_after_reset,
        "stats_after_close": stats_after_close,
    }


# ============================================================================
# Section 3: Backward Probe (Gradient Check without Adapter)
# ============================================================================


def verify_backward_probe(
    model: MossInternVL,
    observations: List[Observation],
    prompt: str,
    adapter_loaded: bool,
) -> Dict[str, Any]:
    """Execute backward probe on 2 consecutive batches to verify gradient flow.

    Computes loss = deep.square().mean() + shallow.square().mean(), calls backward(),
    and verifies that every delta layer has finite, non-zero gradients across its
    k_proj, v_proj, write_logits, memory_gate, and readout parameters, while frozen
    base policy parameters receive no gradients.
    """
    if adapter_loaded:
        return {
            "skipped": True,
            "reason": "保留加载策略用于推理验收，梯度路径另由无adapter probe检查",
            "overall_passed": True,
        }

    model.train()
    model.zero_grad(set_to_none=True)

    obs_batch1 = observations[:5]
    obs_batch2 = observations[5:10]

    frames_b1 = [
        model.project_frame(model.encode_image(list(obs.images)), frame_id=obs.frame_id)
        for obs in obs_batch1
    ]
    frames_b2 = [
        model.project_frame(model.encode_image(list(obs.images)), frame_id=obs.frame_id)
        for obs in obs_batch2
    ]

    deep1, shallow1, s1 = model.read_delta(frames_b1, prompt, previous=None)
    deep2, shallow2, s2 = model.read_delta(frames_b2, prompt, previous=s1)

    loss = deep2.square().mean() + shallow2.square().mean()
    loss.backward()

    delta_grads: Dict[str, Dict[str, Any]] = {}
    policy_grads_none = True

    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            policy_grads_none = False
            break

    # Strictly check EVERY cross_block layer for: k_proj, v_proj, write_logits, memory_gate
    # and check readout_embeddings
    per_layer_checks: Dict[str, Dict[str, bool]] = {}
    all_layers_valid = True

    for lay_idx in model.config.cross_layers:
        block = model.cross_blocks[str(lay_idx)]
        layer_res: Dict[str, bool] = {}

        for component_name, param in [
            ("k_proj", block.k_proj.weight),
            ("v_proj", block.v_proj.weight),
            ("write_logits", block.write_logits),
            ("memory_gate", block.memory_gate),
        ]:
            if param.grad is None:
                layer_res[component_name] = False
                all_layers_valid = False
            else:
                g = param.grad
                finite = bool(torch.isfinite(g).all().item())
                norm = float(g.norm().item())
                valid = finite and (norm > 1e-12)
                layer_res[component_name] = valid
                if not valid:
                    all_layers_valid = False

        per_layer_checks[str(lay_idx)] = layer_res

    # Check readout_embeddings
    readout_valid = False
    if model.readout_embeddings.grad is not None:
        g = model.readout_embeddings.grad
        finite = bool(torch.isfinite(g).all().item())
        norm = float(g.norm().item())
        readout_valid = finite and (norm > 1e-12)

    # Collect all trainable parameter grads for diagnostics
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            delta_grads[name] = {"has_grad": False}
        else:
            g = param.grad
            g_finite = bool(torch.isfinite(g).all().item())
            g_norm = float(g.norm().item())
            delta_grads[name] = {
                "has_grad": True,
                "finite": g_finite,
                "norm": g_norm,
                "is_nonzero": bool(g_norm > 1e-12),
            }

    # Zero grads after probe without calling optimizer.step
    model.zero_grad(set_to_none=True)
    model.eval()

    probe_pass = bool(policy_grads_none and all_layers_valid and readout_valid)

    return {
        "skipped": False,
        "overall_passed": bool(probe_pass),
        "policy_grads_none": bool(policy_grads_none),
        "all_layers_valid": bool(all_layers_valid),
        "readout_valid": bool(readout_valid),
        "per_layer_checks": per_layer_checks,
        "delta_grads": delta_grads,
    }


# ============================================================================
# CLI & Main Entrypoint
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-GPU Bounded Delta Memory Verification")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/models/FabriVLA/checkpoint_step_93000.pt",
        help="Base checkpoint path",
    )
    parser.add_argument(
        "--fabri-root",
        type=str,
        default="/root/FabriVLA",
        help="Path to native FabriVLA repository",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="/root/models/InternVL3_5-1B",
        help="Path to local InternVL3.5 directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to MetaWorld LeRobot dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save report.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Compute device (must be CUDA for official acceptance)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional path to MOSS delta adapter checkpoint",
    )
    parser.add_argument(
        "--adapter-stage",
        type=str,
        default="bridge",
        choices=["bridge", "expert"],
        help="Stage when loading adapter",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Torch CPU thread count (must be positive integer)",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=50,
        help="Flow integration steps for action sampling (must be positive integer)",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=148,
        help="Target episode ID in dataset (default: 148)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt matching dataset task prompt (arbitrary overrides forbidden)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_VERIFY_SEED,
        help=f"Deterministic random seed (default: {DEFAULT_VERIFY_SEED})",
    )
    parser.add_argument(
        "--arm-key",
        type=str,
        default="metaworld_sawyer",
        help="Arm key in norm stats",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for async pipeline operations (must be positive float)",
    )
    return parser


def _verify_delta_impl(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    # Parameter sanity checks
    if args.threads <= 0:
        raise ValueError(f"threads must be a positive integer, got {args.threads}")
    if args.flow_steps <= 0:
        raise ValueError(f"flow_steps must be a positive integer, got {args.flow_steps}")
    if not (isinstance(args.wait_timeout, (int, float)) and math.isfinite(args.wait_timeout) and args.wait_timeout > 0):
        raise ValueError(f"wait_timeout must be a finite positive number, got {args.wait_timeout}")

    # CUDA device verification
    target_device = torch.device(args.device)
    if target_device.type != "cuda":
        raise ValueError(
            f"Verification requires a CUDA device, got {args.device!r}. CPU execution is forbidden."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available on this host; cannot run on {args.device}")

    torch.cuda.set_device(target_device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(target_device)

    seed = args.seed if args.seed is not None else DEFAULT_VERIFY_SEED
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(args.threads)

    # 1. Load native base checkpoint
    policy, ckpt_cfg, norm_stats, meta = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm,
        device=args.device,
        arm_key=args.arm_key,
    )

    # 2. Strict FlashAttention-2 assertion
    fa2_diagnostics = assert_native_fa2(policy)

    # 3. Configure action head and MOSS delta model
    policy.action_head.config.num_inference_timesteps = args.flow_steps
    moss_config = MossConfig(max_frames=5, memory_mode="delta")
    moss_model = MossInternVL(policy, config=moss_config)
    moss_model.set_training_stage(args.adapter_stage)

    # Ensure all parameters on target device
    moss_model.to(args.device)

    # Verify every parameter resides strictly on target CUDA device
    for p_name, param in moss_model.named_parameters():
        p_dev = param.device
        if p_dev.type != target_device.type or (target_device.index is not None and p_dev.index != target_device.index):
            raise RuntimeError(
                f"Model parameter '{p_name}' device mismatch: {p_dev} vs target {target_device}"
            )

    # 4. Handle adapter or synthetic open gates
    adapter_provenance = None
    adapter_sha = None
    synthetic_open_gate = False

    if args.adapter is not None:
        adapter_path = Path(args.adapter).resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")
        adapter_sha = compute_file_sha256(adapter_path)
        adapter_provenance = initialize_adapter_weights(
            path=adapter_path,
            student_model=moss_model,
            expected_norm_stats=norm_stats,
            current_base_metadata=meta,
            device="cpu",
        )
        moss_model.to(args.device)
    else:
        # Without adapter: for mechanism acceptance, set attn_gate=0.1, mlp_gate=0.1
        synthetic_open_gate = True
        for block in moss_model.cross_blocks.values():
            with torch.no_grad():
                block.attn_gate.fill_(0.1)
                block.mlp_gate.fill_(0.1)

    # 5. Load real MetaWorld dataset window=1 split=all
    dataset = MetaWorldWindows(
        root=args.data_root,
        norm_stats=norm_stats,
        window=1,
        frame_stride=1,
        split="all",
        max_episodes=None,
    )

    # Find target episode (e.g. 148) with 10 consecutive frames
    ep_id, ep_samples = find_episode_samples(
        dataset=dataset,
        target_episode_id=args.episode_id,
        required_frames=10,
    )

    # Resolve prompt strictly from dataset sample
    dataset_prompt = str(ep_samples[0]["prompt"])
    if args.prompt is not None and args.prompt.strip() != dataset_prompt.strip():
        raise ValueError(
            f"CLI --prompt override mismatch: provided '{args.prompt}', "
            f"but episode {ep_id} task prompt is '{dataset_prompt}'. "
            f"Arbitrary prompt mismatch is strictly forbidden."
        )
    prompt = dataset_prompt
    observations = convert_samples_to_observations(ep_samples)

    # ========================================================================
    # Execute Probes
    # ========================================================================

    # Section 1: Direct Two-Batch Parity & Memory Invariants
    direct_report = verify_direct_batches(
        model=moss_model,
        observations=observations,
        prompt=prompt,
        synthetic_open_gate=synthetic_open_gate,
    )

    # Section 2: Real Asynchronous Pipeline Replay
    async_report = verify_async_delta_pipeline(
        model=moss_model,
        observations=observations,
        prompt=prompt,
        direct_batch2_outputs=direct_report["batch2_outputs"],
        wait_timeout=args.wait_timeout,
    )

    # Section 3: Backward Probe (only when no adapter loaded)
    backward_report = verify_backward_probe(
        model=moss_model,
        observations=observations,
        prompt=prompt,
        adapter_loaded=(args.adapter is not None),
    )

    # Assert real 2 MiB memory state size (2,097,152 bytes)
    expected_real_nbytes = 2097152
    actual_state_nbytes = direct_report["state_nbytes"]
    if actual_state_nbytes != expected_real_nbytes:
        raise AssertionError(
            f"State size assertion failed: expected exactly {expected_real_nbytes} bytes (2 MiB), "
            f"got {actual_state_nbytes} bytes"
        )

    # Overall Acceptance Decision
    overall_pass = bool(
        direct_report["overall_passed"]
        and async_report["overall_passed"]
        and backward_report["overall_passed"]
    )

    peak_allocated = torch.cuda.max_memory_allocated(target_device)
    peak_reserved = torch.cuda.max_memory_reserved(target_device)

    report: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.time(),
            "device": str(args.device),
            "seed": seed,
            "base_checkpoint": str(args.checkpoint),
            "base_metadata": meta,
            "adapter_path": str(args.adapter) if args.adapter else None,
            "adapter_sha256": adapter_sha,
            "adapter_provenance": adapter_provenance,
            "adapter_stage": args.adapter_stage,
            "synthetic_open_gate": synthetic_open_gate,
            "flow_steps": args.flow_steps,
            "threads": args.threads,
            "episode_id": ep_id,
            "prompt": prompt,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        },
        "flash_attention_2": fa2_diagnostics,
        "config": dataclasses.asdict(moss_config),
        "direct_verification": {
            k: v for k, v in direct_report.items() if not k.endswith("_outputs")
        },
        "async_verification": async_report,
        "backward_probe": backward_report,
        "summary": {
            "overall_pass": bool(overall_pass),
            "direct_pass": bool(direct_report["overall_passed"]),
            "async_pass": bool(async_report["overall_passed"]),
            "backward_pass": bool(backward_report["overall_passed"]),
        },
    }

    return report


def run_verification(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"Output directory {output_dir} exists and is not empty. Refusing to overwrite."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"

    try:
        report = _verify_delta_impl(args, output_dir)
    except Exception as exc:
        err_report: Dict[str, Any] = {
            "passed": False,
            "overall_pass": False,
            "error": str(exc),
            "timestamp": time.time(),
        }
        try:
            report_path.write_text(json.dumps(err_report, indent=2), encoding="utf-8")
        except Exception:
            pass
        raise

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not report.get("summary", {}).get("overall_pass", False):
        error_msg = f"Delta verification failed! See details in {report_path}"
        report["error"] = error_msg
        report["passed"] = False
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(error_msg)

    report["passed"] = True
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run_verification(args)
        print(f"[verify_delta] Verification PASSED successfully! Report saved to {args.output_dir}/report.json")
    except Exception as e:
        print(f"[verify_delta] Verification FAILED with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
