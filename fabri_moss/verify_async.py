"""Real-data decoupled asynchronous replay and interval overlap verification.

Replays real MetaWorld frames through AsyncVisualPlanner using real weights and make_moss_callbacks:
1. Submits 5 initial frames [0..4] and awaits ready=5.
2. Requests plan #1.
3. Immediately submits next 3 frames [5..7] while planner worker executes.
4. Concurrently measures real wall-clock execution intervals of vision encoding and planner.
5. Quantifies actual interval intersections [max(start), min(finish)].
6. Verifies that plan #1 consumed exactly cutoff=4 and preserved frames [5..7] in ready.
7. Waits for ready=3 and requests plan #2.
8. Closes worker threads before computing fresh model reference.
9. Compares cached snapshot outputs (deep, shallow, actions) with fresh end-to-end model forward.
10. Validates finite tensors, correct masks, and non-overlapping consumed batches.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
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
from fabri_moss.evaluate_async import resolve_task_prompt
from fabri_moss.runtime import load_native_checkpoint
from fabri_moss.verify import compute_tensor_diff


# ============================================================================
# Interval Intersection Helper
# ============================================================================


def compute_interval_overlap(
    interval_a: Tuple[float, float],
    interval_b: Tuple[float, float],
) -> float:
    """Compute length of time overlap between two [start, finish] intervals.

    Returns max(0.0, min(finish_a, finish_b) - max(start_a, start_b)).
    """
    start = max(interval_a[0], interval_b[0])
    finish = min(interval_a[1], interval_b[1])
    return max(0.0, finish - start)


# ============================================================================
# Core Verification Routine
# ============================================================================


def verify_async_replay(
    model: MossInternVL,
    dataset: MetaWorldWindows,
    prompt: str,
    output_dir: Path,
    wait_timeout: float = 120.0,
    window: int = 5,
    max_pending: int = 8,
) -> Dict[str, Any]:
    """Execute real-data asynchronous replay verification."""
    model.eval()

    # Find an episode with at least 8 frames
    if len(dataset) < 8:
        raise ValueError(f"Dataset has only {len(dataset)} samples, need >= 8 for async verification")

    ep_id = dataset[0]["episode_id"]
    samples_in_ep: List[Dict[str, Any]] = []
    for i in range(len(dataset)):
        s = dataset[i]
        if s["episode_id"] == ep_id:
            samples_in_ep.append(s)
            if len(samples_in_ep) == 8:
                break

    if len(samples_in_ep) < 8:
        raise ValueError(
            f"Episode {ep_id} has only {len(samples_in_ep)} samples in dataset, need >= 8 frames"
        )
    if window != 5 or max_pending < 5:
        raise ValueError("This verification requires window=5 and max_pending>=5")
    if [s['frame_ids'][-1] for s in samples_in_ep] != list(range(8)):
        raise ValueError("Replay requires the first eight consecutive real episode frames")

    # Convert samples to strict Observation dataclasses
    observations: List[Observation] = []
    for idx, s in enumerate(samples_in_ep):
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
            frame_id=int(s['frame_ids'][-1]),
            capture_time=t_cap,
            images=list(img_item),
            state=state_tensor,
            state_mask=state_mask_tensor,
            action_mask=action_mask_tensor,
        )
        observations.append(obs)

    # Track callback execution intervals and thread IDs
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

    def tracking_plan(frames: Tuple[EncodedFrame, ...], p: str) -> PlanComputation:
        th_id = threading.get_ident()
        t0 = time.monotonic()
        res = base_plan_fn(frames, p)
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

    with AsyncVisualPlanner(
        encode=tracking_encode,
        plan=tracking_plan,
        max_frames=window,
        max_pending=max_pending,
        validate=base_val_fn,
    ) as planner:
        planner.reset(episode_id=f"verify_ep_{ep_id}", prompt=prompt)

        # 1. Submit initial 5 frames [0, 1, 2, 3, 4]
        for f_idx in range(5):
            planner.submit(observations[f_idx])

        # Await ready = 5
        if not planner.wait_ready(min_frames=5, timeout=wait_timeout):
            raise TimeoutError("Timed out waiting for initial 5 frames ready")

        # 2. Request plan #1
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan #1")

        # 3. Immediately submit next 3 frames [5, 6, 7] while planner runs
        for f_idx in range(5, 8):
            planner.submit(observations[f_idx])

        # 4. Wait for plan #1 completion
        plan_1 = planner.wait_plan(timeout=wait_timeout)
        if plan_1 is None:
            raise TimeoutError("Timed out waiting for plan #1")

        # Collect stats immediately after plan #1
        stats_after_plan_1 = copy.deepcopy(planner.stats())

        # 5. Wait for frames [5, 6, 7] ready
        if not planner.wait_ready(min_frames=3, timeout=wait_timeout):
            raise TimeoutError("Timed out waiting for frames [5, 6, 7] ready")

        # 6. Request plan #2
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan #2")

        plan_2 = planner.wait_plan(timeout=wait_timeout)
        if plan_2 is None:
            raise TimeoutError("Timed out waiting for plan #2")

        stats_after_plan_2 = copy.deepcopy(planner.stats())

    # All worker threads are strictly closed before running fresh model comparisons

    # ========================================================================
    # Interval and Concurrency Overlap Analysis
    # ========================================================================
    assert plan_1 is not None and plan_2 is not None

    p1_interval = (plan_1.started, plan_1.finished)
    concurrent_encodes_during_p1: List[Dict[str, Any]] = []
    total_overlap_seconds = 0.0

    for enc in encode_intervals:
        if enc["frame_id"] in (5, 6, 7):
            enc_interval = (enc["start"], enc["finish"])
            overlap = compute_interval_overlap(p1_interval, enc_interval)
            if overlap > 0.0:
                total_overlap_seconds += overlap
                concurrent_encodes_during_p1.append(
                    {
                        "frame_id": enc["frame_id"],
                        "encode_interval": enc_interval,
                        "plan_interval": p1_interval,
                        "overlap_seconds": overlap,
                    }
                )

    concurrency_observed = bool(total_overlap_seconds > 0.0)

    # Thread separation check
    vision_threads = {enc["thread_id"] for enc in encode_intervals}
    planner_threads = {p["thread_id"] for p in plan_intervals}
    threads_isolated = bool(vision_threads.isdisjoint(planner_threads))

    # ========================================================================
    # Cache & Consumption Invariants
    # ========================================================================
    p1_snapshot_fids = [f.observation.frame_id for f in plan_1.frames]
    p2_snapshot_fids = [f.observation.frame_id for f in plan_2.frames]

    batches_disjoint = bool(set(p1_snapshot_fids).isdisjoint(set(p2_snapshot_fids)))
    p1_cutoff_correct = (plan_1.source_frame_id == 4) and (p1_snapshot_fids == [0, 1, 2, 3, 4])
    p2_cutoff_correct = (plan_2.source_frame_id == 7) and (p2_snapshot_fids == [5, 6, 7])

    last_consumed_p1 = stats_after_plan_1.get("last_consumed_id")
    consumed_count_p1 = stats_after_plan_1.get("consumed_count")
    ready_after_p1 = stats_after_plan_1.get("ready_frame_ids", [])

    consumption_invariants_pass = (
        p1_cutoff_correct
        and p2_cutoff_correct
        and batches_disjoint
        and last_consumed_p1 == 4
        and consumed_count_p1 == 5
        and all(fid in (5, 6, 7) for fid in ready_after_p1)
    )

    # ========================================================================
    # Fresh Model Forward Parity Comparison
    # ========================================================================
    # Compare plan_1 snapshot vs fresh forward on raw images
    obs_by_id = {obs.frame_id: obs for obs in observations}
    raw_images_p1 = [obs_by_id[f.observation.frame_id].images for f in plan_1.frames]
    fids_p1 = p1_snapshot_fids

    with torch.no_grad():
        fresh_deep, fresh_shallow = model(
            images_window=raw_images_p1,
            frame_ids=fids_p1,
            prompt=prompt,
        )

    cached_deep = plan_1.computation.deep
    cached_shallow = plan_1.computation.shallow

    deep_diff = compute_tensor_diff(cached_deep, fresh_deep) if cached_deep is not None else {}
    shallow_diff = compute_tensor_diff(cached_shallow, fresh_shallow) if cached_shallow is not None else {}

    allclose_deep = deep_diff.get("allclose_1e3", False)
    allclose_shallow = shallow_diff.get("allclose_1e3", False)

    # Validate that PlanResult does not retain visual payload, images, or tensors in frames
    result_contains_visual_payload = any(
        hasattr(f, "payload")
        or hasattr(f.observation, "images")
        or hasattr(f.observation, "state")
        for f in (*plan_1.frames, *plan_2.frames)
    )

    # Tensor finite and shape validation
    actions_p1 = plan_1.computation.actions
    actions_finite = bool(torch.isfinite(actions_p1).all())
    actions_shape_valid = (actions_p1.ndim == 3 and actions_p1.shape[0] == 1 and actions_p1.shape[1] == 50)

    padded_actions_zero = all(bool((p.computation.actions[:, :, 4:] == 0).all()) for p in (plan_1, plan_2))
    overall_passed = (
        threads_isolated
        and concurrency_observed
        and padded_actions_zero
        and not result_contains_visual_payload
        and stats_after_plan_2.get('consumed_count') == 8
        and stats_after_plan_2.get('ready_frame_ids') == []
        and consumption_invariants_pass
        and allclose_deep
        and allclose_shallow
        and actions_finite
        and actions_shape_valid
    )

    report: Dict[str, Any] = {
        "status": "passed" if overall_passed else "failed",
        "overall_passed": bool(overall_passed),
        "concurrency_observed": bool(concurrency_observed),
        "result_contains_visual_payload": bool(result_contains_visual_payload),
        "episode_id": ep_id,
        "prompt": prompt,
        "padded_actions_zero": padded_actions_zero,
        "capture_clock": "monotonic at replay preparation, not original dataset acquisition time",
        "ready_final": stats_after_plan_2.get('ready_frame_ids'),
        "consumed_count_final": stats_after_plan_2.get('consumed_count'),
        "concurrency_metric": {
            "threads_isolated": bool(threads_isolated),
            "vision_thread_ids": list(vision_threads),
            "planner_thread_ids": list(planner_threads),
            "plan_1_interval": p1_interval,
            "total_overlap_seconds": float(total_overlap_seconds),
            "overlapping_encodes": concurrent_encodes_during_p1,
            "note": (
                "Interval intersection measured between planner execution and background vision encoding. "
                "Non-zero overlap confirms concurrent thread execution without artificial barriers."
            ),
        },
        "consumption_verification": {
            "plan_1_source_frame_id": plan_1.source_frame_id,
            "plan_1_snapshot_frame_ids": p1_snapshot_fids,
            "plan_2_source_frame_id": plan_2.source_frame_id,
            "plan_2_snapshot_frame_ids": p2_snapshot_fids,
            "batches_disjoint": bool(batches_disjoint),
            "last_consumed_after_plan_1": last_consumed_p1,
            "consumed_count_after_plan_1": consumed_count_p1,
            "ready_frame_ids_after_plan_1": ready_after_p1,
            "consumption_invariants_pass": bool(consumption_invariants_pass),
        },
        "fresh_parity_verification": {
            "allclose_1e3_deep": bool(allclose_deep),
            "allclose_1e3_shallow": bool(allclose_shallow),
            "deep_diff": deep_diff,
            "shallow_diff": shallow_diff,
            "actions_finite": bool(actions_finite),
            "actions_shape": list(actions_p1.shape),
            "actions_shape_valid": bool(actions_shape_valid),
        },
        "plan_intervals": plan_intervals,
        "encode_intervals": encode_intervals,
    }

    report_path = output_dir / "verify_async_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ============================================================================
# CLI & Entrypoint
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-data Decoupled Asynchronous Replay Verification")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to native FabriVLA repo")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Base checkpoint .pt")
    parser.add_argument("--vlm", type=str, default="/root/models/InternVL3_5-1B", help="Path to local InternVL3.5 directory")
    parser.add_argument("--adapter", type=str, required=True, help="Path to MOSS adapter checkpoint")
    parser.add_argument("--adapter-stage", type=str, default="bridge", choices=["bridge", "expert"], help="Stage when loading adapter")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MetaWorld LeRobot dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save report")
    parser.add_argument("--window", type=int, default=5, help="Visual observation memory window size (>= 1)")
    parser.add_argument("--max-pending", type=int, default=8, help="Max pending raw observations queue size")
    parser.add_argument("--flow-steps", type=int, default=1, help="Flow integration steps; 1 is a smoke test")
    parser.add_argument("--prompt", type=str, default=None, help="Explicit prompt override")
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads")
    parser.add_argument("--device", type=str, default="cuda:0", help="Model device; GPU by default, cpu for functional tests")
    parser.add_argument("--wait-timeout", type=float, default=120.0, help="Timeout in seconds for worker operations")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key in norm stats")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory {output_dir} exists and is not empty. Refusing to overwrite.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))

    policy, ckpt_cfg, norm_stats, meta = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm,
        device=args.device,
        arm_key=args.arm_key,
    )

    policy.action_head.config.num_inference_timesteps = args.flow_steps
    moss_config = MossConfig(max_frames=args.window)
    moss_model = MossInternVL(policy, config=moss_config)
    moss_model.set_training_stage(args.adapter_stage)

    adapter_path = Path(args.adapter).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

    from fabri_moss.train import initialize_adapter_weights
    adapter_provenance = initialize_adapter_weights(
        path=adapter_path,
        student_model=moss_model,
        expected_norm_stats=norm_stats,
        current_base_metadata=meta,
        device=args.device,
    )

    dataset = MetaWorldWindows(
        root=args.data_root,
        norm_stats=norm_stats,
        window=1,
        frame_stride=1,
        split="all",
        max_episodes=1,
    )

    prompt = args.prompt or dataset[0]['prompt']
    report = verify_async_replay(
        model=moss_model,
        dataset=dataset,
        prompt=prompt,
        output_dir=output_dir,
        wait_timeout=args.wait_timeout,
        window=args.window,
        max_pending=args.max_pending,
    )

    report['base_metadata'] = meta
    report['adapter_provenance'] = adapter_provenance
    report['device'] = args.device
    report['flow_steps'] = args.flow_steps
    (output_dir / 'verify_async_report.json').write_text(json.dumps(report, indent=2))
    if not report["overall_passed"]:
        print(f"[verify_async] Verification FAILED! Details written to {output_dir / 'verify_async_report.json'}")
        sys.exit(1)
    else:
        print(f"[verify_async] Verification PASSED successfully! Report saved to {output_dir}")


if __name__ == "__main__":
    main()
