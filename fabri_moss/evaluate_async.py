"""Asynchronous decoupled evaluation runner for FabriVLA / MOSS on MetaWorld tasks.

Supports:
- Evaluation modes: native single-frame baseline vs moss decoupled visual planning
- Control modes: step_wait (simulation waits for inference) vs realtime (clock-bounded with zero fallback)
- Strict action ledger alignment: actions anchored at plan.source_frame_id, executed at step - source_frame_id
- Deadlock-free starvation recovery: coordinates inflight plans, pending tasks, and ready queues with total deadline
- Realtime zero fallback directly in raw environment action space (np.zeros(action_space.shape)) without denormalization
- Provenance and diagnostic logging in per-command logs, JSONL, and summary report
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
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
from fabri_moss.native_async import make_native_cache_callbacks, validate_native_memory
from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.data import normalize_and_mask
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint


# ============================================================================
# Action Ledger & Acceptance Functions
# ============================================================================


def action_for_step(plan: PlanResult, current_step: int) -> Optional[np.ndarray]:
    """Extract action slice for current_step from plan anchored at source_frame_id.

    Returns normalized action [D] if 0 <= current_step - source_frame_id < H, else None.
    """
    if not isinstance(plan, PlanResult):
        raise TypeError(f"plan must be a PlanResult, got {type(plan).__name__}")

    idx = current_step - plan.source_frame_id
    actions = plan.computation.actions
    if actions.ndim == 3:
        actions = actions[0]
    horizon = actions.shape[0]

    if 0 <= idx < horizon:
        return actions[idx].detach().cpu().numpy()
    return None


def should_accept_plan(
    candidate: PlanResult,
    current_generation: int,
    accepted_source_frame_id: int,
    current_step: int,
) -> bool:
    """Determine whether candidate PlanResult should replace currently accepted plan.

    Invariants:
    1. Candidate must belong to the active episode generation.
    2. Candidate source_frame_id must be strictly newer than already accepted source_frame_id.
    3. Candidate source_frame_id cannot be in the future beyond current_step (source <= current_step).
    4. Candidate plan must not already be expired at current_step:
       (current_step - candidate.source_frame_id) < candidate.computation.actions.shape[1 if 3D else 0].
    """
    if not isinstance(candidate, PlanResult):
        return False
    if candidate.generation != current_generation:
        return False
    if candidate.source_frame_id <= accepted_source_frame_id:
        return False
    if candidate.source_frame_id > current_step:
        return False

    act = candidate.computation.actions
    horizon = act.shape[1] if act.ndim == 3 else act.shape[0]
    if (current_step - candidate.source_frame_id) >= horizon:
        return False
    return True


def preprocess_observation_image(
    raw_rgb: np.ndarray,
    image_size: int = 448,
) -> Image.Image:
    """Official MetaWorld evaluation image normalization pipeline.

    np.uint8 env.render -> rotate 180 -> center crop 2/3 -> resize 448 INTER_LINEAR -> PIL.Image.
    """
    if not isinstance(raw_rgb, np.ndarray):
        raise TypeError(f"raw_rgb must be np.ndarray, got {type(raw_rgb).__name__}")
    if raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
        raise ValueError(f"raw_rgb must have shape [H, W, 3], got {raw_rgb.shape}")

    rgb = cv2.rotate(raw_rgb, cv2.ROTATE_180)
    rgb = np.ascontiguousarray(rgb)

    h, w = rgb.shape[:2]
    keep_ratio = 2.0 / 3.0
    new_h = max(1, int(round(h * keep_ratio)))
    new_w = max(1, int(round(w * keep_ratio)))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    rgb = rgb[y0 : y0 + new_h, x0 : x0 + new_w, :].copy()

    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    return Image.fromarray(rgb)


def denormalize_action(
    action_norm: np.ndarray,
    action_stats: Dict[str, Any],
    action_space_low: Optional[np.ndarray] = None,
    action_space_high: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Denormalize action vector using native formula:

    (act + 1.0) / 2.0 * (max - min + 1e-8) + min, clipping to env bounds for the first 4 dims.
    """
    act = np.asarray(action_norm, dtype=np.float32).copy()
    a_min = np.asarray(action_stats["min"], dtype=np.float32)
    a_max = np.asarray(action_stats["max"], dtype=np.float32)

    dim = min(act.shape[-1], a_min.shape[0])
    act_slice = act[:dim]
    denorm = (act_slice + 1.0) / 2.0 * (a_max[:dim] - a_min[:dim] + 1e-8) + a_min[:dim]

    first4 = denorm[:4]
    if action_space_low is not None and action_space_high is not None:
        first4 = np.clip(first4, action_space_low[:4], action_space_high[:4])
    return first4.astype(np.float32)


# ============================================================================
# Native Callbacks Helper (Single-frame Baseline via AsyncVisualPlanner)
# ============================================================================


def make_native_callbacks(
    policy: Any,
    device: str = "cpu",
) -> Tuple[
    Callable[[Observation], Any],
    Callable[[Tuple[EncodedFrame, ...], str], PlanComputation],
    Callable[[], None],
]:
    """Construct callbacks for native single-frame baseline under AsyncVisualPlanner.

    - encode: lightweight pass-through (no ViT pre-computation, payload is raw observation).
    - plan: runs full embedder + action_head on the latest observation only.
    - validate: verifies policy is in eval mode.
    """
    target_device = torch.device(device)

    def validate() -> None:
        if policy.training:
            raise RuntimeError("Native policy must be in eval mode during evaluation")

    def encode(obs: Observation) -> Any:
        validate()
        return {"raw_frame_id": obs.frame_id}

    def plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        validate()
        latest_obs = frames[-1].observation

        img_list = list(latest_obs.images)
        image_mask = torch.ones(len(img_list), dtype=torch.bool, device=target_device)

        state = latest_obs.state.to(target_device)
        state_mask = latest_obs.state_mask.to(target_device)
        action_mask = latest_obs.action_mask.to(target_device)

        with torch.no_grad():
            if hasattr(policy, "run_inference"):
                actions = policy.run_inference(
                    images=img_list,
                    image_mask=image_mask,
                    prompt=prompt,
                    state=state,
                    action_mask=action_mask,
                    state_mask=state_mask,
                )
            else:
                actions = policy(
                    images=img_list,
                    image_mask=image_mask,
                    prompt=prompt,
                    state=state,
                    action_mask=action_mask,
                    state_mask=state_mask,
                )

            actions_cpu = actions.detach().to(dtype=torch.float32, device="cpu")
            if actions_cpu.ndim == 2:
                actions_cpu = actions_cpu.unsqueeze(0)

        return PlanComputation(actions=actions_cpu, deep=None, shallow=None)

    return encode, plan, validate


# ============================================================================
# Environment Factory & Task Mapping
# ============================================================================


def resolve_task_prompt(
    task_name: str,
    fabri_root: str | Path = "/root/FabriVLA",
    prompt_override: Optional[str] = None,
) -> str:
    """Resolve exact task prompt according to official MetaWorld task metadata.

    Reads mt50_order.json and tasks.jsonl if available. Falls back to CLI prompt_override.
    """
    if prompt_override is not None and prompt_override.strip():
        return prompt_override.strip()

    fabri_root = Path(fabri_root)
    candidate_paths = [fabri_root / "evaluations" / "metaworld"]

    for eval_dir in candidate_paths:
        order_path = eval_dir / "mt50_order.json"
        tasks_path = eval_dir / "tasks.jsonl"
        if order_path.exists() and tasks_path.exists():
            try:
                with order_path.open("r", encoding="utf-8") as f:
                    order = json.load(f)
                idx_to_slug = order.get("idx_to_slug", {})
                slug_to_idx = {slug: int(idx) for idx, slug in idx_to_slug.items()}

                if task_name in slug_to_idx:
                    target_idx = slug_to_idx[task_name]
                    with tasks_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            item = json.loads(line)
                            if item.get("task_index") == target_idx:
                                return str(item.get("task"))
            except Exception:
                pass

    raise RuntimeError(
        f"Could not resolve official prompt for task '{task_name}'. "
        f"Provide explicit prompt via --prompt or ensure mt50_order.json and tasks.jsonl exist."
    )


def make_metaworld_env(task_name: str, seed: int) -> Tuple[Any, Any]:
    """Instantiate MetaWorld environment.

    Uses metaworld.MT1(task_name, seed=seed), env = mt1.train_classes[task_name](...).
    """
    try:
        import metaworld
    except ImportError as e:
        raise RuntimeError("metaworld package is required for evaluation but not installed.") from e

    mt1 = metaworld.MT1(task_name, seed=seed)
    if task_name not in mt1.train_classes:
        available = list(mt1.train_classes.keys())
        raise ValueError(f"Task '{task_name}' not found in MetaWorld MT1 tasks: {available}")

    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode="rgb_array", camera_name="corner2", width=480, height=480)

    train_tasks = mt1.train_tasks
    if not train_tasks:
        raise RuntimeError(f"No train_tasks available for task {task_name}")

    env.set_task(train_tasks[0])
    return env, train_tasks


def capture_observation(
    env: Any,
    step_id: int,
    norm_stats: Dict[str, Any],
    real_env_obs: np.ndarray,
    capture_time: Optional[float] = None,
    image_size: int = 448,
    state_dim: int = 24,
    action_dim: int = 24,
    observation_time: Optional[float] = None,
) -> Observation:
    """Capture RGB from environment and build normalized Observation dataclass."""
    t_cap = time.monotonic() if capture_time is None else capture_time

    # Resolve observation_time: if not provided, try positive finite env.dt * step_id
    obs_time = observation_time
    if obs_time is None:
        env_dt = getattr(env, "dt", None)
        if env_dt is not None and not isinstance(env_dt, bool) and isinstance(env_dt, (int, float)) and math.isfinite(env_dt) and env_dt > 0.0:
            obs_time = float(step_id * env_dt)

    rgb_raw = np.ascontiguousarray(env.render(), dtype=np.uint8)
    pil_img = preprocess_observation_image(rgb_raw, image_size=image_size)

    raw_obs = np.asarray(real_env_obs, dtype=np.float32)
    st_stats = norm_stats["observation.state"]
    raw_st_len = len(st_stats["min"])
    if raw_obs.shape[0] < raw_st_len:
        raise ValueError(f"real_env_obs length {raw_obs.shape[0]} is shorter than stats length {raw_st_len}")
    valid_state = raw_obs[:raw_st_len]

    state_pad, state_mask = normalize_and_mask(
        valid_state,
        st_stats["min"],
        st_stats["max"],
        target_dim=state_dim,
    )

    action_mask = np.zeros(action_dim, dtype=np.float32)
    action_mask[:4] = 1.0

    return Observation(
        frame_id=step_id,
        capture_time=t_cap,
        images=[pil_img],
        state=torch.from_numpy(state_pad).unsqueeze(0),
        state_mask=torch.from_numpy(state_mask).unsqueeze(0).bool(),
        action_mask=torch.from_numpy(action_mask).unsqueeze(0).bool(),
        observation_time=obs_time,
    )


# ============================================================================
# Starvation Recovery Helper
# ============================================================================


def recover_starvation_plan(
    planner: AsyncVisualPlanner,
    env: Any,
    current_step: int,
    real_env_obs: np.ndarray,
    norm_stats: Dict[str, Any],
    current_generation: int,
    accepted_source_id: int,
    wait_timeout: float,
    image_size: int = 448,
    state_dim: int = 24,
    action_dim: int = 24,
) -> Tuple[PlanResult, int]:
    """Recover from action exhaustion in step_wait mode without deadlock.

    Handles states where:
    - A planner task is already running/queued (ready is empty, planner busy/pending).
    - Result is ready or completes while waiting.
    - No planner task is active: submits current observation, waits for ready, and requests plan.
    Enforces a single total deadline across the recovery attempts.
    Returns (accepted_plan, rejected_count).
    """
    deadline = time.monotonic() + wait_timeout
    rejected_count = 0

    while time.monotonic() < deadline:
        # Check if a plan is already completed
        polled = planner.poll_plan()
        if polled is not None:
            if should_accept_plan(
                candidate=polled,
                current_generation=current_generation,
                accepted_source_frame_id=accepted_source_id,
                current_step=current_step,
            ):
                return polled, rejected_count
            else:
                rejected_count += 1
                # Expired or stale plan, continue loop to wait or request fresh one
                continue

        st = planner.stats()
        is_busy = (
            st.get("planner_busy", False)
            or st.get("plan_pending", False)
            or st.get("result_ready", False)
            or bool(st.get("planner_inflight_frame_ids"))
            or bool(st.get("reservation_frame_ids"))
        )

        if is_busy:
            # Planner is already working; wait for plan with remaining timeout
            rem = max(0.01, deadline - time.monotonic())
            new_res = planner.wait_plan(timeout=rem)
            if new_res is not None:
                if should_accept_plan(
                    candidate=new_res,
                    current_generation=current_generation,
                    accepted_source_frame_id=accepted_source_id,
                    current_step=current_step,
                ):
                    return new_res, rejected_count
                else:
                    rejected_count += 1
                    continue
            else:
                # Timeout or empty poll
                continue

        # Planner is idle: ensure latest observation is submitted
        cur_st = planner.stats()
        if cur_st["last_submitted_id"] < current_step:
            obs = capture_observation(
                env=env,
                step_id=current_step,
                norm_stats=norm_stats,
                real_env_obs=real_env_obs,
                image_size=image_size,
                state_dim=state_dim,
                action_dim=action_dim,
            )
            planner.submit(obs)

        # Wait for at least 1 frame ready
        rem = max(0.01, deadline - time.monotonic())
        if not planner.wait_ready(min_frames=1, timeout=rem):
            continue

        if planner.request_plan():
            rem = max(0.01, deadline - time.monotonic())
            new_res = planner.wait_plan(timeout=rem)
            if new_res is not None:
                if should_accept_plan(
                    candidate=new_res,
                    current_generation=current_generation,
                    accepted_source_frame_id=accepted_source_id,
                    current_step=current_step,
                ):
                    return new_res, rejected_count
                else:
                    rejected_count += 1
                    continue

    raise TimeoutError(
        f"Step {current_step}: Starvation recovery timed out after {wait_timeout}s "
        f"(rejected stale/expired plans: {rejected_count})"
    )


# ============================================================================
# Main Episode Rollout Loop
# ============================================================================


def run_episode(
    env: Any,
    planner: AsyncVisualPlanner,
    episode_id: str,
    prompt: str,
    norm_stats: Dict[str, Any],
    control_mode: str = "step_wait",
    episode_horizon: int = 20,
    exec_horizon: int = 5,
    control_hz: float = 30.0,
    observation_stride: int = 1,
    wait_timeout: float = 120.0,
    action_dim: int = 24,
    state_dim: int = 24,
    image_size: int = 448,
    seed: int = 4048,
) -> Dict[str, Any]:
    """Execute a single episode rollout using AsyncVisualPlanner."""
    planner.reset(episode_id=episode_id, prompt=prompt)
    curr_gen = planner.stats()["generation"]

    # Reset environment
    env_obs, _ = env.reset(seed=seed)
    # Zero-action warmup
    try:
        zero_warmup = np.zeros(env.action_space.shape, dtype=np.float32)
        env_obs, _, _, _, _ = env.step(zero_warmup)
    except Exception as e:
        logging.warning("Warmup step raised: %s", e)

    current_step = 0
    accepted_plan: Optional[PlanResult] = None
    accepted_source_id = -1
    steps_since_plan_request = 0

    # Command logs and diagnostics
    command_log: List[Dict[str, Any]] = []
    plan_records: List[Dict[str, Any]] = []
    action_offsets: List[int] = []
    wait_durations: List[float] = []
    deadline_misses = 0
    realtime_zero_fallbacks = 0
    observation_ages: List[float] = []
    executed_actions_count = 0
    rejected_plans_count = 0
    success = False
    stop_reason = "horizon_reached"

    # Initial frame 0 submission
    obs_0 = capture_observation(
        env=env,
        step_id=0,
        norm_stats=norm_stats,
        real_env_obs=env_obs,
        image_size=image_size,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    planner.submit(obs_0)

    # Initial plan bootstrap
    t_wait_start = time.monotonic()
    if not planner.wait_ready(min_frames=1, timeout=wait_timeout):
        raise TimeoutError(f"Timed out waiting for initial observation frame 0 ready (timeout={wait_timeout}s)")

    if not planner.request_plan():
        # A previous episode's planner worker may still be retiring a stale
        # generation.  Wait for that task to clear instead of treating the
        # transient false return as a model failure.
        deadline = time.monotonic() + wait_timeout
        requested = False
        while time.monotonic() < deadline:
            if planner.request_plan():
                requested = True
                break
            time.sleep(0.01)
        if not requested:
            raise RuntimeError("Failed to request initial plan from frame 0 after waiting for planner idle")

    init_res = planner.wait_plan(timeout=wait_timeout)
    if init_res is None or not should_accept_plan(init_res, curr_gen, -1, 0):
        raise TimeoutError(f"Timed out waiting for valid initial plan (timeout={wait_timeout}s)")

    init_wait = time.monotonic() - t_wait_start
    wait_durations.append(init_wait)

    accepted_plan = init_res
    accepted_source_id = init_res.source_frame_id
    steps_since_plan_request = 0

    plan_records.append(
        {
            "plan_index": len(plan_records),
            "source_frame_id": accepted_plan.source_frame_id,
            "snapshot_frame_ids": [f.observation.frame_id for f in accepted_plan.frames],
            "snapshot_observation_times": [f.observation.observation_time for f in accepted_plan.frames],
            "latest_observation_time": accepted_plan.frames[-1].observation.observation_time if accepted_plan.frames else None,
            "capture_time": accepted_plan.capture_time,
            "start": accepted_plan.started,
            "finish": accepted_plan.finished,
            "latency": accepted_plan.finished - accepted_plan.started,
        }
    )

    # Realtime clock baseline: anchor directly after initial bootstrap
    dt_tick = 1.0 / float(control_hz)
    next_tick_deadline = time.monotonic() + dt_tick

    try:
        while current_step < episode_horizon:
            # 1. Asynchronously poll completed plan
            polled = planner.poll_plan()
            if polled is not None:
                if should_accept_plan(
                    candidate=polled,
                    current_generation=curr_gen,
                    accepted_source_frame_id=accepted_source_id,
                    current_step=current_step,
                ):
                    accepted_plan = polled
                    accepted_source_id = polled.source_frame_id
                    steps_since_plan_request = 0
                    plan_records.append(
                        {
                            "plan_index": len(plan_records),
                            "source_frame_id": accepted_plan.source_frame_id,
                            "snapshot_frame_ids": [f.observation.frame_id for f in accepted_plan.frames],
                            "snapshot_observation_times": [f.observation.observation_time for f in accepted_plan.frames],
                            "latest_observation_time": accepted_plan.frames[-1].observation.observation_time if accepted_plan.frames else None,
                            "capture_time": accepted_plan.capture_time,
                            "start": accepted_plan.started,
                            "finish": accepted_plan.finished,
                            "latency": accepted_plan.finished - accepted_plan.started,
                        }
                    )
                else:
                    rejected_plans_count += 1

            # 2. Extract action for current step
            action_norm = None
            if accepted_plan is not None:
                action_norm = action_for_step(accepted_plan, current_step)

            is_fallback = False
            raw_action_to_step: Optional[np.ndarray] = None

            # 3. Handle action exhaustion / starvation
            if action_norm is None:
                if control_mode == "step_wait":
                    t_w0 = time.monotonic()
                    new_plan, rej = recover_starvation_plan(
                        planner=planner,
                        env=env,
                        current_step=current_step,
                        real_env_obs=env_obs,
                        norm_stats=norm_stats,
                        current_generation=curr_gen,
                        accepted_source_id=accepted_source_id,
                        wait_timeout=wait_timeout,
                        image_size=image_size,
                        state_dim=state_dim,
                        action_dim=action_dim,
                    )
                    wait_dur = time.monotonic() - t_w0
                    wait_durations.append(wait_dur)
                    rejected_plans_count += rej

                    accepted_plan = new_plan
                    accepted_source_id = new_plan.source_frame_id
                    steps_since_plan_request = 0
                    plan_records.append(
                        {
                            "plan_index": len(plan_records),
                            "source_frame_id": accepted_plan.source_frame_id,
                            "snapshot_frame_ids": [f.observation.frame_id for f in accepted_plan.frames],
                            "snapshot_observation_times": [f.observation.observation_time for f in accepted_plan.frames],
                            "latest_observation_time": accepted_plan.frames[-1].observation.observation_time if accepted_plan.frames else None,
                            "capture_time": accepted_plan.capture_time,
                            "start": accepted_plan.started,
                            "finish": accepted_plan.finished,
                            "latency": accepted_plan.finished - accepted_plan.started,
                        }
                    )

                    action_norm = action_for_step(accepted_plan, current_step)
                    if action_norm is None:
                        raise RuntimeError(
                            f"Step {current_step}: could not extract action from recovered plan "
                            f"(source={accepted_plan.source_frame_id})"
                        )
                elif control_mode == "realtime":
                    # P0 1: raw zero action directly in env space without denormalization
                    is_fallback = True
                    realtime_zero_fallbacks += 1
                    raw_action_to_step = np.zeros(env.action_space.shape, dtype=np.float32)

            # 4. Denormalize action if not fallback
            if not is_fallback:
                assert action_norm is not None
                raw_action_to_step = denormalize_action(
                    action_norm=action_norm,
                    action_stats=norm_stats["action"],
                    action_space_low=env.action_space.low,
                    action_space_high=env.action_space.high,
                )
                action_offset = current_step - accepted_plan.source_frame_id
                action_offsets.append(action_offset)
                obs_age = time.monotonic() - accepted_plan.capture_time
                observation_ages.append(obs_age)
            else:
                action_offset = -1
                obs_age = -1.0

            # Record per-command log
            command_log.append(
                {
                    "step": current_step,
                    "action_offset": action_offset,
                    "source_frame_id": accepted_plan.source_frame_id if (not is_fallback and accepted_plan) else None,
                    "snapshot_frame_ids": (
                        [f.observation.frame_id for f in accepted_plan.frames]
                        if (not is_fallback and accepted_plan)
                        else []
                    ),
                    "snapshot_observation_times": (
                        [f.observation.observation_time for f in accepted_plan.frames]
                        if (not is_fallback and accepted_plan)
                        else []
                    ),
                    "latest_observation_time": (
                        accepted_plan.frames[-1].observation.observation_time
                        if (not is_fallback and accepted_plan and accepted_plan.frames)
                        else None
                    ),
                    "capture_time": accepted_plan.capture_time if (not is_fallback and accepted_plan) else None,
                    "is_fallback": is_fallback,
                    "action": raw_action_to_step.tolist(),
                }
            )

            # 5. Step environment
            env_obs, reward, terminated, truncated, info = env.step(raw_action_to_step)
            executed_actions_count += 1
            current_step += 1
            steps_since_plan_request += 1

            if bool(info.get("success", False)):
                success = True
                stop_reason = "env_success"
                break

            if terminated or truncated:
                stop_reason = "env_terminal"
                break

            # 6. Submit next observation according to observation_stride
            if current_step % observation_stride == 0:
                obs_next = capture_observation(
                    env=env,
                    step_id=current_step,
                    norm_stats=norm_stats,
                    real_env_obs=env_obs,
                    image_size=image_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                )
                planner.submit(obs_next)

            # 7. Request next plan if interval reached and planner idle
            if steps_since_plan_request >= exec_horizon:
                cur_st = planner.stats()
                is_busy = (
                    cur_st.get("planner_busy", False)
                    or cur_st.get("plan_pending", False)
                    or bool(cur_st.get("planner_inflight_frame_ids"))
                    or bool(cur_st.get("reservation_frame_ids"))
                )
                if not is_busy and len(cur_st["ready_frame_ids"]) > 0:
                    if planner.request_plan():
                        steps_since_plan_request = 0

            # 8. Realtime clock pacing (no catch-up burst)
            if control_mode == "realtime":
                now = time.monotonic()
                sleep_time = next_tick_deadline - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    next_tick_deadline += dt_tick
                else:
                    deadline_misses += 1
                    # P0 7: Re-base deadline to prevent burst
                    next_tick_deadline = now + dt_tick

    finally:
        final_st = planner.stats()

    return {
        "episode_id": episode_id,
        "success": bool(success),
        "steps": current_step,
        "stop_reason": stop_reason,
        "executed_actions_count": executed_actions_count,
        "plans_completed": final_st.get("plans_completed", 0),
        "dropped_pending": final_st.get("dropped_pending", 0),
        "dropped_ready": final_st.get("dropped_ready", 0),
        "rejected_pending": final_st.get("rejected_pending", 0),
        "time_source": final_st.get("time_source", None),
        "stale_outputs_discarded": final_st.get("stale_outputs_discarded", 0),
        "rejected_plans_count": rejected_plans_count,
        "last_consumed_id": final_st.get("last_consumed_id", -1),
        "action_offsets": action_offsets,
        "mean_action_offset": float(np.mean(action_offsets)) if action_offsets else 0.0,
        "wait_durations": wait_durations,
        "total_wait_time": float(np.sum(wait_durations)) if wait_durations else 0.0,
        "deadline_misses": deadline_misses,
        "realtime_zero_fallbacks": realtime_zero_fallbacks,
        "observation_ages": observation_ages,
        "plans": plan_records,
        "command_log": command_log,
        "pipeline_events": final_st.get("events", []),
    }


# ============================================================================
# CLI & Entrypoint
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabriVLA Decoupled Asynchronous Evaluation Runner")
    parser.add_argument("--mode", type=str, required=True, choices=["native", "moss", "native-cache"], help="Evaluation mode")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to native FabriVLA repo")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Base checkpoint .pt")
    parser.add_argument("--vlm", type=str, default="/root/models/InternVL3_5-1B", help="Path to local InternVL3.5 directory")
    parser.add_argument("--adapter", type=str, default=None, help="Path to MOSS adapter checkpoint (required for moss mode)")
    parser.add_argument("--adapter-stage", type=str, default="bridge", choices=["bridge", "expert", "joint"], help="Stage when loading adapter")
    parser.add_argument("--memory-mode", type=str, default="consume", choices=["consume", "delta"], help="Visual memory mode for MOSS")
    parser.add_argument("--native-memory", type=str, default="window", choices=["window", "periodic", "compact-temporal"], help="Native cache memory compression mode (native-cache mode only)")
    parser.add_argument("--window", type=int, default=None, help="Visual observation memory window size (moss/native mode only)")
    parser.add_argument("--history-frames", type=int, default=16, help="Native cache historical frame capacity (native-cache mode only)")
    parser.add_argument("--max-pending", type=int, default=None, help="Max pending raw observations queue size (None for unbounded dynamic native)")
    parser.add_argument("--max-ready", type=int, default=None, help="Max ready observations capacity limit (native dynamic mode only)")
    parser.add_argument("--timestamp-mode", type=str, default="text", choices=["text", "none"], help="Timestamp mode for native-cache (text enables visible timestamps, none for baseline)")
    parser.add_argument("--task", type=str, default="reach-v3", help="MetaWorld task name (e.g. reach-v3)")
    parser.add_argument("--prompt", type=str, default=None, help="Explicit prompt override")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--episode-horizon", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--exec-horizon", type=int, default=5, help="Steps between plan requests")
    parser.add_argument("--num-inference-timesteps", type=int, default=1, help="Diffusion inference timesteps")
    parser.add_argument("--seed", type=int, default=4048, help="Random seed")
    parser.add_argument("--control-mode", type=str, default="step_wait", choices=["step_wait", "realtime"], help="Control pacing mode")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Control loop rate for realtime mode (Hz)")
    parser.add_argument("--observation-stride", type=int, default=1, help="Observation capture stride in steps")
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads")
    parser.add_argument("--device", type=str, default="cuda:0", help="Model device; GPU by default, cpu for functional tests")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results and logs")
    parser.add_argument("--wait-timeout", type=float, default=120.0, help="Timeout in seconds for worker operations")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key in norm stats")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Input validation
    if args.num_inference_timesteps <= 0:
        raise ValueError("--num-inference-timesteps must be positive")
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    if args.episode_horizon <= 0:
        raise ValueError(f"--episode-horizon must be positive, got {args.episode_horizon}")
    if args.exec_horizon <= 0:
        raise ValueError(f"--exec-horizon must be positive, got {args.exec_horizon}")
    if not math.isfinite(args.control_hz) or args.control_hz <= 0.0:
        raise ValueError(f"--control-hz must be positive finite float, got {args.control_hz}")

    # Mode-specific window and queue validation
    if args.mode == "native-cache":
        if args.window is not None:
            raise ValueError(
                "--window is forbidden and unsupported when --mode=native-cache; "
                "native cache uses dynamic unconstrained snapshots of all ready frames. "
                "Use --history-frames to control historical memory capacity and "
                "--max-pending / --max-ready to set explicit resource limits."
            )
        resolved_window = None
        resolved_max_pending = args.max_pending  # None by default (unbounded dynamic)
        if resolved_max_pending is not None and resolved_max_pending <= 0:
            raise ValueError(f"--max-pending must be positive if specified, got {resolved_max_pending}")
        resolved_max_ready = args.max_ready
        if resolved_max_ready is not None and resolved_max_ready <= 0:
            raise ValueError(f"--max-ready must be positive if specified, got {resolved_max_ready}")
    elif args.mode == "moss":
        resolved_window = args.window if args.window is not None else 5
        if resolved_window <= 0:
            raise ValueError(f"--window must be positive, got {resolved_window}")
        resolved_max_pending = args.max_pending if args.max_pending is not None else 8
        if resolved_max_pending <= 0:
            raise ValueError(f"--max-pending must be positive, got {resolved_max_pending}")
        if args.max_ready is not None:
            raise ValueError("--max-ready is only allowed for native dynamic cache mode")
        resolved_max_ready = None
    else:  # native single-frame baseline
        resolved_window = args.window if args.window is not None else 1
        if resolved_window <= 0:
            raise ValueError(f"--window must be positive, got {resolved_window}")
        resolved_max_pending = args.max_pending if args.max_pending is not None else 8
        if resolved_max_pending <= 0:
            raise ValueError(f"--max-pending must be positive, got {resolved_max_pending}")
        if args.max_ready is not None:
            raise ValueError("--max-ready is only allowed for native dynamic cache mode")
        resolved_max_ready = None

    if args.native_memory == "periodic":
        if args.mode != "native-cache":
            raise ValueError("--native-memory=periodic is only allowed when --mode=native-cache")
        if args.timestamp_mode != "text":
            raise ValueError("--native-memory=periodic requires --timestamp-mode=text (periodic memory requires real model observation timestamps)")
    elif args.native_memory == "compact-temporal":
        if args.mode != "native-cache":
            raise ValueError("--native-memory=compact-temporal is only allowed when --mode=native-cache")
        if args.timestamp_mode != "text":
            raise ValueError("--native-memory=compact-temporal requires --timestamp-mode=text (compact temporal memory requires real model observation timestamps)")

    if args.history_frames <= 0:
        raise ValueError(f"--history-frames must be positive, got {args.history_frames}")
    if args.observation_stride <= 0:
        raise ValueError(f"--observation-stride must be positive, got {args.observation_stride}")
    if not math.isfinite(args.wait_timeout) or args.wait_timeout <= 0.0:
        raise ValueError(f"--wait-timeout must be positive finite float, got {args.wait_timeout}")
    if args.threads <= 0:
        raise ValueError(f"--threads must be positive, got {args.threads}")

    if args.mode == "moss" and not args.adapter:
        raise ValueError("--adapter is strictly required when --mode=moss")
    if args.mode == "native-cache" and args.adapter is not None:
        raise ValueError("--adapter is forbidden and unsupported when --mode=native-cache (native cache uses zero trainable adapter parameters)")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory {output_dir} exists and is not empty. Refusing to overwrite.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Seed all generators
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    # Smoke test classification: any of flow_steps < 50, episode_horizon < 400, or episodes < 10
    is_smoke = (args.num_inference_timesteps < 50 or args.episode_horizon < 400 or args.episodes < 10)

    # Resolve official prompt
    prompt = resolve_task_prompt(
        task_name=args.task,
        fabri_root=args.fabri_root,
        prompt_override=args.prompt,
    )

    use_timestamps = (args.timestamp_mode == "text") if args.mode == "native-cache" else False

    provenance: Dict[str, Any] = {
        "mode": args.mode,
        "control_mode": args.control_mode,
        "memory_mode": args.memory_mode if args.mode == "moss" else None,
        "is_smoke": is_smoke,
        "task": args.task,
        "prompt": prompt,
        "seed": args.seed,
        "window": resolved_window,
        "history_frames": args.history_frames if args.mode == "native-cache" else None,
        "snapshot_mode": "all_ready" if args.mode == "native-cache" else "latest_bounded",
        "new_frames_per_plan": None if args.mode == "native-cache" else resolved_window,
        "max_pending": resolved_max_pending,
        "max_ready": resolved_max_ready,
        "overflow_policy": "error" if args.mode == "native-cache" else "drop_oldest",
        "timestamp_mode": args.timestamp_mode if args.mode == "native-cache" else None,
        "use_timestamps": use_timestamps,
        "exec_horizon": args.exec_horizon,
        "episode_horizon": args.episode_horizon,
        "num_inference_timesteps": args.num_inference_timesteps,
        "fabri_root": args.fabri_root,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "adapter_stage": args.adapter_stage,
        "device": args.device,
        "control_hz": args.control_hz,
        "started_at": time.time(),
    }

    if args.control_mode == "step_wait":
        provenance["note"] = (
            "Simulation pauses only when no usable action remains; asynchronous requests still occur "
            "at exec_horizon intervals. This is not a real-time or official MT50 benchmark."
        )

    episode_results: List[Dict[str, Any]] = []
    jsonl_path = output_dir / "episodes.jsonl"
    env = None
    compact_adapter = None

    try:
        policy, ckpt_cfg, norm_stats, meta = load_native_checkpoint(
            fabri_root=args.fabri_root,
            checkpoint_path=args.checkpoint,
            vlm_path=args.vlm,
            device=args.device,
            arm_key=args.arm_key,
        )
        provenance["base_metadata"] = meta

        if (args.mode == "moss" and args.memory_mode == "delta") or args.mode == "native-cache":
            fa2_diag = assert_native_fa2(policy)
            provenance["native_fa2_diagnostics"] = fa2_diag

        if hasattr(policy, "action_head") and hasattr(policy.action_head, "config"):
            policy.action_head.config.num_inference_timesteps = args.num_inference_timesteps
        elif hasattr(policy, "config"):
            policy.config.num_inference_timesteps = args.num_inference_timesteps

        moss_model: Optional[MossInternVL] = None
        stateful_planner: bool = False
        memory_validator_fn: Optional[Callable[..., Any]] = None
        if args.mode == "moss":
            moss_config = MossConfig(max_frames=resolved_window, max_text_tokens=1024, memory_mode=args.memory_mode)
            moss_model = MossInternVL(policy, config=moss_config)

            # Set the requested stage before loading adapter weights.
            moss_model.set_training_stage(args.adapter_stage)

            adapter_path = Path(args.adapter).resolve()
            if not adapter_path.exists():
                raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

            adapter_ckpt = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
            if adapter_ckpt.get("format") == "moss_cross_adapter_v2":
                if adapter_ckpt.get("source_checkpoint_sha256") != meta.get("checkpoint_sha256"):
                    raise ValueError("MOSS adapter source checkpoint SHA256 mismatch")
                if adapter_ckpt.get("stage") != args.adapter_stage:
                    raise ValueError(
                        f"MOSS adapter stage {adapter_ckpt.get('stage')!r} does not match requested {args.adapter_stage!r}"
                    )
                if dict(adapter_ckpt.get("config", {})) != dataclasses.asdict(moss_model.config):
                    raise ValueError("MOSS adapter config mismatch")
                moss_model.cross_blocks.load_state_dict(adapter_ckpt["cross_blocks"], strict=True)
                with torch.no_grad():
                    moss_model.readout_embeddings.copy_(adapter_ckpt["readout_embeddings"].to(moss_model.readout_embeddings.device))
                if args.adapter_stage in ("expert", "joint"):
                    moss_model.policy.action_head.load_state_dict(adapter_ckpt["action_head"], strict=True)
                if args.adapter_stage == "joint":
                    missing, unexpected = moss_model.policy.load_state_dict(adapter_ckpt["base_policy"], strict=False)
                    if unexpected or any("action_head." not in k for k in missing):
                        raise ValueError(f"invalid joint base_policy state: missing={missing}, unexpected={unexpected}")
                adapter_provenance = {"format": adapter_ckpt["format"], "stage": adapter_ckpt["stage"], "step": adapter_ckpt.get("step")}
            else:
                from fabri_moss.train import initialize_adapter_weights
                adapter_provenance = initialize_adapter_weights(
                    path=adapter_path,
                    student_model=moss_model,
                    expected_norm_stats=norm_stats,
                    current_base_metadata=meta,
                    device=args.device,
                )
            provenance["adapter_provenance"] = adapter_provenance
            moss_model.eval()

            encode_fn, plan_fn, val_fn = make_moss_callbacks(moss_model)
            max_planner_frames = resolved_window
            stateful_planner = (args.memory_mode == "delta")
            memory_validator_fn = None
            planner_overflow_policy = "drop_oldest"
        elif args.mode == "native-cache":
            # Native cache mode: no extra adapter weights/parameters, retains full native attention
            from fabri_moss.native_cache import NativeCacheAdapter, NativeCacheConfig

            if args.native_memory == "periodic":
                from fabri_moss.memory_cache import NativeMemoryCacheAdapter, validate_memory_cache
                from fabri_moss.periodic_memory import PeriodicMemoryConfig

                mem_cfg = PeriodicMemoryConfig()
                cache_config = NativeCacheConfig(
                    max_frames=args.history_frames,
                    shallow_layer=6,
                    use_timestamps=use_timestamps,
                )
                native_model = NativeMemoryCacheAdapter(policy, config=cache_config, memory_config=mem_cfg)
                native_model.eval()

                provenance["native_cache_provenance"] = {
                    "architecture": "native_periodic_visual_memory",
                    "visual_layer": "visual_last_layer",
                    "new_trainable_params": 0,
                    "expert_context_tokens": getattr(policy.embedder, "max_text_length", 1024),
                    "history_frames": None,
                    "native_memory": "periodic",
                    "memory_slots": None,
                    "anchor_policy": "snapshot_latest",
                    "protect_decision_frames": mem_cfg.protect_decision_frames,
                    "bank_capacity": "episode_growing_protected_anchors_and_interval_summaries",
                    "spatial_grid": mem_cfg.spatial_grid,
                    "recent_frames": mem_cfg.recent_frames,
                    "consolidate_every": mem_cfg.consolidate_every,
                    "new_frames_per_plan": None,
                    "snapshot_mode": "all_ready",
                    "use_timestamps": use_timestamps,
                    "overflow_policy": "error",
                }

                encode_fn, plan_fn, val_fn = make_native_cache_callbacks(native_model, prompt=prompt)
                max_planner_frames = None
                stateful_planner = True
                memory_validator_fn = validate_memory_cache
                planner_overflow_policy = "error"
            elif args.native_memory == "compact-temporal":
                from fabri_moss.compact_cache import NativeCompactMemoryCacheAdapter, validate_compact_memory
                from fabri_moss.compact_memory import CompactMemoryConfig
                from fabri_moss.compact_protocol import get_compact_protocol_contract

                compact_cfg = CompactMemoryConfig()
                cache_config = NativeCacheConfig(
                    max_frames=args.history_frames,
                    shallow_layer=6,
                    use_timestamps=use_timestamps,
                )
                native_model = NativeCompactMemoryCacheAdapter(
                    policy,
                    config=cache_config,
                    compact_config=compact_cfg,
                    background_rebuild=True,
                )
                native_model.eval()
                compact_adapter = native_model

                provenance["native_cache_provenance"] = {
                    "architecture": "native_compact_temporal_memory",
                    "visual_layer": "visual_last_layer",
                    "new_trainable_params": 0,
                    "expert_context_tokens": getattr(policy.embedder, "max_text_length", 1024),
                    "history_frames": None,
                    "native_memory": "compact-temporal",
                    "memory_slots": None,
                    "full_current_tokens": 1024,
                    "compact_protocol_contract": get_compact_protocol_contract(),
                    "protect_decision_frames": compact_cfg.memory.protect_decision_frames,
                    "intermediate_grid": compact_cfg.intermediate_grid,
                    "temporal_rope": {
                        "strength": compact_cfg.temporal.strength,
                        "time_unit_seconds": compact_cfg.temporal.time_unit_seconds,
                        "rotary_fraction": compact_cfg.temporal.rotary_fraction,
                    },
                    "lifecycle": {
                        "postquery_consolidation": True,
                        "nextquery_effective": True,
                        "rebuild_future_resolution": "before_next_query",
                    },
                    "recent_frames": compact_cfg.memory.recent_frames,
                    "consolidate_every": compact_cfg.memory.consolidate_every,
                    "new_frames_per_plan": None,
                    "snapshot_mode": "all_ready",
                    "use_timestamps": use_timestamps,
                    "overflow_policy": "error",
                }

                encode_fn, plan_fn, val_fn = make_native_cache_callbacks(native_model, prompt=prompt)
                max_planner_frames = None
                stateful_planner = True
                memory_validator_fn = validate_compact_memory
                planner_overflow_policy = "error"
            else:
                provenance["native_cache_provenance"] = {
                    "architecture": "native_joint_kv",
                    "visual_layer": "visual_last_layer",
                    "new_trainable_params": 0,
                    "expert_context_tokens": getattr(policy.embedder, "max_text_length", 1024),
                    "history_frames": args.history_frames,
                    "native_memory": "window",
                    "new_frames_per_plan": None,
                    "snapshot_mode": "all_ready",
                    "use_timestamps": use_timestamps,
                    "overflow_policy": "error",
                }

                cache_config = NativeCacheConfig(
                    max_frames=args.history_frames,
                    shallow_layer=6,
                    use_timestamps=use_timestamps,
                )
                native_model = NativeCacheAdapter(policy, config=cache_config)
                native_model.eval()

                encode_fn, plan_fn, val_fn = make_native_cache_callbacks(native_model, prompt=prompt)
                max_planner_frames = None  # Native cache uses dynamic unbounded ready snapshot
                stateful_planner = True
                memory_validator_fn = validate_native_memory
                planner_overflow_policy = "error"
        else:
            policy.eval()
            encode_fn, plan_fn, val_fn = make_native_callbacks(
                policy=policy,
                device=args.device,
            )
            max_planner_frames = 1
            stateful_planner = False
            memory_validator_fn = None
            planner_overflow_policy = "drop_oldest"

        env, tasks = make_metaworld_env(task_name=args.task, seed=args.seed)

        with AsyncVisualPlanner(
            encode=encode_fn,
            plan=plan_fn,
            max_frames=max_planner_frames,
            max_pending=resolved_max_pending,
            max_ready=resolved_max_ready,
            overflow_policy=planner_overflow_policy,
            validate=val_fn,
            stateful=stateful_planner,
            memory_validator=memory_validator_fn,
        ) as planner:
            for ep_idx in range(args.episodes):
                ep_id = f"ep_{ep_idx:03d}"
                task_obj = tasks[ep_idx % len(tasks)]
                env.set_task(task_obj)

                ep_res = run_episode(
                    env=env,
                    planner=planner,
                    episode_id=ep_id,
                    prompt=prompt,
                    norm_stats=norm_stats,
                    control_mode=args.control_mode,
                    episode_horizon=args.episode_horizon,
                    exec_horizon=args.exec_horizon,
                    control_hz=args.control_hz,
                    observation_stride=args.observation_stride,
                    wait_timeout=args.wait_timeout,
                    seed=args.seed + ep_idx,
                )
                episode_results.append(ep_res)

                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ep_res) + "\n")

    except Exception as exc:
        partial_report = {
            "status": "aborted_due_to_exception",
            "error": str(exc),
            "provenance": provenance,
            "episodes_completed": len(episode_results),
            "episodes": episode_results,
        }
        with (output_dir / "summary_report.json").open("w", encoding="utf-8") as f:
            json.dump(partial_report, f, indent=2)
        raise exc
    finally:
        if compact_adapter is not None:
            try:
                compact_adapter.close()
            except Exception as e:
                logging.warning("Error closing compact_adapter: %s", e)
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logging.warning("Error closing environment: %s", e)

    total_episodes = len(episode_results)
    success_count = sum(1 for r in episode_results if r["success"])
    success_rate = (success_count / total_episodes) if total_episodes > 0 else 0.0

    final_report = {
        "status": "completed",
        "provenance": provenance,
        "summary": {
            "total_episodes": total_episodes,
            "success_count": success_count,
            "success_rate": success_rate,
            "is_smoke": is_smoke,
            "control_mode": args.control_mode,
            "mode": args.mode,
        },
        "episodes": episode_results,
    }

    with (output_dir / "summary_report.json").open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"[evaluate_async] Finished {total_episodes} episodes. Success rate: {success_rate:.2f}. Saved to {output_dir}")


if __name__ == "__main__":
    main()
