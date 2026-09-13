"""Synchronous MT50 evaluation runner for FabriVLA / MOSS predictive memory closed-loop.

Implements official MT50 vector evaluation using synchronous direct stepping and
predictive memory closed-loop planning with full-prefix retention.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fabri_moss.evaluate_async import (
    capture_observation,
    denormalize_action,
    resolve_task_prompt,
)


logger = logging.getLogger("evaluate_predictive")


def _seed_torch_device(seed: int, device: torch.device) -> None:
    """Seed CPU and the selected CUDA device without touching other devices."""
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def compute_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_source_fingerprints(repo_root: Path) -> Dict[str, str]:
    """Compute SHA256 hashes of core predictive evaluation files."""
    files_to_hash = [
        repo_root / "fabri_moss" / "evaluate_predictive.py",
        repo_root / "fabri_moss" / "native_single_inference.py",
        repo_root / "fabri_moss" / "compact_inference.py",
        repo_root / "fabri_moss" / "compact_training.py",
        repo_root / "fabri_moss" / "compact_memory.py",
        repo_root / "fabri_moss" / "periodic_memory.py",
        repo_root / "fabri_moss" / "predictive_inference.py",
        repo_root / "fabri_moss" / "predictive_memory.py",
        repo_root / "fabri_moss" / "predictive_policy.py",
        repo_root / "fabri_moss" / "native_training.py",
        repo_root / "fabri_moss" / "native_cache.py",
        repo_root / "fabri_moss" / "temporal_rope.py",
        repo_root / "fabri_moss" / "evaluate_async.py",
        repo_root / "fabri_moss" / "runtime.py",
    ]
    fingerprints: Dict[str, str] = {}
    for f in files_to_hash:
        if f.exists() and f.is_file():
            fingerprints[f.name] = compute_sha256(f)
    return fingerprints


# ============================================================================
# MT50 Manifest & Tasks Resolution
# ============================================================================


def load_mt50_metadata(
    fabri_root: Union[str, Path] = "/root/FabriVLA",
) -> Tuple[List[int], Dict[str, str], Dict[str, List[str]], str]:
    """Load MT50 order, idx-to-slug mapping, groups, and compute metadata SHA."""
    eval_dir = Path(fabri_root) / "evaluations" / "metaworld"
    order_path = eval_dir / "mt50_order.json"
    if not order_path.exists():
        raise FileNotFoundError(
            f"MT50 metadata file not found at {order_path}. "
            "Please ensure --fabri-root points to the FabriVLA repository containing evaluations/metaworld."
        )

    metadata_sha256 = compute_sha256(order_path)
    with order_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ordered_indices = [int(x) for x in data.get("ordered_indices", [])]
    idx_to_slug = {str(k): str(v) for k, v in data.get("idx_to_slug", {}).items()}
    groups = {str(k): [str(x) for x in v] for k, v in data.get("groups", {}).items()}

    if not ordered_indices or not idx_to_slug:
        raise ValueError(f"Invalid or empty MT50 metadata in {order_path}")
    if len(set(ordered_indices)) != len(ordered_indices) or any(i < 0 for i in ordered_indices):
        raise ValueError("MT50 task indices must be unique and nonnegative")
    if {str(i) for i in ordered_indices} != set(idx_to_slug) or len(set(idx_to_slug.values())) != len(idx_to_slug):
        raise ValueError("MT50 order and task mapping must be complete and unique")

    return ordered_indices, idx_to_slug, groups, metadata_sha256


def resolve_task_manifest(
    tasks_arg: Sequence[str],
    fabri_root: Union[str, Path] = "/root/FabriVLA",
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, List[str]], str]:
    """Resolve and validate the selected tasks manifest.

    Validates that:
    - 'all' cannot be combined with specific slugs
    - Slugs are valid and unique (no duplicates)
    - Order strictly follows ordered_indices when 'all' is selected
    Returns (manifest, idx_to_slug, groups, metadata_sha256).
    """
    ordered_indices, idx_to_slug, groups, metadata_sha = load_mt50_metadata(fabri_root)
    slug_to_idx = {slug: int(idx) for idx, slug in idx_to_slug.items()}

    if not tasks_arg:
        raise ValueError("--tasks argument cannot be empty")

    if "all" in tasks_arg:
        if len(tasks_arg) > 1:
            raise ValueError("--tasks 'all' cannot be combined with specific task slugs")
        selected_slugs = [idx_to_slug[str(idx)] for idx in ordered_indices]
    else:
        seen = set()
        selected_slugs = []
        for s in tasks_arg:
            if s in seen:
                raise ValueError(f"Duplicate task slug specified in --tasks: '{s}'")
            if s not in slug_to_idx:
                valid_slugs = sorted(list(slug_to_idx.keys()))
                raise ValueError(
                    f"Unknown task slug '{s}'. Valid MT50 task slugs are: {valid_slugs[:5]}... ({len(valid_slugs)} total)"
                )
            seen.add(s)
            selected_slugs.append(s)

    # Map slug to difficulty group
    slug_to_group: Dict[str, str] = {}
    for grp_name, grp_slugs in groups.items():
        for s in grp_slugs:
            slug_to_group[s] = grp_name

    manifest: List[Dict[str, Any]] = []
    for slug in selected_slugs:
        idx = slug_to_idx[slug]
        prompt = resolve_task_prompt(slug, fabri_root=fabri_root)
        manifest.append(
            {
                "env_task_index": idx,
                "task_slug": slug,
                "prompt": prompt,
                "group": slug_to_group.get(slug, "unknown"),
            }
        )

    return manifest, idx_to_slug, groups, metadata_sha


def make_mt50_vector_env(seed: int = 4048) -> Any:
    """Lazy instantiate official MT50 vector environment via gymnasium."""
    try:
        import metaworld  # noqa: F401
    except ImportError as e:
        raise RuntimeError("metaworld package is required for MT50 evaluation but not installed.") from e

    try:
        import gymnasium as gym
    except ImportError as e:
        raise RuntimeError("gymnasium package is required for MT50 evaluation but not installed.") from e

    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=seed,
        render_mode="rgb_array",
        camera_name="corner2",
    )
    return envs


# ============================================================================
# Single Episode Runner
# ============================================================================


def run_episode(
    env: Any,
    model: Any,
    prompt: str,
    norm_stats: Dict[str, Any],
    *,
    episode_horizon: int = 400,
    exec_horizon: int = 5,
    observation_stride: int = 1,
    seed: int = 4048,
    step_seconds: Optional[float] = None,
    state_dim: int = 24,
    action_dim: int = 24,
    image_size: int = 448,
    policy_kind: str = "predictive",
    seed_policy: str = "once",
    master_seed: Optional[int] = None,
    task_slug: Optional[str] = None,
    episode_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single predictive closed-loop evaluation episode.

    Direct synchronous stepping without AsyncVisualPlanner.
    ``per-plan`` requires an explicit master seed, task slug and zero-based
    episode index; ``seed`` remains exclusively the environment reset seed.
    """
    if episode_horizon <= 0:
        raise ValueError(f"episode_horizon must be positive, got {episode_horizon}")
    if exec_horizon <= 0:
        raise ValueError(f"exec_horizon must be positive, got {exec_horizon}")
    if observation_stride <= 0:
        raise ValueError(f"observation_stride must be positive, got {observation_stride}")
    if policy_kind not in ("predictive", "compact-temporal", "native-single"):
        raise ValueError(f"Unknown policy_kind: {policy_kind}")
    if seed_policy not in ("once", "per-plan"):
        raise ValueError(f"Unknown seed_policy: {seed_policy}")
    if seed_policy == "per-plan":
        if type(master_seed) is not int or not isinstance(task_slug, str) or not task_slug:
            raise ValueError("per-plan seeding requires an integer master_seed and nonempty task_slug")
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("per-plan seeding requires a nonnegative integer episode_index")

    # Lazy import predictive_inference
    try:
        from fabri_moss.predictive_inference import PredictiveEpisodeHistory
    except ImportError as e:
        raise RuntimeError("Could not import PredictiveEpisodeHistory from fabri_moss.predictive_inference") from e

    # Determine physical step dt
    dt: Optional[float] = None
    if step_seconds is not None:
        if isinstance(step_seconds, bool) or not isinstance(step_seconds, (int, float)) or not math.isfinite(step_seconds) or step_seconds <= 0.0:
            raise ValueError(f"--step-seconds must be a positive finite float, got {step_seconds}")
        dt = float(step_seconds)
    else:
        env_dt = getattr(env, "dt", None)
        if env_dt is None:
            unwrapped = getattr(env, "unwrapped", None)
            env_dt = getattr(unwrapped, "dt", None)
        if env_dt is not None and not isinstance(env_dt, bool) and isinstance(env_dt, (int, float)) and math.isfinite(env_dt) and env_dt > 0.0:
            dt = float(env_dt)
        else:
            raise RuntimeError(
                "Could not determine positive finite dt from env.dt or env.unwrapped.dt. "
                "Explicit --step-seconds is strictly required."
            )

    # 1. Iterate goal position if available
    goal_iter = getattr(env, "iterate_goal_position", None)
    if goal_iter is None:
        unwrapped = getattr(env, "unwrapped", None)
        goal_iter = getattr(unwrapped, "iterate_goal_position", None)
    if callable(goal_iter):
        try:
            goal_iter()
        except Exception as e:
            logger.warning("Error during iterate_goal_position: %s", e)
            raise

    raw_obs, reset_info = env.reset(seed=seed)
    if env.action_space.shape != (4,):
        raise ValueError("MetaWorld evaluation requires a four-dimensional action space")
    zero_act = np.clip(np.zeros(4, dtype=np.float32), env.action_space.low, env.action_space.high)
    raw_obs, _, warmup_term, warmup_trunc, _ = env.step(zero_act)

    # Model parameters and device resolution
    device = next(model.parameters()).device
    cuda_devices = []
    if seed_policy == "per-plan":
        if device.type not in ("cpu", "cuda"):
            raise ValueError("per-plan seeding supports CPU and CUDA devices only")
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    action_stats = norm_stats.get("action", norm_stats.get("actions", {}))
    act_space_low = getattr(env.action_space, "low", None)
    act_space_high = getattr(env.action_space, "high", None)

    # Validate action head horizon
    ah_config = getattr(getattr(model, "policy", model), "action_head", None)
    ah_cfg = getattr(ah_config, "config", None)
    head_horizon = getattr(ah_cfg, "horizon", None)
    if type(head_horizon) is not int or head_horizon <= 0:
        raise ValueError("Action head must declare a positive horizon")
    if exec_horizon > head_horizon:
        raise ValueError(
            f"exec_horizon ({exec_horizon}) cannot exceed model action head horizon ({head_horizon})"
        )

    model.eval()
    history = PredictiveEpisodeHistory(prompt=prompt)
    if policy_kind == "native-single":
        if getattr(model, "writer", None) is not None or getattr(model.policy, "writer", None) is not None:
            raise ValueError("Native single-frame evaluation strictly forbids learned writer")
        learned_writer_enabled = False
        memory_kind = "native_single_frame"
    elif policy_kind == "compact-temporal":
        if not hasattr(model, "compact_config") or getattr(model, "writer", None) is not None:
            raise ValueError(
                "Compact evaluation requires compact_config and strictly forbids learned writer"
            )
        learned_writer_enabled = False
        memory_kind = "fixed_compact_temporal"
    else:
        if not isinstance(getattr(model, "writer", None), torch.nn.Module):
            raise ValueError("Predictive evaluation requires the loaded memory writer")
        learned_writer_enabled = True
        memory_kind = "predictive_causal_writer"

    # Attach forward hook on model.writer to count calls without mutating model
    writer_calls_step: List[int] = [0]
    hook_handle = None
    if policy_kind == "predictive":
        writer_mod = getattr(model, "writer", None)
        if writer_mod is not None and isinstance(writer_mod, torch.nn.Module):
            def _hook(m: torch.nn.Module, inp: Any, out: Any) -> None:
                writer_calls_step[0] += 1
            hook_handle = writer_mod.register_forward_hook(_hook)

    # Tracking records
    commands_log: List[Dict[str, Any]] = []
    plan_records: List[Dict[str, Any]] = []
    current_plan_actions: Optional[np.ndarray] = None  # Shape [H, D] normalized
    current_plan_source_step: int = -1
    success: bool = False
    stop_reason: str = "horizon_reached"
    executed_actions_count: int = 0
    t_step: int = 0

    try:
        for t in range(episode_horizon):
            t_step = t
            needs_new_plan = (current_plan_actions is None) or ((t - current_plan_source_step) >= exec_horizon)

            # Observation capture rule:
            # Capture if t % observation_stride == 0 OR needs_new_plan
            # Captured exactly once at step t
            captured_this_step = False
            if (t % observation_stride == 0) or needs_new_plan:
                if isinstance(raw_obs, dict):
                    state_array = (np.asarray(raw_obs["observation"]).ravel() if "observation" in raw_obs
                                   else np.concatenate([np.asarray(v).ravel() for v in raw_obs.values()]))
                else:
                    state_array = np.asarray(raw_obs).ravel()
                obs_t = capture_observation(
                    env=env,
                    step_id=t,
                    norm_stats=norm_stats,
                    real_env_obs=state_array,
                    image_size=image_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    observation_time=float(t * dt),
                )
                history.append(obs_t)
                captured_this_step = True

            # Planning phase
            if needs_new_plan:
                writer_calls_before = writer_calls_step[0]
                t_plan_start = time.monotonic()

                sample = history.sample_for_decision(device=device)

                plan_seed = None
                if seed_policy == "per-plan":
                    identity = json.dumps(
                        [master_seed, task_slug, episode_index, t], separators=(",", ":"), ensure_ascii=True
                    ).encode("utf-8")
                    plan_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2 ** 63)

                # Restore CPU/selected CUDA RNG after each paired prediction.
                # The default path neither forks nor reseeds the caller's RNG.
                with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices, enabled=plan_seed is not None):
                    if plan_seed is not None:
                        _seed_torch_device(plan_seed, device)
                    pred_out = model.predict_actions(sample)

                writer_calls_during = writer_calls_step[0] - writer_calls_before

                # Validate prediction output
                if not isinstance(pred_out, torch.Tensor):
                    raise TypeError(f"predict_actions must return torch.Tensor, got {type(pred_out)}")
                if tuple(pred_out.shape) != (1, head_horizon, action_dim):
                    raise ValueError(f"predict_actions output must have shape {(1, head_horizon, action_dim)}, got {tuple(pred_out.shape)}")
                if not torch.isfinite(pred_out).all():
                    raise ValueError("predict_actions returned non-finite action values (NaN/Inf)")

                H = pred_out.shape[1]
                if exec_horizon > H:
                    raise ValueError(f"exec_horizon ({exec_horizon}) exceeds predicted action horizon H={H}")

                history.commit_decision()

                current_plan_actions = pred_out[0].detach().to(device="cpu", dtype=torch.float32).numpy()
                plan_latency = time.monotonic() - t_plan_start
                current_plan_source_step = t

                plan_record = {
                    "source_step": t,
                    "source_time": float(t * dt),
                    "plan_latency_sec": plan_latency,
                    "writer_calls": writer_calls_during,
                    "memory_kind": memory_kind,
                    "learned_writer_enabled": learned_writer_enabled,
                    "history_frames_count": len(history.frame_ids),
                    "model_input_frames_count": 1 if policy_kind == "native-single" else len(history.frame_ids),
                    "history_frame_ids": list(history.frame_ids),
                    "decision_indices": list(history.decision_indices),
                    "decision_frame_ids": [history.frame_ids[i] for i in history.decision_indices],
                    "history_observation_times": list(history.observation_times),
                    "captured_this_step": captured_this_step,
                }
                if plan_seed is not None:
                    plan_record["torch_seed"] = plan_seed
                plan_records.append(plan_record)

            # Action extraction and execution
            offset = t - current_plan_source_step
            if offset < 0 or offset >= current_plan_actions.shape[0]:
                raise RuntimeError(
                    f"Action offset {offset} out of bounds for plan anchored at {current_plan_source_step} "
                    f"with horizon {current_plan_actions.shape[0]} at step {t}"
                )

            norm_action = current_plan_actions[offset]
            exec_action = denormalize_action(
                action_norm=norm_action,
                action_stats=action_stats,
                action_space_low=act_space_low,
                action_space_high=act_space_high,
            )

            cmd_entry = {
                "step": t,
                "source_step": current_plan_source_step,
                "offset": offset,
                "action": [float(x) for x in exec_action],
            }
            commands_log.append(cmd_entry)

            # Step environment
            raw_obs, reward, terminated, truncated, info = env.step(exec_action)
            executed_actions_count += 1
            info_success = isinstance(info, dict) and info.get("success", 0) == 1

            if info_success:
                success = True
                stop_reason = "env_success"
                break

            if terminated or truncated:
                stop_reason = "env_terminal"
                break

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return {
        "success": bool(success),
        "steps": t_step + 1,
        "stop_reason": stop_reason,
        "policy_kind": policy_kind,
        "memory_kind": memory_kind,
        "learned_writer_enabled": learned_writer_enabled,
        "executed_actions_count": executed_actions_count,
        "plans_completed": len(plan_records),
        "total_writer_calls": writer_calls_step[0],
        "history_final_frames": list(history.frame_ids),
        "history_final_observation_times": list(history.observation_times),
        "history_final_decisions": list(history.decision_indices),
        "decision_frame_ids": [history.frame_ids[i] for i in history.decision_indices],
        "step_seconds": dt,
        "warmup_terminal": bool(warmup_term or warmup_trunc),
        "plans": plan_records,
        "command_log": commands_log,
    }


# ============================================================================
# Statistics & Summary Reporter
# ============================================================================


def compute_evaluation_summary(
    manifest: List[Dict[str, Any]],
    episode_results: List[Dict[str, Any]],
    episodes_per_task: int,
    groups: Dict[str, List[str]],
    status: str = "running",
    is_full_mt50: bool = False,
) -> Dict[str, Any]:
    """Compute hierarchical per-task, difficulty-group, and overall summary statistics."""
    planned_episodes = len(manifest) * episodes_per_task
    completed_episodes = len(episode_results)
    successful_episodes = sum(1 for r in episode_results if r.get("success", False))
    allowed_tasks = {item["task_slug"] for item in manifest}
    seen = set()
    for result in episode_results:
        key = (result.get("task_slug"), result.get("episode_index"))
        if key[0] not in allowed_tasks or type(key[1]) is not int or not 0 <= key[1] < episodes_per_task or key in seen:
            raise ValueError("Evaluation results contain a duplicate or invalid task/episode")
        seen.add(key)
    complete = status == "completed" and completed_episodes == planned_episodes

    overall_success_rate = (successful_episodes / completed_episodes) if completed_episodes > 0 else 0.0
    completion_fraction = (completed_episodes / planned_episodes) if planned_episodes > 0 else 0.0

    # Group results by task_slug
    results_by_task: Dict[str, List[Dict[str, Any]]] = {item["task_slug"]: [] for item in manifest}
    for r in episode_results:
        slug = r.get("task_slug")
        if slug in results_by_task:
            results_by_task[slug].append(r)

    per_task: Dict[str, Dict[str, Any]] = {}
    task_success_rates: List[float] = []

    for item in manifest:
        slug = item["task_slug"]
        runs = results_by_task.get(slug, [])
        t_completed = len(runs)
        t_success = sum(1 for r in runs if r.get("success", False))
        t_rate = (t_success / t_completed) if t_completed > 0 else 0.0
        per_task[slug] = {
            "task_index": item["env_task_index"],
            "group": item["group"],
            "planned": episodes_per_task,
            "completed": t_completed,
            "successes": t_success,
            "success_rate": t_rate,
        }
        if t_completed > 0:
            task_success_rates.append(t_rate)

    task_macro_rate = float(np.mean(task_success_rates)) if task_success_rates else 0.0

    # Difficulty groups statistics
    difficulty_groups: Dict[str, Dict[str, Any]] = {}
    group_rates: List[float] = []

    for grp_name, grp_slugs in groups.items():
        grp_manifest_slugs = [s for s in grp_slugs if s in results_by_task]
        if not grp_manifest_slugs:
            continue
        g_runs = [r for s in grp_manifest_slugs for r in results_by_task.get(s, [])]
        g_completed = len(g_runs)
        g_success = sum(1 for r in g_runs if r.get("success", False))
        g_rate = (g_success / g_completed) if g_completed > 0 else 0.0
        difficulty_groups[grp_name] = {
            "tasks_count": len(grp_manifest_slugs),
            "planned": len(grp_manifest_slugs) * episodes_per_task,
            "completed": g_completed,
            "successes": g_success,
            "success_rate": g_rate,
        }
        if g_completed > 0:
            group_rates.append(g_rate)

    group_macro_rate = float(np.mean(group_rates)) if group_rates else 0.0

    return {
        "status": status,
        "complete": complete,
        "configured_full_mt50": bool(is_full_mt50),
        "is_full_mt50_benchmark": bool(is_full_mt50 and complete),
        "counts": {
            "planned_episodes": planned_episodes,
            "completed_episodes": completed_episodes,
            "successful_episodes": successful_episodes,
            "completion_fraction": completion_fraction,
        },
        "overall": {
            "success_rate": overall_success_rate,
            "task_macro_success_rate": task_macro_rate,
            "group_macro_success_rate": group_macro_rate,
        },
        "difficulty_groups": difficulty_groups,
        "per_task": per_task,
    }


# ============================================================================
# CLI Parser & Main Evaluation Orchestrator
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for predictive MT50 evaluation."""
    parser = argparse.ArgumentParser(
        description="FabriVLA / MOSS Predictive Memory Closed-Loop Evaluation on MetaWorld MT50"
    )
    parser.add_argument(
        "--policy-kind",
        type=str,
        choices=["predictive", "compact-temporal", "native-single"],
        default="predictive",
        help="Policy architecture; native-single calls the original FabriVLA API on the latest frame",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to new output directory")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to FabriVLA repository")
    parser.add_argument("--vlm-path", type=str, default="/root/models/InternVL3_5-1B", help="Path to VLM directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per task")
    parser.add_argument("--episode-horizon", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument("--exec-horizon", type=int, default=5, help="Execution horizon before replan")
    parser.add_argument("--num-inference-timesteps", type=int, default=50, help="Action head diffusion timesteps")
    parser.add_argument("--seed", type=int, default=4048, help="Master evaluation seed")
    parser.add_argument(
        "--seed-policy", choices=["once", "per-plan"], default="once",
        help="once: original RNG stream; per-plan: paired Torch noise by master seed/task/episode/source step",
    )
    parser.add_argument(
        "--seed-before-load", action="store_true",
        help="match official FabriVLA: seed Torch before constructing/loading the model",
    )
    parser.add_argument("--observation-stride", type=int, default=1, help="Observation stride")
    parser.add_argument("--step-seconds", type=float, default=None, help="Explicit physical step dt override")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Tasks to evaluate: 'all' or list of unique valid MetaWorld task slugs",
    )
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads")
    parser.add_argument("--expected-sha256", type=str, default=None, help="Optional expected checkpoint SHA256")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key in norm_stats")
    return parser


def parse_args(raw_args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(raw_args)

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    if args.episode_horizon <= 0:
        raise ValueError(f"--episode-horizon must be positive, got {args.episode_horizon}")
    if args.exec_horizon <= 0:
        raise ValueError(f"--exec-horizon must be positive, got {args.exec_horizon}")
    if args.num_inference_timesteps <= 0:
        raise ValueError(f"--num-inference-timesteps must be positive, got {args.num_inference_timesteps}")
    if args.observation_stride <= 0:
        raise ValueError(f"--observation-stride must be positive, got {args.observation_stride}")
    if args.threads <= 0:
        raise ValueError(f"--threads must be positive, got {args.threads}")
    if args.step_seconds is not None:
        if isinstance(args.step_seconds, bool) or not isinstance(args.step_seconds, (int, float)) or not math.isfinite(args.step_seconds) or args.step_seconds <= 0.0:
            raise ValueError(f"--step-seconds must be a positive finite float, got {args.step_seconds}")

    return args


def run_evaluation(
    args: argparse.Namespace,
    device: Optional[Union[str, torch.device]] = None,
    env_factory: Optional[Callable[[int], Any]] = None,
) -> Dict[str, Any]:
    """Execute complete predictive closed-loop evaluation workflow.

    Args:
        args: Parsed CLI namespace
        device: Device override (e.g. 'cpu' for unit tests)
        env_factory: Optional factory for environment (e.g. for testing)
    """
    repo_root = Path(__file__).resolve().parent.parent
    seed_policy = getattr(args, "seed_policy", "once")
    if seed_policy not in ("once", "per-plan"):
        raise ValueError(f"Unknown seed_policy: {seed_policy}")

    # 1. Output directory validation: reject non-empty existing directory
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory '{output_dir}' exists and is not empty. Refusing to overwrite.")

    # 2. Resolve target device
    # Formal evaluation requires CUDA; CPU device is only permitted via explicit parameter injection
    if device is None and "cuda" not in str(args.device).lower():
        raise ValueError(
            f"CLI formal evaluation requires a CUDA device (e.g. --device cuda:0). "
            f"Got device '{args.device}'. For CPU tests, invoke run_evaluation via python API with device='cpu'."
        )
    target_device = device if device is not None else args.device
    device_str = str(target_device)
    if "cuda" in device_str.lower():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ('{device_str}') but torch.cuda is not available.")

    # Official FabriVLA seeds Torch before model construction.
    seed_before_load = bool(getattr(args, "seed_before_load", False))
    if seed_before_load:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available() and "cuda" in device_str.lower():
            torch.cuda.manual_seed_all(args.seed)

    # 3. Build selected manifest before loading checkpoints / GPU resources
    manifest, idx_to_slug, groups, metadata_sha256 = resolve_task_manifest(
        tasks_arg=args.tasks,
        fabri_root=args.fabri_root,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest.json
    manifest_payload = {
        "metadata_sha256": metadata_sha256,
        "total_tasks": len(manifest),
        "tasks": manifest,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    # Check full mt50 benchmark flag
    unique_slugs = {item["task_slug"] for item in manifest}
    is_full_mt50 = (
        "cuda" in device_str.lower()
        and len(manifest) == 50
        and len(unique_slugs) == 50
        and args.episodes == 10
        and args.episode_horizon == 400
        and args.exec_horizon == 5
        and args.num_inference_timesteps == 50
        and args.observation_stride == 1
    )

    source_hashes = compute_source_fingerprints(repo_root)

    tasks_path = Path(args.fabri_root) / "evaluations" / "metaworld" / "tasks.jsonl"
    tasks_jsonl_sha256 = compute_sha256(tasks_path) if tasks_path.exists() and tasks_path.is_file() else None

    kind = getattr(args, "policy_kind", "predictive")
    protocol_names = {
        "predictive": ("predictive_sync_full_prefix_v1", "predictive_causal_writer"),
        "compact-temporal": ("compact_sync_full_prefix_v1", "fixed_compact_temporal"),
        "native-single": ("native_single_sync_v1", "native_single_frame"),
    }
    if kind not in protocol_names:
        raise ValueError(f"Unknown policy_kind: {kind}")
    proto_name, memory_kind = protocol_names[kind]

    provenance: Dict[str, Any] = {
        "protocol_name": proto_name,
        "policy_kind": kind,
        "learned_writer_enabled": (kind == "predictive"),
        "memory_kind": memory_kind,
        "checkpoint_source": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir),
        "fabri_root": str(Path(args.fabri_root).resolve()),
        "vlm_path": str(Path(args.vlm_path).resolve()),
        "device": device_str,
        "episodes_per_task": args.episodes,
        "episode_horizon": args.episode_horizon,
        "exec_horizon": args.exec_horizon,
        "num_inference_timesteps": args.num_inference_timesteps,
        "seed": args.seed,
        "seed_policy": (
            "official_preload_once"
            if seed_before_load and seed_policy == "once"
            else ("explicit_after_load_once" if seed_policy == "once" else "per_plan_torch_sha256_v1")
        ),
        "latency_mode": "single_frame_with_history_logging" if kind == "native-single" else "full_prefix_retention_not_realtime",
        "observation_stride": args.observation_stride,
        "step_seconds": args.step_seconds,
        "tasks_count": len(manifest),
        "configured_full_mt50": is_full_mt50,
        "metadata_sha256": metadata_sha256,
        "order_sha256": metadata_sha256,
        "tasks_jsonl_sha256": tasks_jsonl_sha256,
        "source_hashes": source_hashes,
        "started_at": time.time(),
        "checkpoint_metadata": None,
    }
    if seed_policy == "per-plan":
        provenance["plan_seed_derivation"] = (
            "int.from_bytes(SHA256(JSON([master_seed, task_slug, zero_based_episode_index, source_step], "
            "separators=(',', ':'), ensure_ascii=True).encode('utf-8'))[:8], 'big') % 2**63"
        )
        provenance["plan_rng_scope"] = (
            "Seed CPU and selected CUDA device immediately before predict_actions, restore both afterwards. "
            "Python/NumPy seeded once after model load; environment reset seed remains master_seed + episode_index."
        )

    episode_results: List[Dict[str, Any]] = []
    envs: Any = None
    active_task: Optional[str] = None
    active_episode: Optional[str] = None
    jsonl_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary_report.json"

    def write_summary(status: str, error: Optional[str] = None) -> Dict[str, Any]:
        summary = compute_evaluation_summary(
            manifest=manifest,
            episode_results=episode_results,
            episodes_per_task=args.episodes,
            groups=groups,
            status=status,
            is_full_mt50=is_full_mt50,
        )
        summary["provenance"] = copy.deepcopy(provenance)
        if error is not None:
            summary["error"] = str(error)
            if active_task is not None:
                summary["active_task"] = active_task
            if active_episode is not None:
                summary["active_episode"] = active_episode
        tmp_path = summary_path.with_name(f"{summary_path.name}.tmp.{time.time_ns()}")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        tmp_path.replace(summary_path)
        return summary

    # Write initial summary report before loading model or environments
    write_summary(status="running")

    # Set threads before loading model
    torch.set_num_threads(args.threads)

    try:
        if kind == "native-single":
            from fabri_moss.native_single_inference import load_native_single_inference as load_inference
        elif kind == "compact-temporal":
            try:
                from fabri_moss.compact_inference import load_compact_inference as load_inference
            except ImportError as e:
                raise RuntimeError("Could not import load_compact_inference from fabri_moss.compact_inference") from e
        else:
            try:
                from fabri_moss.predictive_inference import load_predictive_inference as load_inference
            except ImportError as e:
                raise RuntimeError("Could not import load_predictive_inference from fabri_moss.predictive_inference") from e

        model, norm_stats, ckpt_metadata = load_inference(
            checkpoint_path=args.checkpoint,
            snapshot_dir=snapshot_dir,
            fabri_root=args.fabri_root,
            vlm_path=args.vlm_path,
            device=target_device,
            arm_key=args.arm_key,
            expected_sha256=args.expected_sha256,
        )
        provenance["checkpoint_metadata"] = ckpt_metadata
        model.eval()

        # Validate action head configuration
        ah_config = getattr(getattr(model, "policy", model), "action_head", None)
        ah_cfg = getattr(ah_config, "config", None)
        if ah_cfg is None:
            raise ValueError("Model action head must declare a config")
        head_horizon = getattr(ah_cfg, "horizon", None)
        head_state_dim = getattr(ah_cfg, "state_dim", None)
        head_per_action_dim = getattr(ah_cfg, "per_action_dim", None)

        if type(head_horizon) is not int or head_horizon <= 0:
            raise ValueError(f"Action head must declare a positive horizon, got {head_horizon}")
        if type(head_state_dim) is not int or head_state_dim <= 0:
            raise ValueError(f"Action head must declare a positive state_dim, got {head_state_dim}")
        if type(head_per_action_dim) is not int or head_per_action_dim <= 0:
            raise ValueError(f"Action head must declare a positive per_action_dim, got {head_per_action_dim}")

        if args.exec_horizon > head_horizon:
            raise ValueError(
                f"exec_horizon ({args.exec_horizon}) cannot exceed model action head horizon ({head_horizon})"
            )

        # Configure diffusion timesteps on action head if present
        if hasattr(ah_cfg, "num_inference_timesteps"):
            ah_cfg.num_inference_timesteps = args.num_inference_timesteps
        elif hasattr(model, "config") and hasattr(model.config, "num_inference_timesteps"):
            model.config.num_inference_timesteps = args.num_inference_timesteps

        image_size = 448
        if isinstance(ckpt_metadata, dict) and "image_size" in ckpt_metadata:
            image_size = int(ckpt_metadata["image_size"])

        # Preserve the official preload RNG stream when requested.
        random.seed(args.seed)
        np.random.seed(args.seed)
        if not seed_before_load:
            if seed_policy == "per-plan":
                _seed_torch_device(args.seed, torch.device(target_device))
            else:
                torch.manual_seed(args.seed)
                if torch.cuda.is_available() and "cuda" in device_str.lower():
                    torch.cuda.manual_seed_all(args.seed)

        # Instantiate environment
        if env_factory is not None:
            envs = env_factory(args.seed)
        else:
            envs = make_mt50_vector_env(seed=args.seed)

        # Validate environment task coverage
        sub_envs = getattr(envs, "envs", None)
        if sub_envs is None and hasattr(envs, "unwrapped"):
            sub_envs = getattr(envs.unwrapped, "envs", None)

        if sub_envs is None:
            raise RuntimeError("Environment must provide .envs containing sub-environments for MT50 tasks.")

        max_idx = max(item["env_task_index"] for item in manifest)
        if len(sub_envs) <= max_idx:
            raise RuntimeError(
                f"MT50 environment contains {len(sub_envs)} sub-environments, "
                f"which cannot cover required task index {max_idx}"
            )

        for task_info in manifest:
            task_idx = task_info["env_task_index"]
            task_slug = task_info["task_slug"]
            task_prompt = task_info["prompt"]
            task_group = task_info["group"]
            active_task = task_slug
            sub_env = sub_envs[task_idx]

            for ep_idx in range(args.episodes):
                ep_id = f"{task_slug}_ep_{ep_idx:03d}"
                active_episode = ep_id
                ep_seed = args.seed + ep_idx
                plan_seed_kwargs = {}
                if seed_policy == "per-plan":
                    plan_seed_kwargs = dict(
                        seed_policy=seed_policy, master_seed=args.seed,
                        task_slug=task_slug, episode_index=ep_idx,
                    )

                ep_data = run_episode(
                    env=sub_env,
                    model=model,
                    prompt=task_prompt,
                    norm_stats=norm_stats,
                    episode_horizon=args.episode_horizon,
                    exec_horizon=args.exec_horizon,
                    observation_stride=args.observation_stride,
                    seed=ep_seed,
                    step_seconds=args.step_seconds,
                    state_dim=head_state_dim,
                    action_dim=head_per_action_dim,
                    image_size=image_size,
                    policy_kind=kind,
                    **plan_seed_kwargs,
                )

                ep_data["episode_id"] = ep_id
                ep_data["task_slug"] = task_slug
                ep_data["task_index"] = task_idx
                ep_data["group"] = task_group
                ep_data["episode_index"] = ep_idx
                ep_data["seed"] = ep_seed

                episode_results.append(ep_data)

                # Incremental flush to JSONL
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ep_data) + "\n")
                    f.flush()

                # Incremental flush to summary_report.json
                write_summary(status="running")
                print(f"[{len(episode_results)}/{len(manifest) * args.episodes}] "
                      f"{task_slug} episode={ep_idx + 1} success={int(ep_data['success'])} "
                      f"steps={ep_data['steps']} writer_calls={ep_data['total_writer_calls']}", flush=True)

    except Exception as exc:
        logger.error("Predictive evaluation aborted due to exception: %s", exc, exc_info=True)
        try:
            write_summary(status="failed", error=str(exc))
        except Exception as write_err:
            logger.warning("Failed to write failure summary: %s", write_err)
        raise
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as close_err:
                logger.warning("Error closing MT50 environment: %s", close_err)

    provenance["finished_at"] = time.time()
    final_summary = write_summary(status="completed")
    return final_summary


def main() -> None:
    """CLI Entrypoint. Strict CUDA-only for formal runs."""
    args = parse_args()
    device_str = args.device.lower()
    if "cuda" not in device_str:
        raise ValueError(
            f"CLI formal evaluation requires a CUDA device (e.g. --device cuda:0). "
            f"Got device '{args.device}'. For CPU tests, invoke run_evaluation via python API."
        )
    run_evaluation(args)


if __name__ == "__main__":
    main()
