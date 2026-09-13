"""Official synchronous MOSS cross-attention evaluation on MetaWorld MT50.

The legacy :mod:`fabri_moss.evaluate_async` command remains useful for a
single-task timing/queue check.  This module is the explicit closed-loop
entrypoint for MOSS: it uses the official vector environment, fixed image
preprocessing, 400-step episodes, five-step execution chunks, and 50 flow
steps.  Its report is kept separate from asynchronous success numbers and the
native FabriVLA SOTA baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fabri_moss.core import MOSS_ARCHITECTURE_REVISION, MossConfig, MossInternVL, VisionSession
try:
    # New core exposes an unbounded episode-scoped consume session.  Keep the
    # fallback import for older checkouts used by CPU mechanism tests.
    from fabri_moss.core import FrameKVSession
except ImportError:  # pragma: no cover - compatibility with pre-session core
    FrameKVSession = VisionSession  # type: ignore[misc,assignment]
from fabri_moss.evaluate_async import (
    OFFICIAL_CAMERA_NAME,
    OFFICIAL_EPISODE_HORIZON,
    OFFICIAL_EXEC_HORIZON,
    OFFICIAL_FLOW_STEPS,
    OFFICIAL_IMAGE_SIZE,
    OFFICIAL_MT50_EPISODES_PER_TASK,
    OFFICIAL_MT50_TASK_COUNT,
    MOSS_TRAINER_ARCHITECTURE_REVISION,
    build_moss_provenance,
    capture_observation,
    compute_file_sha256,
    describe_fa2_runtime,
    derive_plan_seed,
    fixed_diffusion_seed,
    make_official_mt50_env,
    moss_architecture_contract,
    official_mt50_contract,
    denormalize_action,
)
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint


def _cross_lora_paths(model: MossInternVL) -> List[str]:
    """Resolve the trainer's native q/k/v/o targets without hard-coded wrappers."""
    paths: List[str] = []
    for layer_idx in model.config.cross_layers:
        attention = model.native_core.layers[int(layer_idx) - 1].self_attn
        prefix = next((name for name, module in model.policy.named_modules() if module is attention), None)
        if not prefix:
            raise ValueError(f"native attention layer {layer_idx} is not reachable from policy")
        paths.extend(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    return paths


def _restore_joint_lora(model: MossInternVL, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Inject and restore the strict FP32 LoRA surface used by joint MOSS."""
    from fabri_moss.lora import inject_lora, load_lora_state_dict, lora_spec

    spec = checkpoint.get("lora_spec")
    state = checkpoint.get("lora_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("joint MOSS adapter must contain lora_spec and lora_state")
    rank = int(spec.get("rank", 0))
    alpha = float(spec.get("alpha", 0.0))
    dropout = float(spec.get("dropout", 0.0))
    modules = inject_lora(
        model.policy,
        _cross_lora_paths(model),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    actual = lora_spec(model.policy)
    if actual != spec:
        raise ValueError(f"LoRA architecture contract mismatch: checkpoint={spec!r}, model={actual!r}")
    load_lora_state_dict(model.policy, state, strict=True)
    return {"lora_spec": actual, "lora_modules": sorted(modules)}


logger = logging.getLogger("evaluate_moss")


def _json_contract(value: Any) -> Any:
    """Keep provenance JSON serializable without touching checkpoint tensors."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)

def _load_adapter_model(
    *,
    checkpoint: Union[str, Path],
    adapter: Union[str, Path],
    fabri_root: Union[str, Path],
    vlm_path: Union[str, Path],
    device: Union[str, torch.device],
    stage: str,
    memory_mode: str,
    window: Optional[int],
    arm_key: str,
) -> Tuple[MossInternVL, Dict[str, Any], Dict[str, Any]]:
    """Load a strict ``moss_cross_adapter_v2`` model and provenance."""
    if memory_mode != "consume":
        raise ValueError(
            "Official synchronous MOSS evaluation uses the consume VisionSession contract; "
            "delta requires its explicit stateful async protocol."
        )
    adapter_path = Path(adapter).resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(f"MOSS adapter checkpoint not found: {adapter_path}")

    policy, _ckpt_cfg, norm_stats, base_meta = load_native_checkpoint(
        fabri_root=fabri_root,
        checkpoint_path=checkpoint,
        vlm_path=vlm_path,
        device=str(device),
        arm_key=arm_key,
        trainable=False,
    )
    native_fa2: Dict[str, Any] = {}
    if torch.device(device).type == "cuda":
        native_fa2 = assert_native_fa2(policy)

    adapter_sha = compute_file_sha256(adapter_path)
    ckpt = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format") != "moss_cross_adapter_v2":
        raise ValueError("Official MOSS evaluation requires format='moss_cross_adapter_v2'")
    architecture_revision = ckpt.get("architecture_revision")
    if architecture_revision != MOSS_TRAINER_ARCHITECTURE_REVISION:
        raise ValueError(
            f"MOSS adapter architecture_revision {architecture_revision!r} != "
            f"{MOSS_TRAINER_ARCHITECTURE_REVISION!r}"
        )
    source_sha = ckpt.get("source_checkpoint_sha256", ckpt.get("base_sha256"))
    if source_sha != base_meta.get("checkpoint_sha256"):
        raise ValueError("MOSS adapter/source checkpoint SHA256 mismatch")
    if ckpt.get("stage") != stage:
        raise ValueError(f"MOSS adapter stage {ckpt.get('stage')!r} != requested {stage!r}")
    config_payload = ckpt.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("MOSS adapter is missing its config contract")
    config = MossConfig(**dict(config_payload))
    if getattr(config, "architecture_revision", None) not in (
        MOSS_ARCHITECTURE_REVISION,
        MOSS_TRAINER_ARCHITECTURE_REVISION,
    ):
        raise ValueError("MOSS config architecture_revision is unsupported")
    if config.memory_mode != memory_mode:
        raise ValueError(
            f"MOSS adapter memory_mode {config.memory_mode!r} != requested {memory_mode!r}"
        )
    if window is not None and config.max_frames != window:
        raise ValueError(
            f"MOSS adapter max_frames {config.max_frames!r} != requested window {window!r}"
        )

    model = MossInternVL(policy, config=config)
    # Joint checkpoints carry LoRA tensors rather than a copied full native
    # policy.  Inject the exact wrapper surface before restoring state.
    lora_meta: Dict[str, Any] = {}
    if stage == "joint":
        lora_meta = _restore_joint_lora(model, ckpt)
    model.set_training_stage(stage)
    try:
        model.cross_blocks.load_state_dict(ckpt["cross_blocks"], strict=True)
        with torch.no_grad():
            model.readout_embeddings.copy_(ckpt["readout_embeddings"].to(model.readout_embeddings.device))
        if stage in ("expert", "joint"):
            model.policy.action_head.load_state_dict(ckpt["action_head"], strict=True)
        # ``moss_cross_adapter_v2`` joint checkpoints intentionally omit a
        # duplicate base_policy; frozen native weights come from source SHA
        # and LoRA state above.
    except KeyError as exc:
        raise ValueError(f"MOSS adapter missing required state: {exc.args[0]}") from exc
    model.eval()

    architecture_contract = moss_architecture_contract(
        memory_mode=memory_mode,
        cross_layers=config.cross_layers,
        max_frames=config.max_frames,
        architecture_revision=architecture_revision,
    )
    architecture_contract.update(
        {
            "name": "fabri_vla_moss_cross_v2",
            "adapter_format": "moss_cross_adapter_v2",
            "native_visual_and_language_attention": "flash_attention_2 (asserted on GPU)",
            "shallow_layer": config.shallow_layer,
            "core_architecture_revision": getattr(config, "architecture_revision", None),
            "lora": lora_meta,
        }
    )
    data_contract = {
        "name": "official_metaworld_mt50_moss_sync_v1",
        "camera_name": OFFICIAL_CAMERA_NAME,
        "image_size": OFFICIAL_IMAGE_SIZE,
        "image_preprocess": "rotate_180_center_crop_2_3_resize_448",
        "state_action_contract": "FabriVLA norm_stats; four effective action dimensions",
        "state_dim": 24,
        "action_dim": 24,
        "time_source": "env.dt_or_explicit_step_seconds",
        "episode_horizon": OFFICIAL_EPISODE_HORIZON,
        "exec_horizon": OFFICIAL_EXEC_HORIZON,
        "flow_steps": OFFICIAL_FLOW_STEPS,
        "seed_contract": "environment master_seed+episode_index; per-plan diffusion SHA256 seed",
        "adapter_training_contract": _json_contract(ckpt.get("data_contract")),
    }
    provenance = build_moss_provenance(
        checkpoint=checkpoint,
        adapter=adapter_path,
        architecture_contract=architecture_contract,
        data_contract=data_contract,
        native_fa2=native_fa2,
        source_files={
            "evaluate_async.py": compute_file_sha256(Path(__file__).with_name("evaluate_async.py")),
            "evaluate_moss.py": compute_file_sha256(Path(__file__)),
        },
    )
    # Keep loader-derived values explicit even when a test double supplies a
    # metadata dictionary rather than a real file.
    provenance["adapter_checkpoint_sha256"] = adapter_sha
    provenance["adapter_sha256"] = adapter_sha
    provenance["adapter_sha"] = adapter_sha
    provenance["base_checkpoint_sha256"] = base_meta.get("checkpoint_sha256")
    provenance["source_sha256"] = base_meta.get("checkpoint_sha256")
    provenance["adapter_stage"] = stage
    provenance["adapter_step"] = ckpt.get("step")
    provenance["fa2_package"] = describe_fa2_runtime()
    provenance["fa2_package"]["native_fa2_verified"] = bool(native_fa2.get("native_fa2_enabled", False))
    return model, norm_stats, provenance


def _env_dt(env: Any, step_seconds: Optional[float]) -> float:
    if step_seconds is not None:
        if isinstance(step_seconds, bool) or not isinstance(step_seconds, (int, float)) or not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError(f"step_seconds must be a positive finite float, got {step_seconds}")
        return float(step_seconds)
    value = getattr(env, "dt", None)
    if value is None:
        value = getattr(getattr(env, "unwrapped", None), "dt", None)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(
            "Could not determine positive finite dt from env.dt or env.unwrapped.dt; "
            "pass --step-seconds explicitly."
        )
    return float(value)


def _env_time_source(env: Any, step_seconds: Optional[float]) -> str:
    if step_seconds is not None:
        return "explicit_step_seconds"
    if getattr(env, "dt", None) is not None:
        return "env.dt"
    return "env.unwrapped.dt"


def _advance_goal(env: Any) -> None:
    for obj in (env, getattr(env, "unwrapped", None)):
        fn = getattr(obj, "iterate_goal_position", None)
        if callable(fn):
            fn()
            return


def _as_state_array(raw_obs: Any) -> np.ndarray:
    if isinstance(raw_obs, dict):
        if "observation" in raw_obs:
            return np.asarray(raw_obs["observation"], dtype=np.float32).ravel()
        return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in raw_obs.values()])
    return np.asarray(raw_obs, dtype=np.float32).ravel()


def run_episode(
    env: Any,
    model: MossInternVL,
    prompt: str,
    norm_stats: Dict[str, Any],
    episode_id: str = "episode",
    task_slug: str = "task",
    episode_index: int = 0,
    master_seed: int = 4048,
    episode_horizon: int = OFFICIAL_EPISODE_HORIZON,
    exec_horizon: int = OFFICIAL_EXEC_HORIZON,
    observation_stride: int = 1,
    seed: Optional[int] = None,
    seed_policy: str = "per-plan",
    step_seconds: Optional[float] = None,
    image_size: int = OFFICIAL_IMAGE_SIZE,
    state_dim: int = 24,
    action_dim: int = 24,
    flow_steps: int = OFFICIAL_FLOW_STEPS,
) -> Dict[str, Any]:
    """Run one synchronous consume-session episode with an auditable ledger."""
    if episode_horizon <= 0 or exec_horizon <= 0 or observation_stride <= 0 or flow_steps <= 0:
        raise ValueError("episode_horizon, exec_horizon, observation_stride, and flow_steps must be positive")
    if seed_policy not in ("per-plan", "once"):
        raise ValueError(f"Unknown seed_policy {seed_policy!r}")
    if seed_policy == "per-plan" and (not isinstance(task_slug, str) or type(episode_index) is not int):
        raise ValueError("per-plan seeding requires task_slug and episode_index")
    if getattr(model.config, "memory_mode", "consume") != "consume":
        raise ValueError("run_episode requires MossConfig(memory_mode='consume')")

    dt = _env_dt(env, step_seconds)
    time_source = _env_time_source(env, step_seconds)
    _advance_goal(env)
    env_seed = master_seed + episode_index if seed is None else int(seed)
    raw_obs, _ = env.reset(seed=env_seed)
    if tuple(getattr(env.action_space, "shape", ())) != (4,):
        raise ValueError("MetaWorld evaluation requires a four-dimensional action space")
    zero = np.zeros(4, dtype=np.float32)
    raw_obs, _, warmup_term, warmup_trunc, _ = env.step(
        np.clip(zero, env.action_space.low, env.action_space.high)
    )

    device = next(model.parameters()).device
    if getattr(model, "training", False) and hasattr(model, "eval"):
        model.eval()
    head = getattr(model.policy, "action_head", None)
    head_cfg = getattr(head, "config", None)
    head_horizon = getattr(head_cfg, "horizon", None)
    if type(head_horizon) is not int or head_horizon <= 0:
        raise ValueError("Action head must declare a positive horizon")
    if exec_horizon > head_horizon:
        raise ValueError(f"exec_horizon ({exec_horizon}) exceeds model horizon ({head_horizon})")
    if hasattr(head_cfg, "num_inference_timesteps"):
        head_cfg.num_inference_timesteps = flow_steps

    session = FrameKVSession(model)
    session.reset(episode_id=episode_id, prompt=prompt)
    current_plan: Optional[np.ndarray] = None
    source_step = -1
    latest_obs = None
    plans: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    success = False
    stop_reason = "horizon_reached"
    fallback_count = 0
    drop_count = 0
    steps = 0

    try:
        for step in range(episode_horizon):
            needs_plan = current_plan is None or (step - source_step) >= exec_horizon
            captured = False
            if (step % observation_stride == 0) or needs_plan:
                latest_obs = capture_observation(
                    env=env,
                    step_id=step,
                    norm_stats=norm_stats,
                    real_env_obs=_as_state_array(raw_obs),
                    image_size=image_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    observation_time=float(step * dt),
                )
                with torch.no_grad():
                    session.append(list(latest_obs.images), step, latest_obs.observation_time)
                captured = True
                observations.append(
                    {
                        "frame_id": step,
                        "observation_time": float(step * dt),
                        "capture_time": latest_obs.capture_time,
                    }
                )

            plan_seed: Optional[int] = None
            if needs_plan:
                if latest_obs is None:
                    raise RuntimeError("A planning step requires a captured observation")
                source_step = step
                if seed_policy == "per-plan":
                    plan_seed = derive_plan_seed(master_seed, task_slug, episode_index, source_step)
                started = time.monotonic()
                with torch.no_grad(), fixed_diffusion_seed(plan_seed, device):
                    deep, shallow = session.query()
                    prediction = model.policy.action_head.sample(
                        deep,
                        state=latest_obs.state.to(device),
                        state_mask=latest_obs.state_mask.to(device),
                        action_mask=latest_obs.action_mask.to(device),
                        shallow_tokens=shallow,
                    )
                if isinstance(prediction, torch.Tensor) and prediction.ndim == 2:
                    prediction = prediction.unsqueeze(0)
                if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, head_horizon, action_dim):
                    raise ValueError(
                        f"MOSS action prediction must have shape {(1, head_horizon, action_dim)}, got {getattr(prediction, 'shape', None)}"
                    )
                if not torch.isfinite(prediction).all():
                    raise ValueError("MOSS action prediction contains non-finite values")
                current_plan = prediction[0].detach().to("cpu", dtype=torch.float32).numpy()
                frames = list(session.frames)
                plan_record: Dict[str, Any] = {
                    "plan_index": len(plans),
                    "source_frame_id": source_step,
                    "source_time": float(source_step * dt),
                    "plan_latency_sec": time.monotonic() - started,
                    "frame_ids": [f.frame_id for f in frames],
                    "frame_offsets": [int(f.frame_id - source_step) for f in frames],
                    "frame_ages_sec": [float(source_step * dt - (f.frame_id * dt)) for f in frames],
                    "captured_this_step": captured,
                    "dropped_frame_ids": [],
                    "drop_count": drop_count,
                    "fallback": False,
                    "observation_time_source": time_source,
                    "memory_mode": "consume_session",
                    "flow_steps": flow_steps,
                    "diffusion_seed": plan_seed,
                }
                if plan_seed is not None:
                    plan_record["torch_seed"] = plan_seed
                plans.append(plan_record)

            if current_plan is None:
                raise RuntimeError("No MOSS action plan available")
            offset = step - source_step
            if offset < 0 or offset >= current_plan.shape[0]:
                raise RuntimeError(f"Action offset {offset} out of bounds for source {source_step}")
            action = denormalize_action(
                current_plan[offset],
                norm_stats.get("action", norm_stats.get("actions", {})),
                env.action_space.low,
                env.action_space.high,
            )
            if not np.isfinite(action).all():
                raise ValueError("denormalized action contains non-finite values")
            commands.append(
                {
                    "step": step,
                    "source_frame_id": source_step,
                    "offset": offset,
                    "action_offset": offset,
                    "frame_ids": [int(f.frame_id) for f in session.frames],
                    "frame_offsets": [int(f.frame_id - source_step) for f in session.frames],
                    "source_observation_age_sec": float(time.monotonic() - latest_obs.capture_time) if latest_obs is not None else None,
                    "observation_time_source": time_source,
                    "fallback": False,
                    "drop_count": drop_count,
                    "dropped": False,
                    "action": [float(x) for x in action],
                }
            )
            raw_obs, _, terminated, truncated, info = env.step(action)
            steps = step + 1
            if isinstance(info, dict) and info.get("success", 0):
                success = True
                stop_reason = "env_success"
                break
            if terminated or truncated:
                stop_reason = "env_terminal"
                break
    finally:
        # The session owns only bounded FrameKV tensors; releasing the local
        # reference here makes long MT50 runs independent across episodes.
        try:
            clear = getattr(session, "clear", None)
            if callable(clear):
                clear()
            else:
                session._cache.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

    return {
        "episode_id": episode_id,
        "success": bool(success),
        "steps": steps,
        "stop_reason": stop_reason,
        "executed_actions_count": steps,
        "plans_completed": len(plans),
        "memory_mode": "consume_session",
        "observation_time_source": time_source,
        "observations": observations,
        "plans": plans,
        "command_log": commands,
        "drop_count": drop_count,
        "fallback_count": fallback_count,
        "warmup_terminal": bool(warmup_term or warmup_trunc),
        "step_seconds": dt,
        "async_success_rate": None,
        "native_sota_comparable": False,
    }


def compute_moss_evaluation_summary(
    manifest: Sequence[Dict[str, Any]],
    episodes: Sequence[Dict[str, Any]],
    episodes_per_task: int,
    *,
    status: str,
    contract: Dict[str, Any],
    provenance: Dict[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a scope-explicit summary; async and native SOTA stay separate."""
    if type(episodes_per_task) is not int or episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be a positive integer")
    allowed = {str(item["task_slug"]): item for item in manifest}
    if len(allowed) != len(manifest):
        raise ValueError("Evaluation manifest contains duplicate task slugs")
    seen = set()
    for result in episodes:
        key = (result.get("task_slug"), result.get("episode_index"))
        if (
            key[0] not in allowed
            or type(key[1]) is not int
            or not 0 <= key[1] < episodes_per_task
            or key in seen
        ):
            raise ValueError("Evaluation results contain duplicate or invalid task/episode")
        seen.add(key)
    planned = len(manifest) * episodes_per_task
    completed = len(episodes)
    successes = sum(bool(e.get("success")) for e in episodes)
    by_task: Dict[str, List[Dict[str, Any]]] = {slug: [] for slug in allowed}
    for item in episodes:
        by_task[str(item["task_slug"])].append(item)
    per_task: Dict[str, Any] = {}
    rates: List[float] = []
    for slug, info in allowed.items():
        rows = by_task[slug]
        ok = sum(bool(e.get("success")) for e in rows)
        rate = ok / len(rows) if rows else 0.0
        per_task[slug] = {
            "task_index": info.get("env_task_index"),
            "group": info.get("group", "unknown"),
            "planned": episodes_per_task,
            "completed": len(rows),
            "successes": ok,
            "success_rate": rate,
        }
        if rows:
            rates.append(rate)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for slug, rows in by_task.items():
        groups.setdefault(str(allowed[slug].get("group", "unknown")), []).extend(rows)
    group_summary = {}
    group_rates = []
    for name, rows in groups.items():
        ok = sum(bool(e.get("success")) for e in rows)
        rate = ok / len(rows) if rows else 0.0
        group_summary[name] = {"completed": len(rows), "successes": ok, "success_rate": rate}
        if rows:
            group_rates.append(rate)
    full = bool(contract.get("official_mt50_500") and status == "completed" and completed == planned)
    out: Dict[str, Any] = {
        "status": status,
        "complete": bool(status == "completed" and completed == planned),
        "evaluation_kind": "official_mt50_moss_sync" if full else "moss_sync_partial_or_single_task",
        "official_mt50_500": full,
        "is_smoke": not full,
        "configured_full_mt50": bool(contract.get("configured_official_mt50", False)),
        "is_full_mt50_benchmark": full,
        "scope": contract,
        "counts": {
            "planned_episodes": planned,
            "completed_episodes": completed,
            "successful_episodes": successes,
            "completion_fraction": completed / planned if planned else 0.0,
        },
        "overall": {
            "success_rate": successes / completed if completed else 0.0,
            "task_macro_success_rate": float(np.mean(rates)) if rates else 0.0,
            "group_macro_success_rate": float(np.mean(group_rates)) if group_rates else 0.0,
        },
        "per_task": per_task,
        "difficulty_groups": group_summary,
        "comparison_boundary": {
            "async_success_rate": None,
            "native_sota_success_rate": None,
            "note": "MOSS synchronous consume-session success is reported separately from asynchronous smoke results and native FabriVLA SOTA.",
        },
        "provenance": copy.deepcopy(provenance),
    }
    if error is not None:
        out["error"] = str(error)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--adapter", required=False)
    parser.add_argument("--adapter-stage", choices=("bridge", "expert", "joint"), default="bridge")
    parser.add_argument("--memory-mode", choices=("consume",), default="consume")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--fabri-root", default="/root/FabriVLA")
    parser.add_argument("--vlm-path", "--vlm", dest="vlm_path", default="/root/models/InternVL3_5-1B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--episodes", type=int, default=OFFICIAL_MT50_EPISODES_PER_TASK)
    parser.add_argument("--episode-horizon", type=int, default=OFFICIAL_EPISODE_HORIZON)
    parser.add_argument("--exec-horizon", type=int, default=OFFICIAL_EXEC_HORIZON)
    parser.add_argument("--num-inference-timesteps", "--flow-steps", dest="num_inference_timesteps", type=int, default=OFFICIAL_FLOW_STEPS)
    parser.add_argument("--observation-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4048)
    parser.add_argument("--seed-policy", choices=("per-plan", "once"), default="per-plan")
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--arm-key", default="metaworld_sawyer")
    parser.add_argument("--self-check", action="store_true", help="Run parser/contract checks without loading a model")
    return parser


def parse_args(raw_args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(raw_args)
    if not args.self_check and (not args.checkpoint or not args.adapter or not args.output_dir):
        raise ValueError("--checkpoint, --adapter, and --output-dir are required")
    for name in ("episodes", "episode_horizon", "exec_horizon", "num_inference_timesteps", "observation_stride", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.window is not None and args.window <= 0:
        raise ValueError("--window must be positive")
    if args.step_seconds is not None and (not math.isfinite(args.step_seconds) or args.step_seconds <= 0):
        raise ValueError("--step-seconds must be positive and finite")
    return args


def run_evaluation(
    args: argparse.Namespace,
    *,
    device: Optional[Union[str, torch.device]] = None,
    env_factory: Optional[Callable[[int], Any]] = None,
    loader: Optional[Callable[..., Tuple[MossInternVL, Dict[str, Any], Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Run official MOSS evaluation; ``device='cpu'`` is for mechanism tests."""
    loader = loader or _load_adapter_model
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory '{output_dir}' exists and is not empty. Refusing to overwrite.")
    target_device = torch.device(device if device is not None else args.device)
    if device is None and target_device.type != "cuda":
        raise ValueError("Formal MOSS evaluation requires CUDA; inject device='cpu' only in tests")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({target_device}) but CUDA is unavailable")
    if args.num_inference_timesteps != OFFICIAL_FLOW_STEPS:
        # Keep smoke runs possible, but make their scope unambiguous in the report.
        logger.warning("flow steps=%s; this is not the official 50-step contract", args.num_inference_timesteps)

    from fabri_moss.evaluate_predictive import resolve_task_manifest

    manifest, _idx_to_slug, _groups, metadata_sha = resolve_task_manifest(args.tasks, fabri_root=args.fabri_root)
    tasks_metadata_path = Path(args.fabri_root) / "evaluations" / "metaworld" / "tasks.jsonl"
    tasks_metadata_sha = compute_file_sha256(tasks_metadata_path) if tasks_metadata_path.is_file() else None
    contract = official_mt50_contract(
        task_count=len(manifest),
        episodes_per_task=args.episodes,
        episode_horizon=args.episode_horizon,
        exec_horizon=args.exec_horizon,
        flow_steps=args.num_inference_timesteps,
        observation_stride=args.observation_stride,
    )
    if target_device.type != "cuda":
        # CPU injection is a mechanism test and cannot carry an official
        # benchmark label even when its loop dimensions happen to match.
        contract["configured_official_mt50"] = False
        contract["official_mt50_500"] = False
        contract["is_full_mt50_benchmark"] = False
    if args.step_seconds is not None:
        # An explicit override is useful for mechanism tests, but it is a
        # different timing contract from the official MetaWorld environment.
        contract["configured_official_mt50"] = False
        contract["official_mt50_500"] = False
        contract["is_full_mt50_benchmark"] = False
    if args.seed_policy != "per-plan":
        contract["configured_official_mt50"] = False
        contract["official_mt50_500"] = False
        contract["is_full_mt50_benchmark"] = False
    provenance: Dict[str, Any] = {
        "protocol_name": "official_moss_sync_mt50_v1",
        "source_metadata_sha256": metadata_sha,
        "tasks_metadata_sha256": tasks_metadata_sha,
        "fabri_root": str(Path(args.fabri_root).resolve()),
        "vlm_path": str(Path(args.vlm_path).resolve()),
        "device": str(target_device),
        "seed": args.seed,
        "seed_policy": args.seed_policy,
        "episodes_per_task": args.episodes,
        "episode_horizon": args.episode_horizon,
        "exec_horizon": args.exec_horizon,
        "flow_steps": args.num_inference_timesteps,
        "observation_stride": args.observation_stride,
        "step_seconds_override": args.step_seconds,
        "configured_contract": contract,
        "is_smoke": not bool(contract.get("official_mt50_500")),
        "async_success_rate": None,
        "native_sota_success_rate": None,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "fa2_package": describe_fa2_runtime(),
    }
    checkpoint_meta_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    adapter_meta_path = Path(args.adapter).resolve() if args.adapter else None
    if checkpoint_meta_path is not None and checkpoint_meta_path.is_file():
        provenance["source_checkpoint_sha256"] = compute_file_sha256(checkpoint_meta_path)
        provenance["source_sha256"] = provenance["source_checkpoint_sha256"]
    if adapter_meta_path is not None and adapter_meta_path.is_file():
        provenance["adapter_sha256"] = compute_file_sha256(adapter_meta_path)
        provenance["adapter_sha"] = provenance["adapter_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"metadata_sha256": metadata_sha, "tasks_metadata_sha256": tasks_metadata_sha, "tasks": manifest}, indent=2), encoding="utf-8"
    )
    episodes: List[Dict[str, Any]] = []
    summary_path = output_dir / "summary_report.json"
    jsonl_path = output_dir / "episodes.jsonl"
    envs = None
    active_task: Optional[str] = None
    active_episode: Optional[str] = None

    def flush(status: str, error: Optional[str] = None) -> Dict[str, Any]:
        report = compute_moss_evaluation_summary(
            manifest, episodes, args.episodes, status=status, contract=contract, provenance=provenance, error=error
        )
        if error is not None:
            if active_task is not None:
                report["active_task"] = active_task
            if active_episode is not None:
                report["active_episode"] = active_episode
        tmp = summary_path.with_name(f"{summary_path.name}.tmp.{time.time_ns()}")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp.replace(summary_path)
        return report

    flush("running")
    try:
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32 - 1))
        # Seed model construction once; per-plan sampling is subsequently
        # isolated by ``fixed_diffusion_seed``.
        torch.manual_seed(args.seed)
        if target_device.type == "cuda":
            with torch.cuda.device(target_device):
                torch.cuda.manual_seed(args.seed)
        model, norm_stats, model_provenance = loader(
            checkpoint=args.checkpoint,
            adapter=args.adapter,
            fabri_root=args.fabri_root,
            vlm_path=args.vlm_path,
            device=target_device,
            stage=args.adapter_stage,
            memory_mode=args.memory_mode,
            window=args.window,
            arm_key=args.arm_key,
        )
        if hasattr(model, "eval"):
            model.eval()
        provenance.update(model_provenance)
        native_meta = provenance.get("native_fa2")
        if target_device.type == "cuda" and not (isinstance(native_meta, dict) and native_meta.get("native_fa2_enabled", False)):
            # Keep the CUDA gate at the evaluation boundary as well as in the
            # loader: a custom loader cannot silently substitute eager/SDPA
            # native layers for the calibrated FA2 path.
            provenance["native_fa2"] = assert_native_fa2(getattr(model, "policy", model))
            fa2_meta = provenance.setdefault("fa2_package", describe_fa2_runtime())
            if isinstance(fa2_meta, dict):
                fa2_meta["native_fa2_verified"] = True
        if args.expected_sha256 is not None and provenance.get("source_checkpoint_sha256") != args.expected_sha256.lower():
            raise ValueError("source checkpoint SHA256 does not match --expected-sha256")
        ah_cfg = getattr(getattr(model.policy, "action_head", None), "config", None)
        if ah_cfg is None:
            raise ValueError("MOSS model action head has no config")
        ah_cfg.num_inference_timesteps = args.num_inference_timesteps
        envs = env_factory(args.seed) if env_factory is not None else make_official_mt50_env(args.seed)
        sub_envs = getattr(envs, "envs", None)
        if sub_envs is None:
            sub_envs = getattr(getattr(envs, "unwrapped", None), "envs", None)
        if sub_envs is None:
            raise RuntimeError("MT50 vector environment must expose .envs")
        for task in manifest:
            idx = int(task["env_task_index"])
            if idx >= len(sub_envs):
                raise RuntimeError(f"MT50 environment has {len(sub_envs)} tasks; missing index {idx}")
            for ep_idx in range(args.episodes):
                ep_id = f"{task['task_slug']}_ep_{ep_idx:03d}"
                active_task = str(task["task_slug"])
                active_episode = ep_id
                result = run_episode(
                    env=sub_envs[idx],
                    model=model,
                    prompt=task["prompt"],
                    norm_stats=norm_stats,
                    episode_id=ep_id,
                    task_slug=task["task_slug"],
                    episode_index=ep_idx,
                    master_seed=args.seed,
                    episode_horizon=args.episode_horizon,
                    exec_horizon=args.exec_horizon,
                    observation_stride=args.observation_stride,
                    seed=args.seed + ep_idx,
                    seed_policy=args.seed_policy,
                    flow_steps=args.num_inference_timesteps,
                    step_seconds=args.step_seconds,
                    state_dim=int(getattr(ah_cfg, "state_dim", 24)),
                    action_dim=int(getattr(ah_cfg, "per_action_dim", 24)),
                )
                result.update(
                    {
                        "task_slug": task["task_slug"],
                        "task_index": idx,
                        "group": task.get("group", "unknown"),
                        "episode_index": ep_idx,
                        "seed": args.seed + ep_idx,
                        "source_checkpoint_sha256": provenance.get("source_checkpoint_sha256"),
                        "adapter_sha256": provenance.get("adapter_sha256"),
                        "architecture_revision": provenance.get("architecture_contract", {}).get("architecture_revision")
                        if isinstance(provenance.get("architecture_contract"), dict)
                        else None,
                    }
                )
                for ledger in result.get("plans", []) + result.get("command_log", []):
                    ledger["source_checkpoint_sha256"] = provenance.get("source_checkpoint_sha256")
                    ledger["adapter_sha256"] = provenance.get("adapter_sha256")
                episodes.append(result)
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result) + "\n")
                flush("running")
        provenance["finished_at"] = time.time()
        return flush("completed")
    except Exception as exc:
        flush("failed", str(exc))
        raise
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception:
                logger.exception("Failed to close MT50 environment")


def main() -> None:
    args = parse_args()
    if args.self_check:
        contract = official_mt50_contract(
            task_count=OFFICIAL_MT50_TASK_COUNT,
            episodes_per_task=OFFICIAL_MT50_EPISODES_PER_TASK,
            episode_horizon=OFFICIAL_EPISODE_HORIZON,
            exec_horizon=OFFICIAL_EXEC_HORIZON,
            flow_steps=OFFICIAL_FLOW_STEPS,
        )
        assert contract["official_mt50_500"] is True
        assert derive_plan_seed(4048, "reach-v3", 0, 0) == derive_plan_seed(4048, "reach-v3", 0, 0)
        print("MOSS evaluation self-check passed")
        return
    run_evaluation(args)


if __name__ == "__main__":
    main()
