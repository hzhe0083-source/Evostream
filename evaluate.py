"""LIBERO evaluation with blocking or split perception/action streaming."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from data import normalize_state, prepare_streaming_moss_inputs, upright_libero_image
from model import (
    MossActionConfig,
    MossActionVLA,
    checkpoint_fingerprint,
    load_trainable_state_dict,
    load_truncated_moss,
)
from streaming import (
    AsyncPerception,
    ChunkExecutor,
    ChunkPlanner,
    LatestObservation,
    Observation,
)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

SUITE_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _task_ids(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if (
        not result
        or len(result) != len(set(result))
        or any(not 0 <= item < 10 for item in result)
    ):
        raise argparse.ArgumentTypeError("task ids must be unique values in 0..9")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--moss-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(SUITE_HORIZONS), default="libero_10")
    parser.add_argument("--task-ids", type=_task_ids, default=list(range(10)))
    parser.add_argument("--trials-per-task", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--mode", choices=("async", "blocking"), default="async")
    parser.add_argument(
        "--backbone-mode",
        choices=("streaming-kv", "full-recompute"),
        default="streaming-kv",
    )
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument(
        "--plan-hz", type=float, default=5.0, help="micro-chunk replanning rate"
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=0,
        help="blocking mode: actions executed per chunk (0 = half the chunk)",
    )
    parser.add_argument(
        "--ensemble-lambda",
        type=float,
        help="enable ACT-style temporal ensembling with this decay; "
        "omit to let the newest chunk win and measure raw boundary jumps",
    )
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--action-delay-ms", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attention-backend", default="flash_attention_2")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    if not 1 <= args.trials_per_task <= 50:
        parser.error("--trials-per-task must be in 1..50")
    if args.horizon < 0 or args.control_hz <= 0:
        parser.error("horizon must be nonnegative and frequency must be positive")
    if args.settle_steps < 0 or args.startup_timeout <= 0 or args.camera_size < 16:
        parser.error("invalid settle, timeout, or camera size")
    if args.action_delay_ms < 0:
        parser.error("--action-delay-ms must be nonnegative")
    if args.plan_hz <= 0 or args.plan_hz > args.control_hz:
        parser.error("--plan-hz must be positive and at most --control-hz")
    if args.execute_steps < 0:
        parser.error("--execute-steps must be nonnegative")
    if args.ensemble_lambda is not None and args.ensemble_lambda < 0:
        parser.error("--ensemble-lambda must be nonnegative")
    return args


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _fixed_init_states(suite: Any, task_id: int) -> np.ndarray:
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def _robot_state(
    observation: Mapping[str, np.ndarray], normalization: Mapping[str, Any]
) -> np.ndarray:
    raw = np.concatenate(
        (observation["robot0_joint_pos"], observation["robot0_gripper_qpos"])
    ).astype(np.float32)
    return normalize_state(
        raw,
        np.asarray(normalization["state_q01"], dtype=np.float32),
        np.asarray(normalization["state_q99"], dtype=np.float32),
    )


def _image(observation: Mapping[str, np.ndarray]) -> Image.Image:
    return Image.fromarray(upright_libero_image(observation["agentview_image"]))


def _build_policy(args: argparse.Namespace):
    payload = torch.load(args.policy, map_location="cpu", weights_only=True)
    if payload.get("format") != "moss_action_v6":
        raise ValueError("policy is not a moss_action_v6 checkpoint")
    contract = payload.get("training_contract", {})
    required_contract = {
        "retained_layers": 24,
        "retained_cross_attention_layers": [2, 6, 10, 14, 18, 22],
        "action_decoder": "ephemeral_action_queries",
        "action_memory": "final_hidden_state",
        "action_history": "none",
        "runtime": "streaming_action_query_decoder_v1",
        "supervision": "one_micro_chunk_per_planning_time",
        "action_generation": "continuous_micro_chunk_l1",
        "delay_aware": "per_query_visual_age",
        "stage": "streaming",
        "training_context": "single_planning_time",
        "backbone_training": "full",
    }
    mismatched = {
        key: (contract.get(key), value)
        for key, value in required_contract.items()
        if contract.get(key) != value
    }
    if mismatched:
        raise ValueError(f"policy architecture contract mismatch: {mismatched}")
    if (
        int(contract.get("frame_stride", 0)) < 1
        or float(contract.get("frame_interval", 0.0)) <= 0
    ):
        raise ValueError("policy was not trained with a valid causal stream")
    config = MossActionConfig(**payload["config"])
    if (config.state_dim, config.action_dim) != (9, 7):
        raise ValueError("LIBERO evaluation requires state D9 and action A7")
    normalization = payload.get("normalization", {})
    low = np.asarray(normalization.get("state_q01"), dtype=np.float32)
    high = np.asarray(normalization.get("state_q99"), dtype=np.float32)
    if (
        low.shape != (9,)
        or high.shape != (9,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or not np.all(high > low)
    ):
        raise ValueError("policy has invalid LIBERO state normalization")
    actual_fingerprint = checkpoint_fingerprint(args.moss_checkpoint)
    if (
        actual_fingerprint["combined_sha256"]
        != payload["moss_checkpoint"]["combined_sha256"]
    ):
        raise ValueError("MOSS-VL checkpoint SHA256 does not match policy training")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = _dtype(args.dtype) if device.type == "cuda" else torch.float32
    backbone, audit = load_truncated_moss(
        args.moss_checkpoint,
        dtype=dtype,
        device_map={"": str(device)},
        attention_backend=args.attention_backend if device.type == "cuda" else "eager",
    )
    if audit.to_dict() != payload["checkpoint_audit"]:
        raise ValueError(
            "checkpoint loading audit differs from the training-time audit"
    )
    policy = MossActionVLA(backbone, config).to(device)
    load_trainable_state_dict(policy, payload["trainable_state"])
    del payload["trainable_state"]
    policy.eval()
    processor = AutoProcessor.from_pretrained(
        str(args.moss_checkpoint.expanduser().resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    return policy, processor, payload, device, dtype


def _pipeline_functions(
    policy: MossActionVLA,
    processor: Any,
    instruction: str,
    device: torch.device,
    dtype: torch.dtype,
    action_delay_ms: float,
    backbone_mode: str,
    control_interval: float,
):
    """Return (encode_frame, plan_chunk).

    `encode_frame` only grows the visual cache; `plan_chunk` decodes one
    micro-chunk from ephemeral queries and returns it with the timing metadata
    the executor needs to place each action on the physical timeline.
    """
    stream = None
    history_images: list[Any] = []
    history_timestamps: list[float] = []
    history_origin: float | None = None
    previous_state: np.ndarray | None = None
    previous_time: float | None = None

    def encode(observation: Observation) -> float:
        nonlocal history_origin, stream
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            if backbone_mode == "streaming-kv":
                if stream is None:
                    stream = policy.create_stream(processor, instruction)
                return stream.encode_frame(
                    observation.image, timestamp=observation.timestamp
                )
            if history_origin is None:
                history_origin = observation.timestamp
            history_images.append(observation.image)
            history_timestamps.append(observation.timestamp - history_origin)
            return observation.timestamp

    def plan(observation: Observation) -> dict[str, Any]:
        nonlocal previous_state, previous_time
        current = observation.robot_state
        started = time.monotonic()
        obs_time = float(observation.timestamp)
        if previous_state is None or previous_time is None:
            velocity = np.zeros_like(current)
        else:
            dt = max(1e-4, obs_time - previous_time)
            velocity = (current - previous_state) / dt
        previous_state = current.copy()
        previous_time = obs_time

        state = torch.from_numpy(current).to(device=device, dtype=dtype).unsqueeze(0)
        vel = torch.from_numpy(velocity).to(device=device, dtype=dtype).unsqueeze(0)
        injected_latency = (action_delay_ms / 1000.0) if action_delay_ms else 0.0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            if backbone_mode == "streaming-kv":
                if stream is None:
                    raise RuntimeError("a frame must be encoded before planning")
                # Add injected delay on top of tracked EMA compute latency
                total_latency = stream.plan_latency_ema + injected_latency
                chunk = stream.plan(
                    state,
                    vel,
                    plan_timestamp=started,
                    inference_latency=total_latency,
                )
                actions = chunk.actions
                visual_age = chunk.visual_age
            else:
                moss_inputs = _move(
                    prepare_streaming_moss_inputs(
                        processor,
                        [history_images],
                        [instruction],
                        [history_timestamps],
                    ),
                    device,
                )
                visual_age = max(
                    0.0,
                    started - (history_origin + history_timestamps[-1]),
                )
                age = torch.full((1,), visual_age, device=device, dtype=dtype)
                total_latency = 0.05 + injected_latency
                actions = policy.predict_chunk(
                    moss_inputs,
                    state,
                    vel,
                    age,
                    query_delays=policy.default_query_delays(
                        age, inference_latency=total_latency
                    ),
                )[0].clamp(-1.0, 1.0)
        result = actions.float().cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("policy produced non-finite actions")
        if action_delay_ms:
            time.sleep(action_delay_ms / 1000.0)
        return {
            "actions": result,
            "start_time": time.monotonic(),
            "interval": control_interval,
            "visual_age": visual_age,
            "plan_latency": time.monotonic() - started,
        }

    return encode, plan


def _settle(
    env: Any,
    observation: Mapping[str, np.ndarray],
    steps: int,
) -> Mapping[str, np.ndarray]:
    action = np.zeros(7, dtype=np.float32)
    action[-1] = -1.0
    for _ in range(steps):
        observation, _, _, _ = env.step(action)
    return observation


def _blocking_rollout(
    env: Any,
    observation: Mapping[str, np.ndarray],
    encode_fn: Any,
    plan_fn: Any,
    normalization: Mapping[str, Any],
    horizon: int,
    execute_steps: int,
    ensemble_lambda: float | None,
    control_interval: float,
) -> tuple[bool, int, dict[str, Any]]:
    """Receding horizon: replan every `execute_steps` actions, drop the rest."""
    executor = ChunkExecutor(ensemble_lambda=ensemble_lambda)
    gripper = -1.0
    steps = 0
    chunks_planned = 0
    jumps: list[float] = []
    previous = None
    while steps < horizon:
        snapshot = Observation(
            steps + 1,
            _image(observation),
            _robot_state(observation, normalization),
            time.monotonic(),
        )
        encode_fn(snapshot)
        plan = plan_fn(snapshot)
        # A virtual clock keeps blocking mode deterministic: chunk index j is
        # simply the j-th executed step, independent of wall-clock jitter.
        executor.submit(plan["actions"], float(steps), 1.0, plan["visual_age"])
        chunks_planned += 1
        for offset in range(execute_steps):
            if steps >= horizon:
                break
            action = executor.action_at(float(steps)).copy()
            if previous is not None and offset == 0:
                jumps.append(float(np.abs(action - previous).max()))
            previous = action.copy()
            gripper = (
                gripper if abs(float(action[-1])) <= 0.2 else float(np.sign(action[-1]))
            )
            action[-1] = gripper
            observation, _, done, _ = env.step(action)
            steps += 1
            if env.check_success():
                return True, steps, _chunk_stats(chunks_planned, jumps)
            if done:
                return False, steps, _chunk_stats(chunks_planned, jumps)
    return False, steps, _chunk_stats(chunks_planned, jumps)


def _chunk_stats(chunks_planned: int, jumps: list[float]) -> dict[str, Any]:
    return {
        "chunks_planned": chunks_planned,
        "mean_boundary_jump": float(np.mean(jumps)) if jumps else 0.0,
        "max_boundary_jump": max(jumps, default=0.0),
    }


def _async_rollout(
    env: Any,
    observation: Mapping[str, np.ndarray],
    encode_fn: Any,
    plan_fn: Any,
    normalization: Mapping[str, Any],
    horizon: int,
    args: argparse.Namespace,
) -> tuple[bool, int, dict[str, Any]]:
    period = 1.0 / args.control_hz
    mailbox = LatestObservation()
    features = LatestObservation()
    executor = ChunkExecutor(ensemble_lambda=args.ensemble_lambda)
    perception = AsyncPerception(mailbox, features, encode_fn)
    planner = ChunkPlanner(
        features,
        executor,
        plan_fn,
        period=1.0 / args.plan_hz,
        states=mailbox,
    )
    perception.start()
    planner.start()
    mailbox.publish(_image(observation), _robot_state(observation, normalization))
    if not planner.wait_for_first_chunk(args.startup_timeout):
        planner.stop()
        perception.stop()
        perception.raise_if_failed()
        planner.raise_if_failed()
        raise TimeoutError("the first planned chunk did not arrive in time")
    planner.raise_if_failed()
    success = False
    steps = 0
    gripper = -1.0
    stalls = 0
    jumps: list[float] = []
    previous = None
    deadline = time.monotonic()
    rollout_started = deadline
    try:
        for steps in range(1, horizon + 1):
            try:
                action = executor.action_at(time.monotonic()).copy()
            except RuntimeError:
                # The newest chunk ran out before a replan landed; hold the last
                # action rather than pretending a fresh one existed.
                if previous is None:
                    raise
                stalls += 1
                action = previous.copy()
            if previous is not None:
                jumps.append(float(np.abs(action - previous).max()))
            previous = action.copy()
            gripper = (
                gripper
                if abs(float(action[-1])) <= 0.2
                else float(np.sign(action[-1]))
            )
            action[-1] = gripper
            observation, _, done, _ = env.step(action)
            success = bool(env.check_success())
            if success or done:
                break
            mailbox.publish(
                _image(observation), _robot_state(observation, normalization)
            )
            perception.raise_if_failed()
            planner.raise_if_failed()
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        planner.stop()
        perception.stop()
    perception.raise_if_failed()
    planner.raise_if_failed()
    elapsed = max(time.monotonic() - rollout_started, 1e-6)
    return success, steps, {
        "chunks_planned": planner.chunks_planned,
        "chunk_stalls": stalls,
        "stall_fraction": stalls / max(steps, 1),
        "mean_step_delta": float(np.mean(jumps)) if jumps else 0.0,
        "max_step_delta": max(jumps, default=0.0),
        "plan_last_latency": planner.last_latency,
        "plan_frequency_hz": planner.chunks_planned / elapsed,
        "last_visual_age": planner.last_visual_age,
        "perception_last_latency": perception.last_latency,
        "frames_encoded": perception.frames_encoded,
        "perception_frequency_hz": perception.frames_encoded / elapsed,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy, processor, payload, device, dtype = _build_policy(args)

    expected_interval = float(policy.config.control_interval)
    actual_interval = 1.0 / float(args.control_hz)
    if abs(actual_interval - expected_interval) > 1e-4:
        raise ValueError(
            f"control frequency mismatch: --control-hz {args.control_hz} corresponds to "
            f"{actual_interval:.4f}s per step, but policy was trained with control_interval "
            f"= {expected_interval:.4f}s. LIBERO evaluation requires --control-hz {1.0 / expected_interval:.1f}"
        )

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    horizon = args.horizon or SUITE_HORIZONS[args.suite]
    execute_steps = args.execute_steps or max(1, policy.config.chunk_size // 2)
    if execute_steps > policy.config.chunk_size:
        raise ValueError(
            f"--execute-steps {execute_steps} exceeds chunk size "
            f"{policy.config.chunk_size}"
        )
    records = []
    per_task_success = []
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(task_id),
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
            camera_names="agentview",
        )
        init_states = _fixed_init_states(suite, task_id)
        if len(init_states) < args.trials_per_task:
            raise ValueError(
                f"task {task_id} has only {len(init_states)} fixed initial states"
            )
        init_states = init_states[: args.trials_per_task]
        wins = 0
        try:
            for trial, init_state in enumerate(init_states):
                seed = args.seed + task_id * 100 + trial
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
                observation = env.set_init_state(init_state)
                observation = _settle(env, observation, args.settle_steps)
                encode_fn, plan_fn = _pipeline_functions(
                    policy,
                    processor,
                    task.language.strip(),
                    device,
                    dtype,
                    args.action_delay_ms,
                    args.backbone_mode,
                    1.0 / args.control_hz,
                )
                if args.mode == "async":
                    success, steps, metrics = _async_rollout(
                        env,
                        observation,
                        encode_fn,
                        plan_fn,
                        payload["normalization"],
                        horizon,
                        args,
                    )
                else:
                    success, steps, metrics = _blocking_rollout(
                        env,
                        observation,
                        encode_fn,
                        plan_fn,
                        payload["normalization"],
                        horizon,
                        execute_steps,
                        args.ensemble_lambda,
                        1.0 / args.control_hz,
                    )
                wins += int(success)
                record = {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task": task.language,
                    "trial": trial,
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "mode": args.mode,
                    "backbone_mode": args.backbone_mode,
                    **metrics,
                }
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        finally:
            env.close()
        rate = wins / len(init_states)
        per_task_success.append(rate)
        print(f"task={task_id} success={wins}/{len(init_states)}", flush=True)

    summary = {
        "suite": args.suite,
        "mode": args.mode,
        "backbone_mode": args.backbone_mode,
        "action_delay_ms": args.action_delay_ms,
        "chunk_size": policy.config.chunk_size,
        "execute_steps": execute_steps,
        "plan_hz": args.plan_hz,
        "ensemble_lambda": args.ensemble_lambda,
        "task_macro_success": float(np.mean(per_task_success)),
        "overall_success": float(np.mean([record["success"] for record in records])),
        "per_task_success": dict(zip(map(str, args.task_ids), per_task_success)),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
