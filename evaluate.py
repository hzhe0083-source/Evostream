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
    LatestAction,
    LatestObservation,
    Observation,
    StreamingActionWorker,
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
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--action-delay-ms", type=float, default=0.0)
    parser.add_argument("--fixed-noise", action="store_true")
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
    if payload.get("format") != "moss_action_v5":
        raise ValueError("policy is not a moss_action_v5 checkpoint")
    contract = payload.get("training_contract", {})
    required_contract = {
        "retained_layers": 24,
        "retained_cross_attention_layers": [2, 6, 10, 14, 18, 22],
        "raw_taps": [14, 18, 23],
        "action_decoder": "cross_attention_streaming_flow",
        "action_memory": "fresh_fused_h14_h18_h23",
        "runtime": "continuous_moss_action_stream_v1",
        "supervision": "every_frame_end",
        "action_generation": "one_velocity_step_per_control_tick",
        "flow_objective": "online_stabilized_plus_trajectory_cfm",
        "stage": "streaming",
        "training_context": "causal_full_episode_prefix",
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
    if (config.state_dim, config.action_dim, config.horizon) != (9, 7, 50):
        raise ValueError(
            "LIBERO evaluation requires state D9, action A7, and horizon H50"
        )
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
    fixed_noise: bool,
    seed: int,
    action_delay_ms: float,
    backbone_mode: str,
):
    noise = None
    if fixed_noise:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            1,
            policy.config.action_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    stream = None
    history_images = []
    history_timestamps = []
    history_origin = None
    action_state = None
    initial_action = torch.zeros(
        1, policy.config.action_dim, device=device, dtype=dtype
    )
    initial_action[0, -1] = -1.0

    def encode(observation: Observation) -> torch.Tensor:
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
            moss_inputs = _move(
                prepare_streaming_moss_inputs(
                    processor,
                    [history_images],
                    [instruction],
                    [history_timestamps],
                ),
                device,
            )
            return policy.encode_memory(moss_inputs)

    def next_action(observation: Observation) -> np.ndarray:
        nonlocal action_state
        state = torch.from_numpy(observation.robot_state).to(device).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            action, action_state = policy.stream_action(
                observation.image,
                state,
                state=action_state,
                reference_action=initial_action if action_state is None else None,
                noise=noise,
                clamp=True,
            )
        result = action[0].float().cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("policy produced non-finite actions")
        if action_delay_ms:
            time.sleep(action_delay_ms / 1000.0)
        return result

    return encode, next_action


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
    action_fn: Any,
    normalization: Mapping[str, Any],
    horizon: int,
) -> tuple[bool, int, dict[str, Any]]:
    gripper = -1.0
    for step in range(horizon):
        snapshot = Observation(
            step + 1,
            _image(observation),
            _robot_state(observation, normalization),
            time.monotonic(),
        )
        encoded = Observation(
            snapshot.version,
            encode_fn(snapshot),
            snapshot.robot_state,
            snapshot.timestamp,
        )
        action = action_fn(encoded)
        gripper = (
            gripper if abs(float(action[-1])) <= 0.2 else float(np.sign(action[-1]))
        )
        action[-1] = gripper
        observation, _, done, _ = env.step(action)
        if env.check_success():
            return True, step + 1, {}
        if done:
            return False, step + 1, {}
    return False, horizon, {}


def _async_rollout(
    env: Any,
    observation: Mapping[str, np.ndarray],
    encode_fn: Any,
    action_fn: Any,
    normalization: Mapping[str, Any],
    horizon: int,
    args: argparse.Namespace,
) -> tuple[bool, int, dict[str, Any]]:
    period = 1.0 / args.control_hz
    mailbox = LatestObservation()
    features = LatestObservation()
    actions = LatestAction()
    perception = AsyncPerception(mailbox, features, encode_fn)
    action_worker = StreamingActionWorker(
        features, actions, action_fn, period=period, states=mailbox
    )
    perception.start()
    action_worker.start()
    mailbox.publish(_image(observation), _robot_state(observation, normalization))
    current = actions.wait_for_new(0, timeout=args.startup_timeout)
    if current is None:
        action_worker.stop()
        perception.stop()
        perception.raise_if_failed()
        action_worker.raise_if_failed()
        raise TimeoutError("the first streamed action did not arrive in time")
    success = False
    steps = 0
    version = current.version
    gripper = -1.0
    repeated_actions = 0
    action_ages = []
    deadline = time.monotonic()
    rollout_started = deadline
    try:
        for steps in range(1, horizon + 1):
            if steps > 1:
                latest = actions.wait_for_new(version, timeout=0.0)
                if latest is None:
                    repeated_actions += 1
                else:
                    current = latest
                    version = latest.version
            action = current.value.copy()
            gripper = (
                gripper
                if abs(float(action[-1])) <= 0.2
                else float(np.sign(action[-1]))
            )
            action[-1] = gripper
            action_ages.append(max(0.0, time.monotonic() - current.observation_time))
            observation, _, done, _ = env.step(action)
            success = bool(env.check_success())
            if success or done:
                break
            mailbox.publish(
                _image(observation), _robot_state(observation, normalization)
            )
            perception.raise_if_failed()
            action_worker.raise_if_failed()
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        action_worker.stop()
        perception.stop()
    perception.raise_if_failed()
    action_worker.raise_if_failed()
    elapsed = max(time.monotonic() - rollout_started, 1e-6)
    stats = {
        "actions_generated": action_worker.actions_generated,
        "repeated_actions": repeated_actions,
        "stale_action_fraction": repeated_actions / max(steps, 1),
        "mean_action_age": float(np.mean(action_ages)) if action_ages else 0.0,
        "max_action_age": max(action_ages, default=0.0),
        "action_flow_last_latency": action_worker.last_latency,
        "action_frequency_hz": action_worker.actions_generated / elapsed,
    }
    stats["perception_last_latency"] = perception.last_latency
    stats["frames_encoded"] = perception.frames_encoded
    stats["perception_frequency_hz"] = perception.frames_encoded / elapsed
    return success, steps, stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy, processor, payload, device, dtype = _build_policy(args)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    horizon = args.horizon or SUITE_HORIZONS[args.suite]
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
                encode_fn, action_fn = _pipeline_functions(
                    policy,
                    processor,
                    task.language.strip(),
                    device,
                    dtype,
                    args.fixed_noise,
                    seed,
                    args.action_delay_ms,
                    args.backbone_mode,
                )
                rollout = _async_rollout if args.mode == "async" else _blocking_rollout
                if args.mode == "async":
                    success, steps, metrics = rollout(
                        env,
                        observation,
                        encode_fn,
                        action_fn,
                        payload["normalization"],
                        horizon,
                        args,
                    )
                else:
                    success, steps, metrics = rollout(
                        env,
                        observation,
                        encode_fn,
                        action_fn,
                        payload["normalization"],
                        horizon,
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
