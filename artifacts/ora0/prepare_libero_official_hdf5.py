#!/usr/bin/env python
"""Convert official LIBERO HDF5 demos to the existing DINO LongTraj format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


DECISION_OFFSETS = np.array((0, 2, 4, 6), dtype=np.int64)
VISION_OFFSETS = np.array((-6, -4, -2, 0), dtype=np.int64)
ACTION_HORIZON = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-dir", type=Path, required=True)
    parser.add_argument("--language-reference", type=Path, required=True)
    parser.add_argument("--longtraj-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-demo", type=int, default=16)
    return parser.parse_args()


def normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scale = np.where(np.abs(high - low) < 1e-6, 1.0, high - low)
    return np.clip(2.0 * (values - low) / scale - 1.0, -1.0, 1.0).astype(np.float32)


def task_description(handle: h5py.File) -> str:
    return json.loads(handle["data"].attrs["problem_info"])["language_instruction"].strip()


def save_new(payload: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary file: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.windows_per_demo < 1:
        raise ValueError("--windows-per-demo must be positive")
    sources = sorted(args.hdf5_dir.glob("*.hdf5"))
    if len(sources) != 10:
        raise ValueError(f"expected 10 LIBERO-Spatial HDF5 files, got {len(sources)}")
    language = torch.load(args.language_reference, map_location="cpu", weights_only=True)
    tasks = list(language["tasks"])
    if len(tasks) != 10:
        raise ValueError("language reference must contain 10 tasks")
    task_to_id = {task: index for index, task in enumerate(tasks)}

    state_rows: list[np.ndarray] = []
    ordered: list[tuple[int, Path]] = []
    for source in sources:
        with h5py.File(source, "r") as handle:
            description = task_description(handle)
            if description not in task_to_id:
                raise ValueError(f"unknown task description in {source}: {description}")
            ordered.append((task_to_id[description], source))
            for name in handle["data"]:
                obs = handle["data"][name]["obs"]
                state_rows.append(
                    np.concatenate((obs["joint_states"][()], obs["gripper_states"][()]), axis=1)
                )
    ordered.sort()
    all_states = np.concatenate(state_rows).astype(np.float32)
    state_low, state_high = np.quantile(all_states, (0.01, 0.99), axis=0).astype(np.float32)

    actions_out: list[np.ndarray] = []
    previous_out: list[np.ndarray] = []
    proprio_out: list[np.ndarray] = []
    instruction_ids: list[int] = []
    episode_ids: list[int] = []
    frame_refs: list[tuple[str, int, list[list[int]]]] = []
    task_counts = [0] * 10
    global_episode = 0
    args.longtraj_dir.mkdir(parents=True, exist_ok=True)

    for task_id, source in ordered:
        task_key = f"libero_spatial_t{task_id:02d}"
        target = args.longtraj_dir / f"metaworld_longtraj_{task_key}.pt"
        episodes = []
        with h5py.File(source, "r") as handle:
            demo_names = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
            if len(demo_names) != 50:
                raise ValueError(f"{source} has {len(demo_names)} demos, expected 50")
            for demo_index, name in enumerate(demo_names):
                demo = handle["data"][name]
                raw_actions = demo["actions"][()].astype(np.float32)
                obs = demo["obs"]
                raw_state = np.concatenate(
                    (obs["joint_states"][()], obs["gripper_states"][()]), axis=1
                ).astype(np.float32)
                # Official files and live robosuite observations use OpenGL orientation.
                # Store upright images for the frozen ImageNet-pretrained DINO tower.
                frames = np.flip(obs["agentview_rgb"][()], axis=1).copy()
                if frames.dtype != np.uint8 or frames.shape[0] != len(raw_actions):
                    raise ValueError(f"invalid frame/action timeline in {source}:{name}")
                episodes.append({"frames": frames})

                max_start = len(raw_actions) - 1 - int(DECISION_OFFSETS[-1]) - ACTION_HORIZON
                if max_start < 0:
                    raise ValueError(f"demo too short for T4/H8/P2: {source}:{name}")
                starts = np.unique(
                    np.linspace(
                        0,
                        max_start,
                        min(args.windows_per_demo, max_start + 1),
                        dtype=np.int64,
                    )
                )
                for start in starts:
                    decisions = start + DECISION_OFFSETS
                    actions_out.append(
                        np.stack([raw_actions[d + 1 : d + 1 + ACTION_HORIZON] for d in decisions])
                    )
                    previous_out.append(raw_actions[decisions])
                    proprio_out.append(normalize(raw_state[decisions], state_low, state_high))
                    frame_index = np.maximum(decisions[:, None] + VISION_OFFSETS[None], 0)
                    frame_refs.append((task_key, demo_index, frame_index.tolist()))
                    instruction_ids.append(task_id)
                    episode_ids.append(global_episode)
                    task_counts[task_id] += 1
                global_episode += 1
        save_new(
            {
                "episodes": episodes,
                "metadata": {
                    "contract": "libero_spatial_official_upright_frames_v1",
                    "task_id": task_id,
                    "task": tasks[task_id],
                    "source": str(source.resolve()),
                },
            },
            target,
        )

    instruction_id = torch.tensor(instruction_ids, dtype=torch.long)
    actions = torch.from_numpy(np.stack(actions_out)).float()
    previous = torch.from_numpy(np.stack(previous_out)).float()
    proprio = torch.from_numpy(np.stack(proprio_out)).float()
    task_hidden = language["language_hidden"].to(torch.float16)
    task_mask = language["language_mask"].bool()
    payload = {
        "actions": actions,
        "previous_action": previous,
        "proprio": proprio,
        "language_hidden": task_hidden[instruction_id],
        "language_mask": task_mask[instruction_id],
        "instruction_id": instruction_id,
        "episode_id": torch.tensor(episode_ids, dtype=torch.long),
        "pair_id": torch.arange(len(actions), dtype=torch.long),
        "frame_refs": frame_refs,
        "normalization": {
            "action_q01": torch.full((7,), -1.0),
            "action_q99": torch.full((7,), 1.0),
            "state_q01": torch.from_numpy(state_low),
            "state_q99": torch.from_numpy(state_high),
        },
        "metadata": {
            "contract": "libero_spatial_official_h8p2_t4_v1",
            "tasks": tasks,
            "n_tasks": 10,
            "n_demos": global_episode,
            "windows_per_demo": args.windows_per_demo,
            "task_counts": task_counts,
            "sequence_length": 4,
            "action_horizon": 8,
            "planning_stride": 2,
            "decision_offsets": DECISION_OFFSETS.tolist(),
            "vision_offsets": VISION_OFFSETS.tolist(),
            "observation_action_alignment": "obs[d]_after_action[d]",
            "target_alignment": "obs[d]_to_actions[d+1:d+9]",
            "previous_action_alignment": "actions[d]",
            "orientation_contract": "vertical_flip_opengl_to_upright_once",
            "action_contract": "raw_libero_osc_pose_minus1_plus1",
            "proprio_contract": "q01q99(joint_states7+gripper_states2)",
        },
    }

    expected = 10 * 50 * args.windows_per_demo
    if len(actions) != expected or task_counts != [50 * args.windows_per_demo] * 10:
        raise AssertionError(f"sample balance mismatch: N={len(actions)}, counts={task_counts}")
    if tuple(actions.shape[1:]) != (4, 8, 7) or tuple(proprio.shape[1:]) != (4, 9):
        raise AssertionError("unexpected model tensor shapes")
    if not torch.equal(previous[:, 1:], actions[:, :-1, 1]):
        raise AssertionError("P2 previous-action alignment failed")
    if not torch.isfinite(actions).all() or float(actions.abs().max()) > 1.000001:
        raise AssertionError("actions are non-finite or outside LIBERO [-1,1]")
    save_new(payload, args.output)
    print(
        f"PASS output={args.output} samples={len(actions)} demos={global_episode} "
        f"actions={tuple(actions.shape)} proprio={tuple(proprio.shape)}"
    )


if __name__ == "__main__":
    main()
