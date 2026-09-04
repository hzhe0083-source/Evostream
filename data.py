"""Direct LIBERO HDF5 adapter for MOSS-Action."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model import realtime_frame_segment


@dataclass(frozen=True)
class LiberoSampleRef:
    path: Path
    demo: str
    instruction: str


def discover_hdf5(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    files = [path] if path.is_file() else sorted(path.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no LIBERO HDF5 files found under {path}")
    return files


def upright_libero_image(image: np.ndarray) -> np.ndarray:
    """Convert one OpenGL-oriented LIBERO frame to upright RGB exactly once."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image[::-1])


def normalize_state(state: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if state.shape[-1] != low.shape[0] or low.shape != high.shape:
        raise ValueError("state and normalization bounds do not align")
    scale = np.where(np.abs(high - low) < 1e-6, 1.0, high - low)
    return np.clip(2.0 * (state - low) / scale - 1.0, -1.0, 1.0).astype(np.float32)


def _instruction(handle: Any, path: Path) -> str:
    raw = handle["data"].attrs.get("problem_info")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(raw)["language_instruction"].strip()
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid LIBERO problem_info in {path}") from error
    if not value:
        raise ValueError(f"empty LIBERO instruction in {path}")
    return value


def _demo_names(data_group: Any) -> list[str]:
    try:
        return sorted(data_group, key=lambda name: int(name.rsplit("_", 1)[1]))
    except (IndexError, ValueError) as error:
        raise ValueError("LIBERO demo groups must end in numeric ids") from error


class LiberoHDF5Dataset(Dataset):
    """One full causal episode with an action target at every sampled frame."""

    def __init__(
        self,
        data: str | Path | Sequence[str | Path],
        *,
        horizon: int = 50,
        action_offset: int = 1,
        frame_stride: int = 1,
        frame_interval: float = 0.1,
        state_low: np.ndarray | None = None,
        state_high: np.ndarray | None = None,
    ) -> None:
        if (
            horizon < 2
            or action_offset < 0
            or frame_stride < 1
            or frame_interval <= 0
        ):
            raise ValueError(
                "horizon must be at least two; stride/frame_interval positive; "
                "action_offset nonnegative"
            )
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("LIBERO data loading requires h5py") from error

        entries = [data] if isinstance(data, (str, Path)) else list(data)
        self.files = sorted(
            {file for entry in entries for file in discover_hdf5(entry)}
        )
        self.horizon = horizon
        self.action_offset = action_offset
        self.frame_stride = frame_stride
        self.frame_interval = frame_interval
        self.samples: list[LiberoSampleRef] = []
        states: list[np.ndarray] = []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                instruction = _instruction(handle, path)
                for demo_name in _demo_names(handle["data"]):
                    demo = handle["data"][demo_name]
                    actions = demo["actions"]
                    observations = demo["obs"]
                    episode_states = np.concatenate(
                        (
                            observations["joint_states"][()],
                            observations["gripper_states"][()],
                        ),
                        axis=-1,
                    ).astype(np.float32)
                    if episode_states.ndim != 2 or episode_states.shape[1] != 9:
                        raise ValueError(
                            f"{path}:{demo_name} does not contain 9-D state"
                        )
                    if actions.ndim != 2 or actions.shape[1] != 7:
                        raise ValueError(
                            f"{path}:{demo_name} does not contain 7-D actions"
                        )
                    last_observation = min(
                        len(episode_states),
                        len(observations["agentview_rgb"]),
                        len(actions) - action_offset,
                    )
                    if last_observation < 1:
                        raise ValueError(
                            f"{path}:{demo_name} has no causally aligned action"
                        )
                    states.append(episode_states)
                    self.samples.append(
                        LiberoSampleRef(path, demo_name, instruction)
                    )

        if not self.samples:
            raise ValueError("no causally aligned LIBERO training samples found")
        all_states = np.concatenate(states, axis=0)
        computed_low, computed_high = np.quantile(all_states, (0.01, 0.99), axis=0)
        self.state_low = np.asarray(
            computed_low if state_low is None else state_low, dtype=np.float32
        )
        self.state_high = np.asarray(
            computed_high if state_high is None else state_high, dtype=np.float32
        )
        if self.state_low.shape != (9,) or self.state_high.shape != (9,):
            raise ValueError("LIBERO state normalization bounds must have shape [9]")
        if (
            not np.isfinite(self.state_low).all()
            or not np.isfinite(self.state_high).all()
        ):
            raise ValueError("LIBERO state normalization bounds must be finite")
        if not np.all(self.state_high > self.state_low):
            raise ValueError(
                "every LIBERO state upper bound must exceed its lower bound"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import h5py

        ref = self.samples[index]
        # ponytail: reopen per sample for worker safety; cache handles only if I/O profiles hot.
        with h5py.File(ref.path, "r") as handle:
            demo = handle["data"][ref.demo]
            observations = demo["obs"]
            last_observation = min(
                len(observations["joint_states"]),
                len(observations["agentview_rgb"]),
                len(demo["actions"]) - self.action_offset,
            )
            supervised_indices = list(range(0, last_observation, self.frame_stride))
            frame_indices = supervised_indices
            images = [
                Image.fromarray(upright_libero_image(observations["agentview_rgb"][step]))
                for step in frame_indices
            ]
            raw_states = np.concatenate(
                (
                    observations["joint_states"][supervised_indices],
                    observations["gripper_states"][supervised_indices],
                ),
                axis=-1,
            ).astype(np.float32)
            chunks = []
            valid_masks = []
            flow_centers = []
            flow_times = []
            for step in supervised_indices:
                start = step + self.action_offset
                chunk = demo["actions"][start : start + self.horizon].astype(
                    np.float32
                )
                valid = np.arange(self.horizon) < len(chunk)
                if not valid.any():
                    raise RuntimeError("stream frame produced an empty action target")
                if len(chunk) < self.horizon:
                    chunk = np.concatenate(
                        (
                            chunk,
                            np.repeat(chunk[-1:], self.horizon - len(chunk), axis=0),
                        )
                    )
                chunks.append(chunk)
                valid_masks.append(valid)
                flow_centers.append(
                    demo["actions"][max(0, start - 1)].astype(np.float32)
                )
                flow_times.append(
                    (step % (self.horizon - 1)) / (self.horizon - 1)
                )

        actions = np.stack(chunks)
        if not np.isfinite(actions).all() or np.abs(actions).max(initial=0.0) > 1.0001:
            raise ValueError(
                f"raw OSC actions outside [-1,1] at {ref.path}:{ref.demo}"
            )
        return {
            "stream_id": (str(ref.path), ref.demo),
            "images": images,
            "supervised_frames": len(supervised_indices),
            "frame_timestamps": np.asarray(frame_indices, dtype=np.float32)
            * self.frame_interval,
            "instruction": ref.instruction,
            "robot_state": normalize_state(
                raw_states, self.state_low, self.state_high
            ),
            "flow_centers": np.stack(flow_centers),
            "flow_times": np.asarray(flow_times, dtype=np.float32),
            "actions": actions,
            "action_valid_mask": np.stack(valid_masks),
        }

    @property
    def normalization(self) -> dict[str, list[float]]:
        return {
            "state_q01": self.state_low.tolist(),
            "state_q99": self.state_high.tolist(),
            "action_low": [-1.0] * 7,
            "action_high": [1.0] * 7,
        }


def prepare_moss_inputs(
    processor: Any,
    images: Iterable[Image.Image],
    instructions: Iterable[str],
) -> dict[str, Any]:
    images = list(images)
    instructions = list(instructions)
    if len(images) != len(instructions) or not images:
        raise ValueError(
            "images and instructions must be nonempty and have equal length"
        )
    texts = [
        processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image|>\n{instruction}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for instruction in instructions
    ]
    return dict(
        processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
    )


def prepare_streaming_moss_inputs(
    processor: Any,
    image_sequences: Iterable[Sequence[Image.Image]],
    instructions: Iterable[str],
    frame_timestamps: Iterable[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Build a causal multi-frame pass equivalent to incremental stream appends."""
    sequences = [list(sequence) for sequence in image_sequences]
    instructions = list(instructions)
    if len(sequences) != len(instructions) or not sequences:
        raise ValueError(
            "image sequences and instructions must be nonempty and have equal length"
        )
    if any(not sequence for sequence in sequences):
        raise ValueError("every training sample needs at least one frame")
    timestamps = (
        [list(values) for values in frame_timestamps]
        if frame_timestamps is not None
        else [list(np.arange(len(sequence), dtype=float) * 0.1) for sequence in sequences]
    )
    if len(timestamps) != len(sequences) or any(
        len(values) != len(sequence)
        for values, sequence in zip(timestamps, sequences)
    ):
        raise ValueError("frame timestamps must align with every image sequence")
    texts = []
    for instruction, values in zip(instructions, timestamps):
        if any(
            not np.isfinite(value)
            or value < 0
            or (index and value < values[index - 1])
            for index, value in enumerate(values)
        ):
            raise ValueError("frame timestamps must be finite and non-decreasing")
        prefix = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(
            prefix + "".join(realtime_frame_segment(float(value)) for value in values)
        )
    return dict(
        processor(
            text=texts,
            images=[image for sequence in sequences for image in sequence],
            padding=True,
            return_tensors="pt",
        )
    )


class MossActionCollator:
    def __init__(self, processor: Any):
        self.processor = processor
        self.frame_end_token_id = getattr(
            processor,
            "vision_end_token_id",
            processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
        if not isinstance(self.frame_end_token_id, int) or self.frame_end_token_id < 0:
            raise ValueError("MOSS tokenizer is missing <|vision_end|>")

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        moss_inputs = prepare_streaming_moss_inputs(
            self.processor,
            (row["images"] for row in rows),
            (row["instruction"] for row in rows),
            (row["frame_timestamps"] for row in rows),
        )
        frame_token_mask = moss_inputs["input_ids"].eq(self.frame_end_token_id)
        actual = frame_token_mask.sum(dim=1).tolist()
        expected = [len(row["images"]) for row in rows]
        if actual != expected:
            raise ValueError(
                f"frame-end tokens do not align with stream frames: {actual} != {expected}"
            )
        action_token_mask = torch.zeros_like(frame_token_mask)
        for sample, row in enumerate(rows):
            positions = frame_token_mask[sample].nonzero(as_tuple=True)[0]
            action_token_mask[sample, positions[-row["supervised_frames"] :]] = True
        return {
            "moss_inputs": moss_inputs,
            "action_token_mask": action_token_mask,
            "robot_state": torch.from_numpy(
                np.concatenate([row["robot_state"] for row in rows], axis=0)
            ),
            "flow_centers": torch.from_numpy(
                np.concatenate([row["flow_centers"] for row in rows], axis=0)
            ),
            "flow_times": torch.from_numpy(
                np.concatenate([row["flow_times"] for row in rows], axis=0)
            ),
            "actions": torch.from_numpy(
                np.concatenate([row["actions"] for row in rows], axis=0)
            ),
            "action_valid_mask": torch.from_numpy(
                np.concatenate([row["action_valid_mask"] for row in rows], axis=0)
            ),
        }
