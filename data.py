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
    """One planning time per sample: a visual prefix plus one action chunk.

    Every item is a single planning decision, not a whole episode. That keeps the
    ephemeral action queries ephemeral: because a sample stops at the planning
    time, no later frame can ever attend to a chunk that was only planned. The
    prefix is deliberately cut `visual_age` seconds before the planning time so
    training sees the same stale-vision regime the asynchronous runtime produces.
    """

    def __init__(
        self,
        data: str | Path | Sequence[str | Path],
        *,
        chunk_size: int = 8,
        action_offset: int = 1,
        frame_stride: int = 1,
        frame_interval: float = 0.1,
        context_frames: int = 8,
        max_visual_age_steps: int = 2,
        state_low: np.ndarray | None = None,
        state_high: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if (
            chunk_size < 1
            or action_offset < 0
            or frame_stride < 1
            or frame_interval <= 0
            or context_frames < 1
            or max_visual_age_steps < 0
        ):
            raise ValueError(
                "chunk_size/context_frames/stride/frame_interval must be positive; "
                "action_offset and max_visual_age_steps nonnegative"
            )
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("LIBERO data loading requires h5py") from error

        entries = [data] if isinstance(data, (str, Path)) else list(data)
        self.files = sorted(
            {file for entry in entries for file in discover_hdf5(entry)}
        )
        self.chunk_size = chunk_size
        self.action_offset = action_offset
        self.frame_stride = frame_stride
        self.frame_interval = frame_interval
        self.context_frames = context_frames
        self.max_visual_age_steps = max_visual_age_steps
        self.seed = seed
        self.samples: list[LiberoSampleRef] = []
        self.planning_times: list[int] = []
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
                    reference = LiberoSampleRef(path, demo_name, instruction)
                    for planning_time in range(0, last_observation, frame_stride):
                        self.samples.append(reference)
                        self.planning_times.append(planning_time)

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

    def _visual_age_steps(self, index: int) -> int:
        """Deterministic per-sample staleness so epochs stay reproducible."""
        if self.max_visual_age_steps == 0:
            return 0
        generator = np.random.default_rng((self.seed, index))
        return int(generator.integers(0, self.max_visual_age_steps + 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        import h5py

        ref = self.samples[index]
        planning_time = self.planning_times[index]
        # Stale-vision conditioning: the newest frame may be several steps behind
        # the planning time, but cannot be before the start of the episode.
        raw_age_steps = self._visual_age_steps(index)
        age_steps = min(raw_age_steps, planning_time)
        # ponytail: reopen per sample for worker safety; cache handles only if I/O profiles hot.
        with h5py.File(ref.path, "r") as handle:
            demo = handle["data"][ref.demo]
            observations = demo["obs"]

            newest_frame = planning_time - age_steps
            first_frame = max(
                0, newest_frame - (self.context_frames - 1) * self.frame_stride
            )
            frame_indices = list(
                range(first_frame, newest_frame + 1, self.frame_stride)
            )
            if frame_indices[-1] != newest_frame:
                frame_indices.append(newest_frame)
            images = [
                Image.fromarray(upright_libero_image(observations["agentview_rgb"][step]))
                for step in frame_indices
            ]

            previous_time = max(0, planning_time - 1)
            # h5py fancy indexing requires increasing order, so read the pair in
            # chronological order and keep [current, previous] afterwards.
            rows = sorted({previous_time, planning_time})
            raw_states = np.concatenate(
                (
                    observations["joint_states"][rows],
                    observations["gripper_states"][rows],
                ),
                axis=-1,
            ).astype(np.float32)
            current_row = raw_states[-1]
            previous_row = raw_states[0]

            start = planning_time + self.action_offset
            chunk = demo["actions"][start : start + self.chunk_size].astype(np.float32)
            valid = np.arange(self.chunk_size) < len(chunk)
            if not valid.any():
                raise RuntimeError("planning time produced an empty action target")
            if len(chunk) < self.chunk_size:
                chunk = np.concatenate(
                    (chunk, np.repeat(chunk[-1:], self.chunk_size - len(chunk), axis=0))
                )

        if not np.isfinite(chunk).all() or np.abs(chunk).max(initial=0.0) > 1.0001:
            raise ValueError(
                f"raw OSC actions outside [-1,1] at {ref.path}:{ref.demo}"
            )
        normalized = normalize_state(
            np.stack((current_row, previous_row)), self.state_low, self.state_high
        )
        visual_age = float(age_steps * self.frame_interval)
        # Sample realistic planning latency L ~ U(10ms, 60ms) to train delay-aware readiness
        simulated_latency = float(
            np.random.default_rng((self.seed + 1, index)).uniform(0.01, 0.06)
        )
        query_delays = (
            visual_age
            + simulated_latency
            + np.arange(self.chunk_size, dtype=np.float32) * self.frame_interval
        )
        # Normalize state difference into physical velocity (per-second change)
        state_velocity = (
            (normalized[0] - normalized[1]) / self.frame_interval
        ).astype(np.float32)
        return {
            "stream_id": (str(ref.path), ref.demo),
            "planning_time": planning_time,
            "images": images,
            "frame_timestamps": np.asarray(frame_indices, dtype=np.float32)
            * self.frame_interval,
            "instruction": ref.instruction,
            "robot_state": normalized[0],
            "state_velocity": state_velocity,
            "visual_age": np.float32(visual_age),
            "query_delays": query_delays.astype(np.float32),
            "actions": chunk,
            "action_valid_mask": valid,
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
        return {
            "moss_inputs": moss_inputs,
            "robot_state": torch.from_numpy(
                np.stack([row["robot_state"] for row in rows])
            ),
            "state_velocity": torch.from_numpy(
                np.stack([row["state_velocity"] for row in rows])
            ),
            "visual_age": torch.from_numpy(
                np.asarray([row["visual_age"] for row in rows], dtype=np.float32)
            ),
            "query_delays": torch.from_numpy(
                np.stack([row["query_delays"] for row in rows])
            ),
            "actions": torch.from_numpy(np.stack([row["actions"] for row in rows])),
            "action_valid_mask": torch.from_numpy(
                np.stack([row["action_valid_mask"] for row in rows])
            ),
        }
