"""Dataset and schema definitions for sentinel dynamics prediction and drift monitoring."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

__all__ = [
    "FeatureSpec",
    "TimingContract",
    "ContextSpec",
    "TransitionContractMetadata",
    "SentinelTransition",
    "SentinelTransitionDataset",
    "extract_transitions_from_recordings",
    "verify_disjoint_episodes",
    "collate_sentinel_batch",
]


@dataclass(frozen=True)
class FeatureSpec:
    num_regions: int
    feature_dim: int
    grid_h: int | None = None
    grid_w: int | None = None

    def __post_init__(self) -> None:
        if self.num_regions <= 0 or self.feature_dim <= 0:
            raise ValueError("num_regions and feature_dim must be positive")
        if (self.grid_h is None) != (self.grid_w is None):
            raise ValueError("grid_h and grid_w must be specified together")
        if self.grid_h is not None and self.grid_w is not None:
            if self.grid_h <= 0 or self.grid_w <= 0:
                raise ValueError("spatial grid dimensions must be positive")
            if self.grid_h * self.grid_w != self.num_regions:
                raise ValueError(
                    f"grid_h * grid_w ({self.grid_h} * {self.grid_w}) != num_regions ({self.num_regions})"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingContract:
    frame_interval: float
    action_interval: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.frame_interval) or self.frame_interval <= 0:
            raise ValueError("frame_interval must be finite positive")
        if not math.isfinite(self.action_interval) or self.action_interval <= 0:
            raise ValueError("action_interval must be finite positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextSpec:
    context_dim: int
    context_method: str = "held_accepted_prefix"

    def __post_init__(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not self.context_method:
            raise ValueError("context_method must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionContractMetadata:
    policy_hash: str
    vision_source_fingerprint: str
    preprocessing_fingerprint: str
    shallow_weights_fingerprint: str
    split_depth: str
    feature_spec: FeatureSpec
    timing_contract: TimingContract
    context_spec: ContextSpec

    def __post_init__(self) -> None:
        if not self.policy_hash:
            raise ValueError("policy_hash must be non-empty")
        if not self.vision_source_fingerprint:
            raise ValueError("vision_source_fingerprint must be non-empty")
        if not self.preprocessing_fingerprint:
            raise ValueError("preprocessing_fingerprint must be non-empty")
        if not self.shallow_weights_fingerprint:
            raise ValueError("shallow_weights_fingerprint must be non-empty")
        if not self.split_depth:
            raise ValueError("split_depth must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "vision_source_fingerprint": self.vision_source_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "shallow_weights_fingerprint": self.shallow_weights_fingerprint,
            "split_depth": self.split_depth,
            "feature_spec": self.feature_spec.to_dict(),
            "timing_contract": self.timing_contract.to_dict(),
            "context_spec": self.context_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionContractMetadata:
        shallow_fp = data.get("shallow_weights_fingerprint")
        if not shallow_fp:
            raise ValueError("TransitionContractMetadata requires non-empty shallow_weights_fingerprint")
        return cls(
            policy_hash=str(data["policy_hash"]),
            vision_source_fingerprint=str(data["vision_source_fingerprint"]),
            preprocessing_fingerprint=str(data["preprocessing_fingerprint"]),
            shallow_weights_fingerprint=str(shallow_fp),
            split_depth=str(data["split_depth"]),
            feature_spec=FeatureSpec(**data["feature_spec"]),
            timing_contract=TimingContract(**data["timing_contract"]),
            context_spec=ContextSpec(**data["context_spec"]),
        )


@dataclass(frozen=True)
class SentinelTransition:
    start_z: torch.Tensor
    target_z: torch.Tensor
    start_state: torch.Tensor
    start_velocity: torch.Tensor
    action_mean: torch.Tensor
    action_last: torch.Tensor
    dt: torch.Tensor
    task_context: torch.Tensor
    context_age: torch.Tensor
    start_time: float
    end_time: float
    episode_id: str
    step_span: int


class SentinelTransitionDataset(Dataset):
    def __init__(
        self,
        transitions: list[SentinelTransition],
        metadata: TransitionContractMetadata,
    ) -> None:
        if not transitions:
            raise ValueError("Transitions list cannot be empty")
        self.transitions = transitions
        self.metadata = metadata
        self._validate_transitions()

    def _validate_transitions(self) -> None:
        num_regions = self.metadata.feature_spec.num_regions
        feature_dim = self.metadata.feature_spec.feature_dim
        context_dim = self.metadata.context_spec.context_dim

        for i, t in enumerate(self.transitions):
            if any(not bool(torch.isfinite(value).all()) for value in
                   (t.start_z, t.target_z, t.start_state, t.start_velocity,
                    t.action_mean, t.action_last, t.task_context)):
                raise ValueError(f"Transition {i} contains non-finite data")
            if not t.episode_id or t.step_span < 1:
                raise ValueError("transition needs an episode ID and positive step span")
            if not math.isclose(float(t.dt), t.end_time - t.start_time, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("transition dt disagrees with observation timestamps")
            if t.start_z.shape != (num_regions, feature_dim):
                raise ValueError(
                    f"Transition {i} start_z shape {t.start_z.shape} != expected ({num_regions}, {feature_dim})"
                )
            if t.target_z.shape != (num_regions, feature_dim):
                raise ValueError(
                    f"Transition {i} target_z shape {t.target_z.shape} != expected ({num_regions}, {feature_dim})"
                )
            if t.task_context.shape[-1] != context_dim:
                raise ValueError(
                    f"Transition {i} task_context dim {t.task_context.shape[-1]} != expected {context_dim}"
                )
            dt_val = float(t.dt.item() if isinstance(t.dt, torch.Tensor) else t.dt)
            if not math.isfinite(dt_val) or dt_val <= 0:
                raise ValueError(f"Transition {i} dt must be finite positive, got {dt_val}")
            if not math.isfinite(t.start_time) or not math.isfinite(t.end_time):
                raise ValueError(f"Transition {i} start_time and end_time must be finite")
            if t.end_time <= t.start_time:
                raise ValueError(f"Transition {i} end_time {t.end_time} must be > start_time {t.start_time}")
            ctx_age = float(t.context_age.item() if isinstance(t.context_age, torch.Tensor) else t.context_age)
            if not math.isfinite(ctx_age) or ctx_age < 0:
                raise ValueError(
                    f"Transition {i} context_age must be finite >= 0 (no future context), got {ctx_age}"
                )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.transitions[idx]
        return {
            "start_z": t.start_z,
            "target_z": t.target_z,
            "start_state": t.start_state,
            "start_velocity": t.start_velocity,
            "action_mean": t.action_mean,
            "action_last": t.action_last,
            "dt": t.dt if isinstance(t.dt, torch.Tensor) else torch.tensor(t.dt, dtype=torch.float32),
            "task_context": t.task_context,
            "context_age": (
                t.context_age
                if isinstance(t.context_age, torch.Tensor)
                else torch.tensor(t.context_age, dtype=torch.float32)
            ),
            "start_time": t.start_time,
            "end_time": t.end_time,
            "episode_id": t.episode_id,
            "step_span": t.step_span,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self.metadata.to_dict(),
                "transitions": self.transitions,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SentinelTransitionDataset:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "metadata" not in data or "transitions" not in data:
            raise ValueError(f"Invalid dataset file format at {path}")
        metadata = TransitionContractMetadata.from_dict(data["metadata"])
        transitions = data["transitions"]
        return cls(transitions=transitions, metadata=metadata)


def collate_sentinel_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start_z": torch.stack([b["start_z"] for b in batch]),
        "target_z": torch.stack([b["target_z"] for b in batch]),
        "start_state": torch.stack([b["start_state"] for b in batch]),
        "start_velocity": torch.stack([b["start_velocity"] for b in batch]),
        "action_mean": torch.stack([b["action_mean"] for b in batch]),
        "action_last": torch.stack([b["action_last"] for b in batch]),
        "dt": torch.stack([b["dt"] for b in batch]).view(-1, 1),
        "task_context": torch.stack([b["task_context"] for b in batch]),
        "context_age": torch.stack([b["context_age"] for b in batch]).view(-1, 1),
        "start_time": [b["start_time"] for b in batch],
        "end_time": [b["end_time"] for b in batch],
        "episode_id": [b["episode_id"] for b in batch],
        "step_span": torch.tensor([b["step_span"] for b in batch], dtype=torch.long),
    }


def extract_transitions_from_recordings(
    recordings: list[dict[str, Any]],
    metadata: TransitionContractMetadata,
    spans: Sequence[int] = (1, 2, 4),
    time_tolerance: float = 1e-4,
) -> list[SentinelTransition]:
    """Extract transitions from offline recordings with exact duration-weighted action summaries."""
    transitions: list[SentinelTransition] = []
    if not math.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("time_tolerance must be finite and nonnegative")
    if not spans or any(not isinstance(span, int) or span < 1 for span in spans):
        raise ValueError("spans must be positive integer control-step counts")
    if len(set(spans)) != len(spans):
        raise ValueError("duplicate spans would duplicate shadow timestamps")

    for rec in recordings:
        ep_id = str(rec["episode_id"])
        timestamps = torch.as_tensor(rec["timestamps"], dtype=torch.float64)
        if timestamps.ndim != 1 or not torch.all(torch.isfinite(timestamps)):
            raise ValueError(f"Recording {ep_id} timestamps must be 1D finite tensor")
        T = len(timestamps)
        if T < 2:
            continue

        diffs = timestamps[1:] - timestamps[:-1]
        if torch.any(diffs <= 0):
            raise ValueError(f"Recording {ep_id} timestamps must be strictly increasing")

        features = torch.as_tensor(rec["frozen_features"], dtype=torch.float32)
        if (
            features.ndim != 3
            or features.shape[0] != T
            or features.shape[1] != metadata.feature_spec.num_regions
            or features.shape[2] != metadata.feature_spec.feature_dim
            or not torch.all(torch.isfinite(features))
        ):
            raise ValueError(
                f"Recording {ep_id} frozen_features shape must be [T={T}, regions={metadata.feature_spec.num_regions}, dim={metadata.feature_spec.feature_dim}] and finite"
            )

        states = torch.as_tensor(rec["states"], dtype=torch.float32)
        if states.ndim != 2 or states.shape[0] != T or states.shape[1] <= 0 or not torch.all(torch.isfinite(states)):
            raise ValueError(f"Recording {ep_id} states shape must be [T={T}, state_dim > 0] and finite")

        velocities = torch.as_tensor(rec["velocities"], dtype=torch.float32)
        if velocities.shape != states.shape or not torch.all(torch.isfinite(velocities)):
            raise ValueError(f"Recording {ep_id} velocities shape must match states shape {states.shape} and be finite")

        actions = torch.as_tensor(rec["actions"], dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] <= 0 or not torch.all(torch.isfinite(actions)):
            raise ValueError(f"Recording {ep_id} actions must be 2D finite tensor with positive action_dim")
        A = actions.shape[0]

        if "action_end_timestamps" not in rec:
            raise KeyError(
                f"Recording {ep_id} missing required 'action_end_timestamps'."
            )
        action_start_ts = torch.as_tensor(rec["action_timestamps"], dtype=torch.float64)
        action_end_ts = torch.as_tensor(rec["action_end_timestamps"], dtype=torch.float64)

        if (
            action_start_ts.ndim != 1
            or action_end_ts.ndim != 1
            or len(action_start_ts) != A
            or len(action_end_ts) != A
            or not torch.all(torch.isfinite(action_start_ts))
            or not torch.all(torch.isfinite(action_end_ts))
        ):
            raise ValueError(
                f"Recording {ep_id} action timestamp intervals must have shape [{A}] matching actions and be finite"
            )

        act_durations = action_end_ts - action_start_ts
        if torch.any(act_durations <= 0):
            raise ValueError(f"Recording {ep_id} action intervals must have positive duration")

        if A > 1:
            start_diffs = action_start_ts[1:] - action_start_ts[:-1]
            if torch.any(start_diffs <= 0):
                raise ValueError(f"Recording {ep_id} action_timestamps must be strictly increasing")
            overlaps = action_start_ts[1:] - action_end_ts[:-1]
            if torch.any(overlaps < -time_tolerance):
                raise ValueError(f"Recording {ep_id} has overlapping action command executions")

        context_raw = torch.as_tensor(rec["context"], dtype=torch.float32).clone()
        context_ts_raw = torch.as_tensor(rec["context_timestamps"], dtype=torch.float64).clone()

        if not torch.all(torch.isfinite(context_raw)):
            raise ValueError(f"Recording {ep_id} context values must be finite")
        if not torch.all(torch.isfinite(context_ts_raw)):
            raise ValueError(f"Recording {ep_id} context timestamps must be finite")

        C_exp = metadata.context_spec.context_dim
        # Schema: only fixed [C] + scalar/singleton time, or updates [N, C] + [N] (or [N] + [N] when C_exp==1)
        if context_raw.ndim == 1 and (context_ts_raw.ndim == 0 or context_ts_raw.numel() == 1):
            if context_raw.shape[0] != C_exp:
                raise ValueError(
                    f"Recording {ep_id} fixed context dim {context_raw.shape[0]} != metadata context_dim {C_exp}"
                )
            context_vals = context_raw.unsqueeze(0)  # [1, C_exp]
            context_ts = context_ts_raw.view(1)
        elif context_ts_raw.ndim == 1 and context_ts_raw.numel() > 0:
            N = len(context_ts_raw)
            context_ts = context_ts_raw
            if context_raw.ndim == 2:
                if context_raw.shape != (N, C_exp):
                    raise ValueError(
                        f"Recording {ep_id} updates context shape {context_raw.shape} != (N={N}, C={C_exp})"
                    )
                context_vals = context_raw
            elif context_raw.ndim == 1 and C_exp == 1 and len(context_raw) == N:
                context_vals = context_raw.unsqueeze(-1)  # [N, 1]
            else:
                raise ValueError(
                    f"Recording {ep_id} updates context shape {context_raw.shape} invalid for N={N}, context_dim={C_exp}"
                )
        else:
            raise ValueError(
                f"Recording {ep_id} context shape {context_raw.shape} and timestamps shape {context_ts_raw.shape} unsupported"
            )

        if len(context_ts) > 1:
            ts_diffs = context_ts[1:] - context_ts[:-1]
            if torch.any(ts_diffs < 0):
                raise ValueError(f"Recording {ep_id} context timestamps must be nondecreasing")
            for i in range(len(context_ts) - 1):
                if context_ts[i + 1] == context_ts[i]:
                    if not torch.equal(context_vals[i + 1], context_vals[i]):
                        raise ValueError(
                            f"Recording {ep_id} contradictory duplicate context updates at timestamp {context_ts[i].item()}"
                        )

        for span in spans:
            for t_start in range(T - span):
                t_end = t_start + span
                t0 = float(timestamps[t_start])
                t1 = float(timestamps[t_end])
                total_dt = t1 - t0
                if total_dt <= 0:
                    continue

                # Strict actual overlap endpoints: (action_end_ts > t0) & (action_start_ts < t1)
                intersect_mask = (action_end_ts > t0) & (action_start_ts < t1)
                indices = torch.nonzero(intersect_mask, as_tuple=False).squeeze(-1)

                if len(indices) == 0:
                    raise ValueError(
                        f"Recording {ep_id} span {span} [{t0:.4f}, {t1:.4f}]: no action coverage found"
                    )

                first_idx = indices[0].item()
                last_idx = indices[-1].item()
                if float(action_start_ts[first_idx]) > t0 + time_tolerance:
                    raise ValueError(
                        f"Recording {ep_id} span {span} [{t0:.4f}, {t1:.4f}]: coverage gap at start, action starts at {action_start_ts[first_idx]:.4f}"
                    )
                if float(action_end_ts[last_idx]) < t1 - time_tolerance:
                    raise ValueError(
                        f"Recording {ep_id} span {span} [{t0:.4f}, {t1:.4f}]: coverage gap at end, action ends at {action_end_ts[last_idx]:.4f}"
                    )

                for k in range(len(indices) - 1):
                    gap = float(action_start_ts[indices[k + 1]] - action_end_ts[indices[k]])
                    if gap > time_tolerance:
                        raise ValueError(
                            f"Recording {ep_id} span {span} [{t0:.4f}, {t1:.4f}]: action gap of {gap:.6f}s"
                        )

                weighted_sum = torch.zeros_like(actions[0])
                covered_duration = 0.0
                for idx_i in indices:
                    i = idx_i.item()
                    a_start = max(t0, float(action_start_ts[i]))
                    a_end = min(t1, float(action_end_ts[i]))
                    dur = a_end - a_start
                    if dur > 0:
                        weighted_sum += actions[i] * dur
                        covered_duration += dur

                # Exact contract: tolerance plus float64 summation roundoff sized to endpoints
                cov_tol = time_tolerance + len(indices) * 1e-12
                if abs(covered_duration - total_dt) > cov_tol:
                    raise ValueError(
                        f"Recording {ep_id} span {span}: covered duration {covered_duration:.6f} != interval {total_dt:.6f} (tol={cov_tol})"
                    )

                action_mean = weighted_sum / total_dt
                action_last = actions[last_idx].clone()

                idx = torch.searchsorted(
                    context_ts,
                    torch.tensor(t0, dtype=torch.float64),
                    right=True,
                ).item()
                if idx == 0:
                    raise ValueError(
                        f"Recording {ep_id} span {span} [{t0:.4f}, {t1:.4f}]: no prior context update found at or before t0={t0:.4f}"
                    )
                ctx_idx = idx - 1
                ctx = context_vals[ctx_idx].clone()
                ctx_t = float(context_ts[ctx_idx])
                context_age = max(0.0, t0 - ctx_t)

                transitions.append(
                    SentinelTransition(
                        start_z=features[t_start],
                        target_z=features[t_end],
                        start_state=states[t_start],
                        start_velocity=velocities[t_start],
                        action_mean=action_mean,
                        action_last=action_last,
                        dt=torch.tensor([total_dt], dtype=torch.float32),
                        task_context=ctx,
                        context_age=torch.tensor([context_age], dtype=torch.float32),
                        start_time=t0,
                        end_time=t1,
                        episode_id=ep_id,
                        step_span=span,
                    )
                )

    return transitions


def verify_disjoint_episodes(
    dataset_a: list[SentinelTransition] | SentinelTransitionDataset,
    dataset_b: list[SentinelTransition] | SentinelTransitionDataset,
) -> set[str]:
    eps_a = {t.episode_id if isinstance(t, SentinelTransition) else t["episode_id"] for t in dataset_a}
    eps_b = {t.episode_id if isinstance(t, SentinelTransition) else t["episode_id"] for t in dataset_b}
    overlap = eps_a.intersection(eps_b)
    if overlap:
        raise ValueError(f"Episode sets must be strictly disjoint! Overlapping episodes: {overlap}")
    return eps_a
