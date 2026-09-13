"""P2 Sentinel: Shallow residual dynamics predictor and latent drift monitor."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel_data import TransitionContractMetadata

__all__ = [
    "FixedFeatureNormalizer",
    "ShallowDynamicsPredictor",
    "LatentDriftMonitor",
    "SentinelCalibrationProfile",
    "DtBinThreshold",
    "huber_residual",
]


def huber_residual(
    input: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Huber residual between prediction and target."""
    return F.huber_loss(input, target, delta=delta, reduction=reduction)


class FixedFeatureNormalizer(nn.Module):
    """Fixed per-feature normalization using registered buffers (no learnable params)."""

    def __init__(
        self,
        feature_dim: int,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        if mean is None:
            mean = torch.zeros(feature_dim, dtype=torch.float32)
        if std is None:
            std = torch.ones(feature_dim, dtype=torch.float32)

        std = torch.clamp(std, min=1e-5)
        self.register_buffer("mean", mean.view(1, 1, feature_dim))
        self.register_buffer("std", std.view(1, 1, feature_dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class ShallowDynamicsPredictor(nn.Module):
    """Predicts residual delta per spatial region: delta_z = f(z_start, coords, state_cond)."""

    def __init__(
        self,
        num_regions: int,
        feature_dim: int,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        grid_h: int | None = None,
        grid_w: int | None = None,
        normalizer: FixedFeatureNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.num_regions = num_regions
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.normalizer = (
            normalizer if normalizer is not None else FixedFeatureNormalizer(feature_dim)
        )

        if grid_h is None or grid_w is None:
            side = int(math.isqrt(num_regions))
            if side * side == num_regions:
                grid_h, grid_w = side, side
            else:
                grid_h, grid_w = 1, num_regions
        if grid_h * grid_w != num_regions:
            raise ValueError(f"grid_h * grid_w ({grid_h}*{grid_w}) != num_regions ({num_regions})")

        self.grid_h = grid_h
        self.grid_w = grid_w

        ys = torch.linspace(-1.0, 1.0, grid_h, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, grid_w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
        self.register_buffer("region_coords", coords)

        self.condition_dim = 2 * state_dim + 2 * action_dim + 1 + context_dim + 1

        in_dim = feature_dim + 2 + self.condition_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        start_z: torch.Tensor,
        start_state: torch.Tensor,
        start_velocity: torch.Tensor,
        action_mean: torch.Tensor,
        action_last: torch.Tensor,
        dt: torch.Tensor,
        task_context: torch.Tensor,
        context_age: torch.Tensor,
    ) -> torch.Tensor:
        if start_z.ndim == 2:
            start_z = start_z.unsqueeze(0)
            single_batch = True
        else:
            single_batch = False

        B, R, D = start_z.shape
        if R != self.num_regions or D != self.feature_dim:
            raise ValueError(
                f"Expected start_z shape [B, {self.num_regions}, {self.feature_dim}], got {list(start_z.shape)}"
            )

        start_state = start_state.view(B, -1)
        start_velocity = start_velocity.view(B, -1)
        action_mean = action_mean.view(B, -1)
        action_last = action_last.view(B, -1)
        dt = dt.view(B, 1)
        task_context = task_context.view(B, -1)
        context_age = context_age.view(B, 1)

        cond = torch.cat(
            [
                start_state,
                start_velocity,
                action_mean,
                action_last,
                dt,
                task_context,
                context_age,
            ],
            dim=-1,
        )

        z_norm = self.normalizer.normalize(start_z)
        coords_expanded = self.region_coords.unsqueeze(0).expand(B, R, 2)
        cond_expanded = cond.unsqueeze(1).expand(B, R, self.condition_dim)

        x = torch.cat([z_norm, coords_expanded, cond_expanded], dim=-1)
        delta_norm = self.mlp(x)
        delta = delta_norm * self.normalizer.std

        if single_batch:
            delta = delta.squeeze(0)
        return delta

    def predict_target(
        self,
        start_z: torch.Tensor,
        start_state: torch.Tensor,
        start_velocity: torch.Tensor,
        action_mean: torch.Tensor,
        action_last: torch.Tensor,
        dt: torch.Tensor,
        task_context: torch.Tensor,
        context_age: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.forward(
            start_z=start_z,
            start_state=start_state,
            start_velocity=start_velocity,
            action_mean=action_mean,
            action_last=action_last,
            dt=dt,
            task_context=task_context,
            context_age=context_age,
        )
        return start_z + delta


@dataclass(frozen=True)
class DtBinThreshold:
    dt_min: float
    dt_max: float
    region_quantile_threshold: float
    global_threshold: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.dt_min) and math.isfinite(self.dt_max)):
            raise ValueError("dt bounds must be finite")
        if self.dt_min <= 0 or self.dt_max <= self.dt_min:
            raise ValueError(f"Invalid dt bounds [{self.dt_min}, {self.dt_max}]")
        if any(not math.isfinite(t) or t < 0 for t in
               (self.region_quantile_threshold, self.global_threshold)):
            raise ValueError("Thresholds must be finite and nonnegative")

    def matches(self, dt: float, is_last_bin: bool = False) -> bool:
        # Half-open [dt_min, dt_max) except optionally closed on the last bin's upper edge
        if is_last_bin:
            return self.dt_min <= dt <= self.dt_max
        return self.dt_min <= dt < self.dt_max


@dataclass(frozen=True)
class SentinelCalibrationProfile:
    """Fixed calibration profile for LatentDriftMonitor across dt bins."""

    bins: tuple[DtBinThreshold, ...]
    persistence_seconds: float
    quantile: float
    quantile_weight: float
    metadata: TransitionContractMetadata
    predictor_state_hash: str
    calibration_episodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bins:
            raise ValueError("Profile must have at least one dt bin")
        if not math.isfinite(self.persistence_seconds) or self.persistence_seconds < 0:
            raise ValueError("persistence_seconds must be finite nonnegative")
        if not (0.0 < self.quantile < 1.0):
            raise ValueError("quantile must be between 0 and 1")
        if not (0.0 <= self.quantile_weight <= 1.0):
            raise ValueError("quantile_weight must be in [0, 1]")
        if not self.predictor_state_hash:
            raise ValueError("predictor_state_hash must be non-empty")
        if not self.calibration_episodes or isinstance(self.calibration_episodes, str) or not isinstance(self.calibration_episodes, (list, tuple)):
            raise ValueError("Profile must have at least one calibration episode in a list/tuple")
        if any(not isinstance(ep, str) or not ep for ep in self.calibration_episodes):
            raise ValueError("All calibration_episodes must be non-empty strings")
        if list(self.calibration_episodes) != sorted(set(self.calibration_episodes)):
            raise ValueError("calibration_episodes must be sorted unique episode IDs")

        # Validate bins are sorted half-open intervals [min, max), allowing adjacent endpoints
        for i in range(len(self.bins) - 1):
            curr_b = self.bins[i]
            next_b = self.bins[i + 1]
            if curr_b.dt_max > next_b.dt_min + 1e-7:
                raise ValueError(
                    f"Bins overlap: bin {i} max ({curr_b.dt_max}) > bin {i+1} min ({next_b.dt_min})"
                )
            if curr_b.dt_min >= next_b.dt_min:
                raise ValueError(
                    f"Bins not strictly sorted: bin {i} min ({curr_b.dt_min}) >= bin {i+1} min ({next_b.dt_min})"
                )

    def get_thresholds(self, dt: float) -> DtBinThreshold:
        for i, b in enumerate(self.bins):
            is_last = (i == len(self.bins) - 1)
            if b.matches(dt, is_last_bin=is_last):
                return b
        raise ValueError(f"dt {dt:.4f} does not match any calibrated dt bin in profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": [asdict(b) for b in self.bins],
            "persistence_seconds": self.persistence_seconds,
            "quantile": self.quantile,
            "quantile_weight": self.quantile_weight,
            "metadata": self.metadata.to_dict(),
            "predictor_state_hash": self.predictor_state_hash,
            "calibration_episodes": list(self.calibration_episodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentinelCalibrationProfile:
        bins = tuple(DtBinThreshold(**b) for b in data["bins"])
        pred_hash = data.get("predictor_state_hash")
        if not pred_hash:
            raise ValueError("SentinelCalibrationProfile requires predictor_state_hash")
        cal_eps_raw = data.get("calibration_episodes")
        if not cal_eps_raw or isinstance(cal_eps_raw, str) or not isinstance(cal_eps_raw, (list, tuple)):
            raise ValueError("SentinelCalibrationProfile requires non-empty list/tuple of calibration_episodes")
        if any(not isinstance(ep, str) or not ep for ep in cal_eps_raw):
            raise ValueError("All calibration_episodes must be non-empty strings")
        calibration_episodes = tuple(sorted(set(cal_eps_raw)))
        return cls(
            bins=bins,
            persistence_seconds=float(data["persistence_seconds"]),
            quantile=float(data["quantile"]),
            quantile_weight=float(data.get("quantile_weight", 0.5)),
            metadata=TransitionContractMetadata.from_dict(data["metadata"]),
            predictor_state_hash=str(pred_hash),
            calibration_episodes=calibration_episodes,
        )


@dataclass
class AnchorPersistenceTracker:
    span_name: str
    persistence_seconds: float
    drift_start_time: float | None = None
    drift_duration: float = 0.0
    last_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.persistence_seconds) or self.persistence_seconds < 0:
            raise ValueError("anchor persistence must be finite and nonnegative")

    def update(self, is_drift: bool, timestamp: float) -> bool:
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("anchor timestamps must strictly increase")
        self.last_timestamp = timestamp
        if not is_drift:
            self.drift_start_time = None
            self.drift_duration = 0.0
            return False

        if self.drift_start_time is None:
            self.drift_start_time = timestamp
        self.drift_duration = timestamp - self.drift_start_time
        return self.drift_duration >= self.persistence_seconds

    def reset(self) -> None:
        self.drift_start_time = None
        self.drift_duration = 0.0
        self.last_timestamp = None


class LatentDriftMonitor:
    def __init__(
        self,
        predictor: ShallowDynamicsPredictor,
        profile: SentinelCalibrationProfile,
        anchor_persistence_seconds: dict[str, float] | None = None,
    ) -> None:
        self.predictor = predictor
        self.profile = profile

        self._anchors: dict[str, AnchorPersistenceTracker] = {
            "default": AnchorPersistenceTracker("default", profile.persistence_seconds),
        }
        if anchor_persistence_seconds:
            for name, dur in anchor_persistence_seconds.items():
                self._anchors[name] = AnchorPersistenceTracker(name, float(dur))

    def reset_monitor(self) -> None:
        for tracker in self._anchors.values():
            tracker.reset()

    def compute_residual_scores(
        self,
        pred_z: torch.Tensor,
        actual_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        elem_loss = huber_residual(pred_z, actual_z, delta=1.0, reduction="none")
        region_residuals = elem_loss.mean(dim=-1)
        q = self.profile.quantile
        region_quantile_score = torch.quantile(region_residuals, q=q, dim=-1)
        global_residual = region_residuals.mean(dim=-1)
        return region_residuals, region_quantile_score, global_residual

    def evaluate_step(
        self,
        start_z: torch.Tensor,
        target_z: torch.Tensor,
        start_state: torch.Tensor,
        start_velocity: torch.Tensor,
        action_mean: torch.Tensor,
        action_last: torch.Tensor,
        dt: float,
        task_context: torch.Tensor,
        context_age: float,
        timestamp: float,
        anchor: str = "default",
    ) -> dict[str, Any]:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(context_age) or context_age < 0:
            raise ValueError("dt must be positive and context_age nonnegative, both finite")
        if any(not bool(torch.isfinite(t).all()) for t in
               (start_z, target_z, start_state, start_velocity, action_mean, action_last, task_context)):
            raise ValueError("monitor inputs must be finite")

        self.predictor.eval()
        with torch.no_grad():
            dt_tensor = torch.tensor([[dt]], dtype=torch.float32, device=start_z.device)
            ctx_age_tensor = torch.tensor([[context_age]], dtype=torch.float32, device=start_z.device)

            pred_z = self.predictor.predict_target(
                start_z=start_z,
                start_state=start_state,
                start_velocity=start_velocity,
                action_mean=action_mean,
                action_last=action_last,
                dt=dt_tensor,
                task_context=task_context,
                context_age=ctx_age_tensor,
            )
            region_res, q_score, g_res = self.compute_residual_scores(pred_z, target_z)

        q_val = float(q_score.item() if q_score.numel() == 1 else q_score.mean().item())
        g_val = float(g_res.item() if g_res.numel() == 1 else g_res.mean().item())

        if not math.isfinite(q_val) or not math.isfinite(g_val):
            raise ValueError("non-finite predictor residual")
        thresholds = self.profile.get_thresholds(dt)
        q_thresh = thresholds.region_quantile_threshold
        g_thresh = thresholds.global_threshold

        w = self.profile.quantile_weight
        composite_score = w * q_val + (1.0 - w) * g_val
        composite_threshold = w * q_thresh + (1.0 - w) * g_thresh

        is_step_drift = (q_val > q_thresh) or (g_val > g_thresh)

        if anchor not in self._anchors:
            self._anchors[anchor] = AnchorPersistenceTracker(anchor, self.profile.persistence_seconds)

        persistent_alert = self._anchors[anchor].update(is_step_drift, timestamp)
        drift_duration = self._anchors[anchor].drift_duration

        return {
            "is_step_drift": is_step_drift,
            "persistent_alert": persistent_alert,
            "drift_duration": drift_duration,
            "region_quantile_score": q_val,
            "region_quantile_threshold": q_thresh,
            "global_residual": g_val,
            "global_threshold": g_thresh,
            "composite_score": composite_score,
            "composite_threshold": composite_threshold,
            "anchor": anchor,
            "dt": dt,
        }
