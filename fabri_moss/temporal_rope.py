"""Temporal Rotary Position Embedding (RoPE) for compact memory.

Superimposes physical time phase onto rotary position embeddings
without modifying model weights or attention implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TemporalRoPEConfig:
    strength: float = 1.0
    time_unit_seconds: float = 1.0
    rotary_fraction: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise TypeError(f"strength must be a float or int (not bool), got {type(self.strength).__name__}")
        strength_f = float(self.strength)
        if not math.isfinite(strength_f) or strength_f < 0.0:
            raise ValueError(f"strength must be a finite non-negative float, got {self.strength}")

        if isinstance(self.time_unit_seconds, bool) or not isinstance(self.time_unit_seconds, (int, float)):
            raise TypeError(
                f"time_unit_seconds must be a float or int (not bool), got {type(self.time_unit_seconds).__name__}"
            )
        time_unit_f = float(self.time_unit_seconds)
        if not math.isfinite(time_unit_f) or time_unit_f <= 0.0:
            raise ValueError(f"time_unit_seconds must be a finite positive float, got {self.time_unit_seconds}")

        if isinstance(self.rotary_fraction, bool) or not isinstance(self.rotary_fraction, (int, float)):
            raise TypeError(
                f"rotary_fraction must be a float or int (not bool), got {type(self.rotary_fraction).__name__}"
            )
        fraction_f = float(self.rotary_fraction)
        if not math.isfinite(fraction_f) or not (0.0 < fraction_f <= 1.0):
            raise ValueError(f"rotary_fraction must be in (0, 1], got {self.rotary_fraction}")


def temporal_position_embeddings(
    core: nn.Module,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    token_times: torch.Tensor,
    config: TemporalRoPEConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute combined spatial + temporal rotary position embeddings.

    Superimposes physical time phase onto rotary position embeddings
    using the core module's rotary_emb.

    Args:
        core: Module with rotary_emb attribute
        inputs_embeds: [1, seq_len, hidden_dim]
        position_ids: [1, seq_len]
        token_times: [1, seq_len] tensor with physical seconds
        config: TemporalRoPEConfig

    Returns:
        (cos, sin) tuple with same layout and dtype as core.rotary_emb(inputs_embeds, position_ids)
    """
    if not isinstance(config, TemporalRoPEConfig):
        raise TypeError(f"config must be a TemporalRoPEConfig, got {type(config).__name__}")

    rotary_emb = getattr(core, "rotary_emb", None)
    if rotary_emb is None:
        raise AttributeError("core module must have a 'rotary_emb' attribute")

    # Validate inputs_embeds
    if not isinstance(inputs_embeds, torch.Tensor):
        raise TypeError(f"inputs_embeds must be a torch.Tensor, got {type(inputs_embeds).__name__}")
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError(f"inputs_embeds must have shape [1, seq_len, hidden_dim], got {list(inputs_embeds.shape)}")

    seq_len = inputs_embeds.shape[1]
    device = inputs_embeds.device

    # Validate position_ids
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError(f"position_ids must be a torch.Tensor, got {type(position_ids).__name__}")
    if position_ids.ndim != 2 or position_ids.shape != (1, seq_len):
        raise ValueError(f"position_ids must have shape [1, {seq_len}], got {list(position_ids.shape)}")

    # Validate token_times
    if not isinstance(token_times, torch.Tensor):
        raise TypeError(f"token_times must be a torch.Tensor, got {type(token_times).__name__}")
    if token_times.ndim != 2 or token_times.shape != (1, seq_len):
        raise ValueError(f"token_times must have shape [1, {seq_len}], got {list(token_times.shape)}")
    if token_times.device != device:
        raise ValueError(f"token_times device {token_times.device} does not match inputs_embeds device {device}")
    if not torch.is_floating_point(token_times):
        raise TypeError(f"token_times must be floating point dtype, got {token_times.dtype}")
    if not torch.isfinite(token_times).all():
        raise ValueError("token_times must contain only finite numbers")
    if (token_times < 0.0).any():
        raise ValueError("token_times must be non-negative")

    # Base rotary embeddings from core
    base_cos, base_sin = rotary_emb(inputs_embeds, position_ids)

    # If strength == 0.0, return original tuple directly for exact numerical preservation
    if float(config.strength) == 0.0:
        return base_cos, base_sin

    # Check inv_freq
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if inv_freq is None:
        raise AttributeError("core.rotary_emb does not have 'inv_freq' for temporal RoPE")

    # num_pairs is the number of rotary frequencies: inv_freq.shape[0]
    num_pairs = inv_freq.shape[0]
    k_pairs = max(1, math.floor(num_pairs * float(config.rotary_fraction)))

    # Compute time phase in FP32
    # token_times: [1, seq_len] -> [1, seq_len, 1]
    # inv_freq: [num_pairs]
    # active frequencies: inv_freq[:k_pairs]
    times_fp32 = (token_times.to(torch.float32) / float(config.time_unit_seconds)) * float(config.strength)  # [1, seq_len]
    inv_freq_fp32 = inv_freq.to(dtype=torch.float32, device=device)

    # Active time phases: [1, seq_len, k_pairs]
    active_freqs = inv_freq_fp32[:k_pairs]  # [k_pairs]
    active_phase = times_fp32.unsqueeze(-1) * active_freqs.unsqueeze(0).unsqueeze(0)  # [1, seq_len, k_pairs]

    # Full time phase: [1, seq_len, num_pairs], with zeros for inactive frequencies
    if k_pairs < num_pairs:
        zeros_phase = torch.zeros(
            (1, seq_len, num_pairs - k_pairs),
            dtype=torch.float32,
            device=device,
        )
        time_phase_half = torch.cat([active_phase, zeros_phase], dim=-1)  # [1, seq_len, num_pairs]
    else:
        time_phase_half = active_phase

    # Match Qwen rotate_half layout: emb = cat((freqs, freqs), dim=-1)
    # The half-dimension layout pairs first half with second half:
    # x1 = x[..., : d // 2], x2 = x[..., d // 2 :]
    # So duplicate time_phase_half across the dimension:
    time_phase = torch.cat([time_phase_half, time_phase_half], dim=-1)  # [1, seq_len, 2 * num_pairs]

    # Base cos/sin shape is [1, seq_len, head_dim] or [1, seq_len, 2 * num_pairs]
    # Using trigonometric angle addition formula:
    # cos(A + B) = cos(A) cos(B) - sin(A) sin(B)
    # sin(A + B) = sin(A) cos(B) + cos(A) sin(B)
    # Accounting for attention_scaling: base_cos = cos(A) * scaling, base_sin = sin(A) * scaling
    # So:
    # cos(A + B) * scaling = (base_cos * cos(B) - base_sin * sin(B))
    # sin(A + B) * scaling = (base_sin * cos(B) + base_cos * sin(B))
    # We perform all calculations in FP32
    cos_t = time_phase.cos()
    sin_t = time_phase.sin()

    base_cos_fp32 = base_cos.to(torch.float32)
    base_sin_fp32 = base_sin.to(torch.float32)

    new_cos_fp32 = base_cos_fp32 * cos_t - base_sin_fp32 * sin_t
    new_sin_fp32 = base_sin_fp32 * cos_t + base_cos_fp32 * sin_t

    return new_cos_fp32.to(dtype=base_cos.dtype), new_sin_fp32.to(dtype=base_sin.dtype)
