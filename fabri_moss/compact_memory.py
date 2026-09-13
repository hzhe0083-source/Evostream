"""Compact intermediate memory rendering and prefix management.

Preserves full visual tokens on decision frames and keyframe anchors,
while compactly pooling non-decision recent observations.
Superimposes physical token times onto rendered sequences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn

from fabri_moss.periodic_memory import (
    MemoryFrame,
    PeriodicMemoryConfig,
    PeriodicMemoryState,
    _build_header_embeds,
    _spatial_pool_visual_tokens,
    format_frame_timestamp,
)
from fabri_moss.temporal_rope import TemporalRoPEConfig


@dataclass(frozen=True)
class CompactMemoryConfig:
    memory: PeriodicMemoryConfig = field(default_factory=PeriodicMemoryConfig)
    intermediate_grid: int | None = 8
    temporal: TemporalRoPEConfig = field(default_factory=TemporalRoPEConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.memory, PeriodicMemoryConfig):
            raise TypeError(f"memory must be a PeriodicMemoryConfig, got {type(self.memory).__name__}")
        if not self.memory.protect_decision_frames:
            raise ValueError("CompactMemoryConfig requires memory.protect_decision_frames to be True")

        if self.intermediate_grid is not None:
            if type(self.intermediate_grid) is not int or self.intermediate_grid <= 0:
                raise ValueError(
                    f"intermediate_grid must be a positive int or None, got {self.intermediate_grid}"
                )

        if not isinstance(self.temporal, TemporalRoPEConfig):
            raise TypeError(f"temporal must be a TemporalRoPEConfig, got {type(self.temporal).__name__}")


@dataclass(frozen=True)
class RenderedMemory:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    token_times: torch.Tensor
    current_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.inputs_embeds, torch.Tensor):
            raise TypeError(f"inputs_embeds must be a torch.Tensor, got {type(self.inputs_embeds).__name__}")
        if not isinstance(self.attention_mask, torch.Tensor):
            raise TypeError(f"attention_mask must be a torch.Tensor, got {type(self.attention_mask).__name__}")
        if not isinstance(self.token_times, torch.Tensor):
            raise TypeError(f"token_times must be a torch.Tensor, got {type(self.token_times).__name__}")

        if self.inputs_embeds.ndim != 3 or self.inputs_embeds.shape[0] != 1:
            raise ValueError(f"inputs_embeds must have shape [1, L, H], got {list(self.inputs_embeds.shape)}")
        seq_len = self.inputs_embeds.shape[1]

        if self.attention_mask.shape != (1, seq_len):
            raise ValueError(f"attention_mask must have shape [1, {seq_len}], got {list(self.attention_mask.shape)}")
        if self.token_times.shape != (1, seq_len):
            raise ValueError(f"token_times must have shape [1, {seq_len}], got {list(self.token_times.shape)}")

        if type(self.current_start) is not int or self.current_start < 0 or self.current_start > seq_len:
            raise ValueError(f"current_start must be an int in [0, {seq_len}], got {self.current_start}")


def _resolve_embedder_core(embedder: Any) -> Tuple[nn.Module, Any, Any]:
    model = getattr(embedder, "model", None)
    if model is None:
        raise AttributeError("embedder must have a 'model' attribute")
    lm = getattr(model, "language_model", None)
    if lm is None:
        raise AttributeError("embedder.model must have a 'language_model' attribute")
    core = getattr(lm, "model", lm)
    embed_tokens = getattr(core, "embed_tokens", None)
    if embed_tokens is None or not callable(embed_tokens):
        raise AttributeError("core module must have an 'embed_tokens' callable")
    tokenizer = getattr(embedder, "tokenizer", None)
    return core, embed_tokens, tokenizer


def _render_single_frame(
    rf: MemoryFrame,
    tokenizer: Any,
    embed_tokens: Any,
    core_device: torch.device,
    target_device: torch.device,
    target_dtype: torch.dtype,
    intermediate_grid: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render a single recent frame (either compact non-decision or full decision)."""
    p = rf.visual_tokens.shape[1]
    side = int(math.isqrt(p))
    if intermediate_grid is not None and intermediate_grid > side:
        raise ValueError(
            f"intermediate_grid {intermediate_grid} exceeds visual_tokens spatial side {side} (upsampling forbidden)"
        )

    if rf.is_decision or intermediate_grid is None:
        # Full inputs_embeds and attention_mask
        f_embeds = rf.inputs_embeds.to(dtype=target_dtype, device=target_device)
        f_mask = rf.attention_mask.to(device=target_device)
        f_times = torch.full(
            (1, f_embeds.shape[1]),
            float(rf.observation_time),
            dtype=torch.float32,
            device=target_device,
        )
        return f_embeds, f_mask, f_times
    else:
        # Compact non-decision frame: header + pooled visual tokens, mask all 1s, no padding
        if tokenizer is None:
            raise AttributeError("embedder must have a 'tokenizer' attribute to render compact headers")
        header_text = format_frame_timestamp(rf.frame_id, rf.observation_time)
        h_embeds, h_mask = _build_header_embeds(
            header_text, tokenizer, embed_tokens, core_device, target_device, target_dtype
        )
        pooled_vis = _spatial_pool_visual_tokens(rf.visual_tokens, intermediate_grid)
        p_vis = pooled_vis.to(dtype=target_dtype, device=target_device)
        p_vis_mask = torch.ones((1, p_vis.shape[1]), dtype=torch.long, device=target_device)

        f_embeds = torch.cat([h_embeds, p_vis], dim=1)
        f_mask = torch.cat([h_mask, p_vis_mask], dim=1)
        f_times = torch.full(
            (1, f_embeds.shape[1]),
            float(rf.observation_time),
            dtype=torch.float32,
            device=target_device,
        )
        return f_embeds, f_mask, f_times


def render_compact_memory(
    state: PeriodicMemoryState,
    embedder: Any,
    config: CompactMemoryConfig,
) -> RenderedMemory:
    """Render full compact sequence embeddings, attention mask, and token times from memory state.

    Prefix items (anchors and entries) are sorted chronologically by original starting frame_id:
    - Anchors: header + full visual tokens (time = observation_time)
    - Entries: summary header + 16 tokens (time = end_time, summary text includes range and count)
    Followed by recent frames:
    - Non-decision frames: compact header + GxG pooled visual (no padding, mask=1, time = observation_time)
      (or full if intermediate_grid is None)
    - Decision frames: full inputs_embeds and attention_mask (time = observation_time across all tokens including padding)

    current_start points to the start of the final recent frame (which is a decision frame if valid).
    """
    if not isinstance(config, CompactMemoryConfig):
        raise TypeError(f"config must be a CompactMemoryConfig, got {type(config).__name__}")
    if not state.recent and not state.entries and not state.anchors:
        raise ValueError("Cannot render memory from empty state: recent, entries, and anchors are all empty.")

    core, embed_tokens, tokenizer = _resolve_embedder_core(embedder)
    core_device = embed_tokens.weight.device

    if state.recent:
        target_dtype = state.recent[0].inputs_embeds.dtype
        target_device = state.recent[0].inputs_embeds.device
    elif state.anchors:
        target_dtype = state.anchors[0].inputs_embeds.dtype
        target_device = state.anchors[0].inputs_embeds.device
    else:
        target_dtype = embed_tokens.weight.dtype
        target_device = core_device

    if (state.entries or state.anchors or state.recent) and tokenizer is None:
        raise AttributeError("embedder must have a 'tokenizer' attribute")

    g2 = config.memory.spatial_grid * config.memory.spatial_grid

    # Collect and sort prefix items
    prefix_items: list[tuple[int, str, Any]] = []
    for a in state.anchors:
        prefix_items.append((a.frame_id, "anchor", a))
    for e in state.entries:
        if e.visual_tokens.shape[1] != g2:
            raise ValueError(
                f"Entry visual tokens length {e.visual_tokens.shape[1]} does not match config spatial_grid^2 {g2}"
            )
        prefix_items.append((e.start_frame_id, "entry", e))

    prefix_items.sort(key=lambda item: item[0])

    # Validate strictly non-overlapping intervals
    last_end_id = -1
    for start_id, kind, obj in prefix_items:
        if start_id <= last_end_id:
            raise ValueError(
                f"Overlapping or non-monotonic prefix items: start {start_id} <= previous end {last_end_id}"
            )
        if kind == "anchor":
            last_end_id = obj.frame_id
        else:
            last_end_id = obj.end_frame_id

    if state.recent and prefix_items:
        first_recent_id = state.recent[0].frame_id
        if last_end_id >= first_recent_id:
            raise ValueError(
                f"Prefix item end frame {last_end_id} overlaps or exceeds first recent frame {first_recent_id}"
            )

    embeds_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    times_parts: list[torch.Tensor] = []

    for _, kind, obj in prefix_items:
        if kind == "anchor":
            header_text = format_frame_timestamp(obj.frame_id, obj.observation_time)
            h_embeds, h_mask = _build_header_embeds(
                header_text, tokenizer, embed_tokens, core_device, target_device, target_dtype
            )
            a_vis = obj.visual_tokens.to(dtype=target_dtype, device=target_device)
            a_vis_mask = torch.ones((1, a_vis.shape[1]), dtype=torch.long, device=target_device)

            item_embeds = torch.cat([h_embeds, a_vis], dim=1)
            item_mask = torch.cat([h_mask, a_vis_mask], dim=1)
            item_times = torch.full(
                (1, item_embeds.shape[1]),
                float(obj.observation_time),
                dtype=torch.float32,
                device=target_device,
            )
            embeds_parts.append(item_embeds)
            mask_parts.append(item_mask)
            times_parts.append(item_times)
        else:
            header_text = (
                f"Memory frames {obj.start_frame_id}-{obj.end_frame_id}, "
                f"time {obj.start_time:.6f}-{obj.end_time:.6f} s, count {obj.count}.\n"
            )
            h_embeds, h_mask = _build_header_embeds(
                header_text, tokenizer, embed_tokens, core_device, target_device, target_dtype
            )
            e_vis = obj.visual_tokens.to(dtype=target_dtype, device=target_device)
            e_vis_mask = torch.ones((1, e_vis.shape[1]), dtype=torch.long, device=target_device)

            item_embeds = torch.cat([h_embeds, e_vis], dim=1)
            item_mask = torch.cat([h_mask, e_vis_mask], dim=1)
            # summary uses end_time for all tokens
            item_times = torch.full(
                (1, item_embeds.shape[1]),
                float(obj.end_time),
                dtype=torch.float32,
                device=target_device,
            )
            embeds_parts.append(item_embeds)
            mask_parts.append(item_mask)
            times_parts.append(item_times)

    # Process recent frames
    for rf in state.recent:
        f_embeds, f_mask, f_times = _render_single_frame(
            rf=rf,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            core_device=core_device,
            target_device=target_device,
            target_dtype=target_dtype,
            intermediate_grid=config.intermediate_grid,
        )
        embeds_parts.append(f_embeds)
        mask_parts.append(f_mask)
        times_parts.append(f_times)

    inputs_embeds = torch.cat(embeds_parts, dim=1)
    attention_mask = torch.cat(mask_parts, dim=1)
    token_times = torch.cat(times_parts, dim=1)

    total_seq_len = inputs_embeds.shape[1]
    if state.recent:
        last_recent_len = embeds_parts[-1].shape[1]
        current_start = total_seq_len - last_recent_len
    else:
        current_start = total_seq_len

    return RenderedMemory(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        token_times=token_times,
        current_start=current_start,
    )


def render_compact_frames(
    frames: Sequence[MemoryFrame],
    embedder: Any,
    config: CompactMemoryConfig,
) -> RenderedMemory:
    """Render a sequence of frames without prefix bank (used for incremental appends).

    Validates monotonicity across frames.
    """
    if not isinstance(config, CompactMemoryConfig):
        raise TypeError(f"config must be a CompactMemoryConfig, got {type(config).__name__}")
    if not frames:
        raise ValueError("frames must not be empty.")

    # Validate monotonicity of frames
    for i in range(len(frames)):
        if i > 0:
            if frames[i].frame_id <= frames[i - 1].frame_id:
                raise ValueError(
                    f"frames frame_id must be strictly increasing: frame[{i}].frame_id={frames[i].frame_id} <= frame[{i-1}].frame_id={frames[i-1].frame_id}"
                )
            if frames[i].observation_time < frames[i - 1].observation_time:
                raise ValueError(
                    f"frames observation_time must be non-decreasing: frame[{i}].observation_time={frames[i].observation_time} < frame[{i-1}].observation_time={frames[i-1].observation_time}"
                )

    core, embed_tokens, tokenizer = _resolve_embedder_core(embedder)
    core_device = embed_tokens.weight.device

    target_dtype = frames[0].inputs_embeds.dtype
    target_device = frames[0].inputs_embeds.device

    embeds_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    times_parts: list[torch.Tensor] = []

    for rf in frames:
        f_embeds, f_mask, f_times = _render_single_frame(
            rf=rf,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            core_device=core_device,
            target_device=target_device,
            target_dtype=target_dtype,
            intermediate_grid=config.intermediate_grid,
        )
        embeds_parts.append(f_embeds)
        mask_parts.append(f_mask)
        times_parts.append(f_times)

    inputs_embeds = torch.cat(embeds_parts, dim=1)
    attention_mask = torch.cat(mask_parts, dim=1)
    token_times = torch.cat(times_parts, dim=1)

    total_seq_len = inputs_embeds.shape[1]
    last_frame_len = embeds_parts[-1].shape[1]
    current_start = total_seq_len - last_frame_len

    return RenderedMemory(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        token_times=token_times,
        current_start=current_start,
    )


def cold_prefix_end(
    decision_indices: Sequence[int],
    first_target: int,
    memory_config: PeriodicMemoryConfig,
) -> int:
    """Return the frame prefix boundary that has retired before the first target decision.

    Contract:
    Find the last decision d prior to first_target.
    If no prior decision exists, return 0.
    Otherwise, return max(0, ((d + 1 - R) // K) * K).
    """
    if not isinstance(memory_config, PeriodicMemoryConfig):
        raise TypeError(f"memory_config must be a PeriodicMemoryConfig, got {type(memory_config).__name__}")
    if type(first_target) is not int or first_target < 0:
        raise ValueError(f"first_target must be a non-negative int, got {first_target}")

    r = memory_config.recent_frames
    k = memory_config.consolidate_every

    prior_decisions = [d for d in decision_indices if d < first_target]
    if not prior_decisions:
        return 0

    last_d = max(prior_decisions)
    return max(0, ((last_d + 1 - r) // k) * k)
