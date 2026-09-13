"""Predictive learned memory writer, state consolidation, and numeric memory rendering.

Implements Section 2 of PREDICTIVE_WRITER_PLAN.md:
- WriterConfig dataclass
- CausalMemoryWriter: per-cell causal SDPA across observation time with Qwen-style
  half-dimension paired relative time RoPE and zero-initialized out_proj residual
  to count-weighted mean tokens.
- consolidate_predictive: maintains decision anchors and merges non-decision frames
  within boundary groups using CausalMemoryWriter without text headers.
- render_predictive_memory: renders visual-only anchors (full), summaries (16 tokens),
  recent decision frames (full 1024 embeds), and compact non-decision frames (pooled)
  with numeric token times and zero extra text headers or tokenizer calls.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.compact_memory import RenderedMemory
from fabri_moss.periodic_memory import (
    MemoryEntry,
    MemoryFrame,
    PeriodicMemoryState,
    _spatial_pool_visual_tokens,
    append_memory_frame,
    detached_state,
)


@dataclass(frozen=True)
class WriterConfig:
    input_dim: int = 1024
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    grid: int = 4
    intermediate_grid: int = 8
    recent_frames: int = 4
    consolidate_every: int = 4
    tbptt_decisions: int = 4

    def __post_init__(self) -> None:
        int_fields = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "grid": self.grid,
            "intermediate_grid": self.intermediate_grid,
            "recent_frames": self.recent_frames,
            "consolidate_every": self.consolidate_every,
            "tbptt_decisions": self.tbptt_decisions,
        }
        for name, val in int_fields.items():
            if type(val) is not int or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        head_dim = self.hidden_dim // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim} = hidden_dim // num_heads) must be even for half-dim RoPE pairing"
            )


def _apply_half_rope(x: torch.Tensor, rel_times: torch.Tensor) -> torch.Tensor:
    """Apply Qwen-style rotate_half RoPE using relative time on query or key.

    x: [B, num_heads, T, head_dim]
    rel_times: [B, T] or [1, T] in seconds (float32)
    head_dim must be even.
    inv_freq = 10000^(-2i / head_dim) for i in [0, head_dim // 2).
    rotate_half: [-x2, x1] where x1, x2 = x[..., :d//2], x[..., d//2:].
    rotated = x * cos(phase) + rotate_half(x) * sin(phase)
    where phase = cat([p, p], dim=-1) matching the half-dim layout.
    """
    b, num_heads, t, head_dim = x.shape
    half_dim = head_dim // 2
    device = x.device
    dtype = x.dtype

    # Non-persistent inv_freq: 10000^(-2i / head_dim)
    indices = torch.arange(half_dim, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (2.0 * indices / float(head_dim)))  # [half_dim]

    times_fp32 = rel_times.to(dtype=torch.float32, device=device)  # [B, T]
    # Phase: [B, 1, T, half_dim]
    phase_half = times_fp32.unsqueeze(1).unsqueeze(-1) * inv_freq.view(1, 1, 1, half_dim)
    phase = torch.cat([phase_half, phase_half], dim=-1)  # [B, 1, T, head_dim]

    cos_t = phase.cos().to(dtype=dtype)
    sin_t = phase.sin().to(dtype=dtype)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotate_half_x = torch.cat([-x2, x1], dim=-1)

    return x * cos_t + rotate_half_x * sin_t


class CausalSDPABlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, rel_times: torch.Tensor) -> torch.Tensor:
        """Forward pass of Pre-LN Causal SDPA Block.

        x: [B, T, hidden_dim] where B = spatial_cells (e.g. 16)
        rel_times: [1, T] float32
        """
        b, t, d = x.shape
        # Attention sub-layer with Pre-LN
        norm_x = self.ln1(x)

        q = self.q_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply half-rope relative time RoPE to Q and K
        q = _apply_half_rope(q, rel_times)
        k = _apply_half_rope(k, rel_times)

        # PyTorch causal SDPA (is_causal=True)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )  # [B, num_heads, T, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out

        # MLP sub-layer with Pre-LN
        norm_x2 = self.ln2(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out
        return x


class CausalMemoryWriter(nn.Module):
    def __init__(self, config: WriterConfig) -> None:
        super().__init__()
        if not isinstance(config, WriterConfig):
            raise TypeError(f"config must be a WriterConfig, got {type(config).__name__}")
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        # Metadata projection for each input token: (span, log1p(count)) -> hidden_dim
        self.meta_proj = nn.Linear(2, config.hidden_dim)

        self.blocks = nn.ModuleList(
            [CausalSDPABlock(config.hidden_dim, config.num_heads) for _ in range(config.num_layers)]
        )
        self.final_ln = nn.LayerNorm(config.hidden_dim)

        # Output projection back to input_dim; zero initialized so initial residual is 0
        self.out_proj = nn.Linear(config.hidden_dim, config.input_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        frames: Sequence[MemoryFrame],
        previous: MemoryEntry | None = None,
    ) -> MemoryEntry:
        """Consolidate frames (and optional previous entry) into a new MemoryEntry.

        frames: strictly increasing frame_id, same boundary_frame_id, non-decision.
        previous: optional seed token representing older merged interval.
        """
        if not frames:
            raise ValueError("frames sequence cannot be empty")

        boundary_id = frames[0].boundary_frame_id
        for f in frames:
            if f.is_decision:
                raise ValueError(f"Decision frame {f.frame_id} cannot be consolidated by CausalMemoryWriter")
            if f.boundary_frame_id != boundary_id:
                raise ValueError(
                    f"Frame {f.frame_id} boundary {f.boundary_frame_id} mismatch with batch boundary {boundary_id}"
                )

        if previous is not None and previous.boundary_frame_id != boundary_id:
            raise ValueError(
                f"Previous entry boundary {previous.boundary_frame_id} mismatch with frames boundary {boundary_id}"
            )

        # Validate strictly increasing frame_id and non-decreasing observation_time
        prev_fid = None
        prev_time = None
        if previous is not None:
            prev_fid = previous.end_frame_id
            prev_time = previous.end_time

        for f in frames:
            if prev_fid is not None and f.frame_id <= prev_fid:
                raise ValueError(
                    f"Frames must be strictly increasing: frame_id {f.frame_id} <= previous {prev_fid}"
                )
            if prev_time is not None and f.observation_time < prev_time:
                raise ValueError(
                    f"Observation times must be non-decreasing: time {f.observation_time} < previous {prev_time}"
                )
            prev_fid = f.frame_id
            prev_time = f.observation_time

        g = self.config.grid
        g2 = g * g
        target_dtype = frames[0].visual_tokens.dtype
        target_device = frames[0].visual_tokens.device

        # 1. Pool each frame spatially to [1, G^2, input_dim]
        # Then arrange into [G^2, T_frames, input_dim]
        pooled_frames = []
        raw_times = []
        raw_spans = []
        raw_counts = []

        for f in frames:
            p_tok = _spatial_pool_visual_tokens(f.visual_tokens.float(), g)  # [1, G^2, input_dim]
            pooled_frames.append(p_tok)
            raw_times.append(float(f.observation_time))
            raw_spans.append(0.0)
            raw_counts.append(1.0)

        # Stack frames: [T_frames, G^2, input_dim] -> transpose to [G^2, T_frames, input_dim]
        stacked_frames = torch.cat(pooled_frames, dim=0).transpose(0, 1)  # [G^2, T_frames, input_dim]

        if previous is not None:
            # previous visual_tokens is already [1, G^2, input_dim]
            if previous.visual_tokens.shape[1] != g2:
                raise ValueError(
                    f"previous visual_tokens length {previous.visual_tokens.shape[1]} != grid^2 {g2}"
                )
            prev_tok = previous.visual_tokens.transpose(0, 1)  # [G^2, 1, input_dim]
            all_tokens = torch.cat([prev_tok, stacked_frames], dim=1)  # [G^2, T, input_dim]

            all_times = [float(previous.end_time)] + raw_times
            prev_span = float(previous.end_time - previous.start_time)
            all_spans = [prev_span] + raw_spans
            all_counts = [float(previous.count)] + raw_counts

            start_fid = previous.start_frame_id
            start_t = previous.start_time
            total_count = previous.count + len(frames)
        else:
            all_tokens = stacked_frames  # [G^2, T, input_dim]
            all_times = raw_times
            all_spans = raw_spans
            all_counts = raw_counts

            start_fid = frames[0].frame_id
            start_t = float(frames[0].observation_time)
            total_count = len(frames)

        end_fid = frames[-1].frame_id
        end_t = float(frames[-1].observation_time)
        total_span = end_t - start_t

        t_steps = all_tokens.shape[1]

        # 2. Compute count-weighted base tokens in FP32
        if previous is not None:
            w_old = float(previous.count) / float(total_count)
            t_old = previous.visual_tokens.to(torch.float32)  # [1, G^2, input_dim]
            # sum of new pooled frames
            t_new_sum = torch.stack(pooled_frames, dim=0).sum(dim=0).to(torch.float32)  # [1, G^2, input_dim]
            base_tokens = w_old * t_old + (t_new_sum / float(total_count))
        else:
            t_new_sum = torch.stack(pooled_frames, dim=0).sum(dim=0).to(torch.float32)
            base_tokens = t_new_sum / float(total_count)

        # 3. Build sequence embeddings for writer:
        # relative times = obs_time - start_t
        rel_times_tensor = torch.tensor(
            [[t - start_t for t in all_times]], dtype=torch.float32, device=target_device
        )  # [1, T]

        meta_inputs = torch.tensor(
            [[s, math.log1p(c)] for s, c in zip(all_spans, all_counts)],
            dtype=torch.float32,
            device=target_device,
        )  # [T, 2]
        meta_emb = self.meta_proj(meta_inputs).unsqueeze(0)  # [1, T, hidden_dim]

        # tokens: [G^2, T, input_dim]
        # input_proj -> [G^2, T, hidden_dim]
        h = self.input_proj(all_tokens) + meta_emb  # [G^2, T, hidden_dim]

        # Pre-LN Causal SDPA blocks
        for block in self.blocks:
            h = block(h, rel_times_tensor)

        h = self.final_ln(h)

        # Extract last hidden state at time T-1: [G^2, hidden_dim]
        last_h = h[:, -1, :]  # [G^2, hidden_dim]

        # Condition on total span and total count
        total_meta_input = torch.tensor(
            [[total_span, math.log1p(total_count)]],
            dtype=torch.float32,
            device=target_device,
        )  # [1, 2]
        total_meta_emb = self.meta_proj(total_meta_input).squeeze(0)  # [hidden_dim]

        delta = self.out_proj(last_h + total_meta_emb)  # [G^2, input_dim]
        delta = delta.unsqueeze(0).to(torch.float32)  # [1, G^2, input_dim]

        final_tokens = (base_tokens + delta).to(dtype=target_dtype)

        return MemoryEntry(
            visual_tokens=final_tokens,
            start_frame_id=start_fid,
            end_frame_id=end_fid,
            start_time=start_t,
            end_time=end_t,
            count=total_count,
            boundary_frame_id=boundary_id,
        )


def consolidate_predictive(
    state: PeriodicMemoryState,
    writer: CausalMemoryWriter,
    config: WriterConfig | None = None,
) -> PeriodicMemoryState:
    """Consolidate memory state by retiring oldest K frames while len(recent) >= R + K.

    - Protects decision frames by detaching and cloning them into state.anchors.
    - Groups retired non-decision frames by boundary_frame_id.
    - Consolidates each boundary group with existing entry (if any) via CausalMemoryWriter.
    - Does NOT force torch.no_grad(); gradient flow is determined by the caller.
    - Maintains frame_count, consolidations, and merges statistics.
    """
    cfg = config if config is not None else writer.config
    r = cfg.recent_frames
    k = cfg.consolidate_every

    recent_list = list(state.recent)
    entries_list = list(state.entries)
    anchors_list = list(state.anchors)
    consolidations = state.consolidations
    merges = state.merges

    while len(recent_list) >= r + k:
        retired_frames = recent_list[:k]
        recent_list = recent_list[k:]

        consolidations += 1

        batch_anchors = []
        non_anchors = []
        for rf in retired_frames:
            if rf.is_decision:
                anchor_frame = MemoryFrame(
                    frame_id=rf.frame_id,
                    observation_time=rf.observation_time,
                    inputs_embeds=rf.inputs_embeds.detach().clone(),
                    attention_mask=rf.attention_mask.detach().clone(),
                    visual_tokens=rf.visual_tokens.detach().clone(),
                    is_decision=True,
                    boundary_frame_id=rf.boundary_frame_id,
                )
                batch_anchors.append(anchor_frame)
            else:
                non_anchors.append(rf)

        anchors_list.extend(batch_anchors)

        if non_anchors:
            boundaries: dict[int | None, list[MemoryFrame]] = collections.OrderedDict()
            for rf in non_anchors:
                b = rf.boundary_frame_id
                if b not in boundaries:
                    boundaries[b] = []
                boundaries[b].append(rf)

            for b_id, b_frames in boundaries.items():
                existing_idx = None
                for idx, e in enumerate(entries_list):
                    if e.boundary_frame_id == b_id:
                        existing_idx = idx
                        break

                if existing_idx is not None:
                    old_entry = entries_list[existing_idx]
                    merged_entry = writer(b_frames, previous=old_entry)
                    entries_list[existing_idx] = merged_entry
                    merges += 1
                else:
                    new_entry = writer(b_frames, previous=None)
                    entries_list.append(new_entry)
                    entries_list.sort(key=lambda x: x.start_frame_id)

    return PeriodicMemoryState(
        recent=tuple(recent_list),
        entries=tuple(entries_list),
        frame_count=state.frame_count,
        consolidations=consolidations,
        merges=merges,
        anchors=tuple(anchors_list),
        last_decision_frame_id=state.last_decision_frame_id,
    )


append_predictive_frame = append_memory_frame


def advance_predictive(
    state: PeriodicMemoryState,
    frame: MemoryFrame,
    writer: CausalMemoryWriter,
    config: WriterConfig | None = None,
) -> PeriodicMemoryState:
    """Convenience helper: append frame then consolidate_predictive."""
    appended = append_memory_frame(state, frame)
    return consolidate_predictive(appended, writer, config)


def render_predictive_memory(
    state: PeriodicMemoryState,
    config: WriterConfig,
) -> RenderedMemory:
    """Render memory state purely with visual representations and numeric token times.

    - No text headers, no tokenizer, no embed_tokens calls.
    - Anchors: full visual tokens [1, 256, H] (or P tokens), time = observation_time.
    - Entries: summary 16 tokens [1, 16, H], time = end_time.
    - Anchors and entries sorted chronologically by original start time (observation_time / start_time).
    - Recent frames:
      - Non-decision frames: pooled to intermediate_grid x intermediate_grid (e.g. 8x8 = 64),
        attention mask all 1s, time = observation_time.
        If spatial side < intermediate_grid, clamped to side (no upsampling).
      - Decision frames: full inputs_embeds (e.g. 1024), full attention_mask, time = observation_time.
    - current_start points to the start of the final recent frame (which must be a decision frame).
    """
    if not isinstance(config, WriterConfig):
        raise TypeError(f"config must be a WriterConfig, got {type(config).__name__}")
    if not state.recent and not state.entries and not state.anchors:
        raise ValueError("Cannot render memory from empty state: recent, entries, and anchors are all empty.")

    # Contract requirement: last frame in recent must be is_decision
    if not state.recent:
        raise ValueError("Cannot render memory without recent frames (query decision frame required)")
    if not state.recent[-1].is_decision:
        raise ValueError(
            f"Final frame {state.recent[-1].frame_id} in recent must be a decision frame to serve as query"
        )

    # Reference dtype and device
    target_dtype = state.recent[-1].inputs_embeds.dtype
    target_device = state.recent[-1].inputs_embeds.device

    # Prefix items: (start_sort_key, kind, obj)
    # Chronological sort by start_time / observation_time (or start_frame_id)
    prefix_items: list[tuple[int, str, Any]] = []
    for a in state.anchors:
        prefix_items.append((a.frame_id, "anchor", a))
    for e in state.entries:
        prefix_items.append((e.start_frame_id, "entry", e))

    prefix_items.sort(key=lambda item: item[0])

    # Validate non-overlapping intervals
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
            a_vis = obj.visual_tokens.to(dtype=target_dtype, device=target_device)
            seq_len = a_vis.shape[1]
            a_mask = torch.ones((1, seq_len), dtype=torch.long, device=target_device)
            a_times = torch.full(
                (1, seq_len),
                float(obj.observation_time),
                dtype=torch.float32,
                device=target_device,
            )
            embeds_parts.append(a_vis)
            mask_parts.append(a_mask)
            times_parts.append(a_times)
        else:
            e_vis = obj.visual_tokens.to(dtype=target_dtype, device=target_device)
            seq_len = e_vis.shape[1]
            e_mask = torch.ones((1, seq_len), dtype=torch.long, device=target_device)
            # Learned summary anchors to end_time
            e_times = torch.full(
                (1, seq_len),
                float(obj.end_time),
                dtype=torch.float32,
                device=target_device,
            )
            embeds_parts.append(e_vis)
            mask_parts.append(e_mask)
            times_parts.append(e_times)

    # Process recent frames
    for rf in state.recent:
        if rf.is_decision:
            f_embeds = rf.inputs_embeds.to(dtype=target_dtype, device=target_device)
            f_mask = rf.attention_mask.to(device=target_device)
            seq_len = f_embeds.shape[1]
            f_times = torch.full(
                (1, seq_len),
                float(rf.observation_time),
                dtype=torch.float32,
                device=target_device,
            )
            embeds_parts.append(f_embeds)
            mask_parts.append(f_mask)
            times_parts.append(f_times)
        else:
            p = rf.visual_tokens.shape[1]
            side = int(math.isqrt(p))
            target_g = min(config.intermediate_grid, side)
            pooled_vis = _spatial_pool_visual_tokens(rf.visual_tokens, target_g)
            f_embeds = pooled_vis.to(dtype=target_dtype, device=target_device)
            seq_len = f_embeds.shape[1]
            f_mask = torch.ones((1, seq_len), dtype=torch.long, device=target_device)
            f_times = torch.full(
                (1, seq_len),
                float(rf.observation_time),
                dtype=torch.float32,
                device=target_device,
            )
            embeds_parts.append(f_embeds)
            mask_parts.append(f_mask)
            times_parts.append(f_times)

    inputs_embeds = torch.cat(embeds_parts, dim=1)
    attention_mask = torch.cat(mask_parts, dim=1)
    token_times = torch.cat(times_parts, dim=1)

    # current_start points to the start of the final recent frame
    last_frame_len = state.recent[-1].inputs_embeds.shape[1]
    current_start = inputs_embeds.shape[1] - last_frame_len

    return RenderedMemory(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        token_times=token_times,
        current_start=current_start,
    )
