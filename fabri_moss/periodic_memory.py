"""Protect decision-frame visual tokens and pool intermediate historical observations.

Protected memory grows with episode decisions. The optional unprotected path
uses a bounded bank with adjacent similarity merging.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PeriodicMemoryConfig:
    recent_frames: int = 4
    consolidate_every: int = 4
    memory_slots: int = 16
    spatial_grid: int = 4
    protect_decision_frames: bool = True

    def __post_init__(self) -> None:
        if type(self.recent_frames) is not int or self.recent_frames <= 0:
            raise ValueError(f"recent_frames must be a positive integer, got {self.recent_frames}")
        if type(self.consolidate_every) is not int or self.consolidate_every <= 0:
            raise ValueError(f"consolidate_every must be a positive integer, got {self.consolidate_every}")
        if type(self.memory_slots) is not int or self.memory_slots <= 0:
            raise ValueError(f"memory_slots must be a positive integer, got {self.memory_slots}")
        if type(self.spatial_grid) is not int or self.spatial_grid <= 0:
            raise ValueError(f"spatial_grid must be a positive integer, got {self.spatial_grid}")
        if not isinstance(self.protect_decision_frames, bool):
            raise TypeError(f"protect_decision_frames must be a bool, got {type(self.protect_decision_frames).__name__}")


@dataclass(frozen=True)
class MemoryFrame:
    frame_id: int
    observation_time: float
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    visual_tokens: torch.Tensor
    is_decision: bool = False
    boundary_frame_id: int | None = None

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise TypeError(f"frame_id must be a non-negative int, got {type(self.frame_id).__name__}")
        if isinstance(self.observation_time, bool) or not isinstance(self.observation_time, (int, float)):
            raise TypeError(f"observation_time must be a float or int, got {type(self.observation_time).__name__}")
        obs_time = float(self.observation_time)
        if not math.isfinite(obs_time) or obs_time < 0.0:
            raise ValueError(f"observation_time must be a finite non-negative float, got {self.observation_time}")

        if not isinstance(self.is_decision, bool):
            raise TypeError(f"is_decision must be a bool, got {type(self.is_decision).__name__}")
        if self.boundary_frame_id is not None:
            if type(self.boundary_frame_id) is not int or self.boundary_frame_id < 0:
                raise TypeError(f"boundary_frame_id must be a non-negative int or None, got {self.boundary_frame_id}")

        # Validate tensors
        if not isinstance(self.inputs_embeds, torch.Tensor):
            raise TypeError(f"inputs_embeds must be a torch.Tensor, got {type(self.inputs_embeds).__name__}")
        if not isinstance(self.attention_mask, torch.Tensor):
            raise TypeError(f"attention_mask must be a torch.Tensor, got {type(self.attention_mask).__name__}")
        if not isinstance(self.visual_tokens, torch.Tensor):
            raise TypeError(f"visual_tokens must be a torch.Tensor, got {type(self.visual_tokens).__name__}")

        if self.inputs_embeds.ndim != 3 or self.inputs_embeds.shape[0] != 1:
            raise ValueError(f"inputs_embeds must have shape [1, L, H], got {list(self.inputs_embeds.shape)}")
        if self.attention_mask.ndim != 2 or self.attention_mask.shape[0] != 1:
            raise ValueError(f"attention_mask must have shape [1, L], got {list(self.attention_mask.shape)}")
        if self.visual_tokens.ndim != 3 or self.visual_tokens.shape[0] != 1:
            raise ValueError(f"visual_tokens must have shape [1, P, H], got {list(self.visual_tokens.shape)}")

        seq_len = self.inputs_embeds.shape[1]
        mask_len = self.attention_mask.shape[1]
        if seq_len != mask_len:
            raise ValueError(f"inputs_embeds sequence length {seq_len} != attention_mask length {mask_len}")

        hidden_dim = self.inputs_embeds.shape[2]
        vis_hidden_dim = self.visual_tokens.shape[2]
        if hidden_dim != vis_hidden_dim:
            raise ValueError(f"inputs_embeds hidden_dim {hidden_dim} != visual_tokens hidden_dim {vis_hidden_dim}")

        p = self.visual_tokens.shape[1]
        sqrt_p = int(math.isqrt(p))
        if sqrt_p * sqrt_p != p or p <= 0:
            raise ValueError(f"visual_tokens token count P={p} must be a positive perfect square (e.g. 256 for 16x16)")


@dataclass(frozen=True)
class MemoryEntry:
    visual_tokens: torch.Tensor
    start_frame_id: int
    end_frame_id: int
    start_time: float
    end_time: float
    count: int
    boundary_frame_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.visual_tokens, torch.Tensor):
            raise TypeError(f"visual_tokens must be a torch.Tensor, got {type(self.visual_tokens).__name__}")
        if self.visual_tokens.ndim != 3 or self.visual_tokens.shape[0] != 1:
            raise ValueError(f"visual_tokens must have shape [1, G^2, H], got {list(self.visual_tokens.shape)}")

        if not isinstance(self.start_frame_id, int) or not isinstance(self.end_frame_id, int):
            raise TypeError("start_frame_id and end_frame_id must be integers")
        if self.start_frame_id > self.end_frame_id:
            raise ValueError(f"start_frame_id ({self.start_frame_id}) must be <= end_frame_id ({self.end_frame_id})")

        if not isinstance(self.start_time, (int, float)) or not isinstance(self.end_time, (int, float)):
            raise TypeError("start_time and end_time must be floats")
        st, et = float(self.start_time), float(self.end_time)
        if not math.isfinite(st) or st < 0.0 or not math.isfinite(et) or et < 0.0:
            raise ValueError("start_time and end_time must be finite non-negative numbers")
        if st > et:
            raise ValueError(f"start_time ({st}) must be <= end_time ({et})")

        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError(f"count must be a positive integer, got {self.count}")

        if self.boundary_frame_id is not None:
            if type(self.boundary_frame_id) is not int or self.boundary_frame_id < 0:
                raise TypeError(f"boundary_frame_id must be a non-negative int or None, got {self.boundary_frame_id}")


@dataclass(frozen=True)
class PeriodicMemoryState:
    recent: Tuple[MemoryFrame, ...] = ()
    entries: Tuple[MemoryEntry, ...] = ()
    frame_count: int = 0
    consolidations: int = 0
    merges: int = 0
    anchors: Tuple[MemoryFrame, ...] = ()
    last_decision_frame_id: int | None = None

    @property
    def last_frame_id(self) -> int | None:
        if self.recent:
            return self.recent[-1].frame_id
        if self.entries:
            return self.entries[-1].end_frame_id
        if self.anchors:
            return self.anchors[-1].frame_id
        return None

    @property
    def last_observation_time(self) -> float | None:
        if self.recent:
            return float(self.recent[-1].observation_time)
        if self.entries:
            return float(self.entries[-1].end_time)
        if self.anchors:
            return float(self.anchors[-1].observation_time)
        return None


def _spatial_pool_visual_tokens(visual_tokens: torch.Tensor, target_grid: int) -> torch.Tensor:
    """Pool [1, P, H] where P = S * S into [1, G^2, H] where G = target_grid.

    Preserves spatial row-major ordering.
    """
    _, p, h = visual_tokens.shape
    s = int(math.isqrt(p))
    x = visual_tokens.transpose(1, 2).view(1, h, s, s)
    pooled = F.adaptive_avg_pool2d(x, (target_grid, target_grid))
    g2 = target_grid * target_grid
    return pooled.view(1, h, g2).transpose(1, 2)


def _pairwise_token_cosine_similarity(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> float:
    """Compute mean cosine similarity across corresponding tokens in FP32 without graph."""
    a = tokens_a.squeeze(0).to(torch.float32)  # [G^2, H]
    b = tokens_b.squeeze(0).to(torch.float32)  # [G^2, H]
    cos_sim = F.cosine_similarity(a, b, dim=-1)  # [G^2]
    return float(cos_sim.mean().item())


def format_frame_timestamp(frame_id: int, observation_time: float) -> str:
    """Format standard text timestamp header for a keyframe anchor."""
    return f"Frame {frame_id}, time {observation_time:.6f} s.\n"


def _build_header_embeds(
    header_text: str,
    tokenizer: Any,
    embed_tokens: Any,
    core_device: torch.device,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared helper to tokenize and embed a header text."""
    tok_res = tokenizer(header_text, return_tensors="pt", add_special_tokens=False)
    if hasattr(tok_res, "input_ids"):
        input_ids = tok_res.input_ids
    elif isinstance(tok_res, dict) and "input_ids" in tok_res:
        input_ids = tok_res["input_ids"]
    elif isinstance(tok_res, torch.Tensor):
        input_ids = tok_res
    else:
        raise TypeError(f"Unexpected tokenizer output type: {type(tok_res)}")

    if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
        raise ValueError(f"Header token IDs must be a non-empty torch.Tensor, got {input_ids}")

    input_ids = input_ids.to(core_device)
    header_embeds = embed_tokens(input_ids).to(dtype=target_dtype, device=target_device)
    if header_embeds.ndim != 3 or header_embeds.shape[0] != 1:
        raise ValueError(f"Header embeds must have shape [1, L, H], got {list(header_embeds.shape)}")

    header_len = header_embeds.shape[1]
    header_mask = torch.ones((1, header_len), dtype=torch.long, device=target_device)
    return header_embeds, header_mask


def append_memory_frame(
    state: PeriodicMemoryState,
    frame: MemoryFrame,
) -> PeriodicMemoryState:
    """Append a new frame to the memory state without retiring frames.

    - Validates strictly increasing frame_id and non-decreasing observation_time.
    - Validates visual_tokens consistency across frames.
    - Sets boundary_frame_id to state.last_decision_frame_id before adding to recent.
    - Updates last_decision_frame_id if frame.is_decision is True.
    - Increments frame_count by 1.
    - Does NOT consolidate or retire frames from recent.
    """
    if state.recent:
        last_id = state.recent[-1].frame_id
        last_time = state.recent[-1].observation_time
        ref_frame = state.recent[0]
    elif state.entries:
        last_id = state.entries[-1].end_frame_id
        last_time = state.entries[-1].end_time
        ref_frame = None
    elif state.anchors:
        last_id = state.anchors[-1].frame_id
        last_time = state.anchors[-1].observation_time
        ref_frame = state.anchors[0]
    else:
        last_id = None
        last_time = None
        ref_frame = None

    if last_id is not None and frame.frame_id <= last_id:
        raise ValueError(f"frame_id must be strictly increasing: new {frame.frame_id} <= last {last_id}")
    if last_time is not None and frame.observation_time < last_time:
        raise ValueError(
            f"observation_time must be non-decreasing: new {frame.observation_time} < last {last_time}"
        )

    # Validate uniformity of visual_tokens across frames if reference exists
    if ref_frame is not None:
        if frame.visual_tokens.shape != ref_frame.visual_tokens.shape:
            raise ValueError(
                f"visual_tokens shape mismatch: got {list(frame.visual_tokens.shape)}, expected {list(ref_frame.visual_tokens.shape)}"
            )
        if frame.visual_tokens.dtype != ref_frame.visual_tokens.dtype:
            raise ValueError(
                f"visual_tokens dtype mismatch: got {frame.visual_tokens.dtype}, expected {ref_frame.visual_tokens.dtype}"
            )
        if frame.visual_tokens.device != ref_frame.visual_tokens.device:
            raise ValueError(
                f"visual_tokens device mismatch: got {frame.visual_tokens.device}, expected {ref_frame.visual_tokens.device}"
            )
    elif state.entries:
        # Check hidden dim matches entry
        entry_h = state.entries[0].visual_tokens.shape[2]
        if frame.visual_tokens.shape[2] != entry_h:
            raise ValueError(
                f"visual_tokens hidden dim mismatch with entries: got {frame.visual_tokens.shape[2]}, expected {entry_h}"
            )

    assigned_boundary = state.last_decision_frame_id
    stored_frame = dataclasses.replace(frame, boundary_frame_id=assigned_boundary)

    if frame.is_decision:
        new_last_decision = frame.frame_id
    else:
        new_last_decision = state.last_decision_frame_id

    recent_list = list(state.recent)
    recent_list.append(stored_frame)

    return PeriodicMemoryState(
        recent=tuple(recent_list),
        entries=state.entries,
        frame_count=state.frame_count + 1,
        consolidations=state.consolidations,
        merges=state.merges,
        anchors=state.anchors,
        last_decision_frame_id=new_last_decision,
    )


def consolidate_memory(
    state: PeriodicMemoryState,
    config: PeriodicMemoryConfig,
) -> PeriodicMemoryState:
    """Consolidate memory state by retiring frames while len(recent) >= R + K.

    - Does NOT increment frame_count.
    - Handles multiple retirement batches if len(recent) >= R + K.
    - Preserves anchors and pools entries exactly as defined in advance_memory.
    """
    r = config.recent_frames
    k = config.consolidate_every
    slots = config.memory_slots
    g = config.spatial_grid
    protect = config.protect_decision_frames

    recent_list = list(state.recent)
    entries_list = list(state.entries)
    anchors_list = list(state.anchors)
    consolidations = state.consolidations
    merges = state.merges

    # While len(recent) >= R + K: retire oldest K frames
    while len(recent_list) >= r + k:
        retired_frames = recent_list[:k]
        recent_list = recent_list[k:]

        if protect:
            # Explicit decision boundaries mode
            consolidations += 1

            # 1. Split retired batch into decision anchors and non-decision frames
            batch_anchors = []
            non_anchors = []
            for rf in retired_frames:
                if rf.is_decision:
                    # Detach and clone tensors to have independent storage (avoid view pinning cold batch)
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

            # 2. Group non-decision frames by boundary_frame_id
            if non_anchors:
                from collections import OrderedDict
                boundaries: dict[int | None, list[MemoryFrame]] = OrderedDict()
                for rf in non_anchors:
                    b = rf.boundary_frame_id
                    if b not in boundaries:
                        boundaries[b] = []
                    boundaries[b].append(rf)

                for b_id, b_frames in boundaries.items():
                    # Spatial pool each non-decision frame and average across batch frames in this boundary
                    with torch.no_grad():
                        pooled_list = []
                        orig_dtype = b_frames[0].visual_tokens.dtype
                        for bf in b_frames:
                            p_tok = _spatial_pool_visual_tokens(bf.visual_tokens, g)
                            pooled_list.append(p_tok.to(torch.float32))

                        stacked = torch.stack(pooled_list, dim=0)  # [len(b_frames), 1, G^2, H]
                        batch_bin_tokens = stacked.mean(dim=0).to(orig_dtype).detach().clone()

                    b_start_id = b_frames[0].frame_id
                    b_end_id = b_frames[-1].frame_id
                    b_start_time = b_frames[0].observation_time
                    b_end_time = b_frames[-1].observation_time
                    b_count = len(b_frames)

                    # Check if an entry for this boundary already exists in entries_list
                    existing_idx = None
                    for idx, e in enumerate(entries_list):
                        if e.boundary_frame_id == b_id:
                            existing_idx = idx
                            break

                    if existing_idx is not None:
                        # Merge with existing entry in the same boundary
                        old_entry = entries_list[existing_idx]
                        total_count = old_entry.count + b_count
                        w_old = float(old_entry.count) / float(total_count)
                        w_new = float(b_count) / float(total_count)

                        with torch.no_grad():
                            dtype = old_entry.visual_tokens.dtype
                            t_old = old_entry.visual_tokens.to(torch.float32)
                            t_new = batch_bin_tokens.to(torch.float32)
                            merged_tokens = (w_old * t_old + w_new * t_new).to(dtype).detach().clone()

                        merged_entry = MemoryEntry(
                            visual_tokens=merged_tokens,
                            start_frame_id=min(old_entry.start_frame_id, b_start_id),
                            end_frame_id=max(old_entry.end_frame_id, b_end_id),
                            start_time=min(old_entry.start_time, b_start_time),
                            end_time=max(old_entry.end_time, b_end_time),
                            count=total_count,
                            boundary_frame_id=b_id,
                        )
                        entries_list[existing_idx] = merged_entry
                        merges += 1
                    else:
                        new_entry = MemoryEntry(
                            visual_tokens=batch_bin_tokens,
                            start_frame_id=b_start_id,
                            end_frame_id=b_end_id,
                            start_time=b_start_time,
                            end_time=b_end_time,
                            count=b_count,
                            boundary_frame_id=b_id,
                        )
                        entries_list.append(new_entry)
                        # Keep entries_list sorted by start_frame_id
                        entries_list.sort(key=lambda x: x.start_frame_id)

        else:
            with torch.no_grad():
                pooled_tokens_list = []
                orig_dtype = retired_frames[0].visual_tokens.dtype
                for rf in retired_frames:
                    p_tok = _spatial_pool_visual_tokens(rf.visual_tokens, g)
                    pooled_tokens_list.append(p_tok.to(torch.float32))

                stacked = torch.stack(pooled_tokens_list, dim=0)  # [K, 1, G^2, H]
                mean_entry_tokens = stacked.mean(dim=0).to(orig_dtype).detach().clone()

            start_frame_id = retired_frames[0].frame_id
            end_frame_id = retired_frames[-1].frame_id
            start_time = retired_frames[0].observation_time
            end_time = retired_frames[-1].observation_time
            count = len(retired_frames)

            new_entry = MemoryEntry(
                visual_tokens=mean_entry_tokens,
                start_frame_id=start_frame_id,
                end_frame_id=end_frame_id,
                start_time=start_time,
                end_time=end_time,
                count=count,
                boundary_frame_id=None,
            )
            entries_list.append(new_entry)
            consolidations += 1

            # Capacity management: if len(entries_list) > slots, merge adjacent pair with highest similarity
            while len(entries_list) > slots:
                best_idx = 0
                best_sim = -float("inf")
                with torch.no_grad():
                    for i in range(len(entries_list) - 1):
                        sim = _pairwise_token_cosine_similarity(
                            entries_list[i].visual_tokens,
                            entries_list[i + 1].visual_tokens,
                        )
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = i

                e1 = entries_list[best_idx]
                e2 = entries_list[best_idx + 1]

                total_count = e1.count + e2.count
                w1 = float(e1.count) / float(total_count)
                w2 = float(e2.count) / float(total_count)

                with torch.no_grad():
                    dtype = e1.visual_tokens.dtype
                    t1 = e1.visual_tokens.to(torch.float32)
                    t2 = e2.visual_tokens.to(torch.float32)
                    merged_tokens = (w1 * t1 + w2 * t2).to(dtype).detach()

                merged_entry = MemoryEntry(
                    visual_tokens=merged_tokens,
                    start_frame_id=e1.start_frame_id,
                    end_frame_id=e2.end_frame_id,
                    start_time=e1.start_time,
                    end_time=e2.end_time,
                    count=total_count,
                    boundary_frame_id=None,
                )

                entries_list[best_idx : best_idx + 2] = [merged_entry]
                merges += 1

    return PeriodicMemoryState(
        recent=tuple(recent_list),
        entries=tuple(entries_list),
        frame_count=state.frame_count,
        consolidations=consolidations,
        merges=merges,
        anchors=tuple(anchors_list),
        last_decision_frame_id=state.last_decision_frame_id,
    )


def advance_memory(
    state: PeriodicMemoryState,
    frame: MemoryFrame,
    config: PeriodicMemoryConfig,
) -> PeriodicMemoryState:
    """Advance memory state with a new frame.

    Equivalent to consolidate_memory(append_memory_frame(state, frame), config).
    """
    return consolidate_memory(append_memory_frame(state, frame), config)


def materialize_memory(
    state: PeriodicMemoryState,
    embedder: Any,
    config: PeriodicMemoryConfig,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Materialize full sequence prefix embeddings and attention mask from memory state.

    Prefix items (anchors and entries) are sorted chronologically by original starting frame_id:
    - For each MemoryEntry:
        - Text header: 'Memory frames {start}-{end}, time {start:.6f}-{end:.6f} s, count {count}.\n'
        - Embedded via core.embed_tokens
        - Followed by entry.visual_tokens [1, G^2, H]
        - Attention mask is all 1s
    - For each Anchor (MemoryFrame):
        - Text header: format_frame_timestamp(anchor.frame_id, anchor.observation_time)
          i.e. 'Frame {frame_id}, time {observation_time:.6f} s.\n'
        - Embedded via core.embed_tokens
        - Followed by anchor.visual_tokens [1, P, H] (exact, unpooled)
        - Attention mask is all 1s
    Followed by recent frames (in chronological order):
        - inputs_embeds and attention_mask as stored

    Returns:
      inputs_embeds: [1, total_len, H]
      attention_mask: [1, total_len]
      current_start: int, total_len - len(state.recent[-1].inputs_embeds)
    """
    if not state.recent and not state.entries and not state.anchors:
        raise ValueError("Cannot materialize memory from empty state: recent, entries, and anchors are all empty.")

    # Resolve native core: embedder.model.language_model.model (or language_model itself)
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

    # Determine reference device and dtype from recent, anchor, or core
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

    tokenizer = getattr(embedder, "tokenizer", None)
    if (state.entries or state.anchors) and tokenizer is None:
        raise AttributeError("embedder must have a 'tokenizer' attribute when entries or anchors are present")

    g2 = config.spatial_grid * config.spatial_grid

    # Interleave / sort prefix items by starting frame_id
    # Item tuple: (start_frame_id, 'anchor'|'entry', obj)
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

    # Validate strictly non-overlapping intervals across prefix items
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

    # If there are recent frames, ensure prefix does not overlap recent
    if state.recent and prefix_items:
        first_recent_id = state.recent[0].frame_id
        if last_end_id >= first_recent_id:
            raise ValueError(
                f"Prefix item end frame {last_end_id} overlaps or exceeds first recent frame {first_recent_id}"
            )

    embeds_parts = []
    mask_parts = []

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
            embeds_parts.append(item_embeds)
            mask_parts.append(item_mask)
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
            embeds_parts.append(item_embeds)
            mask_parts.append(item_mask)

    # Process recent frames
    for rf in state.recent:
        embeds_parts.append(rf.inputs_embeds.to(dtype=target_dtype, device=target_device))
        mask_parts.append(rf.attention_mask.to(device=target_device))

    inputs_embeds = torch.cat(embeds_parts, dim=1)
    attention_mask = torch.cat(mask_parts, dim=1)

    total_seq_len = inputs_embeds.shape[1]
    if state.recent:
        current_start = total_seq_len - state.recent[-1].inputs_embeds.shape[1]
    else:
        current_start = total_seq_len

    return inputs_embeds, attention_mask, current_start


def detached_state(state: PeriodicMemoryState) -> PeriodicMemoryState:
    """Return a new PeriodicMemoryState where all tensors are detached.

    Preserves dataclass immutability.
    """
    detached_recent = tuple(
        MemoryFrame(
            frame_id=f.frame_id,
            observation_time=f.observation_time,
            inputs_embeds=f.inputs_embeds.detach(),
            attention_mask=f.attention_mask.detach(),
            visual_tokens=f.visual_tokens.detach(),
            is_decision=f.is_decision,
            boundary_frame_id=f.boundary_frame_id,
        )
        for f in state.recent
    )
    detached_entries = tuple(
        MemoryEntry(
            visual_tokens=e.visual_tokens.detach(),
            start_frame_id=e.start_frame_id,
            end_frame_id=e.end_frame_id,
            start_time=e.start_time,
            end_time=e.end_time,
            count=e.count,
            boundary_frame_id=e.boundary_frame_id,
        )
        for e in state.entries
    )
    detached_anchors = tuple(
        MemoryFrame(
            frame_id=a.frame_id,
            observation_time=a.observation_time,
            inputs_embeds=a.inputs_embeds.detach(),
            attention_mask=a.attention_mask.detach(),
            visual_tokens=a.visual_tokens.detach(),
            is_decision=a.is_decision,
            boundary_frame_id=a.boundary_frame_id,
        )
        for a in state.anchors
    )
    return PeriodicMemoryState(
        recent=detached_recent,
        entries=detached_entries,
        frame_count=state.frame_count,
        consolidations=state.consolidations,
        merges=state.merges,
        anchors=detached_anchors,
        last_decision_frame_id=state.last_decision_frame_id,
    )


def state_nbytes(state: PeriodicMemoryState) -> int:
    """Calculate the total storage in bytes of tensors contained in the state.

    Avoids double-counting when tensors share underlying storage.
    """
    seen_storage_ptrs = set()
    total_bytes = 0

    def _add_tensor(t: torch.Tensor) -> None:
        nonlocal total_bytes
        if not isinstance(t, torch.Tensor):
            return
        try:
            storage = t.untyped_storage()
            ptr = storage.data_ptr()
            if ptr not in seen_storage_ptrs:
                seen_storage_ptrs.add(ptr)
                total_bytes += storage.nbytes()
        except Exception:
            total_bytes += t.numel() * t.element_size()

    for e in state.entries:
        _add_tensor(e.visual_tokens)

    for a in state.anchors:
        _add_tensor(a.inputs_embeds)
        _add_tensor(a.attention_mask)
        _add_tensor(a.visual_tokens)

    for f in state.recent:
        _add_tensor(f.inputs_embeds)
        _add_tensor(f.attention_mask)
        _add_tensor(f.visual_tokens)

    return total_bytes
