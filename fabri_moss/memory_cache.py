"""Periodic memory KV-cache management, adapter, and async validator for FabriVLA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from fabri_moss.native_cache import (
    NativeCacheAdapter,
    NativeCacheConfig,
    NativeEmbeddingBlock,
    extract_layer_kv,
    populate_cache_from_layer_kv,
)
from fabri_moss.periodic_memory import (
    MemoryFrame,
    PeriodicMemoryConfig,
    PeriodicMemoryState,
    advance_memory,
    detached_state as detached_memory_state,
    materialize_memory,
    state_nbytes as memory_state_nbytes,
)


@dataclass(frozen=True)
class NativeMemoryKVState:
    """Persistent native memory state holding layer KV and PeriodicMemoryState."""

    layer_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    memory: PeriodicMemoryState
    attention_mask: torch.Tensor
    last_frame_id: int
    frame_count: int
    prompt: str
    owner: Any
    revision: int
    rebuild_count: int

    def __post_init__(self) -> None:
        if self.frame_count < 0:
            raise ValueError(f"frame_count must be non-negative, got {self.frame_count}")
        if self.last_frame_id < -1:
            raise ValueError(f"last_frame_id must be >= -1, got {self.last_frame_id}")
        if self.rebuild_count < 0:
            raise ValueError(f"rebuild_count must be non-negative, got {self.rebuild_count}")
        if not isinstance(self.layer_kv, tuple) or not isinstance(self.memory, PeriodicMemoryState):
            raise TypeError("layer_kv must be tuple and memory must be PeriodicMemoryState")
        if self.attention_mask.ndim != 2 or self.attention_mask.shape[0] != 1:
            raise ValueError(f"attention_mask must have shape [1, total_len], got {tuple(self.attention_mask.shape)}")

    @property
    def blocks(self) -> Tuple[MemoryFrame, ...]:
        return self.memory.recent

    def detached(self) -> NativeMemoryKVState:
        return NativeMemoryKVState(
            layer_kv=tuple((k.detach(), v.detach()) for k, v in self.layer_kv),
            memory=detached_memory_state(self.memory),
            attention_mask=self.attention_mask.detach(),
            last_frame_id=self.last_frame_id,
            frame_count=self.frame_count,
            prompt=self.prompt,
            owner=self.owner,
            revision=self.revision,
            rebuild_count=self.rebuild_count,
        )

    @property
    def kv_nbytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.layer_kv)

    @property
    def nbytes(self) -> int:
        return self.kv_nbytes + memory_state_nbytes(self.memory) + (self.attention_mask.numel() * self.attention_mask.element_size())


class NativeMemoryCacheAdapter(NativeCacheAdapter):
    """Adapter wrapping FabriVLA policy with periodic memory KV cache."""

    def __init__(
        self,
        policy: nn.Module,
        config: Optional[NativeCacheConfig] = None,
        memory_config: Optional[PeriodicMemoryConfig] = None,
    ) -> None:
        super().__init__(policy=policy, config=config)
        self.memory_config = (
            memory_config
            if memory_config is not None
            else PeriodicMemoryConfig(protect_decision_frames=True)
        )
        if not self.config.use_timestamps:
            raise ValueError("Periodic memory requires real observation timestamps")

    @torch.no_grad()
    def encode_frame(
        self,
        images: List[Any],
        frame_id: int,
        prompt: str,
        image_mask: Optional[torch.Tensor] = None,
        capture_time: Optional[float] = None,
        observation_time: Optional[float] = None,
        preserve_visual_tokens: bool = True,
    ) -> NativeEmbeddingBlock:
        if not preserve_visual_tokens:
            raise ValueError("Periodic memory requires preserve_visual_tokens=True to store visual tokens")
        return super().encode_frame(
            images=images,
            frame_id=frame_id,
            prompt=prompt,
            image_mask=image_mask,
            capture_time=capture_time,
            observation_time=observation_time,
            preserve_visual_tokens=True,
        )

    def _validate_previous_memory_state(self, previous: NativeMemoryKVState, prompt: str) -> None:
        if not isinstance(previous, NativeMemoryKVState):
            raise TypeError(f"previous must be NativeMemoryKVState, got {type(previous)}")
        if previous.owner is not self or previous.revision != self._revision or previous.prompt != prompt:
            raise ValueError("previous owner/revision/prompt mismatch")
        if previous.frame_count != previous.memory.frame_count:
            raise ValueError(f"previous frame_count {previous.frame_count} != memory.frame_count {previous.memory.frame_count}")
        if previous.memory.last_frame_id is not None and previous.last_frame_id != previous.memory.last_frame_id:
            raise ValueError(f"previous last_frame_id {previous.last_frame_id} != memory.last_frame_id {previous.memory.last_frame_id}")
        num_layers = len(self.native_core.layers)
        if len(previous.layer_kv) != num_layers:
            raise ValueError(f"previous layer_kv count {len(previous.layer_kv)} != layers count {num_layers}")
        total_len = previous.attention_mask.shape[1]
        num_kv_heads, head_dim = self._get_kv_heads_and_dim()
        for idx, (k, v) in enumerate(previous.layer_kv):
            if k.shape != (1, num_kv_heads, total_len, head_dim) or v.shape != k.shape:
                raise ValueError(f"layer_kv[{idx}] shape mismatch with total_len {total_len}")
            if not torch.isfinite(k).all() or not torch.isfinite(v).all():
                raise ValueError(f"layer_kv[{idx}] contains non-finite values")

    @torch.no_grad()
    def read_blocks(
        self,
        new_blocks: Sequence[NativeEmbeddingBlock],
        prompt: str,
        previous: Optional[NativeMemoryKVState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, NativeMemoryKVState]:
        if self.training:
            raise RuntimeError("read_blocks is only allowed in eval mode.")
        if not new_blocks:
            raise ValueError("new_blocks must not be empty.")

        core = self.native_core
        core_param = next(core.parameters())
        core_dtype, core_device = core_param.dtype, core_param.device
        expected_seq_len = getattr(getattr(self.policy, "embedder", None), "max_text_length", 1024)

        for i, b in enumerate(new_blocks):
            if b.owner is not self or b.revision != self._revision or b.prompt != prompt:
                raise ValueError(f"Block[{i}] metadata mismatch with adapter/prompt")
            if b.inputs_embeds.dtype != core_dtype or b.inputs_embeds.device != core_device:
                raise ValueError(f"Block[{i}] inputs_embeds dtype/device must match core ({core_dtype}, {core_device})")
            if b.inputs_embeds.ndim != 3 or b.inputs_embeds.shape[0] != 1 or b.inputs_embeds.shape[1] != expected_seq_len:
                raise ValueError(f"Block[{i}] inputs_embeds shape mismatch: expected [1, {expected_seq_len}, H], got {tuple(b.inputs_embeds.shape)}")
            if b.attention_mask.ndim != 2 or b.attention_mask.shape[0] != 1 or b.attention_mask.shape[1] != expected_seq_len:
                raise ValueError(f"Block[{i}] attention_mask shape mismatch: expected [1, {expected_seq_len}], got {tuple(b.attention_mask.shape)}")
            if b.visual_tokens is None:
                raise ValueError(f"Block[{i}] must preserve visual_tokens for periodic memory")
            if b.visual_tokens.dtype != core_dtype or b.visual_tokens.device != core_device:
                raise ValueError(f"Block[{i}] visual_tokens dtype/device must match core ({core_dtype}, {core_device})")
            if i > 0 and b.frame_id <= new_blocks[i - 1].frame_id:
                raise ValueError("new_blocks frame_id must be strictly increasing")
            if self.config.use_timestamps and b.observation_time is None:
                raise ValueError("Block observation_time must not be None when use_timestamps=True")

        if previous is not None:
            self._validate_previous_memory_state(previous, prompt)
            if new_blocks[0].frame_id <= previous.last_frame_id:
                raise ValueError(f"new block frame_id {new_blocks[0].frame_id} <= previous {previous.last_frame_id}")

        all_times = ([previous.memory.last_observation_time] if previous is not None and previous.memory.last_observation_time is not None else []) + [b.observation_time for b in new_blocks if b.observation_time is not None]
        for i in range(1, len(all_times)):
            if all_times[i] < all_times[i - 1]:
                raise ValueError(f"observation_time must be non-decreasing: {all_times[i]} < {all_times[i - 1]}")

        mem_state = previous.memory if previous is not None else PeriodicMemoryState()
        layer_kv = previous.layer_kv if previous is not None else None
        att_mask = previous.attention_mask if previous is not None else None
        rebuild_count = previous.rebuild_count if previous is not None else 0
        frame_count = previous.frame_count if previous is not None else 0

        embedder = getattr(self.policy, "embedder", None)
        num_layers, shallow_idx = len(self.native_core.layers), self.config.shallow_layer
        deep, shallow = None, None

        for i, b in enumerate(new_blocks):
            obs_time = float(b.observation_time)
            is_dec = (i == len(new_blocks) - 1)
            mf = MemoryFrame(
                frame_id=b.frame_id,
                observation_time=obs_time,
                inputs_embeds=b.inputs_embeds,
                attention_mask=b.attention_mask,
                visual_tokens=b.visual_tokens,
                is_decision=is_dec,
            )
            old_c = mem_state.consolidations
            mem_state = advance_memory(mem_state, mf, self.memory_config)
            frame_count += 1
            cur_len = b.seq_len

            if layer_kv is None or mem_state.consolidations != old_c:
                if layer_kv is not None:
                    rebuild_count += 1
                p_embeds, p_mask, _ = materialize_memory(mem_state, embedder, self.memory_config)
                cache = DynamicCache()
                final_norm_h, inter_h = self._execute_native_layers(inputs_embeds=p_embeds, attention_mask_2d=p_mask, cache=cache, start_pos=0)
                layer_kv, att_mask = extract_layer_kv(cache), p_mask
            else:
                old_seq_len = att_mask.shape[1]
                cache = populate_cache_from_layer_kv(layer_kv)
                comb_mask = torch.cat([att_mask, b.attention_mask], dim=1)
                final_norm_h, inter_h = self._execute_native_layers(inputs_embeds=b.inputs_embeds, attention_mask_2d=comb_mask, cache=cache, start_pos=old_seq_len)
                layer_kv, att_mask = extract_layer_kv(cache), comb_mask

            deep = final_norm_h[:, -cur_len:, :].to(torch.float32)
            shallow = deep if shallow_idx == num_layers else inter_h[shallow_idx][:, -cur_len:, :].to(torch.float32)

        return deep, shallow, NativeMemoryKVState(
            layer_kv=layer_kv, memory=mem_state, attention_mask=att_mask, last_frame_id=new_blocks[-1].frame_id,
            frame_count=frame_count, prompt=prompt, owner=self, revision=self._revision, rebuild_count=rebuild_count,
        )


def validate_memory_cache(candidate: Any, previous: Optional[Any], snapshot: Tuple[Any, ...], prompt: str) -> NativeMemoryKVState:
    """Validate candidate NativeMemoryKVState against snapshot and previous state."""
    if candidate is None or not isinstance(candidate, NativeMemoryKVState):
        raise TypeError(f"candidate must be NativeMemoryKVState, got {type(candidate).__name__ if candidate else None}")
    if previous is not None and not isinstance(previous, NativeMemoryKVState):
        raise TypeError(f"previous must be NativeMemoryKVState, got {type(previous).__name__}")
    if not snapshot:
        raise ValueError("snapshot cannot be empty")

    owner = candidate.owner
    if not isinstance(owner, NativeMemoryCacheAdapter):
        raise TypeError(f"candidate owner must be NativeMemoryCacheAdapter, got {type(owner).__name__}")

    core = owner.native_core
    core_dtype = next(core.parameters()).dtype
    core_device = next(core.parameters()).device
    expected_seq_len = getattr(getattr(owner.policy, "embedder", None), "max_text_length", 1024)

    for idx, frame in enumerate(snapshot):
        blk = getattr(frame, "payload", frame)
        if not isinstance(blk, NativeEmbeddingBlock):
            raise TypeError(f"snapshot[{idx}] payload must be NativeEmbeddingBlock")
        if blk.owner is not owner:
            raise ValueError(f"snapshot[{idx}] payload owner {blk.owner} != candidate owner {owner}")
        if blk.revision != owner.revision or blk.prompt != prompt:
            raise ValueError(f"snapshot[{idx}] payload revision or prompt mismatch")
        if blk.inputs_embeds.dtype != core_dtype:
            raise TypeError(f"snapshot[{idx}] inputs_embeds dtype {blk.inputs_embeds.dtype} != core dtype {core_dtype}")
        if blk.inputs_embeds.device != core_device:
            raise ValueError(f"snapshot[{idx}] inputs_embeds device {blk.inputs_embeds.device} != core device {core_device}")
        if blk.inputs_embeds.ndim != 3 or blk.inputs_embeds.shape[0] != 1 or blk.inputs_embeds.shape[1] != expected_seq_len:
            raise ValueError(f"snapshot[{idx}] inputs_embeds shape mismatch: expected [1, {expected_seq_len}, H], got {tuple(blk.inputs_embeds.shape)}")
        if blk.attention_mask.ndim != 2 or blk.attention_mask.shape[0] != 1 or blk.attention_mask.shape[1] != expected_seq_len:
            raise ValueError(f"snapshot[{idx}] attention_mask shape mismatch: expected [1, {expected_seq_len}], got {tuple(blk.attention_mask.shape)}")
        if blk.visual_tokens is None:
            raise ValueError(f"snapshot[{idx}] payload must have visual_tokens")
        if blk.visual_tokens.dtype != core_dtype:
            raise TypeError(f"snapshot[{idx}] visual_tokens dtype {blk.visual_tokens.dtype} != core dtype {core_dtype}")
        if blk.visual_tokens.device != core_device:
            raise ValueError(f"snapshot[{idx}] visual_tokens device {blk.visual_tokens.device} != core device {core_device}")
        if blk.observation_time is None:
            raise ValueError("Periodic memory snapshot requires observation_time")
        obs = getattr(frame, "observation", None)
        if obs is not None and (blk.frame_id, blk.observation_time, blk.capture_time) != (obs.frame_id, obs.observation_time, obs.capture_time):
            raise ValueError(f"snapshot[{idx}] payload frame/time mismatch with observation")
        if idx > 0:
            prev_blk = getattr(snapshot[idx - 1], "payload", snapshot[idx - 1])
            if blk.frame_id <= prev_blk.frame_id:
                raise ValueError("snapshot payload frame IDs must be strictly increasing")

    if candidate.revision != owner.revision or candidate.prompt != prompt:
        raise ValueError("candidate revision or prompt mismatch")

    if previous is not None:
        if previous.owner is not owner or previous.revision != owner.revision or previous.prompt != prompt:
            raise ValueError("previous state owner/revision/prompt mismatch")
        if previous.frame_count != previous.memory.frame_count:
            raise ValueError(f"previous frame_count {previous.frame_count} != memory.frame_count {previous.memory.frame_count}")
        if previous.memory.last_frame_id is not None and previous.last_frame_id != previous.memory.last_frame_id:
            raise ValueError(f"previous last_frame_id {previous.last_frame_id} != memory.last_frame_id {previous.memory.last_frame_id}")
        first_snap_id = getattr(snapshot[0], "payload", snapshot[0]).frame_id
        if first_snap_id <= previous.last_frame_id:
            raise ValueError(f"snapshot first frame_id {first_snap_id} <= previous {previous.last_frame_id}")

    expected_frame_count = (previous.frame_count if previous is not None else 0) + len(snapshot)
    if candidate.frame_count != expected_frame_count:
        raise ValueError(f"candidate frame_count {candidate.frame_count} != expected {expected_frame_count}")

    last_snap_id = getattr(snapshot[-1], "payload", snapshot[-1]).frame_id
    if candidate.last_frame_id != last_snap_id:
        raise ValueError(f"candidate last_frame_id {candidate.last_frame_id} != snapshot {last_snap_id}")

    with torch.no_grad():
        expected_mem = previous.memory if previous is not None else PeriodicMemoryState()
        expected_rebuild = previous.rebuild_count if previous is not None else 0
        had_layer_kv = previous is not None

        for i, frame in enumerate(snapshot):
            blk = getattr(frame, "payload", frame)
            obs_time = float(blk.observation_time)
            is_dec = (i == len(snapshot) - 1)
            mf = MemoryFrame(
                frame_id=blk.frame_id,
                observation_time=obs_time,
                inputs_embeds=blk.inputs_embeds,
                attention_mask=blk.attention_mask,
                visual_tokens=blk.visual_tokens,
                is_decision=is_dec,
            )
            old_c = expected_mem.consolidations
            expected_mem = advance_memory(expected_mem, mf, owner.memory_config)
            if not had_layer_kv or expected_mem.consolidations != old_c:
                if had_layer_kv:
                    expected_rebuild += 1
                had_layer_kv = True

    if candidate.rebuild_count != expected_rebuild:
        raise ValueError(f"candidate rebuild_count {candidate.rebuild_count} != expected {expected_rebuild}")

    cand_mem = candidate.memory
    if (cand_mem.frame_count, cand_mem.consolidations, cand_mem.merges) != (expected_mem.frame_count, expected_mem.consolidations, expected_mem.merges):
        raise ValueError("candidate memory frame_count/consolidations/merges mismatch")
    if len(cand_mem.entries) != len(expected_mem.entries) or len(cand_mem.recent) != len(expected_mem.recent) or len(cand_mem.anchors) != len(expected_mem.anchors):
        raise ValueError("candidate memory entries, recent, or anchors count mismatch")

    if not owner.memory_config.protect_decision_frames:
        if len(cand_mem.entries) > owner.memory_config.memory_slots:
            raise ValueError(f"entries count {len(cand_mem.entries)} exceeds slots {owner.memory_config.memory_slots}")
    else:
        total_items_count = sum(e.count for e in cand_mem.entries) + len(cand_mem.anchors) + len(cand_mem.recent)
        if total_items_count != expected_frame_count:
            raise ValueError(
                f"expected state counts sum entries count ({sum(e.count for e in cand_mem.entries)}) + "
                f"len(anchors) ({len(cand_mem.anchors)}) + len(recent) ({len(cand_mem.recent)}) = "
                f"{total_items_count} != expected_frame_count {expected_frame_count}"
            )
        if cand_mem.last_decision_frame_id != expected_mem.last_decision_frame_id:
            raise ValueError(
                f"candidate last_decision_frame_id {cand_mem.last_decision_frame_id} != expected {expected_mem.last_decision_frame_id}"
            )

    # Validate anchors
    for i, (c_a, e_a) in enumerate(zip(cand_mem.anchors, expected_mem.anchors)):
        if owner.memory_config.protect_decision_frames:
            if not c_a.is_decision:
                raise ValueError(f"anchor[{i}] is_decision must be True")
            if c_a.is_decision != e_a.is_decision:
                raise ValueError(f"anchor[{i}] is_decision {c_a.is_decision} != expected {e_a.is_decision}")
            if c_a.boundary_frame_id != e_a.boundary_frame_id:
                raise ValueError(f"anchor[{i}] boundary_frame_id {c_a.boundary_frame_id} != expected {e_a.boundary_frame_id}")
        if (c_a.frame_id, c_a.observation_time) != (e_a.frame_id, e_a.observation_time):
            raise ValueError(f"anchor[{i}] metadata mismatch")
        if c_a.visual_tokens.shape != e_a.visual_tokens.shape:
            raise ValueError(f"anchor[{i}] visual_tokens shape {tuple(c_a.visual_tokens.shape)} != expected {tuple(e_a.visual_tokens.shape)}")
        if c_a.visual_tokens.dtype != e_a.visual_tokens.dtype:
            raise TypeError(f"anchor[{i}] visual_tokens dtype {c_a.visual_tokens.dtype} != expected {e_a.visual_tokens.dtype}")
        if c_a.visual_tokens.device != e_a.visual_tokens.device:
            raise ValueError(f"anchor[{i}] visual_tokens device {c_a.visual_tokens.device} != expected {e_a.visual_tokens.device}")
        if not torch.equal(c_a.visual_tokens, e_a.visual_tokens):
            raise ValueError(f"anchor[{i}] visual_tokens content mismatch with expected recomputed visual_tokens")
        if not torch.equal(c_a.inputs_embeds, e_a.inputs_embeds):
            raise ValueError(f"anchor[{i}] inputs_embeds content mismatch with expected")
        if not torch.equal(c_a.attention_mask, e_a.attention_mask):
            raise ValueError(f"anchor[{i}] attention_mask content mismatch with expected")

    # Validate recent frames
    for i, (c_rf, e_rf) in enumerate(zip(cand_mem.recent, expected_mem.recent)):
        if (c_rf.frame_id, c_rf.observation_time) != (e_rf.frame_id, e_rf.observation_time):
            raise ValueError(f"recent frame[{i}] metadata mismatch")
        if owner.memory_config.protect_decision_frames:
            if c_rf.is_decision != e_rf.is_decision:
                raise ValueError(f"recent frame[{i}] is_decision {c_rf.is_decision} != expected {e_rf.is_decision}")
            if c_rf.boundary_frame_id != e_rf.boundary_frame_id:
                raise ValueError(f"recent frame[{i}] boundary_frame_id {c_rf.boundary_frame_id} != expected {e_rf.boundary_frame_id}")
        if c_rf.inputs_embeds.data_ptr() != e_rf.inputs_embeds.data_ptr() or c_rf.visual_tokens.data_ptr() != e_rf.visual_tokens.data_ptr():
            raise ValueError(f"recent frame[{i}] storage identity preserved mismatch")
        if c_rf.inputs_embeds.shape != e_rf.inputs_embeds.shape or c_rf.inputs_embeds.stride() != e_rf.inputs_embeds.stride() or c_rf.inputs_embeds.storage_offset() != e_rf.inputs_embeds.storage_offset():
            raise ValueError(f"recent frame[{i}] inputs_embeds shape/stride/offset mismatch")
        if c_rf.visual_tokens.shape != e_rf.visual_tokens.shape or c_rf.visual_tokens.stride() != e_rf.visual_tokens.stride() or c_rf.visual_tokens.storage_offset() != e_rf.visual_tokens.storage_offset():
            raise ValueError(f"recent frame[{i}] visual_tokens shape/stride/offset mismatch")
        if not torch.equal(c_rf.attention_mask, e_rf.attention_mask):
            raise ValueError(f"recent frame[{i}] attention_mask content mismatch")

    if cand_mem.recent:
        latest_cand_frame = cand_mem.recent[-1]
        latest_snap_blk = getattr(snapshot[-1], "payload", snapshot[-1])
        if latest_cand_frame.frame_id != latest_snap_blk.frame_id:
            raise ValueError(f"latest recent frame_id {latest_cand_frame.frame_id} != latest snapshot block {latest_snap_blk.frame_id}")
        if latest_cand_frame.visual_tokens.shape != latest_snap_blk.visual_tokens.shape:
            raise ValueError(f"latest frame visual_tokens shape mismatch: {tuple(latest_cand_frame.visual_tokens.shape)} != {tuple(latest_snap_blk.visual_tokens.shape)}")

    # Validate memory entries
    for i, (c_e, e_e) in enumerate(zip(cand_mem.entries, expected_mem.entries)):
        if owner.memory_config.protect_decision_frames:
            if c_e.boundary_frame_id != e_e.boundary_frame_id:
                raise ValueError(
                    f"memory entry[{i}] boundary_frame_id {c_e.boundary_frame_id} != expected {e_e.boundary_frame_id}"
                )
            for a in cand_mem.anchors:
                if c_e.start_frame_id <= a.frame_id <= c_e.end_frame_id:
                    raise ValueError(
                        f"memory entry[{i}] contains anchor frame {a.frame_id} in range [{c_e.start_frame_id}, {c_e.end_frame_id}]"
                    )
        if (c_e.start_frame_id, c_e.end_frame_id, c_e.start_time, c_e.end_time, c_e.count) != (e_e.start_frame_id, e_e.end_frame_id, e_e.start_time, e_e.end_time, e_e.count):
            raise ValueError(f"memory entry[{i}] metadata mismatch")
        if c_e.visual_tokens.shape != e_e.visual_tokens.shape:
            raise ValueError(f"memory entry[{i}] visual_tokens shape mismatch")
        if c_e.visual_tokens.dtype != e_e.visual_tokens.dtype:
            raise TypeError(f"memory entry[{i}] visual_tokens dtype mismatch")
        if c_e.visual_tokens.device != e_e.visual_tokens.device:
            raise ValueError(f"memory entry[{i}] visual_tokens device mismatch")
        if not torch.equal(c_e.visual_tokens, e_e.visual_tokens):
            raise ValueError(f"memory entry[{i}] visual_tokens content mismatch")
        if c_e.start_time > c_e.end_time:
            raise ValueError(f"memory entry[{i}] start_time {c_e.start_time} > end_time {c_e.end_time}")

    with torch.no_grad():
        _, exp_mask, _ = materialize_memory(expected_mem, owner.policy.embedder, owner.memory_config)
    if candidate.attention_mask.shape != exp_mask.shape or not torch.equal(candidate.attention_mask, exp_mask):
        raise ValueError("candidate attention_mask does not match expected materialize mask")

    native_core = owner.native_core
    layers = getattr(native_core, "layers", None)
    if layers is None or len(candidate.layer_kv) != len(layers):
        raise ValueError("layer_kv count does not match native_core layers")

    core_param = next(native_core.parameters())
    num_kv_heads, head_dim = owner._get_kv_heads_and_dim()
    expected_kv_shape = (1, num_kv_heads, exp_mask.shape[1], head_dim)

    for idx, (k, v) in enumerate(candidate.layer_kv):
        if k.shape != expected_kv_shape or v.shape != expected_kv_shape:
            raise ValueError(f"layer_kv[{idx}] shape mismatch: expected {expected_kv_shape}, got {tuple(k.shape)}")
        if k.dtype != core_param.dtype or v.dtype != core_param.dtype:
            raise TypeError(f"layer_kv[{idx}] dtype mismatch with core")
        if k.device != core_param.device or v.device != core_param.device:
            raise ValueError(f"layer_kv[{idx}] device mismatch with core")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError(f"layer_kv[{idx}] contains non-finite values")

    return candidate
