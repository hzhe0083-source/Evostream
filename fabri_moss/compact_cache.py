"""Compact intermediate memory KV-cache management, adapter, and async validator for FabriVLA."""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
from dataclasses import dataclass
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from fabri_moss.compact_memory import (
    CompactMemoryConfig,
    render_compact_frames,
    render_compact_memory,
)
from fabri_moss.memory_cache import NativeMemoryCacheAdapter
from fabri_moss.native_cache import (
    NativeCacheConfig,
    NativeEmbeddingBlock,
    execute_native_layers,
    extract_layer_kv,
    populate_cache_from_layer_kv,
)
from fabri_moss.periodic_memory import (
    MemoryFrame,
    PeriodicMemoryConfig,
    PeriodicMemoryState,
    append_memory_frame,
    consolidate_memory,
    detached_state as detached_memory_state,
    state_nbytes as memory_state_nbytes,
)


@dataclass(frozen=True)
class CompactRebuildPayload:
    """Canonical materialized prefix produced by the background consolidation worker."""

    memory: PeriodicMemoryState
    layer_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor
    token_times: torch.Tensor
    owner: Any
    revision: int
    prompt: str
    episode_token: Any
    raw_version_token: Any
    frame_count: int
    latest_frame_id: int
    rebuild_count: int
    config: CompactMemoryConfig


@dataclass(frozen=True)
class PendingRebuild:
    """Handle linking a CompactKVState to its asynchronous or immediate rebuild task."""

    future: concurrent.futures.Future[CompactRebuildPayload]
    owner: Any
    revision: int
    prompt: str
    episode_token: Any
    raw_version_token: Any
    frame_count: int
    latest_frame_id: int
    config: CompactMemoryConfig


@dataclass(frozen=True)
class CompactKVState:
    """Frozen state holding KV, raw PeriodicMemoryState, temporal masks, and version tokens."""

    layer_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    memory: PeriodicMemoryState
    attention_mask: torch.Tensor
    token_times: torch.Tensor
    last_frame_id: int
    frame_count: int
    prompt: str
    owner: Any
    revision: int
    rebuild_count: int
    episode_token: Any
    version_token: Any
    parent_version_token: Any
    pending: Optional[PendingRebuild] = None
    config: Optional[CompactMemoryConfig] = None

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
        if self.token_times.ndim != 2 or self.token_times.shape != self.attention_mask.shape:
            raise ValueError(
                f"token_times must have shape {tuple(self.attention_mask.shape)}, got {tuple(self.token_times.shape)}"
            )
        if self.episode_token is None or self.version_token is None:
            raise ValueError("episode_token and version_token must not be None")
        if self.pending is not None and not isinstance(self.pending, PendingRebuild):
            raise TypeError("pending must be an instance of PendingRebuild or None")
        if self.config is not None and not isinstance(self.config, CompactMemoryConfig):
            raise TypeError(f"config must be CompactMemoryConfig or None, got {type(self.config).__name__}")

    @property
    def blocks(self) -> Tuple[MemoryFrame, ...]:
        """Expose recent frames as blocks for compatibility with callbacks and pipelines."""
        return self.memory.recent

    def detached(self) -> CompactKVState:
        """Return a detached copy preserving tokens, pending handle, and detaching tensors."""
        detached_kv = tuple((k.detach(), v.detach()) for k, v in self.layer_kv)
        detached_mem = detached_memory_state(self.memory)
        return CompactKVState(
            layer_kv=detached_kv,
            memory=detached_mem,
            attention_mask=self.attention_mask.detach(),
            token_times=self.token_times.detach(),
            last_frame_id=self.last_frame_id,
            frame_count=self.frame_count,
            prompt=self.prompt,
            owner=self.owner,
            revision=self.revision,
            rebuild_count=self.rebuild_count,
            episode_token=self.episode_token,
            version_token=self.version_token,
            parent_version_token=self.parent_version_token,
            pending=self.pending,
            config=self.config,
        )

    @property
    def kv_nbytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.layer_kv)

    @property
    def nbytes(self) -> int:
        """Total memory in bytes strictly for materialized tensors held in this state.

        Does not block pending future.
        """
        return (
            self.kv_nbytes
            + memory_state_nbytes(self.memory)
            + (self.attention_mask.numel() * self.attention_mask.element_size())
            + (self.token_times.numel() * self.token_times.element_size())
        )

    def record_stream(self, stream: Any) -> None:
        """Recursively record CUDA stream for all tensors in this state."""
        if stream is None:
            return
        for k, v in self.layer_kv:
            k.record_stream(stream)
            v.record_stream(stream)
        if hasattr(self.attention_mask, "record_stream"):
            self.attention_mask.record_stream(stream)
        if hasattr(self.token_times, "record_stream"):
            self.token_times.record_stream(stream)
        for anchor in self.memory.anchors:
            if hasattr(anchor.inputs_embeds, "record_stream"):
                anchor.inputs_embeds.record_stream(stream)
            if hasattr(anchor.attention_mask, "record_stream"):
                anchor.attention_mask.record_stream(stream)
            if hasattr(anchor.visual_tokens, "record_stream"):
                anchor.visual_tokens.record_stream(stream)
        for entry in self.memory.entries:
            if hasattr(entry.visual_tokens, "record_stream"):
                entry.visual_tokens.record_stream(stream)
        for rf in self.memory.recent:
            if hasattr(rf.inputs_embeds, "record_stream"):
                rf.inputs_embeds.record_stream(stream)
            if hasattr(rf.attention_mask, "record_stream"):
                rf.attention_mask.record_stream(stream)
            if hasattr(rf.visual_tokens, "record_stream"):
                rf.visual_tokens.record_stream(stream)


def _validate_memory_equal(cand_mem: PeriodicMemoryState, exp_mem: PeriodicMemoryState) -> None:
    """Strictly compare actual candidate memory against expected reconstructed memory.

    Validates:
    - Counts and scalar metadata: frame_count, consolidations, merges, last_decision_frame_id
    - Entries, recent, and anchors lengths
    - Anchors: frame_id, observation_time, is_decision, boundary_frame_id, tensor shapes, dtypes, devices, and exact values
    - Recent frames: frame_id, observation_time, is_decision, boundary_frame_id, tensor shapes, dtypes, devices, exact values, and storage offset/stride
    - Memory entries: start/end frame IDs, start/end times, count, boundary_frame_id, tensor shapes, dtypes, devices, exact values
    """
    if (cand_mem.frame_count, cand_mem.consolidations, cand_mem.merges) != (
        exp_mem.frame_count,
        exp_mem.consolidations,
        exp_mem.merges,
    ):
        raise ValueError("candidate raw memory frame_count/consolidations/merges mismatch")
    if (
        len(cand_mem.entries) != len(exp_mem.entries)
        or len(cand_mem.recent) != len(exp_mem.recent)
        or len(cand_mem.anchors) != len(exp_mem.anchors)
    ):
        raise ValueError("candidate raw memory entries, recent, or anchors count mismatch")
    if cand_mem.last_decision_frame_id != exp_mem.last_decision_frame_id:
        raise ValueError(
            f"candidate last_decision_frame_id {cand_mem.last_decision_frame_id} != expected {exp_mem.last_decision_frame_id}"
        )

    # Validate anchors
    for i, (c_a, e_a) in enumerate(zip(cand_mem.anchors, exp_mem.anchors)):
        if (c_a.frame_id, c_a.observation_time) != (e_a.frame_id, e_a.observation_time):
            raise ValueError(f"anchor[{i}] metadata mismatch (frame_id or observation_time)")
        if c_a.is_decision != e_a.is_decision:
            raise ValueError(f"anchor[{i}] is_decision {c_a.is_decision} != expected {e_a.is_decision}")
        if c_a.boundary_frame_id != e_a.boundary_frame_id:
            raise ValueError(f"anchor[{i}] boundary_frame_id {c_a.boundary_frame_id} != expected {e_a.boundary_frame_id}")
        if c_a.visual_tokens.shape != e_a.visual_tokens.shape:
            raise ValueError(f"anchor[{i}] visual_tokens shape {tuple(c_a.visual_tokens.shape)} != expected {tuple(e_a.visual_tokens.shape)}")
        if c_a.visual_tokens.dtype != e_a.visual_tokens.dtype or c_a.visual_tokens.device != e_a.visual_tokens.device:
            raise TypeError(f"anchor[{i}] visual_tokens dtype/device mismatch")
        if not torch.equal(c_a.visual_tokens, e_a.visual_tokens):
            raise ValueError(f"anchor[{i}] visual_tokens content mismatch")
        if not torch.equal(c_a.inputs_embeds, e_a.inputs_embeds):
            raise ValueError(f"anchor[{i}] inputs_embeds content mismatch")
        if not torch.equal(c_a.attention_mask, e_a.attention_mask):
            raise ValueError(f"anchor[{i}] attention_mask content mismatch")

    # Validate recent frames
    for i, (c_rf, e_rf) in enumerate(zip(cand_mem.recent, exp_mem.recent)):
        if (c_rf.frame_id, c_rf.observation_time) != (e_rf.frame_id, e_rf.observation_time):
            raise ValueError(f"recent frame[{i}] metadata mismatch (frame_id or observation_time)")
        if c_rf.is_decision != e_rf.is_decision:
            raise ValueError(f"recent frame[{i}] is_decision {c_rf.is_decision} != expected {e_rf.is_decision}")
        if c_rf.boundary_frame_id != e_rf.boundary_frame_id:
            raise ValueError(f"recent frame[{i}] boundary_frame_id {c_rf.boundary_frame_id} != expected {e_rf.boundary_frame_id}")
        if c_rf.inputs_embeds.shape != e_rf.inputs_embeds.shape or c_rf.inputs_embeds.dtype != e_rf.inputs_embeds.dtype or c_rf.inputs_embeds.device != e_rf.inputs_embeds.device:
            raise ValueError(f"recent frame[{i}] inputs_embeds shape/dtype/device mismatch")
        if c_rf.inputs_embeds.stride() != e_rf.inputs_embeds.stride() or c_rf.inputs_embeds.storage_offset() != e_rf.inputs_embeds.storage_offset():
            raise ValueError(f"recent frame[{i}] inputs_embeds stride/storage_offset mismatch")
        if not torch.equal(c_rf.inputs_embeds, e_rf.inputs_embeds):
            raise ValueError(f"recent frame[{i}] inputs_embeds content mismatch")
        if not torch.equal(c_rf.attention_mask, e_rf.attention_mask):
            raise ValueError(f"recent frame[{i}] attention_mask content mismatch")
        if c_rf.visual_tokens.shape != e_rf.visual_tokens.shape or c_rf.visual_tokens.dtype != e_rf.visual_tokens.dtype or c_rf.visual_tokens.device != e_rf.visual_tokens.device:
            raise ValueError(f"recent frame[{i}] visual_tokens shape/dtype/device mismatch")
        if c_rf.visual_tokens.stride() != e_rf.visual_tokens.stride() or c_rf.visual_tokens.storage_offset() != e_rf.visual_tokens.storage_offset():
            raise ValueError(f"recent frame[{i}] visual_tokens stride/storage_offset mismatch")
        if not torch.equal(c_rf.visual_tokens, e_rf.visual_tokens):
            raise ValueError(f"recent frame[{i}] visual_tokens content mismatch")

    # Validate memory entries
    for i, (c_e, e_e) in enumerate(zip(cand_mem.entries, exp_mem.entries)):
        if (c_e.start_frame_id, c_e.end_frame_id, c_e.start_time, c_e.end_time, c_e.count) != (
            e_e.start_frame_id,
            e_e.end_frame_id,
            e_e.start_time,
            e_e.end_time,
            e_e.count,
        ):
            raise ValueError(f"memory entry[{i}] metadata mismatch")
        if c_e.boundary_frame_id != e_e.boundary_frame_id:
            raise ValueError(f"memory entry[{i}] boundary_frame_id {c_e.boundary_frame_id} != expected {e_e.boundary_frame_id}")
        if c_e.visual_tokens.shape != e_e.visual_tokens.shape or c_e.visual_tokens.dtype != e_e.visual_tokens.dtype or c_e.visual_tokens.device != e_e.visual_tokens.device:
            raise ValueError(f"memory entry[{i}] visual_tokens shape/dtype/device mismatch")
        if not torch.equal(c_e.visual_tokens, e_e.visual_tokens):
            raise ValueError(f"memory entry[{i}] visual_tokens content mismatch")
        if c_e.start_time > c_e.end_time:
            raise ValueError(f"memory entry[{i}] start_time {c_e.start_time} > end_time {c_e.end_time}")


class NativeCompactMemoryCacheAdapter(NativeMemoryCacheAdapter):
    """Adapter wrapping FabriVLA with compact intermediate observations and async rebuild."""

    def __init__(
        self,
        policy: nn.Module,
        config: Optional[NativeCacheConfig] = None,
        compact_config: Optional[CompactMemoryConfig] = None,
        background_rebuild: bool = True,
    ) -> None:
        # Construct parent NativeMemoryCacheAdapter
        mem_cfg = compact_config.memory if compact_config is not None else PeriodicMemoryConfig(protect_decision_frames=True)
        super().__init__(policy=policy, config=config, memory_config=mem_cfg)

        self.compact_config = compact_config if compact_config is not None else CompactMemoryConfig()
        if not self.compact_config.memory.protect_decision_frames:
            raise ValueError("Compact memory requires protect_decision_frames=True")
        if type(background_rebuild) is not bool:
            raise TypeError(f"background_rebuild must be a bool, got {type(background_rebuild).__name__}")
        self.background_rebuild = background_rebuild

        # Dedicated render embedders with isolated tokenizer clones for foreground and background rebuild
        embedder = getattr(self.policy, "embedder", None)
        if embedder is not None:
            model = getattr(embedder, "model", None)
            source_tok = getattr(embedder, "tokenizer", None)
            try:
                render_tok = copy.deepcopy(source_tok) if source_tok is not None else None
                rebuild_tok = copy.deepcopy(source_tok) if source_tok is not None else None
            except Exception as e:
                raise RuntimeError(f"Failed to clone tokenizer for compact memory renderers: {e}") from e
            self._render_embedder = SimpleNamespace(model=model, tokenizer=render_tok)
            self._rebuild_embedder = SimpleNamespace(model=model, tokenizer=rebuild_tok)
        else:
            self._render_embedder = None
            self._rebuild_embedder = None

        # Lazy single-worker executor and dedicated CUDA stream
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._cuda_stream: Optional[Any] = None
        self._active_future: Optional[concurrent.futures.Future[CompactRebuildPayload]] = None
        self._lock = threading.Lock()
        self._closed = False

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._closed:
            raise RuntimeError("NativeCompactMemoryCacheAdapter is closed.")
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="compact_rebuild")
        return self._executor

    def _execute_native_layers(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        cache: DynamicCache,
        start_pos: int,
        token_times: Optional[torch.Tensor] = None,
        temporal_config: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        return execute_native_layers(
            core=self.native_core,
            inputs_embeds=inputs_embeds,
            attention_mask_2d=attention_mask_2d,
            cache=cache,
            start_pos=start_pos,
            gradient_checkpointing=False,
            token_times=token_times,
            temporal_config=temporal_config,
        )

    def _get_cuda_stream(self, device: torch.device) -> Optional[Any]:
        if device.type == "cuda":
            if self._cuda_stream is None:
                self._cuda_stream = torch.cuda.Stream(device=device)
            return self._cuda_stream
        return None

    def close(self) -> None:
        """Cancel queued rebuild tasks, wait for running ones, and release executor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._active_future is not None:
                self._active_future.cancel()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self._cuda_stream = None
            self._active_future = None

    def __enter__(self) -> NativeCompactMemoryCacheAdapter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _validate_compact_state(self, previous: CompactKVState, prompt: str) -> None:
        if not isinstance(previous, CompactKVState):
            raise TypeError(f"previous must be CompactKVState, got {type(previous)}")
        if previous.owner is not self or previous.revision != self._revision or previous.prompt != prompt:
            raise ValueError("previous owner/revision/prompt mismatch")
        if previous.config != self.compact_config:
            raise ValueError("previous config mismatch with adapter compact_config")
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

    def _resolve_pending(self, previous: CompactKVState, prompt: str) -> Tuple[PeriodicMemoryState, Tuple[Tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor, torch.Tensor, int, int]:
        """Resolve previous.pending if present, validating identity, shape, counts, and revision."""
        if previous.pending is None:
            return (
                previous.memory,
                previous.layer_kv,
                previous.attention_mask,
                previous.token_times,
                previous.rebuild_count,
                previous.frame_count,
            )

        pending = previous.pending
        # Validate pending descriptor matches previous state
        if pending.owner is not self or pending.revision != self._revision or pending.prompt != prompt:
            raise ValueError("Pending rebuild owner/revision/prompt mismatch with current adapter")
        if pending.episode_token is not previous.episode_token or pending.raw_version_token is not previous.version_token:
            raise ValueError("Pending rebuild episode_token or raw_version_token mismatch with previous state")
        if pending.frame_count != previous.frame_count or pending.latest_frame_id != previous.last_frame_id:
            raise ValueError("Pending rebuild frame_count or latest_frame_id mismatch with previous state")
        if pending.config != self.compact_config:
            raise ValueError("Pending rebuild config mismatch with adapter compact_config")

        # Wait / retrieve payload from future (exceptions will bubble up)
        payload = pending.future.result()

        # Deep verification of payload
        if not isinstance(payload, CompactRebuildPayload):
            raise TypeError(f"Rebuild worker returned invalid payload type: {type(payload)}")
        if payload.owner is not self or payload.revision != self._revision or payload.prompt != prompt:
            raise ValueError("Payload owner/revision/prompt mismatch")
        if payload.episode_token is not previous.episode_token or payload.raw_version_token is not previous.version_token:
            raise ValueError("Payload episode_token or raw_version_token mismatch with previous state")
        if payload.frame_count != previous.frame_count or payload.latest_frame_id != previous.last_frame_id:
            raise ValueError("Payload frame_count or latest_frame_id mismatch with previous state")
        if payload.config != self.compact_config:
            raise ValueError("Payload config mismatch with adapter compact_config")
        if payload.rebuild_count != previous.rebuild_count + 1:
            raise ValueError(f"Payload rebuild_count {payload.rebuild_count} != expected {previous.rebuild_count + 1}")

        # Verify consolidated memory matches expected
        expected_mem = consolidate_memory(previous.memory, self.compact_config.memory)
        _validate_memory_equal(payload.memory, expected_mem)

        # Render expected memory to check attention_mask and token_times content, dtype, device
        embedder = self._render_embedder if self._render_embedder is not None else getattr(self.policy, "embedder", None)
        rendered = render_compact_memory(expected_mem, embedder, self.compact_config)
        if payload.attention_mask.dtype != rendered.attention_mask.dtype or payload.attention_mask.device != rendered.attention_mask.device:
            raise TypeError("Payload attention_mask dtype or device mismatch")
        if not torch.equal(payload.attention_mask, rendered.attention_mask):
            raise ValueError("Payload attention_mask content mismatch with rendered memory")
        if payload.token_times.dtype != rendered.token_times.dtype or payload.token_times.device != rendered.token_times.device:
            raise TypeError("Payload token_times dtype or device mismatch")
        if not torch.equal(payload.token_times, rendered.token_times):
            raise ValueError("Payload token_times content mismatch with rendered memory")

        # Verify layer_kv shape, dtype, device, and finite
        core_param = next(self.native_core.parameters())
        core_dtype, core_device = core_param.dtype, core_param.device
        num_layers = len(self.native_core.layers)
        if len(payload.layer_kv) != num_layers:
            raise ValueError(f"Payload layer_kv count {len(payload.layer_kv)} != layers count {num_layers}")
        total_len = payload.attention_mask.shape[1]
        if payload.token_times.shape != (1, total_len):
            raise ValueError("Payload token_times shape mismatch with attention_mask")
        num_kv_heads, head_dim = self._get_kv_heads_and_dim()
        for idx, (k, v) in enumerate(payload.layer_kv):
            if k.shape != (1, num_kv_heads, total_len, head_dim) or v.shape != k.shape:
                raise ValueError(f"Payload layer_kv[{idx}] shape mismatch with total_len {total_len}")
            if k.dtype != core_dtype or v.dtype != core_dtype:
                raise TypeError(f"Payload layer_kv[{idx}] dtype mismatch with core ({core_dtype})")
            if k.device != core_device or v.device != core_device:
                raise ValueError(f"Payload layer_kv[{idx}] device mismatch with core ({core_device})")
            if not torch.isfinite(k).all() or not torch.isfinite(v).all():
                raise ValueError(f"Payload layer_kv[{idx}] contains non-finite values")

        # If on CUDA, record payload tensors to current stream on core_device
        if core_device.type == "cuda" and torch.cuda.is_available():
            curr_stream = torch.cuda.current_stream(core_device)
            temp_state = CompactKVState(
                layer_kv=payload.layer_kv,
                memory=payload.memory,
                attention_mask=payload.attention_mask,
                token_times=payload.token_times,
                last_frame_id=payload.latest_frame_id,
                frame_count=payload.frame_count,
                prompt=payload.prompt,
                owner=payload.owner,
                revision=payload.revision,
                rebuild_count=payload.rebuild_count,
                episode_token=payload.episode_token,
                version_token=payload.raw_version_token,
                parent_version_token=None,
                config=payload.config,
            )
            temp_state.record_stream(curr_stream)

        with self._lock:
            if self._active_future is pending.future:
                self._active_future = None

        return (
            payload.memory,
            payload.layer_kv,
            payload.attention_mask,
            payload.token_times,
            payload.rebuild_count,
            payload.frame_count,
        )

    def _execute_worker_rebuild(
        self,
        raw_mem: PeriodicMemoryState,
        raw_version_token: Any,
        episode_token: Any,
        frame_count: int,
        latest_frame_id: int,
        rebuild_count: int,
        prompt: str,
        revision: int,
        config: CompactMemoryConfig,
        source_event: Optional[Any],
        stream: Optional[Any],
    ) -> CompactRebuildPayload:
        """Worker task executing inside background thread with @no_grad."""
        with torch.no_grad():
            stream_ctx = (
                torch.cuda.stream(stream)
                if (stream is not None and torch.cuda.is_available())
                else contextlib.nullcontext()
            )
            with stream_ctx:
                if stream is not None and torch.cuda.is_available() and source_event is not None:
                    stream.wait_event(source_event)

                # Record all tensors to stream
                if stream is not None and torch.cuda.is_available():
                    for rf in raw_mem.recent:
                        rf.inputs_embeds.record_stream(stream)
                        rf.attention_mask.record_stream(stream)
                        if rf.visual_tokens is not None:
                            rf.visual_tokens.record_stream(stream)
                    for a in raw_mem.anchors:
                        a.inputs_embeds.record_stream(stream)
                        a.attention_mask.record_stream(stream)
                        if a.visual_tokens is not None:
                            a.visual_tokens.record_stream(stream)
                    for e in raw_mem.entries:
                        e.visual_tokens.record_stream(stream)

                # Consolidate raw memory
                consolidated_mem = consolidate_memory(raw_mem, config.memory)

                # Render canonical compact memory
                embedder = self._rebuild_embedder if self._rebuild_embedder is not None else getattr(self.policy, "embedder", None)
                rendered = render_compact_memory(consolidated_mem, embedder, config)

                # Build full DynamicCache using native LLM
                cache = DynamicCache()
                self._execute_native_layers(
                    inputs_embeds=rendered.inputs_embeds,
                    attention_mask_2d=rendered.attention_mask,
                    cache=cache,
                    start_pos=0,
                    token_times=rendered.token_times,
                    temporal_config=config.temporal,
                )

                new_layer_kv = extract_layer_kv(cache)

            if stream is not None and torch.cuda.is_available():
                stream.synchronize()

            return CompactRebuildPayload(
                memory=consolidated_mem,
                layer_kv=new_layer_kv,
                attention_mask=rendered.attention_mask,
                token_times=rendered.token_times,
                owner=self,
                revision=revision,
                prompt=prompt,
                episode_token=episode_token,
                raw_version_token=raw_version_token,
                frame_count=frame_count,
                latest_frame_id=latest_frame_id,
                rebuild_count=rebuild_count + 1,
                config=config,
            )

    @torch.no_grad()
    def encode_frame(self, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("NativeCompactMemoryCacheAdapter is closed.")
        return super().encode_frame(*args, **kwargs)

    @torch.no_grad()
    def read_blocks(
        self,
        new_blocks: Sequence[NativeEmbeddingBlock],
        prompt: str,
        previous: Optional[CompactKVState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, CompactKVState]:
        if self._closed:
            raise RuntimeError("NativeCompactMemoryCacheAdapter is closed.")
        if self.training:
            raise RuntimeError("read_blocks is only allowed in eval mode.")
        if not new_blocks:
            raise ValueError("new_blocks must not be empty.")

        core = self.native_core
        core_param = next(core.parameters())
        core_dtype, core_device = core_param.dtype, core_param.device
        expected_seq_len = getattr(getattr(self.policy, "embedder", None), "max_text_length", 1024)

        # 1. Validate incoming new_blocks
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
                raise ValueError(f"Block[{i}] must preserve visual_tokens for compact memory")
            if b.visual_tokens.dtype != core_dtype or b.visual_tokens.device != core_device:
                raise ValueError(f"Block[{i}] visual_tokens dtype/device must match core ({core_dtype}, {core_device})")
            if i > 0 and b.frame_id <= new_blocks[i - 1].frame_id:
                raise ValueError("new_blocks frame_id must be strictly increasing")
            if self.config.use_timestamps and b.observation_time is None:
                raise ValueError("Block observation_time must not be None when use_timestamps=True")

        # 2. Validate and handle previous state / episode token
        target_future = previous.pending.future if (previous is not None and previous.pending is not None) else None
        stale_future: Optional[concurrent.futures.Future[CompactRebuildPayload]] = None
        with self._lock:
            if self._active_future is not None and self._active_future is not target_future:
                stale_future = self._active_future

        if stale_future is not None:
            try:
                if not stale_future.cancel():
                    stale_future.result()
            finally:
                with self._lock:
                    if self._active_future is stale_future:
                        self._active_future = None

        if previous is not None:
            self._validate_compact_state(previous, prompt)
            if new_blocks[0].frame_id <= previous.last_frame_id:
                raise ValueError(f"new block frame_id {new_blocks[0].frame_id} <= previous {previous.last_frame_id}")
            episode_token = previous.episode_token
            parent_version_token = previous.version_token
            # Resolve previous pending rebuild (waits if necessary)
            (
                effective_mem,
                effective_layer_kv,
                effective_att_mask,
                effective_token_times,
                rebuild_count,
                base_frame_count,
            ) = self._resolve_pending(previous, prompt)
        else:
            episode_token = object()
            parent_version_token = None
            effective_mem = PeriodicMemoryState()
            effective_layer_kv = None
            effective_att_mask = None
            effective_token_times = None
            rebuild_count = 0
            base_frame_count = 0

        # Monotonic time check across boundaries
        last_obs_time = effective_mem.last_observation_time
        all_times = ([last_obs_time] if last_obs_time is not None else []) + [
            b.observation_time for b in new_blocks if b.observation_time is not None
        ]
        for i in range(1, len(all_times)):
            if all_times[i] < all_times[i - 1]:
                raise ValueError(f"observation_time must be non-decreasing: {all_times[i]} < {all_times[i - 1]}")

        # 3. Append all new frames to effective memory without consolidation
        #    Only the final frame in new_blocks has is_decision=True
        new_memory_frames: List[MemoryFrame] = []
        raw_mem = effective_mem
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
            raw_mem = append_memory_frame(raw_mem, mf)
            # Retrieve the frame stored in recent which has assigned boundary_frame_id
            new_memory_frames.append(raw_mem.recent[-1])

        total_frame_count = base_frame_count + len(new_blocks)
        last_frame_id = new_blocks[-1].frame_id

        # 4. Render new blocks using render_compact_frames (or render_compact_memory if cold)
        embedder = self._render_embedder if self._render_embedder is not None else getattr(self.policy, "embedder", None)
        shallow_idx = self.config.shallow_layer

        if effective_layer_kv is None:
            # Cold start query: render full compact memory for all initial frames
            rendered = render_compact_memory(raw_mem, embedder, self.compact_config)
            cache = DynamicCache()
            final_norm_h, inter_h = self._execute_native_layers(
                inputs_embeds=rendered.inputs_embeds,
                attention_mask_2d=rendered.attention_mask,
                cache=cache,
                start_pos=0,
                token_times=rendered.token_times,
                temporal_config=self.compact_config.temporal,
            )
            full_layer_kv = extract_layer_kv(cache)
            full_att_mask = rendered.attention_mask
            full_token_times = rendered.token_times
            cur_start = rendered.current_start
            deep = final_norm_h[:, cur_start:, :]
            shallow = deep if shallow_idx == len(core.layers) else inter_h[shallow_idx][:, cur_start:, :]
        else:
            # Incremental append: render only new frames
            rendered_new = render_compact_frames(new_memory_frames, embedder, self.compact_config)
            cache = populate_cache_from_layer_kv(effective_layer_kv)
            start_pos = effective_att_mask.shape[1]
            cum_mask = torch.cat([effective_att_mask, rendered_new.attention_mask], dim=1)
            cum_times = torch.cat([effective_token_times, rendered_new.token_times], dim=1)

            final_norm_h, inter_h = self._execute_native_layers(
                inputs_embeds=rendered_new.inputs_embeds,
                attention_mask_2d=cum_mask,
                cache=cache,
                start_pos=start_pos,
                token_times=rendered_new.token_times,
                temporal_config=self.compact_config.temporal,
            )
            full_layer_kv = extract_layer_kv(cache)
            full_att_mask = cum_mask
            full_token_times = cum_times
            cur_start = rendered_new.current_start
            deep = final_norm_h[:, cur_start:, :]
            shallow = deep if shallow_idx == len(core.layers) else inter_h[shallow_idx][:, cur_start:, :]

        # 5. Determine if consolidation rebuild is triggered: len(raw_mem.recent) >= R + K
        r = self.compact_config.memory.recent_frames
        k = self.compact_config.memory.consolidate_every
        needs_rebuild = len(raw_mem.recent) >= (r + k)

        raw_version_token = object()
        pending_obj: Optional[PendingRebuild] = None

        if needs_rebuild:
            # Prepare rebuild parameters
            stream = self._get_cuda_stream(core_device)
            source_event = None
            if stream is not None and torch.cuda.is_available():
                source_event = torch.cuda.Event()
                source_event.record(torch.cuda.current_stream(core_device))

            if self.background_rebuild:
                executor = self._ensure_executor()
                fut = executor.submit(
                    self._execute_worker_rebuild,
                    raw_mem=raw_mem,
                    raw_version_token=raw_version_token,
                    episode_token=episode_token,
                    frame_count=total_frame_count,
                    latest_frame_id=last_frame_id,
                    rebuild_count=rebuild_count,
                    prompt=prompt,
                    revision=self._revision,
                    config=self.compact_config,
                    source_event=source_event,
                    stream=stream,
                )
                with self._lock:
                    self._active_future = fut
                pending_obj = PendingRebuild(
                    future=fut,
                    owner=self,
                    revision=self._revision,
                    prompt=prompt,
                    episode_token=episode_token,
                    raw_version_token=raw_version_token,
                    frame_count=total_frame_count,
                    latest_frame_id=last_frame_id,
                    config=self.compact_config,
                )
            else:
                # Synchronous control: compute immediately but return as completed Future
                payload = self._execute_worker_rebuild(
                    raw_mem=raw_mem,
                    raw_version_token=raw_version_token,
                    episode_token=episode_token,
                    frame_count=total_frame_count,
                    latest_frame_id=last_frame_id,
                    rebuild_count=rebuild_count,
                    prompt=prompt,
                    revision=self._revision,
                    config=self.compact_config,
                    source_event=source_event,
                    stream=stream,
                )
                sync_fut: concurrent.futures.Future[CompactRebuildPayload] = concurrent.futures.Future()
                sync_fut.set_result(payload)
                pending_obj = PendingRebuild(
                    future=sync_fut,
                    owner=self,
                    revision=self._revision,
                    prompt=prompt,
                    episode_token=episode_token,
                    raw_version_token=raw_version_token,
                    frame_count=total_frame_count,
                    latest_frame_id=last_frame_id,
                    config=self.compact_config,
                )

        next_state = CompactKVState(
            layer_kv=full_layer_kv,
            memory=raw_mem,
            attention_mask=full_att_mask,
            token_times=full_token_times,
            last_frame_id=last_frame_id,
            frame_count=total_frame_count,
            prompt=prompt,
            owner=self,
            revision=self._revision,
            rebuild_count=rebuild_count,
            episode_token=episode_token,
            version_token=raw_version_token,
            parent_version_token=parent_version_token,
            pending=pending_obj,
            config=self.compact_config,
        )

        return deep.float(), shallow.float(), next_state


def validate_compact_memory(
    candidate: CompactKVState,
    previous: Optional[CompactKVState],
    snapshot: Sequence[Any],
    prompt: str,
) -> CompactKVState:
    """Strictly validate CompactKVState without running additional LLM forwards.

    Replays previous effective memory appended with snapshot frames,
    verifying exact raw memory, counts, tokens, shapes, and pending bindings.
    """
    if not isinstance(candidate, CompactKVState):
        raise TypeError(f"candidate must be CompactKVState, got {type(candidate)}")
    if not snapshot:
        raise ValueError("snapshot must not be empty")

    owner = candidate.owner
    if not isinstance(owner, NativeCompactMemoryCacheAdapter):
        raise TypeError(f"candidate.owner must be NativeCompactMemoryCacheAdapter, got {type(owner)}")

    core = owner.native_core
    core_dtype = next(core.parameters()).dtype
    core_device = next(core.parameters()).device
    expected_seq_len = getattr(getattr(owner.policy, "embedder", None), "max_text_length", 1024)

    # 1. Validate snapshot frames
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
            raise ValueError("Compact memory snapshot requires observation_time")
        obs = getattr(frame, "observation", None)
        if obs is not None and (blk.frame_id, blk.observation_time, blk.capture_time) != (obs.frame_id, obs.observation_time, obs.capture_time):
            raise ValueError(f"snapshot[{idx}] payload frame/time mismatch with observation")
        if idx > 0:
            prev_blk = getattr(snapshot[idx - 1], "payload", snapshot[idx - 1])
            if blk.frame_id <= prev_blk.frame_id:
                raise ValueError("snapshot payload frame IDs must be strictly increasing")

    # 2. Validate candidate vs adapter
    if candidate.revision != owner.revision or candidate.prompt != prompt:
        raise ValueError("candidate revision or prompt mismatch")
    if candidate.config != owner.compact_config:
        raise ValueError("candidate config mismatch with owner compact_config")

    # 3. Validate previous and replay expected raw memory
    if previous is not None:
        if not isinstance(previous, CompactKVState):
            raise TypeError("previous must be CompactKVState")
        if previous.owner is not owner or previous.revision != owner.revision or previous.prompt != prompt:
            raise ValueError("previous state owner/revision/prompt mismatch")
        if candidate.episode_token is not previous.episode_token:
            raise ValueError("candidate episode_token must match previous episode_token")
        if candidate.parent_version_token is not previous.version_token:
            raise ValueError("candidate parent_version_token must match previous version_token")
        if previous.frame_count != previous.memory.frame_count:
            raise ValueError(f"previous frame_count {previous.frame_count} != memory.frame_count {previous.memory.frame_count}")
        if previous.memory.last_frame_id is not None and previous.last_frame_id != previous.memory.last_frame_id:
            raise ValueError(f"previous last_frame_id {previous.last_frame_id} != memory.last_frame_id {previous.memory.last_frame_id}")
        first_snap_id = getattr(snapshot[0], "payload", snapshot[0]).frame_id
        if first_snap_id <= previous.last_frame_id:
            raise ValueError(f"snapshot first frame_id {first_snap_id} <= previous {previous.last_frame_id}")

        # Resolve previous effective memory (waiting if pending)
        if previous.pending is not None:
            eff_mem, _, _, _, eff_rebuild, _ = owner._resolve_pending(previous, prompt)
            effective_mem = eff_mem
            expected_rebuild = eff_rebuild
        else:
            effective_mem = previous.memory
            expected_rebuild = previous.rebuild_count
    else:
        if candidate.parent_version_token is not None:
            raise ValueError("candidate parent_version_token must be None when previous is None")
        effective_mem = PeriodicMemoryState()
        expected_rebuild = 0

    expected_frame_count = (previous.frame_count if previous is not None else 0) + len(snapshot)
    if candidate.frame_count != expected_frame_count:
        raise ValueError(f"candidate frame_count {candidate.frame_count} != expected {expected_frame_count}")

    last_snap_id = getattr(snapshot[-1], "payload", snapshot[-1]).frame_id
    if candidate.last_frame_id != last_snap_id:
        raise ValueError(f"candidate last_frame_id {candidate.last_frame_id} != snapshot {last_snap_id}")

    if candidate.rebuild_count != expected_rebuild:
        raise ValueError(f"candidate rebuild_count {candidate.rebuild_count} != expected {expected_rebuild}")

    # Replay append_memory_frame on effective_mem
    expected_raw_mem = effective_mem
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
        expected_raw_mem = append_memory_frame(expected_raw_mem, mf)

    cand_mem = candidate.memory
    if (cand_mem.frame_count, cand_mem.consolidations, cand_mem.merges) != (
        expected_raw_mem.frame_count,
        expected_raw_mem.consolidations,
        expected_raw_mem.merges,
    ):
        raise ValueError("candidate raw memory frame_count/consolidations/merges mismatch")
    if len(cand_mem.entries) != len(expected_raw_mem.entries) or len(cand_mem.recent) != len(expected_raw_mem.recent) or len(cand_mem.anchors) != len(expected_raw_mem.anchors):
        raise ValueError("candidate raw memory entries, recent, or anchors count mismatch")

    total_items_count = sum(e.count for e in cand_mem.entries) + len(cand_mem.anchors) + len(cand_mem.recent)
    if total_items_count != expected_frame_count:
        raise ValueError(
            f"expected state counts sum entries count ({sum(e.count for e in cand_mem.entries)}) + "
            f"len(anchors) ({len(cand_mem.anchors)}) + len(recent) ({len(cand_mem.recent)}) = "
            f"{total_items_count} != expected_frame_count {expected_frame_count}"
        )
    if cand_mem.last_decision_frame_id != expected_raw_mem.last_decision_frame_id:
        raise ValueError(
            f"candidate last_decision_frame_id {cand_mem.last_decision_frame_id} != expected {expected_raw_mem.last_decision_frame_id}"
        )

    # Validate decision protection on recent frames
    if not cand_mem.recent[-1].is_decision:
        raise ValueError("The last recent frame in candidate must be a decision frame (is_decision=True)")

    # Validate tensors in candidate KV and masks
    num_layers = len(core.layers)
    if len(candidate.layer_kv) != num_layers:
        raise ValueError(f"candidate layer_kv count {len(candidate.layer_kv)} != layers count {num_layers}")
    total_len = candidate.attention_mask.shape[1]
    if candidate.token_times.shape != (1, total_len):
        raise ValueError("candidate token_times shape mismatch with attention_mask")

    num_kv_heads, head_dim = owner._get_kv_heads_and_dim()
    for idx, (k, v) in enumerate(candidate.layer_kv):
        if k.shape != (1, num_kv_heads, total_len, head_dim) or v.shape != k.shape:
            raise ValueError(f"candidate layer_kv[{idx}] shape mismatch with total_len {total_len}")
        if k.dtype != core_dtype or v.dtype != core_dtype:
            raise TypeError(f"candidate layer_kv[{idx}] dtype mismatch with core")
        if k.device != core_device or v.device != core_device:
            raise ValueError(f"candidate layer_kv[{idx}] device mismatch with core")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError(f"candidate layer_kv[{idx}] contains non-finite values")

    # Validate pending requirements
    r = owner.compact_config.memory.recent_frames
    k_cons = owner.compact_config.memory.consolidate_every
    needs_rebuild = len(cand_mem.recent) >= (r + k_cons)

    if needs_rebuild:
        if candidate.pending is None:
            raise ValueError(f"candidate.pending must be present when len(recent) >= R + K ({len(cand_mem.recent)} >= {r + k_cons})")
        pending = candidate.pending
        if pending.owner is not owner or pending.revision != owner.revision or pending.prompt != prompt:
            raise ValueError("candidate.pending owner/revision/prompt mismatch")
        if pending.episode_token is not candidate.episode_token or pending.raw_version_token is not candidate.version_token:
            raise ValueError("candidate.pending version tokens mismatch with candidate")
        if pending.frame_count != candidate.frame_count or pending.latest_frame_id != candidate.last_frame_id:
            raise ValueError("candidate.pending frame_count or latest_frame_id mismatch with candidate")
        if pending.config != owner.compact_config:
            raise ValueError("candidate.pending config mismatch with owner.compact_config")
    else:
        if candidate.pending is not None:
            raise ValueError("candidate.pending must be None when len(recent) < R + K")

    _validate_memory_equal(cand_mem, expected_raw_mem)
    embedder = owner._render_embedder if getattr(owner, "_render_embedder", None) is not None else getattr(owner.policy, "embedder", None)
    with torch.no_grad():
        rendered = render_compact_memory(expected_raw_mem, embedder, owner.compact_config)
    if not torch.equal(candidate.attention_mask, rendered.attention_mask):
        raise ValueError("candidate attention_mask does not match rendered memory attention_mask")
    if not torch.equal(candidate.token_times, rendered.token_times):
        raise ValueError("candidate token_times does not match rendered memory token_times")

    return candidate
