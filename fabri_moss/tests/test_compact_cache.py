"""Unit and integration tests for fabri_moss.compact_cache."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from fabri_moss.async_pipeline import EncodedFrame, Observation
from fabri_moss.compact_cache import (
    CompactKVState,
    CompactRebuildPayload,
    NativeCompactMemoryCacheAdapter,
    PendingRebuild,
    validate_compact_memory,
)
from fabri_moss.compact_memory import (
    CompactMemoryConfig,
)
from fabri_moss.native_cache import (
    NativeCacheConfig,
    NativeEmbeddingBlock,
)
from fabri_moss.periodic_memory import (
    MemoryFrame,
    PeriodicMemoryConfig,
    PeriodicMemoryState,
)
from fabri_moss.temporal_rope import TemporalRoPEConfig


class TinyTokenizer:
    """Mock tokenizer supporting optional add_special_tokens."""

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False) -> Any:
        class TokenizerOutput:
            pass

        out = TokenizerOutput()
        out.input_ids = torch.ones((1, 8), dtype=torch.long)
        return out


class TinyEmbedder(nn.Module):
    """Embedder matching FabriVLA contract for native memory cache tests."""

    def __init__(self, language_model: nn.Module, max_text_length: int = 16):
        super().__init__()
        self.device = torch.device("cpu")
        self.max_text_length = max_text_length
        self.extract_feature_call_count = 0

        self.model = nn.Module()
        self.model.language_model = language_model
        self.model.extract_feature = self._extract_feature

        self.tokenizer = TinyTokenizer()

    def _extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.extract_feature_call_count += 1
        return pixel_values

    def _preprocess_images(self, images: Sequence[Any]) -> Tuple[torch.Tensor, List[int]]:
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            return torch.zeros((1, 4, 64)), [1, 1]
        img = images[0]
        if isinstance(img, str):
            key = img.encode("utf-8")
        elif isinstance(img, bytes):
            key = img
        elif hasattr(img, "tobytes"):
            key = img.tobytes()
        else:
            key = str(img).encode("utf-8")

        h_val = int(hashlib.md5(key).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed(h_val % (2**31 - 1))
        # 4 visual tokens of dim 64 (2x2 grid for spatial_grid=1)
        feats = torch.randn((1, 4, 64), generator=gen)
        return feats, [1]

    def _build_multimodal_prompt(self, num_tiles_list: List[int], prompt: str) -> str:
        return f"<prompt>{prompt}</prompt>"

    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        core = self.model.language_model.model
        h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:6], 16)
        tok_val = (h % 50) + 1
        tokens = torch.full((1, 12), tok_val, dtype=torch.long)
        tok_embeds = core.embed_tokens(tokens)
        inputs_embeds = torch.cat([vit_embeds, tok_embeds], dim=1)  # shape [1, 16, 64]
        attention_mask = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], dim=1)  # shape [1, 16]
        return inputs_embeds, attention_mask


class TinyActionHead(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 50 * 24)

    def sample(self, deep: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        b = deep.shape[0]
        out = self.linear(deep[:, 0, :])
        return out.view(b, 50, 24)


class TinyPolicy(nn.Module):
    def __init__(self, language_model: nn.Module, max_text_length: int = 16):
        super().__init__()
        self.embedder = TinyEmbedder(language_model, max_text_length=max_text_length)
        self.action_head = TinyActionHead(hidden_dim=64)


def make_tiny_compact_adapter(
    max_frames: int = 4,
    shallow_layer: int = 1,
    max_text_length: int = 16,
    use_timestamps: bool = True,
    recent_frames: int = 2,
    consolidate_every: int = 2,
    memory_slots: int = 2,
    spatial_grid: int = 1,
    intermediate_grid: int | None = 1,
    strength: float = 1.0,
    background_rebuild: bool = True,
) -> NativeCompactMemoryCacheAdapter:
    config = Qwen3Config(
        hidden_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        intermediate_size=128,
        vocab_size=64,
        pad_token_id=0,
        max_position_embeddings=512,
        use_cache=True,
        _attn_implementation="eager",
    )
    language_model = Qwen3ForCausalLM(config)
    policy = TinyPolicy(language_model, max_text_length=max_text_length)
    cache_config = NativeCacheConfig(
        max_frames=max_frames,
        shallow_layer=shallow_layer,
        use_timestamps=use_timestamps,
    )
    mem_config = PeriodicMemoryConfig(
        recent_frames=recent_frames,
        consolidate_every=consolidate_every,
        memory_slots=memory_slots,
        spatial_grid=spatial_grid,
        protect_decision_frames=True,
    )
    temporal_config = TemporalRoPEConfig(
        strength=strength,
        time_unit_seconds=1.0,
        rotary_fraction=0.25,
    )
    compact_config = CompactMemoryConfig(
        memory=mem_config,
        intermediate_grid=intermediate_grid,
        temporal=temporal_config,
    )
    adapter = NativeCompactMemoryCacheAdapter(
        policy,
        config=cache_config,
        compact_config=compact_config,
        background_rebuild=background_rebuild,
    )
    adapter.eval()
    return adapter


def test_initial_and_incremental_read_blocks_no_rebuild():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2)
    prompt = "pick block"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    deep0, shallow0, s0 = adapter.read_blocks([b0], prompt=prompt)

    assert deep0.shape == (1, 16, 64)
    assert shallow0.shape == (1, 16, 64)
    assert s0.frame_count == 1
    assert s0.rebuild_count == 0
    assert len(s0.blocks) == 1
    assert s0.pending is None

    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    deep1, shallow1, s1 = adapter.read_blocks([b1], prompt=prompt, previous=s0)

    assert deep1.shape == (1, 16, 64)
    assert shallow1.shape == (1, 16, 64)
    assert s1.frame_count == 2
    assert s1.rebuild_count == 0
    assert len(s1.blocks) == 2
    assert s1.pending is None
    adapter.close()


def test_rebuild_triggered_at_r_plus_k_sync_vs_async_parity():
    # R=2, K=2 -> R+K=4 triggers rebuild
    prompt = "sync async parity"

    # 1. Sync adapter
    adapter_sync = make_tiny_compact_adapter(
        recent_frames=2,
        consolidate_every=2,
        spatial_grid=1,
        intermediate_grid=1,
        background_rebuild=False,
    )
    # 2. Async adapter
    adapter_async = make_tiny_compact_adapter(
        recent_frames=2,
        consolidate_every=2,
        spatial_grid=1,
        intermediate_grid=1,
        background_rebuild=True,
    )
    # Share weights
    adapter_async.load_state_dict(adapter_sync.state_dict())

    b0 = adapter_sync.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1 = adapter_sync.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    b2 = adapter_sync.encode_frame(images=["img2"], frame_id=2, prompt=prompt, observation_time=0.2)
    b3 = adapter_sync.encode_frame(images=["img3"], frame_id=3, prompt=prompt, observation_time=0.3)

    _, _, s0_sync = adapter_sync.read_blocks([b0], prompt=prompt)
    _, _, s1_sync = adapter_sync.read_blocks([b1], prompt=prompt, previous=s0_sync)
    _, _, s2_sync = adapter_sync.read_blocks([b2], prompt=prompt, previous=s1_sync)
    d3_sync, s3_sync, s3_state_sync = adapter_sync.read_blocks([b3], prompt=prompt, previous=s2_sync)

    b0_async = adapter_async.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1_async = adapter_async.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    b2_async = adapter_async.encode_frame(images=["img2"], frame_id=2, prompt=prompt, observation_time=0.2)
    b3_async = adapter_async.encode_frame(images=["img3"], frame_id=3, prompt=prompt, observation_time=0.3)

    _, _, s0_async = adapter_async.read_blocks([b0_async], prompt=prompt)
    _, _, s1_async = adapter_async.read_blocks([b1_async], prompt=prompt, previous=s0_async)
    _, _, s2_async = adapter_async.read_blocks([b2_async], prompt=prompt, previous=s1_async)
    d3_async, s3_async, s3_state_async = adapter_async.read_blocks([b3_async], prompt=prompt, previous=s2_async)

    # Output features on query 3 must be identical
    assert torch.allclose(d3_sync, d3_async, atol=1e-5)
    assert torch.allclose(s3_sync, s3_async, atol=1e-5)

    # Both states have pending rebuild triggered
    assert s3_state_sync.pending is not None
    assert s3_state_async.pending is not None
    assert len(s3_state_sync.blocks) == 4
    assert len(s3_state_async.blocks) == 4

    # Query 4 resolves rebuild and applies incremental new frame
    b4 = adapter_sync.encode_frame(images=["img4"], frame_id=4, prompt=prompt, observation_time=0.4)
    b4_async = adapter_async.encode_frame(images=["img4"], frame_id=4, prompt=prompt, observation_time=0.4)
    d4_sync, _, s4_sync = adapter_sync.read_blocks([b4], prompt=prompt, previous=s3_state_sync)
    d4_async, _, s4_async = adapter_async.read_blocks([b4_async], prompt=prompt, previous=s3_state_async)

    assert torch.allclose(d4_sync, d4_async, atol=1e-5)
    assert s4_sync.rebuild_count == 1
    assert s4_async.rebuild_count == 1
    assert torch.equal(s4_sync.attention_mask, s4_async.attention_mask)
    for (k1, v1), (k2, v2) in zip(s4_sync.layer_kv, s4_async.layer_kv):
        assert torch.allclose(k1, k2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)

    adapter_sync.close()
    adapter_async.close()


def test_zero_time_strength_and_noop_grid():
    adapter = make_tiny_compact_adapter(
        recent_frames=2,
        consolidate_every=2,
        intermediate_grid=None,
        strength=0.0,
    )
    prompt = "zero strength"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    deep0, _, s0 = adapter.read_blocks([b0], prompt=prompt)
    assert deep0.shape == (1, 16, 64)
    assert s0.token_times.shape == s0.attention_mask.shape
    adapter.close()


def test_compact_intermediate_reduces_sequence_length():
    # Intermediate grid 1 -> pooled visual tokens 1x1 = 1 token vs original 4 tokens
    adapter_compact = make_tiny_compact_adapter(
        recent_frames=4,
        consolidate_every=4,
        intermediate_grid=1,
    )
    adapter_full = make_tiny_compact_adapter(
        recent_frames=4,
        consolidate_every=4,
        intermediate_grid=None,
    )
    prompt = "compact length"
    b0_c = adapter_compact.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1_c = adapter_compact.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)

    b0_f = adapter_full.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1_f = adapter_full.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)

    # Multi-block query: b0 is non-decision intermediate, b1 is decision final
    _, _, s_compact = adapter_compact.read_blocks([b0_c, b1_c], prompt=prompt)
    _, _, s_full = adapter_full.read_blocks([b0_f, b1_f], prompt=prompt)

    # Compact should have shorter total sequence length than full
    assert s_compact.attention_mask.shape[1] < s_full.attention_mask.shape[1]
    # Decision frame visual tokens are kept intact on both
    assert s_compact.memory.recent[1].visual_tokens.shape == (1, 4, 64)

    adapter_compact.close()
    adapter_full.close()


def test_background_worker_truly_runs_asynchronously():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2, background_rebuild=True)
    prompt = "async event check"

    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    b2 = adapter.encode_frame(images=["img2"], frame_id=2, prompt=prompt, observation_time=0.2)
    b3 = adapter.encode_frame(images=["img3"], frame_id=3, prompt=prompt, observation_time=0.3)

    _, _, s0 = adapter.read_blocks([b0], prompt=prompt)
    _, _, s1 = adapter.read_blocks([b1], prompt=prompt, previous=s0)
    _, _, s2 = adapter.read_blocks([b2], prompt=prompt, previous=s1)

    # We patch _execute_worker_rebuild to synchronize on an Event
    worker_started = threading.Event()
    worker_can_proceed = threading.Event()
    orig_worker = adapter._execute_worker_rebuild

    def delayed_worker(*args: Any, **kwargs: Any) -> Any:
        worker_started.set()
        worker_can_proceed.wait()
        return orig_worker(*args, **kwargs)

    adapter._execute_worker_rebuild = delayed_worker  # type: ignore[assignment]

    # read_blocks for b3 should return IMMEDIATELY with pending future still running
    d3, s3, s3_state = adapter.read_blocks([b3], prompt=prompt, previous=s2)
    assert s3_state.pending is not None
    assert not s3_state.pending.future.done()
    assert worker_started.wait(timeout=2.0)

    # Foreground can inspect and use s3_state without waiting for worker
    assert d3.shape == (1, 16, 64)
    assert s3.shape == (1, 16, 64)
    assert len(s3_state.blocks) == 4

    # Allow worker to finish
    worker_can_proceed.set()

    # Next query b4 will wait for worker and resolve
    b4 = adapter.encode_frame(images=["img4"], frame_id=4, prompt=prompt, observation_time=0.4)
    d4, _, s4 = adapter.read_blocks([b4], prompt=prompt, previous=s3_state)
    assert s4.rebuild_count == 1
    assert s4.pending is None

    adapter.close()


def test_old_state_and_tensors_immutability():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2)
    prompt = "immutable"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    _, _, s0 = adapter.read_blocks([b0], prompt=prompt)

    kv0_clone = s0.layer_kv[0][0].clone()
    mask0_clone = s0.attention_mask.clone()
    times0_clone = s0.token_times.clone()

    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    _, _, s1 = adapter.read_blocks([b1], prompt=prompt, previous=s0)

    assert torch.equal(s0.layer_kv[0][0], kv0_clone)
    assert torch.equal(s0.attention_mask, mask0_clone)
    assert torch.equal(s0.token_times, times0_clone)
    adapter.close()


def test_version_and_episode_token_corruption_rejection():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2, background_rebuild=False)
    prompt = "token check"

    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    b2 = adapter.encode_frame(images=["img2"], frame_id=2, prompt=prompt, observation_time=0.2)
    b3 = adapter.encode_frame(images=["img3"], frame_id=3, prompt=prompt, observation_time=0.3)

    _, _, s0 = adapter.read_blocks([b0, b1, b2, b3], prompt=prompt)
    assert s0.pending is not None

    # Corrupt raw_version_token on pending
    corrupted_pending = dataclasses.replace(s0.pending, raw_version_token=object())
    s0_corrupt = dataclasses.replace(s0, pending=corrupted_pending)

    b4 = adapter.encode_frame(images=["img4"], frame_id=4, prompt=prompt, observation_time=0.4)
    with pytest.raises(ValueError, match="raw_version_token"):
        adapter.read_blocks([b4], prompt=prompt, previous=s0_corrupt)

    # Corrupt episode_token
    corrupted_ep = dataclasses.replace(s0.pending, episode_token=object())
    s0_corrupt_ep = dataclasses.replace(s0, pending=corrupted_ep)
    with pytest.raises(ValueError, match="episode_token"):
        adapter.read_blocks([b4], prompt=prompt, previous=s0_corrupt_ep)

    adapter.close()


def test_previous_none_resets_episode_token():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2)
    prompt = "reset ep"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    _, _, s0 = adapter.read_blocks([b0], prompt=prompt)

    # Next episode with previous=None
    b_new = adapter.encode_frame(images=["img_new"], frame_id=0, prompt=prompt, observation_time=0.0)
    _, _, s_new = adapter.read_blocks([b_new], prompt=prompt, previous=None)

    assert s0.episode_token is not s_new.episode_token
    assert s_new.parent_version_token is None
    adapter.close()


def test_validator_comprehensive_checks():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2, background_rebuild=False)
    prompt = "val check"

    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0, capture_time=0.0)
    _, _, s0 = adapter.read_blocks([b0], prompt=prompt)

    obs0 = Observation(
        frame_id=0,
        capture_time=0.0,
        images=[torch.zeros((1, 3, 16, 16))],
        state=torch.zeros((1, 10)),
        state_mask=torch.ones((1, 10), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
        observation_time=0.0,
    )
    snap0 = (EncodedFrame(observation=obs0, payload=b0, encode_started=0.0, encode_finished=0.01),)

    # Valid candidate check
    validate_compact_memory(s0, previous=None, snapshot=snap0, prompt=prompt)

    # Prompt mismatch
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_compact_memory(s0, previous=None, snapshot=snap0, prompt="wrong prompt")

    # Frame count mismatch
    s0_bad_count = dataclasses.replace(s0, frame_count=999)
    with pytest.raises(ValueError, match="frame_count"):
        validate_compact_memory(s0_bad_count, previous=None, snapshot=snap0, prompt=prompt)

    adapter.close()


def test_future_exception_bubbles_up():
    adapter = make_tiny_compact_adapter(recent_frames=2, consolidate_every=2, background_rebuild=False)
    prompt = "err check"

    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    b2 = adapter.encode_frame(images=["img2"], frame_id=2, prompt=prompt, observation_time=0.2)
    b3 = adapter.encode_frame(images=["img3"], frame_id=3, prompt=prompt, observation_time=0.3)

    _, _, s3 = adapter.read_blocks([b0, b1, b2, b3], prompt=prompt)
    assert s3.pending is not None

    # Inject a failing future
    err_fut: concurrent.futures.Future[CompactRebuildPayload] = concurrent.futures.Future()
    err_fut.set_exception(RuntimeError("Background rebuild failed with OOM"))
    bad_pending = dataclasses.replace(s3.pending, future=err_fut)
    s3_bad = dataclasses.replace(s3, pending=bad_pending)

    b4 = adapter.encode_frame(images=["img4"], frame_id=4, prompt=prompt, observation_time=0.4)
    with pytest.raises(RuntimeError, match="Background rebuild failed with OOM"):
        adapter.read_blocks([b4], prompt=prompt, previous=s3_bad)

    adapter.close()


def test_close_and_context_manager():
    with make_tiny_compact_adapter(recent_frames=2, consolidate_every=2) as adapter:
        assert not adapter._closed
        _ = adapter._ensure_executor()
        assert adapter._executor is not None
    assert adapter._closed
    assert adapter._executor is None
