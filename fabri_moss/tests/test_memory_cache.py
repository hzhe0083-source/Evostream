"""Unit and integration tests for fabri_moss.memory_cache."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from fabri_moss.async_pipeline import EncodedFrame, Observation
from fabri_moss.memory_cache import (
    NativeMemoryCacheAdapter,
    NativeMemoryKVState,
    validate_memory_cache,
)
from fabri_moss.native_cache import (
    NativeCacheConfig,
    NativeEmbeddingBlock,
)
from fabri_moss.periodic_memory import (
    PeriodicMemoryConfig,
)


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
        # 4 visual tokens of dim 64 (4 is perfect square 2x2 for spatial_grid=1 or 2)
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


def make_tiny_memory_adapter(
    max_frames: int = 4,
    shallow_layer: int = 1,
    max_text_length: int = 16,
    use_timestamps: bool = True,
    recent_frames: int = 2,
    consolidate_every: int = 2,
    memory_slots: int = 2,
    spatial_grid: int = 1,
    protect_decision_frames: bool = True,
) -> NativeMemoryCacheAdapter:
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
        protect_decision_frames=protect_decision_frames,
    )
    adapter = NativeMemoryCacheAdapter(policy, config=cache_config, memory_config=mem_config)
    adapter.eval()
    return adapter


def test_encode_frame_preserves_visual_tokens():
    adapter = make_tiny_memory_adapter()
    prompt = "find cup"
    block = adapter.encode_frame(
        images=["img0"],
        frame_id=0,
        prompt=prompt,
        observation_time=0.0,
    )
    assert block.visual_tokens is not None
    assert block.visual_tokens.shape == (1, 4, 64)
    assert block.inputs_embeds.shape == (1, 16, 64)
    assert block.frame_id == 0


def test_initial_and_incremental_read_blocks():
    adapter = make_tiny_memory_adapter(recent_frames=2, consolidate_every=2)
    prompt = "pick block"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    deep0, shallow0, s0 = adapter.read_blocks([b0], prompt=prompt)

    assert deep0.shape == (1, 16, 64)
    assert shallow0.shape == (1, 16, 64)
    assert s0.frame_count == 1
    assert s0.rebuild_count == 0
    assert len(s0.blocks) == 1

    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    deep1, shallow1, s1 = adapter.read_blocks([b1], prompt=prompt, previous=s0)

    assert s1.frame_count == 2
    assert s1.rebuild_count == 0
    assert len(s1.blocks) == 2


def test_long_history_24_frames_with_consolidation_and_merges():
    # Legacy bounded mode with protect_decision_frames=False: R=2, K=2, slots=2, grid=1
    adapter = make_tiny_memory_adapter(
        recent_frames=2,
        consolidate_every=2,
        memory_slots=2,
        spatial_grid=1,
        protect_decision_frames=False,
    )
    prompt = "long episode"
    N = 24
    blocks = [
        adapter.encode_frame(images=[f"img_{i}"], frame_id=i, prompt=prompt, observation_time=i * 0.1)
        for i in range(N)
    ]

    # Process sequentially one by one
    state = None
    for b in blocks:
        _, _, state = adapter.read_blocks([b], prompt=prompt, previous=state)

    assert state.frame_count == 24
    assert state.last_frame_id == 23
    assert len(state.memory.entries) <= 2
    assert len(state.memory.recent) >= 2
    assert state.memory.consolidations > 0
    assert state.memory.merges > 0
    assert state.rebuild_count > 0


def test_chunking_parity_single_vs_chunks_vs_all():
    # Sequential vs 2-chunk vs all-at-once parity (legacy protect_decision_frames=False)
    adapter = make_tiny_memory_adapter(
        recent_frames=2,
        consolidate_every=2,
        memory_slots=2,
        spatial_grid=1,
        protect_decision_frames=False,
    )
    prompt = "parity task"
    N = 8
    blocks = [
        adapter.encode_frame(images=[f"f_{i}"], frame_id=i, prompt=prompt, observation_time=i * 0.1)
        for i in range(N)
    ]

    # Run all at once
    d_all, s_all, state_all = adapter.read_blocks(blocks, prompt=prompt)

    # Run one by one
    state_seq = None
    d_seq, s_seq = None, None
    for b in blocks:
        d_seq, s_seq, state_seq = adapter.read_blocks([b], prompt=prompt, previous=state_seq)

    # Run in 2 chunks of 4
    _, _, state_c1 = adapter.read_blocks(blocks[:4], prompt=prompt)
    d_chunk, s_chunk, state_chunk = adapter.read_blocks(blocks[4:], prompt=prompt, previous=state_c1)

    assert torch.allclose(d_all, d_seq, atol=1e-5)
    assert torch.allclose(s_all, s_seq, atol=1e-5)
    assert torch.allclose(d_all, d_chunk, atol=1e-5)
    assert torch.allclose(s_all, s_chunk, atol=1e-5)

    # KV and attention mask equality
    assert torch.equal(state_all.attention_mask, state_seq.attention_mask)
    assert torch.equal(state_all.attention_mask, state_chunk.attention_mask)
    for (k1, v1), (k2, v2) in zip(state_all.layer_kv, state_seq.layer_kv):
        assert torch.allclose(k1, k2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)


def test_early_perturbation_changes_memory_summary():
    adapter = make_tiny_memory_adapter(
        recent_frames=2,
        consolidate_every=2,
        memory_slots=2,
        spatial_grid=1,
        protect_decision_frames=False,
    )
    prompt = "perturb task"
    N = 8

    blocks_orig = [
        adapter.encode_frame(images=[f"orig_{i}"], frame_id=i, prompt=prompt, observation_time=i * 0.1)
        for i in range(N)
    ]
    _, _, state_orig = adapter.read_blocks(blocks_orig, prompt=prompt)

    # Perturb frame 0 (early history that gets retired and merged)
    blocks_perturbed = [
        adapter.encode_frame(images=[f"perturbed_{i}" if i == 0 else f"orig_{i}"], frame_id=i, prompt=prompt, observation_time=i * 0.1)
        for i in range(N)
    ]
    _, _, state_pert = adapter.read_blocks(blocks_perturbed, prompt=prompt)

    # Verify early perturbation propagated into memory entries
    e_orig = state_orig.memory.entries[0].visual_tokens
    e_pert = state_pert.memory.entries[0].visual_tokens
    assert not torch.allclose(e_orig, e_pert)


def test_old_state_immutability():
    adapter = make_tiny_memory_adapter(recent_frames=2, consolidate_every=2)
    prompt = "immutability"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    _, _, s0 = adapter.read_blocks([b0], prompt=prompt)

    s0_kv0 = s0.layer_kv[0][0].clone()
    s0_mask = s0.attention_mask.clone()

    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
    _, _, s1 = adapter.read_blocks([b1], prompt=prompt, previous=s0)

    # s0 must not be mutated
    assert torch.equal(s0.layer_kv[0][0], s0_kv0)
    assert torch.equal(s0.attention_mask, s0_mask)
    assert s0.frame_count == 1
    assert s1.frame_count == 2


def test_validator_accepts_valid_and_rejects_foreign_owner():
    adapter1 = make_tiny_memory_adapter(recent_frames=2, consolidate_every=2)
    adapter2 = make_tiny_memory_adapter(recent_frames=2, consolidate_every=2)
    prompt = "val task"

    b0 = adapter1.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0, capture_time=0.0)
    _, _, s0 = adapter1.read_blocks([b0], prompt=prompt)

    obs0 = Observation(
        frame_id=0,
        capture_time=0.0,
        images=[torch.zeros((1, 3, 16, 16))],
        state=torch.zeros((1, 10)),
        state_mask=torch.ones((1, 10), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
        observation_time=0.0,
    )
    snap = (EncodedFrame(observation=obs0, payload=b0, encode_started=0.0, encode_finished=0.01),)

    # Valid candidate check
    validated = validate_memory_cache(s0, previous=None, snapshot=snap, prompt=prompt)
    assert validated is not None
    assert validated.frame_count == 1

    # Foreign block in snapshot
    foreign_b = adapter2.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0, capture_time=0.0)
    foreign_snap = (EncodedFrame(observation=obs0, payload=foreign_b, encode_started=0.0, encode_finished=0.01),)
    with pytest.raises(ValueError, match="payload owner"):
        validate_memory_cache(s0, previous=None, snapshot=foreign_snap, prompt=prompt)


def test_action_head_sample_with_adapter_features():
    adapter = make_tiny_memory_adapter()
    prompt = "sample head"
    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
    deep, shallow, _ = adapter.read_blocks([b0], prompt=prompt)

    actions = adapter.policy.action_head.sample(deep)
    assert actions.shape == (1, 50, 24)


def test_encode_frame_rejects_preserve_visual_tokens_false():
    adapter = make_tiny_memory_adapter()
    prompt = "reject test"
    with pytest.raises(ValueError, match="preserve_visual_tokens"):
        adapter.encode_frame(
            images=["img0"],
            frame_id=0,
            prompt=prompt,
            observation_time=0.0,
            preserve_visual_tokens=False,
        )


def test_snapshot_boundaries_decision_anchor_flow_and_visual_tokens_exactness():
    """Snapshot boundaries [0..8], [9..13], [14..23] => decisions 8, 13, 23.
    - Protected decision frames retired into anchors: 8 and 13.
    - Anchors 8 and 13 must have unpooled full visual tokens [1, 4, 64] torch.equal to original.
    - Last frame of snapshot is protected as is_decision=True.
    - Memory entries summarize non-decision frames without crossing anchor boundaries.
    - State accounting: entries count + len(anchors) + len(recent) == 24.
    - Validator accepts candidate against previous and snapshot.
    - Changing snapshot grouping changes protected set (e.g. [0..5], [6..11], [12..23] => decisions 5, 11, 23).
    """
    adapter = make_tiny_memory_adapter(
        recent_frames=2,
        consolidate_every=2,
        memory_slots=2,
        spatial_grid=1,
        protect_decision_frames=True,
    )
    prompt = "snapshot stream"
    N = 24
    raw_blocks = [
        adapter.encode_frame(images=[f"img_{i}"], frame_id=i, prompt=prompt, observation_time=i * 0.1, capture_time=i * 0.1)
        for i in range(N)
    ]

    # Snapshots: [0..8], [9..13], [14..23] => decisions 8, 13, 23
    chunk1 = raw_blocks[0:9]    # 0..8
    chunk2 = raw_blocks[9:14]   # 9..13
    chunk3 = raw_blocks[14:24]  # 14..23

    # Process chunk 1
    d1, s1, state1 = adapter.read_blocks(chunk1, prompt=prompt)
    assert state1.frame_count == 9
    assert state1.last_frame_id == 8
    assert state1.memory.last_decision_frame_id == 8
    assert state1.memory.recent[-1].frame_id == 8
    assert state1.memory.recent[-1].is_decision is True
    assert state1.memory.recent[-1].visual_tokens.shape == (1, 4, 64)

    # Process chunk 2
    d2, s2, state2 = adapter.read_blocks(chunk2, prompt=prompt, previous=state1)
    assert state2.frame_count == 14
    assert state2.last_frame_id == 13
    assert state2.memory.last_decision_frame_id == 13
    assert state2.memory.recent[-1].frame_id == 13
    assert state2.memory.recent[-1].is_decision is True
    # Frame 8 retired as anchor!
    assert len(state2.memory.anchors) == 1
    assert state2.memory.anchors[0].frame_id == 8
    assert state2.memory.anchors[0].is_decision is True
    assert state2.memory.anchors[0].visual_tokens.shape == (1, 4, 64)
    assert torch.equal(state2.memory.anchors[0].visual_tokens, raw_blocks[8].visual_tokens)

    # Process chunk 3
    d3, s3, state3 = adapter.read_blocks(chunk3, prompt=prompt, previous=state2)
    assert state3.frame_count == 24
    assert state3.last_frame_id == 23
    assert state3.memory.last_decision_frame_id == 23
    assert state3.memory.recent[-1].frame_id == 23
    assert state3.memory.recent[-1].is_decision is True

    # Retired anchors: 8 and 13 (not 0, 10, 20)
    assert len(state3.memory.anchors) == 2
    anchor_ids = [a.frame_id for a in state3.memory.anchors]
    assert anchor_ids == [8, 13]

    for a in state3.memory.anchors:
        orig = raw_blocks[a.frame_id]
        assert a.is_decision is True
        assert a.visual_tokens.shape == (1, 4, 64)
        assert torch.equal(a.visual_tokens, orig.visual_tokens)

    # Entries must not cross any anchor
    for e in state3.memory.entries:
        for a in state3.memory.anchors:
            assert not (e.start_frame_id <= a.frame_id <= e.end_frame_id)

    # Total accounting
    total_acc = sum(e.count for e in state3.memory.entries) + len(state3.memory.anchors) + len(state3.memory.recent)
    assert total_acc == 24

    # Validator accepts candidate state3 with chunk3 snapshot and previous state2
    obs_chunk3 = [
        Observation(
            frame_id=b.frame_id,
            capture_time=b.capture_time,
            images=[torch.zeros((1, 3, 16, 16))],
            state=torch.zeros((1, 10)),
            state_mask=torch.ones((1, 10), dtype=torch.bool),
            action_mask=torch.ones((1, 4), dtype=torch.bool),
            observation_time=b.observation_time,
        )
        for b in chunk3
    ]
    snap3 = tuple(
        EncodedFrame(observation=obs, payload=blk, encode_started=0.0, encode_finished=0.01)
        for obs, blk in zip(obs_chunk3, chunk3)
    )
    val = validate_memory_cache(state3, previous=state2, snapshot=snap3, prompt=prompt)
    assert val.frame_count == 24

    # Changing snapshot grouping changes protected set: [0..5], [6..11], [12..23] => decisions 5, 11, 23
    alt_c1 = raw_blocks[0:6]
    alt_c2 = raw_blocks[6:12]
    alt_c3 = raw_blocks[12:24]
    _, _, alt_s1 = adapter.read_blocks(alt_c1, prompt=prompt)
    _, _, alt_s2 = adapter.read_blocks(alt_c2, prompt=prompt, previous=alt_s1)
    _, _, alt_s3 = adapter.read_blocks(alt_c3, prompt=prompt, previous=alt_s2)
    assert [a.frame_id for a in alt_s3.memory.anchors] == [5, 11]


def test_validator_rejects_boundary_corruption_and_is_decision():
    adapter = make_tiny_memory_adapter(
        recent_frames=2,
        consolidate_every=2,
        memory_slots=2,
        spatial_grid=1,
        protect_decision_frames=True,
    )
    prompt = "reject checks"
    blocks = [
        adapter.encode_frame(images=[f"img_{i}"], frame_id=i, prompt=prompt, observation_time=i * 0.1, capture_time=i * 0.1)
        for i in range(8)
    ]
    chunk1 = blocks[0:5]  # frames 0..4, decision frame 4
    chunk2 = blocks[5:8]  # frames 5..7, decision frame 7
    _, _, s1 = adapter.read_blocks(chunk1, prompt=prompt)
    _, _, s2 = adapter.read_blocks(chunk2, prompt=prompt, previous=s1)

    # In s2, frame 4 has retired as anchor
    assert len(s2.memory.anchors) == 1
    assert s2.memory.anchors[0].frame_id == 4
    assert s2.memory.anchors[0].is_decision is True

    obs_chunk2 = [
        Observation(
            frame_id=b.frame_id,
            capture_time=b.capture_time,
            images=[torch.zeros((1, 3, 16, 16))],
            state=torch.zeros((1, 10)),
            state_mask=torch.ones((1, 10), dtype=torch.bool),
            action_mask=torch.ones((1, 4), dtype=torch.bool),
            observation_time=b.observation_time,
        )
        for b in chunk2
    ]
    snap2 = tuple(
        EncodedFrame(observation=obs, payload=blk, encode_started=0.0, encode_finished=0.01)
        for obs, blk in zip(obs_chunk2, chunk2)
    )

    from fabri_moss.periodic_memory import MemoryEntry, MemoryFrame
    anc = s2.memory.anchors[0]

    # 1. Corrupt anchor is_decision to False -> must reject
    bad_anchor1 = MemoryFrame(
        frame_id=anc.frame_id,
        observation_time=anc.observation_time,
        inputs_embeds=anc.inputs_embeds,
        attention_mask=anc.attention_mask,
        visual_tokens=anc.visual_tokens,
        is_decision=False,
        boundary_frame_id=anc.boundary_frame_id,
    )
    bad_mem1 = copy.copy(s2.memory)
    object.__setattr__(bad_mem1, "anchors", (bad_anchor1,))
    bad_state1 = copy.copy(s2)
    object.__setattr__(bad_state1, "memory", bad_mem1)
    with pytest.raises(ValueError, match="is_decision"):
        validate_memory_cache(bad_state1, previous=s1, snapshot=snap2, prompt=prompt)

    # 2. Corrupt anchor boundary_frame_id -> must reject
    bad_anchor2 = MemoryFrame(
        frame_id=anc.frame_id,
        observation_time=anc.observation_time,
        inputs_embeds=anc.inputs_embeds,
        attention_mask=anc.attention_mask,
        visual_tokens=anc.visual_tokens,
        is_decision=True,
        boundary_frame_id=999,
    )
    bad_mem2 = copy.copy(s2.memory)
    object.__setattr__(bad_mem2, "anchors", (bad_anchor2,))
    bad_state2 = copy.copy(s2)
    object.__setattr__(bad_state2, "memory", bad_mem2)
    with pytest.raises(ValueError, match="boundary_frame_id"):
        validate_memory_cache(bad_state2, previous=s1, snapshot=snap2, prompt=prompt)

    # 3. Corrupt anchor visual tokens -> must reject
    bad_anchor3 = MemoryFrame(
        frame_id=anc.frame_id,
        observation_time=anc.observation_time,
        inputs_embeds=anc.inputs_embeds,
        attention_mask=anc.attention_mask,
        visual_tokens=anc.visual_tokens + 1.0,
        is_decision=True,
        boundary_frame_id=anc.boundary_frame_id,
    )
    bad_mem3 = copy.copy(s2.memory)
    object.__setattr__(bad_mem3, "anchors", (bad_anchor3,))
    bad_state3 = copy.copy(s2)
    object.__setattr__(bad_state3, "memory", bad_mem3)
    with pytest.raises(ValueError, match="visual_tokens content mismatch"):
        validate_memory_cache(bad_state3, previous=s1, snapshot=snap2, prompt=prompt)

    # 4. Corrupt entry boundary_frame_id -> must reject
    ent = s2.memory.entries[0]
    bad_entry1 = MemoryEntry(
        visual_tokens=ent.visual_tokens,
        start_frame_id=ent.start_frame_id,
        end_frame_id=ent.end_frame_id,
        start_time=ent.start_time,
        end_time=ent.end_time,
        count=ent.count,
        boundary_frame_id=999,
    )
    bad_mem4 = copy.copy(s2.memory)
    object.__setattr__(bad_mem4, "entries", (bad_entry1,) + s2.memory.entries[1:])
    bad_state4 = copy.copy(s2)
    object.__setattr__(bad_state4, "memory", bad_mem4)
    with pytest.raises(ValueError, match="boundary_frame_id"):
        validate_memory_cache(bad_state4, previous=s1, snapshot=snap2, prompt=prompt)

    # 5. Corrupt entry to span across anchor frame (anchor is frame 4) -> must reject
    bad_entry2 = MemoryEntry(
        visual_tokens=ent.visual_tokens,
        start_frame_id=0,
        end_frame_id=6,  # covers anchor 4!
        start_time=0.0,
        end_time=0.6,
        count=ent.count,
        boundary_frame_id=ent.boundary_frame_id,
    )
    bad_mem5 = copy.copy(s2.memory)
    object.__setattr__(bad_mem5, "entries", (bad_entry2,) + s2.memory.entries[1:])
    bad_state5 = copy.copy(s2)
    object.__setattr__(bad_state5, "memory", bad_mem5)
    with pytest.raises(ValueError, match="contains anchor frame"):
        validate_memory_cache(bad_state5, previous=s1, snapshot=snap2, prompt=prompt)

    # 6. Corrupt recent frame is_decision -> must reject
    rf = s2.memory.recent[0]
    bad_recent1 = MemoryFrame(
        frame_id=rf.frame_id,
        observation_time=rf.observation_time,
        inputs_embeds=rf.inputs_embeds,
        attention_mask=rf.attention_mask,
        visual_tokens=rf.visual_tokens,
        is_decision=not rf.is_decision,
        boundary_frame_id=rf.boundary_frame_id,
    )
    bad_mem6 = copy.copy(s2.memory)
    object.__setattr__(bad_mem6, "recent", (bad_recent1,) + s2.memory.recent[1:])
    bad_state6 = copy.copy(s2)
    object.__setattr__(bad_state6, "memory", bad_mem6)
    with pytest.raises(ValueError, match="is_decision"):
        validate_memory_cache(bad_state6, previous=s1, snapshot=snap2, prompt=prompt)

    # 7. Corrupt recent frame boundary_frame_id -> must reject
    bad_recent2 = MemoryFrame(
        frame_id=rf.frame_id,
        observation_time=rf.observation_time,
        inputs_embeds=rf.inputs_embeds,
        attention_mask=rf.attention_mask,
        visual_tokens=rf.visual_tokens,
        is_decision=rf.is_decision,
        boundary_frame_id=999,
    )
    bad_mem7 = copy.copy(s2.memory)
    object.__setattr__(bad_mem7, "recent", (bad_recent2,) + s2.memory.recent[1:])
    bad_state7 = copy.copy(s2)
    object.__setattr__(bad_state7, "memory", bad_mem7)
    with pytest.raises(ValueError, match="boundary_frame_id"):
        validate_memory_cache(bad_state7, previous=s1, snapshot=snap2, prompt=prompt)

    # 8. Observation timestamp mismatch -> must reject
    obs_bad = copy.copy(obs_chunk2[-1])
    object.__setattr__(obs_bad, "observation_time", 0.999)
    snap_bad = snap2[:-1] + (EncodedFrame(observation=obs_bad, payload=chunk2[-1], encode_started=0.0, encode_finished=0.01),)
    with pytest.raises(ValueError, match="frame/time mismatch with observation"):
        validate_memory_cache(s2, previous=s1, snapshot=snap_bad, prompt=prompt)


def test_training_wrapper_incremental_query_features_comparison():
    """Compare training wrapper NativeMemorySequencePolicy against incremental adapter.
    Same weights, N=24, decision_indices=[3, 7, 12, 18, 23], targets=[18, 23].
    Check incremental query features vs recomputed training features are allclose in CPU FP32.
    """
    from fabri_moss.memory_training import NativeMemorySequencePolicy
    from fabri_moss.tests.test_memory_training import MockTokenizerForMemory
    from fabri_moss.tests.test_native_training import make_sample, make_tiny_training_policy

    base_seq_policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy = base_seq_policy.policy
    policy.embedder.tokenizer = MockTokenizerForMemory(policy.embedder.tokenizer)

    # Attach single-frame adapter methods for encode_frame on FullDifferentiableEmbedder
    def _preprocess_images(images: Sequence[Any]) -> Tuple[torch.Tensor, List[int]]:
        return policy.embedder._preprocess_images_on_cpu(images)

    def _prepare_and_fuse_embeddings(
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_embeds, batch_masks = policy.embedder._prepare_batch_and_fuse_embeddings(
            prompts=[prompt],
            vit_embeds_batch=[vit_embeds],
            image_masks=[image_mask],
            batch_num_tiles_list=[num_tiles_list],
        )
        return batch_embeds, batch_masks

    policy.embedder._preprocess_images = _preprocess_images
    policy.embedder._prepare_and_fuse_embeddings = _prepare_and_fuse_embeddings

    mem_config = PeriodicMemoryConfig(
        recent_frames=2,
        consolidate_every=2,
        memory_slots=2,
        spatial_grid=1,
        protect_decision_frames=True,
    )
    mem_policy = NativeMemorySequencePolicy(
        policy=policy,
        shallow_layer=1,
        memory_config=mem_config,
        gradient_checkpointing=False,
    )
    mem_policy.eval()

    N = 24
    decision_indices = [3, 7, 12, 18, 23]
    target_indices = [18, 23]
    sample = make_sample(N=N, target_indices=target_indices, horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = decision_indices

    # Compute training wrapper features
    with torch.no_grad():
        deep_train, shallow_train = mem_policy.features(sample)

    assert deep_train.shape == (2, 16, 64)
    assert shallow_train.shape == (2, 16, 64)

    # Build incremental adapter sharing the exact same policy instance and weights
    cache_config = NativeCacheConfig(
        max_frames=4,
        shallow_layer=1,
        use_timestamps=True,
    )
    adapter = NativeMemoryCacheAdapter(policy=policy, config=cache_config, memory_config=mem_config)
    adapter.eval()

    prompt = sample["prompt"]
    # Encode all frames into blocks
    blocks = []
    for i in range(N):
        b = adapter.encode_frame(
            images=[sample["images_window"][i]],
            frame_id=sample["frame_ids"][i],
            prompt=prompt,
            observation_time=sample["observation_times"][i],
            capture_time=sample["observation_times"][i],
        )
        blocks.append(b)

    # Chunk blocks according to decision_indices so each chunk ends at a decision frame:
    # chunk 0: 0..3
    # chunk 1: 4..7
    # chunk 2: 8..12
    # chunk 3: 13..18 (target 18)
    # chunk 4: 19..23 (target 23)
    chunks = []
    start_idx = 0
    for d_idx in decision_indices:
        chunks.append(blocks[start_idx : d_idx + 1])
        start_idx = d_idx + 1

    state = None
    deep_cache_list = []
    shallow_cache_list = []

    for chunk in chunks:
        deep_c, shallow_c, state = adapter.read_blocks(chunk, prompt=prompt, previous=state)
        last_fid = chunk[-1].frame_id
        if last_fid in target_indices:
            deep_cache_list.append(deep_c)
            shallow_cache_list.append(shallow_c)

    deep_cache = torch.cat(deep_cache_list, dim=0)
    shallow_cache = torch.cat(shallow_cache_list, dim=0)

    # Both pathways use identical weights, identical tokenizer & prompt headers, identical anchor/summary prefix
    assert torch.allclose(deep_train, deep_cache, atol=1e-5)
    assert torch.allclose(shallow_train, shallow_cache, atol=1e-5)
