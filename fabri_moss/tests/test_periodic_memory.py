"""Unit tests for periodic memory data structures, consolidation, merging, and materialization."""

import math
import pytest
import torch
import torch.nn as nn

from fabri_moss.periodic_memory import (
    MemoryEntry,
    MemoryFrame,
    PeriodicMemoryConfig,
    PeriodicMemoryState,
    advance_memory,
    detached_state,
    format_frame_timestamp,
    materialize_memory,
    state_nbytes,
    _spatial_pool_visual_tokens,
    _pairwise_token_cosine_similarity,
)


def _make_frame(
    frame_id: int,
    obs_time: float,
    seq_len: int = 24,
    hidden_dim: int = 32,
    num_visual_tokens: int = 16,
    vis_fill: float = 1.0,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
    is_decision: bool = False,
    boundary_frame_id: int | None = None,
) -> MemoryFrame:
    embeds = torch.randn(1, seq_len, hidden_dim, dtype=dtype)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    vis = torch.full((1, num_visual_tokens, hidden_dim), vis_fill, dtype=dtype)
    if requires_grad:
        embeds.requires_grad_(True)
        vis.requires_grad_(True)
    return MemoryFrame(
        frame_id=frame_id,
        observation_time=obs_time,
        inputs_embeds=embeds,
        attention_mask=mask,
        visual_tokens=vis,
        is_decision=is_decision,
        boundary_frame_id=boundary_frame_id,
    )


class MockTokenizer:
    def __init__(self, token_len: int = 8):
        self.token_len = token_len

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        return {"input_ids": torch.ones((1, self.token_len), dtype=torch.long)}


class MockEmbedder(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 100):
        super().__init__()
        self.tokenizer = MockTokenizer()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.model = nn.Module()
        self.model.language_model.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)


def test_config_validation():
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=4, protect_decision_frames=True)
    assert cfg.recent_frames == 4
    assert cfg.protect_decision_frames is True

    # Legacy mode is allowed
    cfg_legacy = PeriodicMemoryConfig(protect_decision_frames=False)
    assert cfg_legacy.protect_decision_frames is False

    # Invalid validations
    with pytest.raises(ValueError, match="recent_frames must be a positive integer"):
        PeriodicMemoryConfig(recent_frames=0)
    with pytest.raises(ValueError, match="consolidate_every must be a positive integer"):
        PeriodicMemoryConfig(consolidate_every=-1)
    with pytest.raises(ValueError, match="memory_slots must be a positive integer"):
        PeriodicMemoryConfig(memory_slots=0)
    with pytest.raises(ValueError, match="spatial_grid must be a positive integer"):
        PeriodicMemoryConfig(spatial_grid=0)
    with pytest.raises(TypeError, match="protect_decision_frames must be a bool"):
        PeriodicMemoryConfig(protect_decision_frames="yes")  # type: ignore


def test_frame_and_entry_validation():
    # Negative time
    with pytest.raises(ValueError, match="finite non-negative float"):
        _make_frame(frame_id=0, obs_time=-1.0)

    # Infinite / NaN time
    with pytest.raises(ValueError, match="finite non-negative float"):
        _make_frame(frame_id=0, obs_time=float("inf"))
    with pytest.raises(ValueError, match="finite non-negative float"):
        _make_frame(frame_id=0, obs_time=float("nan"))

    # Negative or bool frame_id
    with pytest.raises(TypeError, match="frame_id must be a non-negative int"):
        _make_frame(frame_id=-1, obs_time=0.1)
    with pytest.raises(TypeError, match="frame_id must be a non-negative int"):
        _make_frame(frame_id=True, obs_time=0.1)  # type: ignore

    # Non-square visual tokens
    with pytest.raises(ValueError, match="positive perfect square"):
        MemoryFrame(
            frame_id=1,
            observation_time=0.1,
            inputs_embeds=torch.randn(1, 10, 16),
            attention_mask=torch.ones(1, 10, dtype=torch.long),
            visual_tokens=torch.randn(1, 15, 16),  # 15 is not square
        )

    # Length mismatch
    with pytest.raises(ValueError, match="sequence length 10 != attention_mask length 9"):
        MemoryFrame(
            frame_id=1,
            observation_time=0.1,
            inputs_embeds=torch.randn(1, 10, 16),
            attention_mask=torch.ones(1, 9, dtype=torch.long),
            visual_tokens=torch.randn(1, 16, 16),
        )

    # Hidden dim mismatch
    with pytest.raises(ValueError, match="hidden_dim 16 != visual_tokens hidden_dim 32"):
        MemoryFrame(
            frame_id=1,
            observation_time=0.1,
            inputs_embeds=torch.randn(1, 10, 16),
            attention_mask=torch.ones(1, 10, dtype=torch.long),
            visual_tokens=torch.randn(1, 16, 32),
        )

    # MemoryEntry validation
    with pytest.raises(ValueError, match="start_frame_id .* must be <= end_frame_id"):
        MemoryEntry(
            visual_tokens=torch.randn(1, 16, 32),
            start_frame_id=10,
            end_frame_id=5,
            start_time=1.0,
            end_time=2.0,
            count=2,
        )
    with pytest.raises(ValueError, match="start_time .* must be <= end_time"):
        MemoryEntry(
            visual_tokens=torch.randn(1, 16, 32),
            start_frame_id=5,
            end_frame_id=10,
            start_time=2.5,
            end_time=2.0,
            count=2,
        )
    with pytest.raises(ValueError, match="count must be a positive integer"):
        MemoryEntry(
            visual_tokens=torch.randn(1, 16, 32),
            start_frame_id=5,
            end_frame_id=10,
            start_time=1.0,
            end_time=2.0,
            count=0,
        )

    # is_decision and boundary_frame_id validation
    with pytest.raises(TypeError, match="is_decision must be a bool"):
        _make_frame(frame_id=0, obs_time=0.0, is_decision="true")  # type: ignore
    with pytest.raises(TypeError, match="boundary_frame_id must be a non-negative int or None"):
        _make_frame(frame_id=0, obs_time=0.0, boundary_frame_id=-1)
    with pytest.raises(TypeError, match="boundary_frame_id must be a non-negative int or None"):
        MemoryEntry(
            visual_tokens=torch.randn(1, 16, 32),
            start_frame_id=0,
            end_frame_id=0,
            start_time=0.0,
            end_time=0.0,
            count=1,
            boundary_frame_id=-1,
        )


def test_spatial_pooling_known_patterns():
    # 4x4 spatial grid (P=16) pooled to 2x2 (G=2, G^2=4)
    grid_vals = [
        0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
        2.0, 2.0, 3.0, 3.0,
        2.0, 2.0, 3.0, 3.0,
    ]
    tokens = torch.tensor(grid_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 16, 1]
    pooled = _spatial_pool_visual_tokens(tokens, target_grid=2)  # [1, 4, 1]
    expected = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]], dtype=torch.float32)
    torch.testing.assert_close(pooled, expected)

    # 16x16 (P=256) pooled to 4x4 (G=4, G^2=16)
    p256 = torch.zeros(16, 16, 1, dtype=torch.float32)
    for br in range(4):
        for bc in range(4):
            val = float(br * 4 + bc)
            p256[br * 4 : (br + 1) * 4, bc * 4 : (bc + 1) * 4, 0] = val
    tokens256 = p256.view(1, 256, 1)
    pooled256 = _spatial_pool_visual_tokens(tokens256, target_grid=4)  # [1, 16, 1]
    expected16 = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    torch.testing.assert_close(pooled256, expected16)


def test_first_consolidation_at_8_frames():
    # R=4, K=4, protect_decision_frames=False (legacy test path)
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2, protect_decision_frames=False)
    state = PeriodicMemoryState()

    for i in range(1, 8):
        f = _make_frame(frame_id=i, obs_time=float(i) * 0.1, num_visual_tokens=16)
        state = advance_memory(state, f, cfg)
        assert len(state.recent) == i
        assert len(state.entries) == 0
        assert state.consolidations == 0

    # Add 8th frame
    f8 = _make_frame(frame_id=8, obs_time=0.8, num_visual_tokens=16)
    state = advance_memory(state, f8, cfg)
    assert len(state.recent) == 4
    assert [f.frame_id for f in state.recent] == [5, 6, 7, 8]
    assert len(state.entries) == 1
    assert state.consolidations == 1
    assert state.merges == 0
    entry = state.entries[0]
    assert entry.start_frame_id == 1
    assert entry.end_frame_id == 4
    assert entry.start_time == 0.1
    assert entry.end_time == 0.4
    assert entry.count == 4
    assert entry.visual_tokens.shape == (1, 4, 32)


def test_pure_merge_weighted_unequal_counts():
    # Merge entries with counts 4 vs 8 and check weighted average in legacy mode (protect_decision_frames=False)
    cfg = PeriodicMemoryConfig(recent_frames=1, consolidate_every=1, memory_slots=1, spatial_grid=2, protect_decision_frames=False)
    e1 = MemoryEntry(
        visual_tokens=torch.full((1, 4, 16), 2.0),
        start_frame_id=1,
        end_frame_id=4,
        start_time=0.1,
        end_time=0.4,
        count=4,
    )
    e2 = MemoryEntry(
        visual_tokens=torch.full((1, 4, 16), 5.0),
        start_frame_id=5,
        end_frame_id=12,
        start_time=0.5,
        end_time=1.2,
        count=8,
    )
    state = PeriodicMemoryState(entries=(e1, e2), recent=(), frame_count=12, consolidations=2, merges=0)
    f13 = _make_frame(frame_id=13, obs_time=1.3, hidden_dim=16, num_visual_tokens=4, vis_fill=5.0)
    f14 = _make_frame(frame_id=14, obs_time=1.4, hidden_dim=16, num_visual_tokens=4, vis_fill=5.0)
    state = advance_memory(state, f13, cfg)
    assert len(state.entries) == 2
    state = advance_memory(state, f14, cfg)
    assert len(state.entries) == 1
    final_entry = state.entries[0]
    assert final_entry.count == 4 + 8 + 1
    assert final_entry.start_frame_id == 1
    assert final_entry.end_frame_id == 13
    torch.testing.assert_close(final_entry.visual_tokens, torch.full_like(final_entry.visual_tokens, (4 * 2.0 + 9 * 5.0) / 13))


def test_tie_breaking_deterministic_argmax():
    # Legacy mode tie break
    cfg = PeriodicMemoryConfig(recent_frames=1, consolidate_every=1, memory_slots=2, spatial_grid=1, protect_decision_frames=False)
    e0 = MemoryEntry(
        visual_tokens=torch.ones((1, 1, 8)),
        start_frame_id=1,
        end_frame_id=1,
        start_time=0.1,
        end_time=0.1,
        count=1,
    )
    e1 = MemoryEntry(
        visual_tokens=torch.ones((1, 1, 8)),
        start_frame_id=2,
        end_frame_id=2,
        start_time=0.2,
        end_time=0.2,
        count=2,
    )
    state = PeriodicMemoryState(entries=(e0, e1), recent=(), frame_count=2)

    f3 = _make_frame(frame_id=3, obs_time=0.3, hidden_dim=8, num_visual_tokens=1, vis_fill=1.0)
    f4 = _make_frame(frame_id=4, obs_time=0.4, hidden_dim=8, num_visual_tokens=1, vis_fill=1.0)
    state = advance_memory(state, f3, cfg)
    state = advance_memory(state, f4, cfg)

    assert len(state.entries) == 2
    assert state.entries[0].start_frame_id == 1
    assert state.entries[0].end_frame_id == 2
    assert state.entries[0].count == 3
    assert state.entries[1].start_frame_id == 3
    assert state.entries[1].end_frame_id == 3
    assert state.entries[1].count == 1


def test_long_stream_200_frames_legacy_none():
    # Long stream of 200 frames with protect_decision_frames=False:
    # Memory slots bounded at 16
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2, protect_decision_frames=False)
    state = PeriodicMemoryState()

    for i in range(1, 201):
        f = _make_frame(
            frame_id=i,
            obs_time=i * 0.05,
            num_visual_tokens=16,
            vis_fill=float(i % 10),
        )
        state = advance_memory(state, f, cfg)

    assert state.frame_count == 200
    assert len(state.entries) <= cfg.memory_slots
    assert len(state.entries) == 16
    assert len(state.recent) <= cfg.recent_frames + cfg.consolidate_every - 1
    total_entry_count = sum(e.count for e in state.entries)
    total_accounted = total_entry_count + len(state.recent)
    assert total_accounted == 200

    all_ranges = [(e.start_frame_id, e.end_frame_id, e.count) for e in state.entries]
    for idx in range(len(all_ranges) - 1):
        assert all_ranges[idx][1] < all_ranges[idx + 1][0]

    first_recent_id = state.recent[0].frame_id
    last_entry_end_id = state.entries[-1].end_frame_id
    assert last_entry_end_id < first_recent_id
    assert state.recent[-1].frame_id == 200


def test_chunking_grouping_invariance():
    # Same explicit decision flags fed 1 by 1 vs in groups should yield identical state
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2, protect_decision_frames=True)
    decision_fids = {9, 14, 31}
    frames = [
        _make_frame(
            frame_id=i,
            obs_time=i * 0.1,
            num_visual_tokens=16,
            vis_fill=float(i),
            is_decision=(i in decision_fids),
        )
        for i in range(0, 40)
    ]

    state1 = PeriodicMemoryState()
    for f in frames:
        state1 = advance_memory(state1, f, cfg)

    state2 = PeriodicMemoryState()
    for chunk in [frames[:10], frames[10:25], frames[25:]]:
        for f in chunk:
            state2 = advance_memory(state2, f, cfg)

    assert state1.frame_count == state2.frame_count
    assert state1.consolidations == state2.consolidations
    assert state1.merges == state2.merges
    assert len(state1.anchors) == len(state2.anchors)
    assert len(state1.entries) == len(state2.entries)
    for a1, a2 in zip(state1.anchors, state2.anchors):
        assert a1.frame_id == a2.frame_id
        torch.testing.assert_close(a1.visual_tokens, a2.visual_tokens)
    for e1, e2 in zip(state1.entries, state2.entries):
        assert e1.start_frame_id == e2.start_frame_id
        assert e1.end_frame_id == e2.end_frame_id
        assert e1.count == e2.count
        torch.testing.assert_close(e1.visual_tokens, e2.visual_tokens)


def test_immutability_and_forking():
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=8, spatial_grid=2, protect_decision_frames=True)
    state0 = PeriodicMemoryState()
    for i in range(0, 8):
        # Mark frame 0 as decision frame
        f = _make_frame(frame_id=i, obs_time=i * 0.1, num_visual_tokens=16, is_decision=(i == 0))
        state0 = advance_memory(state0, f, cfg)

    # State0 has frame 0 as anchor, frames 1..3 as entry (boundary 0), recent frames 4..7
    assert len(state0.anchors) == 1
    assert state0.anchors[0].frame_id == 0
    assert len(state0.entries) == 1
    assert len(state0.recent) == 4
    entry0_tokens_clone = state0.entries[0].visual_tokens.clone()
    anchor0_tokens_clone = state0.anchors[0].visual_tokens.clone()

    # Fork A: advance with frame 8a
    f8a = _make_frame(frame_id=8, obs_time=0.8, num_visual_tokens=16, vis_fill=10.0)
    state_a = advance_memory(state0, f8a, cfg)

    # Fork B: advance with frame 8b
    f8b = _make_frame(frame_id=8, obs_time=0.8, num_visual_tokens=16, vis_fill=-10.0)
    state_b = advance_memory(state0, f8b, cfg)

    # State0 should remain totally untouched
    assert len(state0.recent) == 4
    assert state0.recent[-1].frame_id == 7
    torch.testing.assert_close(state0.entries[0].visual_tokens, entry0_tokens_clone)
    torch.testing.assert_close(state0.anchors[0].visual_tokens, anchor0_tokens_clone)


def test_early_frame_perturbation_propagates_into_merged_entry():
    # Changing an early historical frame's tokens must result in different merged entry tokens
    cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=2, spatial_grid=2, protect_decision_frames=False)

    frames_base = [
        _make_frame(frame_id=i, obs_time=i * 0.1, num_visual_tokens=16, vis_fill=1.0)
        for i in range(1, 15)
    ]
    frames_perturbed = list(frames_base)
    frames_perturbed[0] = _make_frame(frame_id=1, obs_time=0.1, num_visual_tokens=16, vis_fill=99.0)

    state_base = PeriodicMemoryState()
    for f in frames_base:
        state_base = advance_memory(state_base, f, cfg)

    state_pert = PeriodicMemoryState()
    for f in frames_perturbed:
        state_pert = advance_memory(state_pert, f, cfg)

    assert state_base.merges > 0
    assert not torch.allclose(state_base.entries[0].visual_tokens, state_pert.entries[0].visual_tokens)


def test_detached_state_and_gradients():
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2, protect_decision_frames=True)
    state = PeriodicMemoryState()

    for i in range(0, 8):
        f = _make_frame(
            frame_id=i,
            obs_time=i * 0.1,
            num_visual_tokens=16,
            requires_grad=True,
            is_decision=(i == 0),
        )
        state = advance_memory(state, f, cfg)

    # Frame 0 is anchor, frames 1..3 is entry
    assert state.anchors[0].visual_tokens.requires_grad is False
    assert state.entries[0].visual_tokens.requires_grad is False

    # Recent frames retain original grad requirement
    assert state.recent[0].inputs_embeds.requires_grad is True

    # Detached state detaches all
    det_state = detached_state(state)
    assert det_state.recent[0].inputs_embeds.requires_grad is False
    assert det_state.anchors[0].inputs_embeds.requires_grad is False
    assert det_state.anchors[0].is_decision is True
    assert det_state.entries[0].boundary_frame_id == 0


def test_state_nbytes():
    cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=4, spatial_grid=2, protect_decision_frames=True)
    state = PeriodicMemoryState()
    for i in range(0, 10):
        f = _make_frame(frame_id=i, obs_time=i * 0.1, num_visual_tokens=16, is_decision=(i == 2))
        state = advance_memory(state, f, cfg)

    nbytes = state_nbytes(state)
    assert isinstance(nbytes, int)
    assert nbytes > 0


def test_invalid_advances():
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2)
    f1 = _make_frame(frame_id=5, obs_time=1.0, num_visual_tokens=16)
    state = advance_memory(PeriodicMemoryState(), f1, cfg)

    # Non-increasing frame_id
    with pytest.raises(ValueError, match="frame_id must be strictly increasing"):
        f_bad_id = _make_frame(frame_id=5, obs_time=1.5, num_visual_tokens=16)
        advance_memory(state, f_bad_id, cfg)

    with pytest.raises(ValueError, match="frame_id must be strictly increasing"):
        f_bad_id2 = _make_frame(frame_id=3, obs_time=1.5, num_visual_tokens=16)
        advance_memory(state, f_bad_id2, cfg)

    # Decreasing observation_time
    with pytest.raises(ValueError, match="observation_time must be non-decreasing"):
        f_bad_time = _make_frame(frame_id=6, obs_time=0.5, num_visual_tokens=16)
        advance_memory(state, f_bad_time, cfg)

    # Shape mismatch in visual_tokens
    with pytest.raises(ValueError, match="visual_tokens shape mismatch"):
        f_bad_shape = _make_frame(frame_id=6, obs_time=1.2, num_visual_tokens=4)
        advance_memory(state, f_bad_shape, cfg)

    # Dtype mismatch in visual_tokens
    with pytest.raises(ValueError, match="visual_tokens dtype mismatch"):
        f_bad_dtype = _make_frame(frame_id=6, obs_time=1.2, num_visual_tokens=16, dtype=torch.float64)
        advance_memory(state, f_bad_dtype, cfg)


def test_materialize_memory_legacy_none():
    cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=4, spatial_grid=2, protect_decision_frames=False)
    embedder = MockEmbedder(hidden_dim=32)

    with pytest.raises(ValueError, match="Cannot materialize memory from empty state"):
        materialize_memory(PeriodicMemoryState(), embedder, cfg)

    state = PeriodicMemoryState()
    for i in range(1, 7):
        f = _make_frame(
            frame_id=i,
            obs_time=i * 0.1,
            seq_len=20,
            hidden_dim=32,
            num_visual_tokens=16,
        )
        state = advance_memory(state, f, cfg)

    assert len(state.entries) == 2
    assert len(state.recent) == 2

    inputs_embeds, attention_mask, current_start = materialize_memory(state, embedder, cfg)

    assert inputs_embeds.ndim == 3
    assert inputs_embeds.shape[0] == 1
    assert inputs_embeds.shape[2] == 32
    assert attention_mask.shape == (1, inputs_embeds.shape[1])
    assert inputs_embeds.shape[1] == 64
    assert current_start == 44
    assert torch.all(attention_mask[:, :24] == 1)


# =========================================================================
# New comprehensive tests for explicit decision frame mechanism (N=40, nonconsecutive, etc.)
# =========================================================================

def test_explicit_decision_frames_n40_default_flow():
    """N=40 frames (ids 0..39), default R=4, K=4, protect_decision_frames=True:
    Explicit decision frames at ids 9, 14, 31, 39:
    - Retires frames in batches of 4.
    - Retired decision anchors: 9, 14, 31 (frame 39 remains in recent).
    - Each anchor's visual_tokens must be byte-equal to original unpooled visual_tokens.
    - Frame 39 is still recent and protected; advancing past it later will retire it as an anchor.
    - Summary entries must never cross 9, 14, 31 decision boundaries and must not include decision frames.
    - Boundaries:
        None: frames 0..8
        9: frames 10..13
        14: frames 15..30
        31: frames 32..35 (since 36..39 are in recent)
    - Exact accounting: sum(e.count) + len(anchors) + len(recent) == 40.
    """
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2, protect_decision_frames=True)
    embedder = MockEmbedder(hidden_dim=32)
    state = PeriodicMemoryState()

    decision_ids = {9, 14, 31, 39}
    raw_frames: dict[int, MemoryFrame] = {}
    for i in range(40):
        is_dec = (i in decision_ids)
        f = _make_frame(
            frame_id=i,
            obs_time=i * 0.1,
            seq_len=20,
            hidden_dim=32,
            num_visual_tokens=16,
            vis_fill=float(i + 1),
            is_decision=is_dec,
        )
        raw_frames[i] = f
        state = advance_memory(state, f, cfg)

    assert state.frame_count == 40
    # Recent must be exactly 36..39
    assert [f.frame_id for f in state.recent] == [36, 37, 38, 39]
    # Frame 39 in recent is marked is_decision=True
    assert state.recent[-1].is_decision is True
    assert state.last_decision_frame_id == 39

    # Retired anchors: 9, 14, 31 (frame 39 is not yet retired)
    assert len(state.anchors) == 3
    assert [a.frame_id for a in state.anchors] == [9, 14, 31]

    # Byte-equal unpooled visual tokens for anchors
    for a in state.anchors:
        orig = raw_frames[a.frame_id]
        assert a.visual_tokens.shape == (1, 16, 32)
        assert torch.equal(a.visual_tokens, orig.visual_tokens)
        assert a.is_decision is True
        # Check independent storage
        assert a.visual_tokens.untyped_storage().data_ptr() != orig.visual_tokens.untyped_storage().data_ptr()

    # Entries check:
    # Boundary None: frames 0..8 (count=9)
    # Boundary 9: frames 10..13 (count=4)
    # Boundary 14: frames 15..30 (count=16)
    # Boundary 31: frames 32..35 (count=4)
    assert len(state.entries) == 4
    expected_entries = [
        (0, 8, 9, None),
        (10, 13, 4, 9),
        (15, 30, 16, 14),
        (32, 35, 4, 31),
    ]
    for e, (exp_s, exp_e, exp_cnt, exp_b) in zip(state.entries, expected_entries):
        assert e.start_frame_id == exp_s
        assert e.end_frame_id == exp_e
        assert e.count == exp_cnt
        assert e.boundary_frame_id == exp_b
        assert e.visual_tokens.shape == (1, 4, 32)

    # Accounting
    total_entry_count = sum(e.count for e in state.entries)
    assert total_entry_count + len(state.anchors) + len(state.recent) == 40

    # Advance 4 more frames: 40, 41, 42, 43 -> retires 36..39
    # Frame 39 MUST retire into anchors!
    for next_id in range(40, 44):
        fn = _make_frame(frame_id=next_id, obs_time=next_id * 0.1, num_visual_tokens=16)
        state = advance_memory(state, fn, cfg)

    assert len(state.anchors) == 4
    assert [a.frame_id for a in state.anchors] == [9, 14, 31, 39]
    # Check entry for boundary 31: now merged with 36, 37, 38 (count 4 + 3 = 7, start=32, end=38)
    e_last = [e for e in state.entries if e.boundary_frame_id == 31][0]
    assert e_last.start_frame_id == 32
    assert e_last.end_frame_id == 38
    assert e_last.count == 7

    # Materialize memory on initial state (40 frames) and verify ordering
    state_40 = PeriodicMemoryState()
    for i in range(40):
        state_40 = advance_memory(state_40, raw_frames[i], cfg)
    inputs_embeds, attention_mask, current_start = materialize_memory(state_40, embedder, cfg)
    assert inputs_embeds.ndim == 3
    assert inputs_embeds.shape[0] == 1
    assert inputs_embeds.shape[2] == 32
    # Prefix ordering:
    # summary 0..8 -> anchor 9 -> summary 10..13 -> anchor 14 -> summary 15..30 -> anchor 31 -> summary 32..35
    # Total prefix: 3 anchors (3*24=72), 4 entries (4*12=48) -> 120 tokens
    # 4 recent frames: 4 * 20 = 80 tokens -> total seq_len = 200 tokens
    assert inputs_embeds.shape[1] == 200
    assert current_start == 200 - 20


def test_decision_frames_n200_no_slot_clipping():
    """N=200 stream with decision frames:
    - Does NOT drop or clip to memory_slots=16 when protect_decision_frames=True.
    - Decisions at 0, 10, 20, ..., 190 (20 decisions).
    - Every retired decision frame is preserved as an unpooled exact anchor.
    - Exact accounting: sum(e.count) + len(anchors) + len(recent) == 200.
    """
    cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, memory_slots=16, spatial_grid=2, protect_decision_frames=True)
    state = PeriodicMemoryState()

    raw_anchors = {}
    for i in range(200):
        is_dec = (i % 10 == 0)
        f = _make_frame(
            frame_id=i,
            obs_time=i * 0.05,
            num_visual_tokens=16,
            vis_fill=float(i + 1),
            is_decision=is_dec,
        )
        if is_dec:
            raw_anchors[i] = f.visual_tokens.clone()
        state = advance_memory(state, f, cfg)

    assert state.frame_count == 200
    # Retired anchors: 0, 10, ..., 190 -> 20 anchors
    # (Frame 196..199 are recent, frame 190 was retired)
    assert len(state.anchors) >= 19
    retired_anchor_ids = [a.frame_id for a in state.anchors]
    for exp_id in range(0, 191, 10):
        assert exp_id in retired_anchor_ids
        # Exact match
        matching_anchor = next(a for a in state.anchors if a.frame_id == exp_id)
        torch.testing.assert_close(matching_anchor.visual_tokens, raw_anchors[exp_id])
        assert matching_anchor.is_decision is True

    # Ensure entries were not capped at 16
    assert len(state.entries) >= 19

    # Exact accounting
    total_entry_count = sum(e.count for e in state.entries)
    assert total_entry_count + len(state.anchors) + len(state.recent) == 200


def test_nonconsecutive_ids_and_decisions():
    """Non-consecutive frame IDs [2, 7, 100, 101, 205] with decisions at 7 and 205.
    - R=1, K=1
    - Step by step:
      fid 2 (non-dec): recent [2]
      fid 7 (decision): retires 2 -> entry start=2, end=2, count=1, boundary=None; recent [7 (dec)]
      fid 100 (non-dec): retires 7 (dec) -> anchor 7; recent [100 (boundary=7)]
      fid 101 (non-dec): retires 100 -> entry start=100, end=100, count=1, boundary=7; recent [101 (boundary=7)]
      fid 205 (decision): retires 101 -> merges into boundary 7 entry -> entry start=100, end=101, count=2, boundary=7; recent [205 (dec)]
    - Counts and boundaries must be exact.
    """
    cfg = PeriodicMemoryConfig(recent_frames=1, consolidate_every=1, memory_slots=10, spatial_grid=2, protect_decision_frames=True)
    frames_info = [
        (2, False),
        (7, True),
        (100, False),
        (101, False),
        (205, True),
    ]
    state = PeriodicMemoryState()
    for idx, (fid, is_dec) in enumerate(frames_info):
        f = _make_frame(frame_id=fid, obs_time=idx * 0.1, num_visual_tokens=16, is_decision=is_dec)
        state = advance_memory(state, f, cfg)

    assert state.frame_count == 5
    # Anchors: frame 7 was retired when frame 100 was added
    assert len(state.anchors) == 1
    assert state.anchors[0].frame_id == 7
    assert state.anchors[0].is_decision is True

    # Recent: frame 205 (which is decision=True)
    assert len(state.recent) == 1
    assert state.recent[0].frame_id == 205
    assert state.recent[0].is_decision is True
    assert state.last_decision_frame_id == 205

    # Entries:
    # Entry 0 (boundary=None): frame 2 (count=1)
    # Entry 1 (boundary=7): frames 100, 101 (count=2, start=100, end=101)
    assert len(state.entries) == 2
    assert state.entries[0].start_frame_id == 2
    assert state.entries[0].end_frame_id == 2
    assert state.entries[0].count == 1
    assert state.entries[0].boundary_frame_id is None

    assert state.entries[1].start_frame_id == 100
    assert state.entries[1].end_frame_id == 101
    assert state.entries[1].count == 2
    assert state.entries[1].boundary_frame_id == 7

    # Exact accounting
    assert sum(e.count for e in state.entries) + len(state.anchors) + len(state.recent) == 5


def test_early_anchor_perturbation_propagates_into_materialized():
    """Perturbing an early decision anchor keyframe must directly propagate to the materialized sequence."""
    cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=10, spatial_grid=2, protect_decision_frames=True)
    embedder = MockEmbedder(hidden_dim=32)

    frames_base = [
        _make_frame(frame_id=i, obs_time=i * 0.1, num_visual_tokens=16, vis_fill=1.0, is_decision=(i == 0))
        for i in range(12)
    ]
    frames_pert = list(frames_base)
    # Perturb frame 0 (which is an anchor)
    frames_pert[0] = _make_frame(frame_id=0, obs_time=0.0, num_visual_tokens=16, vis_fill=999.0, is_decision=True)

    state_base = PeriodicMemoryState()
    for f in frames_base:
        state_base = advance_memory(state_base, f, cfg)

    state_pert = PeriodicMemoryState()
    for f in frames_pert:
        state_pert = advance_memory(state_pert, f, cfg)

    # Frame 0 retired into anchors[0]
    assert state_base.anchors[0].frame_id == 0
    assert state_pert.anchors[0].frame_id == 0
    assert not torch.allclose(state_base.anchors[0].visual_tokens, state_pert.anchors[0].visual_tokens)

    # Materialize and verify difference in the anchor tokens position
    emb_base, _, _ = materialize_memory(state_base, embedder, cfg)
    emb_pert, _, _ = materialize_memory(state_pert, embedder, cfg)

    # The anchor visual tokens follow the first 8 header tokens
    torch.testing.assert_close(emb_base[:, 8:24], torch.full((1, 16, 32), 1.0))
    torch.testing.assert_close(emb_pert[:, 8:24], torch.full((1, 16, 32), 999.0))


def test_different_decision_flags_yield_different_anchors():
    """Different decision flags must produce different anchors and summaries."""
    cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=10, spatial_grid=2, protect_decision_frames=True)
    # Stream A: decisions at 0, 4
    # Stream B: decisions at 2, 4
    frames_a = [
        _make_frame(frame_id=i, obs_time=i * 0.1, num_visual_tokens=16, is_decision=(i in {0, 4}))
        for i in range(8)
    ]
    frames_b = [
        _make_frame(frame_id=i, obs_time=i * 0.1, num_visual_tokens=16, is_decision=(i in {2, 4}))
        for i in range(8)
    ]

    state_a = PeriodicMemoryState()
    for f in frames_a:
        state_a = advance_memory(state_a, f, cfg)

    state_b = PeriodicMemoryState()
    for f in frames_b:
        state_b = advance_memory(state_b, f, cfg)

    # Stream A retired anchors: 0, 4
    # Stream B retired anchors: 2, 4
    assert [a.frame_id for a in state_a.anchors] == [0, 4]
    assert [a.frame_id for a in state_b.anchors] == [2, 4]
    assert state_a.anchors != state_b.anchors


def test_frame_ids_remain_original_unrenumbered():
    """Frame IDs after consolidation must remain the original input IDs, never renumbered by array position."""
    cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=10, spatial_grid=2, protect_decision_frames=True)
    # Gaps in frame IDs, decisions at 0, 15
    ids = [0, 5, 10, 15, 20, 25]
    state = PeriodicMemoryState()
    for idx, fid in enumerate(ids):
        f = _make_frame(frame_id=fid, obs_time=idx * 0.1, num_visual_tokens=16, is_decision=(fid in {0, 15}))
        state = advance_memory(state, f, cfg)

    # Check anchors
    assert all(a.frame_id in (0, 15) for a in state.anchors)
    # Check entries
    for e in state.entries:
        assert e.start_frame_id in ids
        assert e.end_frame_id in ids
