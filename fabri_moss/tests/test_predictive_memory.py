"""Tests for predictive learned memory writer, consolidation, and numeric rendering."""

import math
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from fabri_moss.periodic_memory import MemoryEntry, MemoryFrame, PeriodicMemoryState, _spatial_pool_visual_tokens
from fabri_moss.predictive_memory import (
    CausalMemoryWriter,
    WriterConfig,
    consolidate_predictive,
    render_predictive_memory,
)


def _make_frame(
    frame_id: int,
    obs_time: float,
    seq_len: int = 16,
    dim: int = 16,
    vis_tokens: int = 4,
    is_decision: bool = False,
    boundary_id: int | None = 0,
    vis_val: float | None = None,
) -> MemoryFrame:
    embeds = torch.randn(1, seq_len, dim)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    vis = torch.full((1, vis_tokens, dim), vis_val) if vis_val is not None else torch.randn(1, vis_tokens, dim)
    return MemoryFrame(frame_id, obs_time, embeds, mask, vis, is_decision, boundary_id)


@pytest.fixture
def cfg() -> WriterConfig:
    return WriterConfig(
        input_dim=16, hidden_dim=16, num_heads=2, num_layers=2,
        grid=1, intermediate_grid=2, recent_frames=2, consolidate_every=2, tbptt_decisions=2,
    )


def test_zero_outproj_equals_count_weighted_mean(cfg: WriterConfig) -> None:
    writer = CausalMemoryWriter(cfg)
    f1 = _make_frame(0, 0.0, vis_val=2.0)
    f2 = _make_frame(1, 1.0, vis_val=4.0)
    entry = writer([f1, f2])
    expected = (_spatial_pool_visual_tokens(f1.visual_tokens, 1) + _spatial_pool_visual_tokens(f2.visual_tokens, 1)) / 2.0
    assert torch.allclose(entry.visual_tokens, expected, atol=1e-5)
    assert entry.count == 2
    assert entry.start_time == 0.0 and entry.end_time == 1.0


def test_two_step_gradient_propagation(cfg: WriterConfig) -> None:
    writer = CausalMemoryWriter(cfg)
    opt = optim.AdamW(writer.parameters(), lr=0.01)
    f1 = _make_frame(0, 0.0)
    f2 = _make_frame(1, 1.0)
    target = torch.randn(1, 1, 16)

    # Step 1: out_proj gradient is non-zero, upstream layers receive zero gradient
    loss1 = nn.functional.mse_loss(writer([f1, f2]).visual_tokens, target)
    loss1.backward()
    assert writer.out_proj.weight.grad is not None and writer.out_proj.weight.grad.abs().sum() > 0
    assert writer.input_proj.weight.grad is not None and writer.input_proj.weight.grad.abs().sum() == 0

    opt.step()
    opt.zero_grad()

    # Step 2: upstream layers receive non-zero gradient after out_proj became non-zero
    loss2 = nn.functional.mse_loss(writer([f1, f2]).visual_tokens, target)
    loss2.backward()
    assert writer.input_proj.weight.grad.abs().sum() > 0


def test_frame_order_sensitivity_with_nonzero_outproj(cfg: WriterConfig) -> None:
    writer = CausalMemoryWriter(cfg)
    nn.init.normal_(writer.out_proj.weight, std=0.5)
    f1 = _make_frame(0, 0.0, vis_val=1.0)
    f2 = _make_frame(1, 1.0, vis_val=3.0)
    e_orig = writer([f1, f2])

    f1_swap = MemoryFrame(0, 0.0, f2.inputs_embeds, f2.attention_mask, f2.visual_tokens, False, 0)
    f2_swap = MemoryFrame(1, 1.0, f1.inputs_embeds, f1.attention_mask, f1.visual_tokens, False, 0)
    e_swap = writer([f1_swap, f2_swap])

    assert not torch.allclose(e_orig.visual_tokens, e_swap.visual_tokens, atol=1e-4)


def test_previous_entry_merge_and_seed_gradient(cfg: WriterConfig) -> None:
    writer = CausalMemoryWriter(cfg)
    seed_vis = torch.randn(1, 1, 16, requires_grad=True)
    prev = MemoryEntry(seed_vis, start_frame_id=0, end_frame_id=1, start_time=0.0, end_time=1.0, count=3, boundary_frame_id=0)
    f_new = _make_frame(2, 2.0)

    merged = writer([f_new], previous=prev)
    assert merged.count == 4
    assert merged.start_frame_id == 0 and merged.end_frame_id == 2
    assert merged.start_time == 0.0 and merged.end_time == 2.0

    loss = merged.visual_tokens.sum()
    loss.backward()
    assert seed_vis.grad is not None and seed_vis.grad.abs().sum() > 0
    assert writer.out_proj.weight.grad is not None and writer.out_proj.weight.grad.abs().sum() > 0


def test_consolidate_protects_anchor_and_conserves_count(cfg: WriterConfig) -> None:
    writer = CausalMemoryWriter(cfg)
    frames = [
        _make_frame(0, 0.0, is_decision=False, boundary_id=0),
        _make_frame(1, 1.0, is_decision=True, boundary_id=0),
        _make_frame(2, 2.0, is_decision=False, boundary_id=1),
        _make_frame(3, 3.0, is_decision=False, boundary_id=1),
        _make_frame(4, 4.0, is_decision=True, boundary_id=1),
    ]
    st = PeriodicMemoryState(recent=tuple(frames), frame_count=5)
    st_cons = consolidate_predictive(st, writer, cfg)

    # Protected anchor check
    assert len(st_cons.anchors) == 1
    anchor = st_cons.anchors[0]
    assert anchor.frame_id == 1 and anchor.is_decision
    assert torch.allclose(anchor.visual_tokens, frames[1].visual_tokens)

    # Boundary separation & count conservation
    assert len(st_cons.entries) == 1
    assert st_cons.entries[0].boundary_frame_id == 0 and st_cons.entries[0].count == 1
    total_retained = len(st_cons.recent) + len(st_cons.anchors) + sum(e.count for e in st_cons.entries)
    assert total_retained == st.frame_count == 5


def test_render_predictive_memory_shapes_and_times(cfg: WriterConfig) -> None:
    anchor_frame = _make_frame(0, 0.0, vis_tokens=4, is_decision=True, boundary_id=0)
    entry = MemoryEntry(torch.randn(1, 1, 16), start_frame_id=1, end_frame_id=1, start_time=1.0, end_time=1.0, count=1, boundary_frame_id=0)
    recent_ordinary = _make_frame(2, 2.0, seq_len=8, vis_tokens=4, is_decision=False, boundary_id=1)
    current_decision = _make_frame(3, 3.0, seq_len=16, vis_tokens=4, is_decision=True, boundary_id=1)

    state = PeriodicMemoryState(
        recent=(recent_ordinary, current_decision),
        entries=(entry,),
        anchors=(anchor_frame,),
        frame_count=4,
    )
    rendered = render_predictive_memory(state, cfg)

    # ordinarypool 4 + entry 1 + anchor 4 + current L 16 = 25 tokens total
    assert rendered.inputs_embeds.shape == (1, 25, 16)
    assert rendered.attention_mask.shape == (1, 25)
    assert rendered.token_times.shape == (1, 25)
    assert rendered.current_start == 9
    assert torch.allclose(rendered.inputs_embeds[:, rendered.current_start:, :], current_decision.inputs_embeds)
    expected_times = [0.0] * 4 + [1.0] * 1 + [2.0] * 4 + [3.0] * 16
    assert torch.allclose(rendered.token_times.squeeze(0), torch.tensor(expected_times, dtype=torch.float32))


def test_rejections_and_invalid_inputs(cfg: WriterConfig) -> None:
    writer = CausalMemoryWriter(cfg)
    dec_frame = _make_frame(0, 0.0, is_decision=True)
    with pytest.raises(ValueError, match="Decision frame"):
        writer([dec_frame])

    f1 = _make_frame(1, 1.0, boundary_id=0)
    f0 = _make_frame(0, 0.0, boundary_id=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        writer([f1, f0])

    f_b0 = _make_frame(0, 0.0, boundary_id=0)
    f_b1 = _make_frame(1, 1.0, boundary_id=1)
    with pytest.raises(ValueError, match="boundary.*mismatch"):
        writer([f_b0, f_b1])

    bad_state = PeriodicMemoryState(recent=(_make_frame(0, 0.0, is_decision=False),))
    with pytest.raises(ValueError, match="must be a decision frame"):
        render_predictive_memory(bad_state, cfg)
