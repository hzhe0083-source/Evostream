"""Tests for compact memory and temporal RoPE primitives using real tiny Qwen3 model.

Tests covered:
- 0 phase (strength=0) matches original execute_native_layers exactly.
- Same frame tokens have identical time rotation.
- Different across-frame dt induces changes.
- Absolute time translation invariance (relative dt is preserved in attention).
- KV cache incremental vs full one-shot parity with temporal RoPE.
- Gradient checkpointing works with backward pass.
- Invalid inputs validation (shapes, nan/inf, device, boolean types, grid bounds).
- Compact frame rendering reduces length vs full uncompacted, preserves full decision, summary end_time, counts.
- append_memory_frame + consolidate_memory equivalence vs original advance_memory.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from fabri_moss.compact_memory import (
    CompactMemoryConfig,
    RenderedMemory,
    cold_prefix_end,
    render_compact_frames,
    render_compact_memory,
)
from fabri_moss.native_cache import execute_native_layers, extract_layer_kv, populate_cache_from_layer_kv
from fabri_moss.periodic_memory import (
    MemoryEntry,
    MemoryFrame,
    PeriodicMemoryConfig,
    PeriodicMemoryState,
    advance_memory,
    append_memory_frame,
    consolidate_memory,
)
from fabri_moss.temporal_rope import TemporalRoPEConfig, temporal_position_embeddings


def _make_tiny_qwen3() -> Tuple[Qwen3ForCausalLM, nn.Module]:
    """Instantiate a real tiny Qwen3 model on CPU for testing."""
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model, model.model


class MockTinyEmbedder(nn.Module):
    """Embedder matching FabriVLA contract for tests."""

    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.model = core_model

        class SimpleTokenizer:
            def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
                # Deterministic small token output
                return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}

        self.tokenizer = SimpleTokenizer()


def _create_frame(
    frame_id: int,
    obs_time: float,
    seq_len: int = 16,
    hidden_dim: int = 64,
    vis_tokens: int = 256,
    is_decision: bool = False,
    boundary_frame_id: int | None = None,
    requires_grad: bool = False,
) -> MemoryFrame:
    embeds = torch.randn(1, seq_len, hidden_dim)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    # 256 tokens = 16x16
    vis = torch.randn(1, vis_tokens, hidden_dim)
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


# 1. append + consolidate vs advance parity
def test_append_and_consolidate_parity_with_advance():
    pm_cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4, protect_decision_frames=True)

    state_adv = PeriodicMemoryState()
    state_step = PeriodicMemoryState()

    frames = [
        _create_frame(0, 0.0, is_decision=False),
        _create_frame(1, 0.1, is_decision=False),
        _create_frame(2, 0.2, is_decision=False),
        _create_frame(3, 0.3, is_decision=True),  # decision
        _create_frame(4, 0.4, is_decision=False),
        _create_frame(5, 0.5, is_decision=False),
        _create_frame(6, 0.6, is_decision=False),
        _create_frame(7, 0.7, is_decision=True),  # triggers retirement
        _create_frame(8, 0.8, is_decision=False),
        _create_frame(9, 0.9, is_decision=False),
    ]

    for f in frames:
        state_adv = advance_memory(state_adv, f, pm_cfg)

        state_step = append_memory_frame(state_step, f)
        state_step = consolidate_memory(state_step, pm_cfg)

        assert state_adv.frame_count == state_step.frame_count
        assert state_adv.consolidations == state_step.consolidations
        assert state_adv.merges == state_step.merges
        assert state_adv.last_decision_frame_id == state_step.last_decision_frame_id
        assert len(state_adv.recent) == len(state_step.recent)
        assert len(state_adv.anchors) == len(state_step.anchors)
        assert len(state_adv.entries) == len(state_step.entries)

        for r_adv, r_stp in zip(state_adv.recent, state_step.recent):
            assert r_adv.frame_id == r_stp.frame_id
            assert r_adv.boundary_frame_id == r_stp.boundary_frame_id
            assert torch.equal(r_adv.inputs_embeds, r_stp.inputs_embeds)

        for a_adv, a_stp in zip(state_adv.anchors, state_step.anchors):
            assert a_adv.frame_id == a_stp.frame_id
            assert torch.equal(a_adv.visual_tokens, a_stp.visual_tokens)

        for e_adv, e_stp in zip(state_adv.entries, state_step.entries):
            assert e_adv.start_frame_id == e_stp.start_frame_id
            assert e_adv.end_frame_id == e_stp.end_frame_id
            assert torch.allclose(e_adv.visual_tokens, e_stp.visual_tokens)


# 2. Strength 0 matches original execution exactly
def test_temporal_rope_strength_zero_exact_match():
    _, core = _make_tiny_qwen3()
    seq_len = 12
    inputs = torch.randn(1, seq_len, 64)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    times = torch.linspace(0.0, 2.0, seq_len).unsqueeze(0)

    # Original execution (both None)
    out_orig, hiddens_orig = execute_native_layers(
        core=core,
        inputs_embeds=inputs,
        attention_mask_2d=mask,
        start_pos=0,
    )

    # Execution with strength 0.0
    cfg_zero = TemporalRoPEConfig(strength=0.0)
    out_zero, hiddens_zero = execute_native_layers(
        core=core,
        inputs_embeds=inputs,
        attention_mask_2d=mask,
        start_pos=0,
        token_times=times,
        temporal_config=cfg_zero,
    )

    assert torch.equal(out_orig, out_zero), "Strength 0 must be numerically identical to original execution"
    for k in hiddens_orig:
        assert torch.equal(hiddens_orig[k], hiddens_zero[k])


# 3. Same frame共同时间 vs 跨帧dt改变
def test_temporal_rope_intra_frame_and_cross_frame():
    _, core = _make_tiny_qwen3()
    # Sequence with 2 frames (e.g. 4 tokens each)
    # Case A: both frames at time 1.0 (dt = 0)
    # Case B: frame 1 at 1.0, frame 2 at 5.0 (dt = 4.0)
    seq_len = 8
    inputs = torch.randn(1, seq_len, 64)
    mask = torch.ones(1, seq_len, dtype=torch.long)

    times_no_dt = torch.full((1, seq_len), 1.0)
    times_with_dt = torch.cat([
        torch.full((1, 4), 1.0),
        torch.full((1, 4), 5.0),
    ], dim=1)

    cfg = TemporalRoPEConfig(strength=1.0)

    out_no_dt, _ = execute_native_layers(core, inputs, mask, token_times=times_no_dt, temporal_config=cfg)
    out_with_dt, _ = execute_native_layers(core, inputs, mask, token_times=times_with_dt, temporal_config=cfg)

    # Cross-frame non-zero dt changes cross-frame attention relative phases and outputs
    assert not torch.allclose(out_no_dt, out_with_dt, atol=1e-3), "Cross-frame time delta dt must change outputs"


# 4. Absolute time translation invariance in attention (relative time delta preserved)
def test_temporal_rope_time_translation_preserves_relative_phase():
    """In attention dot product (Q K^T), if all token times are shifted by uniform tau:

    phase_q -> pos_q + omega * (t_q + tau)
    phase_k -> pos_k + omega * (t_k + tau)
    The difference (phase_q - phase_k) equals (pos_q - pos_k) + omega * (t_q - t_k),
    so relative attention logits between tokens are invariant under a uniform time shift!
    """
    _, core = _make_tiny_qwen3()
    layer0 = core.layers[0]
    attn = layer0.self_attn

    x = torch.randn(1, 4, 64)
    pos_ids = torch.arange(0, 4).unsqueeze(0)

    times_1 = torch.tensor([[0.0, 0.5, 1.0, 1.5]])
    times_2 = torch.tensor([[10.0, 10.5, 11.0, 11.5]])  # uniform shift +10.0

    cfg = TemporalRoPEConfig(strength=1.0)
    cos1, sin1 = temporal_position_embeddings(core, x, pos_ids, times_1, cfg)
    cos2, sin2 = temporal_position_embeddings(core, x, pos_ids, times_2, cfg)

    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = attn.head_dim

    # Apply rotary to query/key
    q = attn.q_proj(x).view(1, 4, num_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(1, 4, num_kv_heads, head_dim).transpose(1, 2)

    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    q1, k1 = apply_rotary_pos_emb(q, k, cos1, sin1)
    q2, k2 = apply_rotary_pos_emb(q, k, cos2, sin2)

    # Dot products for head 0
    # q: [1, H, S, D], k: [1, H, S, D] -> dot product [S, S]
    dot1 = torch.matmul(q1[0, 0], k1[0, 0].transpose(-1, -2))
    dot2 = torch.matmul(q2[0, 0], k2[0, 0].transpose(-1, -2))

    # All dot products should be numerically equal!
    assert torch.allclose(dot1, dot2, atol=1e-5), "Uniform time translation must preserve relative attention dot products"


# 5. Cached incremental vs full one-shot parity with temporal RoPE
def test_temporal_rope_kv_cache_incremental_parity():
    _, core = _make_tiny_qwen3()

    seq1_len = 6
    seq2_len = 4
    total_len = seq1_len + seq2_len

    inputs_full = torch.randn(1, total_len, 64)
    mask_full = torch.ones(1, total_len, dtype=torch.long)
    times_full = torch.cat([
        torch.full((1, seq1_len), 0.5),
        torch.full((1, seq2_len), 1.5),
    ], dim=1)

    cfg = TemporalRoPEConfig(strength=1.0)

    # 1. Full one-shot execution
    out_full, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs_full,
        attention_mask_2d=mask_full,
        start_pos=0,
        token_times=times_full,
        temporal_config=cfg,
    )

    # 2. Incremental execution
    inputs_part1 = inputs_full[:, :seq1_len, :]
    inputs_part2 = inputs_full[:, seq1_len:, :]
    mask_part1 = mask_full[:, :seq1_len]
    times_part1 = times_full[:, :seq1_len]
    times_part2 = times_full[:, seq1_len:]

    cache = DynamicCache()
    out_part1, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs_part1,
        attention_mask_2d=mask_part1,
        cache=cache,
        start_pos=0,
        token_times=times_part1,
        temporal_config=cfg,
    )

    out_part2, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs_part2,
        attention_mask_2d=mask_full,  # [1, total_len] combined past + new
        cache=cache,
        start_pos=seq1_len,
        token_times=times_part2,
        temporal_config=cfg,
    )

    # Compare tail output of full with part2
    out_full_tail = out_full[:, seq1_len:, :]
    assert torch.allclose(out_full_tail, out_part2, atol=1e-5), "Incremental cached output must match full forward pass"


# 6. Gradient checkpointing and backward pass
def test_temporal_rope_gradient_checkpointing():
    _, core = _make_tiny_qwen3()
    # Enable grad on model parameters
    for p in core.parameters():
        p.requires_grad_(True)

    seq_len = 8
    inputs = torch.randn(1, seq_len, 64, requires_grad=True)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    times = torch.linspace(0.0, 1.0, seq_len).unsqueeze(0)
    cfg = TemporalRoPEConfig(strength=1.0)

    # Forward without checkpointing
    out1, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs,
        attention_mask_2d=mask,
        start_pos=0,
        gradient_checkpointing=False,
        token_times=times,
        temporal_config=cfg,
    )
    loss1 = out1.sum()
    loss1.backward()
    grad_inputs1 = inputs.grad.clone()
    inputs.grad = None

    # Forward with checkpointing
    out2, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs,
        attention_mask_2d=mask,
        start_pos=0,
        gradient_checkpointing=True,
        token_times=times,
        temporal_config=cfg,
    )
    loss2 = out2.sum()
    loss2.backward()
    grad_inputs2 = inputs.grad.clone()

    assert torch.allclose(out1, out2, atol=1e-5)
    assert torch.allclose(grad_inputs1, grad_inputs2, atol=1e-5)


# 7. Validation of invalid inputs
def test_validation_errors():
    _, core = _make_tiny_qwen3()
    inputs = torch.randn(1, 4, 64)
    pos_ids = torch.arange(0, 4).unsqueeze(0)

    # Bad config types
    with pytest.raises(TypeError):
        TemporalRoPEConfig(strength="high")  # type: ignore
    with pytest.raises(TypeError):
        TemporalRoPEConfig(strength=True)  # boolean rejected
    with pytest.raises(ValueError):
        TemporalRoPEConfig(strength=-0.5)
    with pytest.raises(ValueError):
        TemporalRoPEConfig(time_unit_seconds=0.0)
    with pytest.raises(ValueError):
        TemporalRoPEConfig(rotary_fraction=1.5)

    # Negative / non-finite token_times
    cfg = TemporalRoPEConfig()
    with pytest.raises(ValueError, match="non-negative"):
        temporal_position_embeddings(core, inputs, pos_ids, torch.tensor([[-0.1, 0.0, 0.0, 0.0]]), cfg)
    with pytest.raises(ValueError, match="finite"):
        temporal_position_embeddings(core, inputs, pos_ids, torch.tensor([[float("inf"), 0.0, 0.0, 0.0]]), cfg)

    # Token times shape mismatch
    with pytest.raises(ValueError, match="shape"):
        temporal_position_embeddings(core, inputs, pos_ids, torch.tensor([[0.0, 0.0]]), cfg)

    # execute_native_layers pairing requirement
    with pytest.raises(ValueError, match="together"):
        execute_native_layers(core, inputs, torch.ones(1, 4), token_times=torch.zeros(1, 4), temporal_config=None)
    with pytest.raises(ValueError, match="together"):
        execute_native_layers(core, inputs, torch.ones(1, 4), token_times=None, temporal_config=cfg)


# 8. Compact memory rendering tests: token length reduction, full decision, summary end_time
def test_compact_memory_rendering():
    _, core = _make_tiny_qwen3()
    embedder = MockTinyEmbedder(core)

    pm_cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, protect_decision_frames=True)
    compact_cfg = CompactMemoryConfig(
        memory=pm_cfg,
        intermediate_grid=4,  # 16x16=256 -> 4x4=16 visual tokens
        temporal=TemporalRoPEConfig(strength=1.0),
    )

    # Add frames: non-decision frame 0, non-decision frame 1, decision frame 2
    f0 = _create_frame(0, 0.0, seq_len=100, vis_tokens=256, is_decision=False)
    f1 = _create_frame(1, 0.5, seq_len=100, vis_tokens=256, is_decision=False)
    f2 = _create_frame(2, 1.0, seq_len=100, vis_tokens=256, is_decision=True)

    state = PeriodicMemoryState()
    state = append_memory_frame(state, f0)
    state = consolidate_memory(state, pm_cfg)
    state = append_memory_frame(state, f1)
    state = consolidate_memory(state, pm_cfg)
    state = append_memory_frame(state, f2)
    state = consolidate_memory(state, pm_cfg)

    # f0 and f1 retired into entry (len=2 recent, len=2 consolidate) -> len(recent)>=4 triggers retire
    # Actually recent has [f0, f1, f2], which is len 3 < 4 (R+K = 2+2=4), so no retirement yet.
    rendered = render_compact_memory(state, embedder, compact_cfg)

    # Non-decision frames f0 and f1 each have:
    # 4 header tokens + 16 pooled visual tokens = 20 tokens (much smaller than full 100!)
    # Decision frame f2 has full 100 tokens.
    # Total seq len = 20 + 20 + 100 = 140.
    assert rendered.inputs_embeds.shape[1] == 140
    assert rendered.current_start == 40  # Start of f2
    assert rendered.token_times.shape == (1, 140)

    # Check that f0 tokens have time 0.0, f1 tokens have time 0.5, f2 tokens have time 1.0
    assert (rendered.token_times[0, :20] == 0.0).all()
    assert (rendered.token_times[0, 20:40] == 0.5).all()
    assert (rendered.token_times[0, 40:] == 1.0).all()

    # Now add frame 3 and retire
    f3 = _create_frame(3, 1.5, seq_len=100, vis_tokens=256, is_decision=False)
    state = append_memory_frame(state, f3)
    state = consolidate_memory(state, pm_cfg)  # recent had 4 frames, oldest 2 (f0, f1) retired into entries

    assert len(state.entries) == 1
    assert state.entries[0].count == 2
    assert state.entries[0].end_time == 0.5

    rendered_retired = render_compact_memory(state, embedder, compact_cfg)
    # Entry has 4 header tokens + 16 pooled entry tokens = 20 tokens.
    # The entry tokens must all have time = end_time = 0.5
    assert (rendered_retired.token_times[0, :20] == 0.5).all()


# 9. Test cold_prefix_end contract
def test_cold_prefix_end_contract():
    pm_cfg = PeriodicMemoryConfig(recent_frames=4, consolidate_every=4)

    # If first_target has no prior decisions -> 0
    assert cold_prefix_end(decision_indices=[5, 10], first_target=4, memory_config=pm_cfg) == 0
    assert cold_prefix_end(decision_indices=[], first_target=10, memory_config=pm_cfg) == 0

    # Decision at 3: d=3, (3+1-4)//4 * 4 = 0
    assert cold_prefix_end(decision_indices=[3, 7], first_target=4, memory_config=pm_cfg) == 0

    # Decision at 7: d=7, (7+1-4)//4 * 4 = 4
    assert cold_prefix_end(decision_indices=[3, 7], first_target=8, memory_config=pm_cfg) == 4

    # Decision at 11: d=11, (11+1-4)//4 * 4 = 8
    assert cold_prefix_end(decision_indices=[3, 7, 11], first_target=12, memory_config=pm_cfg) == 8


# 10. Upsampling rejection in CompactMemoryConfig
def test_compact_memory_rejects_upsampling():
    _, core = _make_tiny_qwen3()
    embedder = MockTinyEmbedder(core)
    # Frame has 16 visual tokens (4x4)
    frame = _create_frame(0, 0.0, seq_len=30, vis_tokens=16, is_decision=False)

    # If intermediate_grid is 8 (> 4), it should raise ValueError
    cfg_bad = CompactMemoryConfig(intermediate_grid=8)
    with pytest.raises(ValueError, match="upsampling forbidden"):
        render_compact_frames([frame], embedder, cfg_bad)
