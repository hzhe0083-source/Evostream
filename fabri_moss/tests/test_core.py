"""Comprehensive unit tests for fabri_moss.

Tests:
1. Tiny Qwen3 fixture (native-like architecture without remote checkpoints)
2. Zero gates equivalence to native_core readout positions
3. Gates opened -> gradients on K/V projections, gates, and readout
4. Frozen expert / frozen embedder backprop passing gradients to bridge
5. Strict single-view / single-tile enforcement
6. Bounded VisionSession caching, fresh vs cached parity, eviction, reset
7. Foreign owner and stale revision rejection
8. Frame ordering and strict monotonicity checks
9. Mixed precision (bfloat16 native + float32 cross) gradient flow
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.core import (
    FrameKV,
    MossConfig,
    MossInternVL,
    VisionSession,
    build_causal_cross_mask,
    compute_3d_rope,
    apply_3d_rotary_pos_emb,
)


# ============================================================================
# Tiny Fixture Architecture
# ============================================================================

class TinyQwen3Attention(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, num_kv_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            while cos.ndim < q.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            if cos.shape[1] == 1 and q.shape[1] > 1:
                pass
            elif cos.shape[1] != q.shape[1]:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=scale
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), None

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class TinyQwen3MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, intermediate_size: int = 256):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyQwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, num_kv_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.self_attn = TinyQwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = TinyQwen3MLP(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(normed, attention_mask=attention_mask, position_embeddings=position_embeddings)
        hidden_states = residual + attn_out

        residual = hidden_states
        normed_mlp = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp)
        hidden_states = residual + mlp_out
        return (hidden_states,)


class TinyRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int = 32, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = torch.outer(position_ids.squeeze(0).float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


class TinyConfig:
    def __init__(self, hidden_size: int = 128, num_hidden_layers: int = 6):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class TinyNativeCore(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 6, vocab_size: int = 256):
        super().__init__()
        self.config = TinyConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyQwen3DecoderLayer(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.rotary_emb = TinyRotaryEmbedding(head_dim=32)


class TinyTokenizer:
    def __init__(self):
        pass

    def __call__(self, text: str, return_tensors: str = "pt"):
        class Output:
            pass
        tokens = [ord(c) % 256 for c in text]
        if not tokens:
            tokens = [1]
        out = Output()
        out.input_ids = torch.tensor([tokens], dtype=torch.long)
        return out


class TinyEmbedder(nn.Module):
    def __init__(self, native_core: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.model = native_core
        self.tokenizer = TinyTokenizer()
        self.img_context_token_id = 42

        # Mock ViT
        self.vision_proj = nn.Linear(64, native_core.config.hidden_size)
        self.model.extract_feature = self._mock_extract_feature
        self._preprocess_tiles = [1]
        self._preprocess_calls = 0
        self._extract_calls = 0

    def _mock_extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._extract_calls += 1
        # pixel_values shape [1, 64] -> [1, 16, 128] (16 tokens, square 4x4)
        return self.vision_proj(pixel_values).unsqueeze(1).repeat(1, 16, 1)

    def _preprocess_images(self, images: List[Any]) -> Tuple[torch.Tensor, List[int]]:
        self._preprocess_calls += 1
        img_key = str(images[0]) if images else "default"
        seed = int(hashlib.sha256(img_key.encode("utf-8")).hexdigest()[:8], 16)
        gen = torch.Generator()
        gen.manual_seed(seed)
        return torch.randn(1, 64, generator=gen), list(self._preprocess_tiles)


class TinyActionHead(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 4)

    def forward(self, deep_tokens: torch.Tensor, shallow_tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(deep_tokens.mean(dim=1) + shallow_tokens.mean(dim=1))


class TinyFabriVLAPolicy(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 6):
        super().__init__()
        self.core = TinyNativeCore(hidden_size=hidden_size, num_layers=num_layers)
        self.embedder = TinyEmbedder(self.core)
        self.action_head = TinyActionHead(hidden_size=hidden_size)


# ============================================================================
# Unit Tests
# ============================================================================


def test_causal_cross_mask_expands_frame_tokens():
    mask = build_causal_cross_mask(
        key_times=[0.0, 1.0, 2.0],
        key_token_counts=[2, 1, 2],
        query_times=[0.0, 1.0, 2.0],
    )
    assert mask.shape == (1, 1, 3, 5)
    # Query at t=0 sees only frame 0's two patch tokens.
    assert torch.isfinite(mask[0, 0, 0, :2]).all()
    assert torch.isneginf(mask[0, 0, 0, 2:]).all()
    # Query at t=1 additionally sees frame 1, but not frame 2.
    assert torch.isfinite(mask[0, 0, 1, :3]).all()
    assert torch.isneginf(mask[0, 0, 1, 3:]).all()
    assert torch.isfinite(mask[0, 0, 2]).all()


def test_causal_cross_mask_rejects_mismatched_metadata():
    with pytest.raises(ValueError, match="equal length"):
        build_causal_cross_mask([0.0], [1, 1], [0.0])
    with pytest.raises(ValueError, match="finite"):
        build_causal_cross_mask([0.0], [1], [float("nan")])

@pytest.fixture
def tiny_setup():
    torch.manual_seed(42)
    policy = TinyFabriVLAPolicy(hidden_size=128, num_layers=6)
    config = MossConfig(
        cross_layers=(2, 4, 6),
        num_readout_tokens=8,
        max_frames=2,
        shallow_layer=4,
        max_text_tokens=64,
    )
    moss = MossInternVL(policy, config=config)
    return policy, moss, config


def test_zero_gates_native_parity(tiny_setup):
    """When gates are zero, readout positions must match native_core pure forward on identical prompt+readout sequence."""
    policy, moss, config = tiny_setup
    moss.eval()

    prompt = "move robot"
    features = torch.randn(1, 16, 128)
    frame = moss.project_frame(features, frame_id=0)

    # 1. Moss read_memory
    with torch.no_grad():
        deep_moss, shallow_moss = moss.read_memory([frame], prompt)

    # 2. Native forward on exact same tokens
    tokenizer = policy.embedder.tokenizer
    tokens = tokenizer(prompt.strip(), return_tensors="pt")
    text_embeds = moss.native_core.embed_tokens(tokens.input_ids)
    readout_embeds = moss.readout_embeddings.unsqueeze(0)
    full_embeds = torch.cat([text_embeds, readout_embeds], dim=1)

    total_seq_len = full_embeds.shape[1]
    causal_mask = torch.triu(
        torch.full((total_seq_len, total_seq_len), float("-inf")),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    pos_ids = torch.arange(total_seq_len, dtype=torch.long).unsqueeze(0)
    pos_emb = moss.native_core.rotary_emb(full_embeds, pos_ids)

    hs = full_embeds
    shallow_native = None
    with torch.no_grad():
        for l_idx, layer in enumerate(moss.native_core.layers):
            hs = layer(hs, attention_mask=causal_mask, position_embeddings=pos_emb)[0]
            if (l_idx + 1) == config.shallow_layer:
                shallow_native = hs[:, -config.num_readout_tokens :, :].float()
        deep_native = moss.native_core.norm(hs)[:, -config.num_readout_tokens :, :].float()

    assert torch.allclose(deep_moss, deep_native, atol=1e-5), "Zero-gate deep output must match native"
    assert torch.allclose(shallow_moss, shallow_native, atol=1e-5), "Zero-gate shallow output must match native"


def test_zero_gates_preserve_full_native_multimodal_sequence(tiny_setup):
    """Native image fusion must stay full length; only historical KV is optional."""
    policy, moss, config = tiny_setup
    moss.eval()
    embedder = policy.embedder

    # Emulate the real InternVL embedder contract (fused image tokens plus
    # right-padding).  The old path appended 16 synthetic readout tokens and
    # discarded this mask, so it could never be gate=0 equivalent.
    embedder._build_multimodal_prompt = lambda tiles, prompt: prompt

    def fuse(*, prompts, vit_embeds_batch, image_masks, batch_num_tiles_list):
        seq_len = 24
        ids = torch.arange(seq_len).unsqueeze(0)
        fused = policy.core.embed_tokens(ids).float()
        fused[:, : vit_embeds_batch[0].shape[1], :] = vit_embeds_batch[0]
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        mask[:, -4:] = False
        return fused, mask

    embedder._prepare_batch_and_fuse_embeddings = fuse
    features = torch.randn(1, 16, 128)
    frame = moss.project_frame(features, frame_id=0)
    with torch.no_grad():
        deep, shallow = moss.read_memory([frame], "move robot")

        native = fuse(
            prompts=["move robot"],
            vit_embeds_batch=[features],
            image_masks=[torch.ones(1, dtype=torch.bool)],
            batch_num_tiles_list=[[1]],
        )
        inputs, mask = native
        h = inputs.to(dtype=policy.core.embed_tokens.weight.dtype)
        seq_len = h.shape[1]
        q = torch.arange(seq_len).view(seq_len, 1)
        k = torch.arange(seq_len).view(1, seq_len)
        allowed = (k <= q) & mask.bool().view(1, seq_len)
        attn_mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(h.dtype).min)
        attn_mask.masked_fill_(allowed.view(1, 1, seq_len, seq_len), 0.0)
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        pos = policy.core.rotary_emb(h, pos_ids)
        hs, native_shallow = h, None
        for idx, layer in enumerate(policy.core.layers, start=1):
            hs = layer(hs, attention_mask=attn_mask, position_embeddings=pos)[0]
            if idx == config.shallow_layer:
                native_shallow = hs.float()
        native_deep = policy.core.norm(hs).float()

    assert deep.shape == (1, seq_len, 128)
    assert shallow.shape == (1, seq_len, 128)
    assert torch.allclose(deep, native_deep, atol=1e-5)
    assert torch.allclose(shallow, native_shallow, atol=1e-5)


def _native_query_inputs(moss, prompt: str = "move robot"):
    toks = moss.policy.embedder.tokenizer(prompt, return_tensors="pt")
    text = moss.native_core.embed_tokens(toks.input_ids)
    readout = moss.readout_embeddings.unsqueeze(0).to(dtype=text.dtype)
    embeds = torch.cat([text, readout], dim=1)
    return embeds, torch.ones(1, embeds.shape[1], dtype=torch.bool)


def test_native_query_batch_matches_single_and_hides_future(tiny_setup):
    _, moss, config = tiny_setup
    moss.eval()
    features0 = torch.randn(1, 16, 128)
    features1 = torch.randn(1, 16, 128)
    f0 = moss.project_frame(features0, 0)
    f1 = moss.project_frame(features1, 1)
    embeds, mask = _native_query_inputs(moss)
    batch_embeds = embeds.expand(2, -1, -1).clone()
    batch_mask = mask.expand(2, -1).clone()

    with torch.no_grad():
        deep_b, shallow_b = moss.read_native_queries_batch(
            [[f0], [f0, f1]], batch_embeds, batch_mask, [0, 1]
        )
        deep0, shallow0 = moss.read_native_queries([f0], embeds, mask, 0)
        deep1, shallow1 = moss.read_native_queries([f0, f1], embeds, mask, 1)
    assert torch.allclose(deep_b[0:1], deep0, atol=1e-5)
    assert torch.allclose(deep_b[1:2], deep1, atol=1e-5)
    assert torch.allclose(shallow_b[0:1], shallow0, atol=1e-5)
    assert torch.allclose(shallow_b[1:2], shallow1, atol=1e-5)

    # Open the cross gate and verify a future frame cannot affect a past query.
    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.2)
    with torch.no_grad():
        deep_future, _ = moss.read_native_queries([f0, f1], embeds, mask, 0)
        deep_past, _ = moss.read_native_queries([f0], embeds, mask, 0)
    assert torch.allclose(deep_future, deep_past, atol=1e-5)


def test_native_query_batch_cross_gradients(tiny_setup):
    _, moss, _ = tiny_setup
    moss.train(True)
    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.2)
    f0 = moss.project_frame(torch.randn(1, 16, 128), 0)
    embeds, mask = _native_query_inputs(moss)
    deep, shallow = moss.read_native_queries_batch([[f0]], embeds, mask, [0])
    (deep.square().mean() + shallow.square().mean()).backward()
    first = moss.cross_blocks[str(moss.config.cross_layers[0])]
    assert first.k_proj.weight.grad is not None and torch.isfinite(first.k_proj.weight.grad).all()
    assert first.v_proj.weight.grad is not None and torch.isfinite(first.v_proj.weight.grad).all()


def test_project_frame_metadata_appends_one_memory_token(tiny_setup):
    _, moss, _ = tiny_setup
    features = torch.randn(1, 16, 128)
    frame = moss.project_frame(features, frame_id=3, observation_time=1.25)
    assert frame.num_tokens == 17
    assert all(k.shape[2] == 17 and v.shape[2] == 17 for k, v in zip(frame.keys, frame.values))


def test_gate_opened_cross_gradients(tiny_setup):
    """When gates are opened, gradients flow to cross attention weights, gates, and readout embeddings."""
    policy, moss, config = tiny_setup
    moss.train(True)
    moss.set_training_stage("bridge")

    # Set non-zero gates
    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.5)
        block.mlp_gate.data.fill_(0.5)

    features = torch.randn(1, 16, 128, requires_grad=True)
    frame = moss.project_frame(features, frame_id=0)

    deep, shallow = moss.read_memory([frame], prompt="grasp cup")
    loss = deep.sum() + shallow.sum()
    loss.backward()

    # Bridge params have gradients
    assert moss.readout_embeddings.grad is not None
    assert moss.readout_embeddings.grad.abs().sum() > 0

    first_block = moss.cross_blocks["2"]
    assert first_block.attn_gate.grad is not None
    assert first_block.k_proj.weight.grad is not None
    assert first_block.k_proj.weight.grad.abs().sum() > 0

    # Policy base parameters remain frozen
    for p in policy.core.parameters():
        assert p.grad is None


def test_frozen_expert_backprop(tiny_setup):
    """Gradients pass through frozen action_head to bridge parameters."""
    policy, moss, config = tiny_setup
    moss.train(True)
    moss.set_training_stage("bridge")

    # Action head is frozen
    for p in policy.action_head.parameters():
        assert not p.requires_grad

    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.5)

    features = torch.randn(1, 16, 128)
    frame = moss.project_frame(features, frame_id=1)
    deep, shallow = moss.read_memory([frame], prompt="lift object")

    action_pred = policy.action_head(deep, shallow)
    loss = action_pred.sum()
    loss.backward()

    # Bridge params updated through action head
    assert moss.readout_embeddings.grad is not None
    assert torch.isfinite(moss.readout_embeddings.grad).all()
    assert (moss.readout_embeddings.grad != 0).any()

    first_block = moss.cross_blocks[str(config.cross_layers[0])]
    assert first_block.k_proj.weight.grad is not None
    assert torch.isfinite(first_block.k_proj.weight.grad).all()
    assert (first_block.k_proj.weight.grad != 0).any()
    assert first_block.v_proj.weight.grad is not None
    assert torch.isfinite(first_block.v_proj.weight.grad).all()
    assert (first_block.v_proj.weight.grad != 0).any()

    # Action head params received NO gradient
    for p in policy.action_head.parameters():
        assert p.grad is None


def test_strict_single_view_tile_check(tiny_setup):
    """encode_image strictly validates single-view and single-tile."""
    policy, moss, _ = tiny_setup

    # Valid: single tile
    policy.embedder._preprocess_tiles = [1]
    feats = moss.encode_image(["dummy.png"])
    assert feats.shape == (1, 16, 128)

    # Invalid: multi-tile
    policy.embedder._preprocess_tiles = [2]
    with pytest.raises(ValueError, match="Strict single-view / single-tile required"):
        moss.encode_image(["dummy.png"])

    # Invalid: multiple views
    policy.embedder._preprocess_tiles = [1, 1]
    with pytest.raises(ValueError, match="Strict single-view / single-tile required"):
        moss.encode_image(["img1.png", "img2.png"])


def test_vision_session_cache_and_parity(tiny_setup):
    """VisionSession caches KV, achieves identical output to fresh forward, and evicts properly."""
    policy, moss, config = tiny_setup
    moss.eval()

    # Open gates to 0.1 so cross-attention is genuinely active
    with torch.no_grad():
        for block in moss.cross_blocks.values():
            block.attn_gate.fill_(0.1)
            block.mlp_gate.fill_(0.1)

    # Hooks to monitor k_proj/v_proj and native layer invocations
    num_cross_layers = len(config.cross_layers)
    cross_kv_calls = 0
    native_layer_calls = 0

    def cross_kv_hook(module, inp, out):
        nonlocal cross_kv_calls
        cross_kv_calls += 1

    def native_layer_hook(module, inp, out):
        nonlocal native_layer_calls
        native_layer_calls += 1

    for block in moss.cross_blocks.values():
        block.k_proj.register_forward_hook(cross_kv_hook)
        block.v_proj.register_forward_hook(cross_kv_hook)

    for layer in moss.native_core.layers:
        layer.register_forward_hook(native_layer_hook)

    session = VisionSession(moss)
    session.reset(episode_id="ep_001", prompt="clean table")

    with torch.no_grad():
        # Frame 1 append: must call k_proj/v_proj 2*layers times, native layers 0 times
        prev_cross = cross_kv_calls
        prev_native = native_layer_calls
        session.append(["img0.png"], frame_id=0)
        assert len(session.frames) == 1
        assert cross_kv_calls - prev_cross == 2 * num_cross_layers
        assert native_layer_calls == prev_native

        # Frame 1 query: clone keys/values before query, verify equal after query, cross hook unchanged
        cloned_kv1 = [
            (tuple(k.clone() for k in f.keys), tuple(v.clone() for v in f.values))
            for f in session.frames
        ]
        cnt_before_q1 = cross_kv_calls
        d1, s1 = session.query()
        assert cross_kv_calls == cnt_before_q1
        for (orig_k, orig_v), f in zip(cloned_kv1, session.frames):
            for ok, fk in zip(orig_k, f.keys):
                assert torch.equal(ok, fk)
            for ov, fv in zip(orig_v, f.values):
                assert torch.equal(ov, fv)

        # Frame 2 append: must call k_proj/v_proj 2*layers times, native layers 0 times
        prev_cross = cross_kv_calls
        prev_native = native_layer_calls
        session.append(["img1.png"], frame_id=1)
        assert len(session.frames) == 2
        assert cross_kv_calls - prev_cross == 2 * num_cross_layers
        assert native_layer_calls == prev_native

        # Frame 2 query: clone keys/values before query, verify equal after query, cross hook unchanged
        cloned_kv2 = [
            (tuple(k.clone() for k in f.keys), tuple(v.clone() for v in f.values))
            for f in session.frames
        ]
        cnt_before_q2 = cross_kv_calls
        d2, s2 = session.query()
        assert cross_kv_calls == cnt_before_q2
        for (orig_k, orig_v), f in zip(cloned_kv2, session.frames):
            for ok, fk in zip(orig_k, f.keys):
                assert torch.equal(ok, fk)
            for ov, fv in zip(orig_v, f.values):
                assert torch.equal(ov, fv)

        # Query calls ViT 0 additional times
        pre_extract = policy.embedder._extract_calls
        session.query()
        assert policy.embedder._extract_calls == pre_extract, "query must not call ViT extract_feature"

        # Compare with fresh model.forward
        d_fresh, s_fresh = moss.forward([["img0.png"], ["img1.png"]], [0, 1], "clean table")
        assert torch.allclose(d2, d_fresh, atol=1e-5), "Cached session must match fresh forward"
        assert torch.allclose(s2, s_fresh, atol=1e-5), "Cached session must match fresh forward"

        # Frame 3 causes eviction (max_frames=2)
        prev_cross = cross_kv_calls
        prev_native = native_layer_calls
        session.append(["img2.png"], frame_id=2)
        assert len(session.frames) == 2
        assert session.frames[0].frame_id == 1
        assert session.frames[1].frame_id == 2
        assert cross_kv_calls - prev_cross == 2 * num_cross_layers
        assert native_layer_calls == prev_native

        # Frame 3 query: clone keys/values before query, verify equal after query, cross hook unchanged
        cloned_kv3 = [
            (tuple(k.clone() for k in f.keys), tuple(v.clone() for v in f.values))
            for f in session.frames
        ]
        cnt_before_q3 = cross_kv_calls
        d3, s3 = session.query()
        assert cross_kv_calls == cnt_before_q3
        for (orig_k, orig_v), f in zip(cloned_kv3, session.frames):
            for ok, fk in zip(orig_k, f.keys):
                assert torch.equal(ok, fk)
            for ov, fv in zip(orig_v, f.values):
                assert torch.equal(ov, fv)

        # Eviction parity check: [1, 2] fresh forward
        d_fresh_12, s_fresh_12 = moss.forward([["img1.png"], ["img2.png"]], [1, 2], "clean table")
        assert torch.allclose(d3, d_fresh_12, atol=1e-5), "Evicted cached session must match fresh forward for [1, 2]"
        assert torch.allclose(s3, s_fresh_12, atol=1e-5), "Evicted cached session must match fresh forward for [1, 2]"

        # Change history input to ensure output differs
        d_diff_hist, s_diff_hist = moss.forward([["img0.png"], ["img2.png"]], [0, 2], "clean table")
        assert not torch.allclose(d3, d_diff_hist, atol=1e-3), "Different history input must produce different deep output"
        assert not torch.allclose(s3, s_diff_hist, atol=1e-3), "Different history input must produce different shallow output"


def test_session_state_guards(tiny_setup):
    """Guards against invalid operations on VisionSession."""
    _, moss, _ = tiny_setup
    moss.eval()
    session = VisionSession(moss)

    # Append before reset
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="must be set via reset"):
            session.append(["img.png"], frame_id=0)

    session.reset("ep1", "task prompt")

    # Append with grad enabled
    with pytest.raises(RuntimeError, match="requires torch.no_grad"):
        session.append(["img.png"], frame_id=0)

    # Non-monotonic frame_id
    with torch.no_grad():
        session.append(["img.png"], frame_id=5)
        with pytest.raises(ValueError, match="strictly greater"):
            session.append(["img.png"], frame_id=3)

    # Invalidation on training
    moss.train(True)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="eval mode"):
            session.query()


def test_joint_stage_unfreezes_action_and_base_but_not_vision(tiny_setup):
    policy, moss, _ = tiny_setup
    moss.set_training_stage("bridge")
    assert all(not p.requires_grad for p in policy.action_head.parameters())
    moss.set_training_stage("joint")
    assert any(p.requires_grad for p in policy.action_head.parameters())
    assert any(p.requires_grad for p in policy.core.parameters())
    assert moss.training_stage == "joint"


def test_foreign_owner_and_revision_rejection(tiny_setup):
    """read_memory rejects FrameKV from foreign owner or with stale revision."""
    policy, moss1, config = tiny_setup
    moss2 = MossInternVL(policy, config=config)

    feats = torch.randn(1, 16, 128)
    frame_foreign = moss2.project_frame(feats, frame_id=0)

    with pytest.raises(ValueError, match="foreign owner"):
        moss1.read_memory([frame_foreign], prompt="task")

    frame_valid = moss1.project_frame(feats, frame_id=0)
    moss1.set_training_stage("expert")  # increments revision
    with pytest.raises(ValueError, match="stale revision"):
        moss1.read_memory([frame_valid], prompt="task")


def test_mixed_precision_bf16_native_fp32_cross():
    """Verify bfloat16 native weights + float32 cross blocks execution."""
    torch.manual_seed(42)
    policy = TinyFabriVLAPolicy(hidden_size=128, num_layers=4)
    policy.core.to(torch.bfloat16)

    config = MossConfig(
        cross_layers=(2, 4),
        num_readout_tokens=4,
        max_frames=2,
        shallow_layer=2,
    )
    moss = MossInternVL(policy, config=config)
    moss.train(True)
    moss.set_training_stage("bridge")

    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.1)
        block.mlp_gate.data.fill_(0.1)

    feats = torch.randn(1, 16, 128)
    frame = moss.project_frame(feats, frame_id=0)

    deep, shallow = moss.read_memory([frame], prompt="test mixed precision")
    assert deep.dtype == torch.float32
    assert shallow.dtype == torch.float32
    assert not torch.isnan(deep).any()
    assert not torch.isnan(shallow).any()

    # Action head is frozen in bridge stage
    for p in policy.action_head.parameters():
        assert not p.requires_grad

    loss = policy.action_head(deep, shallow).square().mean()
    loss.backward()

    # Check all native policy parameters have grad None
    for p in policy.parameters():
        assert p.grad is None

    # At least first block k/v is non-zero and finite
    first_block = moss.cross_blocks[str(config.cross_layers[0])]
    assert first_block.k_proj.weight.grad is not None
    assert torch.isfinite(first_block.k_proj.weight.grad).all()
    assert (first_block.k_proj.weight.grad != 0).any()
    assert first_block.v_proj.weight.grad is not None
    assert torch.isfinite(first_block.v_proj.weight.grad).all()
    assert (first_block.v_proj.weight.grad != 0).any()

    # Readout embeddings grad is non-zero and finite
    assert moss.readout_embeddings.grad is not None
    assert torch.isfinite(moss.readout_embeddings.grad).all()
    assert (moss.readout_embeddings.grad != 0).any()


def test_transformers_actual_qwen3_zero_gates_parity():
    """Test actual transformers Qwen3Model if installed."""
    try:
        from transformers import Qwen3Config, Qwen3Model
    except ImportError:
        pytest.skip("transformers Qwen3Model not available")

    # Real Qwen3 configuration matching FabriVLA / MOSS (head_dim=128, etc)
    hf_config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
    )
    qwen3 = Qwen3Model(hf_config)

    class MockEmbedder(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.model = core_model
            self.tokenizer = TinyTokenizer()
            self.img_context_token_id = 42
            self.model.extract_feature = lambda pv: torch.randn(1, 16, 256)

        def _preprocess_images(self, imgs):
            return torch.randn(1, 64), [1]

    class MockPolicy(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.embedder = MockEmbedder(core_model)
            self.action_head = nn.Linear(256, 4)

    policy = MockPolicy(qwen3)
    moss_config = MossConfig(
        cross_layers=(2, 4),
        num_readout_tokens=4,
        max_frames=2,
        shallow_layer=2,
    )
    moss = MossInternVL(policy, config=moss_config)
    moss.eval()

    prompt = "real qwen3 prompt"
    feats = torch.randn(1, 16, 256)
    frame = moss.project_frame(feats, frame_id=0)

    with torch.no_grad():
        deep_moss, shallow_moss = moss.read_memory([frame], prompt)

    # Direct Qwen3Model forward with identical tokens
    toks = policy.embedder.tokenizer(prompt.strip(), return_tensors="pt")
    text_emb = qwen3.embed_tokens(toks.input_ids)
    readout_emb = moss.readout_embeddings.unsqueeze(0)
    full_emb = torch.cat([text_emb, readout_emb], dim=1)

    with torch.no_grad():
        hf_out = qwen3(inputs_embeds=full_emb, output_hidden_states=True)
        deep_hf = hf_out.last_hidden_state[:, -moss_config.num_readout_tokens :, :].float()
        shallow_hf = hf_out.hidden_states[moss_config.shallow_layer][:, -moss_config.num_readout_tokens :, :].float()

    assert torch.allclose(deep_moss, deep_hf, atol=1e-5), "Actual Qwen3 zero gate deep output must match native Qwen3"
    assert torch.allclose(shallow_moss, shallow_hf, atol=1e-5), "Actual Qwen3 zero gate shallow output must match native Qwen3"
