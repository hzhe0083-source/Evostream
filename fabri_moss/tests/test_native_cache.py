"""Unit and integration tests for fabri_moss.native_cache: native sliding-window KV cache and adapter."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Any, List, Optional, Sequence, Tuple, Union

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from fabri_moss.native_cache import (
    NativeCacheAdapter,
    NativeCacheConfig,
    NativeEmbeddingBlock,
    NativeKVState,
    extract_layer_kv,
    format_frame_timestamp,
    populate_cache_from_layer_kv,
)


class TinyTokenizer:
    """Mock tokenizer returning token IDs with length <= 16."""

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        class TokenizerOutput:
            pass

        out = TokenizerOutput()
        out.input_ids = torch.ones((1, 8), dtype=torch.long)
        return out


class TinyEmbedder(nn.Module):
    """Embedder matching FabriVLA contract for native cache adapter tests."""

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
        # Deterministic hashing from image representation
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
        # 4 visual tokens of dim 64
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
        # Simple prompt-dependent token generation so text length/content can reflect timestamp prefix
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


def make_tiny_native_adapter(
    max_frames: int = 4,
    shallow_layer: int = 1,
    max_text_length: int = 16,
    use_timestamps: bool = False,
) -> NativeCacheAdapter:
    """Exported helper for native_async and native_cache tests to reuse.

    Instantiates a genuine Qwen3ForCausalLM model with eager attention on CPU,
    wraps it in TinyPolicy, and configures NativeCacheAdapter.
    """
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
    adapter = NativeCacheAdapter(policy, config=cache_config)
    adapter.eval()
    return adapter


def test_1_single_adapter_read_vs_native_core_full_forward():
    """Test 1: single adapter read_blocks vs native core full forward (use_cache=False).

    Validates deep features and shallow features (layer 1) match exactly within 1e-5.
    """
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "execute task"
    block = adapter.encode_frame(images=["image_0"], frame_id=0, prompt=prompt)

    deep, shallow, state = adapter.read_blocks([block], prompt=prompt)

    # Native core forward with use_cache=False
    core = adapter.native_core
    out = core(
        inputs_embeds=block.inputs_embeds,
        attention_mask=block.attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )

    expected_deep = out.last_hidden_state
    expected_shallow = out.hidden_states[1]

    assert deep.dtype == torch.float32
    assert shallow.dtype == torch.float32
    assert torch.allclose(deep, expected_deep, atol=1e-5, rtol=0)
    assert torch.allclose(shallow, expected_shallow, atol=1e-5, rtol=0)
    assert state.frame_count == 1
    assert state.last_frame_id == 0
    assert state.rebuild_count == 0


def test_2_incremental_append_vs_full_cat_forward():
    """Test 2: block0 append block1 vs full concatenation (16*2 tokens) with mask.

    Physical positions 0..31; deep and shallow features on the last 16 tokens match within 1e-5.
    """
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "track object"

    block0 = adapter.encode_frame(images=["frame_0"], frame_id=0, prompt=prompt)
    deep0, shallow0, state0 = adapter.read_blocks([block0], prompt=prompt)

    block1 = adapter.encode_frame(images=["frame_1"], frame_id=1, prompt=prompt)
    deep1, shallow1, state1 = adapter.read_blocks([block1], prompt=prompt, previous=state0)

    # Full concatenation forward through native core
    core = adapter.native_core
    full_embeds = torch.cat([block0.inputs_embeds, block1.inputs_embeds], dim=1)
    full_mask = torch.cat([block0.attention_mask, block1.attention_mask], dim=1)

    out_full = core(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        use_cache=False,
        output_hidden_states=True,
    )

    expected_deep = out_full.last_hidden_state[:, 16:, :]
    expected_shallow = out_full.hidden_states[1][:, 16:, :]

    assert torch.allclose(deep1, expected_deep, atol=1e-5, rtol=0)
    assert torch.allclose(shallow1, expected_shallow, atol=1e-5, rtol=0)
    assert state1.frame_count == 2
    assert state1.last_frame_id == 1
    assert state1.rebuild_count == 0
    assert len(state1.blocks) == 2


def test_3_sliding_window_overflow_rebuild_and_vit_cache():
    """Test 3: max_frames=2; append 3rd block triggers rebuild count 1.

    Remaining blocks are [1, 2], matching fresh read of blocks [1, 2],
    and ViT extract_feature is not re-invoked during read_blocks.
    """
    adapter = make_tiny_native_adapter(max_frames=2, shallow_layer=1)
    prompt = "slide window"

    b0 = adapter.encode_frame(["img_0"], frame_id=0, prompt=prompt)
    _, _, state0 = adapter.read_blocks([b0], prompt=prompt)

    b1 = adapter.encode_frame(["img_1"], frame_id=1, prompt=prompt)
    _, _, state1 = adapter.read_blocks([b1], prompt=prompt, previous=state0)
    assert state1.rebuild_count == 0
    assert [b.frame_id for b in state1.blocks] == [0, 1]

    # Pre-record ViT calls before appending 3rd block
    vit_calls_before = adapter.policy.embedder.extract_feature_call_count
    b2 = adapter.encode_frame(["img_2"], frame_id=2, prompt=prompt)
    vit_calls_after_encode = adapter.policy.embedder.extract_feature_call_count
    assert vit_calls_after_encode == vit_calls_before + 1

    # Appending 3rd block when max_frames=2 triggers overflow rebuild
    deep2, shallow2, state2 = adapter.read_blocks([b2], prompt=prompt, previous=state1)
    vit_calls_after_read = adapter.policy.embedder.extract_feature_call_count
    # ViT calls must not increase during read_blocks
    assert vit_calls_after_read == vit_calls_after_encode

    assert state2.rebuild_count == 1
    assert [b.frame_id for b in state2.blocks] == [1, 2]
    assert state2.last_frame_id == 2
    assert state2.frame_count == 3

    # Compare with fresh prefill on blocks [b1, b2] using the same adapter
    deep_fresh, shallow_fresh, state_fresh = adapter.read_blocks([b1, b2], prompt=prompt)

    assert torch.allclose(deep2, deep_fresh, atol=1e-5, rtol=0)
    assert torch.allclose(shallow2, shallow_fresh, atol=1e-5, rtol=0)
    for (k2, v2), (kf, vf) in zip(state2.layer_kv, state_fresh.layer_kv):
        assert torch.allclose(k2, kf, atol=1e-5, rtol=0)
        assert torch.allclose(v2, vf, atol=1e-5, rtol=0)


def test_4_immutability_old_state_and_cache_wrapper_isolation():
    """Test 4: old state KV is not modified; new cache is not the same wrapper."""
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "immutable check"

    b0 = adapter.encode_frame(["img_0"], frame_id=0, prompt=prompt)
    _, _, state0 = adapter.read_blocks([b0], prompt=prompt)

    # Snapshot state0 layer_kv clones
    old_kv_snapshots = [(k.clone(), v.clone()) for k, v in state0.layer_kv]

    b1 = adapter.encode_frame(["img_1"], frame_id=1, prompt=prompt)
    _, _, state1 = adapter.read_blocks([b1], prompt=prompt, previous=state0)

    # Check state0 tensors remain intact
    for (orig_k, orig_v), (curr_k, curr_v) in zip(old_kv_snapshots, state0.layer_kv):
        assert torch.equal(orig_k, curr_k)
        assert torch.equal(orig_v, curr_v)
        assert curr_k.shape[2] == 16  # remains length 16

    # Verify state1 has updated sequence length
    assert state1.layer_kv[0][0].shape[2] == 32

    # Verify new cache wrapper created during read_blocks is fresh and isolated
    cache_a = populate_cache_from_layer_kv(state0.layer_kv)
    cache_b = populate_cache_from_layer_kv(state0.layer_kv)
    assert cache_a is not cache_b

    # Verify detached state produces isolated object with detached tensors
    detached_state = state0.detached()
    assert detached_state is not state0
    assert not detached_state.layer_kv[0][0].requires_grad
    assert torch.equal(detached_state.layer_kv[0][0], state0.layer_kv[0][0])


def test_5_zero_new_parameters_and_frozen_policy():
    """Test 5: adapter adds 0 new parameters; parameters identity and weights unchanged."""
    policy_cfg = Qwen3Config(
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
    lm = Qwen3ForCausalLM(policy_cfg)
    policy = TinyPolicy(lm)

    # Record parameter names, shapes, and weights before wrapping
    orig_params = list(policy.parameters())
    orig_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}

    adapter = NativeCacheAdapter(policy, NativeCacheConfig(max_frames=4, shallow_layer=1))

    adapter_params = list(adapter.parameters())
    assert len(adapter_params) == len(orig_params)

    # All adapter parameters are identical references to policy parameters
    for p_orig, p_adapt in zip(orig_params, adapter_params):
        assert p_orig is p_adapt
        assert not p_adapt.requires_grad

    # Verify weights unchanged
    curr_state_dict = policy.state_dict()
    for k, v in orig_state_dict.items():
        assert torch.equal(v, curr_state_dict[k])


def test_6_encode_frame_strict_validation_before_feature_extraction():
    """Test 6: frame bool/negative/prompt empty/multi-images/multi-tiles reject before extract."""
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "valid prompt"

    initial_vit_count = adapter.policy.embedder.extract_feature_call_count

    # 1. frame_id is bool
    with pytest.raises(ValueError, match="frame_id must be non-negative int"):
        adapter.encode_frame(["img_0"], frame_id=True, prompt=prompt)  # type: ignore

    # 2. frame_id is negative
    with pytest.raises(ValueError, match="frame_id must be non-negative int"):
        adapter.encode_frame(["img_0"], frame_id=-1, prompt=prompt)

    # 3. prompt empty or whitespace only
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        adapter.encode_frame(["img_0"], frame_id=0, prompt="")
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        adapter.encode_frame(["img_0"], frame_id=0, prompt="   ")

    # 4. multi-images
    with pytest.raises(ValueError, match="len\\(images\\) == 1"):
        adapter.encode_frame(["img_0", "img_1"], frame_id=0, prompt=prompt)
    with pytest.raises(ValueError, match="len\\(images\\) == 1"):
        adapter.encode_frame([], frame_id=0, prompt=prompt)

    # ViT calls must not have been triggered for these argument validation errors
    assert adapter.policy.embedder.extract_feature_call_count == initial_vit_count

    # 5. multi-tiles rejection before extract
    class MultiTileEmbedder(TinyEmbedder):
        def _preprocess_images(self, images):
            return torch.zeros((1, 8, 64)), [2]  # tile count != [1]

    adapter.policy.embedder = MultiTileEmbedder(adapter.native_core.embed_tokens)
    with pytest.raises(ValueError, match="single tile \\[1\\]"):
        adapter.encode_frame(["img_0"], frame_id=0, prompt=prompt)

    assert adapter.policy.embedder.extract_feature_call_count == 0


def test_7_metadata_validation_owner_revision_prompt_order():
    """Test 7: validate owner, revision, prompt mismatch, and stale block frame ordering."""
    adapter1 = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    adapter2 = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "metadata check"

    b0 = adapter1.encode_frame(["img_0"], frame_id=0, prompt=prompt)
    _, _, state0 = adapter1.read_blocks([b0], prompt=prompt)

    # 1. Owner mismatch in block
    b_foreign = adapter2.encode_frame(["img_1"], frame_id=1, prompt=prompt)
    with pytest.raises(ValueError, match="does not match adapter"):
        adapter1.read_blocks([b_foreign], prompt=prompt, previous=state0)

    # 2. Owner mismatch in previous state
    with pytest.raises(ValueError, match="previous owner"):
        adapter2.read_blocks([b_foreign], prompt=prompt, previous=state0)

    # 3. Revision mismatch in block
    b_stale_rev = NativeEmbeddingBlock(
        frame_id=1,
        inputs_embeds=b0.inputs_embeds,
        attention_mask=b0.attention_mask,
        owner=adapter1,
        revision=999,
        prompt=prompt,
    )
    with pytest.raises(ValueError, match="revision .* does not match"):
        adapter1.read_blocks([b_stale_rev], prompt=prompt)

    # 4. Prompt mismatch
    adapter1 = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    b0_valid = adapter1.encode_frame(["img_0"], frame_id=0, prompt="prompt A")
    with pytest.raises(ValueError, match="prompt .* does not match"):
        adapter1.read_blocks([b0_valid], prompt="prompt B")

    # 5. Non-increasing frame_id order in new_blocks
    b1_id1 = adapter1.encode_frame(["img_1"], frame_id=1, prompt="prompt A")
    b2_id1 = adapter1.encode_frame(["img_2"], frame_id=1, prompt="prompt A")
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter1.read_blocks([b1_id1, b2_id1], prompt="prompt A")

    # 6. First new block frame_id <= previous.last_frame_id
    _, _, st1 = adapter1.read_blocks([b0_valid], prompt="prompt A")
    b_retro = adapter1.encode_frame(["img_retro"], frame_id=0, prompt="prompt A")
    with pytest.raises(ValueError, match="must be > previous last_frame_id"):
        adapter1.read_blocks([b_retro], prompt="prompt A", previous=st1)


def test_8_wrong_previous_layer_kv_length_and_mask_shape():
    """Test 8: wrong layer count, wrong K/V sequence length, or mismatched attention mask."""
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "corrupt state test"

    b0 = adapter.encode_frame(["img_0"], frame_id=0, prompt=prompt)
    _, _, state0 = adapter.read_blocks([b0], prompt=prompt)

    # 1. Wrong previous layer_kv count (e.g. 2 instead of 3)
    corrupted_kv_count = NativeKVState(
        layer_kv=state0.layer_kv[:2],
        blocks=state0.blocks,
        attention_mask=state0.attention_mask,
        last_frame_id=state0.last_frame_id,
        frame_count=state0.frame_count,
        prompt=state0.prompt,
        owner=state0.owner,
        revision=state0.revision,
        rebuild_count=state0.rebuild_count,
    )
    b1 = adapter.encode_frame(["img_1"], frame_id=1, prompt=prompt)
    with pytest.raises(ValueError, match="layer_kv count .* does not match"):
        adapter.read_blocks([b1], prompt=prompt, previous=corrupted_kv_count)

    # 2. Corrupted K/V tensor shape along sequence dimension
    bad_k = torch.zeros((1, 2, 24, 16))  # seq_len 24 != blocks seq_len 16
    bad_v = torch.zeros((1, 2, 24, 16))
    corrupted_kv_shape = NativeKVState(
        layer_kv=((bad_k, bad_v), state0.layer_kv[1], state0.layer_kv[2]),
        blocks=state0.blocks,
        attention_mask=state0.attention_mask,
        last_frame_id=state0.last_frame_id,
        frame_count=state0.frame_count,
        prompt=state0.prompt,
        owner=state0.owner,
        revision=state0.revision,
        rebuild_count=state0.rebuild_count,
    )
    with pytest.raises(ValueError, match="seq_len .* does not match"):
        adapter.read_blocks([b1], prompt=prompt, previous=corrupted_kv_shape)

    # 3. Attention mask length does not match total blocks sequence length
    corrupted_mask = NativeKVState(
        layer_kv=state0.layer_kv,
        blocks=state0.blocks,
        attention_mask=torch.ones((1, 32)),  # 32 != 16
        last_frame_id=state0.last_frame_id,
        frame_count=state0.frame_count,
        prompt=state0.prompt,
        owner=state0.owner,
        revision=state0.revision,
        rebuild_count=state0.rebuild_count,
    )
    with pytest.raises(ValueError, match="attention_mask seq_len .* does not match"):
        adapter.read_blocks([b1], prompt=prompt, previous=corrupted_mask)


def test_9_nbytes_and_kv_nbytes_exact_calculation():
    """Test 9: exact analytical verification of kv_nbytes and nbytes."""
    seq_len = 16
    hidden_dim = 64
    num_layers = 3
    num_kv_heads = 2
    head_dim = 16

    inputs_embeds = torch.zeros((1, seq_len, hidden_dim), dtype=torch.float32)
    mask = torch.ones((1, seq_len), dtype=torch.bool)
    blk = NativeEmbeddingBlock(
        frame_id=0,
        inputs_embeds=inputs_embeds,
        attention_mask=mask,
        owner="dummy",
        revision=0,
        prompt="calc nbytes",
    )

    k = torch.zeros((1, num_kv_heads, seq_len, head_dim), dtype=torch.float32)
    v = torch.zeros((1, num_kv_heads, seq_len, head_dim), dtype=torch.float32)
    layer_kv = tuple((k, v) for _ in range(num_layers))

    state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk,),
        attention_mask=mask,
        last_frame_id=0,
        frame_count=1,
        prompt="calc nbytes",
        owner="dummy",
        revision=0,
        rebuild_count=0,
    )

    # Analytical calculation:
    # kv_nbytes = 3 layers * 2 tensors (k and v) * (1 * 2 * 16 * 16 elements) * 4 bytes
    # = 6 * 512 * 4 = 12,288 bytes
    expected_kv_nbytes = 3 * 2 * (1 * 2 * 16 * 16) * 4
    assert state.kv_nbytes == expected_kv_nbytes

    # nbytes = kv_nbytes + blk.inputs_embeds (1 * 16 * 64 * 4 = 4096 bytes)
    #          + blk.attention_mask (1 * 16 * 1 byte = 16 bytes)
    #          + state.attention_mask (1 * 16 * 1 byte = 16 bytes)
    # = 12,288 + 4,096 + 16 + 16 = 16,416 bytes
    expected_total_nbytes = expected_kv_nbytes + (1 * 16 * 64 * 4) + (1 * 16 * 1) + (1 * 16 * 1)
    assert state.nbytes == expected_total_nbytes


def test_10_train_mode_rejection_and_eval_revision_stability():
    """Test 10: train mode operations rejected; switching to train bumps revision; eval mode does not."""
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "train mode check"

    assert adapter.revision == 0
    assert not adapter.training

    # Transition to train mode bumps revision
    adapter.train(True)
    assert adapter.training
    assert adapter.revision == 1

    # Operations in train mode are rejected
    with pytest.raises(RuntimeError, match="encode_frame is only allowed in eval mode"):
        adapter.encode_frame(["img_0"], frame_id=0, prompt=prompt)

    # Dummy block for testing read_blocks under train mode
    dummy_blk = NativeEmbeddingBlock(
        frame_id=0,
        inputs_embeds=torch.zeros((1, 16, 64)),
        attention_mask=torch.ones((1, 16)),
        owner=adapter,
        revision=adapter.revision,
        prompt=prompt,
    )
    with pytest.raises(RuntimeError, match="read_blocks is only allowed in eval mode"):
        adapter.read_blocks([dummy_blk], prompt=prompt)

    # Calling eval does not bump revision
    adapter.eval()
    assert not adapter.training
    assert adapter.revision == 1

    adapter.eval()
    assert adapter.revision == 1


def test_11_shallow_layer_equals_num_layers_norm_consistency():
    """Test 11: shallow_layer == num_layers produces output identical to deep features."""
    # num_hidden_layers = 3, set shallow_layer = 3
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=3)
    prompt = "norm consistency check"

    block = adapter.encode_frame(["img_0"], frame_id=0, prompt=prompt)
    deep, shallow, _ = adapter.read_blocks([block], prompt=prompt)

    assert deep.shape == shallow.shape
    assert torch.allclose(deep, shallow, atol=1e-5, rtol=0)


def test_12_invalid_config_shallow_layer_greater_than_num_layers():
    """Test 12: shallow_layer > num_layers is rejected during adapter initialization."""
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
    lm = Qwen3ForCausalLM(config)
    policy = TinyPolicy(lm)

    # shallow_layer = 4 while num_hidden_layers = 3
    invalid_cfg = NativeCacheConfig(max_frames=4, shallow_layer=4)
    with pytest.raises(ValueError, match="shallow_layer must be in \\[1, 3\\]"):
        NativeCacheAdapter(policy, config=invalid_cfg)

    # shallow_layer = 0 is also rejected by config validation
    with pytest.raises(ValueError, match="shallow_layer must be > 0"):
        NativeCacheConfig(max_frames=4, shallow_layer=0)
