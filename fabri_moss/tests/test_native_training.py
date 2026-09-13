"""Unit and integration tests for fabri_moss.native_training."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from fabri_moss.native_cache import NativeCacheAdapter, NativeCacheConfig, format_frame_timestamp
from fabri_moss.native_training import NativeSequencePolicy


class LearnedFakeViT(nn.Module):
    """Differentiable learned fake ViT encoder with parameters."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Linear(in_dim, hidden_dim)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, in_dim] -> [B, 4, hidden_dim]
        return self.conv(x)


class LearnedProjector(nn.Module):
    """Differentiable projector from ViT hidden dim to LLM hidden dim."""

    def __init__(self, in_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TinyModelWithVision(nn.Module):
    """Wrapper holding vision_model, extract_feature, and language_model."""

    def __init__(self, language_model: nn.Module, vit_dim: int = 32, llm_dim: int = 64):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.encoder = LearnedFakeViT(in_dim=16, hidden_dim=vit_dim)
        self.projector = LearnedProjector(in_dim=vit_dim, out_dim=llm_dim)
        self.language_model = language_model

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 4, 16] -> [B, 4, 32] -> [B, 4, 64]
        feat = self.vision_model.encoder(pixel_values)
        return self.projector(feat)


class TinyTokenizer:
    """Mock tokenizer returning token IDs."""

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        class TokenizerOutput:
            pass

        out = TokenizerOutput()
        # Return 8 tokens so length <= max_text_length (16)
        out.input_ids = torch.ones((1, 8), dtype=torch.long)
        return out


class FullDifferentiableEmbedder(nn.Module):
    """Embedder matching FabriVLA contract with learned components."""

    def __init__(self, language_model: nn.Module, max_text_length: int = 16):
        super().__init__()
        self.device = torch.device("cpu")
        self.max_text_length = max_text_length
        self.model = TinyModelWithVision(language_model, vit_dim=32, llm_dim=64)
        self.tokenizer = TinyTokenizer()

    def _preprocess_images_on_cpu(self, images: Sequence[Any]) -> Tuple[torch.Tensor, List[int]]:
        N = len(images)
        tensors = []
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                tensors.append(img.view(1, 4, 16).float())
            else:
                # Deterministic float tensor from string/int
                s = str(img).encode("utf-8")
                h_val = int(hashlib.md5(s).hexdigest()[:8], 16)
                gen = torch.Generator().manual_seed(h_val % (2**31 - 1))
                tensors.append(torch.randn(1, 4, 16, generator=gen))
        pixel_values = torch.cat(tensors, dim=0)  # [N, 4, 16]
        return pixel_values, [1] * N

    def _build_multimodal_prompt(self, num_tiles_list: List[int], prompt: str) -> str:
        return f"<prompt>{prompt}</prompt>"

    def _prepare_batch_and_fuse_embeddings(
        self,
        prompts: List[str],
        vit_embeds_batch: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        batch_num_tiles_list: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(prompts)
        core = self.model.language_model.model
        fused_list = []
        mask_list = []
        for i in range(N):
            vit = vit_embeds_batch[i]  # [1, 4, 64]
            # Prompt token embeddings from core.embed_tokens (12 tokens)
            p = prompts[i]
            h = int(hashlib.md5(p.encode("utf-8")).hexdigest()[:6], 16)
            tok_val = (h % 50) + 1
            tokens = torch.full((1, 12), tok_val, dtype=torch.long, device=vit.device)
            tok_embeds = core.embed_tokens(tokens)  # [1, 12, 64]
            fused = torch.cat([vit, tok_embeds], dim=1)  # [1, 16, 64]
            mask = torch.ones(1, 16, dtype=torch.bool, device=vit.device)
            fused_list.append(fused)
            mask_list.append(mask)

        batch_embeds = torch.cat(fused_list, dim=0)  # [N, 16, 64]
        batch_masks = torch.cat(mask_list, dim=0)    # [N, 16]
        return batch_embeds, batch_masks


class ActionHeadConfig:
    def __init__(self, shallow_fusion: str = "concat_proj"):
        self.shallow_fusion = shallow_fusion


class DualTokenActionHead(nn.Module):
    """Action head using BOTH deep and shallow features to compute loss and action prediction."""

    def __init__(self, hidden_dim: int = 64, horizon: int = 2, action_dim: int = 4, shallow_fusion: str = "concat_proj"):
        super().__init__()
        self.config = ActionHeadConfig(shallow_fusion=shallow_fusion)
        self.horizon = horizon
        self.action_dim = action_dim
        in_dim = hidden_dim * 2 if shallow_fusion != "none" else hidden_dim
        self.proj = nn.Linear(in_dim, horizon * action_dim)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        actions_gt: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        state_mask: Optional[torch.Tensor] = None,
        shallow_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        M = fused_tokens.shape[0]
        deep_rep = fused_tokens.mean(dim=1)  # [M, 64]
        if self.config.shallow_fusion != "none" and shallow_tokens is not None:
            shallow_rep = shallow_tokens.mean(dim=1)  # [M, 64]
            comb = torch.cat([deep_rep, shallow_rep], dim=-1)  # [M, 128]
        else:
            comb = deep_rep

        pred = self.proj(comb).view(M, self.horizon, self.action_dim)
        res: Dict[str, Any] = {"action_pred": pred}
        if actions_gt is not None:
            diff = (pred - actions_gt) ** 2
            if action_mask is not None:
                diff = diff * action_mask.float()
            loss = diff.mean()
            res["loss"] = loss
        return res

    def sample(self, deep: torch.Tensor, *args: Any, shallow_tokens: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        out = self.forward(fused_tokens=deep, shallow_tokens=shallow_tokens)
        return out["action_pred"]


class TinyDifferentiablePolicy(nn.Module):
    def __init__(self, language_model: nn.Module, max_text_length: int = 16, shallow_fusion: str = "concat_proj", horizon: int = 2, action_dim: int = 4):
        super().__init__()
        self.embedder = FullDifferentiableEmbedder(language_model, max_text_length=max_text_length)
        self.action_head = DualTokenActionHead(hidden_dim=64, horizon=horizon, action_dim=action_dim, shallow_fusion=shallow_fusion)


def make_tiny_training_policy(
    shallow_layer: int = 1,
    max_text_length: int = 16,
    use_timestamps: bool = True,
    gradient_checkpointing: bool = False,
    shallow_fusion: str = "concat_proj",
    horizon: int = 2,
    action_dim: int = 4,
) -> NativeSequencePolicy:
    """Create a fully differentiable policy wrapping a real tiny Qwen3 model."""
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
    policy = TinyDifferentiablePolicy(
        language_model=language_model,
        max_text_length=max_text_length,
        shallow_fusion=shallow_fusion,
        horizon=horizon,
        action_dim=action_dim,
    )
    seq_policy = NativeSequencePolicy(
        policy=policy,
        shallow_layer=shallow_layer,
        use_timestamps=use_timestamps,
        gradient_checkpointing=gradient_checkpointing,
    )
    return seq_policy


def make_sample(
    N: int = 4,
    target_indices: Sequence[int] = (1, 3),
    horizon: int = 2,
    action_dim: int = 4,
) -> Dict[str, Any]:
    images_window = [torch.randn(4, 16) for _ in range(N)]
    frame_ids = list(range(N))
    observation_times = [i * 0.1 for i in range(N)]
    prompt = "pick up the red cube"
    M = len(target_indices)
    actions = torch.randn(M, horizon, action_dim)
    action_mask = torch.ones(M, horizon, action_dim, dtype=torch.bool)
    return {
        "images_window": images_window,
        "frame_ids": frame_ids,
        "observation_times": observation_times,
        "prompt": prompt,
        "target_indices": list(target_indices),
        "actions": actions,
        "action_mask": action_mask,
    }


# ============================================================================
# Test Cases
# ============================================================================

def test_1_shape_n4_m2_features_and_forward():
    """Test shape N=4, M=2: deep and shallow features are [M, 16, 64], loss dict contains loss_sum and target_count."""
    seq_policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    sample = make_sample(N=4, target_indices=(1, 3), horizon=2, action_dim=4)

    deep, shallow = seq_policy.features(sample)
    assert deep.shape == (2, 16, 64)
    assert shallow.shape == (2, 16, 64)
    assert deep.dtype == torch.float32
    assert shallow.dtype == torch.float32

    out = seq_policy(sample)
    assert "loss" in out
    assert "loss_sum" in out
    assert out["target_count"] == 2
    assert torch.isclose(out["loss_sum"], out["loss"] * 2.0)
    assert out["action_pred"].shape == (2, 2, 4)


def test_2_all_params_fp32_trainable_and_no_param_creation():
    """Test all policy parameters are FP32, requires_grad=True, and wrapper introduces no new parameters."""
    seq_policy = make_tiny_training_policy(shallow_layer=1)
    all_wrapper_params = list(seq_policy.parameters())
    all_inner_params = list(seq_policy.policy.parameters())

    assert len(all_wrapper_params) == len(all_inner_params)
    for p in all_wrapper_params:
        assert p.dtype == torch.float32
        assert p.requires_grad is True

    # Check forward pass creates no new parameters
    count_before = len(list(seq_policy.parameters()))
    sample = make_sample(N=4, target_indices=(1, 3))
    _ = seq_policy(sample)
    count_after = len(list(seq_policy.parameters()))
    assert count_before == count_after


def test_3_gradient_flow_from_last_target_loss_to_first_frame():
    """Test backward from last target frame loss propagates non-zero gradients to:
    first frame visual features, ViT encoder, projector, token embeddings, LLM layers, and action head.
    """
    seq_policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    seq_policy.train()

    # Create sample where image 0 requires grad to check gradient reaching first frame pixel values
    N = 4
    img0 = torch.randn(4, 16, requires_grad=True)
    images = [img0] + [torch.randn(4, 16) for _ in range(N - 1)]
    sample = make_sample(N=4, target_indices=[3], horizon=2, action_dim=4)  # M=1, only last target
    sample["images_window"] = images

    out = seq_policy(sample)
    loss = out["loss"]
    loss.backward()

    # 1. Gradient reaches first frame visual features
    assert img0.grad is not None
    assert torch.any(img0.grad != 0.0)

    # 2. Gradient reaches vision encoder
    vit = seq_policy.policy.embedder.model.vision_model.encoder
    for name, p in vit.named_parameters():
        assert p.grad is not None and torch.any(p.grad != 0.0), f"ViT param {name} has no grad"

    # 3. Gradient reaches projector
    proj = seq_policy.policy.embedder.model.projector
    for name, p in proj.named_parameters():
        assert p.grad is not None and torch.any(p.grad != 0.0), f"Projector param {name} has no grad"

    # 4. Gradient reaches LLM token embeddings
    embed_tokens = seq_policy.native_core.embed_tokens
    assert embed_tokens.weight.grad is not None
    assert torch.any(embed_tokens.weight.grad != 0.0)

    # 5. Gradient reaches LLM layers
    for i, layer in enumerate(seq_policy.native_core.layers):
        assert layer.self_attn.q_proj.weight.grad is not None
        assert torch.any(layer.self_attn.q_proj.weight.grad != 0.0), f"Layer {i} has no grad"

    # 6. Gradient reaches action head
    ah = seq_policy.policy.action_head
    assert ah.proj.weight.grad is not None
    assert torch.any(ah.proj.weight.grad != 0.0)


def test_4_causality_future_perturbation_invariance():
    """Test perturbation of future frame does not change earlier selected features or action predictions,
    while past frame perturbation does change subsequent predictions.
    """
    seq_policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    seq_policy.eval()

    sample_a = make_sample(N=4, target_indices=(0, 2), horizon=2, action_dim=4)
    with torch.no_grad():
        deep_a, shallow_a = seq_policy.features(sample_a)
        out_a = seq_policy(sample_a)

    # Perturb frame 3 (future with respect to target index 0 and 2)
    sample_b = copy.deepcopy(sample_a)
    sample_b["images_window"][3] = sample_b["images_window"][3] + 10.0

    with torch.no_grad():
        deep_b, shallow_b = seq_policy.features(sample_b)
        out_b = seq_policy(sample_b)

    # Future frame perturbation: targets 0 and 2 MUST be completely invariant
    assert torch.allclose(deep_a, deep_b, atol=1e-5)
    assert torch.allclose(shallow_a, shallow_b, atol=1e-5)
    assert torch.allclose(out_a["action_pred"], out_b["action_pred"], atol=1e-5)

    # Now perturb frame 0 (past with respect to target index 2)
    sample_c = copy.deepcopy(sample_a)
    sample_c["images_window"][0] = sample_c["images_window"][0] + 10.0

    with torch.no_grad():
        deep_c, shallow_c = seq_policy.features(sample_c)
        out_c = seq_policy(sample_c)

    # Past frame perturbation: target 2 features MUST change
    assert not torch.allclose(deep_a[1], deep_c[1], atol=1e-4)
    assert not torch.allclose(out_a["action_pred"][1], out_c["action_pred"][1], atol=1e-4)


def test_5_timestamp_prefix_and_validation_before_vit():
    """Test prompt has timestamp prefix before image/task prompt, and invalid order/types rejected before ViT."""
    seq_policy = make_tiny_training_policy(shallow_layer=1, use_timestamps=True)

    # 1. Check prompt timestamp prefix
    captured_prompts = []
    orig_fuse = seq_policy.policy.embedder._prepare_batch_and_fuse_embeddings

    def spy_fuse(prompts, vit_embeds_batch, image_masks, batch_num_tiles_list):
        captured_prompts.extend(prompts)
        return orig_fuse(prompts, vit_embeds_batch, image_masks, batch_num_tiles_list)

    seq_policy.policy.embedder._prepare_batch_and_fuse_embeddings = spy_fuse

    sample = make_sample(N=2, target_indices=(0, 1))
    sample["frame_ids"] = [10, 20]
    sample["observation_times"] = [1.5, 2.5]
    _ = seq_policy.features(sample)

    expected_p0 = format_frame_timestamp(10, 1.5)
    expected_p1 = format_frame_timestamp(20, 2.5)
    assert captured_prompts[0].startswith(expected_p0)
    assert captured_prompts[1].startswith(expected_p1)

    # 2. Validation: invalid frame_ids order rejected before extract_feature
    extract_called = False
    orig_extract = seq_policy.policy.embedder.model.extract_feature

    def mock_extract(pv):
        nonlocal extract_called
        extract_called = True
        return orig_extract(pv)

    seq_policy.policy.embedder.model.extract_feature = mock_extract

    bad_sample_fids = make_sample(N=2, target_indices=(0, 1))
    bad_sample_fids["frame_ids"] = [5, 3]  # not strictly increasing
    with pytest.raises(ValueError, match="frame_ids must be strictly increasing"):
        seq_policy.features(bad_sample_fids)
    assert not extract_called

    bad_sample_targets = make_sample(N=3, target_indices=(2, 1))  # not strictly increasing
    with pytest.raises(ValueError, match="target_indices must be strictly increasing"):
        seq_policy.features(bad_sample_targets)
    assert not extract_called

    bad_sample_masks = make_sample(N=2, target_indices=(0, 1))
    bad_sample_masks["image_masks"] = [torch.ones(1)]  # length != 2
    with pytest.raises(ValueError, match="image_masks length"):
        seq_policy.features(bad_sample_masks)
    assert not extract_called


def test_6_gradient_checkpointing_equivalence():
    """Test execute with checkpointing on vs off gives equal outputs and gradients, and mutable cache rejected."""
    sample = make_sample(N=3, target_indices=(0, 2), horizon=2, action_dim=4)

    # 1. Checkpoint off
    p_off = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    p_off.train()
    out_off = p_off(copy.deepcopy(sample))
    out_off["loss"].backward()
    grads_off = [p.grad.clone() for p in p_off.parameters() if p.grad is not None]

    # 2. Checkpoint on (same initialization)
    p_on = copy.deepcopy(p_off)
    p_on.gradient_checkpointing = True
    p_on.train()
    for p in p_on.parameters():
        p.grad = None
    out_on = p_on(copy.deepcopy(sample))
    out_on["loss"].backward()
    grads_on = [p.grad.clone() for p in p_on.parameters() if p.grad is not None]

    assert torch.allclose(out_off["loss"], out_on["loss"], atol=1e-6)
    assert len(grads_off) == len(grads_on)
    for g_off, g_on in zip(grads_off, grads_on):
        assert torch.allclose(g_off, g_on, atol=1e-5)

    # 3. Mutable cache rejected with checkpointing in execute_native_layers
    from transformers.cache_utils import DynamicCache
    from fabri_moss.native_cache import execute_native_layers
    dummy_cache = DynamicCache()
    core = p_on.native_core
    with pytest.raises(ValueError, match="Cannot enable gradient_checkpointing with mutable cache"):
        execute_native_layers(
            core=core,
            inputs_embeds=torch.randn(1, 16, 64),
            attention_mask_2d=torch.ones(1, 16),
            cache=dummy_cache,
            gradient_checkpointing=True,
        )


def test_7_one_frame_parity_with_native_cache_adapter():
    """Test features for N=1 use_timestamps=False equal NativeCacheAdapter on identical weights."""
    seq_policy = make_tiny_training_policy(shallow_layer=1, use_timestamps=False)
    seq_policy.eval()

    # Build adapter from copied policy
    adapter_policy = copy.deepcopy(seq_policy.policy)
    # Give adapter_policy embedder a _preprocess_images compatible with NativeCacheAdapter
    def mock_prep(images):
        pv, nt = adapter_policy.embedder._preprocess_images_on_cpu(images)
        return pv, nt
    adapter_policy.embedder._preprocess_images = mock_prep

    # Also make _prepare_and_fuse_embeddings call the batch one
    def mock_fuse(prompt, vit_embeds, image_mask, num_tiles_list):
        fe, fm = adapter_policy.embedder._prepare_batch_and_fuse_embeddings(
            prompts=[prompt],
            vit_embeds_batch=[vit_embeds],
            image_masks=[image_mask],
            batch_num_tiles_list=[num_tiles_list],
        )
        return fe, fm
    adapter_policy.embedder._prepare_and_fuse_embeddings = mock_fuse

    cache_config = NativeCacheConfig(max_frames=4, shallow_layer=1, use_timestamps=False)
    adapter = NativeCacheAdapter(adapter_policy, config=cache_config)
    adapter.eval()

    img = torch.randn(4, 16)
    prompt = "execute task"
    sample = {
        "images_window": [img],
        "frame_ids": [0],
        "observation_times": [0.0],
        "prompt": prompt,
        "target_indices": [0],
    }

    with torch.no_grad():
        deep_seq, shallow_seq = seq_policy.features(sample)

        block = adapter.encode_frame(images=[img], frame_id=0, prompt=prompt)
        deep_ad, shallow_ad, state = adapter.read_blocks([block], prompt=prompt)

    assert torch.allclose(deep_seq, deep_ad, atol=1e-5)
    assert torch.allclose(shallow_seq, shallow_ad, atol=1e-5)


def test_8_optimizer_updates_change_parameters_no_persisted_cache():
    """Test actual optimizer updates change policy parameters and no mutable cache is persisted across steps."""
    seq_policy = make_tiny_training_policy(shallow_layer=1)
    seq_policy.train()

    optimizer = torch.optim.SGD(seq_policy.parameters(), lr=1e-2)

    params_before = [p.clone().detach() for p in seq_policy.parameters()]

    # Step 1
    sample1 = make_sample(N=3, target_indices=(0, 2))
    optimizer.zero_grad()
    out1 = seq_policy(sample1)
    out1["loss"].backward()
    optimizer.step()

    params_after_step1 = [p.clone().detach() for p in seq_policy.parameters()]

    # Step 2
    sample2 = make_sample(N=3, target_indices=(1, 2))
    optimizer.zero_grad()
    out2 = seq_policy(sample2)
    out2["loss"].backward()
    optimizer.step()

    params_after_step2 = [p.clone().detach() for p in seq_policy.parameters()]

    # Verify parameters actually changed on both steps
    any_changed_1 = any(not torch.equal(p1, p2) for p1, p2 in zip(params_before, params_after_step1))
    any_changed_2 = any(not torch.equal(p1, p2) for p1, p2 in zip(params_after_step1, params_after_step2))
    assert any_changed_1
    assert any_changed_2

    # Verify no persistent cache attributes exist on policy or wrapper
    assert not hasattr(seq_policy, "cache")
    assert not hasattr(seq_policy.native_core, "past_key_values")
    assert not hasattr(seq_policy, "memory")
