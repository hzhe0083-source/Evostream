"""Unit tests for fabri_moss.memory_training with NativeMemorySequencePolicy."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn

from fabri_moss.memory_training import NativeMemorySequencePolicy
from fabri_moss.periodic_memory import PeriodicMemoryConfig
from fabri_moss.tests.test_native_training import (
    TinyTokenizer,
    make_sample,
    make_tiny_training_policy,
)


class MockTokenizerForMemory:
    """Tokenizer wrapper satisfying the materialize_memory header tokenization contract."""

    def __init__(self, inner: Optional[TinyTokenizer] = None):
        self.inner = inner if inner is not None else TinyTokenizer()

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
    ) -> Any:
        return self.inner(text, return_tensors=return_tensors)


def build_test_memory_policy(
    shallow_layer: int = 1,
    recent_frames: int = 2,
    consolidate_every: int = 2,
    memory_slots: int = 2,
    spatial_grid: int = 1,
    gradient_checkpointing: bool = False,
) -> Tuple[NativeMemorySequencePolicy, nn.Module]:
    """Create a NativeMemorySequencePolicy with MockTokenizerForMemory wrapping the inner TinyTokenizer."""
    base_seq_policy = make_tiny_training_policy(
        shallow_layer=shallow_layer,
        gradient_checkpointing=gradient_checkpointing,
    )
    policy = base_seq_policy.policy
    policy.embedder.tokenizer = MockTokenizerForMemory(policy.embedder.tokenizer)

    mem_config = PeriodicMemoryConfig(
        recent_frames=recent_frames,
        consolidate_every=consolidate_every,
        memory_slots=memory_slots,
        spatial_grid=spatial_grid,
        protect_decision_frames=True,
    )
    mem_policy = NativeMemorySequencePolicy(
        policy=policy,
        shallow_layer=shallow_layer,
        memory_config=mem_config,
        gradient_checkpointing=gradient_checkpointing,
    )
    return mem_policy, base_seq_policy


def test_1_forward_backward_shapes_params_and_gradient_flow():
    """Test 1:

    - Forward/backward output shapes with M=2.
    - Parameters not increased compared to inner policy.
    - image0 set requires_grad: grad should be None/0 (retired and detached into summary).
    - image17 set requires_grad: grad should be non-zero (within recent differentiable window).
    - Four groups of original model parameters have non-zero gradients.
    """
    mem_policy, _ = build_test_memory_policy(
        shallow_layer=1,
        gradient_checkpointing=False,
    )
    mem_policy.train()

    # Verify parameters not increased
    wrapper_params = list(mem_policy.parameters())
    inner_params = list(mem_policy.policy.parameters())
    assert len(wrapper_params) == len(inner_params)
    for p in wrapper_params:
        assert p.requires_grad is True

    # Build sample with N=20, target_indices=[18, 19] (M=2)
    sample = make_sample(N=20, target_indices=[18, 19], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [3, 7, 12, 18, 19]

    # Test feature extraction shapes
    deep, shallow = mem_policy.features(sample)
    assert deep.shape == (2, 16, 64)
    assert shallow.shape == (2, 16, 64)
    assert deep.dtype == torch.float32
    assert shallow.dtype == torch.float32

    # Forward pass
    out = mem_policy(sample)
    assert "loss" in out
    assert "loss_sum" in out
    assert out["target_count"] == 2
    assert out["action_pred"].shape == (2, 2, 4)

    # Gradient flow test: set requires_grad on image0 and image17
    img0 = sample["images_window"][0].clone().detach().requires_grad_(True)
    img17 = sample["images_window"][17].clone().detach().requires_grad_(True)
    sample["images_window"][0] = img0
    sample["images_window"][17] = img17

    mem_policy.zero_grad()
    out_grad = mem_policy(sample)
    out_grad["loss"].backward()

    # image0 is retired before cold_end (or detached during memory advance), so grad is None or 0
    if img0.grad is not None:
        assert torch.all(img0.grad == 0), f"img0 grad should be 0, got norm {img0.grad.norm().item()}"
    else:
        assert img0.grad is None

    # image17 is in recent buffer for targets 18 and 19 (within recent_frames + consolidate_every window)
    assert img17.grad is not None, "img17 grad should not be None"
    assert img17.grad.norm().item() > 0.0, "img17 should receive non-zero gradients"

    # Verify original model parameter groups have gradients:
    # 1. ViT encoder
    vit_grad = mem_policy.policy.embedder.model.vision_model.encoder.conv.weight.grad
    assert vit_grad is not None and vit_grad.norm().item() > 0.0, "ViT encoder must have non-zero grad"

    # 2. Projector
    proj_grad = mem_policy.policy.embedder.model.projector.proj.weight.grad
    assert proj_grad is not None and proj_grad.norm().item() > 0.0, "Projector must have non-zero grad"

    # 3. Token embeddings / LLM layers
    embed_grad = mem_policy.policy.embedder.model.language_model.model.embed_tokens.weight.grad
    assert embed_grad is not None and embed_grad.norm().item() > 0.0, "Embed tokens must have non-zero grad"
    llm_grad = mem_policy.policy.embedder.model.language_model.model.layers[0].self_attn.q_proj.weight.grad
    assert llm_grad is not None and llm_grad.norm().item() > 0.0, "LLM layers must have non-zero grad"

    # 4. Action head
    action_grad = mem_policy.policy.action_head.proj.weight.grad
    assert action_grad is not None and action_grad.norm().item() > 0.0, "Action head must have non-zero grad"


def test_2_causal_and_memory_receptive_field():
    """Test 2:

    - Modifying image0 significantly changes the final target deep features (effect of summary across >16 frames).
    - Modifying image19 does NOT affect target 18 features (strict causality).
    """
    mem_policy, _ = build_test_memory_policy(
        shallow_layer=1,
        gradient_checkpointing=False,
    )
    mem_policy.eval()

    sample_base = make_sample(N=20, target_indices=[18, 19])
    sample_base["memory_replay"] = True
    sample_base["decision_indices"] = [3, 7, 12, 18, 19]

    with torch.no_grad():
        deep_base, _ = mem_policy.features(sample_base)

    # 1. Modifying image0 (far past frame > 16 frames back)
    sample_mod_img0 = copy.deepcopy(sample_base)
    sample_mod_img0["images_window"][0] = sample_mod_img0["images_window"][0] + 50.0

    with torch.no_grad():
        deep_mod_img0, _ = mem_policy.features(sample_mod_img0)

    diff_target19_from_img0 = (deep_base[1] - deep_mod_img0[1]).abs().max().item()
    assert diff_target19_from_img0 > 1e-4, (
        f"Modifying image0 should affect target 19 via memory summary, diff: {diff_target19_from_img0}"
    )

    # 2. Modifying image19 should not affect target 18
    sample_mod_img19 = copy.deepcopy(sample_base)
    sample_mod_img19["images_window"][19] = sample_mod_img19["images_window"][19] + 50.0

    with torch.no_grad():
        deep_mod_img19, _ = mem_policy.features(sample_mod_img19)

    diff_target18_from_img19 = (deep_base[0] - deep_mod_img19[0]).abs().max().item()
    diff_target19_from_img19 = (deep_base[1] - deep_mod_img19[1]).abs().max().item()

    assert diff_target18_from_img19 == 0.0, (
        f"Modifying image 19 should not affect target 18, got diff: {diff_target18_from_img19}"
    )
    assert diff_target19_from_img19 > 1e-4, (
        f"Modifying image 19 must affect target 19, got diff: {diff_target19_from_img19}"
    )


def test_3_checkpointing_equivalence_and_optimizer_steps():
    """Test 3:

    - Identical weights with gradient_checkpointing on/off give allclose outputs and parameter gradients.
    - Two consecutive optimizer steps succeed and backward functions cleanly.
    - Model does not hold any persistent memory_cache attribute.
    """
    torch.manual_seed(42)
    mem_off, _ = build_test_memory_policy(gradient_checkpointing=False)
    mem_off.train()

    policy_on = copy.deepcopy(mem_off.policy)
    mem_on = NativeMemorySequencePolicy(
        policy=policy_on,
        shallow_layer=1,
        memory_config=copy.deepcopy(mem_off.memory_config),
        gradient_checkpointing=True,
    )
    mem_on.train()

    sample = make_sample(N=20, target_indices=[18, 19])
    sample["memory_replay"] = True
    sample["decision_indices"] = [3, 7, 12, 18, 19]

    out_off = mem_off(sample)
    out_on = mem_on(sample)

    assert torch.allclose(out_off["loss"], out_on["loss"], atol=1e-5), "Loss mismatch between checkpointing on/off"
    assert torch.allclose(out_off["action_pred"], out_on["action_pred"], atol=1e-5), (
        "Action prediction mismatch between checkpointing on/off"
    )

    out_off["loss"].backward()
    out_on["loss"].backward()

    for (p_off_name, p_off), (p_on_name, p_on) in zip(mem_off.named_parameters(), mem_on.named_parameters()):
        if p_off.grad is not None:
            assert p_on.grad is not None, f"Parameter {p_on_name} missing gradient in checkpointed model"
            assert torch.allclose(p_off.grad, p_on.grad, atol=1e-5), (
                f"Gradient mismatch for {p_off_name}: max diff {(p_off.grad - p_on.grad).abs().max()}"
            )

    # Verify 2 consecutive optimizer steps
    optimizer = torch.optim.SGD(mem_off.parameters(), lr=0.01)
    for step in range(2):
        optimizer.zero_grad()
        s = make_sample(N=20, target_indices=[18, 19])
        s["memory_replay"] = True
        s["decision_indices"] = [3, 7, 12, 18, 19]
        step_out = mem_off(s)
        step_out["loss"].backward()
        optimizer.step()

        # Model should not retain memory_cache across steps
        assert not hasattr(mem_off, "memory_cache") or getattr(mem_off, "memory_cache", None) is None
        assert not hasattr(mem_off.policy, "memory_cache") or getattr(mem_off.policy, "memory_cache", None) is None


def test_4_memory_flag_absent_backward_compatibility():
    """Test 4:

    - When memory_replay is absent or False, features output matches NativeSequencePolicy on identical policy.
    """
    mem_policy, base_seq_policy = build_test_memory_policy(gradient_checkpointing=False)
    mem_policy.eval()
    base_seq_policy.eval()

    sample = make_sample(N=20, target_indices=[18, 19])
    assert "memory_replay" not in sample

    with torch.no_grad():
        deep_base, shallow_base = base_seq_policy.features(sample)
        deep_mem, shallow_mem = mem_policy.features(sample)

    assert torch.allclose(deep_base, deep_mem, atol=1e-6), "deep features differ when memory_replay is absent"
    assert torch.allclose(shallow_base, shallow_mem, atol=1e-6), "shallow features differ when memory_replay is absent"


def test_5_invalid_decision_indices_validation_before_vit():
    """Test 5: Validation of decision_indices before ViT encoding.

    Checks:
    - missing decision_indices raises ValueError
    - empty decision_indices raises ValueError
    - boolean in decision_indices raises TypeError
    - out of order / non-increasing decision_indices raises ValueError
    - duplicate decision_indices raises ValueError
    - out of bounds decision_indices raises IndexError
    - target missing from decision_indices raises ValueError
    - last decision index != last target index raises ValueError
    """
    mem_policy, _ = build_test_memory_policy(gradient_checkpointing=False)
    mem_policy.eval()

    # 1. missing decision_indices
    sample = make_sample(N=20, target_indices=[18, 19])
    sample["memory_replay"] = True
    with pytest.raises(ValueError, match="must contain 'decision_indices'"):
        mem_policy.features(sample)

    # 2. empty decision_indices
    sample["decision_indices"] = []
    with pytest.raises(ValueError, match="cannot be empty"):
        mem_policy.features(sample)

    # 3. boolean in decision_indices
    sample["decision_indices"] = [True, 18, 19]
    with pytest.raises(TypeError, match="must contain non-bool integers"):
        mem_policy.features(sample)

    # 4. out of order (decreasing)
    sample["decision_indices"] = [10, 5, 18, 19]
    with pytest.raises(ValueError, match="strictly increasing"):
        mem_policy.features(sample)

    # 5. duplicate indices
    sample["decision_indices"] = [5, 5, 18, 19]
    with pytest.raises(ValueError, match="strictly increasing"):
        mem_policy.features(sample)

    # 6. out of bounds
    sample["decision_indices"] = [5, 18, 19, 25]  # N=20
    with pytest.raises(IndexError, match="out of bounds"):
        mem_policy.features(sample)

    # 7. target missing from decision_indices
    sample["decision_indices"] = [3, 7, 19]  # target 18 is missing!
    with pytest.raises(ValueError, match="target_indices must be a subset of decision_indices"):
        mem_policy.features(sample)

    # 8. last decision != last target (e.g. post-target decision)
    sample_large = make_sample(N=25, target_indices=[18, 19])
    sample_large["memory_replay"] = True
    sample_large["decision_indices"] = [18, 19, 22]
    with pytest.raises(ValueError, match="last decision index.*must equal last target index"):
        mem_policy.features(sample_large)


def test_6_is_decision_attribute_and_visual_token_dtype():
    """Test 6: Verify is_decision flag matches (absolute_pool_index in decision_set)

    across both cold and hot frames, and visual_tokens are cast to seq_inputs_embeds dtype.
    """
    mem_policy, _ = build_test_memory_policy(
        shallow_layer=1,
        gradient_checkpointing=False,
    )
    mem_policy.eval()

    sample = make_sample(N=20, target_indices=[18, 19])
    sample["memory_replay"] = True
    # Let decision indices include cold frames (e.g. 3, 7, 12) and hot frames (18, 19)
    sample["decision_indices"] = [3, 7, 12, 18, 19]

    created_frames = []
    orig_advance = None

    import fabri_moss.memory_training as mt
    orig_advance = mt.advance_memory

    def spy_advance(state, frame, config):
        created_frames.append(frame)
        return orig_advance(state, frame, config)

    mt.advance_memory = spy_advance
    try:
        deep, shallow = mem_policy.features(sample)
    finally:
        mt.advance_memory = orig_advance

    assert len(created_frames) == 20
    decision_set = {3, 7, 12, 18, 19}
    for idx, f in enumerate(created_frames):
        assert f.is_decision == (idx in decision_set), f"Frame {idx} is_decision mismatch: {f.is_decision}"
        # visual_tokens dtype matches inputs_embeds dtype
        assert f.visual_tokens.dtype == f.inputs_embeds.dtype, (
            f"Frame {idx} visual_tokens dtype {f.visual_tokens.dtype} != inputs_embeds dtype {f.inputs_embeds.dtype}"
        )
