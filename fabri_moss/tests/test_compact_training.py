"""Unit and integration tests for fabri_moss.compact_training with NativeCompactMemorySequencePolicy."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn

from fabri_moss.compact_cache import NativeCompactMemoryCacheAdapter
from fabri_moss.compact_memory import CompactMemoryConfig
from fabri_moss.compact_training import NativeCompactMemorySequencePolicy
from fabri_moss.native_cache import NativeCacheConfig, execute_native_layers
from fabri_moss.periodic_memory import PeriodicMemoryConfig
from fabri_moss.temporal_rope import TemporalRoPEConfig
from fabri_moss.tests.test_memory_training import MockTokenizerForMemory, build_test_memory_policy
from fabri_moss.tests.test_native_training import make_sample, make_tiny_training_policy


def _build_test_compact_policy(
    shallow_layer: int = 1,
    recent_frames: int = 2,
    consolidate_every: int = 2,
    memory_slots: int = 2,
    spatial_grid: int = 1,
    intermediate_grid: Optional[int] = 1,
    temporal_strength: float = 1.0,
    gradient_checkpointing: bool = False,
) -> Tuple[NativeCompactMemorySequencePolicy, nn.Module]:
    """Helper to instantiate NativeCompactMemorySequencePolicy wrapping tiny Qwen / fake ViT."""
    mem_policy, base_seq_policy = build_test_memory_policy(
        shallow_layer=shallow_layer,
        gradient_checkpointing=gradient_checkpointing,
    )
    policy = base_seq_policy.policy
    compact_cfg = CompactMemoryConfig(
        memory=PeriodicMemoryConfig(
            recent_frames=recent_frames,
            consolidate_every=consolidate_every,
            memory_slots=memory_slots,
            spatial_grid=spatial_grid,
            protect_decision_frames=True,
        ),
        intermediate_grid=intermediate_grid,
        temporal=TemporalRoPEConfig(strength=temporal_strength),
    )
    compact_policy = NativeCompactMemorySequencePolicy(
        policy=policy,
        shallow_layer=shallow_layer,
        use_timestamps=True,
        gradient_checkpointing=gradient_checkpointing,
        compact_config=compact_cfg,
    )
    return compact_policy, base_seq_policy


def test_compact_training_params_dense_fallback_and_constructor():
    """Verify: no added parameters, constructor rejects use_timestamps=False, dense/currentonly fallback."""
    compact_policy, base_seq_policy = _build_test_compact_policy()

    # No added parameters compared to inner policy
    wrapper_params = list(compact_policy.parameters())
    inner_params = list(compact_policy.policy.parameters())
    assert len(wrapper_params) == len(inner_params)
    for p1, p2 in zip(wrapper_params, inner_params):
        assert p1 is p2

    # Constructor rejects use_timestamps=False
    with pytest.raises(ValueError, match="use_timestamps=False is strictly forbidden"):
        NativeCompactMemorySequencePolicy(
            policy=compact_policy.policy,
            shallow_layer=1,
            use_timestamps=False,
        )

    # Dense fallback when memory_replay is False / absent
    dense_sample = make_sample(N=4, target_indices=[1, 3], horizon=2, action_dim=4)
    d_dense, sh_dense = compact_policy.features(dense_sample)
    assert d_dense.shape == (2, 16, 64)
    assert sh_dense.shape == (2, 16, 64)

    # Current-only fallback when replay_groups is present
    curr_sample = make_sample(N=2, target_indices=[0, 1], horizon=2, action_dim=4)
    curr_sample["replay_groups"] = [
        {"observation_indices": [0], "target_positions": [0]},
        {"observation_indices": [1], "target_positions": [1]},
    ]
    d_curr, sh_curr = compact_policy.features(curr_sample)
    assert d_curr.shape == (2, 16, 64)
    assert sh_curr.shape == (2, 16, 64)


def test_compact_training_n24_gradients_and_temporal_causality():
    """Verify: N=24 decisions [3,8,17,23] targets [17,23] full current L, cold/recent grad flow, future causality."""
    compact_policy, _ = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    compact_policy.train()

    sample = make_sample(N=24, target_indices=[17, 23], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [3, 8, 17, 23]

    img_cold = sample["images_window"][0]
    img_cold.requires_grad_(True)
    img_recent = sample["images_window"][18]
    img_recent.requires_grad_(True)

    deep, shallow = compact_policy.features(sample)
    # Target count is 2 (M=2), full current length is tokens_per_frame = 16, dim = 64
    assert deep.shape == (2, 16, 64)
    assert shallow.shape == (2, 16, 64)

    loss = deep.sum() + shallow.sum()
    loss.backward()

    # Cold prefix image 0 was encoded under torch.no_grad() -> grad must be None
    assert img_cold.grad is None
    # Recent frame image 18 was differentiably encoded -> grad must be non-zero
    assert img_recent.grad is not None and img_recent.grad.norm().item() > 0.0

    # Causality: future frames do not affect earlier targets
    compact_policy.eval()
    sample_a = make_sample(N=24, target_indices=[17, 23], horizon=2, action_dim=4)
    sample_a["memory_replay"] = True
    sample_a["decision_indices"] = [3, 8, 17, 23]

    sample_b = copy.deepcopy(sample_a)
    sample_b["images_window"][20] = torch.randn_like(sample_b["images_window"][20])
    sample_b["images_window"][21] = torch.randn_like(sample_b["images_window"][21])

    with torch.no_grad():
        d_a, sh_a = compact_policy.features(sample_a)
        d_b, sh_b = compact_policy.features(sample_b)

    # Earlier target (frame 17, idx 0) remains identical
    assert torch.equal(d_a[0], d_b[0])
    assert torch.equal(sh_a[0], sh_b[0])
    # Later target (frame 23, idx 1) is affected by perturbed frames
    assert not torch.equal(d_a[1], d_b[1])


def test_compact_training_checkpointing_and_strength_zero_core_parity():
    """Verify: checkpoint on/off numerical parity, and 0-strength temporal RoPE vs original core on identical inputs."""
    compact_policy, base_seq_policy = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    p_off = compact_policy
    p_on, _ = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=True)
    p_on.load_state_dict(p_off.state_dict())

    sample = make_sample(N=24, target_indices=[17, 23], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [3, 8, 17, 23]

    p_off.train()
    p_on.train()
    d_off, sh_off = p_off.features(sample)
    d_on, sh_on = p_on.features(sample)
    assert torch.allclose(d_off, d_on, atol=1e-5)
    assert torch.allclose(sh_off, sh_on, atol=1e-5)

    # 0-strength temporal RoPE on identical inputs matches original execute_native_layers exactly
    core = compact_policy.native_core
    inputs_embeds = torch.randn(1, 10, 64)
    attention_mask = torch.ones(1, 10, dtype=torch.long)
    token_times = torch.linspace(0, 5, 10).unsqueeze(0)
    cfg0 = TemporalRoPEConfig(strength=0.0)

    out_temporal, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs_embeds,
        attention_mask_2d=attention_mask,
        token_times=token_times,
        temporal_config=cfg0,
    )
    out_orig, _ = execute_native_layers(
        core=core,
        inputs_embeds=inputs_embeds,
        attention_mask_2d=attention_mask,
        token_times=None,
        temporal_config=None,
    )
    assert torch.allclose(out_temporal, out_orig, atol=1e-6)


def test_compact_training_wrapper_vs_adapter_parity():
    """Verify: NativeCompactMemorySequencePolicy matches NativeCompactMemoryCacheAdapter across decision boundaries."""
    compact_policy, base_seq_policy = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    policy = base_seq_policy.policy

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

    compact_policy.eval()

    N = 24
    decision_indices = [3, 8, 17, 23]
    target_indices = [17, 23]
    sample = make_sample(N=N, target_indices=target_indices, horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = decision_indices

    with torch.no_grad():
        deep_train, shallow_train = compact_policy.features(sample)

    cache_config = NativeCacheConfig(
        max_frames=4,
        shallow_layer=1,
        use_timestamps=True,
    )
    adapter = NativeCompactMemoryCacheAdapter(
        policy=policy,
        config=cache_config,
        compact_config=compact_policy.compact_config,
        background_rebuild=False,
    )
    adapter.eval()

    prompt = sample["prompt"]
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

    assert torch.allclose(deep_train, deep_cache, atol=1e-5)
    assert torch.allclose(shallow_train, shallow_cache, atol=1e-5)
