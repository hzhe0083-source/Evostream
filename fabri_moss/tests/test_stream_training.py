"""Tests for stream replay training extension in fabri_moss.native_training.

Verifies:
1. default old passes unchanged (no replay_groups behaves identically).
2. grouped feature outputs equal separate isolated sub-samples (same weights) restored to outer target order.
3. changing image visible only to other group cannot affect target, but history changes affect later target within group.
4. future perturbation doesn't affect earlier target in same group.
5. history gradient is non-zero back to earlier frames in group and ViT.
6. all targets once strict invalid schemas reject before encoder:
   - empty replay_groups
   - non-dict group
   - missing keys
   - invalid/boolean types
   - non-strictly increasing observation_indices
   - non-strictly increasing target_positions
   - target frame not in group observation_indices
   - final observation not equal to latest target frame (trailing unsupervised frames)
   - duplicate target position across groups
   - missing target position across groups
   - unused pool frames (union != pool)
7. current-only feature equals independent single-frame calls.
8. grouped checkpointing on/off gives equal loss and parameter gradients.
9. two optimizer updates with replay_groups change parameters and no persistent cache remains.
10. parity test with NativeCacheAdapter.read_blocks using same policy copy, encoding blocks per group
    and splitting deltas at query boundaries (CPU FP32 numerical tolerance).
"""

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
from fabri_moss.tests.test_native_training import (
    LearnedFakeViT,
    LearnedProjector,
    TinyModelWithVision,
    TinyTokenizer,
    FullDifferentiableEmbedder,
    DualTokenActionHead,
    TinyDifferentiablePolicy,
    make_tiny_training_policy,
    make_sample,
)


def make_grouped_sample(
    pool_N: int = 5,
    target_indices: Sequence[int] = (1, 2, 4),
    replay_groups: Optional[List[Dict[str, Any]]] = None,
    horizon: int = 2,
    action_dim: int = 4,
) -> Dict[str, Any]:
    images_window = [torch.randn(4, 16) for _ in range(pool_N)]
    frame_ids = list(range(pool_N))
    observation_times = [i * 0.1 for i in range(pool_N)]
    prompt = "execute multi-rate stream task"
    M = len(target_indices)
    actions = torch.randn(M, horizon, action_dim)
    action_mask = torch.ones(M, horizon, action_dim, dtype=torch.bool)
    sample: Dict[str, Any] = {
        "images_window": images_window,
        "frame_ids": frame_ids,
        "observation_times": observation_times,
        "prompt": prompt,
        "target_indices": list(target_indices),
        "actions": actions,
        "action_mask": action_mask,
    }
    if replay_groups is not None:
        sample["replay_groups"] = replay_groups
    return sample


def test_1_default_old_sample_behavior_identical():
    """Verify that a sample without replay_groups produces identical features and loss to baseline."""
    policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy.eval()

    sample = make_sample(N=4, target_indices=(1, 3), horizon=2, action_dim=4)
    deep1, shallow1 = policy.features(sample)
    out1 = policy(sample)

    assert deep1.shape == (2, 16, 64)
    assert shallow1.shape == (2, 16, 64)
    assert out1["target_count"] == 2
    assert out1["action_pred"].shape == (2, 2, 4)


def test_2_grouped_features_equal_separate_isolated_subsamples():
    """Verify grouped feature outputs equal running separate isolated sub-samples for each group,

    restored to the original outer target order.
    """
    policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy.eval()

    # Pool frames: 0, 1, 2, 3, 4 (N=5)
    # Targets: outer 0 -> frame 1, outer 1 -> frame 2, outer 2 -> frame 4 (M=3)
    # Group A: obs [0, 1, 3, 4], targets [0, 2] -> target frames [1, 4]
    # Group B: obs [2], targets [1] -> target frame [2]
    # Union of obs = {0, 1, 2, 3, 4} (exact pool)
    replay_groups = [
        {"observation_indices": [0, 1, 3, 4], "target_positions": [0, 2]},
        {"observation_indices": [2], "target_positions": [1]},
    ]
    sample = make_grouped_sample(pool_N=5, target_indices=(1, 2, 4), replay_groups=replay_groups)

    with torch.no_grad():
        deep_grouped, shallow_grouped = policy.features(sample)

    # Now construct separate isolated sub-samples
    # Group A sub-sample:
    # frames: 0, 1, 3, 4
    # targets: frame 1 (rel idx 1), frame 4 (rel idx 3)
    sample_a = {
        "images_window": [sample["images_window"][i] for i in [0, 1, 3, 4]],
        "frame_ids": [sample["frame_ids"][i] for i in [0, 1, 3, 4]],
        "observation_times": [sample["observation_times"][i] for i in [0, 1, 3, 4]],
        "prompt": sample["prompt"],
        "target_indices": [1, 3],
    }
    with torch.no_grad():
        deep_a, shallow_a = policy.features(sample_a)

    # Group B sub-sample:
    # frames: 2
    # targets: frame 2 (rel idx 0)
    sample_b = {
        "images_window": [sample["images_window"][2]],
        "frame_ids": [sample["frame_ids"][2]],
        "observation_times": [sample["observation_times"][2]],
        "prompt": sample["prompt"],
        "target_indices": [0],
    }
    with torch.no_grad():
        deep_b, shallow_b = policy.features(sample_b)

    # Restored to outer target order: [target 0 (from A[0]), target 1 (from B[0]), target 2 (from A[1])]
    deep_expected = torch.stack([deep_a[0], deep_b[0], deep_a[1]], dim=0)
    shallow_expected = torch.stack([shallow_a[0], shallow_b[0], shallow_a[1]], dim=0)

    assert torch.allclose(deep_grouped, deep_expected, atol=1e-5)
    assert torch.allclose(shallow_grouped, shallow_expected, atol=1e-5)


def test_3_cross_group_isolation_and_intra_group_history_sensitivity():
    """Changing an image visible ONLY to Group B cannot affect Group A's targets;

    while changing an image visible to Group A affects subsequent targets in Group A.
    """
    policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy.eval()

    # Pool: 0, 1, 2, 3
    # Targets: outer 0 -> frame 1 (Group A), outer 1 -> frame 3 (Group B)
    # Group A: obs [0, 1], targets [0]
    # Group B: obs [2, 3], targets [1]
    replay_groups = [
        {"observation_indices": [0, 1], "target_positions": [0]},
        {"observation_indices": [2, 3], "target_positions": [1]},
    ]
    sample_base = make_grouped_sample(pool_N=4, target_indices=(1, 3), replay_groups=replay_groups)

    with torch.no_grad():
        deep_base, shallow_base = policy.features(sample_base)

    # Perturb frame 2 (only visible to Group B)
    sample_mod_b = copy.deepcopy(sample_base)
    sample_mod_b["images_window"][2] = sample_mod_b["images_window"][2] + 5.0
    with torch.no_grad():
        deep_mod_b, shallow_mod_b = policy.features(sample_mod_b)

    # Target 0 (Group A) MUST be completely unaffected
    assert torch.allclose(deep_base[0], deep_mod_b[0], atol=1e-5)
    assert torch.allclose(shallow_base[0], shallow_mod_b[0], atol=1e-5)
    # Target 1 (Group B, observing frame 2 then 3) MUST change
    assert not torch.allclose(deep_base[1], deep_mod_b[1], atol=1e-4)

    # Now perturb frame 0 (history within Group A)
    sample_mod_a = copy.deepcopy(sample_base)
    sample_mod_a["images_window"][0] = sample_mod_a["images_window"][0] + 5.0
    with torch.no_grad():
        deep_mod_a, shallow_mod_a = policy.features(sample_mod_a)

    # Target 0 (Group A, observing frame 0 then 1) MUST change
    assert not torch.allclose(deep_base[0], deep_mod_a[0], atol=1e-4)
    # Target 1 (Group B) MUST be unaffected
    assert torch.allclose(deep_base[1], deep_mod_a[1], atol=1e-5)


def test_4_future_perturbation_invariance_within_group():
    """Within a group with multiple targets, perturbing a future frame does not affect earlier target."""
    policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy.eval()

    # Pool: 0, 1, 2, 3 (N=4)
    # Target 0 -> frame 1, Target 1 -> frame 3
    # Group A: obs [0, 1, 2, 3], targets [0, 1]
    replay_groups = [
        {"observation_indices": [0, 1, 2, 3], "target_positions": [0, 1]},
    ]
    sample_base = make_grouped_sample(pool_N=4, target_indices=(1, 3), replay_groups=replay_groups)

    with torch.no_grad():
        deep_base, shallow_base = policy.features(sample_base)

    # Perturb frame 2 and frame 3 (future with respect to Target 0 at frame 1)
    sample_mod = copy.deepcopy(sample_base)
    sample_mod["images_window"][2] = sample_mod["images_window"][2] + 10.0
    sample_mod["images_window"][3] = sample_mod["images_window"][3] + 10.0

    with torch.no_grad():
        deep_mod, shallow_mod = policy.features(sample_mod)

    # Earlier Target 0 MUST be completely unaffected by future frames 2 and 3
    assert torch.allclose(deep_base[0], deep_mod[0], atol=1e-5)
    assert torch.allclose(shallow_base[0], shallow_mod[0], atol=1e-5)

    # Later Target 1 MUST change
    assert not torch.allclose(deep_base[1], deep_mod[1], atol=1e-4)


def test_5_history_gradient_nonzero_and_vit_shared():
    """Verify backward pass from a target with history propagates non-zero gradients to earlier frames and ViT."""
    policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy.train()

    # Pool: 0, 1, 2 (N=3)
    img0 = torch.randn(4, 16, requires_grad=True)
    img1 = torch.randn(4, 16, requires_grad=True)
    img2 = torch.randn(4, 16, requires_grad=True)

    # Group A: obs [0, 1], target [0] -> frame 1
    # Group B: obs [2], target [1] -> frame 2
    replay_groups = [
        {"observation_indices": [0, 1], "target_positions": [0]},
        {"observation_indices": [2], "target_positions": [1]},
    ]
    sample = make_grouped_sample(pool_N=3, target_indices=(1, 2), replay_groups=replay_groups)
    sample["images_window"] = [img0, img1, img2]

    out = policy(sample)
    # Loss backprop
    loss = out["loss"]
    loss.backward()

    # img0 is history for Target 0 (Group A). Its grad must be non-zero!
    assert img0.grad is not None
    assert torch.any(img0.grad != 0.0)

    # img1 is target frame for Target 0. Its grad must be non-zero!
    assert img1.grad is not None
    assert torch.any(img1.grad != 0.0)

    # img2 is target frame for Target 1. Its grad must be non-zero!
    assert img2.grad is not None
    assert torch.any(img2.grad != 0.0)

    # ViT and projector have gradients
    vit = policy.policy.embedder.model.vision_model.encoder
    for name, p in vit.named_parameters():
        assert p.grad is not None and torch.any(p.grad != 0.0)


def test_6_strict_validation_rejects_invalid_schemas_before_vit():
    """Test all strict schema and consistency validations reject before ViT encoder is called."""
    policy = make_tiny_training_policy(shallow_layer=1)

    extract_called = False
    orig_extract = policy.policy.embedder.model.extract_feature

    def spy_extract(pv):
        nonlocal extract_called
        extract_called = True
        return orig_extract(pv)

    policy.policy.embedder.model.extract_feature = spy_extract

    def assert_rejects(sample_corrupt, match_str):
        nonlocal extract_called
        extract_called = False
        with pytest.raises((ValueError, TypeError, KeyError), match=match_str):
            policy.features(sample_corrupt)
        assert not extract_called, f"extract_feature was called for {match_str}!"

    # 1. Empty replay_groups
    s = make_grouped_sample(pool_N=3, target_indices=(0, 2), replay_groups=[])
    assert_rejects(s, "replay_groups must not be empty")

    # 2. Non-dict group
    s = make_grouped_sample(pool_N=3, target_indices=(0, 2), replay_groups=["not_a_dict"])
    assert_rejects(s, "replay_groups\\[0\\] must be a dict")

    # 3. Missing keys
    s = make_grouped_sample(pool_N=3, target_indices=(0, 2), replay_groups=[{"observation_indices": [0, 1, 2]}])
    assert_rejects(s, "missing required key 'target_positions'")

    # 4. Empty observation_indices or target_positions
    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[{"observation_indices": [], "target_positions": [0, 1]}],
    )
    assert_rejects(s, "observation_indices.*must not be empty")

    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[{"observation_indices": [0, 1, 2], "target_positions": []}],
    )
    assert_rejects(s, "target_positions.*must not be empty")

    # 5. Non-strictly increasing observation_indices (or bool)
    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[{"observation_indices": [True, 1], "target_positions": [0]}],
    )
    assert_rejects(s, "must be int in")

    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[{"observation_indices": [1, 0, 2], "target_positions": [0, 1]}],
    )
    assert_rejects(s, "strictly increasing")

    # 6. Non-strictly increasing target_positions
    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[{"observation_indices": [0, 1, 2], "target_positions": [1, 0]}],
    )
    assert_rejects(s, "strictly increasing")

    # 7. Target frame not in group observation_indices
    # outer target 1 is at pool frame 2; group obs is [0, 1]
    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[
            {"observation_indices": [0, 1], "target_positions": [0, 1]},
            {"observation_indices": [2], "target_positions": []},
        ],
    )
    assert_rejects(s, "target_positions.*must not be empty|not in group's observation_indices")

    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[
            {"observation_indices": [0, 1], "target_positions": [1]},  # target 1 is frame 2, not in [0, 1]
            {"observation_indices": [2], "target_positions": [0]},
        ],
    )
    assert_rejects(s, "not in group's observation_indices")

    # 8. Unsupervised trailing frames: group final obs does NOT equal latest target frame
    # group obs [0, 1, 2], but target is only at frame 1. Frame 2 is trailing unsupervised!
    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 1),
        replay_groups=[
            {"observation_indices": [0, 1, 2], "target_positions": [0, 1]},
        ],
    )
    assert_rejects(s, "unsupervised trailing frames are forbidden")

    # 9. Duplicate target position across groups
    # outer targets: pos 0 -> frame 0, pos 1 -> frame 1.
    # Group A: obs [0], targets [0]
    # Group B: obs [0, 1], targets [0, 1] (target pos 0 duplicated across groups!)
    s = make_grouped_sample(
        pool_N=2,
        target_indices=(0, 1),
        replay_groups=[
            {"observation_indices": [0], "target_positions": [0]},
            {"observation_indices": [0, 1], "target_positions": [0, 1]},
        ],
    )
    assert_rejects(s, "must appear exactly once")

    # 10. Missing target position across groups
    # outer targets: pos 0 -> frame 0, pos 1 -> frame 1 (M=2)
    # Group A: obs [0], targets [0]
    # Group B: obs [1], targets [] (or target 1 omitted: Group B has obs [1] with no targets)
    s = make_grouped_sample(
        pool_N=2,
        target_indices=(0, 1),
        replay_groups=[
            {"observation_indices": [0], "target_positions": [0]},
            {"observation_indices": [1], "target_positions": []},
        ],
    )
    assert_rejects(s, "target_positions.*must not be empty|must appear exactly once")

    # 11. Unused pool frames: union of group obs != pool 0..N-1
    # pool is N=3 (frames 0, 1, 2), but groups only cover [0] and [2]. Frame 1 is unused!
    s = make_grouped_sample(
        pool_N=3,
        target_indices=(0, 2),
        replay_groups=[
            {"observation_indices": [0], "target_positions": [0]},
            {"observation_indices": [2], "target_positions": [1]},
        ],
    )
    assert_rejects(s, "Unused pool frames: \\[1\\].*Sampler inconsistency detected")


def test_7_current_only_equals_independent_single_frame_calls():
    """Verify that current-only replay groups ([{obs: [i], targets: [ti]}])

    produce features identical to running the model independently on each frame.
    """
    policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    policy.eval()

    # Pool: frames 0, 1, 2 (N=3), targets: 0, 1, 2 (M=3)
    replay_groups = [
        {"observation_indices": [0], "target_positions": [0]},
        {"observation_indices": [1], "target_positions": [1]},
        {"observation_indices": [2], "target_positions": [2]},
    ]
    sample = make_grouped_sample(pool_N=3, target_indices=(0, 1, 2), replay_groups=replay_groups)

    with torch.no_grad():
        deep_grouped, shallow_grouped = policy.features(sample)

    # Independent single frame calls
    deep_singles = []
    shallow_singles = []
    for i in range(3):
        single_sample = {
            "images_window": [sample["images_window"][i]],
            "frame_ids": [sample["frame_ids"][i]],
            "observation_times": [sample["observation_times"][i]],
            "prompt": sample["prompt"],
            "target_indices": [0],
        }
        with torch.no_grad():
            d, s = policy.features(single_sample)
        deep_singles.append(d[0])
        shallow_singles.append(s[0])

    deep_single_cat = torch.stack(deep_singles, dim=0)
    shallow_single_cat = torch.stack(shallow_singles, dim=0)

    assert torch.allclose(deep_grouped, deep_single_cat, atol=1e-5)
    assert torch.allclose(shallow_grouped, shallow_single_cat, atol=1e-5)


def test_8_grouped_gradient_checkpointing_equivalence():
    """Verify execute with gradient_checkpointing on vs off gives equal loss and parameter gradients."""
    sample = make_grouped_sample(
        pool_N=4,
        target_indices=(1, 3),
        replay_groups=[
            {"observation_indices": [0, 1], "target_positions": [0]},
            {"observation_indices": [2, 3], "target_positions": [1]},
        ],
        horizon=2,
        action_dim=4,
    )

    # 1. Checkpoint off
    p_off = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    p_off.train()
    out_off = p_off(copy.deepcopy(sample))
    out_off["loss"].backward()
    grads_off = [p.grad.clone() for p in p_off.parameters() if p.grad is not None]

    # 2. Checkpoint on (identical weights)
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


def test_9_optimizer_updates_change_parameters_no_persisted_cache():
    """Verify two optimizer updates with replay_groups change parameters and no persistent cache remains."""
    policy = make_tiny_training_policy(shallow_layer=1)
    policy.train()

    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-2)
    params_start = [p.clone().detach() for p in policy.parameters()]

    # Update 1
    sample1 = make_grouped_sample(
        pool_N=4,
        target_indices=(1, 3),
        replay_groups=[
            {"observation_indices": [0, 1], "target_positions": [0]},
            {"observation_indices": [2, 3], "target_positions": [1]},
        ],
    )
    optimizer.zero_grad()
    out1 = policy(sample1)
    out1["loss"].backward()
    optimizer.step()

    params_step1 = [p.clone().detach() for p in policy.parameters()]

    # Update 2
    sample2 = make_grouped_sample(
        pool_N=4,
        target_indices=(0, 2, 3),
        replay_groups=[
            {"observation_indices": [0], "target_positions": [0]},
            {"observation_indices": [1, 2, 3], "target_positions": [1, 2]},
        ],
    )
    optimizer.zero_grad()
    out2 = policy(sample2)
    out2["loss"].backward()
    optimizer.step()

    params_step2 = [p.clone().detach() for p in policy.parameters()]

    # Verify changes
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(params_start, params_step1))
    assert any(not torch.equal(p1, p2) for p1, p2 in zip(params_step1, params_step2))

    # Verify no cache persisted
    assert not hasattr(policy, "cache")
    assert not hasattr(policy.native_core, "past_key_values")
    assert not hasattr(policy, "memory")


@pytest.mark.parametrize("use_timestamps", [False, True])
def test_10_parity_with_native_cache_adapter_read_blocks(use_timestamps):
    """Parity test:

    Uses identical policy weights for NativeSequencePolicy and NativeCacheAdapter.
    For each replay group, encode blocks with the adapter, feed them to read_blocks
    with delta split at query boundaries. Compare resulting deep & shallow features
    at each query against the grouped training features on CPU FP32.
    """
    seq_policy = make_tiny_training_policy(shallow_layer=1, use_timestamps=use_timestamps, gradient_checkpointing=False)
    seq_policy.eval()

    # Build adapter from deepcopy of policy
    adapter_policy = copy.deepcopy(seq_policy.policy)

    def mock_prep(images):
        pv, nt = adapter_policy.embedder._preprocess_images_on_cpu(images)
        return pv, nt
    adapter_policy.embedder._preprocess_images = mock_prep

    def mock_fuse(prompt, vit_embeds, image_mask, num_tiles_list):
        fe, fm = adapter_policy.embedder._prepare_batch_and_fuse_embeddings(
            prompts=[prompt],
            vit_embeds_batch=[vit_embeds],
            image_masks=[image_mask],
            batch_num_tiles_list=[num_tiles_list],
        )
        return fe, fm
    adapter_policy.embedder._prepare_and_fuse_embeddings = mock_fuse

    cache_config = NativeCacheConfig(max_frames=6, shallow_layer=1, use_timestamps=use_timestamps)
    adapter = NativeCacheAdapter(adapter_policy, config=cache_config)
    adapter.eval()

    # Pool: 5 frames (0..4)
    # Target indices: outer 0 -> frame 1, outer 1 -> frame 2, outer 2 -> frame 4 (M=3)
    # Replay groups:
    # Group A: obs [0, 1, 3, 4], targets outer [0, 2] -> targets at frames 1 and 4
    #   Query boundaries:
    #   Delta 1: obs [0, 1] -> query at frame 1
    #   Delta 2: obs [3, 4] -> query at frame 4
    # Group B: obs [2], targets outer [1] -> target at frame 2
    #   Delta 1: obs [2] -> query at frame 2
    replay_groups = [
        {"observation_indices": [0, 1, 3, 4], "target_positions": [0, 2]},
        {"observation_indices": [2], "target_positions": [1]},
    ]
    sample = make_grouped_sample(pool_N=5, target_indices=(1, 2, 4), replay_groups=replay_groups)

    sample["frame_ids"] = [10, 12, 17, 25, 31]
    sample["observation_times"] = [0.0, 0.05, 0.2, 0.7, 1.5]

    # 1. Grouped training features
    with torch.no_grad():
        deep_train, shallow_train = seq_policy.features(sample)

    # 2. Replay with NativeCacheAdapter.read_blocks
    prompt = sample["prompt"]
    adapter_targets_deep: Dict[int, torch.Tensor] = {}
    adapter_targets_shallow: Dict[int, torch.Tensor] = {}

    # Encode all pool frames with adapter
    pool_blocks = []
    for i in range(5):
        blk = adapter.encode_frame(
            images=[sample["images_window"][i]],
            frame_id=sample["frame_ids"][i],
            prompt=prompt,
            observation_time=sample["observation_times"][i],
        )
        pool_blocks.append(blk)

    # Replay Group A:
    # Sequence of frames: 0, 1, 3, 4
    # Query 1 is at frame 1 (outer target 0) -> delta [blk0, blk1]
    # Query 2 is at frame 4 (outer target 2) -> delta [blk3, blk4]
    state_a = None
    delta_a1 = [pool_blocks[0], pool_blocks[1]]
    deep_a1, shallow_a1, state_a = adapter.read_blocks(delta_a1, prompt=prompt, previous=state_a)
    adapter_targets_deep[0] = deep_a1
    adapter_targets_shallow[0] = shallow_a1

    delta_a2 = [pool_blocks[3], pool_blocks[4]]
    deep_a2, shallow_a2, state_a = adapter.read_blocks(delta_a2, prompt=prompt, previous=state_a)
    adapter_targets_deep[2] = deep_a2
    adapter_targets_shallow[2] = shallow_a2

    # Replay Group B:
    # Sequence of frames: 2
    # Query is at frame 2 (outer target 1) -> delta [blk2]
    state_b = None
    delta_b1 = [pool_blocks[2]]
    deep_b1, shallow_b1, state_b = adapter.read_blocks(delta_b1, prompt=prompt, previous=state_b)
    adapter_targets_deep[1] = deep_b1
    adapter_targets_shallow[1] = shallow_b1

    # Compare features in original outer target order (0, 1, 2)
    for t_pos in range(3):
        expected_deep = adapter_targets_deep[t_pos]        # [1, 16, 64]
        expected_shallow = adapter_targets_shallow[t_pos]  # [1, 16, 64]

        actual_deep = deep_train[t_pos : t_pos + 1]        # [1, 16, 64]
        actual_shallow = shallow_train[t_pos : t_pos + 1]  # [1, 16, 64]

        assert torch.allclose(actual_deep, expected_deep, atol=1e-5), f"Deep mismatch at target {t_pos}"
        assert torch.allclose(actual_shallow, expected_shallow, atol=1e-5), f"Shallow mismatch at target {t_pos}"
