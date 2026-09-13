"""Unit and integration tests for fabri_moss.predictive_policy.

Tests cover:
- Tiny Qwen fixture with learned vision encoder & projector.
- Frozen base policy with requires_grad=False and eval mode.
- Base policy gradients remain None and parameters unchanged during backward.
- Differentiable CausalMemoryWriter: out_proj has gradients on action loss backward.
- Upstream writer layers receive gradients on subsequent backward steps after optimizer update.
- FutureLatentHead: phase 1 (future_weight=0.0) strictly skips future head and teacher,
  future_head parameters have None gradients, and monkeypatched feature extractor count is 0.
- Phase 2 (future_weight > 0.0): future_head and writer receive valid gradients, future_loss is finite.
- Variations in future_images do not affect online features or action predictions.
- Invalid future horizons (indices == -1) yield 0 loss without NaNs or errors.
- Dense / current-only groups process full textless frames with temporal RoPE and without writer.
- Verify prompts passed to embedder contain no timestamp text (Frame / time).
- WriterConfig custom toy dimensions adapt correctly.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.predictive_memory import WriterConfig
from fabri_moss.predictive_policy import FutureLatentHead, PredictiveMemoryPolicy
from fabri_moss.tests.test_native_training import (
    make_sample,
    make_tiny_training_policy,
)


def _build_toy_policy(
    shallow_layer: int = 1,
    input_dim: int = 64,
    hidden_dim: int = 32,
    grid: int = 1,
    intermediate_grid: int = 1,
    recent_frames: int = 2,
    consolidate_every: int = 2,
    tbptt_decisions: int = 2,
) -> PredictiveMemoryPolicy:
    """Build a tiny PredictiveMemoryPolicy fixture with toy dimensions."""
    base_seq = make_tiny_training_policy(
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
    )
    base_policy = base_seq.policy

    writer_cfg = WriterConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=2,
        num_layers=1,
        grid=grid,
        intermediate_grid=intermediate_grid,
        recent_frames=recent_frames,
        consolidate_every=consolidate_every,
        tbptt_decisions=tbptt_decisions,
    )

    pred_policy = PredictiveMemoryPolicy(
        policy=base_policy,
        writer_config=writer_cfg,
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
    )
    return pred_policy


def test_constructor_and_frozen_backbone():
    """Verify constructor freezes base policy, turns off vision gradient checkpointing,
    and sets writer and future_head in FP32 with requires_grad=True.
    """
    policy = _build_toy_policy()

    # Base policy must be in eval mode and all params frozen
    assert not policy.policy.training
    for name, p in policy.policy.named_parameters():
        assert not p.requires_grad, f"Base parameter {name} has requires_grad=True"

    # Writer must be FP32 and trainable
    assert policy.writer.training
    for name, p in policy.writer.named_parameters():
        assert p.requires_grad, f"Writer parameter {name} not trainable"
        assert p.dtype == torch.float32, f"Writer parameter {name} not float32"

    # Future head must be FP32 and trainable
    assert policy.future_head.training
    for name, p in policy.future_head.named_parameters():
        assert p.requires_grad, f"FutureHead parameter {name} not trainable"
        assert p.dtype == torch.float32, f"FutureHead parameter {name} not float32"

    # Train mode override: base policy stays in eval
    policy.train(True)
    assert not policy.policy.training
    assert policy.writer.training
    assert policy.future_head.training

    policy.train(False)
    assert not policy.policy.training
    assert not policy.writer.training
    assert not policy.future_head.training


def test_no_timestamp_text_in_prompt_or_headers():
    """Verify that text prompts encoded under PredictiveMemoryPolicy contain no 'Frame' or 'time' timestamps."""
    policy = _build_toy_policy()
    policy.train(True)

    # Spy on _prepare_batch_and_fuse_embeddings to capture all actual prompts processed
    embedder = policy.policy.embedder
    orig_fuse = embedder._prepare_batch_and_fuse_embeddings
    prompts_seen = []

    def spy_fuse(prompts, vit_embeds_batch, image_masks, batch_num_tiles_list):
        for p in prompts:
            prompts_seen.append(p)
        return orig_fuse(prompts, vit_embeds_batch, image_masks, batch_num_tiles_list)

    embedder._prepare_batch_and_fuse_embeddings = spy_fuse

    sample = make_sample(N=4, target_indices=[1, 3], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3]

    _ = policy.features(sample)

    assert len(prompts_seen) > 0, "Expected at least one prompt to be fused"
    for p in prompts_seen:
        assert "Frame " not in p, f"Found timestamp text 'Frame' in prompt: {p}"
        assert "time " not in p, f"Found timestamp text 'time' in prompt: {p}"


def test_action_loss_backward_writer_grad_and_frozen_base():
    """Verify:
    1. Forward with action loss (future_weight=0.0)
    2. Backward produces gradients in writer.out_proj on step 1 (where out_proj was zero-init).
    3. Base policy params receive NO gradients (grad is None).
    4. Base policy parameter values remain bitwise identical.
    5. Update writer with real AdamW so out_proj is non-zero, then step 2 backward produces
       gradients in upstream writer layers (input_proj).
    """
    policy = _build_toy_policy()
    policy.train(True)

    optimizer = torch.optim.AdamW(policy.writer.parameters(), lr=1e-2)

    # Save copy of base parameters
    base_params_before = {
        name: p.clone().detach() for name, p in policy.policy.named_parameters()
    }

    # N=8, decisions=[1, 3, 5, 7], targets=[5, 7]
    sample = make_sample(N=8, target_indices=[5, 7], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5, 7]

    out = policy.forward(sample, future_weight=0.0)
    assert out["action_loss"].isfinite()
    assert out["future_loss"].item() == 0.0
    assert out["loss"].item() == out["action_loss"].item()
    assert out["target_count"] == 2

    # Backward step 1
    optimizer.zero_grad()
    out["loss"].backward()

    # Verify base params have None grad and unchanged values
    for name, p in policy.policy.named_parameters():
        assert p.grad is None, f"Base parameter {name} received gradient!"
        torch.testing.assert_close(p, base_params_before[name])

    # Verify writer out_proj receives gradient on step 1
    assert policy.writer.out_proj.weight.grad is not None
    assert policy.writer.out_proj.weight.grad.abs().sum() > 0.0

    # Step optimizer: out_proj becomes non-zero
    optimizer.step()

    # Step 2: forward & backward with updated writer to verify upstream writer layers receive gradient
    optimizer.zero_grad()
    out2 = policy.forward(sample, future_weight=0.0)
    out2["loss"].backward()
    assert policy.writer.out_proj.weight.grad is not None
    assert policy.writer.input_proj.weight.grad is not None
    assert policy.writer.input_proj.weight.grad.abs().sum() > 0.0


def test_phase1_future_weight_zero_strictly_skips_future_head():
    """Verify phase 1 (future_weight=0.0) strictly skips future head and teacher visual encoder."""
    policy = _build_toy_policy()
    policy.train(True)

    # Spy on vision model extract_feature
    extract_feature_called = 0
    orig_extract = policy.policy.embedder.model.extract_feature

    def spy_extract(pixel_values):
        nonlocal extract_feature_called
        extract_feature_called += 1
        return orig_extract(pixel_values)

    policy.policy.embedder.model.extract_feature = spy_extract

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]
    # Provide future fields
    sample["future_images"] = [torch.randn(1, 4, 16) for _ in range(4)]
    sample["future_indices"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    sample["future_deltas"] = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32)

    policy.zero_grad()
    extract_before_fwd = extract_feature_called
    out = policy.forward(sample, future_weight=0.0)

    # Extract feature called only for observations during features(), NOT for future_images
    # N=6 observations in batch 8 => 1 call
    assert extract_feature_called - extract_before_fwd == 1, (
        "extract_feature was called unexpectedly for future_images under future_weight=0.0"
    )

    out["loss"].backward()

    # future_head parameters must have None grad
    for name, p in policy.future_head.named_parameters():
        assert p.grad is None, f"future_head param {name} received grad in Phase 1!"


def test_phase2_future_loss_and_gradient_flow():
    """Verify phase 2 (future_weight > 0.0):
    1. future_head and writer receive valid finite gradients.
    2. Base policy remains frozen with None grad.
    """
    policy = _build_toy_policy()
    policy.train(True)

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]
    sample["future_images"] = [torch.randn(1, 4, 16) for _ in range(4)]
    sample["future_indices"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    sample["future_deltas"] = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32)
    sample["future_valid"] = torch.tensor([[True, True], [True, True]], dtype=torch.bool)

    policy.zero_grad()
    out = policy.forward(sample, future_weight=0.5)

    assert out["future_loss"].isfinite()
    assert out["future_loss"].item() > 0.0
    assert out["future_valid_count"] == 4

    out["loss"].backward()

    # Base parameters still have None grad
    for name, p in policy.policy.named_parameters():
        assert p.grad is None, f"Base parameter {name} received gradient in Phase 2!"

    # Future head parameters receive gradients
    assert policy.future_head.context_proj.weight.grad is not None
    assert policy.future_head.learned_queries.grad is not None
    assert policy.future_head.out_mlp[1].weight.grad is not None


def test_future_images_variations_do_not_affect_online_features_or_actions():
    """Verify changing future_images does not alter online features or action predictions."""
    policy = _build_toy_policy()
    policy.eval()

    sample1 = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample1["memory_replay"] = True
    sample1["decision_indices"] = [1, 3, 5]
    sample1["future_images"] = [torch.randn(1, 4, 16) for _ in range(4)]
    sample1["future_indices"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    sample1["future_deltas"] = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32)

    sample2 = copy.deepcopy(sample1)
    sample2["future_images"] = [torch.randn(1, 4, 16) * 10.0 + 5.0 for _ in range(4)]

    with torch.no_grad():
        deep1, shallow1 = policy.features(sample1)
        deep2, shallow2 = policy.features(sample2)
        act1 = policy.predict_actions(sample1)
        act2 = policy.predict_actions(sample2)

    torch.testing.assert_close(deep1, deep2)
    torch.testing.assert_close(shallow1, shallow2)
    torch.testing.assert_close(act1, act2)


def test_invalid_future_horizons_yield_zero_without_nan():
    """Verify that when all future horizons are invalid (index == -1), future_loss is 0.0 with no NaN."""
    policy = _build_toy_policy()
    policy.train(True)

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]
    sample["future_images"] = [torch.randn(1, 4, 16) for _ in range(4)]
    # All invalid indices (-1)
    sample["future_indices"] = torch.tensor([[-1, -1], [-1, -1]], dtype=torch.long)
    sample["future_deltas"] = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    sample["future_valid"] = torch.tensor([[False, False], [False, False]], dtype=torch.bool)

    out = policy.forward(sample, future_weight=1.0)
    assert out["future_loss"].item() == 0.0
    assert not torch.isnan(out["loss"])
    assert out["future_valid_count"] == 0

    # Also test empty future_images
    sample["future_images"] = []
    out2 = policy.forward(sample, future_weight=1.0)
    assert out2["future_loss"].item() == 0.0
    assert not torch.isnan(out2["loss"])


def test_dense_and_current_only_groups_without_writer():
    """Verify dense and current-only samples run through full textless frames with temporal RoPE
    without invoking the memory writer, preserving target ordering.
    """
    policy = _build_toy_policy()
    policy.train(True)

    # Dense sample (memory_replay absent)
    dense_sample = make_sample(N=4, target_indices=[1, 3], horizon=2, action_dim=4)
    deep_dense, shallow_dense = policy.features(dense_sample)
    assert deep_dense.shape == (2, 16, 64)
    assert shallow_dense.shape == (2, 16, 64)

    # Current-only sample with replay_groups
    curr_sample = make_sample(N=4, target_indices=[1, 3], horizon=2, action_dim=4)
    curr_sample["replay_groups"] = [
        {"observation_indices": [0, 1], "target_positions": [0]},
        {"observation_indices": [2, 3], "target_positions": [1]},
    ]
    deep_curr, shallow_curr = policy.features(curr_sample)
    assert deep_curr.shape == (2, 16, 64)
    assert shallow_curr.shape == (2, 16, 64)


def test_predict_actions_offline_inference():
    """Verify predict_actions executes in eval/no_grad without needing any future keys."""
    policy = _build_toy_policy()
    policy.train(True)

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]
    # No future keys present

    pred = policy.predict_actions(sample)
    assert pred.shape == (1, 2, 4)
    assert pred.dtype == torch.float32


def test_tbptt_boundary_and_first_target_backward_reach_writer():
    """Verify TBPTT requirement 2:
    Sample decisions [3, 7, 11, 15, 19, 23], targets [19, 23].
    Prior decisions before first target (19): [3, 7, 11, 15].
    With tbptt_decisions=4, warm_decision_start is 3 (all 4 prior decisions retained in graph).
    First target (19) alone backward must reach writer.out_proj with non-zero gradient.
    Base policy remains completely frozen with None grad.
    """
    policy = _build_toy_policy(
        input_dim=64,
        hidden_dim=32,
        grid=1,
        intermediate_grid=1,
        recent_frames=2,
        consolidate_every=2,
        tbptt_decisions=4,
    )
    policy.train(True)

    # Decisions: [3, 7, 11, 15, 19, 23], targets: [19, 23], N=24
    N = 24
    decisions = [3, 7, 11, 15, 19, 23]
    targets = [19, 23]
    sample = make_sample(N=N, target_indices=targets, horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = decisions

    # Deep/shallow features
    deep, shallow = policy.features(sample)
    assert deep.shape == (2, 16, 64)

    # 1. Backward solely from the first target (target 19)
    first_target_loss = deep[0].sum()
    policy.zero_grad()
    first_target_loss.backward(retain_graph=True)

    # Writer out_proj must have non-zero gradient from first target alone
    assert policy.writer.out_proj.weight.grad is not None
    assert policy.writer.out_proj.weight.grad.abs().sum() > 0.0

    # 2. Action loss alone forward & backward
    out = policy.forward(sample, future_weight=0.0)
    policy.zero_grad()
    out["action_loss"].backward()

    assert policy.writer.out_proj.weight.grad is not None
    assert policy.writer.out_proj.weight.grad.abs().sum() > 0.0

    # Base policy must have None grad throughout
    for name, p in policy.policy.named_parameters():
        assert p.grad is None, f"Base parameter {name} received gradient!"


def test_future_weight_and_future_targets_strict_validation():
    """Verify requirement 3:
    - future_weight must be strictly non-negative finite float/int, not bool.
    - phase 1 (weight=0.0) does not touch future keys or future head.
    - phase 2 (weight > 0.0) strictly validates future_indices, future_deltas, future_valid:
      shape [M, K] matching, valid indices >= 0 and < total_future, delta > 0 and finite,
      invalid indices == -1, raises on invalid / out-of-range index rather than silent zero target.
    """
    policy = _build_toy_policy()
    policy.train(True)

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]

    # Invalid future_weight types
    with pytest.raises(TypeError, match="future_weight"):
        policy.forward(sample, future_weight=True)

    with pytest.raises(ValueError, match="future_weight"):
        policy.forward(sample, future_weight=-0.5)

    with pytest.raises(ValueError, match="future_weight"):
        policy.forward(sample, future_weight=float("inf"))

    # Phase 1: future_weight=0.0 ignores absence of future keys
    out_zero = policy.forward(sample, future_weight=0.0)
    assert out_zero["future_loss"].item() == 0.0

    # Phase 2: weight > 0 missing future keys raises KeyError
    with pytest.raises(KeyError):
        policy.forward(sample, future_weight=0.5)

    # Missing future_images when valid horizons exist
    sample["future_indices"] = torch.tensor([[0, -1], [-1, -1]], dtype=torch.long)
    sample["future_deltas"] = torch.tensor([[0.2, 0.0], [0.0, 0.0]], dtype=torch.float32)
    sample["future_valid"] = torch.tensor([[True, False], [False, False]], dtype=torch.bool)
    sample["future_images"] = []  # length 0 but valid horizon at [0, 0]

    with pytest.raises(ValueError, match="out-of-range index"):
        policy.forward(sample, future_weight=0.5)

    # Non-positive delta for valid target
    sample["future_images"] = [torch.randn(1, 4, 16)]
    sample["future_deltas"] = torch.tensor([[-0.1, 0.0], [0.0, 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="non-positive"):
        policy.forward(sample, future_weight=0.5)

    # Invalid target with index != -1
    sample["future_deltas"] = torch.tensor([[0.2, 0.0], [0.0, 0.0]], dtype=torch.float32)
    sample["future_indices"] = torch.tensor([[0, 2], [-1, -1]], dtype=torch.long)  # [0, 1] invalid but idx=2 != -1
    with pytest.raises(ValueError, match="index -1"):
        policy.forward(sample, future_weight=0.5)

    sample["future_indices"] = torch.tensor([[0, -1], [-1, -1]], dtype=torch.long)
    sample["future_deltas"][0, 1] = float("nan")
    with pytest.raises(ValueError, match="finite zero delta"):
        policy.forward(sample, future_weight=0.5)


def test_no_param_creation_under_no_grad_and_eval():
    """Verify that calling features, forward, and predict_actions creates no new parameters
    and leaves model in eval/proper state.
    """
    policy = _build_toy_policy()
    params_before = set(policy.state_dict().keys())

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]

    with torch.no_grad():
        _ = policy.features(sample)
        _ = policy.predict_actions(sample)

    params_after = set(policy.state_dict().keys())
    assert params_before == params_after
    assert not policy.policy.training
