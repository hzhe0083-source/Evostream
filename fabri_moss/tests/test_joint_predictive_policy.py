"""Unit and integration tests for fabri_moss.joint_predictive_policy.

Tests cover:
- FrozenVisualTeacher equality test using small InternVL-style source fixture:
  exact shuffle parity between InternVL extract_feature and FrozenVisualTeacher forward
  under ps_version v1 and v2, and select_layer options (-1, 0).
- FrozenVisualTeacher deepcopy bounds: verify teacher has no language_model or action_head attributes,
  parameter count is strictly vision + mlp1, params are FP32, eval mode, frozen (requires_grad=False).
- Tiny Qwen fixture integration with TinyTeacher injection:
  JointPredictiveMemoryPolicy unfreezes base backbone (ViT, projector/mlp1, LLM, action_head),
  enables training mode for all modules except teacher which remains strictly eval & requires_grad=False.
- Full gradient flow test across all 6 groups:
  nonzero gradients and actual parameter updates for vision, projector, LLM, action_head, writer, future_head.
- Memory and dense replay observation gradient start:
  Memory replay sets grad_start = cold_prefix_end(decision_indices, first_target).
  Cold prefix frames ViT batch has no_grad, warm frames receive gradients.
  Dense replay sets grad_start = 0 (all frames receive gradients).
- Evaluation and torch.no_grad() inference:
  Evaluation / predict_actions does not create computational graph or mutate parameters.
- Numerical outputs parity:
  Forward identical features/loss between Joint and Predictive policy when teacher matches.
- Perturbing student vision weights does NOT alter teacher targets (teacher frozen at initial weights).
- Robust error handling: unsupported ps_version, non-square token count.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.compact_memory import cold_prefix_end
from fabri_moss.joint_predictive_policy import FrozenVisualTeacher, JointPredictiveMemoryPolicy
from fabri_moss.periodic_memory import PeriodicMemoryConfig
from fabri_moss.predictive_memory import WriterConfig
from fabri_moss.predictive_policy import PredictiveMemoryPolicy
from fabri_moss.tests.test_native_training import (
    LearnedFakeViT,
    LearnedProjector,
    TinyModelWithVision,
    make_sample,
    make_tiny_training_policy,
)


# ---------------------------------------------------------------------------
# InternVL-style Fixture for FrozenVisualTeacher Parity Testing
# ---------------------------------------------------------------------------


class InternVLStyleVisionModel(nn.Module):
    """Small mock ViT module returning last_hidden_state and hidden_states."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Linear(in_dim, hidden_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Any:
        # pixel_values: [B, num_patches, in_dim]
        # output tokens: 1 CLS token + num_patches tokens = 1 + 16 = 17 tokens
        B = pixel_values.shape[0]
        h = self.conv(pixel_values)  # [B, 16, hidden_dim]
        cls_tok = torch.zeros(B, 1, h.shape[-1], device=pixel_values.device, dtype=pixel_values.dtype)
        full_seq = torch.cat([cls_tok, h], dim=1)  # [B, 17, hidden_dim]

        class Output:
            pass

        out = Output()
        out.last_hidden_state = full_seq
        if output_hidden_states:
            # Provide two layers: layer 0 (scaled) and layer 1
            out.hidden_states = [full_seq * 0.5, full_seq]
        return out


class InternVLStyleSourceModel(nn.Module):
    """InternVL-like model source containing vision_model, mlp1, language_model, etc."""

    def __init__(
        self,
        vit_dim: int = 32,
        llm_dim: int = 64,
        select_layer: int = -1,
        downsample_ratio: float = 0.5,
        ps_version: str = "v2",
    ):
        super().__init__()
        self.vision_model = InternVLStyleVisionModel(in_dim=16, hidden_dim=vit_dim)
        # downsample_ratio=0.5 reshapes [B, 4, 4, C] -> [B, 2, 2, C*4]
        in_mlp = int(vit_dim / (downsample_ratio * downsample_ratio))
        self.mlp1 = nn.Sequential(
            nn.Linear(in_mlp, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.language_model = nn.Linear(llm_dim, llm_dim)  # Dummy LM
        self.select_layer = select_layer
        self.downsample_ratio = downsample_ratio
        self.ps_version = ps_version

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        scale = scale_factor
        x = x.view(n, w, int(h * scale), int(c / scale))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale), int(w * scale), int(c / (scale * scale)))
        if self.ps_version != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


# ---------------------------------------------------------------------------
# Tests for FrozenVisualTeacher
# ---------------------------------------------------------------------------


def test_frozen_visual_teacher_internvl_parity_and_no_lm_copy():
    """Verify:
    1. FrozenVisualTeacher copies only vision_model and mlp1, NOT language_model.
    2. Forward exactly matches InternVL extract_feature under ps_version v1 and v2, and select_layer.
    3. All teacher parameters are FP32, eval mode, and frozen (requires_grad=False).
    """
    for ps_ver in ("v1", "v2"):
        for sel_layer in (-1, 0):
            source = InternVLStyleSourceModel(
                vit_dim=16,
                llm_dim=32,
                select_layer=sel_layer,
                downsample_ratio=0.5,
                ps_version=ps_ver,
            )

            teacher = FrozenVisualTeacher(source)

            # Check no language_model copied
            assert hasattr(source, "language_model")
            assert not hasattr(teacher, "language_model")

            # Check params count and requires_grad
            teacher_params = list(teacher.parameters())
            source_vision_mlp_params = list(source.vision_model.parameters()) + list(source.mlp1.parameters())
            assert len(teacher_params) == len(source_vision_mlp_params)
            for p in teacher_params:
                assert not p.requires_grad
                assert p.dtype == torch.float32

            assert not teacher.training
            teacher.train(True)
            assert not teacher.training  # train override maintains False

            # Parity check
            pixel_values = torch.randn(2, 16, 16)
            with torch.no_grad():
                expected = source.extract_feature(pixel_values)
                actual = teacher(pixel_values)

            torch.testing.assert_close(actual, expected)


def test_frozen_visual_teacher_error_cases():
    """Verify FrozenVisualTeacher raises clear errors on invalid inputs."""
    # Unsupported ps_version
    source = InternVLStyleSourceModel(ps_version="v3")
    with pytest.raises(ValueError, match="Unsupported ps_version"):
        FrozenVisualTeacher(source)

    # Missing attributes
    dummy = nn.Module()
    with pytest.raises(AttributeError, match="vision_model"):
        FrozenVisualTeacher(dummy)


# ---------------------------------------------------------------------------
# Helpers for Joint Policy Tiny Qwen Fixture
# ---------------------------------------------------------------------------


class TinyTeacher(nn.Module):
    """Test-only tiny teacher wrapping a frozen copy of vision and projector without LM."""

    def __init__(self, vision_model: nn.Module, projector: nn.Module):
        super().__init__()
        self.vision_model = copy.deepcopy(vision_model)
        self.projector = copy.deepcopy(projector)
        self.float()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> TinyTeacher:
        super().train(False)
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.vision_model.encoder(pixel_values)
            return self.projector(feat)


def _build_toy_joint_policy(
    shallow_layer: int = 1,
    input_dim: int = 64,
    hidden_dim: int = 32,
    grid: int = 1,
    intermediate_grid: int = 1,
    recent_frames: int = 2,
    consolidate_every: int = 2,
    tbptt_decisions: int = 2,
) -> JointPredictiveMemoryPolicy:
    """Build a tiny JointPredictiveMemoryPolicy fixture with injected TinyTeacher."""
    base_seq = make_tiny_training_policy(
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
    )
    base_policy = base_seq.policy

    # Normalize native tiny fixture: alias projector as mlp1 if needed, or inject TinyTeacher
    embedder_model = base_policy.embedder.model
    if not hasattr(embedder_model, "mlp1"):
        embedder_model.mlp1 = embedder_model.projector

    teacher = TinyTeacher(embedder_model.vision_model, embedder_model.projector)

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

    joint_policy = JointPredictiveMemoryPolicy(
        policy=base_policy,
        writer_config=writer_cfg,
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
        future_teacher=teacher,
    )
    return joint_policy


# ---------------------------------------------------------------------------
# Joint Policy Tests
# ---------------------------------------------------------------------------


def test_joint_policy_unfrozen_backbone_and_eval_teacher():
    """Verify:
    1. JointPredictiveMemoryPolicy unfreezes all base policy parameters (requires_grad=True).
    2. Future teacher strictly remains in eval and requires_grad=False.
    3. train(True) / train(False) maintains teacher in eval mode.
    """
    policy = _build_toy_joint_policy()

    # Policy in train mode
    assert policy.training
    assert policy.policy.training

    # Check base policy parameters are trainable
    for name, p in policy.policy.named_parameters():
        assert p.requires_grad, f"Base param {name} should have requires_grad=True"

    # Check writer and future_head trainable
    for name, p in policy.writer.named_parameters():
        assert p.requires_grad, f"Writer param {name} should have requires_grad=True"
    for name, p in policy.future_head.named_parameters():
        assert p.requires_grad, f"FutureHead param {name} should have requires_grad=True"

    # Check future_teacher is frozen and eval
    assert not policy.future_teacher.training
    for name, p in policy.future_teacher.named_parameters():
        assert not p.requires_grad, f"Teacher param {name} should have requires_grad=False"
        assert p.dtype == torch.float32

    # Switch modes
    policy.train(False)
    assert not policy.training
    assert not policy.future_teacher.training

    policy.train(True)
    assert policy.training
    assert not policy.future_teacher.training


def test_joint_policy_full_gradient_flow_and_parameter_updates():
    """Verify all 6 module groups receive non-zero gradients and are updated by AdamW:
    1. Vision encoder (vision_model)
    2. Projector (mlp1)
    3. Language model (LLM)
    4. Action head
    5. CausalMemoryWriter
    6. FutureLatentHead
    Also verify future_teacher parameters have NO gradients and do not change.
    """
    policy = _build_toy_joint_policy()
    policy.train(True)

    # 6 groups optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.policy.embedder.model.vision_model.parameters(), "lr": 1e-3},
            {"params": policy.policy.embedder.model.projector.parameters(), "lr": 1e-3},
            {"params": policy.policy.embedder.model.language_model.parameters(), "lr": 1e-3},
            {"params": policy.policy.action_head.parameters(), "lr": 1e-3},
            {"params": policy.writer.parameters(), "lr": 1e-3},
            {"params": policy.future_head.parameters(), "lr": 1e-3},
        ]
    )

    # Snapshot parameter values before update
    teacher_weights_before = [p.clone().detach() for p in policy.future_teacher.parameters()]
    vit_weight_before = next(policy.policy.embedder.model.vision_model.parameters()).clone().detach()
    proj_weight_before = next(policy.policy.embedder.model.projector.parameters()).clone().detach()
    llm_weight_before = next(policy.policy.embedder.model.language_model.parameters()).clone().detach()
    action_weight_before = next(policy.policy.action_head.parameters()).clone().detach()
    writer_weight_before = policy.writer.out_proj.weight.clone().detach()
    futhead_weight_before = next(policy.future_head.parameters()).clone().detach()

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]
    sample["future_images"] = [torch.randn(1, 4, 16) for _ in range(4)]
    sample["future_indices"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    sample["future_deltas"] = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32)
    sample["future_valid"] = torch.tensor([[True, True], [True, True]], dtype=torch.bool)

    optimizer.zero_grad()
    out = policy.forward(sample, future_weight=0.5)
    assert out["action_loss"].isfinite()
    assert out["future_loss"].isfinite()

    out["loss"].backward()

    # 1. Check gradients on all 6 groups
    assert next(policy.policy.embedder.model.vision_model.parameters()).grad is not None
    assert next(policy.policy.embedder.model.vision_model.parameters()).grad.abs().sum() > 0.0

    assert next(policy.policy.embedder.model.projector.parameters()).grad is not None
    assert next(policy.policy.embedder.model.projector.parameters()).grad.abs().sum() > 0.0

    assert next(policy.policy.embedder.model.language_model.parameters()).grad is not None
    assert next(policy.policy.embedder.model.language_model.parameters()).grad.abs().sum() > 0.0

    assert next(policy.policy.action_head.parameters()).grad is not None
    assert next(policy.policy.action_head.parameters()).grad.abs().sum() > 0.0

    assert policy.writer.out_proj.weight.grad is not None
    assert policy.writer.out_proj.weight.grad.abs().sum() > 0.0

    assert policy.future_head.context_proj.weight.grad is not None
    assert policy.future_head.context_proj.weight.grad.abs().sum() > 0.0

    # 2. Check future_teacher has NO gradients
    for p in policy.future_teacher.parameters():
        assert p.grad is None

    # Step optimizer
    optimizer.step()

    # 3. Check all 6 groups received actual parameter updates
    assert not torch.equal(next(policy.policy.embedder.model.vision_model.parameters()), vit_weight_before)
    assert not torch.equal(next(policy.policy.embedder.model.projector.parameters()), proj_weight_before)
    assert not torch.equal(next(policy.policy.embedder.model.language_model.parameters()), llm_weight_before)
    assert not torch.equal(next(policy.policy.action_head.parameters()), action_weight_before)
    assert not torch.equal(policy.writer.out_proj.weight, writer_weight_before)
    assert not torch.equal(next(policy.future_head.parameters()), futhead_weight_before)

    # 4. Check future_teacher parameters are completely unchanged
    for p, p_before in zip(policy.future_teacher.parameters(), teacher_weights_before):
        torch.testing.assert_close(p, p_before)


def test_warm_cold_scope_proof_memory_and_dense():
    """Verify observation grad hook behavior:
    1. In memory replay, cold prefix frames (before cold_prefix_end) are computed under no_grad,
       while warm frames (>= cold_prefix_end) retain gradients.
    2. In dense replay, grad_start is 0, so all frames retain gradients.
    """
    policy = _build_toy_joint_policy(recent_frames=2, consolidate_every=2)
    policy.train(True)

    # Sample: N=8, decisions=[1, 3, 5, 7], targets=[5, 7]
    # cold_prefix_end: prior decisions before first_target (5) are [1, 3], last prior d=3.
    # cold_prefix_end = max(0, ((3 + 1 - 2) // 2) * 2) = 2.
    # Therefore frames 0, 1 are cold, frames 2..7 are warm.
    sample = make_sample(N=8, target_indices=[5, 7], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5, 7]

    grad_start = policy._observation_grad_start(sample)
    assert grad_start == 2

    # Dense sample: grad_start must be 0
    dense_sample = make_sample(N=8, target_indices=[5, 7], horizon=2, action_dim=4)
    dense_sample["memory_replay"] = False
    assert policy._observation_grad_start(dense_sample) == 0

    # Spy on _encode_sequence_embeddings to verify cold frames called under no_grad
    # and warm frames called under grad_enabled
    orig_encode = policy._encode_sequence_embeddings
    grad_states_during_encode = []

    def spy_encode(*args, **kwargs):
        grad_states_during_encode.append(torch.is_grad_enabled())
        return orig_encode(*args, **kwargs)

    policy._encode_sequence_embeddings = spy_encode
    _ = policy.features(sample)

    # First chunk (frames 0..1, size 2) must be called with grad disabled (False)
    # Second chunk (frames 2..7, size 6) must be called with grad enabled (True)
    assert len(grad_states_during_encode) == 2
    assert grad_states_during_encode[0] is False, "Cold prefix was not called under no_grad!"
    assert grad_states_during_encode[1] is True, "Warm frames were not called with grad enabled!"


def test_compare_phase1_default_numerical_outputs():
    """Verify default old frozen policy behavior and joint policy produce identical outputs
    when given identical weights and future_weight=0.0.
    """
    base_seq = make_tiny_training_policy(
        shallow_layer=1,
        gradient_checkpointing=False,
    )
    base_policy = base_seq.policy

    writer_cfg = WriterConfig(
        input_dim=64,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        grid=1,
        intermediate_grid=1,
        recent_frames=2,
        consolidate_every=2,
        tbptt_decisions=2,
    )

    pred_policy = PredictiveMemoryPolicy(
        policy=base_policy,
        writer_config=writer_cfg,
        shallow_layer=1,
        gradient_checkpointing=False,
    )

    teacher = TinyTeacher(
        base_policy.embedder.model.vision_model,
        base_policy.embedder.model.projector,
    )

    joint_policy = JointPredictiveMemoryPolicy(
        policy=copy.deepcopy(base_policy),
        writer_config=writer_cfg,
        shallow_layer=1,
        gradient_checkpointing=False,
        future_teacher=teacher,
    )
    # Copy writer state to ensure bitwise match
    joint_policy.writer.load_state_dict(pred_policy.writer.state_dict())
    joint_policy.future_head.load_state_dict(pred_policy.future_head.state_dict())

    pred_policy.eval()
    joint_policy.eval()

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]

    with torch.no_grad():
        out_pred = pred_policy.forward(sample, future_weight=0.0)
        out_joint = joint_policy.forward(sample, future_weight=0.0)

    torch.testing.assert_close(out_pred["loss"], out_joint["loss"])
    torch.testing.assert_close(out_pred["action_loss"], out_joint["action_loss"])


def test_perturb_student_vision_does_not_alter_teacher_targets():
    """Verify that perturbing student vision model weights does NOT alter future targets
    extracted by the frozen teacher.
    """
    policy = _build_toy_joint_policy()
    policy.eval()

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["future_images"] = [torch.randn(1, 4, 16) for _ in range(4)]
    sample["future_indices"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    sample["future_deltas"] = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32)
    sample["future_valid"] = torch.tensor([[True, True], [True, True]], dtype=torch.bool)

    # Extract targets before perturbation
    targets_before = policy._future_targets(
        sample, sample["future_indices"], sample["future_valid"]
    )

    # Perturb student vision model weights drastically
    with torch.no_grad():
        for p in policy.policy.embedder.model.vision_model.parameters():
            p.add_(torch.randn_like(p) * 10.0)

    # Extract targets after perturbation
    targets_after = policy._future_targets(
        sample, sample["future_indices"], sample["future_valid"]
    )

    # Must be bitwise identical
    torch.testing.assert_close(targets_before, targets_after)


def test_no_grad_inference_creates_no_graph_or_params():
    """Verify inference in eval/no_grad does not create computational graph or mutate parameters."""
    policy = _build_toy_joint_policy()
    policy.eval()

    sample = make_sample(N=6, target_indices=[3, 5], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3, 5]

    params_before = {k: v.clone() for k, v in policy.state_dict().items()}

    with torch.no_grad():
        pred = policy.predict_actions(sample)
        deep, shallow = policy.features(sample)

    assert pred.shape == (1, 2, 4)
    assert not pred.requires_grad
    assert not deep.requires_grad
    assert not shallow.requires_grad

    params_after = policy.state_dict()
    for k in params_before:
        torch.testing.assert_close(params_before[k], params_after[k])
