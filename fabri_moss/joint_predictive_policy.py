"""Joint Predictive Memory Policy and Frozen Visual Teacher for FabriVLA Stage 3.

Implements:
- FrozenVisualTeacher: deepcopy ONLY source_model.vision_model and .mlp1, scalar
  select_layer, downsample_ratio, ps_version. No language_model or action_head copy.
  Extracts features following InternVL pixel_shuffle and mlp1 projection.
  Frozen in FP32, eval mode, encoder.gradient_checkpointing=False, train() no-op.
- JointPredictiveMemoryPolicy: inherits from PredictiveMemoryPolicy, unfreezes
  base policy (ViT, mlp1, LLM, action_head), sets observation grad start to
  cold_prefix_end (compact memory) or 0 (dense), hooks future feature extraction
  to future_teacher, overrides train() to keep teacher in eval mode.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fabri_moss.compact_memory import cold_prefix_end
from fabri_moss.predictive_memory import WriterConfig
from fabri_moss.predictive_policy import PredictiveMemoryPolicy


class FrozenVisualTeacher(nn.Module):
    """Frozen visual teacher extracting pooled feature tokens from pixel values.

    Deepcopies ONLY source_model.vision_model and source_model.mlp1.
    Never references or copies language_model or action_head.
    """

    def __init__(self, source_model: Any) -> None:
        super().__init__()
        if not hasattr(source_model, "vision_model"):
            raise AttributeError("source_model must have a 'vision_model' attribute")
        if not hasattr(source_model, "mlp1"):
            raise AttributeError("source_model must have an 'mlp1' attribute")

        # Deepcopy only vision_model and mlp1
        self.vision_model = copy.deepcopy(source_model.vision_model)
        self.mlp1 = copy.deepcopy(source_model.mlp1)

        # Extract scalar configs
        self.select_layer = int(getattr(source_model, "select_layer", -1))
        self.downsample_ratio = float(getattr(source_model, "downsample_ratio", 0.5))
        self.ps_version = str(getattr(source_model, "ps_version", "v2"))

        if self.ps_version not in ("v1", "v2"):
            raise ValueError(f"Unsupported ps_version '{self.ps_version}', expected 'v1' or 'v2'")

        # Disable gradient checkpointing if present
        encoder = getattr(self.vision_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "gradient_checkpointing"):
            encoder.gradient_checkpointing = False

        # Convert whole module to float32, freeze all parameters, set eval
        self.float()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenVisualTeacher:
        """Always maintain eval mode and frozen state."""
        super().train(False)
        return self

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """Pixel shuffle operation matching InternVL implementation."""
        n, w, h, c = x.size()
        scale = scale_factor
        if scale <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")

        h_scaled = int(h * scale)
        w_scaled = int(w * scale)
        c_scaled = int(c / scale)
        c_final = int(c / (scale * scale))

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h_scaled, c_scaled)
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h_scaled, w_scaled, c_final)

        if self.ps_version != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual features matching InternVL extract_feature.

        Args:
            pixel_values: [B, C, H, W] or model-specific input

        Returns:
            [B, 256, 1024] or [B, P, H] projected token embeddings
        """
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]

        # Remove CLS token
        vit_embeds = vit_embeds[:, 1:, :]

        num_tokens = vit_embeds.shape[1]
        h = int(math.isqrt(num_tokens))
        if h * h != num_tokens:
            raise ValueError(f"Visual tokens count {num_tokens} is not a square")
        w = h

        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


class JointPredictiveMemoryPolicy(PredictiveMemoryPolicy):
    """Joint Predictive Memory Policy unfreezing base backbone for Stage 3 training.

    Maintains frozen future_teacher for target latent extraction while
    allowing full gradient flow through vision model, projector, language model,
    action head, memory writer, and future head.
    """

    def __init__(
        self,
        policy: nn.Module,
        writer_config: Optional[WriterConfig] = None,
        shallow_layer: int = 6,
        gradient_checkpointing: bool = True,
        future_teacher: Optional[nn.Module] = None,
    ) -> None:
        # Determine policy device and dtype
        embedder = getattr(policy, "embedder", None)
        if embedder is not None and hasattr(embedder, "model"):
            source_model = embedder.model
            policy_device = getattr(embedder, "device", next(policy.parameters()).device)
        else:
            source_model = None
            policy_device = next(policy.parameters()).device

        # Create or assign teacher module instance before unfreezing base
        if future_teacher is not None:
            teacher_mod = future_teacher
        else:
            if source_model is None:
                raise ValueError("Cannot create FrozenVisualTeacher without embedder.model on policy")
            teacher_mod = FrozenVisualTeacher(source_model)

        # Freeze teacher in FP32, set eval, match policy device
        teacher_mod = teacher_mod.to(device=policy_device, dtype=torch.float32)
        teacher_mod.requires_grad_(False)
        teacher_mod.eval()

        # Super init sets up writer, future_head, freezes base policy initially
        super().__init__(
            policy=policy,
            writer_config=writer_config,
            shallow_layer=shallow_layer,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Assign future_teacher module attribute
        self.future_teacher = teacher_mod

        # Unfreeze entire base policy for joint stage 3 training
        self.policy.requires_grad_(True)

        # Enable vision gradient checkpointing if requested
        self._set_vision_gradient_checkpointing(gradient_checkpointing)

        # Set training mode to True
        self.train(True)

    def train(self, mode: bool = True) -> JointPredictiveMemoryPolicy:
        """Override train to use standard nn.Module.train without freezing base, while keeping teacher in eval."""
        nn.Module.train(self, mode)
        # Ensure future_teacher strictly remains in eval and requires_grad=False
        if hasattr(self, "future_teacher") and self.future_teacher is not None:
            self.future_teacher.eval()
            self.future_teacher.requires_grad_(False)
        return self

    def _observation_grad_start(self, sample: Dict[str, Any]) -> int:
        """Return frame index where observation gradients start.

        For memory replay: returns cold_prefix_end(decision_indices, first_target, writer_config).
        For dense replay: returns 0 (all observations receive gradients when externally enabled).
        """
        if not sample.get("memory_replay", False):
            return 0

        decision_indices = sample.get("decision_indices", [])
        target_indices = sample.get("target_indices", [])
        if not target_indices:
            return 0
        first_target = target_indices[0]

        from fabri_moss.periodic_memory import PeriodicMemoryConfig

        pm_cfg = PeriodicMemoryConfig(
            recent_frames=self.writer_config.recent_frames,
            consolidate_every=self.writer_config.consolidate_every,
        )
        return cold_prefix_end(
            decision_indices=decision_indices,
            first_target=first_target,
            memory_config=pm_cfg,
        )

    def _extract_future_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual features for future targets using frozen future_teacher."""
        return self.future_teacher(pixel_values)
