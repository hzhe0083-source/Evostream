"""Compact Memory Policy Offline and Closed-Loop Inference Loader and Policy Wrapper.

Provides:
- CompactInferencePolicy: inherits from NativeCompactMemorySequencePolicy,
  predicts actions using full-prefix compact memory replay without learned writer,
  using native text timestamps and TemporalRoPE.
- load_compact_inference: strictly loads native_compact_memory_replay_v1 checkpoint
  into a frozen CompactInferencePolicy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fabri_moss.compact_memory import CompactMemoryConfig
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.compact_training import NativeCompactMemorySequencePolicy
from fabri_moss.native_training import _autocast_context
from fabri_moss.predictive_inference import snapshot_checkpoint
from fabri_moss.runtime import (
    assert_native_fa2,
    compute_file_sha256,
    load_native_checkpoint,
)


class CompactInferencePolicy(NativeCompactMemorySequencePolicy):
    """Inference policy using native compact memory replay, temporal RoPE, and native action head sampling.

    Inherits from NativeCompactMemorySequencePolicy. Adds no extra modules or parameters.
    Requires memory_replay=True on sample to avoid quiet fallback to dense replay.
    """

    def __init__(
        self,
        policy: nn.Module,
        shallow_layer: int = 6,
        use_timestamps: bool = True,
        gradient_checkpointing: bool = False,
        compact_config: Optional[CompactMemoryConfig] = None,
    ) -> None:
        super().__init__(
            policy=policy,
            shallow_layer=shallow_layer,
            use_timestamps=use_timestamps,
            gradient_checkpointing=gradient_checkpointing,
            compact_config=compact_config,
        )

    @torch.no_grad()
    def predict_actions(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Offline and closed-loop evaluation action prediction.

        Strictly requires memory_replay=True to avoid quiet fallback to dense replay.
        Computes deep/shallow features using native compact memory replay,
        slices the latest target frame, and queries action_head.sample() under float32.
        """
        if not sample.get("memory_replay", False):
            raise ValueError(
                "memory_replay must be True for CompactInferencePolicy.predict_actions to avoid quiet dense fallback"
            )

        deep, shallow = self.features(sample)

        # Slice latest target frame: [1, tokens_per_frame, hidden_dim]
        last_deep = deep[-1:]
        last_shallow = shallow[-1:] if shallow is not None else None

        state = sample.get("state", None)
        state_mask = sample.get("state_mask", None)
        action_mask = sample.get("action_mask", None)

        last_state = state[-1:] if state is not None else None
        last_state_mask = state_mask[-1:] if state_mask is not None else None
        last_action_mask = action_mask[-1:] if action_mask is not None else None

        ah_config = getattr(getattr(self.policy, "action_head", None), "config", None)
        shallow_fusion = getattr(ah_config, "shallow_fusion", "none")
        pass_shallow = last_shallow if shallow_fusion != "none" else None

        kwargs: Dict[str, Any] = {
            "state": last_state,
            "state_mask": last_state_mask,
            "action_mask": last_action_mask,
            "shallow_tokens": pass_shallow,
        }

        with _autocast_context(last_deep.device, enabled=False):
            pred_actions = self.policy.action_head.sample(last_deep, **kwargs)

        return pred_actions


def load_compact_inference(
    checkpoint_path: str | Path,
    *,
    snapshot_dir: str | Path,
    fabri_root: str | Path = "/root/FabriVLA",
    vlm_path: str | Path = "/root/models/InternVL3_5-1B",
    device: str = "cpu",
    arm_key: str = "metaworld_sawyer",
    expected_sha256: Optional[str] = None,
) -> Tuple[CompactInferencePolicy, Dict[str, Any], Dict[str, Any]]:
    """Strictly load a native compact memory checkpoint for evaluation inference."""
    ckpt_src = Path(checkpoint_path).resolve()
    snap_root = Path(snapshot_dir).resolve()
    snap_root.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot checkpoint into snapshot_dir / "checkpoint.pt"
    snap_ckpt_path = snap_root / "checkpoint.pt"
    snapshot_checkpoint(ckpt_src, snap_ckpt_path)

    # Compute sha256 strictly on the snapshot
    ckpt_sha256 = compute_file_sha256(snap_ckpt_path)
    if expected_sha256 is not None and ckpt_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {ckpt_sha256}"
        )

    # Load snapshot dictionary on CPU with mmap
    ckpt_data = torch.load(str(snap_ckpt_path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(ckpt_data, dict):
        raise ValueError("Loaded checkpoint is not a dictionary")

    # Reject joint / predictive formats before loading base
    fmt = ckpt_data.get("format", None)
    if fmt in ("predictive_memory_adapter_v1", "predictive_memory_joint_v1"):
        raise ValueError(
            f"Expected native compact checkpoint, but got predictive format: {fmt}"
        )
    if "writer" in ckpt_data or "future_head" in ckpt_data:
        raise ValueError(
            "Checkpoint contains 'writer' or 'future_head' keys; compact inference strictly forbids learned writer modules"
        )

    # Required bare keys
    for req_key in ("model", "config", "norm_stats"):
        if req_key not in ckpt_data:
            raise KeyError(f"Compact checkpoint missing required bare '{req_key}'")

    # Validate training_contract
    tc = ckpt_data.get("training_contract", None)
    if not isinstance(tc, dict):
        raise ValueError("Compact checkpoint missing 'training_contract'")

    tc_format = tc.get("format", None)
    if tc_format != "native_compact_memory_replay_v1":
        raise ValueError(
            f"Expected training_contract format 'native_compact_memory_replay_v1', got {tc_format!r}"
        )
    if fmt is not None and fmt != "native_compact_memory_replay_v1":
        raise ValueError(
            f"Expected checkpoint format 'native_compact_memory_replay_v1', got {fmt!r}"
        )

    contract_expected = get_compact_protocol_contract()
    tc_stream = tc.get("stream_protocol", None)
    if tc_stream != contract_expected:
        raise ValueError(
            f"Training contract stream_protocol mismatch: expected {contract_expected}, got {tc_stream}"
        )

    if tc.get("use_timestamps") is not True:
        raise ValueError(
            f"Training contract use_timestamps must be True, got {tc.get('use_timestamps')}"
        )

    if "writer_config" in tc or "writer" in tc or "future_head" in tc:
        raise ValueError(
            "Training contract contains writer/future_head keys; compact inference strictly forbids learned writer"
        )

    # Load bare native checkpoint preserving FP32 (trainable=True)
    policy_base, raw_config, norm_stats, base_meta = load_native_checkpoint(
        fabri_root=fabri_root,
        checkpoint_path=snap_ckpt_path,
        vlm_path=vlm_path,
        device=str(device),
        arm_key=arm_key,
        trainable=True,
    )

    # Assert FA2 on GPU
    target_device = torch.device(device)
    backend_info = (
        assert_native_fa2(policy_base)
        if target_device.type == "cuda"
        else {"native_fa2_enabled": False}
    )

    # Resolve shallow layer from action head config or raw_config
    ah_config = getattr(getattr(policy_base, "action_head", None), "config", None)
    shallow_layer = getattr(ah_config, "shallow_layer_index", None)
    if shallow_layer is None:
        shallow_layer = raw_config.get("shallow_layer_index", raw_config.get("shallow_layer", 6))

    # Instantiate CompactInferencePolicy (throws out of range shallow_layer via NativeSequencePolicy)
    policy = CompactInferencePolicy(
        policy=policy_base,
        shallow_layer=shallow_layer,
        use_timestamps=True,
        gradient_checkpointing=False,
        compact_config=CompactMemoryConfig(),
    )

    # Freeze all parameters and set eval mode
    policy.requires_grad_(False)
    policy.eval()

    # Metadata extraction
    action_head_cfg = getattr(policy.policy.action_head, "config", None)
    horizon = getattr(
        action_head_cfg,
        "pred_horizon",
        getattr(action_head_cfg, "horizon", raw_config.get("horizon", None)),
    )
    state_dim = getattr(action_head_cfg, "state_dim", raw_config.get("state_dim", None))
    action_dim = getattr(
        action_head_cfg,
        "action_dim",
        getattr(action_head_cfg, "per_action_dim", raw_config.get("action_dim", None)),
    )
    image_size = getattr(action_head_cfg, "image_size", raw_config.get("image_size", 448))

    metadata: Dict[str, Any] = {
        "format": "native_compact_memory_replay_v1",
        "checkpoint_source": str(ckpt_src),
        "checkpoint_snapshot": str(snap_ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "base_snapshot": str(snap_ckpt_path),
        "base_sha256": ckpt_sha256,
        "update": None,
        "parent": None,
        "global_step": ckpt_data.get("global_step", ckpt_data.get("step", None)),
        "shallow_layer": shallow_layer,
        "compact_protocol_contract": contract_expected,
        "memory_kind": "fixed_compact_temporal",
        "learned_writer_enabled": False,
        "new_trainable_params": 0,
        "writer_params_loaded": 0,
        "future_head_params_loaded": 0,
        "teacher_instantiated": False,
        "training_only_omitted": ["optimizer", "scheduler", "rng_states"],
        "precision": "float32",
        "backend": target_device.type,
        "backend_diagnostics": backend_info,
        "native_metadata": base_meta,
        "image_size": image_size,
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }

    return policy, norm_stats, metadata
