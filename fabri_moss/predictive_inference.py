"""Predictive Memory Policy Offline and Closed-Loop Inference Loader and Episode History.

Provides:
- snapshot_checkpoint: immutable atomic/exclusive hardlink or copy of checkpoint into snapshot_dir.
- load_predictive_inference: strictly loads predictive_memory_adapter_v1 or predictive_memory_joint_v1
  into a frozen PredictiveMemoryPolicy (teacher intentionally omitted online).
- PredictiveEpisodeHistory: strictly captures observed prefix, handles deepcopy/clones,
  formats sample_for_decision, and records committed decisions.
"""

from __future__ import annotations

import copy
import dataclasses
import errno
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from fabri_moss.async_pipeline import Observation
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.predictive_memory import WriterConfig
from fabri_moss.predictive_policy import PredictiveMemoryPolicy
from fabri_moss.runtime import (
    assert_native_fa2,
    compute_file_sha256,
    load_native_checkpoint,
)


def snapshot_checkpoint(source: str | Path, dest: str | Path) -> Path:
    """Create an immutable snapshot of source checkpoint at dest without overwriting.

    Uses exclusive hardlink os.link if possible (atomic replace writer safe on same filesystem),
    or exclusive copy without overwriting. dest must not already exist.
    """
    src_path = Path(source).resolve()
    dst_path = Path(dest).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

    if dst_path.exists():
        raise FileExistsError(f"Snapshot destination already exists: {dst_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt hardlink first (atomic and writer-safe)
    try:
        os.link(src_path, dst_path)
    except (OSError, NotImplementedError) as err:
        # Cross-device link or unsupported -> copy exclusively
        if isinstance(err, OSError) and err.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        # Exclusive copy to prevent race overwrite
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        with open(src_path, "rb") as f_src:
            fd_dst = os.open(dst_path, flags, 0o644)
            with open(fd_dst, "wb", closefd=True) as f_dst:
                shutil.copyfileobj(f_src, f_dst)

    return dst_path


def _validate_temporal_contract(train_contract: Dict[str, Any]) -> None:
    """Strictly assert temporal RoPE contract matches compact protocol contract."""
    compact_contract = get_compact_protocol_contract()
    expected_temporal = compact_contract["temporal"]

    actual_temporal = train_contract.get("temporal")

    if actual_temporal != expected_temporal:
        raise ValueError(
            f"Temporal contract mismatch: expected {expected_temporal}, got {actual_temporal}"
        )

    text_ts = train_contract.get("text_timestamps")
    if text_ts is not False:
        raise ValueError(f"text_timestamps must be False in temporal contract, got {text_ts}")


def _load_and_validate_submodules(
    policy: PredictiveMemoryPolicy,
    writer_dict: Dict[str, Any],
    future_head_dict: Dict[str, Any],
    writer_config: WriterConfig,
) -> Tuple[int, int]:
    """Strictly load and validate writer and future head state dicts on CPU/device."""
    if policy.writer_config != writer_config:
        raise ValueError("Writer configuration differs from the constructed policy")
    if not isinstance(writer_dict, dict) or not isinstance(future_head_dict, dict):
        raise TypeError("Writer and future head state_dicts must be dictionaries")

    # Validate finite and float32 on state dicts
    for name, tensor in writer_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Writer state_dict item '{name}' is not a Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"Writer state_dict item '{name}' must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Writer state_dict item '{name}' contains non-finite values")

    for name, tensor in future_head_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"FutureHead state_dict item '{name}' is not a Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"FutureHead state_dict item '{name}' must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"FutureHead state_dict item '{name}' contains non-finite values")

    # Strict load into policy.writer and policy.future_head
    writer_incompat = policy.writer.load_state_dict(writer_dict, strict=True)
    if writer_incompat.missing_keys or writer_incompat.unexpected_keys:
        raise RuntimeError(f"Writer load mismatch: {writer_incompat}")

    head_incompat = policy.future_head.load_state_dict(future_head_dict, strict=True)
    if head_incompat.missing_keys or head_incompat.unexpected_keys:
        raise RuntimeError(f"FutureHead load mismatch: {head_incompat}")

    # Freeze writer and head
    policy.writer.requires_grad_(False)
    policy.writer.eval()
    policy.future_head.requires_grad_(False)
    policy.future_head.eval()

    writer_param_count = sum(p.numel() for p in policy.writer.parameters())
    head_param_count = sum(p.numel() for p in policy.future_head.parameters())
    return writer_param_count, head_param_count


def load_predictive_inference(
    checkpoint_path: str | Path,
    *,
    snapshot_dir: str | Path,
    fabri_root: str | Path = "/root/FabriVLA",
    vlm_path: str | Path = "/root/models/InternVL3_5-1B",
    device: str = "cpu",
    arm_key: str = "metaworld_sawyer",
    expected_sha256: Optional[str] = None,
) -> Tuple[PredictiveMemoryPolicy, Dict[str, Any], Dict[str, Any]]:
    """Strictly load a predictive checkpoint for evaluation inference.

    Supports:
      - predictive_memory_adapter_v1
      - predictive_memory_joint_v1
    """
    ckpt_src = Path(checkpoint_path).resolve()
    snap_root = Path(snapshot_dir).resolve()
    snap_root.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot primary checkpoint
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

    fmt = ckpt_data.get("format", None)
    if fmt not in ("predictive_memory_adapter_v1", "predictive_memory_joint_v1"):
        raise ValueError(f"Unrecognized or missing checkpoint format: {fmt}")

    # Validate temporal contract
    tc = ckpt_data.get("training_contract", {})
    if not isinstance(tc, dict):
        tc = {}
    _validate_temporal_contract(tc)

    # Shallow layer validation
    shallow_layer = tc.get("shallow_layer")
    if type(shallow_layer) is not int or shallow_layer <= 0:
        raise ValueError(f"Invalid shallow_layer in training_contract: {shallow_layer}")

    # Resolve writer_config
    raw_writer_cfg = ckpt_data.get("writer_config", None)
    if raw_writer_cfg is None:
        raw_writer_cfg = tc.get("writer_config", None)
    if not isinstance(raw_writer_cfg, dict):
        raise ValueError("Missing writer_config in checkpoint or training_contract")
    if raw_writer_cfg != tc.get("writer_config"):
        raise ValueError("Checkpoint writer_config differs from the training contract")
    writer_config = WriterConfig(**raw_writer_cfg)
    if fmt == "predictive_memory_joint_v1":
        for key in ("model", "config", "norm_stats"):
            if key not in ckpt_data:
                raise KeyError(f"Joint checkpoint missing required '{key}'")

    # Writer & FutureHead weights
    writer_dict = ckpt_data.get("writer", None)
    future_head_dict = ckpt_data.get("future_head", None)
    if writer_dict is None:
        raise KeyError("Checkpoint missing required 'writer' state_dict")
    if future_head_dict is None:
        raise KeyError("Checkpoint missing required 'future_head' state_dict")

    base_snapshot_path: Optional[Path] = None
    base_sha256: Optional[str] = None
    raw_config: Dict[str, Any] = {}
    norm_stats: Dict[str, Any] = {}
    policy_base: Any = None
    base_meta: Dict[str, Any] = {}

    if fmt == "predictive_memory_adapter_v1":
        # Must have parent_path and parent_sha256
        parent_path = ckpt_data.get("parent_path", None)
        parent_sha256 = ckpt_data.get("parent_sha256", None)
        if not parent_path or not parent_sha256:
            raise ValueError("Adapter checkpoint missing 'parent_path' or 'parent_sha256'")

        parent_src = Path(parent_path).resolve()
        base_snapshot_path = snap_root / "parent.pt"
        snapshot_checkpoint(parent_src, base_snapshot_path)
        base_sha256 = compute_file_sha256(base_snapshot_path)
        if base_sha256 != parent_sha256:
            raise ValueError(
                f"Parent checkpoint SHA256 mismatch: contract={parent_sha256}, actual={base_sha256}"
            )
        if tc.get("parent_sha256") != parent_sha256 or tc.get("parent_path") != parent_path:
            raise ValueError("Adapter parent lineage differs from its training contract")

        policy_base, raw_config, norm_stats, base_meta = load_native_checkpoint(
            fabri_root=fabri_root,
            checkpoint_path=base_snapshot_path,
            vlm_path=vlm_path,
            device=str(device),
            arm_key=arm_key,
            trainable=True,
        )

    elif fmt == "predictive_memory_joint_v1":
        # Load bare base model directly from the snapshot
        base_snapshot_path = snap_ckpt_path
        base_sha256 = ckpt_sha256

        policy_base, raw_config, norm_stats, base_meta = load_native_checkpoint(
            fabri_root=fabri_root,
            checkpoint_path=base_snapshot_path,
            vlm_path=vlm_path,
            device=str(device),
            arm_key=arm_key,
            trainable=True,
        )

    # Assert FA2 on GPU
    target_device = torch.device(device)
    backend_info = assert_native_fa2(policy_base) if target_device.type == "cuda" else {"native_fa2_enabled": False}
    model_dim = policy_base.embedder.model.language_model.config.hidden_size
    if writer_config.input_dim != model_dim:
        raise ValueError("Writer input dimension does not match the native language model")

    # Wrap into PredictiveMemoryPolicy (FP32, frozen, eval)
    policy = PredictiveMemoryPolicy(
        policy=policy_base,
        writer_config=writer_config,
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
    )

    # Load and validate writer and future head
    writer_count, head_count = _load_and_validate_submodules(
        policy=policy,
        writer_dict=writer_dict,
        future_head_dict=future_head_dict,
        writer_config=writer_config,
    )

    # Ensure entire policy is frozen and in eval
    policy.requires_grad_(False)
    policy.eval()

    # Model metadata extraction
    action_head_cfg = getattr(policy.policy.action_head, "config", None)
    horizon = getattr(action_head_cfg, "pred_horizon", getattr(action_head_cfg, "horizon", raw_config.get("horizon", None)))
    state_dim = getattr(action_head_cfg, "state_dim", raw_config.get("state_dim", None))
    action_dim = getattr(action_head_cfg, "action_dim", getattr(action_head_cfg, "per_action_dim", raw_config.get("action_dim", None)))
    image_size = getattr(action_head_cfg, "image_size", raw_config.get("image_size", 448))

    metadata: Dict[str, Any] = {
        "format": fmt,
        "checkpoint_source": str(ckpt_src),
        "checkpoint_snapshot": str(snap_ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "base_snapshot": str(base_snapshot_path),
        "base_sha256": base_sha256,
        "update": ckpt_data.get("update", None),
        "global_step": ckpt_data.get("global_step", None),
        "parent_global_step": ckpt_data.get("parent_global_step", None),
        "writer_config": dataclasses.asdict(writer_config),
        "shallow_layer": shallow_layer,
        "precision": "float32",
        "backend": target_device.type,
        "backend_diagnostics": backend_info,
        "native_metadata": base_meta,
        "writer_params_loaded": writer_count,
        "future_head_params_loaded": head_count,
        "teacher_instantiated": False,
        "training_only_omitted": ["teacher", "optimizer", "scheduler", "rng_states"],
        "image_size": image_size,
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }

    return policy, norm_stats, metadata


class PredictiveEpisodeHistory:
    """Manages observation history and samples prefix dicts for predictive policy decision.

    Guarantees:
      - Deepcopy / CPU clone for all images and states.
      - Strict strictly increasing frame_id and non-decreasing finite observation_times.
      - Exactly single image per observation frame.
      - Memory replay flags and clean separation of sample_for_decision and commit_decision.
    """

    def __init__(self, prompt: str) -> None:
        self.reset(prompt)

    def reset(self, prompt: str) -> None:
        """Reset history to initial empty state with given prompt."""
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError(f"prompt must be a non-empty string, got {prompt}")
        self._prompt: str = prompt
        self._images: List[Any] = []
        self._frame_ids: List[int] = []
        self._times: List[float] = []
        self._decision_indices: List[int] = []
        self._latest_state: Optional[torch.Tensor] = None
        self._latest_state_mask: Optional[torch.Tensor] = None
        self._latest_action_mask: Optional[torch.Tensor] = None
        self._pending_decision_index: Optional[int] = None

    @property
    def frame_ids(self) -> Tuple[int, ...]:
        return tuple(self._frame_ids)

    @property
    def observation_times(self) -> Tuple[float, ...]:
        return tuple(self._times)

    @property
    def decision_indices(self) -> Tuple[int, ...]:
        return tuple(self._decision_indices)

    def append(self, obs: Observation) -> None:
        """Append an observation to the episode history."""
        if not isinstance(obs, Observation):
            raise TypeError(f"obs must be an Observation instance, got {type(obs).__name__}")

        if len(self._frame_ids) > 0:
            if obs.frame_id <= self._frame_ids[-1]:
                raise ValueError(
                    f"obs.frame_id {obs.frame_id} must be strictly greater than last frame_id {self._frame_ids[-1]}"
                )

        t = obs.observation_time
        if isinstance(t, bool) or not (isinstance(t, (int, float)) and math.isfinite(t) and t >= 0.0):
            raise ValueError(f"Observation time {t} must be finite non-negative float")

        if len(self._times) > 0:
            if t < self._times[-1]:
                raise ValueError(
                    f"Observation time {t} is backwards relative to last time {self._times[-1]}"
                )

        if not isinstance(obs.images, (tuple, list)) or len(obs.images) != 1:
            raise ValueError(f"Observation must contain exactly 1 image, got {len(obs.images)}")

        img = obs.images[0]
        # Deepcopy PIL / numpy / tensor image
        if isinstance(img, Image.Image):
            img_copy = img.copy()
        elif isinstance(img, np.ndarray):
            img_copy = img.copy()
        elif isinstance(img, torch.Tensor):
            img_copy = img.detach().clone().cpu()
        else:
            img_copy = copy.deepcopy(img)

        # A new observation invalidates an uncommitted sample of the preceding prefix.
        self._pending_decision_index = None

        # Clone state and masks to CPU
        st_copy = obs.state.detach().clone().cpu()
        stm_copy = obs.state_mask.detach().clone().cpu()
        acm_copy = obs.action_mask.detach().clone().cpu()

        self._images.append(img_copy)
        self._frame_ids.append(obs.frame_id)
        self._times.append(float(t))
        self._latest_state = st_copy
        self._latest_state_mask = stm_copy
        self._latest_action_mask = acm_copy

    def sample_for_decision(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Produce sample dictionary for predictive policy decision inference.

        Does not commit the decision until commit_decision() is explicitly called.
        """
        N = len(self._images)
        if N == 0:
            raise RuntimeError("Cannot sample_for_decision from empty history")

        latest_idx = N - 1
        if len(self._decision_indices) > 0 and self._decision_indices[-1] == latest_idx:
            raise RuntimeError(
                f"Attempting to sample decision at already committed index {latest_idx}"
            )

        self._pending_decision_index = latest_idx

        # Deepcopy images window
        images_window = [
            img.copy() if isinstance(img, (Image.Image, np.ndarray)) else copy.deepcopy(img)
            for img in self._images
        ]

        # Previous decisions + latest target decision index
        dec_indices = list(self._decision_indices) + [latest_idx]

        # Format state tensors to [1, D] and move to device
        cur_st = self._latest_state.to(device=device, copy=True)
        cur_stm = self._latest_state_mask.to(device=device, copy=True)

        cur_acm = self._latest_action_mask.to(device=device, copy=True)

        sample: Dict[str, Any] = {
            "prompt": self._prompt,
            "images_window": [[image] for image in images_window],
            "frame_ids": list(self._frame_ids),
            "observation_times": list(self._times),
            "target_indices": [latest_idx],
            "decision_indices": dec_indices,
            "memory_replay": True,
            "state": cur_st,
            "state_mask": cur_stm,
            "action_mask": cur_acm,
        }
        return sample

    def commit_decision(self) -> int:
        """Commit the pending decision after successful prediction.

        Raises RuntimeError if no decision is pending or if index is already committed.
        """
        if self._pending_decision_index is None:
            raise RuntimeError("No pending decision to commit")

        idx = self._pending_decision_index
        if len(self._decision_indices) > 0 and idx <= self._decision_indices[-1]:
            raise RuntimeError(
                f"Cannot commit decision index {idx} <= last decision index {self._decision_indices[-1]}"
            )

        self._decision_indices.append(idx)
        self._pending_decision_index = None
        return idx
