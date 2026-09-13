"""Frozen original FabriVLA API baseline with only the latest observation.

Loaded weight values are preserved in FP32; CUDA inference uses BF16 autocast.
This is not a reproduction of the historical benchmark's loading precision.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fabri_moss.native_training import _autocast_context
from fabri_moss.predictive_inference import snapshot_checkpoint
from fabri_moss.runtime import assert_native_fa2, compute_file_sha256, load_native_checkpoint


class NativeSingleInferencePolicy(nn.Module):
    """Call the original API without adding memory, timestamps or parameters."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        if not callable(getattr(policy, "run_inference", None)):
            raise TypeError("Native single-frame policy must implement run_inference")
        if any(getattr(policy, key, None) is not None for key in ("writer", "future_head")):
            raise ValueError("Native single-frame policy forbids writer and future_head")
        self.policy = policy
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def predict_actions(self, sample: Dict[str, Any]) -> torch.Tensor:
        images_window = sample["images_window"]
        if not images_window or not images_window[-1]:
            raise ValueError("Native single-frame inference requires the latest images")
        images = list(images_window[-1])
        device = next(self.policy.parameters()).device
        latest = {
            key: sample[key][-1:].to(device) if sample.get(key) is not None else None
            for key in ("state", "state_mask", "action_mask")
        }
        with _autocast_context(device):
            return self.policy.run_inference(
                images=images,
                image_mask=torch.ones(len(images), dtype=torch.bool, device=device),
                prompt=sample["prompt"],
                **latest,
            )


def load_native_single_inference(
    checkpoint_path: str | Path,
    *,
    snapshot_dir: str | Path,
    fabri_root: str | Path = "/root/FabriVLA",
    vlm_path: str | Path = "/root/models/InternVL3_5-1B",
    device: str = "cpu",
    arm_key: str = "metaworld_sawyer",
    expected_sha256: Optional[str] = None,
) -> Tuple[NativeSingleInferencePolicy, Dict[str, Any], Dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    snapshot = Path(snapshot_dir).resolve() / "checkpoint.pt"
    snapshot_checkpoint(source, snapshot)
    sha256 = compute_file_sha256(snapshot)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {sha256}")

    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Native single-frame checkpoint must be a dictionary")
    if checkpoint.get("format") is not None or any(
        key in checkpoint for key in ("writer", "future_head", "training_contract")
    ):
        raise ValueError("Native single-frame baseline requires an original bare FabriVLA checkpoint")
    for key in ("model", "config", "norm_stats"):
        if key not in checkpoint:
            raise KeyError(f"Native single-frame checkpoint missing '{key}'")
    del checkpoint

    # trainable=True prevents BF16 rounding during loading; the wrapper freezes it.
    base, config, norm_stats, native_metadata = load_native_checkpoint(
        fabri_root=fabri_root,
        checkpoint_path=snapshot,
        vlm_path=vlm_path,
        device=str(device),
        arm_key=arm_key,
        trainable=True,
    )
    policy = NativeSingleInferencePolicy(base)
    target_device = torch.device(device)
    backend = assert_native_fa2(base) if target_device.type == "cuda" else {"native_fa2_enabled": False}
    head_config = base.action_head.config
    metadata = {
        "format": "native_single_frame_v1",
        "checkpoint_source": str(source),
        "checkpoint_snapshot": str(snapshot),
        "checkpoint_sha256": sha256,
        "base_snapshot": str(snapshot),
        "base_sha256": sha256,
        "global_step": native_metadata.get("step"),
        "memory_kind": "native_single_frame",
        "learned_writer_enabled": False,
        "new_trainable_params": 0,
        "writer_params_loaded": 0,
        "future_head_params_loaded": 0,
        "teacher_instantiated": False,
        "inference_api": "original FabriVLA.run_inference",
        "precision": "fp32_preserved_base_bf16_autocast" if target_device.type == "cuda" else "float32",
        "historical_benchmark_precision_reproduced": False,
        "backend": target_device.type,
        "backend_diagnostics": backend,
        "native_metadata": native_metadata,
        "image_size": config.get("image_size", 448),
        "horizon": head_config.horizon,
        "state_dim": head_config.state_dim,
        "action_dim": head_config.per_action_dim,
    }
    return policy, norm_stats, metadata
