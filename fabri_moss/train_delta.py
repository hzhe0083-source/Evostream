"""Trainer for FabriVLA bounded delta visual memory adapter on sequential MetaWorld episodes."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.delta_data import (
    EpisodePermutationSampler,
    MetaWorldEpisodes,
    identity_collate,
)
from fabri_moss.runtime import (
    assert_native_fa2,
    compute_file_sha256,
    load_native_checkpoint,
)
from fabri_moss.train import (
    compute_flow_kd_loss,
    initialize_adapter_weights,
    validate_checkpoint_metadata_and_stats,
)


def require_training_device(policy: nn.Module, device: str) -> None:
    """Helper strictly enforcing CUDA for training and that all policy parameters reside on CUDA."""
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError(
            f"Training requires CUDA device, got {device!r}. CPU bypass is forbidden in formal runs."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    for name, param in policy.named_parameters():
        if param.device.type != "cuda":
            raise RuntimeError(
                f"Policy parameter {name} is on {param.device}, but must be on CUDA"
            )


def create_optimizer_and_scheduler(
    student_model: MossInternVL,
    stage: str,
    lr: float = 1e-4,
    head_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    total_steps: int = 1000,
    warmup_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create AdamW optimizer with parameter groups and warmup-cosine scheduler.

    - Bias, norm, and <= 1D parameters have weight decay 0.0.
    - Other parameters have weight decay weight_decay.
    - If stage is 'expert', student_model.policy.action_head parameters use head_lr.
    - warmup_steps can be 0 (step 0 starts at full lr).
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    head_decay_params: List[nn.Parameter] = []
    head_no_decay_params: List[nn.Parameter] = []

    for name, param in student_model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = "action_head" in name
        is_no_decay = (
            param.ndim <= 1
            or "bias" in name
            or "norm" in name
            or "write_logits" in name
            or "memory_gate" in name
            or "attn_gate" in name
            or "mlp_gate" in name
        )

        if is_head:
            if is_no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)
        else:
            if is_no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if head_decay_params:
        param_groups.append({"params": head_decay_params, "lr": head_lr, "weight_decay": weight_decay})
    if head_no_decay_params:
        param_groups.append({"params": head_no_decay_params, "lr": head_lr, "weight_decay": 0.0})

    optimizer = AdamW(param_groups)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0:
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            progress = float(current_step) / float(max(1, total_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def save_delta_checkpoint(
    output_path: Path,
    student_model: MossInternVL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    epoch: int,
    next_episode_cursor: int,
    stage: str,
    args: argparse.Namespace,
    norm_stats: Dict[str, Any],
    base_metadata: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    training_contract: Dict[str, Any],
    init_provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomic checkpoint saver for fabri_delta_v1 format with complete contracts."""
    config_dict = dataclasses.asdict(student_model.config)
    clean_args = {k: v for k, v in vars(args).items() if not k.startswith("_")}

    ckpt = {
        "format": "fabri_delta_v1",
        "step": step,
        "epoch": epoch,
        "next_episode_cursor": next_episode_cursor,
        "stage": stage,
        "config": config_dict,
        "cross_blocks": student_model.cross_blocks.state_dict(),
        "readout_embeddings": student_model.readout_embeddings.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "norm_stats": norm_stats,
        "base_metadata": base_metadata,
        "train_data_contract": train_data_contract,
        "val_data_contract": val_data_contract,
        "data_contract": train_data_contract,  # backward compatibility alias
        "training_contract": training_contract,
        "args": clean_args,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "random": random.getstate(),
        },
    }

    student_device = next(student_model.parameters()).device
    if student_device.type == "cuda" and torch.cuda.is_available():
        ckpt["rng_state"]["torch_cuda"] = [
            s.cpu() for s in torch.cuda.get_rng_state_all()
        ]

    if init_provenance is not None:
        ckpt["init_provenance"] = init_provenance

    if stage == "expert":
        ckpt["action_head"] = student_model.policy.action_head.state_dict()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = output_path.parent / f"{output_path.name}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    try:
        torch.save(ckpt, str(tmp_file))
        os.replace(str(tmp_file), str(output_path))
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


def load_delta_checkpoint(
    resume_path: Union[str, Path],
    student_model: MossInternVL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    expected_stage: str,
    expected_norm_stats: Dict[str, Any],
    current_base_metadata: Dict[str, Any],
    expected_data_contract: Optional[Dict[str, Any]] = None,
    expected_training_contract: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    expected_val_data_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Strictly resume checkpoint from fabri_delta_v1 format BEFORE any weights are mutated."""
    resume_path = Path(resume_path).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found at {resume_path}")

    ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)

    ckpt_format = ckpt.get("format")
    if ckpt_format != "fabri_delta_v1":
        raise ValueError(
            f"Checkpoint format mismatch: expected 'fabri_delta_v1', got {ckpt_format!r}. "
            f"Legacy consume checkpoints cannot be resumed in delta mode."
        )

    if "stage" not in ckpt:
        raise KeyError("Checkpoint missing required field 'stage'")
    resumed_stage = ckpt["stage"]
    if resumed_stage != expected_stage:
        raise ValueError(f"Stage mismatch! Resumed: {resumed_stage!r}, expected: {expected_stage!r}")

    # Validate base metadata and norm stats
    validate_checkpoint_metadata_and_stats(
        ckpt=ckpt,
        student_model=student_model,
        expected_norm_stats=expected_norm_stats,
        current_base_metadata=current_base_metadata,
        allow_max_frames_mismatch=False,
    )

    # Validate train data contract (exact dict equality)
    if expected_data_contract is not None:
        ckpt_contract = ckpt.get("train_data_contract", ckpt.get("data_contract"))
        if not isinstance(ckpt_contract, dict):
            raise KeyError("Checkpoint missing required 'train_data_contract' or 'data_contract'")
        if ckpt_contract != expected_data_contract:
            raise ValueError(
                f"train_data_contract mismatch: resumed {ckpt_contract} != expected {expected_data_contract}"
            )

    # Validate val data contract (exact dict equality)
    if expected_val_data_contract is not None:
        if "val_data_contract" not in ckpt:
            raise KeyError("Checkpoint missing required 'val_data_contract'")
        ckpt_val_contract = ckpt["val_data_contract"]
        if not isinstance(ckpt_val_contract, dict):
            raise ValueError("Checkpoint 'val_data_contract' must be a dict")
        if ckpt_val_contract != expected_val_data_contract:
            raise ValueError(
                f"val_data_contract mismatch: resumed {ckpt_val_contract} != expected {expected_val_data_contract}"
            )

    # Validate training contract (exact dict equality)
    if expected_training_contract is not None:
        if "training_contract" not in ckpt:
            raise KeyError("Checkpoint missing required 'training_contract' in delta mode")
        ckpt_tr = ckpt["training_contract"]
        if not isinstance(ckpt_tr, dict):
            raise ValueError("Checkpoint 'training_contract' must be a dict")
        if ckpt_tr != expected_training_contract:
            raise ValueError(
                f"training_contract mismatch: resumed {ckpt_tr} != expected {expected_training_contract}"
            )

    # Check required weight keys and shape matches before mutating weights
    if "cross_blocks" not in ckpt or "readout_embeddings" not in ckpt:
        raise KeyError("Checkpoint missing 'cross_blocks' or 'readout_embeddings'")

    target_cb_state = student_model.cross_blocks.state_dict()
    source_cb_state = ckpt["cross_blocks"]
    if set(target_cb_state.keys()) != set(source_cb_state.keys()):
        raise KeyError("cross_blocks state_dict keys mismatch between model and checkpoint")
    for k, v in target_cb_state.items():
        src_v = source_cb_state[k]
        if v.shape != src_v.shape:
            raise ValueError(f"cross_blocks param shape mismatch for {k}: target {v.shape} vs source {src_v.shape}")

    source_readout = ckpt["readout_embeddings"]
    if student_model.readout_embeddings.shape != source_readout.shape:
        raise ValueError(
            f"readout_embeddings shape mismatch: target {student_model.readout_embeddings.shape} vs source {source_readout.shape} (broadcast disallowed)"
        )

    if expected_stage == "expert":
        if "action_head" not in ckpt:
            raise KeyError("Stage 'expert' requested but checkpoint missing 'action_head'")
        target_head_state = student_model.policy.action_head.state_dict()
        source_head_state = ckpt["action_head"]
        if set(target_head_state.keys()) != set(source_head_state.keys()):
            raise KeyError("action_head state_dict keys mismatch between model and checkpoint")
        for k, v in target_head_state.items():
            src_v = source_head_state[k]
            if v.shape != src_v.shape:
                raise ValueError(f"action_head param shape mismatch for {k}: target {v.shape} vs source {src_v.shape}")

    # Verify optimizer, scheduler, and RNG fields exist
    if "optimizer" not in ckpt:
        raise KeyError("Checkpoint missing required 'optimizer' state dict")
    if "scheduler" not in ckpt:
        raise KeyError("Checkpoint missing required 'scheduler' state dict")
    if "rng_state" not in ckpt:
        raise KeyError("Checkpoint missing required 'rng_state'")
    rng = ckpt["rng_state"]
    if "torch" not in rng or "random" not in rng:
        raise KeyError("Checkpoint 'rng_state' missing 'torch' or 'random'")

    # Validate step, epoch, cursor: strict int type, reject bool/float truncation
    for field_name in ("step", "epoch", "next_episode_cursor"):
        if field_name not in ckpt:
            raise KeyError(f"Checkpoint missing required field '{field_name}'")
        val = ckpt[field_name]
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(
                f"Checkpoint field '{field_name}' must be an integer, got {type(val).__name__} ({val!r})"
            )
        if val < 0:
            raise ValueError(f"Invalid negative {field_name} in checkpoint: {val}")

    step = ckpt["step"]
    epoch = ckpt["epoch"]
    next_episode_cursor = ckpt["next_episode_cursor"]

    # Validate cursor & step consistency if full training/data contracts are present
    ckpt_train_c = ckpt.get("train_data_contract", ckpt.get("data_contract"))
    ckpt_tr = ckpt.get("training_contract")

    has_full_contracts = (
        isinstance(ckpt_train_c, dict)
        and "active_episode_ids" in ckpt_train_c
        and isinstance(ckpt_train_c["active_episode_ids"], (list, tuple))
        and isinstance(ckpt_tr, dict)
        and "episodes_per_step" in ckpt_tr
        and "epochs" in ckpt_tr
        and "scheduler_budget" in ckpt_tr
    )

    if has_full_contracts:
        N = len(ckpt_train_c["active_episode_ids"])
        accum = ckpt_tr["episodes_per_step"]
        contract_epochs = ckpt_tr["epochs"]

        if isinstance(accum, bool) or not isinstance(accum, int) or accum <= 0:
            raise ValueError(f"Invalid episodes_per_step in contract: {accum}")
        if isinstance(contract_epochs, bool) or not isinstance(contract_epochs, int) or contract_epochs <= 0:
            raise ValueError(f"Invalid epochs in contract: {contract_epochs}")

        per_epoch = math.ceil(N / accum)

        if epoch > contract_epochs:
            raise ValueError(
                f"Checkpoint epoch {epoch} exceeds training contract epochs {contract_epochs}"
            )
        if next_episode_cursor > N:
            raise ValueError(
                f"Checkpoint next_episode_cursor {next_episode_cursor} exceeds dataset episode count {N}"
            )
        if next_episode_cursor != N and (next_episode_cursor % accum) != 0:
            raise ValueError(
                f"Checkpoint next_episode_cursor {next_episode_cursor} is not on an optimizer boundary "
                f"(accum={accum}, total={N})"
            )
        if epoch == contract_epochs and next_episode_cursor != 0:
            raise ValueError(
                f"Checkpoint at final epoch {epoch} must have next_episode_cursor=0, got {next_episode_cursor}"
            )
        expected_step = epoch * per_epoch + math.ceil(next_episode_cursor / accum)
        if step != expected_step:
            raise ValueError(
                f"Checkpoint step mismatch: step={step} != expected_step={expected_step} "
                f"(epoch={epoch}, per_epoch={per_epoch}, cursor={next_episode_cursor}, accum={accum})"
            )

        # Scheduler last_epoch check
        sched_state = ckpt["scheduler"]
        if not isinstance(sched_state, dict) or "last_epoch" not in sched_state:
            raise KeyError("Checkpoint scheduler state missing 'last_epoch'")
        sched_last_epoch = sched_state["last_epoch"]
        if isinstance(sched_last_epoch, bool) or not isinstance(sched_last_epoch, int):
            raise TypeError(
                f"Scheduler state 'last_epoch' must be an integer, got {type(sched_last_epoch).__name__}"
            )
        if sched_last_epoch != step:
            raise ValueError(
                f"Scheduler state last_epoch {sched_last_epoch} does not match checkpoint step {step}"
            )

    # Validate GPU RNG before mutating weights
    target_device = torch.device(device)
    student_device = next(student_model.parameters()).device
    is_gpu = (student_device.type == "cuda" or target_device.type == "cuda")

    if is_gpu:
        if "torch_cuda" not in rng or rng["torch_cuda"] is None:
            raise KeyError(
                "Resuming GPU model requires 'torch_cuda' in rng_state to prevent silent RNG drift"
            )

    # All checks passed, mutate weights
    student_model.cross_blocks.load_state_dict(source_cb_state, strict=True)
    with torch.no_grad():
        student_model.readout_embeddings.copy_(source_readout)

    if expected_stage == "expert":
        student_model.policy.action_head.load_state_dict(ckpt["action_head"], strict=True)

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    # Restore RNG
    rng_t = rng["torch"]
    if isinstance(rng_t, torch.Tensor):
        rng_t = rng_t.cpu()
    torch.set_rng_state(rng_t)
    random.setstate(rng["random"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        cuda_states = rng["torch_cuda"]
        cpu_cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
        if is_gpu:
            torch.cuda.set_rng_state_all(cpu_cuda_states)

    student_model.set_training_stage(expected_stage)
    student_model.train()

    init_provenance = ckpt.get("init_provenance")
    return {
        "step": step,
        "epoch": epoch,
        "next_episode_cursor": next_episode_cursor,
        "init_provenance": init_provenance,
    }


def evaluate_delta(
    student_model: MossInternVL,
    teacher_policy: nn.Module,
    teacher_head: nn.Module,
    val_dataset: MetaWorldEpisodes,
    device: str = "cpu",
    val_episodes: int = 50,
    kd_weight: float = 1.0,
    seed: int = 4042,
) -> Dict[str, Any]:
    """Run validation over heldout episodes using identical chunk/TBPTT recurrence protocol.

    - Records was_training BEFORE switching to eval mode; try/finally restores training state.
    - Uses fork_rng to isolate CUDA / CPU RNG states strictly.
    - Uniform noise in [-1, 1] generated deterministically using local CPU generator.
    - Deterministic per-task sampling (1 episode per task if 50 tasks available).
    """
    if val_episodes <= 0:
        return {"val_skipped": True}

    was_training = student_model.training
    student_model.eval()
    teacher_head.eval()

    # Devices for fork_rng
    fork_devices = [torch.device(device)] if torch.device(device).type == "cuda" and torch.cuda.is_available() else []

    try:
        with torch.random.fork_rng(devices=fork_devices):
            val_rng = random.Random(seed + 99991)
            val_gen = torch.Generator(device="cpu").manual_seed(seed + 88883)

            total_gt_loss = 0.0
            total_kd_loss = 0.0
            total_decisions = 0
            evaluated_episode_ids = []

            # Deterministic selection: sample evenly across tasks
            # Group val_dataset indices by task
            task_to_indices: Dict[str, List[int]] = {}
            for idx, ep in enumerate(val_dataset.active_episodes):
                task_str = ep["tasks"][0]
                task_to_indices.setdefault(task_str, []).append(idx)

            chosen_indices: List[int] = []
            sorted_tasks = sorted(task_to_indices.keys())
            # Round-robin pick 1 episode per task until val_episodes reached
            round_idx = 0
            while len(chosen_indices) < val_episodes and len(chosen_indices) < len(val_dataset):
                added_any = False
                for t in sorted_tasks:
                    idxs = task_to_indices[t]
                    if round_idx < len(idxs):
                        chosen_indices.append(idxs[round_idx])
                        added_any = True
                        if len(chosen_indices) >= val_episodes:
                            break
                if not added_any:
                    break
                round_idx += 1

            with torch.no_grad():
                for ep_idx in chosen_indices:
                    ep_data = val_dataset[ep_idx]
                    ep_id = ep_data["episode_id"]
                    prompt = ep_data["prompt"]
                    chunks = ep_data["chunks"]
                    evaluated_episode_ids.append(ep_id)

                    delta_state = None
                    for c_info in chunks:
                        images_window = c_info["images"]
                        frame_ids = c_info["frame_ids"]
                        state = c_info["state"].to(device)
                        state_mask = c_info["state_mask"].to(device)
                        state = state * state_mask

                        actions = c_info["actions"].to(device)
                        action_mask = c_info["action_mask"].to(device)

                        student_deep, student_shallow, delta_state = student_model.forward_delta(
                            images_window=images_window,
                            frame_ids=frame_ids,
                            prompt=prompt,
                            previous=delta_state,
                        )

                        last_images = images_window[-1]
                        image_mask = torch.ones(len(last_images), dtype=torch.bool, device=device)

                        teacher_out = teacher_policy.get_vl_embeddings(
                            images=last_images,
                            image_mask=image_mask,
                            prompt=prompt,
                            return_cls_only=False,
                            shallow_layer_index=student_model.config.shallow_layer,
                        )
                        if isinstance(teacher_out, tuple):
                            teacher_deep, teacher_shallow = teacher_out
                        else:
                            teacher_deep = teacher_out
                            teacher_shallow = None

                        # Deterministic uniform noise [-1, 1] using isolated CPU generator
                        raw_noise = (torch.rand(actions.shape, generator=val_gen, dtype=torch.float32) * 2.0 - 1.0)
                        val_noise = raw_noise.to(device=device, dtype=actions.dtype) * action_mask
                        val_t = torch.tensor([0.5], device=device, dtype=actions.dtype)

                        student_head = student_model.policy.action_head
                        _, gt_l, kd_l, _, _ = compute_flow_kd_loss(
                            student_head=student_head,
                            teacher_head=teacher_head,
                            student_deep=student_deep,
                            student_shallow=student_shallow,
                            teacher_deep=teacher_deep,
                            teacher_shallow=teacher_shallow,
                            state=state,
                            actions=actions,
                            action_mask=action_mask,
                            kd_weight=kd_weight,
                            fixed_noise=val_noise,
                            fixed_t=val_t,
                        )

                        total_gt_loss += float(gt_l.item())
                        total_kd_loss += float(kd_l.item())
                        total_decisions += 1

            avg_gt = total_gt_loss / max(1, total_decisions)
            avg_kd = total_kd_loss / max(1, total_decisions)
            return {
                "val_gt_loss": avg_gt,
                "val_kd_loss": avg_kd,
                "val_total_loss": avg_gt + kd_weight * avg_kd,
                "val_decisions": total_decisions,
                "val_episode_count": len(evaluated_episode_ids),
                "val_episode_ids": evaluated_episode_ids,
            }
    finally:
        if was_training:
            student_model.train()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabriVLA Delta Memory Trainer")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MetaWorld dataset root")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to native FabriVLA repo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/models/FabriVLA/checkpoint_step_93000.pt",
        help="Path to base 93k checkpoint",
    )
    parser.add_argument("--vlm", type=str, default="/root/models/InternVL3_5-1B", help="Path to local InternVL model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device (e.g. cuda:0)")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key for norm stats")

    parser.add_argument("--stage", type=str, choices=["bridge", "expert"], default="bridge", help="Training stage")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for bridge / new parameters")
    parser.add_argument("--head-lr", type=float, default=1e-5, help="Learning rate for expert action head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for 2D+ weights")
    parser.add_argument("--kd-weight", type=float, default=1.0, help="Weight for flow distillation KD loss")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--seed", type=int, default=4042, help="Random seed")

    parser.add_argument("--epochs", type=int, default=5, help="Total training epochs")
    parser.add_argument("--episodes-per-step", type=int, default=4, help="Number of complete episodes per optimizer step")
    parser.add_argument("--tbptt-decisions", type=int, default=4, help="Number of decisions per TBPTT window")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup optimizer steps (0 allowed)")
    parser.add_argument("--save-every", type=int, default=25, help="Save last.pt every N optimizer steps")
    parser.add_argument("--val-every", type=int, default=100, help="Run validation every N optimizer steps")
    parser.add_argument("--val-episodes", type=int, default=50, help="Number of validation episodes")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU intraop threads")

    parser.add_argument("--max-episodes", type=int, default=None, help="Explicit smoke cap on total episodes")
    parser.add_argument("--max-updates", type=int, default=None, help="Explicit smoke cap on optimizer steps")

    parser.add_argument("--resume", type=str, default=None, help="Path to delta checkpoint to resume full run")
    parser.add_argument("--init-adapter", type=str, default=None, help="Path to adapter checkpoint for weight initialization only")

    parser.add_argument("--augmentation", action="store_true", default=True, help="Enable episode-level data augmentation")
    parser.add_argument("--no-augmentation", action="store_false", dest="augmentation", help="Disable data augmentation")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line / namespace arguments before creating directories or loading weights."""
    pos_int_fields = [
        ("epochs", args.epochs),
        ("episodes_per_step", args.episodes_per_step),
        ("tbptt_decisions", args.tbptt_decisions),
        ("save_every", args.save_every),
        ("threads", args.threads),
    ]
    for name, val in pos_int_fields:
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ValueError(f"Argument '{name}' must be a positive integer, got {val!r}")

    nonneg_int_fields = [
        ("warmup_steps", args.warmup_steps),
        ("val_every", args.val_every),
        ("val_episodes", args.val_episodes),
        ("num_workers", args.num_workers),
    ]
    for name, val in nonneg_int_fields:
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError(f"Argument '{name}' must be a non-negative integer, got {val!r}")

    if args.max_episodes is not None:
        if isinstance(args.max_episodes, bool) or not isinstance(args.max_episodes, int) or args.max_episodes <= 0:
            raise ValueError(f"Argument 'max_episodes' must be a positive integer if set, got {args.max_episodes!r}")

    if args.max_updates is not None:
        if isinstance(args.max_updates, bool) or not isinstance(args.max_updates, int) or args.max_updates <= 0:
            raise ValueError(f"Argument 'max_updates' must be a positive integer if set, got {args.max_updates!r}")

    pos_float_fields = [
        ("lr", args.lr),
        ("head_lr", args.head_lr),
        ("grad_clip_norm", args.grad_clip_norm),
    ]
    for name, val in pos_float_fields:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"Argument '{name}' must be a finite positive number, got {val!r}")

    nonneg_float_fields = [
        ("weight_decay", args.weight_decay),
        ("kd_weight", args.kd_weight),
    ]
    for name, val in nonneg_float_fields:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Argument '{name}' must be a finite non-negative number, got {val!r}")


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    if args.resume and args.init_adapter:
        raise ValueError("Cannot specify both --resume and --init-adapter. Choose one.")

    validate_args(args)

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume:
            raise RuntimeError(
                f"Output directory {output_dir} already exists and is not empty. Refusing to overwrite without --resume."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[Init] Loading base policy on {args.device}...")
    loaded = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm,
        device=args.device,
        arm_key=args.arm_key,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 4:
        raise TypeError("load_native_checkpoint must return a 4-tuple (policy, raw_config, norm_stats, metadata)")
    teacher_policy, raw_config, norm_stats, base_metadata = loaded

    # Device verification helper
    require_training_device(teacher_policy, args.device)

    # Native FlashAttention-2 assertion
    fa2_diag = assert_native_fa2(teacher_policy)
    print(f"[Native FA2] Verified: {fa2_diag}")

    # Freeze teacher embedder and whole teacher policy
    teacher_policy.eval()
    for p in teacher_policy.parameters():
        p.requires_grad = False

    # In bridge stage: reuse frozen action_head directly without deepcopying entire head.
    # In expert stage: deepcopy teacher action head BEFORE student modifies or loads adapter.
    if args.stage == "expert":
        teacher_head = copy.deepcopy(teacher_policy.action_head)
        teacher_head.eval()
        for p in teacher_head.parameters():
            p.requires_grad = False
    else:
        teacher_head = teacher_policy.action_head
        teacher_head.eval()
        for p in teacher_head.parameters():
            p.requires_grad = False

    # Build student model with delta memory configuration
    moss_config = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student_model = MossInternVL(teacher_policy, moss_config)
    student_model.set_training_stage(args.stage)

    # Verify all trainable parameters are strictly FP32
    for name, param in student_model.named_parameters():
        if param.requires_grad:
            if param.dtype != torch.float32:
                raise TypeError(
                    f"Trainable parameter {name} has dtype {param.dtype}, strictly expected torch.float32"
                )

    # Setup datasets
    train_dataset = MetaWorldEpisodes(
        root=args.data_root,
        norm_stats=norm_stats,
        horizon=50,
        state_dim=24,
        action_dim=24,
        max_chunk_size=5,
        split="train",
        seed=args.seed,
        val_fraction=0.1,
        max_episodes=args.max_episodes,
        augmentation=args.augmentation,
    )
    val_dataset = MetaWorldEpisodes(
        root=args.data_root,
        norm_stats=norm_stats,
        horizon=50,
        state_dim=24,
        action_dim=24,
        max_chunk_size=5,
        split="val",
        seed=args.seed,
        val_fraction=0.1,
        max_episodes=args.max_episodes,
        augmentation=False,
    )

    train_data_contract = train_dataset.get_data_contract()
    val_data_contract = val_dataset.get_data_contract()
    print(f"[Dataset] Train episodes: {len(train_dataset)}, Val episodes: {len(val_dataset)}")

    # Scheduler budget: based on full epochs and ceiling per epoch
    est_steps_per_epoch = math.ceil(len(train_dataset) / args.episodes_per_step)
    scheduler_budget = args.epochs * est_steps_per_epoch

    optimizer, scheduler = create_optimizer_and_scheduler(
        student_model=student_model,
        stage=args.stage,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        total_steps=scheduler_budget,
        warmup_steps=args.warmup_steps,
    )

    training_contract = {
        "format": "fabri_delta_v1",
        "stage": args.stage,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "kd_weight": args.kd_weight,
        "grad_clip_norm": args.grad_clip_norm,
        "epochs": args.epochs,
        "episodes_per_step": args.episodes_per_step,
        "tbptt_decisions": args.tbptt_decisions,
        "warmup_steps": args.warmup_steps,
        "scheduler_budget": scheduler_budget,
        "validation_noise": "uniform_fixed_seed",
        "validation_t": 0.5,
        "val_every": args.val_every,
        "val_episodes": args.val_episodes,
        "teacher_backend": "flash_attention_2",
    }

    init_provenance = None
    if args.init_adapter:
        print(f"[Init] Initializing adapter weights from {args.init_adapter}...")
        init_provenance = initialize_adapter_weights(
            path=args.init_adapter,
            student_model=student_model,
            expected_norm_stats=norm_stats,
            current_base_metadata=base_metadata,
            device="cpu",
        )

    start_step = 0
    start_epoch = 0
    start_episode_cursor = 0

    if args.resume:
        print(f"[Resume] Resuming training from {args.resume}...")
        resume_info = load_delta_checkpoint(
            resume_path=args.resume,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_stage=args.stage,
            expected_norm_stats=norm_stats,
            current_base_metadata=base_metadata,
            expected_data_contract=train_data_contract,
            expected_training_contract=training_contract,
            device=args.device,
            expected_val_data_contract=val_data_contract,
        )
        start_step = resume_info["step"]
        start_epoch = resume_info["epoch"]
        start_episode_cursor = resume_info["next_episode_cursor"]
        if resume_info.get("init_provenance") is not None:
            init_provenance = resume_info["init_provenance"]
        print(f"[Resume] Resumed at step={start_step}, epoch={start_epoch}, cursor={start_episode_cursor}")

        if start_epoch >= args.epochs:
            print(f"[Finished] Training already completed (epoch {start_epoch}/{args.epochs}). Exiting without modification.")
            return

    # Write run_config.json on fresh run or verify existence
    run_config_path = output_dir / "run_config.json"
    if not args.resume or not run_config_path.exists():
        trainable_count = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in student_model.parameters() if not p.requires_grad)
        model_cfg_dict = dataclasses.asdict(student_model.config)

        run_config = {
            "trainable_parameter_count": trainable_count,
            "frozen_parameter_count": frozen_count,
            "model_config": model_cfg_dict,
            "train_data_contract": train_data_contract,
            "val_data_contract": val_data_contract,
            "training_contract": training_contract,
            "base_metadata": base_metadata,
            "fa2_diagnostic": fa2_diag,
            "device": str(args.device),
            "seed": args.seed,
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        }
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2)

    # Line-buffered metrics jsonl path
    metrics_log_path = output_dir / "train_metrics.jsonl"
    metrics_file = open(metrics_log_path, "a", buffering=1)

    # If already reached max_updates, do not process new episodes
    if args.max_updates is not None and start_step >= args.max_updates:
        print(f"[Paused] Resume step {start_step} already reached max_updates={args.max_updates}. Exiting without processing.")
        summary_metrics = {
            "status": "paused_max_updates",
            "step": start_step,
            "epoch": start_epoch,
            "cursor": start_episode_cursor,
            "checkpoint": str(output_dir / "last.pt"),
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(summary_metrics, f, indent=2)
        metrics_file.close()
        return

    global_step = start_step
    first_step_saved = False
    start_time = time.time()
    student_model.train()

    epoch = start_epoch
    ep_cursor = start_episode_cursor

    print("[Train] Starting training loop...")
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        ep_cursor = start_episode_cursor if epoch == start_epoch else 0

        # Sampler start_cursor directly skips processed episodes without decoding them
        sampler = EpisodePermutationSampler(
            num_episodes=len(train_dataset),
            seed=args.seed,
            start_epoch=epoch,
            start_cursor=ep_cursor,
        )

        # DataLoader with multiprocessing spawn context and isolated generator
        loader_generator = torch.Generator().manual_seed(args.seed + epoch)
        loader_kwargs = {
            "dataset": train_dataset,
            "batch_size": None,
            "sampler": sampler,
            "collate_fn": identity_collate,
            "generator": loader_generator,
        }
        if args.num_workers > 0:
            import torch.multiprocessing as mp
            loader_kwargs["num_workers"] = args.num_workers
            loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
            loader_kwargs["persistent_workers"] = False

        loader = DataLoader(**loader_kwargs)

        accum_episodes_data: List[Dict[str, Any]] = []
        total_in_epoch = len(train_dataset)

        for ep_batch in loader:
            # We consumed this episode
            ep_cursor += 1
            accum_episodes_data.append(ep_batch)

            is_epoch_end = (ep_cursor >= total_in_epoch)
            if len(accum_episodes_data) >= args.episodes_per_step or (is_epoch_end and accum_episodes_data):
                num_accum = len(accum_episodes_data)
                optimizer.zero_grad()

                step_gt_loss = 0.0
                step_kd_loss = 0.0
                step_total_loss = 0.0
                step_decisions = 0
                step_ep_ids = []

                # Accumulate gradients across complete episodes
                for ep_data in accum_episodes_data:
                    ep_id = ep_data["episode_id"]
                    prompt = ep_data["prompt"]
                    chunks = ep_data["chunks"]
                    step_ep_ids.append(ep_id)

                    num_ep_decisions = len(chunks)
                    step_decisions += num_ep_decisions

                    delta_state = None
                    tbptt_accum_loss = 0.0
                    tbptt_count = 0

                    for c_info in chunks:
                        images_window = c_info["images"]
                        frame_ids = c_info["frame_ids"]
                        state = c_info["state"].to(args.device)
                        state_mask = c_info["state_mask"].to(args.device)
                        state = state * state_mask

                        actions = c_info["actions"].to(args.device)
                        action_mask = c_info["action_mask"].to(args.device)

                        student_deep, student_shallow, delta_state = student_model.forward_delta(
                            images_window=images_window,
                            frame_ids=frame_ids,
                            prompt=prompt,
                            previous=delta_state,
                        )

                        last_images = images_window[-1]
                        img_mask = torch.ones(len(last_images), dtype=torch.bool, device=args.device)

                        with torch.no_grad():
                            teacher_out = teacher_policy.get_vl_embeddings(
                                images=last_images,
                                image_mask=img_mask,
                                prompt=prompt,
                                return_cls_only=False,
                                shallow_layer_index=student_model.config.shallow_layer,
                            )
                            if isinstance(teacher_out, tuple):
                                teacher_deep, teacher_shallow = teacher_out
                            else:
                                teacher_deep = teacher_out
                                teacher_shallow = None

                        student_head = student_model.policy.action_head
                        loss, gt_l, kd_l, _, _ = compute_flow_kd_loss(
                            student_head=student_head,
                            teacher_head=teacher_head,
                            student_deep=student_deep,
                            student_shallow=student_shallow,
                            teacher_deep=teacher_deep,
                            teacher_shallow=teacher_shallow,
                            state=state,
                            actions=actions,
                            action_mask=action_mask,
                            kd_weight=args.kd_weight,
                        )

                        norm_loss = loss / float(num_ep_decisions * num_accum)
                        tbptt_accum_loss = tbptt_accum_loss + norm_loss
                        tbptt_count += 1

                        step_gt_loss += float(gt_l.item()) / float(num_ep_decisions * num_accum)
                        step_kd_loss += float(kd_l.item()) / float(num_ep_decisions * num_accum)
                        step_total_loss += float(loss.item()) / float(num_ep_decisions * num_accum)

                        if tbptt_count >= args.tbptt_decisions:
                            tbptt_accum_loss.backward()
                            tbptt_accum_loss = 0.0
                            tbptt_count = 0
                            delta_state = delta_state.detached()

                    if tbptt_count > 0:
                        tbptt_accum_loss.backward()
                        delta_state = delta_state.detached()

                # Record delta memory state metrics before clearing state
                if delta_state is not None:
                    memory_bytes = delta_state.nbytes
                    frame_count = delta_state.frame_count
                    state_norm = float(
                        torch.sqrt(
                            sum(torch.sum(m.float() ** 2) for m in delta_state.matrices)
                        ).item()
                    )
                else:
                    memory_bytes = 0
                    frame_count = 0
                    state_norm = 0.0

                step_observed_frames = sum(
                    sum(len(c_info.get("frame_ids", [])) for c_info in ep_data.get("chunks", []))
                    for ep_data in accum_episodes_data
                )

                # Collect memory gate, write_logits, and adapter parameter gradient norms without CPU copies
                memory_gate_grads = {}
                write_logits_grads = {}
                for name, p in student_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        lower_name = name.lower()
                        if any(k in lower_name for k in ("k_proj", "v_proj", "write_logits", "memory_gate", "mem")):
                            g_norm = float(p.grad.norm().item())
                            if "memory_gate" in name:
                                memory_gate_grads[name] = g_norm
                            elif "write_logits" in name:
                                write_logits_grads[name] = g_norm

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in student_model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip_norm,
                    error_if_nonfinite=True,
                )

                # Track lr used for this step
                lr_used = scheduler.get_last_lr()[0]

                optimizer.step()
                scheduler.step()
                global_step += 1

                lr_next = scheduler.get_last_lr()[0]

                # Clear old state immediately after optimizer step
                accum_episodes_data.clear()
                delta_state = None

                elapsed = time.time() - start_time
                target_dev = torch.device(args.device)
                if target_dev.type == "cuda" and torch.cuda.is_available():
                    cuda_peak_mb = torch.cuda.max_memory_allocated(target_dev) / (1024 * 1024)
                else:
                    cuda_peak_mb = 0.0

                metric_record = {
                    "step": global_step,
                    "epoch": epoch,
                    "next_episode_cursor": ep_cursor,
                    "loss": step_total_loss,
                    "gt_loss": step_gt_loss,
                    "kd_loss": step_kd_loss,
                    "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "lr_used": lr_used,
                    "lr_next": lr_next,
                    "elapsed_sec": round(elapsed, 2),
                    "decisions": step_decisions,
                    "observed_frames": step_observed_frames,
                    "episode_ids": step_ep_ids,
                    "memory_bytes": memory_bytes,
                    "frame_count": frame_count,
                    "norm": state_norm,
                    "memory_norm": state_norm,
                    "cuda_peak_mb": round(cuda_peak_mb, 1),
                    "memory_gate_grads": memory_gate_grads,
                    "write_logits_grads": write_logits_grads,
                    "timestamp": time.time(),
                }

                # Validation
                if args.val_every > 0 and global_step % args.val_every == 0:
                    val_res = evaluate_delta(
                        student_model=student_model,
                        teacher_policy=teacher_policy,
                        teacher_head=teacher_head,
                        val_dataset=val_dataset,
                        device=args.device,
                        val_episodes=args.val_episodes,
                        kd_weight=args.kd_weight,
                        seed=args.seed,
                    )
                    metric_record.update(val_res)
                    print(f"[Val Step {global_step}] {val_res}")

                # Line-buffered write
                metrics_file.write(json.dumps(metric_record) + "\n")

                print(
                    f"[Step {global_step}] Ep: {epoch} (cur={ep_cursor}/{total_in_epoch}) | "
                    f"Loss: {step_total_loss:.4f} (GT: {step_gt_loss:.4f}, KD: {step_kd_loss:.4f}) | "
                    f"Grad: {metric_record['grad_norm']:.4f} | LR: {lr_used:.2e} -> {lr_next:.2e} | "
                    f"Decisions: {step_decisions} | Peak: {cuda_peak_mb:.1f}MB"
                )

                if not first_step_saved:
                    first_step_path = output_dir / "last.pt"
                    save_delta_checkpoint(
                        output_path=first_step_path,
                        student_model=student_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        next_episode_cursor=ep_cursor,
                        stage=args.stage,
                        args=args,
                        norm_stats=norm_stats,
                        base_metadata=base_metadata,
                        train_data_contract=train_data_contract,
                        val_data_contract=val_data_contract,
                        training_contract=training_contract,
                        init_provenance=init_provenance,
                    )
                    first_step_saved = True

                elif global_step % args.save_every == 0:
                    last_path = output_dir / "last.pt"
                    save_delta_checkpoint(
                        output_path=last_path,
                        student_model=student_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        next_episode_cursor=ep_cursor,
                        stage=args.stage,
                        args=args,
                        norm_stats=norm_stats,
                        base_metadata=base_metadata,
                        train_data_contract=train_data_contract,
                        val_data_contract=val_data_contract,
                        training_contract=training_contract,
                        init_provenance=init_provenance,
                    )

                if args.max_updates is not None and global_step >= args.max_updates:
                    print(f"[Paused] Reached max_updates={args.max_updates}. Exiting training loop.")
                    break

        # Save milestone checkpoint at end of epoch ONLY IF cursor actually completed full epoch
        if ep_cursor == total_in_epoch:
            epoch_ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_delta_checkpoint(
                output_path=epoch_ckpt_path,
                student_model=student_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch + 1,
                next_episode_cursor=0,
                stage=args.stage,
                args=args,
                norm_stats=norm_stats,
                base_metadata=base_metadata,
                train_data_contract=train_data_contract,
                val_data_contract=val_data_contract,
                training_contract=training_contract,
                init_provenance=init_provenance,
            )
            print(f"[Milestone] Saved epoch milestone to {epoch_ckpt_path}")

        if args.max_updates is not None and global_step >= args.max_updates:
            break

    # Save final checkpoint and last.pt
    # If stopped prematurely by max_updates, epoch and cursor reflect actual processed state
    final_epoch = (epoch + 1) if (ep_cursor == total_in_epoch) else epoch
    final_cursor = 0 if (ep_cursor == total_in_epoch) else ep_cursor
    paused_by_max_updates = (
        args.max_updates is not None and global_step >= args.max_updates and final_epoch < args.epochs
    )

    for p in [output_dir / "final.pt", output_dir / "last.pt"]:
        save_delta_checkpoint(
            output_path=p,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            epoch=final_epoch,
            next_episode_cursor=final_cursor,
            stage=args.stage,
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_metadata,
            train_data_contract=train_data_contract,
            val_data_contract=val_data_contract,
            training_contract=training_contract,
            init_provenance=init_provenance,
        )

    metrics_file.close()

    status = "paused_max_updates" if paused_by_max_updates else "completed"
    final_ckpt_path = output_dir / ("last.pt" if paused_by_max_updates else "final.pt")
    summary_metrics = {
        "status": status,
        "step": global_step,
        "epoch": final_epoch,
        "cursor": final_cursor,
        "checkpoint": str(final_ckpt_path),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    if paused_by_max_updates:
        print(f"[Paused] Training paused at max_updates={args.max_updates}. Saved last.pt and metrics to {output_dir}")
    else:
        print(f"[Done] Training complete. Saved final.pt and metrics to {output_dir}")


if __name__ == "__main__":
    main()
