"""Third-stage joint fine-tuning for FabriVLA with learned predictive memory.

Unfreezes the base policy (vision encoder, projector, language model, action head)
while jointly training the learned causal memory writer and future latent head.
Maintains a frozen visual teacher (deepcopy of vision encoder + projector) to
extract future observation latent targets without leaking gradients.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.predictive_memory import WriterConfig
from fabri_moss.runtime import assert_native_fa2, compute_file_sha256
from fabri_moss.train_native import (
    SegmentSequenceSampler,
    bucketed_gradient_allreduce,
    check_stop_file_requested,
    classify_parameter,
    clip_parameter_groups_norm,
    compute_epoch_batches,
    identity_collate,
)
from fabri_moss.train_predictive import (
    create_predictive_optimizer_and_scheduler,
    evaluate_predictive,
    load_predictive_checkpoint,
    validate_predictive_cursor,
)

if TYPE_CHECKING:
    from fabri_moss.joint_predictive_policy import JointPredictiveMemoryPolicy
    from fabri_moss.predictive_data import PredictiveTrainingDataset


def compute_teacher_hash(teacher: nn.Module) -> str:
    """Compute deterministic SHA256 over all frozen teacher parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(teacher.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def create_joint_optimizer_and_scheduler(
    model: JointPredictiveMemoryPolicy,
    lr_vision: float = 5e-7,
    lr_projector: float = 2.5e-6,
    lr_llm: float = 1e-6,
    lr_head: float = 5e-6,
    lr_writer: float = 1e-5,
    lr_future: float = 1e-5,
    weight_decay: float = 1e-4,
    total_steps: int = 1000,
    warmup_steps: int = 100,
) -> Tuple[AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Create 6-group AdamW optimizer and cosine scheduler with warmup for joint stage 3.

    Parameter groups:
    1. vision: embedder.model.vision_model -> lr_vision
    2. projector: embedder.model.mlp1 -> lr_projector
    3. llm: embedder.model.language_model -> lr_llm
    4. head: action_head -> lr_head
    5. writer: model.writer -> lr_writer
    6. future: model.future_head -> lr_future

    model.future_teacher is strictly excluded.
    1D params, norm, bias have weight_decay=0.0.
    """
    lr_map = {
        "vision": lr_vision,
        "projector": lr_projector,
        "llm": lr_llm,
        "head": lr_head,
        "writer": lr_writer,
        "future": lr_future,
    }

    buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {
        (g, nd): [] for g in ("vision", "projector", "llm", "head", "writer", "future") for nd in (False, True)
    }

    # Classify base policy parameters
    for name, param in model.policy.named_parameters():
        if not param.requires_grad:
            continue
        group = classify_parameter(name)
        is_no_decay = (param.ndim <= 1) or ("bias" in name) or ("norm" in name)
        buckets[(group, is_no_decay)].append(param)

    # Classify writer parameters
    for name, param in model.writer.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.ndim <= 1) or ("bias" in name) or ("norm" in name)
        buckets[("writer", is_no_decay)].append(param)

    # Classify future head parameters
    for name, param in model.future_head.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.ndim <= 1) or ("bias" in name) or ("norm" in name)
        buckets[("future", is_no_decay)].append(param)

    param_groups: List[Dict[str, Any]] = []
    for (group_name, is_no_decay), plist in buckets.items():
        if not plist:
            continue
        param_groups.append({
            "params": plist,
            "lr": lr_map[group_name],
            "weight_decay": 0.0 if is_no_decay else weight_decay,
            "group_name": group_name,
            "no_decay": is_no_decay,
            "initial_lr": lr_map[group_name],
        })

    if not param_groups:
        raise ValueError("No trainable parameters found for joint optimizer.")

    optimizer = AdamW(param_groups)

    def lr_lambda(current_step: int) -> float:
        step_num = current_step + 1
        if warmup_steps > 0 and step_num <= warmup_steps:
            return float(step_num) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(0.0, progress), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def build_joint_checkpoint_payload(
    model: JointPredictiveMemoryPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    update: int,
    epoch: int,
    batch_cursor: int,
    epoch_targets_seen: int,
    parent_global_step: int,
    stage2_source_path: str,
    stage2_source_sha256: str,
    stage2_global_step: int,
    all_rng_states: List[Dict[str, Any]],
    raw_config: Dict[str, Any],
    raw_norm_stats: Dict[str, Any],
    train_contract: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    source_module_fingerprints: Dict[str, Any],
    group_update_checks: Dict[str, bool],
    teacher_hash_match: bool,
) -> Dict[str, Any]:
    """Assemble joint predictive checkpoint matching predictive_memory_joint_v1 format."""
    return {
        "format": "predictive_memory_joint_v1",
        "model": {k: v.cpu() for k, v in model.policy.state_dict().items()},
        "config": raw_config,
        "norm_stats": raw_norm_stats,
        "writer": {k: v.cpu() for k, v in model.writer.state_dict().items()},
        "future_head": {k: v.cpu() for k, v in model.future_head.state_dict().items()},
        "teacher": {k: v.cpu() for k, v in model.future_teacher.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update": int(update),
        "global_step": int(stage2_global_step + update),
        "epoch": int(epoch),
        "batch_cursor": int(batch_cursor),
        "epoch_targets_seen": int(epoch_targets_seen),
        "parent_global_step": int(parent_global_step),
        "stage2_source_path": str(stage2_source_path),
        "stage2_source_sha256": str(stage2_source_sha256),
        "stage2_global_step": int(stage2_global_step),
        "rng_states_per_rank": all_rng_states,
        "world_size": len(all_rng_states),
        "training_contract": train_contract,
        "train_data_contract": train_data_contract,
        "val_data_contract": val_data_contract,
        "source_module_fingerprints": source_module_fingerprints,
        "group_update_checks": group_update_checks,
        "teacher_hash_match": bool(teacher_hash_match),
    }


def save_joint_checkpoint(
    output_dir: Path,
    filename: str,
    model: JointPredictiveMemoryPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    update: int,
    epoch: int,
    batch_cursor: int,
    epoch_targets_seen: int,
    parent_global_step: int,
    stage2_source_path: str,
    stage2_source_sha256: str,
    stage2_global_step: int,
    raw_config: Dict[str, Any],
    raw_norm_stats: Dict[str, Any],
    train_contract: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    source_module_fingerprints: Dict[str, Any],
    group_update_checks: Dict[str, bool],
    teacher_hash_match: bool,
    device: Optional[torch.device] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    update_last: bool = True,
) -> Path:
    """Atomically save joint predictive checkpoint on rank 0."""
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    dev_idx = device.index if (device is not None and device.type == "cuda") else None
    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(dev_idx) if (device is not None and device.type == "cuda" and torch.cuda.is_available()) else None,
    }
    all_rngs = [None] * world_size if rank == 0 else None
    if is_dist and world_size > 1:
        dist.gather_object(local_rng, all_rngs if rank == 0 else None, dst=0, group=process_group)
    else:
        all_rngs = [local_rng]

    out_file = output_dir / filename
    if rank == 0:
        payload = build_joint_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            update=update,
            epoch=epoch,
            batch_cursor=batch_cursor,
            epoch_targets_seen=epoch_targets_seen,
            parent_global_step=parent_global_step,
            stage2_source_path=stage2_source_path,
            stage2_source_sha256=stage2_source_sha256,
            stage2_global_step=stage2_global_step,
            all_rng_states=all_rngs,
            raw_config=raw_config,
            raw_norm_stats=raw_norm_stats,
            train_contract=train_contract,
            train_data_contract=train_data_contract,
            val_data_contract=val_data_contract,
            source_module_fingerprints=source_module_fingerprints,
            group_update_checks=group_update_checks,
            teacher_hash_match=teacher_hash_match,
        )
        tmp_file = output_dir / f".tmp_{filename}_{os.getpid()}_{int(time.time()*1000)}"
        torch.save(payload, str(tmp_file))
        os.replace(str(tmp_file), str(out_file))
        if update_last and filename != "last.pt":
            last_file = output_dir / "last.pt"
            tmp_last = output_dir / f".tmp_last_{os.getpid()}_{int(time.time()*1000)}"
            try:
                os.link(str(out_file), str(tmp_last))
            except OSError:
                shutil.copyfile(str(out_file), str(tmp_last))
            os.replace(str(tmp_last), str(last_file))

    if is_dist and world_size > 1:
        dist.barrier(group=process_group)
    return out_file


def parse_args(raw_args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for train_joint_predictive."""
    parser = argparse.ArgumentParser(description="Third-stage joint training for FabriVLA and learned predictive memory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init-from-stage2", type=str, default=None, help="Stage 2 completed checkpoint path (e.g. stage2.pt).")
    group.add_argument("--resume", type=str, default=None, help="Joint predictive checkpoint path to resume from (e.g. last.pt).")

    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for joint checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=10, help="Target epoch ceiling (default: 10).")
    parser.add_argument("--max-updates", type=int, default=None, help="Optional total update ceiling (for 2-step gate or testing).")
    parser.add_argument("--save-every", type=int, default=250, help="Save interval in updates.")
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluation interval in updates.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--global-batch-size", type=int, default=8, help="Global batch size across all ranks.")
    parser.add_argument("--seed", type=int, default=4042, help="Random seed.")
    parser.add_argument("--warmup-updates", type=int, default=100, help="Warmup updates for scheduler.")
    parser.add_argument("--future-weight", type=float, default=0.001, help="Future latent loss weight.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm clipping.")

    # Peak learning rates for 6 groups
    parser.add_argument("--lr-vision", type=float, default=5e-7, help="Learning rate for vision model.")
    parser.add_argument("--lr-projector", type=float, default=2.5e-6, help="Learning rate for projector (mlp1).")
    parser.add_argument("--lr-llm", type=float, default=1e-6, help="Learning rate for language model.")
    parser.add_argument("--lr-head", type=float, default=5e-6, help="Learning rate for action head.")
    parser.add_argument("--lr-writer", type=float, default=1e-5, help="Learning rate for memory writer.")
    parser.add_argument("--lr-future", type=float, default=1e-5, help="Learning rate for future head.")

    parser.add_argument("--stop-file", type=str, default=None, help="Path to stop sentinel file.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--data-root", type=str, default="/root/evo1_metaworld_dataset")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA")
    parser.add_argument("--vlm-path", type=str, default="/root/models/InternVL3_5-1B")
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    return parser.parse_args(raw_args)


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate CLI arguments strictly before runtime execution."""
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    for name in ("global_batch_size", "save_every", "eval_every", "warmup_updates"):
        val = getattr(args, name)
        if val <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {val}")
    for name in ("lr_vision", "lr_projector", "lr_llm", "lr_head", "lr_writer", "lr_future", "weight_decay", "grad_clip", "future_weight"):
        val = getattr(args, name)
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite, got {val}")
    if args.workers < 0:
        raise ValueError(f"--workers must be non-negative, got {args.workers}")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError(f"--val-fraction must be in (0, 1), got {args.val_fraction}")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError(f"--max-updates must be positive, got {args.max_updates}")
    if args.stop_file is not None and not str(args.stop_file).strip():
        raise ValueError("--stop-file cannot be empty")


def run_training(args: argparse.Namespace, device: Optional[torch.device] = None) -> None:
    """Execute third-stage joint fine-tuning runtime."""
    validate_cli_arguments(args)
    output_dir = Path(args.output_dir).resolve()
    run_config_path = output_dir / "run_config.json"

    # Reject non-empty output directory unless resuming
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is non-empty, but --resume was not specified."
            )
        if not run_config_path.exists():
            raise ValueError(
                f"Cannot resume in {output_dir}: missing run_config.json provenance file."
            )

    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1

    if device is None:
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is strictly required for formal joint run_training execution.")
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)

    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed and world_size > 1:
        dist.barrier()

    metrics_file = None
    try:
        per_rank_seed = args.seed + rank * 10007
        torch.manual_seed(per_rank_seed)
        random.seed(per_rank_seed)
        np.random.seed(per_rank_seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(per_rank_seed)

        src_dir = Path(__file__).resolve().parent
        current_source_fingerprints = {
            "joint_predictive_policy.py": compute_file_sha256(src_dir / "joint_predictive_policy.py"),
            "train_joint_predictive.py": compute_file_sha256(src_dir / "train_joint_predictive.py"),
            "predictive_memory.py": compute_file_sha256(src_dir / "predictive_memory.py"),
            "predictive_policy.py": compute_file_sha256(src_dir / "predictive_policy.py"),
            "predictive_data.py": compute_file_sha256(src_dir / "predictive_data.py"),
        }

        import fabri_moss.joint_predictive_policy as joint_predictive_policy
        import fabri_moss.predictive_data as predictive_data
        import fabri_moss.runtime as runtime

        JointPredictiveMemoryPolicy = joint_predictive_policy.JointPredictiveMemoryPolicy
        PredictiveTrainingDataset = predictive_data.PredictiveTrainingDataset
        load_native_checkpoint = runtime.load_native_checkpoint

        if args.init_from_stage2:
            stage2_path = Path(args.init_from_stage2).resolve()
            if not stage2_path.exists():
                raise FileNotFoundError(f"Stage 2 checkpoint not found: {stage2_path}")
            stage2_sha256 = compute_file_sha256(stage2_path)
            stage2_ckpt = torch.load(str(stage2_path), map_location="cpu", weights_only=False)

            # Strict stage 2 format and state check
            if stage2_ckpt.get("format") != "predictive_memory_adapter_v1":
                raise ValueError(f"Expected format 'predictive_memory_adapter_v1', got {stage2_ckpt.get('format')!r}")
            if stage2_ckpt.get("stage") != "stage2_joint_future":
                raise ValueError(f"Stage 2 checkpoint stage must be 'stage2_joint_future', got {stage2_ckpt.get('stage')!r}")

            s2_tc = stage2_ckpt.get("training_contract")
            if not isinstance(s2_tc, dict):
                raise ValueError("Stage 2 training_contract missing or not a dict")

            if "writer_updates" not in s2_tc or "joint_updates" not in s2_tc:
                raise ValueError("Stage 2 training_contract missing writer_updates or joint_updates")

            s2_writer_updates = s2_tc["writer_updates"]
            s2_joint_updates = s2_tc["joint_updates"]
            if isinstance(s2_writer_updates, bool) or not isinstance(s2_writer_updates, int) or s2_writer_updates < 0:
                raise ValueError(f"Invalid writer_updates: {s2_writer_updates!r}")
            if isinstance(s2_joint_updates, bool) or not isinstance(s2_joint_updates, int) or s2_joint_updates <= 0:
                raise ValueError(f"Invalid joint_updates: {s2_joint_updates!r}")

            expected_s2_updates = s2_writer_updates + s2_joint_updates
            if stage2_ckpt.get("update") != expected_s2_updates:
                raise ValueError(f"Stage 2 checkpoint update {stage2_ckpt.get('update')} != {expected_s2_updates}")

            parent_global_step = int(stage2_ckpt["parent_global_step"])
            stage2_global_step = int(stage2_ckpt["global_step"])
            if stage2_global_step != parent_global_step + expected_s2_updates:
                raise ValueError(f"Stage 2 global_step {stage2_global_step} != parent_global_step {parent_global_step} + update {expected_s2_updates}")

            if s2_tc.get("seed") != args.seed:
                raise ValueError(f"Stage 2 training contract seed {s2_tc.get('seed')} != args.seed {args.seed}")
            if s2_tc.get("global_batch_size") != args.global_batch_size:
                raise ValueError(f"Stage 2 global_batch_size {s2_tc.get('global_batch_size')} != args.global_batch_size {args.global_batch_size}")
            if s2_tc.get("world_size") != world_size:
                raise ValueError(f"Stage 2 world_size {s2_tc.get('world_size')} != current world_size {world_size}")

            parent_path = Path(stage2_ckpt["parent_path"]).resolve()
            if not parent_path.exists():
                raise FileNotFoundError(f"Parent checkpoint not found: {parent_path}")
            parent_sha256 = compute_file_sha256(parent_path)
            if parent_sha256 != stage2_ckpt.get("parent_sha256"):
                raise ValueError(f"Parent SHA256 mismatch: {parent_sha256} vs {stage2_ckpt.get('parent_sha256')}")

            # Verify stage 2 run_config.json
            s2_run_cfg_path = stage2_path.parent / "run_config.json"
            if not s2_run_cfg_path.exists():
                raise ValueError(f"Stage 2 run_config.json not found in {stage2_path.parent}")
            s2_run_cfg = json.loads(s2_run_cfg_path.read_text())
            if s2_run_cfg.get("parent_sha256") != parent_sha256:
                raise ValueError("Stage 2 run_config parent SHA256 mismatch")
            if s2_run_cfg.get("training_contract") != s2_tc:
                raise ValueError("Stage 2 run_config training_contract mismatch")
            if s2_run_cfg.get("source_module_fingerprints") != stage2_ckpt.get("source_module_fingerprints"):
                raise ValueError("Stage 2 run_config source_module_fingerprints mismatch")
            if s2_run_cfg.get("train_data_contract") != stage2_ckpt.get("train_data_contract"):
                raise ValueError("Stage 2 run_config train_data_contract mismatch")
            if s2_run_cfg.get("val_data_contract") != stage2_ckpt.get("val_data_contract"):
                raise ValueError("Stage 2 run_config val_data_contract mismatch")

            # Load parent payload to preserve raw config and raw norm_stats
            parent_payload = torch.load(str(parent_path), map_location="cpu", weights_only=False, mmap=True)
            raw_config = parent_payload.get("config", {})
            raw_norm_stats = parent_payload.get("norm_stats", {})
            del parent_payload

            base_policy, _, selected_norm_stats, _ = load_native_checkpoint(
                checkpoint_path=str(parent_path),
                fabri_root=args.fabri_root,
                vlm_path=args.vlm_path,
                device=str(device),
                trainable=True,
            )
            if device.type == "cuda":
                assert_native_fa2(base_policy)

            train_dataset = PredictiveTrainingDataset(
                root=args.data_root,
                norm_stats=selected_norm_stats,
                split="train",
                history_frames=16,
                target_frames=8,
                augmentation=False,
                seed=args.seed,
                val_fraction=args.val_fraction,
                max_episodes=args.max_train_episodes,
            )
            val_dataset = PredictiveTrainingDataset(
                root=args.data_root,
                norm_stats=selected_norm_stats,
                split="val",
                history_frames=16,
                target_frames=8,
                augmentation=False,
                seed=args.seed,
                val_fraction=args.val_fraction,
                max_episodes=args.max_val_episodes,
            )
            train_data_contract = train_dataset.get_data_contract()
            val_data_contract = val_dataset.get_data_contract()

            if stage2_ckpt.get("train_data_contract") != train_data_contract:
                raise ValueError("Stage 2 train_data_contract mismatch with current environment")
            if stage2_ckpt.get("val_data_contract") != val_data_contract:
                raise ValueError("Stage 2 val_data_contract mismatch with current environment")

            start_epoch = int(stage2_ckpt["epoch"])
            start_cursor = int(stage2_ckpt["batch_cursor"])
            start_epoch_targets_seen = int(stage2_ckpt["epoch_targets_seen"])

            total_segments = len(train_dataset)
            if start_cursor == total_segments:
                start_epoch += 1
                start_cursor = 0
                start_epoch_targets_seen = 0

            if start_epoch >= args.epochs:
                raise ValueError(f"Start epoch {start_epoch} >= target epochs {args.epochs}")

            validate_predictive_cursor(
                total_segments, args.global_batch_size, args.seed,
                start_epoch, start_cursor, start_epoch_targets_seen, train_dataset,
            )

            rem_in_cur_epoch = math.ceil((total_segments - start_cursor) / args.global_batch_size)
            epoch_batches_count = math.ceil(total_segments / args.global_batch_size)
            total_stage3_budget = rem_in_cur_epoch + max(0, args.epochs - start_epoch - 1) * epoch_batches_count

            writer_cfg = stage2_ckpt["writer_config"] if isinstance(stage2_ckpt["writer_config"], WriterConfig) else WriterConfig(**stage2_ckpt["writer_config"])
            shallow_layer = s2_tc.get("shallow_layer", 6)
            gradient_checkpointing = s2_tc.get("gradient_checkpointing", True)
            model = JointPredictiveMemoryPolicy(
                policy=base_policy,
                writer_config=writer_cfg,
                shallow_layer=shallow_layer,
                gradient_checkpointing=gradient_checkpointing,
            ).to(device)

            temp_opt, temp_sched = create_predictive_optimizer_and_scheduler(
                nn.ModuleList([model.writer, model.future_head]),
                lr=s2_tc["lr"],
                weight_decay=s2_tc.get("weight_decay", 1e-4),
                warmup_steps=s2_tc.get("warmup_updates", 50),
            )
            load_predictive_checkpoint(
                checkpoint_path=stage2_path,
                writer_module=model.writer,
                future_head_module=model.future_head,
                optimizer=temp_opt,
                scheduler=temp_sched,
                expected_parent_sha256=parent_sha256,
                expected_training_contract=s2_tc,
                expected_train_data_contract=train_data_contract,
                expected_val_data_contract=val_data_contract,
                device=device,
                expected_source_module_fingerprints=stage2_ckpt.get("source_module_fingerprints"),
            )

            optimizer, scheduler = create_joint_optimizer_and_scheduler(
                model=model,
                lr_vision=args.lr_vision,
                lr_projector=args.lr_projector,
                lr_llm=args.lr_llm,
                lr_head=args.lr_head,
                lr_writer=args.lr_writer,
                lr_future=args.lr_future,
                weight_decay=args.weight_decay,
                total_steps=total_stage3_budget,
                warmup_steps=args.warmup_updates,
            )

            for p, state in temp_opt.state.items():
                if state:
                    transferred = {}
                    for k, v in state.items():
                        transferred[k] = v.clone() if isinstance(v, torch.Tensor) else v
                    optimizer.state[p] = transferred

            del temp_opt, temp_sched, stage2_ckpt

            completed_updates = 0
            current_epoch = start_epoch
            current_cursor = start_cursor
            epoch_targets_seen = start_epoch_targets_seen
            stage2_source_path_str = str(stage2_path)
            stage2_source_sha = stage2_sha256
            stage2_g_step = stage2_global_step
            p_global_step = parent_global_step
            group_update_checks = {g: False for g in ("vision", "projector", "llm", "head", "writer", "future")}

        else:
            # Resuming from joint checkpoint
            joint_ckpt_path = Path(args.resume).resolve()
            if not joint_ckpt_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {joint_ckpt_path}")
            joint_ckpt = torch.load(str(joint_ckpt_path), map_location="cpu", weights_only=False, mmap=True)

            # Strict metadata checks before modifying model
            if joint_ckpt.get("format") != "predictive_memory_joint_v1":
                raise ValueError(f"Expected format 'predictive_memory_joint_v1', got {joint_ckpt.get('format')!r}")

            required_joint_keys = (
                "format", "model", "config", "norm_stats", "writer", "future_head", "teacher",
                "optimizer", "scheduler", "update", "global_step", "epoch", "batch_cursor",
                "epoch_targets_seen", "parent_global_step", "stage2_source_path", "stage2_source_sha256",
                "stage2_global_step", "rng_states_per_rank", "world_size", "training_contract",
                "train_data_contract", "val_data_contract", "source_module_fingerprints",
            )
            for k in required_joint_keys:
                if k not in joint_ckpt:
                    raise ValueError(f"Missing required checkpoint key: '{k}'")

            def _check_nonnegative_int(val: Any, name: str) -> int:
                if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                    raise ValueError(f"{name} must be a non-negative integer, got {val!r}")
                return val

            completed_updates = _check_nonnegative_int(joint_ckpt["update"], "update")
            current_epoch = _check_nonnegative_int(joint_ckpt["epoch"], "epoch")
            current_cursor = _check_nonnegative_int(joint_ckpt["batch_cursor"], "batch_cursor")
            epoch_targets_seen = _check_nonnegative_int(joint_ckpt["epoch_targets_seen"], "epoch_targets_seen")
            p_global_step = _check_nonnegative_int(joint_ckpt["parent_global_step"], "parent_global_step")
            stage2_g_step = _check_nonnegative_int(joint_ckpt["stage2_global_step"], "stage2_global_step")
            ckpt_global_step = _check_nonnegative_int(joint_ckpt["global_step"], "global_step")
            ckpt_world_size = _check_nonnegative_int(joint_ckpt["world_size"], "world_size")

            if ckpt_world_size != world_size:
                raise ValueError(f"Resume world_size {world_size} != checkpoint world_size {ckpt_world_size}")

            rng_per_rank = joint_ckpt["rng_states_per_rank"]
            if not isinstance(rng_per_rank, list) or len(rng_per_rank) != world_size:
                raise ValueError(f"Resume rng_states_per_rank length != world_size {world_size}")

            if ckpt_global_step != stage2_g_step + completed_updates:
                raise ValueError(f"Checkpoint global_step {ckpt_global_step} != stage2_global_step {stage2_g_step} + update {completed_updates}")

            sched_state = joint_ckpt["scheduler"]
            if sched_state.get("last_epoch") != completed_updates:
                raise ValueError(f"Scheduler last_epoch {sched_state.get('last_epoch')} != checkpoint update {completed_updates}")

            if joint_ckpt.get("source_module_fingerprints") != current_source_fingerprints:
                raise ValueError("Resume source_module_fingerprints mismatch with current source fingerprints")

            resume_tc = joint_ckpt["training_contract"]
            if not isinstance(resume_tc, dict):
                raise ValueError("Checkpoint training_contract must be a dict")

            tc_expected_matches = {
                "target_epochs": args.epochs,
                "world_size": world_size,
                "seed": args.seed,
                "global_batch_size": args.global_batch_size,
                "lr_vision": args.lr_vision,
                "lr_projector": args.lr_projector,
                "lr_llm": args.lr_llm,
                "lr_head": args.lr_head,
                "lr_writer": args.lr_writer,
                "lr_future": args.lr_future,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "warmup_updates": args.warmup_updates,
                "future_weight": args.future_weight,
            }
            for tc_k, exp_v in tc_expected_matches.items():
                if tc_k not in resume_tc:
                    raise ValueError(f"Resume training_contract missing '{tc_k}'")
                if resume_tc[tc_k] != exp_v:
                    raise ValueError(f"Resume training_contract '{tc_k}' mismatch: {resume_tc[tc_k]!r} != {exp_v!r}")

            total_stage3_budget = resume_tc["total_stage3_budget"]
            if completed_updates > total_stage3_budget:
                raise ValueError(f"Completed updates {completed_updates} > total_stage3_budget {total_stage3_budget}")

            stage2_source_path_str = str(joint_ckpt["stage2_source_path"])
            stage2_source_sha = str(joint_ckpt["stage2_source_sha256"])
            s2_src_path = Path(stage2_source_path_str)
            if s2_src_path.exists():
                if compute_file_sha256(s2_src_path) != stage2_source_sha:
                    raise ValueError("Stage 2 source SHA256 mismatch on resume")

            rc_existing = json.loads(run_config_path.read_text())
            if rc_existing.get("training_contract") != resume_tc:
                raise ValueError("run_config training_contract mismatch with resume checkpoint")
            if rc_existing.get("train_data_contract") != joint_ckpt.get("train_data_contract"):
                raise ValueError("run_config train_data_contract mismatch with resume checkpoint")
            if rc_existing.get("val_data_contract") != joint_ckpt.get("val_data_contract"):
                raise ValueError("run_config val_data_contract mismatch with resume checkpoint")
            if rc_existing.get("source_module_fingerprints") != current_source_fingerprints:
                raise ValueError("run_config source_module_fingerprints mismatch with current source fingerprints")

            raw_config = joint_ckpt.get("config", {})
            raw_norm_stats = joint_ckpt.get("norm_stats", {})

            base_policy, _, selected_norm_stats, _ = load_native_checkpoint(
                checkpoint_path=str(joint_ckpt_path),
                fabri_root=args.fabri_root,
                vlm_path=args.vlm_path,
                device=str(device),
                trainable=True,
            )
            if device.type == "cuda":
                assert_native_fa2(base_policy)

            train_dataset = PredictiveTrainingDataset(
                root=args.data_root,
                norm_stats=selected_norm_stats,
                split="train",
                history_frames=16,
                target_frames=8,
                augmentation=False,
                seed=args.seed,
                val_fraction=args.val_fraction,
                max_episodes=args.max_train_episodes,
            )
            val_dataset = PredictiveTrainingDataset(
                root=args.data_root,
                norm_stats=selected_norm_stats,
                split="val",
                history_frames=16,
                target_frames=8,
                augmentation=False,
                seed=args.seed,
                val_fraction=args.val_fraction,
                max_episodes=args.max_val_episodes,
            )
            train_data_contract = train_dataset.get_data_contract()
            val_data_contract = val_dataset.get_data_contract()

            if joint_ckpt.get("train_data_contract") != train_data_contract:
                raise ValueError("Resume train_data_contract mismatch")
            if joint_ckpt.get("val_data_contract") != val_data_contract:
                raise ValueError("Resume val_data_contract mismatch")

            start_epoch = resume_tc["start_epoch"]
            start_cursor = resume_tc["start_cursor"]
            start_epoch_targets_seen = resume_tc["start_epoch_targets_seen"]

            validate_predictive_cursor(
                len(train_dataset), args.global_batch_size, args.seed,
                current_epoch, current_cursor, epoch_targets_seen, train_dataset,
            )

            writer_cfg = resume_tc["writer_config"] if isinstance(resume_tc["writer_config"], WriterConfig) else WriterConfig(**resume_tc["writer_config"])
            shallow_layer = resume_tc.get("shallow_layer", 6)
            gradient_checkpointing = resume_tc.get("gradient_checkpointing", True)
            model = JointPredictiveMemoryPolicy(
                policy=base_policy,
                writer_config=writer_cfg,
                shallow_layer=shallow_layer,
                gradient_checkpointing=gradient_checkpointing,
            ).to(device)

            model.writer.load_state_dict(joint_ckpt["writer"], strict=True)
            model.future_head.load_state_dict(joint_ckpt["future_head"], strict=True)
            model.future_teacher.load_state_dict(joint_ckpt["teacher"], strict=True)

            optimizer, scheduler = create_joint_optimizer_and_scheduler(
                model=model,
                lr_vision=args.lr_vision,
                lr_projector=args.lr_projector,
                lr_llm=args.lr_llm,
                lr_head=args.lr_head,
                lr_writer=args.lr_writer,
                lr_future=args.lr_future,
                weight_decay=args.weight_decay,
                total_steps=total_stage3_budget,
                warmup_steps=args.warmup_updates,
            )
            optimizer.load_state_dict(joint_ckpt["optimizer"])
            scheduler.load_state_dict(joint_ckpt["scheduler"])

            # Restore RNG states
            rng_per_rank = joint_ckpt["rng_states_per_rank"]
            my_rng = rng_per_rank[rank]
            random.setstate(my_rng["python"])
            np.random.set_state(my_rng["numpy"])
            torch.set_rng_state(my_rng["torch_cpu"].cpu() if isinstance(my_rng["torch_cpu"], torch.Tensor) else my_rng["torch_cpu"])
            dev_idx = device.index if (device is not None and device.type == "cuda") else None
            if my_rng.get("torch_cuda") is not None and device is not None and device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_rng_state(my_rng["torch_cuda"].cpu() if isinstance(my_rng["torch_cuda"], torch.Tensor) else my_rng["torch_cuda"], dev_idx)

            group_update_checks = joint_ckpt.get("group_update_checks", {g: False for g in ("vision", "projector", "llm", "head", "writer", "future")})
            del joint_ckpt

        validate_predictive_cursor(
            len(train_dataset), args.global_batch_size, args.seed,
            current_epoch, current_cursor, epoch_targets_seen, train_dataset,
        )

        initial_teacher_hash = compute_teacher_hash(model.future_teacher)
        total_segments = len(train_dataset)
        epoch_steps = math.ceil(total_segments / args.global_batch_size)
        consumed = ((current_epoch - start_epoch) * epoch_steps
                    + math.ceil(current_cursor / args.global_batch_size)
                    - math.ceil(start_cursor / args.global_batch_size))
        expected_budget = (math.ceil((total_segments - start_cursor) / args.global_batch_size)
                           + (args.epochs - start_epoch - 1) * epoch_steps)
        if consumed != completed_updates or expected_budget != total_stage3_budget:
            raise ValueError("Stage 3 update count or budget does not match the data cursor")
        if current_epoch > args.epochs or (current_epoch == args.epochs and current_cursor != 0):
            raise ValueError("Stage 3 cursor exceeds the epoch budget")
        training_contract = {
            "format": "predictive_memory_joint_v1",
            "parent_global_step": p_global_step,
            "stage2_source_path": stage2_source_path_str,
            "stage2_source_sha256": stage2_source_sha,
            "stage2_global_step": stage2_g_step,
            "lr_vision": args.lr_vision,
            "lr_projector": args.lr_projector,
            "lr_llm": args.lr_llm,
            "lr_head": args.lr_head,
            "lr_writer": args.lr_writer,
            "lr_future": args.lr_future,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "warmup_updates": args.warmup_updates,
            "total_stage3_budget": total_stage3_budget,
            "world_size": world_size,
            "seed": args.seed,
            "global_batch_size": args.global_batch_size,
            "future_weight": args.future_weight,
            "writer_config": dataclasses.asdict(model.writer.config) if hasattr(model.writer, "config") else {},
            "shallow_layer": model.shallow_layer,
            "gradient_checkpointing": model.gradient_checkpointing,
            "target_epochs": args.epochs,
            "start_epoch": start_epoch, "start_cursor": start_cursor,
            "start_epoch_targets_seen": start_epoch_targets_seen,
            "teacher_sha256": initial_teacher_hash,
            "temporal": get_compact_protocol_contract()["temporal"],
            "text_timestamps": False,
            "precision": "fp32_parameters_gradients_optimizer_bf16_vision_llm",
        }
        if args.resume and training_contract != resume_tc:
            raise ValueError("Resume full training contract mismatch")
        if (current_epoch == args.epochs and current_cursor == 0) or (args.max_updates is not None and completed_updates >= args.max_updates):
            return

        # Provenance run_config.json check and update
        if rank == 0:
            if run_config_path.exists() and args.resume:
                rc = json.loads(run_config_path.read_text())
                rc.setdefault("resume_events", []).append({
                    "resumed_from": str(args.resume), "update": completed_updates,
                    "epoch": current_epoch, "cursor": current_cursor, "timestamp": time.time(),
                })
                run_config_path.write_text(json.dumps(rc, indent=2))
            elif not args.resume:
                rc = {
                    "training_contract": training_contract,
                    "train_data_contract": train_data_contract,
                    "val_data_contract": val_data_contract,
                    "source_module_fingerprints": current_source_fingerprints,
                    "stage2_source_path": stage2_source_path_str,
                    "stage2_source_sha256": stage2_source_sha,
                    "stage2_global_step": stage2_g_step,
                    "parent_global_step": p_global_step,
                    "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                    "resume_events": [],
                }
                run_config_path.write_text(json.dumps(rc, indent=2))

        # Ensure all model trainable except teacher
        for p in model.future_teacher.parameters():
            p.requires_grad = False
        for p in model.policy.parameters():
            p.requires_grad = True
        for p in model.writer.parameters():
            p.requires_grad = True
        for p in model.future_head.parameters():
            p.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Validation indices up to 8
        val_dataset.set_epoch(current_epoch)
        raw_val_indices = val_dataset.validation_indices()
        val_indices = [raw_val_indices[i] for i in np.linspace(
            0, len(raw_val_indices) - 1, min(8, len(raw_val_indices)), dtype=int
        )] if raw_val_indices else []

        metrics_file = (output_dir / "metrics.jsonl").open("a") if rank == 0 else None

        group_candidates = {g: [] for g in group_update_checks}
        for group in optimizer.param_groups:
            group_candidates[group["group_name"]].extend(p for p in group["params"] if p.ndim > 1)
        for candidates in group_candidates.values():
            candidates.sort(key=lambda p: p.numel())

        stop_requested = False
        cursor = current_cursor
        total_segments = len(train_dataset)

        while (current_epoch < args.epochs) and not stop_requested:
            if args.max_updates is not None and completed_updates >= args.max_updates:
                break

            train_dataset.set_epoch(current_epoch)
            _, epoch_batches, _ = compute_epoch_batches(
                total_segments, args.global_batch_size, world_size, args.seed, current_epoch,
            )

            batch_start_idx = math.ceil(cursor / args.global_batch_size)
            remaining_batches = epoch_batches[batch_start_idx:]

            loader_generator = torch.Generator()
            loader_generator.manual_seed(args.seed + current_epoch * 10007 + rank * 997)

            rank_remaining_indices = [idx for bslice in remaining_batches for idx in bslice[rank::world_size]]
            loader = DataLoader(
                train_dataset, batch_size=None, sampler=SegmentSequenceSampler(rank_remaining_indices),
                num_workers=args.workers, collate_fn=identity_collate,
                multiprocessing_context="spawn" if (args.workers > 0 and sys.platform != "win32") else None,
                prefetch_factor=1 if args.workers > 0 else None,
                generator=loader_generator,
            )
            loader_iter = iter(loader)

            for batch_slice in remaining_batches:
                if (args.max_updates is not None and completed_updates >= args.max_updates) or check_stop_file_requested(args.stop_file, device):
                    stop_requested = True
                    break

                if model.future_teacher.training or any(p.requires_grad or p.grad is not None for p in model.future_teacher.parameters()):
                    raise RuntimeError("Frozen teacher invariant violated")

                cursor += len(batch_slice)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                # Joint policy train() sets policy to train and teacher to eval
                model.future_teacher.eval()

                local_targets_sum, local_loss_sum, local_has_grad = 0, 0.0, False
                local_action_sum, local_future_sum, local_future_valid = 0.0, 0.0, 0
                step_start_time = time.time()

                for _ in range(len(batch_slice[rank::world_size])):
                    sample = next(loader_iter)
                    target_count = int(sample["target_count"])
                    sample_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}

                    out = model(sample_dev, future_weight=args.future_weight)
                    loss_sum_t = out["loss_sum"]
                    loss_val = float(loss_sum_t.item())

                    if loss_sum_t.requires_grad:
                        loss_sum_t.backward()
                        local_has_grad = True

                    local_loss_sum += loss_val
                    local_action_sum += float(out["action_loss"].detach()) * target_count
                    local_future_sum += float(out["future_loss"].detach()) * target_count
                    local_future_valid += int(out["future_valid_count"])
                    local_targets_sum += target_count

                if is_distributed and world_size > 1:
                    flag_t = torch.tensor([local_targets_sum, int(local_has_grad), local_loss_sum,
                                           local_action_sum, local_future_sum, local_future_valid], dtype=torch.float64, device=device)
                    dist.all_reduce(flag_t, op=dist.ReduceOp.SUM)
                    global_targets, global_has_grad, global_loss_sum = int(flag_t[0].item()), int(flag_t[1].item()) > 0, float(flag_t[2].item())
                    global_action_sum, global_future_sum, global_future_valid = flag_t[3:].tolist()
                else:
                    global_targets, global_has_grad, global_loss_sum = local_targets_sum, local_has_grad, local_loss_sum
                    global_action_sum, global_future_sum, global_future_valid = local_action_sum, local_future_sum, local_future_valid

                expected_targets = sum(train_dataset.segments[i][2] - train_dataset.segments[i][1] for i in batch_slice)
                if global_targets <= 0 or global_targets != expected_targets:
                    raise ValueError(f"Batch target count {global_targets} != expected {expected_targets}")
                if not all(math.isfinite(x) for x in (global_loss_sum, global_action_sum, global_future_sum)):
                    raise FloatingPointError(f"Non-finite loss encountered in joint training: {global_loss_sum}")

                epoch_targets_seen += global_targets

                if not global_has_grad:
                    raise RuntimeError(f"Joint stage 3 batch at cursor {cursor} unexpectedly produced no gradients.")

                # Assert teacher parameters still frozen with no grad
                for tp in model.future_teacher.parameters():
                    if tp.grad is not None or tp.requires_grad:
                        raise RuntimeError("Teacher parameter invariant violated: requires_grad or grad is not None")

                bucketed_gradient_allreduce(trainable_params, global_target_count=global_targets)
                grad_norm = clip_parameter_groups_norm(trainable_params, max_norm=args.grad_clip)

                representatives = {}
                group_diagnostics = {}
                if completed_updates < 2:
                    for g, candidates in group_candidates.items():
                        for p in candidates:
                            if p.grad is not None and torch.count_nonzero(p.grad).item() > 0:
                                representatives[g] = (p, p.detach().clone())
                                group_diagnostics[g] = {"gradient_norm": float(p.grad.norm()), "dtype": str(p.grad.dtype)}
                                break
                optimizer.step()
                scheduler.step()
                completed_updates += 1

                if completed_updates in (1, 2):
                    for g, (p, before) in representatives.items():
                        delta = float((p.detach() - before).abs().max())
                        group_diagnostics[g]["max_parameter_delta"] = delta
                        group_update_checks[g] = group_update_checks[g] or delta > 0
                    current_t_hash = compute_teacher_hash(model.future_teacher)
                    if current_t_hash != initial_teacher_hash:
                        raise AssertionError("Teacher hash changed during updates!")

                gpu_peak = torch.cuda.max_memory_allocated(device) if (device.type == "cuda" and torch.cuda.is_available()) else 0

                if rank == 0:
                    metrics_file.write(json.dumps({
                        "update": completed_updates,
                        "global_step": stage2_g_step + completed_updates,
                        "epoch": current_epoch,
                        "batch_cursor": cursor,
                        "epoch_targets_seen": epoch_targets_seen,
                        "targets": global_targets,
                        "action_loss": global_action_sum / global_targets,
                        "future_loss": global_future_sum / global_targets,
                        "future_valid_count": int(global_future_valid),
                        "total_loss": global_loss_sum / global_targets,
                        "grad_norm": grad_norm,
                        "lr_next": scheduler.get_last_lr(),
                        "seconds": time.time() - step_start_time,
                        "gpu_peak_bytes": gpu_peak,
                        "group_diagnostics": group_diagnostics,
                        "group_update_checks": dict(group_update_checks),
                    }) + "\n")
                    metrics_file.flush()

                if rank == 0 and (completed_updates == 1 or completed_updates % 10 == 0):
                    print(f"[Joint Update {completed_updates}] epoch={current_epoch} cursor={cursor} "
                          f"loss={global_loss_sum / global_targets:.6f} "
                          f"act={global_action_sum / global_targets:.6f} fut={global_future_sum / global_targets:.6f} "
                          f"norm={grad_norm:.3f} sec={time.time() - step_start_time:.2f}", flush=True)

                # Periodic & First step checkpoint
                if completed_updates == 1 or (completed_updates % args.save_every == 0):
                    teacher_match_now = (compute_teacher_hash(model.future_teacher) == initial_teacher_hash)
                    if not teacher_match_now:
                        raise AssertionError("Frozen teacher changed")
                    save_joint_checkpoint(
                        output_dir=output_dir,
                        filename="last.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        update=completed_updates,
                        epoch=current_epoch,
                        batch_cursor=cursor,
                        epoch_targets_seen=epoch_targets_seen,
                        parent_global_step=p_global_step,
                        stage2_source_path=stage2_source_path_str,
                        stage2_source_sha256=stage2_source_sha,
                        stage2_global_step=stage2_g_step,
                        raw_config=raw_config,
                        raw_norm_stats=raw_norm_stats,
                        train_contract=training_contract,
                        train_data_contract=train_data_contract,
                        val_data_contract=val_data_contract,
                        source_module_fingerprints=current_source_fingerprints,
                        group_update_checks=group_update_checks,
                        teacher_hash_match=teacher_match_now,
                        device=device,
                        update_last=False,
                    )

                if completed_updates % args.eval_every == 0:
                    val_res = evaluate_predictive(model, val_dataset, val_indices, args.future_weight, device)
                    if rank == 0:
                        metrics_file.write(json.dumps({"type": "validation", "update": completed_updates, **val_res}) + "\n")
                        metrics_file.flush()

            # Epoch transition
            if cursor == total_segments:
                current_epoch += 1
                cursor = 0
                epoch_targets_seen = 0

                teacher_match_now = (compute_teacher_hash(model.future_teacher) == initial_teacher_hash)
                save_joint_checkpoint(
                    output_dir=output_dir,
                    filename=f"checkpoint_epoch_{current_epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=completed_updates,
                    epoch=current_epoch,
                    batch_cursor=cursor,
                    epoch_targets_seen=epoch_targets_seen,
                    parent_global_step=p_global_step,
                    stage2_source_path=stage2_source_path_str,
                    stage2_source_sha256=stage2_source_sha,
                    stage2_global_step=stage2_g_step,
                    raw_config=raw_config,
                    raw_norm_stats=raw_norm_stats,
                    train_contract=training_contract,
                    train_data_contract=train_data_contract,
                    val_data_contract=val_data_contract,
                    source_module_fingerprints=current_source_fingerprints,
                    group_update_checks=group_update_checks,
                    teacher_hash_match=teacher_match_now,
                    device=device,
                    update_last=True,
                )

        # Final checkpoint save: final.pt only if full target epoch is completed
        final_teacher_match = (compute_teacher_hash(model.future_teacher) == initial_teacher_hash)
        if not final_teacher_match:
            raise AssertionError("Teacher hash mismatch at end of training!")

        # Always update last.pt with exact ending state
        save_joint_checkpoint(
            output_dir=output_dir,
            filename="last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            update=completed_updates,
            epoch=current_epoch,
            batch_cursor=cursor,
            epoch_targets_seen=epoch_targets_seen,
            parent_global_step=p_global_step,
            stage2_source_path=stage2_source_path_str,
            stage2_source_sha256=stage2_source_sha,
            stage2_global_step=stage2_g_step,
            raw_config=raw_config,
            raw_norm_stats=raw_norm_stats,
            train_contract=training_contract,
            train_data_contract=train_data_contract,
            val_data_contract=val_data_contract,
            source_module_fingerprints=current_source_fingerprints,
            group_update_checks=group_update_checks,
            teacher_hash_match=final_teacher_match,
            device=device,
            update_last=False,
        )

        if current_epoch >= args.epochs and cursor == 0:
            save_joint_checkpoint(
                output_dir=output_dir,
                filename="final.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                update=completed_updates,
                epoch=current_epoch,
                batch_cursor=cursor,
                epoch_targets_seen=epoch_targets_seen,
                parent_global_step=p_global_step,
                stage2_source_path=stage2_source_path_str,
                stage2_source_sha256=stage2_source_sha,
                stage2_global_step=stage2_g_step,
                raw_config=raw_config,
                raw_norm_stats=raw_norm_stats,
                train_contract=training_contract,
                train_data_contract=train_data_contract,
                val_data_contract=val_data_contract,
                source_module_fingerprints=current_source_fingerprints,
                group_update_checks=group_update_checks,
                teacher_hash_match=final_teacher_match,
                device=device,
                update_last=False,
            )

    finally:
        if rank == 0 and metrics_file is not None:
            metrics_file.close()
        if is_distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
