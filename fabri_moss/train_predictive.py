"""Two-stage synchronous trainer for learned predictive memory writer and latent predictor.

Stage 1: Writer-only training (default 500 updates), future loss weight strictly 0.0.
Stage 2: Joint future-prediction training (default 2000 updates), future weight ramps linearly.
Freezes native policy backbone and trains only CausalMemoryWriter and FutureLatentHead.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.runtime import compute_file_sha256
from fabri_moss.train_native import (
    SegmentSequenceSampler,
    bucketed_gradient_allreduce,
    check_stop_file_requested,
    clip_parameter_groups_norm,
    compute_epoch_batches,
    identity_collate,
    require_training_device,
)


def compute_future_loss_weight(
    update: int,
    writer_updates: int,
    joint_updates: int,
    future_weight: float,
    ramp_updates: int = 200,
) -> Tuple[float, str]:
    """Compute phase name and future loss weight for a given completed update count."""
    if update < writer_updates:
        return 0.0, "stage1_writer_only"
    step_in_joint = (update - writer_updates) + 1
    if ramp_updates <= 0:
        weight = float(future_weight)
    else:
        ramp_factor = min(1.0, max(0.0, float(step_in_joint) / float(ramp_updates)))
        weight = float(future_weight) * ramp_factor
    return weight, "stage2_joint_future"


def create_predictive_optimizer_and_scheduler(
    new_modules: nn.ModuleDict | nn.ModuleList | Sequence[nn.Parameter] | nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 50,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create AdamW optimizer and warm-up-then-constant scheduler for new modules."""
    if isinstance(new_modules, nn.Module):
        params = [p for p in new_modules.parameters() if p.requires_grad]
    elif isinstance(new_modules, (list, tuple)):
        params = [p for p in new_modules if isinstance(p, nn.Parameter) and p.requires_grad]
    else:
        params = [p for p in new_modules.parameters() if p.requires_grad]

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for p in params:
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("No trainable parameters found in new modules.")

    optimizer = AdamW(param_groups)

    def lr_lambda(step: int) -> float:
        step_num = step + 1
        return float(step_num) / float(warmup_steps) if (warmup_steps > 0 and step_num <= warmup_steps) else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def compute_parameter_subset_hashes(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    num_samples: int = 4,
) -> Dict[str, str]:
    """Deterministically sample parameter tensors and compute SHA256 hashes as frozen invariants."""
    frozen_items = [(n, p) for n, p in named_params if not p.requires_grad]
    if not frozen_items:
        return {}
    step = max(1, len(frozen_items) // num_samples)
    chosen = [frozen_items[i] for i in range(0, len(frozen_items), step)][:num_samples]
    return {name: hashlib.sha256(p.detach().cpu().float().numpy().tobytes()).hexdigest() for name, p in chosen}


def assert_base_frozen_invariants(named_params: Sequence[Tuple[str, nn.Parameter]]) -> None:
    """Strictly assert that every base parameter has requires_grad=False and grad is None."""
    for name, p in named_params:
        if p.requires_grad:
            raise RuntimeError(f"Base parameter invariant violated: {name} has requires_grad=True")
        if p.grad is not None:
            raise RuntimeError(f"Base parameter invariant violated: {name} has non-None grad")


def build_predictive_checkpoint_payload(
    writer_module: nn.Module,
    future_head_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    update: int,
    stage_name: str,
    epoch: int,
    batch_cursor: int,
    epoch_targets_seen: int,
    parent_path: str,
    parent_sha256: str,
    parent_global_step: int,
    all_rng_states: List[Dict[str, Any]],
    writer_config_dict: Dict[str, Any],
    train_contract: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    source_module_fingerprints: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble predictive adapter v1 checkpoint dictionary."""
    return {
        "format": "predictive_memory_adapter_v1",
        "writer": {k: v.cpu() for k, v in writer_module.state_dict().items()},
        "future_head": {k: v.cpu() for k, v in future_head_module.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update": int(update),
        "stage": stage_name,
        "global_step": int(parent_global_step + update),
        "epoch": int(epoch),
        "batch_cursor": int(batch_cursor),
        "epoch_targets_seen": int(epoch_targets_seen),
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "parent_global_step": int(parent_global_step),
        "rng_states_per_rank": all_rng_states,
        "writer_config": writer_config_dict,
        "training_contract": train_contract,
        "train_data_contract": train_data_contract,
        "val_data_contract": val_data_contract,
        "source_module_fingerprints": source_module_fingerprints,
    }


def save_predictive_checkpoint(
    output_dir: Path,
    filename: str,
    writer_module: nn.Module,
    future_head_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    update: int,
    stage_name: str,
    epoch: int,
    batch_cursor: int,
    epoch_targets_seen: int,
    parent_path: str,
    parent_sha256: str,
    parent_global_step: int,
    writer_config_dict: Dict[str, Any],
    train_contract: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    source_module_fingerprints: Dict[str, Any],
    device: Optional[torch.device] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    update_last: bool = True,
) -> Path:
    """Save predictive checkpoint atomically on rank 0."""
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
        payload = build_predictive_checkpoint_payload(
            writer_module, future_head_module, optimizer, scheduler,
            update, stage_name, epoch, batch_cursor, epoch_targets_seen,
            parent_path, parent_sha256, parent_global_step, all_rngs,
            writer_config_dict, train_contract, train_data_contract, val_data_contract,
            source_module_fingerprints,
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


def load_predictive_checkpoint(
    checkpoint_path: Union[str, Path],
    writer_module: nn.Module,
    future_head_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    expected_parent_sha256: str,
    expected_training_contract: Dict[str, Any],
    expected_train_data_contract: Dict[str, Any],
    expected_val_data_contract: Dict[str, Any],
    device: Optional[torch.device] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    expected_source_module_fingerprints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Strictly load predictive adapter checkpoint and verify contracts."""
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    required_ckpt_keys = (
        "format", "writer", "future_head", "optimizer", "scheduler",
        "update", "stage", "global_step", "epoch", "batch_cursor",
        "epoch_targets_seen", "parent_path", "parent_sha256", "parent_global_step",
        "rng_states_per_rank", "writer_config", "training_contract",
        "train_data_contract", "val_data_contract", "source_module_fingerprints",
    )
    for field in required_ckpt_keys:
        if field not in ckpt:
            raise ValueError(f"Missing required checkpoint field: '{field}'")

    if ckpt.get("format") != "predictive_memory_adapter_v1":
        raise ValueError(f"Expected format 'predictive_memory_adapter_v1', got {ckpt.get('format')!r}")

    ckpt_tc = ckpt.get("training_contract")
    if not isinstance(ckpt_tc, dict):
        raise ValueError("Checkpoint training_contract must be a dict")

    required_tc_keys = (
        "format", "parent_path", "parent_sha256", "parent_global_step",
        "writer_updates", "joint_updates", "lambda_future", "future_ramp_updates",
        "lr", "weight_decay", "grad_clip", "warmup_updates", "world_size", "seed",
        "global_batch_size", "writer_config",
    )
    for k in required_tc_keys:
        if k not in expected_training_contract:
            raise ValueError(f"Training contract mismatch on '{k}': missing in expected_training_contract")
        if k not in ckpt_tc:
            raise ValueError(f"Training contract mismatch on '{k}': missing in checkpoint")

    for k in sorted(set(ckpt_tc.keys()) | set(expected_training_contract.keys())):
        if k not in ckpt_tc or k not in expected_training_contract or ckpt_tc[k] != expected_training_contract[k]:
            raise ValueError(f"Training contract mismatch on '{k}': {ckpt_tc.get(k)!r} != {expected_training_contract.get(k)!r}")

    if ckpt.get("parent_sha256") != expected_parent_sha256:
        raise ValueError(f"Parent SHA mismatch on resume: {ckpt.get('parent_sha256')} vs {expected_parent_sha256}")
    if ckpt.get("parent_path") != expected_training_contract["parent_path"]:
        raise ValueError(f"Parent path mismatch on resume: {ckpt.get('parent_path')} vs {expected_training_contract['parent_path']}")
    if ckpt.get("parent_global_step") != expected_training_contract["parent_global_step"]:
        raise ValueError(f"Parent global step mismatch on resume: {ckpt.get('parent_global_step')} vs {expected_training_contract['parent_global_step']}")

    actual_writer_cfg = dataclasses.asdict(writer_module.config) if hasattr(writer_module, "config") else None
    if actual_writer_cfg is not None:
        if ckpt.get("writer_config") != actual_writer_cfg:
            raise ValueError(f"Writer config mismatch between checkpoint and module: {ckpt.get('writer_config')} vs {actual_writer_cfg}")
        if expected_training_contract.get("writer_config") != actual_writer_cfg:
            raise ValueError(f"Writer config mismatch between expected contract and module: {expected_training_contract.get('writer_config')} vs {actual_writer_cfg}")
    if ckpt.get("writer_config") != expected_training_contract.get("writer_config"):
        raise ValueError(f"Writer config mismatch: {ckpt.get('writer_config')} vs {expected_training_contract.get('writer_config')}")

    if ckpt.get("train_data_contract") != expected_train_data_contract:
        raise ValueError("Resume train_data_contract mismatch")
    if ckpt.get("val_data_contract") != expected_val_data_contract:
        raise ValueError("Resume val_data_contract mismatch")

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    rng_per_rank = ckpt.get("rng_states_per_rank", [])
    if not isinstance(rng_per_rank, list) or len(rng_per_rank) != world_size:
        raise ValueError(f"Resume world_size={world_size} mismatch with checkpoint RNG count={len(rng_per_rank) if isinstance(rng_per_rank, list) else 'invalid'}")
    if expected_training_contract.get("world_size") != world_size:
        raise ValueError(f"Resume world_size={world_size} mismatch with training_contract world_size={expected_training_contract.get('world_size')}")

    for k in ("update", "global_step", "epoch", "batch_cursor", "epoch_targets_seen", "parent_global_step"):
        val = ckpt.get(k)
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError(f"Checkpoint field '{k}' must be a non-negative integer, got {val!r}")

    update = ckpt["update"]
    global_step = ckpt["global_step"]
    epoch = ckpt["epoch"]
    batch_cursor = ckpt["batch_cursor"]
    epoch_targets_seen = ckpt["epoch_targets_seen"]
    parent_global_step = ckpt["parent_global_step"]

    writer_updates = expected_training_contract["writer_updates"]
    joint_updates = expected_training_contract["joint_updates"]
    sum_phase_budgets = writer_updates + joint_updates
    if update > sum_phase_budgets:
        raise ValueError(f"Checkpoint update {update} exceeds total update budget {sum_phase_budgets}")

    if global_step != parent_global_step + update:
        raise ValueError(f"Checkpoint global_step {global_step} != parent_global_step {parent_global_step} + update {update}")

    if update == 0:
        expected_stage = "init"
    elif 0 < update <= writer_updates:
        expected_stage = "stage1_writer_only"
    else:
        expected_stage = "stage2_joint_future"

    if ckpt.get("stage") != expected_stage:
        raise ValueError(f"Checkpoint stage mismatch: got {ckpt.get('stage')!r}, expected {expected_stage!r} for update={update}")

    sched_state = ckpt.get("scheduler")
    if not isinstance(sched_state, dict) or sched_state.get("last_epoch") != update:
        raise ValueError(f"Scheduler last_epoch {sched_state.get('last_epoch') if isinstance(sched_state, dict) else None} != update {update}")

    if expected_source_module_fingerprints is not None:
        if ckpt.get("source_module_fingerprints") != expected_source_module_fingerprints:
            raise ValueError(f"Source module fingerprints mismatch: {ckpt.get('source_module_fingerprints')} != {expected_source_module_fingerprints}")

    for mod_name, mod in (("writer", writer_module), ("future_head", future_head_module)):
        mod_state = ckpt.get(mod_name)
        if not isinstance(mod_state, dict):
            raise ValueError(f"Checkpoint {mod_name} state must be a dict")
        expected_state = mod.state_dict()
        if set(mod_state.keys()) != set(expected_state.keys()):
            diff = set(mod_state.keys()) ^ set(expected_state.keys())
            raise ValueError(f"{mod_name} state_dict keys mismatch: differing keys {sorted(diff)}")
        for param_name, target_tensor in expected_state.items():
            loaded_tensor = mod_state[param_name]
            if not isinstance(loaded_tensor, torch.Tensor):
                raise ValueError(f"{mod_name} parameter '{param_name}' must be a torch.Tensor, got {type(loaded_tensor)}")
            if loaded_tensor.shape != target_tensor.shape:
                raise ValueError(f"{mod_name} parameter '{param_name}' shape mismatch: {loaded_tensor.shape} vs {target_tensor.shape}")
            if loaded_tensor.dtype != torch.float32:
                raise ValueError(f"{mod_name} parameter '{param_name}' dtype mismatch: expected torch.float32, got {loaded_tensor.dtype}")

    opt_state_dict = ckpt.get("optimizer")
    if not isinstance(opt_state_dict, dict):
        raise ValueError("Checkpoint optimizer state must be a dict")
    saved_param_groups = opt_state_dict.get("param_groups", [])
    if not isinstance(saved_param_groups, list) or len(saved_param_groups) != len(optimizer.param_groups):
        raise ValueError(f"Optimizer param_groups count mismatch: {len(saved_param_groups) if isinstance(saved_param_groups, list) else 'invalid'} vs {len(optimizer.param_groups)}")

    param_id_map: Dict[Any, nn.Parameter] = {}
    for g_idx, (saved_g, actual_g) in enumerate(zip(saved_param_groups, optimizer.param_groups)):
        s_params = saved_g.get("params", [])
        a_params = actual_g.get("params", [])
        if len(s_params) != len(a_params):
            raise ValueError(f"Optimizer param group {g_idx} param count mismatch: {len(s_params)} vs {len(a_params)}")
        for pid, actual_p in zip(s_params, a_params):
            param_id_map[pid] = actual_p

    opt_state = opt_state_dict.get("state", {})
    if not isinstance(opt_state, dict):
        raise ValueError("Optimizer state must be a dict")
    for pid, s in opt_state.items():
        if pid not in param_id_map:
            raise ValueError(f"Unknown parameter id {pid} in optimizer state")
        target_p = param_id_map[pid]
        for moment_key in ("exp_avg", "exp_avg_sq"):
            if moment_key in s:
                moment_t = s[moment_key]
                if not isinstance(moment_t, torch.Tensor):
                    raise ValueError(f"Optimizer state {moment_key} must be a torch.Tensor, got {type(moment_t)}")
                if moment_t.shape != target_p.shape:
                    raise ValueError(f"Optimizer state {moment_key} shape mismatch: {moment_t.shape} vs {target_p.shape}")
                if moment_t.dtype != torch.float32:
                    raise ValueError(f"Optimizer state {moment_key} dtype mismatch: expected torch.float32, got {moment_t.dtype}")

    writer_module.load_state_dict(ckpt["writer"], strict=True)
    future_head_module.load_state_dict(ckpt["future_head"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    my_rng = rng_per_rank[rank]
    random.setstate(my_rng["python"])
    np.random.set_state(my_rng["numpy"])
    torch_cpu_state = my_rng["torch_cpu"]
    if isinstance(torch_cpu_state, torch.Tensor):
        torch_cpu_state = torch_cpu_state.cpu()
    torch.set_rng_state(torch_cpu_state)
    dev_idx = device.index if (device is not None and device.type == "cuda") else None
    if my_rng.get("torch_cuda") is not None and device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch_cuda_state = my_rng["torch_cuda"]
        if isinstance(torch_cuda_state, torch.Tensor):
            torch_cuda_state = torch_cuda_state.cpu()
        torch.cuda.set_rng_state(torch_cuda_state, dev_idx)

    return {
        "update": int(ckpt["update"]),
        "stage": ckpt.get("stage", ""),
        "epoch": int(ckpt["epoch"]),
        "batch_cursor": int(ckpt["batch_cursor"]),
        "epoch_targets_seen": int(ckpt.get("epoch_targets_seen", 0)),
        "parent_global_step": int(ckpt.get("parent_global_step", 0)),
        "parent_path": ckpt.get("parent_path", ""),
    }


def validate_predictive_cursor(
    total_segments: int,
    global_batch_size: int,
    seed: int,
    epoch: int,
    cursor: int,
    epoch_targets_seen: int,
    dataset: Any,
) -> None:
    """Validate predictive dataset cursor alignment, boundaries, and targets seen against segment layout."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Epoch must be a non-negative integer, got {epoch!r}")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ValueError(f"Cursor must be a non-negative integer, got {cursor!r}")
    if isinstance(epoch_targets_seen, bool) or not isinstance(epoch_targets_seen, int) or epoch_targets_seen < 0:
        raise ValueError(f"epoch_targets_seen must be a non-negative integer, got {epoch_targets_seen!r}")

    if total_segments <= 0:
        raise ValueError(f"Dataset total segments must be > 0, got {total_segments}")
    if cursor > total_segments:
        raise ValueError(f"Invalid cursor {cursor} for total_segments {total_segments}")
    if cursor % global_batch_size != 0 and cursor != total_segments:
        raise ValueError(
            f"Cursor {cursor} must align with global_batch_size {global_batch_size} boundary or len {total_segments}"
        )

    if cursor == 0:
        expected_targets = 0
    else:
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(seed + epoch * 37)
        perm = torch.randperm(total_segments, generator=epoch_generator).tolist()

        processed_segments = perm[:cursor]
        expected_targets = 0
        for s_idx in processed_segments:
            _, s, e = dataset.segments[s_idx]
            expected_targets += (e - s)

    if epoch_targets_seen != expected_targets:
        raise ValueError(
            f"Cursor state mismatch: recorded epoch_targets_seen {epoch_targets_seen} != "
            f"expected {expected_targets} from {cursor} processed segments"
        )


def evaluate_predictive(
    model: nn.Module,
    val_dataset: Any,
    eval_indices: Sequence[int],
    future_weight: float,
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
    base_seed: int = 20260908,
) -> Dict[str, float]:
    """Evaluate predictive model deterministically on validation segments."""
    if len(eval_indices) == 0:
        return {"action_loss": 0.0, "future_loss": 0.0, "loss": 0.0, "targets": 0.0, "future_valid_count": 0.0}

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    was_training = model.training
    model.eval()
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    dev_idx = device.index if device.type == "cuda" else None
    cuda_rng = torch.cuda.get_rng_state(dev_idx) if (device.type == "cuda" and torch.cuda.is_available()) else None

    rank_indices = [idx for i, idx in enumerate(eval_indices) if i % world_size == rank]
    local_act_sum, local_fut_sum, local_targets, local_fval = 0.0, 0.0, 0, 0

    try:
        with torch.no_grad():
            for seg_idx in rank_indices:
                seg_seed = base_seed + int(seg_idx)
                random.seed(seg_seed)
                np.random.seed(seg_seed % (2**32 - 1))
                torch.manual_seed(seg_seed)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed(seg_seed)

                sample = val_dataset[seg_idx]
                target_count = int(sample.get("target_count", 0))
                if target_count <= 0:
                    continue
                sample_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
                out = model(sample_dev, future_weight=future_weight)
                act_loss = float(out.get("action_loss", out.get("loss", 0.0)))
                fut_loss = float(out.get("future_loss", 0.0))
                fut_cnt = int(out.get("future_valid_count", 0))

                if not math.isfinite(act_loss) or not math.isfinite(fut_loss):
                    raise FloatingPointError("Non-finite validation loss")
                local_act_sum += act_loss * target_count
                local_fut_sum += fut_loss * target_count
                local_targets += target_count
                local_fval += fut_cnt

        if is_dist and world_size > 1:
            stats = torch.tensor([local_act_sum, local_fut_sum, float(local_targets), float(local_fval)], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=process_group)
            g_act_sum, g_fut_sum, g_targets, g_fval = stats.tolist()
        else:
            g_act_sum, g_fut_sum, g_targets, g_fval = local_act_sum, local_fut_sum, float(local_targets), float(local_fval)

        mean_act = (g_act_sum / g_targets) if g_targets > 0 else 0.0
        mean_fut = (g_fut_sum / g_targets) if g_targets > 0 else 0.0
        return {
            "action_loss": mean_act,
            "future_loss": mean_fut,
            "loss": mean_act + future_weight * mean_fut,
            "targets": g_targets,
            "future_valid_count": g_fval,
        }
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng, dev_idx)
        if was_training:
            model.train()


def parse_args(raw_args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for train_predictive."""
    parser = argparse.ArgumentParser(description="Two-stage training for predictive memory writer and latent predictor.")
    parser.add_argument("--init-from", type=str, default=None, help="Parent compact checkpoint path (required unless --resume).")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for predictive checkpoints and logs.")
    parser.add_argument("--resume", type=str, default=None, help="Predictive adapter checkpoint path to resume from.")
    parser.add_argument("--writer-updates", type=int, default=500, help="Number of optimizer updates for stage 1 (writer-only).")
    parser.add_argument("--joint-updates", type=int, default=2000, help="Number of optimizer updates for stage 2 (joint future).")
    parser.add_argument("--max-updates", type=int, default=None, help="Optional total update ceiling (for testing).")
    parser.add_argument("--future-weight", type=float, default=0.001, help="Final weight for future latent MSE loss in stage 2.")
    parser.add_argument("--future-ramp-updates", type=int, default=200, help="Linear ramp updates for future loss in stage 2.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for new modules.")
    parser.add_argument("--warmup-updates", type=int, default=50, help="Warmup updates for learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for new modules.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm clipping.")
    parser.add_argument("--global-batch-size", type=int, default=8, help="Global batch size across all ranks.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=4042, help="Random seed.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--save-every", type=int, default=100, help="Save interval in updates.")
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluation interval in updates.")
    parser.add_argument("--stop-file", type=str, default=None, help="Path to stop sentinel file.")
    parser.add_argument("--device", type=str, default=None, help="Device override for testing (e.g. 'cpu').")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt")
    parser.add_argument("--vlm-path", type=str, default="/root/models/InternVL3_5-1B")
    parser.add_argument("--data-root", type=str, default="/root/evo1_metaworld_dataset")
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    return parser.parse_args(raw_args)


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate CLI arguments strictly before runtime initialization."""
    if args.resume is None and args.init_from is None:
        raise ValueError("Either --init-from or --resume must be provided.")
    if args.writer_updates < 0 or args.joint_updates < 0 or (args.writer_updates + args.joint_updates) <= 0:
        raise ValueError("Invalid update budgets: writer and joint updates must be >= 0 and sum > 0.")
    for arg_name in ("global_batch_size", "lr", "grad_clip", "save_every", "eval_every"):
        if not math.isfinite(getattr(args, arg_name)) or getattr(args, arg_name) <= 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be positive and finite, got {getattr(args, arg_name)}")
    for name in ("workers", "future_ramp_updates", "warmup_updates", "weight_decay", "future_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in ("max_updates", "max_train_episodes", "max_val_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.device is not None and not args.device.startswith("cuda"):
        raise ValueError("Formal training requires CUDA; CPU is only available through test injection")
    if args.stop_file is not None and not str(args.stop_file).strip():
        raise ValueError("--stop-file must be a nonempty path")


def run_training(args: argparse.Namespace, device: Optional[torch.device] = None) -> None:
    """Main execution function for two-stage predictive training."""
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
                raise RuntimeError("CUDA is strictly required for formal run_training execution.")
            device = torch.device(args.device or "cuda:0")
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

        from fabri_moss.predictive_data import PredictiveTrainingDataset
        from fabri_moss.predictive_policy import PredictiveMemoryPolicy

        parent_path_to_load = args.init_from
        if args.resume is not None:
            res_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
            parent_path_to_load = res_ckpt.get("parent_path")
            if args.init_from is not None and str(Path(args.init_from).resolve()) != str(Path(parent_path_to_load).resolve()):
                raise ValueError(f"Cannot change parent --init-from when resuming: {args.init_from} vs {parent_path_to_load}")

        parent_path = Path(parent_path_to_load).resolve()
        if not parent_path.exists():
            raise FileNotFoundError(f"Parent checkpoint not found: {parent_path}")

        parent_sha256 = compute_file_sha256(parent_path)
        parent_ckpt = torch.load(str(parent_path), map_location="cpu", weights_only=False, mmap=True)

        parent_training_contract = parent_ckpt.get("training_contract", {})
        if parent_training_contract.get("format") != "native_compact_memory_replay_v1":
            raise ValueError("Parent checkpoint format strictly must be 'native_compact_memory_replay_v1'")
        if parent_training_contract.get("stream_protocol") != get_compact_protocol_contract():
            raise ValueError("Parent checkpoint stream_protocol does not match get_compact_protocol_contract()")

        # Verify parent training contract invariants against new training hyperparameters
        for k in ("seed", "global_batch_size"):
            if parent_training_contract.get(k) != getattr(args, k):
                raise ValueError(f"Parent training contract mismatch on {k}: {parent_training_contract[k]} vs {getattr(args, k)}")
        if parent_training_contract.get("world_size") != world_size:
            raise ValueError(f"Parent training contract mismatch on world_size: {parent_training_contract['world_size']} vs {world_size}")

        parent_global_step = int(parent_ckpt.get("global_step", 0))
        start_epoch = int(parent_ckpt.get("epoch", 0))
        start_cursor = int(parent_ckpt.get("batch_cursor", 0))
        start_epoch_targets_seen = int(parent_ckpt.get("epoch_targets_seen", 0))

        # First load native checkpoint (trainable=True) to obtain norm_stats
        from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint
        base_policy, _, norm_stats, _ = load_native_checkpoint(
            fabri_root=args.fabri_root, checkpoint_path=str(parent_path), vlm_path=args.vlm_path,
            device=str(device), arm_key="metaworld_sawyer", trainable=True,
        )
        if device.type == "cuda":
            require_training_device(base_policy, str(device))
            assert_native_fa2(base_policy)

        # Dataset initialization with root=args.data_root, norm_stats, and augmentation=False
        train_dataset = PredictiveTrainingDataset(
            root=args.data_root, norm_stats=norm_stats, split="train", history_frames=16, target_frames=8,
            augmentation=False, seed=args.seed, val_fraction=args.val_fraction, max_episodes=args.max_train_episodes,
        )
        val_dataset = PredictiveTrainingDataset(
            root=args.data_root, norm_stats=norm_stats, split="val", history_frames=16, target_frames=8,
            augmentation=False, seed=args.seed, val_fraction=args.val_fraction, max_episodes=args.max_val_episodes,
        )

        # Parent data validation:逐train,val严格校验
        compact_proto = get_compact_protocol_contract()
        compact_proto_sorted_json = json.dumps(compact_proto, sort_keys=True)

        for split_name, ds, ckpt_key in [("train", train_dataset, "train_data_contract"), ("val", val_dataset, "val_data_contract")]:
            parent_dc = parent_ckpt.get(ckpt_key, {})
            expected_dc = dict(ds.get_base_data_contract())
            # augmentation is the only explicitly allowed variation between base and parent compact
            expected_dc["augmentation"] = parent_dc.get("augmentation", expected_dc.get("augmentation"))
            expected_dc["stream_protocol"] = compact_proto
            exp_dfp_raw = f"{expected_dc['data_fingerprint']}:{compact_proto_sorted_json}"
            expected_dc["data_fingerprint"] = hashlib.sha256(exp_dfp_raw.encode("utf-8")).hexdigest()

            if parent_dc != expected_dc:
                for k in sorted(set(parent_dc.keys()) | set(expected_dc.keys())):
                    if parent_dc.get(k) != expected_dc.get(k):
                        raise ValueError(f"Parent {split_name} data contract mismatch on '{k}': parent={parent_dc.get(k)!r}, expected={expected_dc.get(k)!r}")

        validate_predictive_cursor(len(train_dataset), args.global_batch_size, args.seed,
                                   start_epoch, start_cursor, start_epoch_targets_seen, train_dataset)
        del parent_ckpt
        if args.resume:
            del res_ckpt
        model = PredictiveMemoryPolicy(policy=base_policy).to(device)
        base_named = [(n, p) for n, p in model.policy.named_parameters()]
        assert_base_frozen_invariants(base_named)
        frozen_hashes = compute_parameter_subset_hashes(base_named)

        if is_distributed and world_size > 1:
            for p in model.writer.parameters():
                dist.broadcast(p.data, src=0)
            for p in model.future_head.parameters():
                dist.broadcast(p.data, src=0)

        trainable_modules = nn.ModuleList([model.writer, model.future_head])
        trainable_params = list(trainable_modules.parameters())
        optimizer, scheduler = create_predictive_optimizer_and_scheduler(
            trainable_modules, lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_updates,
        )

        writer_config_dict = dataclasses.asdict(model.writer.config) if hasattr(model.writer, "config") else {}
        training_contract = {
            "format": "predictive_memory_adapter_v1", "parent_path": str(parent_path), "parent_sha256": parent_sha256,
            "parent_global_step": parent_global_step,
            "writer_updates": args.writer_updates, "joint_updates": args.joint_updates,
            "lambda_future": args.future_weight, "future_ramp_updates": args.future_ramp_updates,
            "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
            "warmup_updates": args.warmup_updates, "world_size": world_size, "seed": args.seed,
            "global_batch_size": args.global_batch_size, "writer_config": writer_config_dict,
            "temporal": get_compact_protocol_contract()["temporal"],
            "text_timestamps": False, "shallow_layer": model.shallow_layer,
            "gradient_checkpointing": model.gradient_checkpointing,
            "precision": "frozen_fp32_base_bf16_vision_llm_fp32_writer_head_expert",
        }
        train_contract_data = train_dataset.get_data_contract()
        val_contract_data = val_dataset.get_data_contract()

        src_dir = Path(__file__).resolve().parent
        source_module_fingerprints = {
            "predictive_memory.py": compute_file_sha256(src_dir / "predictive_memory.py"),
            "predictive_policy.py": compute_file_sha256(src_dir / "predictive_policy.py"),
            "predictive_data.py": compute_file_sha256(src_dir / "predictive_data.py"),
            "train_predictive.py": compute_file_sha256(src_dir / "train_predictive.py"),
        }

        # Handle resume or new run setup
        completed_updates = 0
        current_epoch = start_epoch
        current_cursor = start_cursor
        epoch_targets_seen = start_epoch_targets_seen

        resume_event = None
        if args.resume:
            resume_info = load_predictive_checkpoint(
                checkpoint_path=args.resume, writer_module=model.writer, future_head_module=model.future_head,
                optimizer=optimizer, scheduler=scheduler, expected_parent_sha256=parent_sha256,
                expected_training_contract=training_contract, expected_train_data_contract=train_contract_data,
                expected_val_data_contract=val_contract_data, device=device,
                expected_source_module_fingerprints=source_module_fingerprints,
            )
            completed_updates = resume_info["update"]
            current_epoch = resume_info["epoch"]
            current_cursor = resume_info["batch_cursor"]
            epoch_targets_seen = resume_info["epoch_targets_seen"]
            resume_event = {
                "resumed_from": str(args.resume), "update": completed_updates,
                "epoch": current_epoch, "cursor": current_cursor, "timestamp": time.time(),
            }

        validate_predictive_cursor(len(train_dataset), args.global_batch_size, args.seed,
                                   current_epoch, current_cursor, epoch_targets_seen, train_dataset)
        if args.resume and run_config_path.exists():
            existing_config = json.loads(run_config_path.read_text())
            for key, expected in (("training_contract", training_contract),
                                  ("train_data_contract", train_contract_data),
                                  ("val_data_contract", val_contract_data),
                                  ("source_module_fingerprints", source_module_fingerprints)):
                if existing_config.get(key) != expected:
                    raise ValueError(f"run_config {key} mismatch")

        # Check already done against total updates budget
        total_target_updates = min(args.writer_updates + args.joint_updates, args.max_updates) if args.max_updates is not None else (args.writer_updates + args.joint_updates)
        if completed_updates >= total_target_updates:
            if rank == 0:
                print(f"[Done] Already completed {completed_updates} >= {total_target_updates}. Exiting.")
            return

        # Write / update run_config.json
        if rank == 0:
            if run_config_path.exists() and args.resume:
                rc = json.loads(run_config_path.read_text())
                if resume_event is not None:
                    rc.setdefault("resume_events", []).append(resume_event)
                run_config_path.write_text(json.dumps(rc, indent=2))
            else:
                rc = {
                    "training_contract": training_contract,
                    "train_data_contract": train_contract_data,
                    "val_data_contract": val_contract_data,
                    "source_module_fingerprints": source_module_fingerprints,
                    "parent_path": str(parent_path),
                    "parent_sha256": parent_sha256,
                    "parent_global_step": parent_global_step,
                    "frozen_parameter_subset_sha256": frozen_hashes,
                    "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                    "resume_events": [resume_event] if resume_event is not None else [],
                }
                run_config_path.write_text(json.dumps(rc, indent=2))

        # Validation indices: from ds.validation_indices() up to 8
        val_dataset.set_epoch(start_epoch)
        raw_val_indices = val_dataset.validation_indices()
        val_indices = [raw_val_indices[i] for i in np.linspace(
            0, len(raw_val_indices) - 1, min(8, len(raw_val_indices)), dtype=int
        )] if raw_val_indices else []

        metrics_file = (output_dir / "metrics.jsonl").open("a") if rank == 0 else None

        writer_out_proj = getattr(model.writer, "out_proj", None)
        writer_out_proj_param = getattr(writer_out_proj, "weight", None) if writer_out_proj is not None else None
        writer_out_proj_snap = writer_out_proj_param.detach().clone() if writer_out_proj_param is not None else None

        future_head_repr_p = next((p for p in model.future_head.parameters() if p.ndim > 1), None)
        future_head_snap = future_head_repr_p.detach().clone() if future_head_repr_p is not None else None

        head_initial = {n: p.detach().clone() for n, p in model.future_head.named_parameters()}
        head_update_verified = any(p in optimizer.state for p in model.future_head.parameters())
        stop_requested = False
        exhausted_without_updates = False
        total_segments = len(train_dataset)
        cursor = current_cursor

        while completed_updates < total_target_updates and not stop_requested:
            epoch_start_updates, epoch_start_cursor = completed_updates, cursor
            train_dataset.set_epoch(current_epoch)
            _, epoch_batches, _ = compute_epoch_batches(
                total_segments, args.global_batch_size, world_size, args.seed, current_epoch,
            )

            # Determine remaining batches in this epoch from current cursor
            batch_start_idx = math.ceil(cursor / args.global_batch_size)
            remaining_batches = epoch_batches[batch_start_idx:]
            rank_remaining_indices = [idx for bslice in remaining_batches for idx in bslice[rank::world_size]]

            loader_generator = torch.Generator()
            loader_generator.manual_seed(args.seed + current_epoch * 10007 + rank * 997)

            loader = DataLoader(
                train_dataset, batch_size=None, sampler=SegmentSequenceSampler(rank_remaining_indices),
                num_workers=args.workers, collate_fn=identity_collate,
                multiprocessing_context="spawn" if (args.workers > 0 and sys.platform != "win32") else None,
                prefetch_factor=1 if args.workers > 0 else None,
                generator=loader_generator,
            )
            loader_iter = iter(loader)

            for batch_slice in remaining_batches:
                # First check stop file / update ceiling before consuming batch
                if completed_updates >= total_target_updates or check_stop_file_requested(args.stop_file, device):
                    stop_requested = True
                    break

                current_future_weight, stage_name = compute_future_loss_weight(
                    completed_updates, args.writer_updates, args.joint_updates, args.future_weight, args.future_ramp_updates,
                )

                cursor += len(batch_slice)
                optimizer.zero_grad(set_to_none=True)
                model.train()

                local_targets_sum, local_loss_sum, local_has_grad = 0, 0.0, False
                local_action_sum, local_future_sum, local_future_valid = 0.0, 0.0, 0
                step_start_time = time.time()

                for _ in range(len(batch_slice[rank::world_size])):
                    sample = next(loader_iter)
                    target_count = int(sample["target_count"])
                    sample_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}

                    out = model(sample_dev, future_weight=current_future_weight)
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
                    raise FloatingPointError(f"Encountered non-finite loss sum: {global_loss_sum}")

                epoch_targets_seen += global_targets

                if not global_has_grad:
                    # Skip batch: advance cursor/epoch_targets_seen without optimizer step or update++
                    if rank == 0:
                        metrics_file.write(json.dumps({
                            "type": "skipped_batch", "epoch": current_epoch, "batch_cursor": cursor,
                            "epoch_targets_seen": epoch_targets_seen, "targets": global_targets,
                            "phase": stage_name, "future_weight": current_future_weight,
                        }) + "\n")
                        metrics_file.flush()
                    continue

                # Every update: assert base parameter invariants
                assert_base_frozen_invariants(base_named)

                bucketed_gradient_allreduce(trainable_params, global_target_count=global_targets)
                if stage_name == "stage1_writer_only":
                    if any(p.grad is not None or p in optimizer.state for p in model.future_head.parameters()):
                        raise AssertionError("Stage 1 must not update the future head")
                verify_head_now = not head_update_verified and any(p.grad is not None for p in model.future_head.parameters())
                grad_norm = clip_parameter_groups_norm(trainable_params, max_norm=args.grad_clip)

                optimizer.step()
                scheduler.step()
                completed_updates += 1

                # Assert model.writer.out_proj.weight changed on first update
                if completed_updates == 1:
                    if writer_out_proj_param is None:
                        raise RuntimeError("model.writer.out_proj.weight not found for first-update verification")
                    if torch.equal(writer_out_proj_param, writer_out_proj_snap):
                        raise AssertionError("Writer first update assertion failed: model.writer.out_proj.weight did not change")

                # Assert future_head parameter delta > 0 on first joint update
                if verify_head_now:
                    if future_head_repr_p is None:
                        raise RuntimeError("model.future_head parameter not found for first joint update verification")
                    head_delta = (future_head_repr_p - future_head_snap).abs().sum().item()
                    if head_delta <= 0.0:
                        raise AssertionError(f"First joint update assertion failed: future_head parameter delta is {head_delta} <= 0")
                    head_update_verified = True

                gpu_peak = torch.cuda.max_memory_allocated(device) if (device.type == "cuda" and torch.cuda.is_available()) else 0

                if rank == 0:
                    metrics_file.write(json.dumps({
                        "update": completed_updates, "phase": stage_name, "future_weight": current_future_weight,
                        "action_loss": global_action_sum / global_targets,
                        "future_loss": global_future_sum / global_targets,
                        "future_valid_count": int(global_future_valid),
                        "future_head_update_verified": head_update_verified,
                        "total_loss": global_loss_sum / max(1, global_targets), "grad_norm": grad_norm,
                        "targets": global_targets, "epoch": current_epoch, "batch_cursor": cursor,
                        "epoch_targets_seen": epoch_targets_seen,
                        "parent_derived_step": parent_global_step + completed_updates,
                        "seconds": time.time() - step_start_time, "gpu_peak_bytes": gpu_peak,
                        "lr_next": scheduler.get_last_lr(),
                    }) + "\n")
                    metrics_file.flush()

                if rank == 0 and (completed_updates == 1 or completed_updates % 10 == 0 or verify_head_now):
                    print(f"[Update {completed_updates}/{args.writer_updates + args.joint_updates}] "
                          f"{stage_name} loss={global_loss_sum / global_targets:.8f} "
                          f"future_weight={current_future_weight:.8g} "
                          f"seconds={time.time() - step_start_time:.2f}", flush=True)

                # Save checkpoints at boundaries: stage1 at 500, stage2 at 2500 (if full run), periodic, and last
                is_stage1_boundary = (completed_updates == args.writer_updates)
                is_stage2_boundary = (completed_updates == (args.writer_updates + args.joint_updates))
                if is_stage1_boundary or is_stage2_boundary:
                    if compute_parameter_subset_hashes(base_named) != frozen_hashes:
                        raise AssertionError("Frozen base parameter subset changed")
                    if is_stage1_boundary and any(not torch.equal(p, head_initial[n]) for n, p in model.future_head.named_parameters()):
                        raise AssertionError("Future head changed during stage 1")
                    if rank == 0:
                        metrics_file.write(json.dumps({"type": "frozen_check", "update": completed_updates,
                                                       "subset_sha256": frozen_hashes, "match": True}) + "\n")
                        metrics_file.flush()

                if is_stage1_boundary:
                    save_predictive_checkpoint(
                        output_dir, "stage1.pt", model.writer, model.future_head, optimizer, scheduler,
                        completed_updates, stage_name, current_epoch, cursor, epoch_targets_seen,
                        str(parent_path), parent_sha256, parent_global_step, writer_config_dict,
                        training_contract, train_contract_data, val_contract_data, source_module_fingerprints, device,
                        update_last=True,
                    )
                elif is_stage2_boundary:
                    save_predictive_checkpoint(
                        output_dir, "stage2.pt", model.writer, model.future_head, optimizer, scheduler,
                        completed_updates, stage_name, current_epoch, cursor, epoch_targets_seen,
                        str(parent_path), parent_sha256, parent_global_step, writer_config_dict,
                        training_contract, train_contract_data, val_contract_data, source_module_fingerprints, device,
                        update_last=True,
                    )
                elif completed_updates == 1 or completed_updates % args.save_every == 0:
                    save_predictive_checkpoint(
                        output_dir, "last.pt", model.writer, model.future_head, optimizer, scheduler,
                        completed_updates, stage_name, current_epoch, cursor, epoch_targets_seen,
                        str(parent_path), parent_sha256, parent_global_step, writer_config_dict,
                        training_contract, train_contract_data, val_contract_data, source_module_fingerprints, device,
                        update_last=False,
                    )

                if completed_updates % args.eval_every == 0 or is_stage1_boundary or completed_updates == total_target_updates:
                    val_res = evaluate_predictive(model, val_dataset, val_indices, current_future_weight, device)
                    if rank == 0:
                        metrics_file.write(json.dumps({"type": "validation", "update": completed_updates, "stage": stage_name, **val_res}) + "\n")
                        metrics_file.flush()

            # Epoch transition: only reset cursor and targets if the full epoch was completely consumed
            if cursor == total_segments:
                if epoch_start_cursor == 0 and completed_updates == epoch_start_updates:
                    exhausted_without_updates = True
                    stop_requested = True
                current_epoch += 1
                cursor = 0
                epoch_targets_seen = 0
            current_cursor = cursor

        # Final checkpoint save: accurate current cursor, stage name, update_last=False
        final_stage_name = ("init" if completed_updates == 0 else
                            "stage1_writer_only" if completed_updates <= args.writer_updates else "stage2_joint_future")
        if compute_parameter_subset_hashes(base_named) != frozen_hashes:
            raise AssertionError("Frozen base parameter subset changed")
        save_predictive_checkpoint(
            output_dir, "last.pt", model.writer, model.future_head, optimizer, scheduler,
            completed_updates, final_stage_name, current_epoch,
            cursor, epoch_targets_seen,
            str(parent_path), parent_sha256, parent_global_step, writer_config_dict,
            training_contract, train_contract_data, val_contract_data, source_module_fingerprints, device,
            update_last=False,
        )
        if exhausted_without_updates:
            raise RuntimeError("A full epoch produced no trainable graph; checkpoint saved without inventing updates")

    finally:
        if rank == 0 and metrics_file is not None:
            metrics_file.close()
        if is_distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
