"""Synchronous data-parallel trainer for native FabriVLA sequence policy fine-tuning.

Implements full gradient differentiation through vision encoder, projector,
language model, and action head across multi-frame causal sequences without
matrix compression, cross-attention adapters, or readout tokens.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from fabri_moss.runtime import compute_file_sha256
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.memory_protocol import get_memory_protocol_contract
from fabri_moss.stream_protocol import get_stream_protocol_contract


def get_protocol_contract_and_format(protocol: str) -> Tuple[Dict[str, Any], str]:
    """Return the stable contract specification and training contract format for a protocol."""
    if protocol == "stream_replay_v1":
        return get_stream_protocol_contract(), "native_stream_replay_v1"
    elif protocol == "memory_replay_v1":
        return get_memory_protocol_contract(), "native_memory_replay_v1"
    elif protocol == "compact_memory_replay_v1":
        return get_compact_protocol_contract(), "native_compact_memory_replay_v1"
    raise ValueError(f"Unknown protocol: {protocol!r}")


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


def classify_parameter(name: str) -> str:
    """Classify a policy parameter name into one of the four required trainable groups.

    Groups:
    1. 'vision': embedder.model.vision_model
    2. 'projector': embedder.model.mlp1
    3. 'llm': embedder.model.language_model
    4. 'head': action_head
    """
    if "embedder.model.vision_model." in name or name.startswith("embedder.model.vision_model"):
        return "vision"
    elif "embedder.model.mlp1." in name or name.startswith("embedder.model.mlp1"):
        return "projector"
    elif "embedder.model.language_model." in name or name.startswith("embedder.model.language_model"):
        return "llm"
    elif "action_head." in name or name.startswith("action_head"):
        return "head"
    raise ValueError(
        f"Unclassified trainable parameter: '{name}'. Must belong to embedder.model.vision_model, embedder.model.mlp1, embedder.model.language_model, or action_head."
    )


def create_native_optimizer_and_scheduler(
    policy: nn.Module,
    lr_vision: float = 1e-6,
    lr_projector: float = 5e-6,
    lr_llm: float = 2e-6,
    lr_head: float = 1e-5,
    weight_decay: float = 1e-4,
    total_steps: int = 1000,
    warmup_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create AdamW optimizer with group-specific learning rates and weight decay rules.

    - Vision: embedder.model.vision_model -> lr_vision
    - Projector: embedder.model.mlp1 -> lr_projector
    - LLM: embedder.model.language_model -> lr_llm
    - Head: action_head -> lr_head
    - Norm, bias, and 1D parameters have weight decay 0.0. Other parameters use weight_decay.
    - Warmup first actual update non-zero: min(1.0, (step + 1) / warmup_steps).
    - Cosine decay to total_steps: 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup))).
    """
    lr_map = {
        "vision": lr_vision,
        "projector": lr_projector,
        "llm": lr_llm,
        "head": lr_head,
    }

    # Buckets: (group_name, is_no_decay) -> list of parameters
    buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {
        (g, nd): [] for g in ("vision", "projector", "llm", "head") for nd in (False, True)
    }

    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        group = classify_parameter(name)
        is_no_decay = (param.ndim <= 1) or ("bias" in name) or ("norm" in name)
        buckets[(group, is_no_decay)].append(param)

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
        raise ValueError("No trainable parameters found to optimize.")

    optimizer = AdamW(param_groups)

    def lr_lambda(current_step: int) -> float:
        # step is 0-based step index passed by PyTorch LambdaLR
        # Warmup first actual update non-zero: min(1.0, (current_step + 1) / warmup_steps)
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


def bucketed_gradient_allreduce(
    parameters: Sequence[nn.Parameter],
    global_target_count: int,
    max_bucket_bytes: int = 64 * 1024 * 1024,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Bucket-based gradient allreduce with global target count normalization.

    Synchronizes gradients in a fixed parameter order in contiguous buckets (<= max_bucket_bytes).
    Note: A single parameter exceeding max_bucket_bytes (e.g. language model embedding table
    vocabulary matrix ~600MiB) is explicitly allowed as an individual oversized bucket without
    artificial splitting or faking the 64MiB bound.

    Detects globally unused gradients:
    - If a parameter has no gradient locally on any rank, its grad remains None globally.
    - If a parameter has grad on some ranks but not others, missing ranks contribute zeros.
    - Gradients are reduced with SUM globally across ranks and divided by global_target_count.
    - All gradient tensors are strictly enforced to be FP32.
    """
    if global_target_count <= 0:
        raise ValueError(f"global_target_count must be > 0, got {global_target_count}")

    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    trainable_params = [p for p in parameters if p.requires_grad]
    if not trainable_params:
        return

    # Enforce strict float32 dtype on all present gradients
    for p in trainable_params:
        if p.grad is not None and p.grad.dtype != torch.float32:
            raise TypeError(
                f"Gradient tensor has dtype {p.grad.dtype}, strictly expected float32"
            )

    # Determine which parameters have local gradients
    device = trainable_params[0].device
    local_has_grad = torch.tensor(
        [1 if p.grad is not None else 0 for p in trainable_params],
        dtype=torch.int32,
        device=device,
    )

    if is_dist and world_size > 1:
        global_has_grad = local_has_grad.clone()
        dist.all_reduce(global_has_grad, op=dist.ReduceOp.SUM, group=process_group)
    else:
        global_has_grad = local_has_grad

    # Convert global flags to CPU list once to eliminate per-parameter GPU synchronization
    global_has_grad_list: List[int] = global_has_grad.tolist()

    # Single-rank fast path
    if not is_dist or world_size == 1:
        for p, g_count in zip(trainable_params, global_has_grad_list):
            if g_count > 0 and p.grad is not None:
                p.grad.div_(float(global_target_count))
        return

    # Distributed bucketed all-reduce
    current_bucket_params: List[nn.Parameter] = []
    current_bucket_bytes = 0

    def reduce_bucket(bucket_params: List[nn.Parameter]) -> None:
        if not bucket_params:
            return
        b_dev = bucket_params[0].device
        grad_tensors = []
        for bp in bucket_params:
            if bp.grad is None:
                grad_tensors.append(torch.zeros_like(bp.data, device=b_dev, dtype=torch.float32))
            else:
                grad_tensors.append(bp.grad.data)

        flat_grad = torch.cat([g.reshape(-1) for g in grad_tensors])
        if flat_grad.dtype != torch.float32:
            flat_grad = flat_grad.float()
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=process_group)

        # Divide by global target count once
        flat_grad.div_(float(global_target_count))

        # Unflatten back into parameters
        offset = 0
        for bp in bucket_params:
            numel = bp.numel()
            reduced_slice = flat_grad[offset : offset + numel].reshape_as(bp.data)
            if bp.grad is None:
                bp.grad = reduced_slice.clone()
            else:
                bp.grad.data.copy_(reduced_slice)
            offset += numel

    for p, g_count in zip(trainable_params, global_has_grad_list):
        if g_count == 0:
            # Globally unused: leave grad as None
            continue
        p_bytes = p.numel() * p.element_size()
        if current_bucket_params and (current_bucket_bytes + p_bytes > max_bucket_bytes):
            reduce_bucket(current_bucket_params)
            current_bucket_params = []
            current_bucket_bytes = 0

        current_bucket_params.append(p)
        current_bucket_bytes += p_bytes

    if current_bucket_params:
        reduce_bucket(current_bucket_params)


def clip_parameter_groups_norm(
    parameters: Sequence[nn.Parameter],
    max_norm: float,
) -> float:
    """Clips gradient norm over parameters that have non-None gradients.

    Passes parameters (which have .grad attributes) to torch.nn.utils.clip_grad_norm_
    with error_if_nonfinite=True so that gradient tensors are correctly scaled in-place.
    """
    valid_params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not valid_params:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(
        valid_params,
        max_norm=max_norm,
        error_if_nonfinite=True,
    )
    return float(total_norm)


class SegmentSequenceSampler(Sampler[int]):
    """Deterministic index sampler consuming a specific sequence of segment indices."""

    def __init__(self, local_indices: Sequence[int]):
        super().__init__()
        self.local_indices = list(local_indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.local_indices)

    def __len__(self) -> int:
        return len(self.local_indices)


def identity_collate(batch: Any) -> Any:
    """Identity collate for dataset items."""
    return batch


def evaluate_native(
    model: nn.Module,
    val_dataset: Any,
    eval_indices: Sequence[int],
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
    base_seed: int = 20260908,
) -> Tuple[float, int]:
    """Evaluate native model on selected validation segments deterministically.

    Saves and restores Python, numpy, PyTorch CPU, and current CUDA device RNG states.
    Applies deterministic per-segment seed (base_seed + seg_idx) so evaluations
    remain strictly layout-independent and reproducible across repeated runs.
    """
    if len(eval_indices) == 0:
        raise ValueError("eval_indices cannot be empty for evaluation")

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    was_training = model.training
    model.eval()

    # Save RNG states
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    cpu_rng_state = torch.get_rng_state()
    dev_idx = device.index if device.type == "cuda" else None
    cuda_rng_state = torch.cuda.get_rng_state(dev_idx) if (device.type == "cuda" and torch.cuda.is_available()) else None

    # Distribute indices among ranks stride-wise
    rank_indices = [idx for i, idx in enumerate(eval_indices) if i % world_size == rank]

    local_loss_sum = 0.0
    local_target_count = 0

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
                    raise ValueError(f"Validation segment {seg_idx} has invalid target_count: {target_count}")

                sample_dev = {}
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        sample_dev[k] = v.to(device)
                    else:
                        sample_dev[k] = v

                out = model(sample_dev)
                loss_sum_val = float(out["loss_sum"].item())
                if not math.isfinite(loss_sum_val):
                    raise FloatingPointError(f"Non-finite loss {loss_sum_val} detected during validation segment {seg_idx}")

                local_loss_sum += loss_sum_val
                local_target_count += target_count

        # Global allreduce for loss_sum and target_count
        if is_dist and world_size > 1:
            stat_tensor = torch.tensor([local_loss_sum, float(local_target_count)], dtype=torch.float64, device=device)
            dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM, group=process_group)
            global_loss_sum = float(stat_tensor[0].item())
            global_targets = int(stat_tensor[1].item())
        else:
            global_loss_sum = local_loss_sum
            global_targets = local_target_count

        if global_targets <= 0:
            raise RuntimeError("Validation evaluated 0 total targets across all ranks")

        mean_loss = global_loss_sum / global_targets
        return mean_loss, global_targets

    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state, dev_idx)
        if was_training:
            model.train()


def check_stop_file_requested(
    stop_file_path: Optional[Union[str, Path]],
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
) -> bool:
    """Check if stop-file exists on rank 0 and broadcast collective decision across all ranks."""
    if stop_file_path is None:
        return False
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    local_stop = 0
    if rank == 0:
        p = Path(stop_file_path)
        if p.is_file():
            local_stop = 1

    if is_dist and world_size > 1:
        dev_to_use = device if (isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
        stop_tensor = torch.tensor([local_stop], dtype=torch.int32, device=dev_to_use)
        dist.broadcast(stop_tensor, src=0, group=process_group)
        return bool(stop_tensor.item() == 1)
    return bool(local_stop == 1)


def evaluate_matched_history_diagnostics(
    model: nn.Module,
    val_dataset: Any,
    dense_val_dataset: Optional[Any],
    eval_indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
    global_step: int,
    epoch: int,
    process_group: Optional[dist.ProcessGroup] = None,
    base_seed: int = 20260908,
    max_diag_segments: int = 8,
) -> Optional[Dict[str, Any]]:
    """Evaluate matched-history diagnostic on a small deterministic validation subset.

    Compares three observation variants on the exact same targets with identical flow seeds:
    1. stream: current stream observation features (new replay groups or stream sample)
    2. dense: dense old dataset without augmentation
    3. current_only: each target observes only its own current frame via single-obs replay groups

    Results are persisted to history_diagnostics.jsonl.
    """
    if len(eval_indices) == 0:
        return None

    if dense_val_dataset is None:
        raise ValueError("dense_val_dataset must be provided for evaluate_matched_history_diagnostics")

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    if type(max_diag_segments) is not int or max_diag_segments <= 0:
        raise ValueError("max_diag_segments must be a positive integer")
    K = len(eval_indices)
    if max_diag_segments == 1:
        subset_indices = [eval_indices[K // 2]]
    elif K <= max_diag_segments:
        subset_indices = list(eval_indices)
    else:
        # Deterministic linspace spread across all tasks/segments (min 8).
        # For a standard 50-task evaluation split (eval_indices 0..49), the 8 positions
        # selected are [0, 7, 14, 21, 28, 35, 42, 49], evenly covering the task distribution.
        spread_positions = [int(round(i * (K - 1) / (max_diag_segments - 1))) for i in range(max_diag_segments)]
        subset_indices = [eval_indices[p] for p in spread_positions]

    was_training = model.training
    model.eval()

    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    cpu_rng_state = torch.get_rng_state()
    dev_idx = device.index if (isinstance(device, torch.device) and device.type == "cuda") else None
    cuda_rng_state = (
        torch.cuda.get_rng_state(dev_idx)
        if (isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available())
        else None
    )

    rank_indices = [idx for i, idx in enumerate(subset_indices) if i % world_size == rank]

    local_stream_loss = 0.0
    local_dense_loss = 0.0
    local_curr_loss = 0.0
    local_targets = 0

    per_segment_records = []

    def set_diag_seed(seg_seed: int) -> None:
        random.seed(seg_seed)
        np.random.seed(seg_seed % (2**32 - 1))
        torch.manual_seed(seg_seed)
        if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(seg_seed)

    def run_sample_forward(sample_data: Dict[str, Any], seg_seed: int) -> Tuple[float, int]:
        set_diag_seed(seg_seed)
        sample_dev = {}
        for k, v in sample_data.items():
            if isinstance(v, torch.Tensor):
                sample_dev[k] = v.to(device)
            else:
                sample_dev[k] = v
        out = model(sample_dev)
        loss_val = float(out["loss_sum"].item())
        target_count = int(out.get("target_count", sample_data.get("target_count", 0)))
        return loss_val, target_count

    try:
        with torch.no_grad():
            for seg_idx in rank_indices:
                seg_seed = base_seed + int(seg_idx)

                # 1. Stream variant
                stream_sample = val_dataset[seg_idx]
                s_loss, s_targets = run_sample_forward(stream_sample, seg_seed)

                # 2. Dense variant
                dense_sample = dense_val_dataset[seg_idx]
                d_loss, d_targets = run_sample_forward(dense_sample, seg_seed)

                # Verify stream and dense targetframeids/count/state/masks/actions torch.equal;
                # assert pervariant counts > 0 all same/finite.
                if s_targets <= 0 or d_targets <= 0 or s_targets != d_targets:
                    raise AssertionError(f"Invalid or mismatched target counts: stream={s_targets}, dense={d_targets}")

                if "target_indices" not in stream_sample or "target_indices" not in dense_sample:
                    raise KeyError("stream_sample and dense_sample must both contain 'target_indices'")
                if "frame_ids" not in stream_sample or "frame_ids" not in dense_sample:
                    raise KeyError("stream_sample and dense_sample must both contain 'frame_ids'")

                s_tgt_fids = [stream_sample["frame_ids"][i] for i in stream_sample["target_indices"]]
                d_tgt_fids = [dense_sample["frame_ids"][i] for i in dense_sample["target_indices"]]
                if s_tgt_fids != d_tgt_fids:
                    raise AssertionError(f"Target frame IDs mismatch: stream={s_tgt_fids} vs dense={d_tgt_fids}")

                for key in ("state", "state_mask", "actions", "action_mask"):
                    s_tensor, d_tensor = stream_sample[key], dense_sample[key]
                    if not isinstance(s_tensor, torch.Tensor) or not isinstance(d_tensor, torch.Tensor):
                        raise TypeError(f"Diagnostic {key} must be a tensor in both variants")
                    if not torch.equal(s_tensor, d_tensor):
                        raise AssertionError(f"Stream and dense {key} tensors are not torch.equal")
                for key in ("episode_id", "prompt", "target_frame_ids"):
                    if stream_sample[key] != dense_sample[key]:
                        raise AssertionError(f"Stream and dense {key} mismatch")

                s_masks = stream_sample.get("image_masks")
                d_masks = dense_sample.get("image_masks")
                if s_masks is not None and d_masks is not None:
                    for s_tidx, d_tidx in zip(stream_sample["target_indices"], dense_sample["target_indices"]):
                        sm = s_masks[s_tidx]
                        dm = d_masks[d_tidx]
                        if isinstance(sm, torch.Tensor) and isinstance(dm, torch.Tensor):
                            if not torch.equal(sm, dm):
                                raise AssertionError("Stream and dense image_masks mismatch on target frames")

                # 3. Current-only variant
                curr_sample = dict(dense_sample)
                tgt_indices = dense_sample["target_indices"]
                if "images_window" not in dense_sample:
                    raise KeyError("dense_sample missing required key 'images_window'")
                curr_sample["images_window"] = [dense_sample["images_window"][i] for i in tgt_indices]
                curr_sample["frame_ids"] = [dense_sample["frame_ids"][i] for i in tgt_indices]
                curr_sample["observation_times"] = [dense_sample["observation_times"][i] for i in tgt_indices]
                curr_sample["target_indices"] = list(range(len(tgt_indices)))
                if "image_masks" in dense_sample and dense_sample["image_masks"] is not None:
                    curr_sample["image_masks"] = [dense_sample["image_masks"][i] for i in tgt_indices]
                curr_sample["replay_groups"] = [
                    {"observation_indices": [i], "target_positions": [i]}
                    for i in range(len(tgt_indices))
                ]
                c_loss, c_targets = run_sample_forward(curr_sample, seg_seed)

                if c_targets != s_targets:
                    raise AssertionError(f"Current-only target count {c_targets} != stream target count {s_targets}")

                if not (math.isfinite(s_loss) and math.isfinite(d_loss) and math.isfinite(c_loss)):
                    raise FloatingPointError(
                        f"Non-finite loss in diagnostic on segment {seg_idx}: stream={s_loss}, dense={d_loss}, curr={c_loss}"
                    )

                local_stream_loss += s_loss
                local_dense_loss += d_loss
                local_curr_loss += c_loss
                local_targets += s_targets

                s_mean = s_loss / s_targets
                d_mean = d_loss / s_targets
                c_mean = c_loss / s_targets

                stream_layout_mode = stream_sample.get(
                    "stream_layout_mode",
                    stream_sample.get("stream_layout", {}).get("mode", "stream")
                    if isinstance(stream_sample.get("stream_layout"), dict)
                    else "stream",
                )

                per_segment_records.append({
                    "segment_index": int(seg_idx),
                    "episode_id": stream_sample.get("episode_id", stream_sample.get("ep_idx", int(seg_idx))),
                    "task_prompt": stream_sample.get("prompt", ""),
                    "target_frame_ids": s_tgt_fids,
                    "actual_stream_layout_mode": stream_layout_mode,
                    "targets": s_targets,
                    "stream_loss": s_mean,
                    "dense_loss": d_mean,
                    "current_only_loss": c_mean,
                    "loss_differences": {
                        "stream_minus_dense": s_mean - d_mean,
                        "current_only_minus_dense": c_mean - d_mean,
                        "stream_minus_current_only": s_mean - c_mean,
                    },
                    "criterion": "no utility claim",
                })

        if is_dist and world_size > 1:
            dev_to_use = device if (isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
            stat_tensor = torch.tensor(
                [local_stream_loss, local_dense_loss, local_curr_loss, float(local_targets)],
                dtype=torch.float64,
                device=dev_to_use,
            )
            dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM, group=process_group)
            global_stream_loss = float(stat_tensor[0].item())
            global_dense_loss = float(stat_tensor[1].item())
            global_curr_loss = float(stat_tensor[2].item())
            global_targets = int(stat_tensor[3].item())

            gathered_records: List[List[Dict[str, Any]]] = [None] * world_size
            dist.all_gather_object(gathered_records, per_segment_records, group=process_group)
            all_records = [r for sublist in gathered_records for r in sublist]
            all_records.sort(key=lambda r: r["segment_index"])
        else:
            global_stream_loss = local_stream_loss
            global_dense_loss = local_dense_loss
            global_curr_loss = local_curr_loss
            global_targets = local_targets
            all_records = per_segment_records

        if rank == 0 and global_targets > 0:
            diag_record = {
                "type": "matched_history_diagnostics",
                "global_step": int(global_step),
                "epoch": int(epoch),
                "num_segments": len(subset_indices),
                "total_targets": global_targets,
                "stream_mean_loss": global_stream_loss / global_targets,
                "dense_mean_loss": global_dense_loss / global_targets,
                "current_only_mean_loss": global_curr_loss / global_targets,
                "segment_records": all_records,
                "timestamp": time.time(),
            }
            diag_path = output_dir / "history_diagnostics.jsonl"
            with open(diag_path, "a", buffering=1) as f:
                f.write(json.dumps(diag_record) + "\n")
            print(
                f"[History Diag @ Step {global_step}] Stream: {global_stream_loss / global_targets:.5f} | "
                f"Dense: {global_dense_loss / global_targets:.5f} | "
                f"Current-only: {global_curr_loss / global_targets:.5f} (targets: {global_targets})"
            )
            sys.stdout.flush()
            return diag_record
        return None

    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None and isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state, dev_idx)
        if was_training:
            model.train()


def build_checkpoint_payload(
    policy: nn.Module,
    config: Dict[str, Any],
    norm_stats: Dict[str, Any],
    training_contract: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    global_step: int,
    epoch: int,
    batch_cursor: int,
    epoch_targets_seen: int,
    source_metadata: Dict[str, Any],
    all_rng_states: List[Any],
    stage_lineage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a standard, complete native checkpoint payload dictionary."""
    raw_policy = getattr(policy, "policy", policy)
    state_dict = {k: v.cpu() for k, v in raw_policy.state_dict().items()}

    eff_lineage = stage_lineage if stage_lineage is not None else source_metadata.get("stage_lineage")

    payload = {
        "model": state_dict,
        "config": config,
        "norm_stats": norm_stats,
        "training_contract": training_contract,
        "train_data_contract": train_data_contract,
        "val_data_contract": val_data_contract,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_cursor": int(batch_cursor),
        "epoch_targets_seen": int(epoch_targets_seen),
        "source_metadata": source_metadata,
        "rng_states_per_rank": all_rng_states,
    }
    if eff_lineage is not None:
        payload["stage_lineage"] = eff_lineage
    return payload


def save_native_checkpoint(
    output_dir: Path,
    filename: str,
    policy: nn.Module,
    config: Dict[str, Any],
    norm_stats: Dict[str, Any],
    training_contract: Dict[str, Any],
    train_data_contract: Dict[str, Any],
    val_data_contract: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    global_step: int,
    epoch: int,
    batch_cursor: int,
    epoch_targets_seen: int,
    source_metadata: Dict[str, Any],
    device: Optional[torch.device] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    update_last: bool = False,
    stage_lineage: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save full unwrapped checkpoint atomically on rank 0 with all ranks contributing RNG states.

    If update_last is True, also atomically updates last.pt (via hardlink or atomic copy).
    """
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    dev_idx = device.index if (device is not None and isinstance(device, torch.device) and device.type == "cuda") else None
    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(dev_idx) if (device is not None and isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()) else None,
    }

    # Gather RNG states from all ranks to rank 0
    all_rngs = [None] * world_size if rank == 0 else None
    if is_dist and world_size > 1:
        dist.gather_object(local_rng, all_rngs if rank == 0 else None, dst=0, group=process_group)
    else:
        all_rngs = [local_rng]

    out_file = output_dir / filename
    if rank == 0:
        eff_lineage = stage_lineage if stage_lineage is not None else source_metadata.get("stage_lineage")
        payload = build_checkpoint_payload(
            policy=policy,
            config=config,
            norm_stats=norm_stats,
            training_contract=training_contract,
            train_data_contract=train_data_contract,
            val_data_contract=val_data_contract,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            batch_cursor=batch_cursor,
            epoch_targets_seen=epoch_targets_seen,
            source_metadata=source_metadata,
            all_rng_states=all_rngs,
            stage_lineage=eff_lineage,
        )

        tmp_file = output_dir / f".tmp_{filename}_{os.getpid()}_{int(time.time()*1000)}"
        torch.save(payload, str(tmp_file))
        os.replace(str(tmp_file), str(out_file))

        if update_last:
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


def load_native_training_checkpoint(
    checkpoint_path: Union[str, Path],
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    expected_source_metadata: Dict[str, Any],
    expected_training_contract: Dict[str, Any],
    expected_train_data_contract: Dict[str, Any],
    expected_val_data_contract: Dict[str, Any],
    device: Optional[torch.device] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    train_dataset: Optional[Any] = None,
    max_epochs: Optional[int] = None,
    mode: str = "resume",
    stream_protocol: Optional[str] = None,
    parent_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Strictly load native training checkpoint for exact deterministic continuation or explicit transition."""
    if mode not in ("resume", "transition"):
        raise ValueError(f"Unknown loading mode {mode!r}; expected 'resume' or 'transition'.")

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if is_dist else 0
    world_size = dist.get_world_size(group=process_group) if is_dist else 1

    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # 1. Validate required keys exist
    for key in (
        "training_contract",
        "train_data_contract",
        "val_data_contract",
        "source_metadata",
        "model",
        "optimizer",
        "scheduler",
        "global_step",
        "epoch",
        "batch_cursor",
    ):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing required contract field '{key}'")

    # 2. Strict source metadata validation
    ckpt_src = ckpt["source_metadata"]
    if ckpt_src.get("checkpoint_sha256") != expected_source_metadata.get("checkpoint_sha256"):
        raise ValueError(
            f"Source checkpoint SHA256 mismatch: {ckpt_src.get('checkpoint_sha256')} != {expected_source_metadata.get('checkpoint_sha256')}"
        )

    # 3. Strict training and data contract validation BEFORE mutating model or optimizer
    ignored_training_keys = {"workers", "max_updates", "save_every", "eval_every", "val_per_task"}
    ckpt_tr = ckpt["training_contract"]
    ckpt_filtered = {k: v for k, v in ckpt_tr.items() if k not in ignored_training_keys}

    if mode == "resume":
        exp_filtered = {k: v for k, v in expected_training_contract.items() if k not in ignored_training_keys}
        if ckpt_filtered != exp_filtered:
            for k in sorted(set(ckpt_filtered.keys()) | set(exp_filtered.keys())):
                if ckpt_filtered.get(k) != exp_filtered.get(k):
                    raise ValueError(
                        f"Training contract mismatch on '{k}': checkpoint has {ckpt_filtered.get(k)!r}, expected {exp_filtered.get(k)!r}"
                    )

        for name, ckpt_dc, exp_dc in [
            ("train_data_contract", ckpt["train_data_contract"], expected_train_data_contract),
            ("val_data_contract", ckpt["val_data_contract"], expected_val_data_contract),
        ]:
            if ckpt_dc != exp_dc:
                for k in sorted(set(ckpt_dc.keys()) | set(exp_dc.keys())):
                    if ckpt_dc.get(k) != exp_dc.get(k):
                        raise ValueError(
                            f"{name} mismatch on '{k}': checkpoint has {ckpt_dc.get(k)!r}, expected {exp_dc.get(k)!r}"
                        )
    elif mode == "transition":
        # Explicit transition:
        # 1) Dense original native_multiframe_v1 -> stream_replay_v1, memory_replay_v1, or compact_memory_replay_v1
        # 2) Memory replay native_memory_replay_v1 (v1.2) -> compact_memory_replay_v1
        source_format = ckpt_tr.get("format")
        if stream_protocol not in ("stream_replay_v1", "memory_replay_v1", "compact_memory_replay_v1"):
            raise ValueError(
                f"Transition requires stream_protocol in ('stream_replay_v1', 'memory_replay_v1', 'compact_memory_replay_v1'), got {stream_protocol!r}"
            )

        real_protocol, expected_format = get_protocol_contract_and_format(stream_protocol)
        if expected_training_contract.get("format") != expected_format:
            raise ValueError(
                f"Transition requires target training contract format {expected_format!r}, got {expected_training_contract.get('format')!r}"
            )
        if expected_training_contract.get("stream_protocol") != real_protocol:
            raise ValueError(
                f"Expected training contract stream_protocol does not match implemented contract: "
                f"{expected_training_contract.get('stream_protocol')} != {real_protocol}"
            )

        if parent_checkpoint_sha256 is None:
            if rank == 0:
                parent_checkpoint_sha256 = compute_file_sha256(ckpt_path)
            if is_dist and world_size > 1:
                sha_list = [parent_checkpoint_sha256]
                dist.broadcast_object_list(sha_list, src=0, group=process_group)
                parent_checkpoint_sha256 = sha_list[0]

        if source_format == "native_multiframe_v1":
            # Dense -> new protocol
            # Full old training contract == expected new with only format restored + stream_protocol removed
            exp_filtered_old = {
                k: v for k, v in expected_training_contract.items()
                if k not in ignored_training_keys and k not in ("format", "stream_protocol")
            }
            exp_filtered_old["format"] = "native_multiframe_v1"

            if ckpt_filtered != exp_filtered_old:
                for k in sorted(set(ckpt_filtered.keys()) | set(exp_filtered_old.keys())):
                    if ckpt_filtered.get(k) != exp_filtered_old.get(k):
                        raise ValueError(
                            f"Transition training contract mismatch on '{k}': parent checkpoint has {ckpt_filtered.get(k)!r}, expected {exp_filtered_old.get(k)!r}"
                        )

            # Old train/val data contracts == newdataset.get_base_data_contract() exactly
            for name, ckpt_dc, exp_dc in [
                ("train_data_contract", ckpt["train_data_contract"], expected_train_data_contract),
                ("val_data_contract", ckpt["val_data_contract"], expected_val_data_contract),
            ]:
                if ckpt_dc != exp_dc:
                    for k in sorted(set(ckpt_dc.keys()) | set(exp_dc.keys())):
                        if ckpt_dc.get(k) != exp_dc.get(k):
                            raise ValueError(
                                f"Transition base {name} mismatch on '{k}': parent checkpoint has {ckpt_dc.get(k)!r}, expected {exp_dc.get(k)!r}"
                            )

        elif source_format == "native_memory_replay_v1" and stream_protocol == "compact_memory_replay_v1":
            # Memory v1.2 -> Compact memory transition
            # Validate parent stream_protocol equals memory_protocol contract
            expected_parent_mem_protocol = get_memory_protocol_contract()
            if ckpt_tr.get("stream_protocol") != expected_parent_mem_protocol:
                raise ValueError(
                    f"Transition from native_memory_replay_v1 requires parent stream_protocol to match get_memory_protocol_contract()"
                )

            # Training contract check: only format and stream_protocol differ (+ ignored_training_keys)
            exp_filtered_old = {
                k: v for k, v in expected_training_contract.items()
                if k not in ignored_training_keys and k not in ("format", "stream_protocol")
            }
            exp_filtered_old["format"] = "native_memory_replay_v1"
            exp_filtered_old["stream_protocol"] = expected_parent_mem_protocol

            if ckpt_filtered != exp_filtered_old:
                for k in sorted(set(ckpt_filtered.keys()) | set(exp_filtered_old.keys())):
                    if ckpt_filtered.get(k) != exp_filtered_old.get(k):
                        raise ValueError(
                            f"Transition training contract mismatch on '{k}': parent checkpoint has {ckpt_filtered.get(k)!r}, expected {exp_filtered_old.get(k)!r}"
                        )

            # Data contract check:
            # Reconstruct expected memory data contracts from new dataset's base data contracts
            for name, ckpt_dc, base_dc in [
                ("train_data_contract", ckpt["train_data_contract"], expected_train_data_contract),
                ("val_data_contract", ckpt["val_data_contract"], expected_val_data_contract),
            ]:
                # Reconstruct memory prior data contract from base contract
                expected_prior_dc = dict(base_dc)
                expected_prior_dc["stream_protocol"] = expected_parent_mem_protocol
                sorted_mem_json = json.dumps(expected_parent_mem_protocol, sort_keys=True)
                prior_dfp_raw = f"{base_dc['data_fingerprint']}:{sorted_mem_json}"
                expected_prior_dc["data_fingerprint"] = hashlib.sha256(prior_dfp_raw.encode("utf-8")).hexdigest()

                if ckpt_dc != expected_prior_dc:
                    for k in sorted(set(ckpt_dc.keys()) | set(expected_prior_dc.keys())):
                        if ckpt_dc.get(k) != expected_prior_dc.get(k):
                            raise ValueError(
                                f"Transition parent {name} mismatch on '{k}': parent checkpoint has {ckpt_dc.get(k)!r}, expected {expected_prior_dc.get(k)!r}"
                            )
        else:
            raise ValueError(
                f"Transition requires source checkpoint format 'native_multiframe_v1' (or 'native_memory_replay_v1' -> 'compact_memory_replay_v1'), got source format {source_format!r} to target {stream_protocol!r}"
            )

    # 4. Strict scheduler last_epoch vs checkpoint global_step check
    global_step = int(ckpt["global_step"])
    ckpt_sched = ckpt["scheduler"]
    sched_last_epoch = ckpt_sched.get("last_epoch")
    if sched_last_epoch is not None and sched_last_epoch != global_step:
        raise ValueError(
            f"Scheduler last_epoch ({sched_last_epoch}) mismatch with checkpoint global_step ({global_step})"
        )

    # 5. Validate RNG states per rank length
    rng_per_rank = ckpt.get("rng_states_per_rank", [])
    if len(rng_per_rank) != world_size:
        raise ValueError(
            f"Resuming with world_size={world_size} but checkpoint has {len(rng_per_rank)} rank RNG states."
        )

    # 6. Validate resume state BEFORE mutating policy, optimizer, scheduler
    if train_dataset is not None:
        validate_resume_state(
            total_segments=len(train_dataset),
            global_batch_size=expected_training_contract["global_batch_size"],
            seed=expected_training_contract["seed"],
            epoch=int(ckpt["epoch"]),
            cursor=int(ckpt["batch_cursor"]),
            step=global_step,
            epoch_targets_seen=int(ckpt.get("epoch_targets_seen", 0)),
            train_dataset=train_dataset,
            max_epochs=max_epochs,
        )

    # 7. Load model state dict strictly
    raw_policy = getattr(policy, "policy", policy)
    model_state = ckpt["model"]
    incompatible = raw_policy.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint loading failed! Missing keys: {incompatible.missing_keys}, Unexpected: {incompatible.unexpected_keys}"
        )

    # 8. Load optimizer & scheduler state
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    # 9. Restore RNG for this rank
    my_rng = rng_per_rank[rank]
    random.setstate(my_rng["python"])
    np.random.set_state(my_rng["numpy"])
    torch.set_rng_state(my_rng["torch_cpu"])
    dev_idx = device.index if (device is not None and isinstance(device, torch.device) and device.type == "cuda") else None
    if my_rng.get("torch_cuda") is not None and device is not None and isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(my_rng["torch_cuda"], dev_idx)

    # 10. Extract / construct stage lineage
    stage_lineage = None
    if mode == "transition":
        trans_ts = time.time() if rank == 0 else None
        if is_dist and world_size > 1:
            ts_list = [trans_ts]
            dist.broadcast_object_list(ts_list, src=0, group=process_group)
            trans_ts = ts_list[0]

        stage_lineage = {
            "parent_checkpoint_path": str(ckpt_path),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "from_protocol": ckpt_tr.get("format", "native_multiframe_v1"),
            "to_protocol": stream_protocol or "stream_replay_v1",
            "parent_global_step": global_step,
            "parent_epoch": int(ckpt["epoch"]),
            "parent_batch_cursor": int(ckpt["batch_cursor"]),
            "parent_epoch_targets_seen": int(ckpt.get("epoch_targets_seen", 0)),
            "transition_timestamp": trans_ts,
        }
    elif mode == "resume":
        stage_lineage = ckpt.get("source_metadata", {}).get("stage_lineage") or ckpt.get("stage_lineage")

    return {
        "global_step": global_step,
        "epoch": int(ckpt["epoch"]),
        "batch_cursor": int(ckpt["batch_cursor"]),
        "epoch_targets_seen": int(ckpt.get("epoch_targets_seen", 0)),
        "stage_lineage": stage_lineage,
    }


def transition_native_checkpoint(
    checkpoint_path: Union[str, Path],
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    expected_source_metadata: Dict[str, Any],
    expected_training_contract: Dict[str, Any],
    expected_train_data_contract: Dict[str, Any],
    expected_val_data_contract: Dict[str, Any],
    stream_protocol: str = "stream_replay_v1",
    device: Optional[torch.device] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    train_dataset: Optional[Any] = None,
    max_epochs: Optional[int] = None,
    parent_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Explicitly transition a dense native_multiframe_v1 (or memory_replay_v1) checkpoint into stream_replay_v1, memory_replay_v1, or compact_memory_replay_v1."""
    return load_native_training_checkpoint(
        checkpoint_path=checkpoint_path,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_source_metadata=expected_source_metadata,
        expected_training_contract=expected_training_contract,
        expected_train_data_contract=expected_train_data_contract,
        expected_val_data_contract=expected_val_data_contract,
        device=device,
        process_group=process_group,
        train_dataset=train_dataset,
        max_epochs=max_epochs,
        mode="transition",
        stream_protocol=stream_protocol,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def compute_epoch_batches(
    total_segments: int,
    global_batch_size: int,
    world_size: int,
    seed: int,
    epoch: int,
) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    """Deterministic epoch segment partitioning helper.

    Returns:
        perm: full permutation of segment indices for this epoch
        global_batches: list of global batch index lists
        per_rank_local_indices: list of len(world_size), each having all segment indices for that rank in epoch order
    """
    epoch_generator = torch.Generator()
    epoch_generator.manual_seed(seed + epoch * 37)
    perm = torch.randperm(total_segments, generator=epoch_generator).tolist()

    global_batches: List[List[int]] = []
    per_rank_indices: List[List[int]] = [[] for _ in range(world_size)]

    cursor = 0
    while cursor < total_segments:
        bslice = perm[cursor : min(cursor + global_batch_size, total_segments)]
        global_batches.append(bslice)
        for r in range(world_size):
            per_rank_indices[r].extend(bslice[r::world_size])
        cursor += len(bslice)

    return perm, global_batches, per_rank_indices


def validate_resume_state(
    total_segments: int,
    global_batch_size: int,
    seed: int,
    epoch: int,
    cursor: int,
    step: int,
    epoch_targets_seen: int,
    train_dataset: Any,
    max_epochs: Optional[int] = None,
) -> None:
    """Validate that resume cursor and epoch targets match deterministic segment ordering."""
    if epoch < 0:
        raise ValueError(f"Epoch {epoch} cannot be negative")
    if max_epochs is not None and epoch > max_epochs:
        raise ValueError(f"Epoch {epoch} exceeds max_epochs {max_epochs}")
    if max_epochs is not None and epoch == max_epochs:
        if cursor != 0 or epoch_targets_seen != 0:
            raise ValueError(
                f"Completed run at epoch {epoch} requires cursor=0 and epoch_targets_seen=0, "
                f"got cursor={cursor}, epoch_targets_seen={epoch_targets_seen}"
            )
    if cursor < 0 or cursor > total_segments:
        raise ValueError(f"Invalid cursor {cursor} for total_segments {total_segments}")
    if cursor % global_batch_size != 0 and cursor != total_segments:
        raise ValueError(
            f"Cursor {cursor} must align with global_batch_size {global_batch_size} boundary or len {total_segments}"
        )

    steps_per_epoch = math.ceil(total_segments / global_batch_size)
    batches_in_curr_epoch = math.ceil(cursor / global_batch_size)
    expected_step = epoch * steps_per_epoch + batches_in_curr_epoch
    if step != expected_step:
        raise ValueError(
            f"Resume step mismatch: step {step} != expected {expected_step} "
            f"(epoch={epoch}, cursor={cursor}, steps_per_epoch={steps_per_epoch})"
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
            _, s, e = train_dataset.segments[s_idx]
            expected_targets += (e - s)

    if epoch_targets_seen != expected_targets:
        raise ValueError(
            f"Resume state mismatch: recorded epoch_targets_seen {epoch_targets_seen} != "
            f"expected {expected_targets} from {cursor} processed segments"
        )


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate CLI arguments strictly before any CUDA initialization or file mutation."""
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be a positive integer, got {args.epochs}")
    if args.history_frames <= 0:
        raise ValueError(f"--history-frames must be a positive integer, got {args.history_frames}")
    if args.target_frames <= 0:
        raise ValueError(f"--target-frames must be a positive integer, got {args.target_frames}")
    if args.target_frames > args.history_frames:
        raise ValueError(
            f"--target-frames ({args.target_frames}) must be <= --history-frames ({args.history_frames})"
        )
    if args.global_batch_size <= 0:
        raise ValueError(f"--global-batch-size must be a positive integer, got {args.global_batch_size}")
    if args.workers < 0:
        raise ValueError(f"--workers must be non-negative, got {args.workers}")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError(f"--val-fraction must be in (0, 1), got {args.val_fraction}")
    if not (math.isfinite(args.lr_vision) and args.lr_vision > 0.0):
        raise ValueError(f"--lr-vision must be finite and positive, got {args.lr_vision}")
    if not (math.isfinite(args.lr_projector) and args.lr_projector > 0.0):
        raise ValueError(f"--lr-projector must be finite and positive, got {args.lr_projector}")
    if not (math.isfinite(args.lr_llm) and args.lr_llm > 0.0):
        raise ValueError(f"--lr-llm must be finite and positive, got {args.lr_llm}")
    if not (math.isfinite(args.lr_head) and args.lr_head > 0.0):
        raise ValueError(f"--lr-head must be finite and positive, got {args.lr_head}")
    if not (math.isfinite(args.weight_decay) and args.weight_decay >= 0.0):
        raise ValueError(f"--weight-decay must be finite and non-negative, got {args.weight_decay}")
    if args.warmup_updates < 0:
        raise ValueError(f"--warmup-updates must be non-negative, got {args.warmup_updates}")
    if not (math.isfinite(args.grad_clip) and args.grad_clip > 0.0):
        raise ValueError(f"--grad-clip must be finite and positive, got {args.grad_clip}")
    if args.save_every <= 0:
        raise ValueError(f"--save-every must be a positive integer, got {args.save_every}")
    if args.eval_every <= 0:
        raise ValueError(f"--eval-every must be a positive integer, got {args.eval_every}")
    if args.val_per_task <= 0:
        raise ValueError(f"--val-per-task must be a positive integer, got {args.val_per_task}")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError(f"--max-updates must be positive, got {args.max_updates}")
    if args.max_train_episodes is not None and args.max_train_episodes <= 0:
        raise ValueError(f"--max-train-episodes must be positive, got {args.max_train_episodes}")
    if args.max_val_episodes is not None and args.max_val_episodes <= 0:
        raise ValueError(f"--max-val-episodes must be positive, got {args.max_val_episodes}")
    if args.resume is not None and args.transition_from is not None:
        raise ValueError("--resume and --transition-from are mutually exclusive.")
    if args.transition_from is not None and args.stream_protocol not in ("stream_replay_v1", "memory_replay_v1", "compact_memory_replay_v1"):
        raise ValueError("--transition-from requires --stream-protocol stream_replay_v1, memory_replay_v1, or compact_memory_replay_v1.")
    if args.stop_file is not None and not str(args.stop_file).strip():
        raise ValueError("--stop-file must be a non-empty string path if provided.")


def parse_args(raw_args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for train_native."""
    parser = argparse.ArgumentParser(
        description="Synchronous distributed training for native FabriVLA sequence policy."
    )
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt")
    parser.add_argument("--vlm-path", type=str, default="/root/models/InternVL3_5-1B")
    parser.add_argument("--data-root", type=str, default="/root/evo1_metaworld_dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--target-frames", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4042)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lr-vision", type=float, default=1e-6)
    parser.add_argument("--lr-projector", type=float, default=5e-6)
    parser.add_argument("--lr-llm", type=float, default=2e-6)
    parser.add_argument("--lr-head", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-per-task", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--transition-from", type=str, default=None)
    parser.add_argument(
        "--stream-protocol",
        type=str,
        default=None,
        choices=["stream_replay_v1", "memory_replay_v1", "compact_memory_replay_v1"],
        help="Stream observation replay protocol (stream_replay_v1, memory_replay_v1, or compact_memory_replay_v1).",
    )
    parser.add_argument("--stop-file", type=str, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--no-augmentation", action="store_true", default=False)
    parser.add_argument("--no-gradient-checkpointing", action="store_true", default=False)

    return parser.parse_args(raw_args)


def run_training(args: argparse.Namespace, device: Optional[torch.device] = None) -> None:
    """Core training loop execution."""
    # 1. Early CLI argument validation BEFORE any CUDA or filesystem mutation
    validate_cli_arguments(args)

    output_dir = Path(args.output_dir).resolve()
    run_config_path = output_dir / "run_config.json"

    # Pre-check output directory and run_config
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.transition_from is not None:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is non-empty, but --transition-from requires a NEW EMPTY output directory."
            )
        if not args.resume:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is non-empty, but --resume was not specified."
            )
        if not run_config_path.exists():
            raise ValueError(
                f"Cannot resume in {output_dir}: missing run_config.json provenance file."
            )

    # 2. Distributed environment setup
    if device is None:
        is_distributed = "WORLD_SIZE" in os.environ
        if is_distributed:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            world_size = 1
            rank = 0
            local_rank = 0
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
            else:
                raise RuntimeError("CUDA is strictly required for formal train_native execution.")
    else:
        is_distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0
        local_rank = 0

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if is_distributed and world_size > 1:
        dist.barrier()

    # Per-rank RNG seed initialization
    per_rank_seed = args.seed + rank * 10007
    torch.manual_seed(per_rank_seed)
    random.seed(per_rank_seed)
    np.random.seed(per_rank_seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(per_rank_seed)

    # 3. Lazy imports
    from fabri_moss.native_data import NativeTrainingDataset
    from fabri_moss.native_training import NativeSequencePolicy
    if args.stream_protocol == "memory_replay_v1":
        from fabri_moss.memory_training import NativeMemorySequencePolicy
        NativeCompactMemorySequencePolicy = None
    elif args.stream_protocol == "compact_memory_replay_v1":
        from fabri_moss.compact_training import NativeCompactMemorySequencePolicy
        NativeMemorySequencePolicy = None
    else:
        NativeMemorySequencePolicy = None
        NativeCompactMemorySequencePolicy = None
    from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint

    if rank == 0:
        print(f"[Init] Loading native FabriVLA checkpoint from {args.checkpoint} (trainable=True)...")

    base_policy, ckpt_config, norm_stats, base_metadata = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm_path,
        device=str(device),
        arm_key="metaworld_sawyer",
        trainable=True,
    )

    # Verify device and FlashAttention-2
    require_training_device(base_policy, str(device))
    fa2_diag = assert_native_fa2(base_policy)
    if rank == 0:
        print(f"[Init] Native FA2 verified: {fa2_diag}")

    # Inspect policy config for dimensions and shallow layer
    action_head = getattr(base_policy, "action_head", None)
    head_config = getattr(action_head, "config", None) if action_head is not None else None
    act_config = getattr(base_policy, "config", head_config)

    shallow_layer = getattr(act_config, "shallow_layer_index", 6)
    use_timestamps = True  # explicit current protocol: True
    state_dim = getattr(act_config, "state_dim", 24)
    if head_config is not None:
        action_dim = getattr(head_config, "per_action_dim", getattr(act_config, "per_action_dim", getattr(act_config, "action_dim", 24)))
        horizon = getattr(head_config, "horizon", getattr(act_config, "horizon", 50))
    else:
        action_dim = getattr(act_config, "per_action_dim", getattr(act_config, "action_dim", 24))
        horizon = getattr(act_config, "horizon", 50)

    gradient_checkpointing = not args.no_gradient_checkpointing
    if args.stream_protocol == "memory_replay_v1":
        policy_wrapper_cls = NativeMemorySequencePolicy
    elif args.stream_protocol == "compact_memory_replay_v1":
        policy_wrapper_cls = NativeCompactMemorySequencePolicy
    else:
        policy_wrapper_cls = NativeSequencePolicy
    seq_model = policy_wrapper_cls(
        policy=base_policy,
        shallow_layer=shallow_layer,
        use_timestamps=use_timestamps,
        gradient_checkpointing=gradient_checkpointing,
    )
    seq_model.to(device)

    # Strict FP32 trainability check: require all parameters enabled
    trainable_count = 0
    for name, p in seq_model.named_parameters():
        if not p.requires_grad:
            raise RuntimeError(
                f"Parameter {name} has requires_grad=False; all parameters must be enabled for native training."
            )
        if p.dtype != torch.float32:
            raise TypeError(f"Trainable parameter {name} has dtype {p.dtype}, strictly expected float32")
        trainable_count += 1

    if rank == 0:
        print(f"[Init] Model parameter status: {trainable_count} trainable (FP32), 0 non-trainable")

    # Broadcast initial parameters/buffers to ensure strict identical weights
    if is_distributed and world_size > 1:
        for p in seq_model.parameters():
            dist.broadcast(p.data, src=0)
        for b in seq_model.buffers():
            dist.broadcast(b.data, src=0)

    # 4. Datasets and Contracts
    dataset_kwargs = {}
    if args.stream_protocol is not None:
        dataset_kwargs["stream_protocol"] = args.stream_protocol

    train_dataset = NativeTrainingDataset(
        root=args.data_root,
        norm_stats=norm_stats,
        history_frames=args.history_frames,
        target_frames=args.target_frames,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        split="train",
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_episodes=args.max_train_episodes,
        augmentation=not args.no_augmentation,
        **dataset_kwargs,
    )

    val_dataset = NativeTrainingDataset(
        root=args.data_root,
        norm_stats=norm_stats,
        history_frames=args.history_frames,
        target_frames=args.target_frames,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        split="val",
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_episodes=args.max_val_episodes,
        augmentation=False,
        **dataset_kwargs,
    )

    dense_val_dataset = None
    if args.stream_protocol in ("stream_replay_v1", "memory_replay_v1", "compact_memory_replay_v1"):
        dense_val_dataset = NativeTrainingDataset(
            root=args.data_root,
            norm_stats=norm_stats,
            history_frames=args.history_frames,
            target_frames=args.target_frames,
            horizon=horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            split="val",
            seed=args.seed,
            val_fraction=args.val_fraction,
            max_episodes=args.max_val_episodes,
            augmentation=False,
        )

    train_contract = train_dataset.get_data_contract()
    val_contract = val_dataset.get_data_contract()

    is_transition = bool(args.transition_from)
    if is_transition:
        if not hasattr(train_dataset, "get_base_data_contract") or not hasattr(val_dataset, "get_base_data_contract"):
            raise AttributeError("Transition requires dataset to implement get_base_data_contract()")
        base_train_contract = train_dataset.get_base_data_contract()
        base_val_contract = val_dataset.get_base_data_contract()
    else:
        base_train_contract = None
        base_val_contract = None

    total_segments = len(train_dataset)
    steps_per_epoch = math.ceil(total_segments / args.global_batch_size)
    total_scheduler_updates = args.epochs * steps_per_epoch

    # 5. Optimizer and Scheduler
    optimizer, scheduler = create_native_optimizer_and_scheduler(
        policy=seq_model.policy,
        lr_vision=args.lr_vision,
        lr_projector=args.lr_projector,
        lr_llm=args.lr_llm,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        total_steps=total_scheduler_updates,
        warmup_steps=args.warmup_updates,
    )

    training_contract = {
        "format": "native_multiframe_v1",
        "epochs": args.epochs,
        "global_batch_size": args.global_batch_size,
        "world_size": world_size,
        "seed": args.seed,
        "use_timestamps": use_timestamps,
        "history_frames": args.history_frames,
        "target_frames": args.target_frames,
        "lr_vision": args.lr_vision,
        "lr_projector": args.lr_projector,
        "lr_llm": args.lr_llm,
        "lr_head": args.lr_head,
        "warmup_updates": args.warmup_updates,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "total_scheduler_updates": total_scheduler_updates,
        "gradient_checkpointing": gradient_checkpointing,
        "augmentation": not args.no_augmentation,
    }
    if args.stream_protocol is not None:
        protocol_contract, contract_format = get_protocol_contract_and_format(args.stream_protocol)
        training_contract["format"] = contract_format
        training_contract["stream_protocol"] = protocol_contract

    current_source_metadata = dict(base_metadata)
    start_step = 0
    start_epoch = 0
    start_cursor = 0
    epoch_targets_seen = 0

    if args.resume:
        # Check existing run_config.json before mutating state
        if run_config_path.exists():
            existing_cfg = json.loads(run_config_path.read_text())
            # Validate source SHA
            if existing_cfg.get("source_metadata", {}).get("checkpoint_sha256") != base_metadata.get("checkpoint_sha256"):
                raise ValueError("Resume run_config.json source checkpoint SHA256 mismatch")
            # Validate training contract (excluding non-runtime keys)
            ignored_keys = {"workers", "max_updates", "save_every", "eval_every", "val_per_task"}
            cfg_tr = {k: v for k, v in existing_cfg.get("training_contract", {}).items() if k not in ignored_keys}
            exp_tr = {k: v for k, v in training_contract.items() if k not in ignored_keys}
            if cfg_tr != exp_tr:
                raise ValueError(f"Resume run_config.json training contract mismatch: {cfg_tr} != {exp_tr}")
            # Validate data contracts (full equality)
            if existing_cfg.get("train_data_contract") != train_contract:
                raise ValueError("Resume run_config.json train_data_contract mismatch")
            if existing_cfg.get("val_data_contract") != val_contract:
                raise ValueError("Resume run_config.json val_data_contract mismatch")

        if rank == 0:
            print(f"[Resume] Loading checkpoint from {args.resume}...")

        resume_info = load_native_training_checkpoint(
            checkpoint_path=args.resume,
            policy=seq_model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_source_metadata=current_source_metadata,
            expected_training_contract=training_contract,
            expected_train_data_contract=train_contract,
            expected_val_data_contract=val_contract,
            device=device,
            train_dataset=train_dataset,
            max_epochs=args.epochs,
            mode="resume",
        )
        start_step = resume_info["global_step"]
        start_epoch = resume_info["epoch"]
        start_cursor = resume_info["batch_cursor"]
        epoch_targets_seen = resume_info["epoch_targets_seen"]
        if resume_info.get("stage_lineage") is not None:
            current_source_metadata["stage_lineage"] = resume_info["stage_lineage"]

        # Guard: if already done, exit cleanly without modifying files or adding resume events
        if (args.max_updates is not None and start_step >= args.max_updates) or (start_epoch >= args.epochs):
            if rank == 0:
                print(f"[Done] Already completed target updates/epochs (step={start_step}, epoch={start_epoch}). Exiting.")
            if is_distributed and world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
            return

        if rank == 0:
            print(f"[Resume] Successfully resumed at step={start_step}, epoch={start_epoch}, cursor={start_cursor}")
            resume_event = {
                "resumed_from": str(args.resume),
                "step": start_step,
                "epoch": start_epoch,
                "cursor": start_cursor,
                "timestamp": time.time(),
            }
            if run_config_path.exists():
                rc = json.loads(run_config_path.read_text())
                rc.setdefault("resume_events", []).append(resume_event)
                run_config_path.write_text(json.dumps(rc, indent=2))
            else:
                run_config = {
                    "training_contract": training_contract,
                    "train_data_contract": train_contract,
                    "val_data_contract": val_contract,
                    "source_metadata": current_source_metadata,
                    "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                    "startup_info": {
                        "world_size": world_size,
                        "trainable_parameters": trainable_count,
                        "total_scheduler_updates": total_scheduler_updates,
                    },
                    "resume_events": [resume_event],
                }
                run_config_path.write_text(json.dumps(run_config, indent=2))

    elif args.transition_from:
        if rank == 0:
            print(f"[Transition] Transitioning from dense checkpoint {args.transition_from} to stream protocol {args.stream_protocol}...")

        resume_info = transition_native_checkpoint(
            checkpoint_path=args.transition_from,
            policy=seq_model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_source_metadata=current_source_metadata,
            expected_training_contract=training_contract,
            expected_train_data_contract=base_train_contract,
            expected_val_data_contract=base_val_contract,
            stream_protocol=args.stream_protocol,
            device=device,
            train_dataset=train_dataset,
            max_epochs=args.epochs,
            parent_checkpoint_sha256=None,
        )
        start_step = resume_info["global_step"]
        start_epoch = resume_info["epoch"]
        start_cursor = resume_info["batch_cursor"]
        epoch_targets_seen = resume_info["epoch_targets_seen"]
        if resume_info.get("stage_lineage") is not None:
            current_source_metadata["stage_lineage"] = resume_info["stage_lineage"]

        if (args.max_updates is not None and start_step >= args.max_updates) or (start_epoch >= args.epochs):
            if rank == 0:
                print(f"[Done] Transition source already reached target updates/epochs (step={start_step}, epoch={start_epoch}). Exiting.")
            if is_distributed and world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
            return

        if rank == 0:
            run_config = {
                "training_contract": training_contract,
                "train_data_contract": train_contract,
                "val_data_contract": val_contract,
                "source_metadata": current_source_metadata,
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "startup_info": {
                    "world_size": world_size,
                    "trainable_parameters": trainable_count,
                    "total_scheduler_updates": total_scheduler_updates,
                },
                "transition_info": current_source_metadata.get("stage_lineage", {}),
            }
            run_config_path.write_text(json.dumps(run_config, indent=2))
            print(f"[Transition] Transition successful: starting at step={start_step}, epoch={start_epoch}, cursor={start_cursor}")

    else:
        # Fresh start: write initial run_config.json on rank 0
        if rank == 0:
            run_config = {
                "training_contract": training_contract,
                "train_data_contract": train_contract,
                "val_data_contract": val_contract,
                "source_metadata": current_source_metadata,
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "startup_info": {
                    "world_size": world_size,
                    "trainable_parameters": trainable_count,
                    "total_scheduler_updates": total_scheduler_updates,
                },
            }
            run_config_path.write_text(json.dumps(run_config, indent=2))

    metrics_log_path = output_dir / "train_metrics.jsonl"
    metrics_file = open(metrics_log_path, "a", buffering=1) if rank == 0 else None

    # Check stop file before initial validation and first update
    if check_stop_file_requested(args.stop_file, device):
        if rank == 0:
            print(f"[StopFile] Stop requested before first update via {args.stop_file}. Exiting cleanly.")
            sys.stdout.flush()
        save_native_checkpoint(
            output_dir=output_dir,
            filename="last.pt",
            policy=seq_model,
            config=ckpt_config,
            norm_stats=norm_stats,
            training_contract=training_contract,
            train_data_contract=train_contract,
            val_data_contract=val_contract,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=start_step,
            epoch=start_epoch,
            batch_cursor=start_cursor,
            epoch_targets_seen=epoch_targets_seen,
            source_metadata=current_source_metadata,
            device=device,
            update_last=False,
        )
        if metrics_file is not None:
            metrics_file.close()
        if is_distributed and world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Initial validation before update 1 (fresh start or transition)
    periodic_val_indices = val_dataset.validation_indices(per_task=args.val_per_task)
    if (start_step == 0 and not args.resume) or is_transition:
        if rank == 0:
            v_type = "transition initial " if is_transition else "initial "
            print(f"[Validation] Running {v_type}validation on {len(periodic_val_indices)} segments...")
        init_val_loss, init_val_targets = evaluate_native(
            model=seq_model,
            val_dataset=val_dataset,
            eval_indices=periodic_val_indices,
            device=device,
        )
        if rank == 0:
            print(f"[Validation] {'Transition' if is_transition else 'Initial'} val loss: {init_val_loss:.5f} across {init_val_targets} targets")
            sys.stdout.flush()
            metrics_file.write(json.dumps({
                "type": "transition_initial_validation" if is_transition else "initial_validation",
                "val_loss": init_val_loss,
                "val_targets": init_val_targets,
                "global_step": start_step,
            }) + "\n")

    # Initial matched-history diagnostic if stream protocol
    if args.stream_protocol in ("stream_replay_v1", "memory_replay_v1", "compact_memory_replay_v1") and ((start_step == 0 and not args.resume) or is_transition):
        evaluate_matched_history_diagnostics(
            model=seq_model,
            val_dataset=val_dataset,
            dense_val_dataset=dense_val_dataset,
            eval_indices=periodic_val_indices,
            device=device,
            output_dir=output_dir,
            global_step=start_step,
            epoch=start_epoch,
        )

    global_step = start_step

    # Training Loop over Epochs
    stop_requested = False
    for epoch in range(start_epoch, args.epochs):
        if (args.max_updates is not None and global_step >= args.max_updates) or stop_requested:
            break

        train_dataset.set_epoch(epoch)

        # Generate epoch partitioning
        perm, epoch_batches, per_rank_indices = compute_epoch_batches(
            total_segments=total_segments,
            global_batch_size=args.global_batch_size,
            world_size=world_size,
            seed=args.seed,
            epoch=epoch,
        )

        cursor = start_cursor if epoch == start_epoch else 0
        if epoch != start_epoch:
            epoch_targets_seen = 0

        # Calculate remaining local indices for this rank from current cursor
        rank_remaining_indices: List[int] = []
        if cursor == total_segments:
            rank_remaining_indices = []
        else:
            batch_idx_start = math.ceil(cursor / args.global_batch_size)
            for bslice in epoch_batches[batch_idx_start:]:
                rank_remaining_indices.extend(bslice[rank::world_size])

        epoch_sampler = SegmentSequenceSampler(rank_remaining_indices)
        loader = DataLoader(
            train_dataset,
            batch_size=None,
            sampler=epoch_sampler,
            num_workers=args.workers,
            collate_fn=identity_collate,
            multiprocessing_context="spawn" if args.workers > 0 else None,
            persistent_workers=False,
            prefetch_factor=1 if args.workers > 0 else None,
            generator=torch.Generator().manual_seed(args.seed + epoch * 10007 + rank),
        )
        loader_iter = iter(loader)

        while cursor < total_segments:
            if args.max_updates is not None and global_step >= args.max_updates:
                break

            step_start_time = time.time()
            batch_slice = epoch_batches[cursor // args.global_batch_size]
            rank_segment_indices = batch_slice[rank::world_size]
            expected_local_count = len(rank_segment_indices)
            next_cursor = cursor + len(batch_slice)

            optimizer.zero_grad(set_to_none=True)
            seq_model.train()

            local_targets_sum = 0
            local_loss_sum = 0.0

            for _ in range(expected_local_count):
                sample = next(loader_iter)
                target_count = int(sample["target_count"])
                if target_count <= 0:
                    raise ValueError(f"Invalid local target_count {target_count} <= 0 for sample")

                sample_dev = {}
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        sample_dev[k] = v.to(device)
                    else:
                        sample_dev[k] = v

                out = seq_model(sample_dev)
                loss_sum_t = out["loss_sum"]
                loss_sum_val = float(loss_sum_t.item())

                if not math.isfinite(loss_sum_val):
                    raise FloatingPointError(f"Non-finite loss {loss_sum_val} detected on rank {rank}!")

                loss_sum_t.backward()
                local_loss_sum += loss_sum_val
                local_targets_sum += target_count

            # Global reduction of total target count
            if is_distributed and world_size > 1:
                target_tensor = torch.tensor([local_targets_sum, local_loss_sum], dtype=torch.float64, device=device)
                dist.all_reduce(target_tensor, op=dist.ReduceOp.SUM)
                global_targets_step = int(target_tensor[0].item())
                global_loss_step = float(target_tensor[1].item())
            else:
                global_targets_step = local_targets_sum
                global_loss_step = local_loss_sum

            if global_targets_step == 0:
                raise RuntimeError("Global targets across all ranks in step is 0; invalid batch!")

            # Verify global target count exactly matches sum of segments in batch
            expected_global_targets = sum(
                train_dataset.segments[s_idx][2] - train_dataset.segments[s_idx][1]
                for s_idx in batch_slice
            )
            if global_targets_step != expected_global_targets:
                raise RuntimeError(
                    f"Global target count mismatch: step produced {global_targets_step} targets, "
                    f"expected {expected_global_targets} from batch segments"
                )

            # Synchronize gradients via bucketed allreduce
            trainable_params = [p for p in seq_model.parameters() if p.requires_grad]
            bucketed_gradient_allreduce(
                parameters=trainable_params,
                global_target_count=global_targets_step,
            )

            # First step diagnostics: find first grad-bearing non-zero parameter per group
            is_first_step_diag = (global_step == 0 and not args.resume) or (is_transition and global_step == start_step)
            if is_first_step_diag:
                group_norms: Dict[str, float] = {}
                group_grad_counts: Dict[str, int] = {}
                chosen_rep_params: Dict[str, Tuple[str, nn.Parameter]] = {}
                group_params_before: Dict[str, torch.Tensor] = {}

                for name, p in seq_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    grp = classify_parameter(name)
                    if p.grad is not None:
                        g_norm = p.grad.norm().item()
                        group_norms[grp] = group_norms.get(grp, 0.0) + g_norm
                        group_grad_counts[grp] = group_grad_counts.get(grp, 0) + 1
                        if grp not in chosen_rep_params and g_norm > 0.0:
                            chosen_rep_params[grp] = (name, p)

                for required_grp in ("vision", "projector", "llm", "head"):
                    if required_grp not in group_norms or group_norms[required_grp] == 0.0:
                        raise RuntimeError(
                            f"First step failed: semantic parameter group '{required_grp}' has ZERO gradient norm!"
                        )
                    if required_grp not in chosen_rep_params:
                        raise RuntimeError(
                            f"First step failed: no parameter in group '{required_grp}' has non-zero gradient!"
                        )
                    p_rep = chosen_rep_params[required_grp][1]
                    group_params_before[required_grp] = p_rep.detach().clone()

            # Clip gradients across synchronized parameters
            grad_norm = clip_parameter_groups_norm(trainable_params, max_norm=args.grad_clip)

            current_lrs = {g["group_name"]: g["lr"] for g in optimizer.param_groups}

            # Step optimizer & scheduler
            optimizer.step()
            scheduler.step()

            # First step post-update diagnostics: check param deltas, moments dtype & finiteness, and cross-rank equality
            if is_first_step_diag:
                max_deltas: Dict[str, float] = {}
                for grp_name, (p_name, p_rep) in chosen_rep_params.items():
                    delta = (p_rep.detach() - group_params_before[grp_name]).abs().max().item()
                    max_deltas[grp_name] = delta
                    if delta == 0.0:
                        raise RuntimeError(
                            f"First step failed: representative parameter '{p_name}' for '{grp_name}' "
                            f"did not change after optimizer.step()!"
                        )

                # Verify optimizer moments are strictly FP32 and finite
                for p in trainable_params:
                    state = optimizer.state.get(p, {})
                    for m_key in ("exp_avg", "exp_avg_sq"):
                        if m_key in state:
                            tensor = state[m_key]
                            if tensor.dtype != torch.float32:
                                raise TypeError(
                                    f"Optimizer moment {m_key} has dtype {tensor.dtype}, strictly expected float32"
                                )
                            if not torch.isfinite(tensor).all():
                                raise FloatingPointError(
                                    f"Optimizer moment {m_key} contains non-finite values after first update!"
                                )

                # Cross-rank tensor SHA256 equality verification (exact full bytes on CPU)
                local_param_hashes: Dict[str, str] = {}
                for grp_name, (p_name, p_rep) in chosen_rep_params.items():
                    data_bytes = p_rep.detach().cpu().contiguous().numpy().tobytes()
                    local_param_hashes[grp_name] = hashlib.sha256(data_bytes).hexdigest()

                if is_distributed and world_size > 1:
                    gathered_hashes = [None] * world_size
                    dist.all_gather_object(gathered_hashes, local_param_hashes)
                    for r_i, chk in enumerate(gathered_hashes):
                        for grp_name in chosen_rep_params:
                            if chk[grp_name] != local_param_hashes[grp_name]:
                                raise RuntimeError(
                                    f"First step weight hash mismatch between rank {rank} and rank {r_i} on group {grp_name}!"
                                )

                if rank == 0:
                    first_step_evidence = {
                        "global_step": global_step + 1,
                        "start_step": start_step,
                        "is_transition": is_transition,
                        "group_norms": group_norms,
                        "group_grad_counts": group_grad_counts,
                        "chosen_representatives": {g: chosen_rep_params[g][0] for g in chosen_rep_params},
                        "max_deltas": max_deltas,
                        "grad_norm": grad_norm,
                        "param_hashes": local_param_hashes,
                        "rank": rank,
                        "pid": os.getpid(),
                        "device": str(device),
                        "precision": str(next(seq_model.parameters()).dtype),
                    }
                    (output_dir / "first_step_diagnostics.json").write_text(json.dumps(first_step_evidence, indent=2))
                    print(f"[Step {global_step + 1} Diagnostics] All 4 groups updated: deltas={max_deltas}")
                    sys.stdout.flush()

            global_step += 1
            epoch_targets_seen += global_targets_step
            cursor = next_cursor

            step_duration = time.time() - step_start_time
            weighted_loss = global_loss_step / global_targets_step

            # Metrics logging
            if rank == 0:
                peak_gpu = (
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    if (isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available())
                    else 0.0
                )
                metric_entry = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "epoch_display": f"{epoch + 1}/{args.epochs}",
                    "loss": weighted_loss,
                    "targets": global_targets_step,
                    "segments": len(batch_slice),
                    "epoch_targets_seen": epoch_targets_seen,
                    "grad_norm": grad_norm,
                    "lr": current_lrs,
                    "step_secs": step_duration,
                    "gpu_peak_mb": peak_gpu,
                }
                metrics_file.write(json.dumps(metric_entry) + "\n")
                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"[Step {global_step}] Epoch {epoch+1}/{args.epochs} | "
                        f"Loss: {weighted_loss:.4f} | GradNorm: {grad_norm:.4f} | "
                        f"Targets: {global_targets_step} | Time: {step_duration:.2f}s"
                    )
                    sys.stdout.flush()

            # Checkpoint after first update of this stage: write last.pt immediately
            if global_step == 1 or (is_transition and global_step == start_step + 1):
                save_native_checkpoint(
                    output_dir=output_dir,
                    filename="last.pt",
                    policy=seq_model,
                    config=ckpt_config,
                    norm_stats=norm_stats,
                    training_contract=training_contract,
                    train_data_contract=train_contract,
                    val_data_contract=val_contract,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    batch_cursor=cursor,
                    epoch_targets_seen=epoch_targets_seen,
                    source_metadata=current_source_metadata,
                    device=device,
                    update_last=False,
                )

            # Periodic evaluation
            if global_step % args.eval_every == 0:
                val_loss, val_targets = evaluate_native(
                    model=seq_model,
                    val_dataset=val_dataset,
                    eval_indices=periodic_val_indices,
                    device=device,
                )
                if rank == 0:
                    print(f"[Eval @ Step {global_step}] Loss: {val_loss:.5f} (targets: {val_targets})")
                    sys.stdout.flush()
                    metrics_file.write(json.dumps({
                        "type": "periodic_validation",
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "val_targets": val_targets,
                    }) + "\n")

                if args.stream_protocol in ("stream_replay_v1", "memory_replay_v1", "compact_memory_replay_v1"):
                    evaluate_matched_history_diagnostics(
                        model=seq_model,
                        val_dataset=val_dataset,
                        dense_val_dataset=dense_val_dataset,
                        eval_indices=periodic_val_indices,
                        device=device,
                        output_dir=output_dir,
                        global_step=global_step,
                        epoch=epoch,
                    )

            # Periodic saving: update last.pt
            if global_step % args.save_every == 0:
                save_native_checkpoint(
                    output_dir=output_dir,
                    filename="last.pt",
                    policy=seq_model,
                    config=ckpt_config,
                    norm_stats=norm_stats,
                    training_contract=training_contract,
                    train_data_contract=train_contract,
                    val_data_contract=val_contract,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    batch_cursor=cursor,
                    epoch_targets_seen=epoch_targets_seen,
                    source_metadata=current_source_metadata,
                    device=device,
                    update_last=False,
                )

            # Check stop-file at completed update boundary
            if check_stop_file_requested(args.stop_file, device):
                if rank == 0:
                    print(
                        f"[StopFile] Stop requested at step {global_step} "
                        f"(epoch {epoch}, cursor {cursor}). Saving last.pt and exiting cleanly."
                    )
                    sys.stdout.flush()
                save_native_checkpoint(
                    output_dir=output_dir,
                    filename="last.pt",
                    policy=seq_model,
                    config=ckpt_config,
                    norm_stats=norm_stats,
                    training_contract=training_contract,
                    train_data_contract=train_contract,
                    val_data_contract=val_contract,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    batch_cursor=cursor,
                    epoch_targets_seen=epoch_targets_seen,
                    source_metadata=current_source_metadata,
                    device=device,
                    update_last=False,
                )
                stop_requested = True
                break

        if stop_requested:
            break

        # Check if stopped early due to max_updates in a partial epoch
        if cursor < total_segments:
            if rank == 0:
                print(f"[Stop] Early stop at step {global_step} (epoch {epoch}, cursor {cursor}). Saving last.pt...")
            save_native_checkpoint(
                output_dir=output_dir,
                filename="last.pt",
                policy=seq_model,
                config=ckpt_config,
                norm_stats=norm_stats,
                training_contract=training_contract,
                train_data_contract=train_contract,
                val_data_contract=val_contract,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch,
                batch_cursor=cursor,
                epoch_targets_seen=epoch_targets_seen,
                source_metadata=current_source_metadata,
                device=device,
                update_last=False,
            )
            break

        # ONLY check loader exhaustion if cursor == total_segments!
        remaining_samples = 0
        try:
            next(loader_iter)
            remaining_samples += 1
        except StopIteration:
            pass
        if remaining_samples > 0:
            raise AssertionError(f"Rank {rank} loader has unconsumed samples at epoch {epoch} boundary!")

        # End of Epoch handling
        if epoch_targets_seen != train_dataset.total_targets:
            raise AssertionError(
                f"Epoch {epoch} completed targets {epoch_targets_seen} != expected total_targets {train_dataset.total_targets}"
            )

        # Full validation across ALL validation segments
        if rank == 0:
            print(f"[Epoch {epoch + 1} Completed] Running full validation across {len(val_dataset)} segments...")
            sys.stdout.flush()
        all_val_indices = list(range(len(val_dataset)))
        full_val_loss, full_val_targets = evaluate_native(
            model=seq_model,
            val_dataset=val_dataset,
            eval_indices=all_val_indices,
            device=device,
        )
        if rank == 0:
            print(f"[Epoch {epoch + 1}] Full val loss: {full_val_loss:.5f} across {full_val_targets} targets")
            sys.stdout.flush()
            metrics_file.write(json.dumps({
                "type": "full_epoch_validation",
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_loss": full_val_loss,
                "val_targets": full_val_targets,
            }) + "\n")

        # Save epoch checkpoint milestone and atomically update last.pt
        save_native_checkpoint(
            output_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch + 1:03d}.pt",
            policy=seq_model,
            config=ckpt_config,
            norm_stats=norm_stats,
            training_contract=training_contract,
            train_data_contract=train_contract,
            val_data_contract=val_contract,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch + 1,
            batch_cursor=0,
            epoch_targets_seen=0,
            source_metadata=current_source_metadata,
            device=device,
            update_last=True,
        )

        if args.max_updates is not None and global_step >= args.max_updates:
            break

    if rank == 0 and metrics_file is not None:
        metrics_file.close()

    if is_distributed and world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main(raw_args: Optional[Sequence[str]] = None, device: Optional[torch.device] = None) -> None:
    """Main training entrypoint."""
    args = parse_args(raw_args)
    run_training(args, device=device)


if __name__ == "__main__":
    main()
