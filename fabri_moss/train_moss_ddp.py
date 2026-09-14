"""CUDA-first DDP trainer for the MOSS FrameKV adapter.

The bridge stage learns the cross-attention adapter for one epoch.  The joint
stage keeps the native ViT/LLM weights frozen and trains the Action Expert plus
small FP32 LoRA residuals on the native projections adjacent to MOSS layers.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import numpy as np
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.data import MetaWorldWindows
from fabri_moss.lora import (
    FP32LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_modules,
    lora_spec,
    lora_state_dict,
    set_lora_train_mode,
)
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint
from fabri_moss.train import compute_file_sha256, compute_flow_kd_loss
from fabri_moss.train_native import (
    SegmentSequenceSampler,
    bucketed_gradient_allreduce,
    clip_parameter_groups_norm,
    compute_epoch_batches,
)


# Keep the public adapter format stable; the architecture revision below
# prevents a legacy/full-base joint checkpoint from being resumed silently.
FORMAT = "moss_cross_adapter_v2"
ARCHITECTURE_REVISION = "moss-cross-qkvo-lora-fp32-r8-a16-d0.1-v1"
MOSS_TRAINER_ARCHITECTURE_REVISION = ARCHITECTURE_REVISION
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

# Public aliases keep tiny external launch/test helpers independent of the
# trainer's private naming.
FP32LoRA = FP32LoRALinear


class _MossArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.epochs is None:
            parsed.epochs = 2 if parsed.stage == "joint" else 1
        return parsed


def parser():
    p = _MossArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="/root/models/FabriVLA/checkpoint_step_93000.pt")
    p.add_argument("--fabri-root", default="/root/FabriVLA")
    p.add_argument("--vlm", "--vlm-path", dest="vlm", default="/root/models/InternVL3_5-1B")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    # ``None`` lets joint default to its absolute two-epoch target while
    # bridge defaults to one epoch.  A supplied value always wins.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--stage", choices=("bridge", "expert", "joint"), default="bridge")
    p.add_argument("--context-mode", choices=("window", "consume", "causal"), default="causal")
    p.add_argument("--max-updates", type=int)
    p.add_argument("--global-batch-size", type=int, default=4)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--min-context-frames", type=int, default=1)
    p.add_argument("--decision-stride", type=int, default=None)
    p.add_argument("--execution-horizon", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4, help="bridge learning rate")
    p.add_argument("--action-lr", "--lr-action", dest="action_lr", type=float, default=1e-5)
    # Kept as a compatibility alias for old launchers.  Joint does not use it:
    # native LLM weights are frozen and only LoRA tensors are optimized.
    p.add_argument("--lora-lr", "--base-lr", "--lr-base", dest="base_lr", type=float, default=5e-6)
    p.add_argument("--vision-lr", type=float, default=None)
    p.add_argument("--train-vision", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    p.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument(
        "--native-kd-weight", type=float, default=0.0,
        help="current-frame-only velocity KD weight (set >0 for recovery diagnostics)",
    )
    p.add_argument(
        "--joint-train-action-expert", action="store_true",
        help="unfreeze Action Expert in joint; default keeps the native expert frozen",
    )
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--seed", type=int, default=4042)
    p.add_argument("--device", default=None, help="defaults to cuda:<LOCAL_RANK>")
    p.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume")
    p.add_argument("--init-adapter")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and normalize CLI arguments without touching CUDA or the filesystem."""
    args = parser().parse_args(argv)
    if args.epochs is None:
        args.epochs = 2 if args.stage == "joint" else 1
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.global_batch_size <= 0:
        raise ValueError(f"--global-batch-size must be positive, got {args.global_batch_size}")
    if args.window <= 0 or args.frame_stride <= 0 or args.min_context_frames <= 0:
        raise ValueError("--window, --frame-stride and --min-context-frames must be positive")
    if args.min_context_frames > args.window:
        raise ValueError("--min-context-frames cannot exceed --window")
    if args.decision_stride is not None and args.decision_stride <= 0:
        raise ValueError("--decision-stride must be positive")
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive")
    if args.num_workers < 0 or args.save_every <= 0:
        raise ValueError("--num-workers must be non-negative and --save-every positive")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    for name in ("lr", "action_lr", "base_lr", "kd_weight", "native_kd_weight", "grad_clip_norm"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0 or (name in ("lr", "action_lr", "base_lr", "grad_clip_norm") and value == 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    lora_rank = getattr(args, "lora_rank", LORA_RANK)
    lora_alpha = getattr(args, "lora_alpha", LORA_ALPHA)
    lora_dropout = getattr(args, "lora_dropout", LORA_DROPOUT)
    if isinstance(lora_rank, bool) or lora_rank <= 0:
        raise ValueError("--lora-rank must be positive")
    if not np.isfinite(lora_alpha) or lora_alpha <= 0:
        raise ValueError("--lora-alpha must be finite and positive")
    if not np.isfinite(lora_dropout) or not 0 <= lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if (args.stage == "joint" and
            (getattr(args, "lora_rank", LORA_RANK) != LORA_RANK or
             not np.isclose(getattr(args, "lora_alpha", LORA_ALPHA), LORA_ALPHA) or
             not np.isclose(getattr(args, "lora_dropout", LORA_DROPOUT), LORA_DROPOUT))):
        raise ValueError("joint LoRA is fixed at rank=8, alpha=16, dropout=0.1")
    if args.stage == "joint" and getattr(args, "train_vision", False):
        raise ValueError("joint stage keeps ViT frozen; remove --train-vision")
    if args.resume and args.init_adapter:
        raise ValueError("--resume and --init-adapter are mutually exclusive")


def _init_dist()->Tuple[int,int,int]:
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if world>1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            try: dist.init_process_group('nccl',device_id=torch.device('cuda',local))
            except TypeError: dist.init_process_group('nccl')
        else:
            # CPU/Gloo keeps synthetic collective tests runnable; launchers
            # still default to CUDA for formal runs.
            dist.init_process_group('gloo')
    return rank,world,local


def _cross_lora_paths(model: MossInternVL, layers: Optional[Sequence[int]] = None) -> List[str]:
    """Return native q/k/v/o paths at the layers carrying MOSS cross memory."""
    selected = tuple(int(x) for x in (layers if layers is not None else model.config.cross_layers))
    core = model.native_core
    paths: List[str] = []
    for layer_idx in selected:
        if layer_idx < 1 or layer_idx > len(core.layers):
            raise ValueError(f"LoRA layer {layer_idx} outside native layer range")
        attention = getattr(core.layers[layer_idx - 1], "self_attn", None)
        if attention is None:
            raise AttributeError(f"native layer {layer_idx} has no self_attn")
        # Paths are relative to policy, which is the root passed to inject_lora.
        # ``native_core`` is usually language_model.model; find its public path
        # rather than hard-coding one InternVL wrapper variant.
        prefix = _module_path(model.policy, attention)
        for name in LORA_TARGET_MODULES:
            target = getattr(attention, name, None)
            if target is None:
                raise AttributeError(f"native layer {layer_idx}.self_attn has no {name}")
            if isinstance(target, FP32LoRALinear):
                paths.append(f"{prefix}.{name}")
            else:
                if not hasattr(target, "weight") or not hasattr(target, "in_features"):
                    raise TypeError(f"LoRA target {layer_idx}.self_attn.{name} is not Linear-like")
                paths.append(f"{prefix}.{name}")
    return paths


def _module_path(root: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in root.named_modules():
        if module is target:
            return name
    raise ValueError("native attention module is not reachable from policy")


def configure_lora(model: MossInternVL, args: argparse.Namespace) -> Dict[str, FP32LoRALinear]:
    """Attach the fixed FP32 q/k/v/o LoRA set for a joint run."""
    paths = _cross_lora_paths(model)
    modules = inject_lora(
        model.policy,
        paths,
        rank=getattr(args, "lora_rank", LORA_RANK),
        alpha=getattr(args, "lora_alpha", LORA_ALPHA),
        dropout=getattr(args, "lora_dropout", LORA_DROPOUT),
    )
    for module in modules.values():
        if module.lora_A.dtype != torch.float32 or module.lora_B.dtype != torch.float32:
            raise TypeError("LoRA parameters must be FP32")
    return modules


def configure_trainable_parameters(
    model: MossInternVL, stage: str, train_action_expert: Optional[bool] = None
) -> List[torch.nn.Parameter]:
    """Set the exact trainable surface for each stage.

    Joint deliberately excludes bridge/readout and every native base tensor;
    only Action Expert parameters and injected LoRA tensors remain trainable.
    """
    if train_action_expert is not None:
        model.joint_train_action_expert = bool(train_action_expert)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "bridge":
        for parameter in model.bridge_parameters():
            parameter.requires_grad_(True)
    elif stage == "expert":
        for parameter in model.bridge_parameters():
            parameter.requires_grad_(True)
        for parameter in model.action_parameters():
            parameter.requires_grad_(True)
    elif stage == "joint":
        if not lora_modules(model.policy):
            raise RuntimeError("joint stage requires injected q/k/v/o LoRA modules")
        train_action_expert = bool(getattr(model, "joint_train_action_expert", True))
        if train_action_expert:
            for parameter in model.action_parameters():
                parameter.requires_grad_(True)
        for module in lora_modules(model.policy).values():
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            for parameter in module.base.parameters():
                parameter.requires_grad_(False)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    head = getattr(model.policy, "action_head", None)
    if head is not None:
        # Freezing weights does not disable dropout.  Keep the frozen expert
        # deterministic while autograd still differentiates its inputs.
        head.train(model.training and any(p.requires_grad for p in head.parameters()))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if stage == "joint" and any(p.dtype != torch.float32 for p in trainable):
        bad = [str(p.dtype) for p in trainable if p.dtype != torch.float32][:3]
        raise TypeError(f"joint trainable parameters must be FP32, got {bad}")
    return trainable


def _trainable_names(model: torch.nn.Module) -> List[str]:
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def _target_tensors(s):
    states = s.get("target_states", s.get("states", s.get("state")))
    actions = s.get("target_actions", s.get("actions"))
    masks = s.get("target_action_mask", s.get("action_masks", s.get("action_mask")))
    if states is None or actions is None or masks is None:
        raise KeyError("sample must contain state(s), action(s), and action mask(s)")
    states, actions, masks = map(torch.as_tensor, (states, actions, masks))
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    n = int(states.shape[0])
    if actions.shape[0] != n or masks.shape[0] != n:
        raise ValueError(
            f"target count mismatch: states={n}, actions={actions.shape[0]}, masks={masks.shape[0]}"
        )
    vis = s.get("visible_frame_counts", s.get("visible_counts"))
    if vis is None:
        idx = s.get("target_indices")
        vis = [int(i) + 1 for i in idx] if idx is not None else [len(s["images_window"])] * n
    vis = [int(x) for x in vis]
    if len(vis) != n:
        raise ValueError(f"visible frame count length {len(vis)} != target count {n}")
    return states, actions, masks, vis


def _sample_loss(
    student, teacher_head, sample, device, kd_weight, teacher_policy=None,
    native_kd_weight: float = 0.0,
):
    images = sample["images_window"]
    frame_ids = [int(x) for x in sample["frame_ids"]]
    prompt = str(sample["prompt"])
    states, actions, masks, visible = _target_tensors(sample)
    if states.shape[0] == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero, zero, 0
    observation_times = sample.get('observation_times')
    frames=[]
    for j, (imgs, fid) in enumerate(zip(images, frame_ids)):
        encoded = student.encode_image(imgs)
        obs_time = float(observation_times[j]) if observation_times is not None else None
        try:
            frames.append(student.project_frame(encoded, fid, observation_time=obs_time))
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            frames.append(student.project_frame(encoded, fid))
    total=gt_total=kd_total=None; total_count=0
    state_mask = sample.get("target_state_mask", sample.get("state_mask"))
    time_mask = sample.get("action_time_mask")
    if time_mask is not None:
        time_mask = torch.as_tensor(time_mask)
        if time_mask.ndim == 1:
            time_mask = time_mask.unsqueeze(0).expand(masks.shape[0], -1)
        elif time_mask.ndim == 2 and time_mask.shape[0] == 1 and masks.shape[0] != 1:
            time_mask = time_mask.expand(masks.shape[0], -1)
        if time_mask.shape[0] != masks.shape[0] or time_mask.shape[1] != masks.shape[1]:
            raise ValueError("action_time_mask must match target/action horizon dimensions")
        masks = masks * time_mask.unsqueeze(-1).to(dtype=masks.dtype)
    for i,count in enumerate(visible):
        if count<1 or count>len(frames): raise ValueError(f'visible frame count {count} outside [1, {len(frames)}]')
        raw_mask = masks[i : i + 1]
        if raw_mask.numel() == 0 or not bool(raw_mask.any().item()):
            continue
        if sample.get("valid_action_lengths") is not None and int(torch.as_tensor(sample["valid_action_lengths"])[i].item()) <= 0:
            continue
        times = observation_times
        prefix_times = list(times[:count]) if times is not None else None
        try:
            deep,shallow=student.read_memory(
                frames[:count], prompt, frame_ids=frame_ids[:count], observation_times=prefix_times
            )
        except TypeError as exc:
            # Keep tiny test doubles and legacy adapters that only expose the
            # original two-argument consume API working.
            if "unexpected keyword argument" not in str(exc):
                raise
            deep,shallow=student.read_memory(frames[:count], prompt)
        with torch.no_grad():
            teacher_embedder = teacher_policy if teacher_policy is not None else student.policy
            out=teacher_embedder.get_vl_embeddings(images=images[count-1],image_mask=torch.ones(len(images[count-1]),dtype=torch.bool,device=device),prompt=prompt,return_cls_only=False,shallow_layer_index=6)
        tdeep,tshallow=out if isinstance(out,tuple) else (out,None)
        state=states[i:i+1].to(device)
        if state_mask is not None:
            sm=torch.as_tensor(state_mask); sm=sm[i:i+1] if sm.ndim>1 and sm.shape[0]>1 else sm[:1]; state=state*sm.to(device)
        mask=masks[i:i+1].to(device); action=actions[i:i+1].to(device)
        sample_time_weights = sample.get("action_time_weights", sample.get("time_weights"))
        if sample_time_weights is not None:
            sample_time_weights = torch.as_tensor(sample_time_weights)
            if sample_time_weights.ndim >= 3 or (
                sample_time_weights.ndim == 2 and sample_time_weights.shape[0] == masks.shape[0]
            ):
                sample_time_weights = sample_time_weights[i : i + 1]
        # Recovery mode can explicitly distill the native current-frame path.
        # Reuse one noise/t pair for history and current-only predictions so
        # the auxiliary term measures representation drift, not sampling noise.
        fixed_noise = fixed_t = None
        if native_kd_weight > 0:
            fixed_noise = sample.get("fixed_noise")
            if fixed_noise is not None:
                fixed_noise = torch.as_tensor(fixed_noise)
                if fixed_noise.ndim >= 3 and fixed_noise.shape[0] == masks.shape[0]:
                    fixed_noise = fixed_noise[i : i + 1]
            if fixed_noise is None:
                fixed_noise = torch.rand_like(action) * 2.0 - 1.0
            fixed_t = sample.get("fixed_t")
            if fixed_t is None:
                concentration = action.new_tensor(2.0)
                fixed_t = torch.distributions.Beta(concentration, concentration).sample(
                    (action.shape[0],)
                ).clamp(0.02, 0.98)
            fixed_t = torch.as_tensor(fixed_t, dtype=action.dtype).reshape(-1)
            if fixed_t.numel() == masks.shape[0]:
                fixed_t = fixed_t[i : i + 1]
            if fixed_t.numel() == 1:
                fixed_t = fixed_t.expand(action.shape[0])
        loss_kwargs = dict(
            student_head=student.policy.action_head,
            teacher_head=teacher_head,
            student_deep=deep,
            student_shallow=shallow,
            teacher_deep=tdeep,
            teacher_shallow=tshallow,
            state=state,
            actions=action,
            action_mask=mask,
            kd_weight=kd_weight,
            fixed_noise=fixed_noise,
            fixed_t=fixed_t,
        )
        if any(k in sample for k in ("action_time_weights", "time_weights", "valid_action_lengths", "execution_horizon")):
            loss_kwargs.update(
                action_time_weights=sample_time_weights,
                valid_action_lengths=(
                    torch.as_tensor(sample["valid_action_lengths"])[i : i + 1]
                    if sample.get("valid_action_lengths") is not None
                    else None
                ),
                execution_horizon=(
                    int(sample["execution_horizon"])
                    if sample.get("execution_horizon") is not None else 5
                ),
            )
        loss, gt, kd, _, _ = compute_flow_kd_loss(**loss_kwargs)
        if native_kd_weight > 0:
            current_deep, current_shallow = student.read_memory(
                frames[count - 1 : count], prompt,
                frame_ids=frame_ids[count - 1 : count],
                observation_times=(prefix_times[count - 1 : count] if prefix_times is not None else None),
            )
            native_kwargs = dict(loss_kwargs)
            native_kwargs.update(student_deep=current_deep, student_shallow=current_shallow, kd_weight=1.0)
            _, _, native_kd, _, _ = compute_flow_kd_loss(**native_kwargs)
            loss = loss + float(native_kd_weight) * native_kd
            kd = kd + float(native_kd_weight) * native_kd
        # Keep the global denominator aligned with compute_flow_kd_loss.  Old
        # hand-written samples without cadence metadata retain raw mask counts.
        c = float(mask.sum().item())
        if any(k in sample for k in ("action_time_weights", "time_weights", "valid_action_lengths", "execution_horizon")):
            valid_len = None
            if sample.get("valid_action_lengths") is not None:
                valid_len = int(torch.as_tensor(sample["valid_action_lengths"])[i].item())
            weighted = mask.float()
            if valid_len is not None:
                weighted[:, valid_len:, :] = 0
            supplied_weights = sample_time_weights
            if supplied_weights is None:
                weighted[:, : min(int(sample.get("execution_horizon", 5)), weighted.shape[1]), :] *= 4.0
            else:
                sw = torch.as_tensor(supplied_weights, dtype=weighted.dtype)
                if sw.ndim == 1:
                    sw = sw.view(1, -1, 1)
                elif sw.ndim == 2:
                    sw = sw.unsqueeze(0) if sw.shape[0] == weighted.shape[1] else sw.unsqueeze(-1)
                weighted = weighted * sw.to(device=weighted.device)
            c = float(weighted.sum().item())
        if c<=0: continue
        total=loss*c if total is None else total+loss*c; gt_total=gt*c if gt_total is None else gt_total+gt*c; kd_total=kd*c if kd_total is None else kd_total+kd*c; total_count+=c
    if total is None:
        # Cadence rows can intentionally carry no action target.  Returning a
        # detached zero keeps every DDP rank on the same collective schedule.
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero, zero, 0
    return total,gt_total,kd_total,total_count


def _rng_states(world, device):
    local = {'torch': torch.get_rng_state(), 'random': random.getstate(), 'numpy': np.random.get_state()}
    if torch.device(device).type == 'cuda':
        cuda_state = torch.cuda.get_rng_state(torch.device(device))
        local['torch_cuda'] = cuda_state
        local['cuda'] = cuda_state
    gathered = [None] * world
    if world > 1: dist.all_gather_object(gathered, local)
    else: gathered[0] = local
    return gathered


def _parameter_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            h.update(name.encode()); h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _state_hash(module: torch.nn.Module) -> str:
    """Hash a frozen module without depending on parameter ``requires_grad``."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _allreduce_mixed_precision_gradients(parameters, global_target_count):
    """Reduce mixed FP32/BF16 gradients without assigning wrong dtypes."""
    trainable = [p for p in parameters if p.requires_grad]
    if not trainable:
        return False
    is_dist = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if is_dist else 1
    device = trainable[0].device
    # All ranks still exchange the per-parameter usage flags for an empty or
    # masked tail batch.  No optimizer step is taken when the global count is 0.
    if global_target_count <= 0:
        flags = torch.tensor(
            [p.grad is not None for p in trainable], dtype=torch.int32, device=device
        )
        if is_dist and world > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.SUM)
        for p in trainable:
            p.grad = None
        return False
    local_has_grad = any(p.grad is not None for p in trainable)
    if is_dist and world > 1:
        flag = torch.tensor(int(local_has_grad), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        global_has_grad = bool(flag.item())
    else:
        global_has_grad = local_has_grad
    if all(p.grad is None or p.grad.dtype == torch.float32 for p in trainable):
        bucketed_gradient_allreduce(trainable, global_target_count)
        return global_has_grad
    flags = torch.tensor([p.grad is not None for p in trainable], dtype=torch.int32, device=device)
    if is_dist and world > 1:
        # The scalar reduction above already establishes whether any rank has
        # a gradient; retain this per-parameter reduction for zero-fill.
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    for p, used in zip(trainable, flags.tolist()):
        if not used:
            continue
        grad = p.grad.float() if p.grad is not None else torch.zeros_like(p.data, dtype=torch.float32)
        if is_dist and world > 1:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(float(global_target_count))
        if p.grad is None:
            p.grad = grad.to(dtype=p.dtype)
        else:
            p.grad.data.copy_(grad.to(dtype=p.grad.dtype))
    return global_has_grad


def _config_dict(model: Any) -> Dict[str, Any]:
    config = getattr(model, "config", {})
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return dict(config) if isinstance(config, Mapping) else {}


def _architecture(model: Any) -> Dict[str, Any]:
    modules = lora_spec(model.policy) if isinstance(getattr(model, "policy", None), torch.nn.Module) else {
        "rank": 0, "alpha": 0.0, "dropout": 0.0, "modules": [], "parameter_dtype": "float32"
    }
    stage = getattr(model, "training_stage", "bridge")
    surface = {
        "bridge": "bridge_readout",
        "expert": "bridge_readout_plus_action_expert",
        "joint": "action_expert_plus_lora",
    }.get(stage, str(stage))
    return {
        "revision": ARCHITECTURE_REVISION,
        "cross_layers": list(getattr(getattr(model, "config", None), "cross_layers", ())),
        "lora": modules,
        "trainable_surface": surface,
        "joint_train_action_expert": bool(getattr(model, "joint_train_action_expert", True)),
    }


def _checkpoint_trainable_names(model: Any) -> List[str]:
    return _trainable_names(model) if isinstance(model, torch.nn.Module) else []


def _save(
    path,
    model,
    optimizer,
    step,
    epoch,
    batch_cursor,
    epoch_targets_seen,
    args,
    norm_stats,
    base_meta,
    contract,
    world,
    rng_states=None,
):
    """Atomically save a self-describing adapter checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = str(getattr(model, "training_stage", getattr(args, "stage", "bridge")))
    source_sha = base_meta.get("checkpoint_sha256")
    architecture = _architecture(model)
    payload = {
        "format": FORMAT,
        "architecture_revision": ARCHITECTURE_REVISION,
        "adapter_architecture_revision": ARCHITECTURE_REVISION,
        "core_architecture_revision": getattr(getattr(model, "config", None), "architecture_revision", None),
        "architecture": architecture,
        "step": int(step),
        "global_step": int(step),
        "epoch": int(epoch),
        "batch_cursor": int(batch_cursor),
        "cursor": int(batch_cursor),
        "epoch_cursor": int(batch_cursor),
        "epoch_targets_seen": int(epoch_targets_seen),
        "stage": stage,
        "world_size": int(world),
        "source_checkpoint_sha256": source_sha,
        "base_sha256": source_sha,
        "base_checkpoint_sha256": source_sha,
        "config": _config_dict(model),
        "cross_blocks": model.cross_blocks.state_dict(),
        "readout_embeddings": model.readout_embeddings.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "norm_stats": norm_stats,
        "base_metadata": base_meta,
        "data_contract": contract,
        "training_contract": _training_contract(args, int(world), model),
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "trainable_parameter_names": _checkpoint_trainable_names(model),
        "lora_spec": architecture["lora"],
        "teacher_immutable": True,
        "teacher_state_sha256": base_meta.get("teacher_state_sha256"),
        "rng_state": {"torch": torch.get_rng_state(), "random": random.getstate(), "numpy": np.random.get_state()},
    }
    if rng_states is None:
        if int(world) != 1:
            raise ValueError("DDP checkpoint save requires one RNG state per rank")
        rng_states = [payload["rng_state"]]
    if rng_states is not None:
        # Keep both spellings while downstream readers migrate.
        payload["rng_states"] = rng_states
        payload["rng_states_per_rank"] = rng_states
    if stage in ("expert", "joint") and hasattr(model, "policy") and getattr(model.policy, "action_head", None) is not None:
        payload["action_head"] = model.policy.action_head.state_dict()
    if stage == "joint" and isinstance(getattr(model, "policy", None), torch.nn.Module):
        payload["lora_state"] = lora_state_dict(model.policy)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _validate_checkpoint_header(ck: Mapping[str, Any], model: Any, args: Any, base_meta: Mapping[str, Any], contract: Mapping[str, Any], world: int) -> None:
    if ck.get("format") != FORMAT:
        raise ValueError(f"resume checkpoint must use {FORMAT}; Writer/legacy checkpoints are rejected")
    if ck.get("teacher_immutable") is not True:
        raise ValueError("checkpoint must declare an immutable teacher")
    if ck.get("architecture_revision") != ARCHITECTURE_REVISION:
        raise ValueError("architecture_revision mismatch; mixed MOSS/LoRA runs are rejected")
    expected_core_revision = getattr(getattr(model, "config", None), "architecture_revision", None)
    if "core_architecture_revision" in ck and ck.get("core_architecture_revision") != expected_core_revision:
        raise ValueError("core architecture_revision mismatch")
    if int(ck.get("world_size", -1)) != int(world):
        raise ValueError(f"world_size mismatch: checkpoint={ck.get('world_size')} current={world}")
    expected_sha = base_meta.get("checkpoint_sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError("current base metadata is missing checkpoint SHA256")
    checkpoint_sha = ck.get("source_checkpoint_sha256") or ck.get("base_sha256") or ck.get("base_checkpoint_sha256")
    if checkpoint_sha is None and isinstance(ck.get("base_metadata"), Mapping):
        checkpoint_sha = ck["base_metadata"].get("checkpoint_sha256")
    if checkpoint_sha != expected_sha:
        raise ValueError("source checkpoint SHA256 mismatch")
    expected_teacher_sha = base_meta.get("teacher_state_sha256")
    if expected_teacher_sha is not None and ck.get("teacher_state_sha256") != expected_teacher_sha:
        raise ValueError("immutable teacher fingerprint mismatch")
    if ck.get("stage") != args.stage:
        raise ValueError(f"stage mismatch: checkpoint={ck.get('stage')} current={args.stage}")
    if ck.get("data_contract", ck.get("training_contract")) != contract:
        raise ValueError("data_contract mismatch on resume")
    saved_training = ck.get("training_contract")
    if isinstance(saved_training, Mapping):
        expected_training = _training_contract(args, world, model)
        # The epoch budget is an absolute target and may be extended on resume;
        # every data/model/optimizer setting remains strict.
        saved_cmp = {k: v for k, v in saved_training.items() if k != "target_epochs"}
        expected_cmp = {k: v for k, v in expected_training.items() if k != "target_epochs"}
        if saved_cmp != expected_cmp:
            raise ValueError("training_contract mismatch on resume")
    if dict(ck.get("config", {})) != _config_dict(model):
        raise ValueError("MOSS config mismatch on resume")
    expected_arch = _architecture(model)
    if ck.get("architecture") != expected_arch:
        raise ValueError("architecture contract mismatch on resume")
    saved_names = ck.get("trainable_parameter_names")
    if saved_names is not None and list(saved_names) != _checkpoint_trainable_names(model):
        raise ValueError("trainable parameter surface mismatch on resume")
    if args.stage == "joint":
        if not isinstance(ck.get("lora_state"), Mapping):
            raise KeyError("joint checkpoint missing lora_state")
        if ck.get("lora_spec") != expected_arch["lora"]:
            raise ValueError("LoRA specification mismatch on resume")
    for key in ("cross_blocks", "readout_embeddings", "optimizer"):
        if key not in ck:
            raise KeyError(f"checkpoint missing required field {key!r}")
    if args.stage == "joint" and isinstance(ck["optimizer"], Mapping):
        for state in ck["optimizer"].get("state", {}).values():
            if isinstance(state, Mapping):
                for value in state.values():
                    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
                        raise TypeError("joint optimizer state must be FP32; BF16 base state is forbidden")
    if not isinstance(ck.get("epoch"), int) or ck["epoch"] < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    cursor_value = ck.get("batch_cursor", ck.get("cursor", ck.get("epoch_cursor")))
    if not isinstance(cursor_value, int) or cursor_value < 0:
        raise ValueError("checkpoint cursor must be a non-negative integer")
    if not isinstance(ck.get("epoch_targets_seen", 0), int) or ck.get("epoch_targets_seen", 0) < 0:
        raise ValueError("checkpoint epoch_targets_seen must be a non-negative integer")
    if hasattr(args, "epochs") and args.epochs is not None:
        target_epochs = int(args.epochs)
        if target_epochs < int(ck["epoch"]):
            raise ValueError("--epochs is an absolute target and cannot be below checkpoint epoch")
        if target_epochs == int(ck["epoch"]) and int(cursor_value) != 0:
            raise ValueError("checkpoint cursor is beyond the requested absolute epoch target")
    step_value = ck.get("step", ck.get("global_step"))
    if not isinstance(step_value, int) or step_value < 0:
        raise ValueError("checkpoint step must be a non-negative integer")


def _restore_rng(rng: Mapping[str, Any], device: str) -> None:
    if "torch" in rng:
        state = rng["torch"].cpu() if isinstance(rng["torch"], torch.Tensor) else rng["torch"]
        torch.set_rng_state(state)
    if "random" in rng:
        random.setstate(rng["random"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    cuda_state = rng.get("torch_cuda", rng.get("cuda"))
    if cuda_state is not None and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(cuda_state, torch.device(device))


def _load_resume(path, model, optimizer, args, base_meta, contract, world, rank=0, device="cpu"):
    ck = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    _validate_checkpoint_header(ck, model, args, base_meta, contract, world)
    model.cross_blocks.load_state_dict(ck["cross_blocks"], strict=True)
    with torch.no_grad():
        model.readout_embeddings.copy_(ck["readout_embeddings"].to(model.readout_embeddings.device))
    if args.stage in ("expert", "joint"):
        if "action_head" not in ck:
            raise KeyError(f"stage {args.stage} checkpoint missing action_head")
        model.policy.action_head.load_state_dict(ck["action_head"], strict=True)
    if args.stage == "joint":
        load_lora_state_dict(model.policy, ck["lora_state"], strict=True)
    rng_states = ck.get("rng_states_per_rank", ck.get("rng_states"))
    if not isinstance(rng_states, list) or len(rng_states) != world:
        raise ValueError("resume checkpoint must contain one RNG state per rank")
    if rank < 0 or rank >= world:
        raise ValueError(f"rank {rank} outside checkpoint world_size {world}")
    optimizer.load_state_dict(ck["optimizer"])
    _restore_rng(rng_states[rank], device)
    if hasattr(model, "set_training_stage"):
        model.set_training_stage(args.stage)
    if hasattr(model, "train"):
        model.train()
    if hasattr(model, "policy"):
        configure_trainable_parameters(model, args.stage)
        set_lora_train_mode(model.policy, args.stage == "joint")
    return int(ck.get("step", ck.get("global_step"))), int(ck["epoch"]), int(ck.get("batch_cursor", ck.get("cursor", ck.get("epoch_cursor", 0)))), int(ck.get("epoch_targets_seen", 0))


def _load_init_adapter(path, model, args, base_meta, contract=None):
    ck = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"init-adapter must use {FORMAT}; Writer/legacy checkpoints are rejected")
    if ck.get("teacher_immutable") is not True:
        raise ValueError("init-adapter must declare an immutable teacher")
    if ck.get("architecture_revision") != ARCHITECTURE_REVISION:
        raise ValueError("architecture_revision mismatch in init-adapter")
    checkpoint_sha = ck.get("source_checkpoint_sha256") or ck.get("base_sha256") or ck.get("base_checkpoint_sha256")
    if checkpoint_sha is None and isinstance(ck.get("base_metadata"), Mapping):
        checkpoint_sha = ck["base_metadata"].get("checkpoint_sha256")
    expected_sha = base_meta.get("checkpoint_sha256")
    if not isinstance(expected_sha, str) or not expected_sha or checkpoint_sha != expected_sha:
        raise ValueError("init-adapter source checkpoint SHA256 mismatch")
    if contract is not None and ck.get("data_contract", ck.get("training_contract")) != contract:
        raise ValueError("init-adapter data_contract mismatch")
    source_stage = ck.get("stage")
    if source_stage not in ("bridge", "expert", "joint"):
        raise ValueError(f"unsupported init-adapter stage {source_stage!r}")
    if source_stage == "joint" and args.stage == "bridge":
        raise ValueError("cannot initialize bridge from joint checkpoint")
    if dict(ck.get("config", {})) != _config_dict(model):
        raise ValueError("init-adapter MOSS config mismatch")
    model.cross_blocks.load_state_dict(ck["cross_blocks"], strict=True)
    with torch.no_grad():
        model.readout_embeddings.copy_(ck["readout_embeddings"].to(model.readout_embeddings.device))
    if args.stage in ("expert", "joint") and source_stage in ("expert", "joint"):
        model.policy.action_head.load_state_dict(ck["action_head"], strict=True)
    if args.stage == "joint" and source_stage == "joint":
        if ck.get("lora_spec") != lora_spec(model.policy) or not isinstance(ck.get("lora_state"), Mapping):
            raise ValueError("init-adapter LoRA specification mismatch")
        load_lora_state_dict(model.policy, ck["lora_state"], strict=True)


def _training_contract(args: argparse.Namespace, world: int, model: Any) -> Dict[str, Any]:
    """Stable trainer-side contract checked in addition to the data contract."""
    spec = lora_spec(model.policy) if isinstance(getattr(model, "policy", None), torch.nn.Module) else {
        "rank": 0, "alpha": 0.0, "dropout": 0.0, "modules": [], "parameter_dtype": "float32"
    }
    return {
        "format": FORMAT,
        "stage": getattr(args, "stage", getattr(model, "training_stage", "bridge")),
        "target_epochs": int(getattr(args, "epochs", 1)),
        "global_batch_size": int(getattr(args, "global_batch_size", 1)),
        "num_workers": int(getattr(args, "num_workers", 0)),
        "save_every": int(getattr(args, "save_every", 0)),
        "max_episodes": getattr(args, "max_episodes", None),
        "world_size": int(world),
        "seed": int(getattr(args, "seed", 0)),
        "context_mode": getattr(args, "context_mode", "window"),
        "window": int(getattr(args, "window", 0)),
        "frame_stride": int(getattr(args, "frame_stride", 0)),
        "min_context_frames": int(getattr(args, "min_context_frames", 0)),
        "decision_stride": getattr(args, "decision_stride", None),
        "execution_horizon": int(getattr(args, "execution_horizon", 0)),
        "lr": float(getattr(args, "lr", 0.0)),
        "action_lr": float(getattr(args, "action_lr", 0.0)),
        "kd_weight": float(getattr(args, "kd_weight", 0.0)),
        "native_kd_weight": float(getattr(args, "native_kd_weight", 0.0)),
        "joint_train_action_expert": bool(getattr(args, "joint_train_action_expert", getattr(model, "joint_train_action_expert", True))),
        "grad_clip_norm": float(getattr(args, "grad_clip_norm", 0.0)),
        "architecture_revision": ARCHITECTURE_REVISION,
        "lora": spec,
    }


def _make_dataset(args: argparse.Namespace, norm_stats: Mapping[str, Any]):
    kwargs = dict(
        root=args.data_root,
        norm_stats=norm_stats,
        horizon=50,
        state_dim=24,
        action_dim=24,
        window=args.window,
        frame_stride=args.frame_stride,
        split="train",
        seed=args.seed,
        max_episodes=args.max_episodes,
        context_mode=args.context_mode,
        min_context_frames=args.min_context_frames,
        decision_stride=args.decision_stride,
        execution_horizon=args.execution_horizon,
    )
    # Keep compatibility with a pre-cadence checkout while using cadence when
    # the dataset exposes it.  Signature inspection avoids swallowing real
    # constructor errors.
    import inspect

    try:
        accepted = inspect.signature(MetaWorldWindows).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        pass
    return MetaWorldWindows(**kwargs)


def main():
    args = parse_args()
    validate_args(args)
    rank, world, local = _init_dist()
    if args.global_batch_size <= 0:
        raise ValueError("global batch size must be positive")
    device = args.device or f"cuda:{local}"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    policy, _, norm_stats, base_meta = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm,
        device=device,
        arm_key="metaworld_sawyer",
        trainable=False,
    )
    fa2 = assert_native_fa2(policy) if torch.device(device).type == "cuda" else {"native_fa2_enabled": False}

    # Copy the immutable teacher before replacing any native projections with
    # LoRA wrappers.  This also keeps teacher dropout/eval state independent.
    if getattr(policy, "action_head", None) is not None:
        policy.action_head.float()
    teacher_policy = copy.deepcopy(policy).to(device)
    teacher_policy.eval()
    for parameter in teacher_policy.parameters():
        parameter.requires_grad_(False)
    teacher_head = getattr(teacher_policy, "action_head", None)
    base_meta = dict(base_meta)
    base_meta["teacher_state_sha256"] = _state_hash(teacher_policy)

    config = MossConfig(
        cross_layers=(3, 6, 10, 14),
        max_frames=args.window,
        max_text_tokens=1024,
        shallow_layer=6,
        train_vision=False,
    )
    model = MossInternVL(policy, config)
    if args.stage == "joint":
        configure_lora(model, args)
    model.set_training_stage(args.stage)
    trainable = configure_trainable_parameters(
        model, args.stage,
        train_action_expert=(args.joint_train_action_expert if args.stage == "joint" else None),
    )
    model.train()
    if args.stage == "joint":
        set_lora_train_mode(model.policy, True)
    if any(parameter.dtype != torch.float32 for parameter in trainable):
        raise TypeError("all MOSS trainer parameters passed to AdamW must be FP32")

    if world > 1:
        # Synchronize both adapter and LoRA initialization; frozen native base
        # tensors are identical because every rank loaded the same SHA.
        for parameter in trainable:
            dist.broadcast(parameter.data, src=0)
        hashes = [None] * world
        dist.all_gather_object(hashes, _parameter_hash(model))
        if len(set(hashes)) != 1:
            raise RuntimeError("Trainable parameters differ across ranks after initialization")

    rank_seed = args.seed + rank * 10007
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    dataset = _make_dataset(args, norm_stats)
    if hasattr(dataset, "get_data_contract"):
        data_contract = dataset.get_data_contract()
    else:
        data_contract = {
            "context_mode": args.context_mode,
            "window": args.window,
            "frame_stride": args.frame_stride,
            "min_context_frames": args.min_context_frames,
            "decision_stride": args.decision_stride,
            "execution_horizon": args.execution_horizon,
            "seed": args.seed,
            "max_episodes": args.max_episodes,
            "split": "train",
            "active_episode_ids": sorted(int(e["episode_index"]) for e in dataset.active_episodes),
        }

    if args.init_adapter:
        _load_init_adapter(args.init_adapter, model, args, base_meta, data_contract)
        # The source adapter can contain bridge/expert weights; restore the
        # exact target-stage trainable surface afterward.
        trainable = configure_trainable_parameters(
            model, args.stage,
            train_action_expert=(args.joint_train_action_expert if args.stage == "joint" else None),
        )
        model.train()
        if args.stage == "joint":
            set_lora_train_mode(model.policy, True)

    groups: List[Dict[str, Any]] = []

    def add_group(name: str, params: Iterable[torch.nn.Parameter], lr: float) -> None:
        seen = {id(p) for group in groups for p in group["params"]}
        selected = [p for p in params if p.requires_grad and id(p) not in seen]
        if selected:
            if any(p.dtype != torch.float32 for p in selected):
                raise TypeError(f"optimizer group {name} contains non-FP32 parameters")
            groups.append({"params": selected, "lr": float(lr), "name": name})

    if args.stage in ("bridge", "expert"):
        add_group("bridge", model.bridge_parameters(), args.lr)
    if args.stage == "expert" or (args.stage == "joint" and args.joint_train_action_expert):
        add_group("action_head", model.action_parameters(), args.action_lr)
    if args.stage == "joint":
        add_group("lora", (p for m in lora_modules(model.policy).values() for p in (m.lora_A, m.lora_B)), args.base_lr)
    if not groups:
        raise RuntimeError(f"no trainable parameters for stage {args.stage}")
    trainable = [p for group in groups for p in group["params"]]
    optimizer = AdamW(groups, weight_decay=0.0)

    trainer_contract = _training_contract(args, world, model)
    out = Path(args.output_dir)
    # Check before rank 0 creates run_config.json; otherwise a slower rank can
    # mistake files written by rank 0 for a pre-existing output and abort.
    preexisting = int(out.exists() and any(out.iterdir()) and not args.resume)
    if world > 1:
        marker = torch.tensor(preexisting, dtype=torch.int32, device=device)
        dist.all_reduce(marker, op=dist.ReduceOp.MAX)
        preexisting = int(marker.item())
    if preexisting:
        raise RuntimeError(
            f"output directory '{out}' is not empty; use a new run directory or --resume"
        )
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "architecture_revision": ARCHITECTURE_REVISION,
                    "stage": args.stage,
                    "world_size": world,
                    "fa2": fa2,
                    "base_sha256": base_meta.get("checkpoint_sha256"),
                    "data_contract": data_contract,
                    "training_contract": trainer_contract,
                },
                indent=2,
            )
        )
    if world > 1:
        dist.barrier()

    step = start_epoch = batch_cursor = epoch_targets_seen = 0
    if args.resume:
        step, start_epoch, batch_cursor, epoch_targets_seen = _load_resume(
            args.resume,
            model,
            optimizer,
            args,
            base_meta,
            data_contract,
            world,
            rank,
            device,
        )
        trainable = [p for p in model.parameters() if p.requires_grad]

    current_epoch = start_epoch
    stop_requested = False
    for epoch in range(start_epoch, args.epochs):
        if args.max_updates is not None and step >= args.max_updates:
            break
        current_epoch = epoch
        _, batches, per_rank = compute_epoch_batches(
            len(dataset), args.global_batch_size, world, args.seed, epoch
        )
        kwargs = {
            "dataset": dataset,
            "batch_size": None,
            "sampler": SegmentSequenceSampler(per_rank[rank]),
            "num_workers": args.num_workers,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            import torch.multiprocessing as mp

            kwargs.update(prefetch_factor=2, multiprocessing_context=mp.get_context("spawn"))
        iterator = iter(DataLoader(**kwargs))
        epoch_complete = True
        for batch_index, global_batch in enumerate(batches):
            local_indices = global_batch[rank::world]
            samples = [next(iterator) for _ in local_indices]
            if epoch == start_epoch and batch_index < batch_cursor:
                continue
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(4, dtype=torch.float64, device=device)
            for sample in samples:
                total, gt, kd, count_local = _sample_loss(
                    model,
                    teacher_head,
                    sample,
                    device,
                    args.kd_weight,
                    teacher_policy=teacher_policy,
                    native_kd_weight=args.native_kd_weight if args.stage == "joint" else 0.0,
                )
                if isinstance(total, torch.Tensor) and total.requires_grad:
                    total.backward()
                if not isinstance(total, torch.Tensor):
                    total = torch.as_tensor(total, dtype=torch.float32, device=device)
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, dtype=torch.float32, device=device)
                if not isinstance(kd, torch.Tensor):
                    kd = torch.as_tensor(kd, dtype=torch.float32, device=device)
                sums += torch.tensor(
                    [total.detach().item(), gt.detach().item(), kd.detach().item(), count_local],
                    dtype=torch.float64,
                    device=device,
                )
            if world > 1:
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            count = int(round(float(sums[3].item())))
            has_grad = _allreduce_mixed_precision_gradients(trainable, count)
            grad_norm = 0.0
            updated = count > 0 and has_grad is not False
            if updated:
                grad_norm = clip_parameter_groups_norm(trainable, args.grad_clip_norm)
                optimizer.step()
                step += 1
            epoch_targets_seen += count
            batch_cursor = batch_index + 1
            if rank == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "batch_cursor": batch_cursor,
                    "loss": sums[0].item() / count if count else 0.0,
                    "gt_loss": sums[1].item() / count if count else 0.0,
                    "kd_loss": sums[2].item() / count if count else 0.0,
                    "grad_norm": float(grad_norm),
                    "target_count": count,
                    "updated": updated,
                }
                with (out / "train_metrics.jsonl").open("a", buffering=1) as handle:
                    handle.write(json.dumps(rec) + "\n")
            if updated and (step == 1 or step % args.save_every == 0):
                rng_states = _rng_states(world, device)
                if rank == 0:
                    _save(
                        out / "last.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        batch_cursor,
                        epoch_targets_seen,
                        args,
                        norm_stats,
                        base_meta,
                        data_contract,
                        world,
                        rng_states,
                    )
            if args.max_updates is not None and step >= args.max_updates:
                if batch_index + 1 < len(batches):
                    epoch_complete = False
                stop_requested = True
                break

        if epoch_complete and not (args.max_updates is not None and step >= args.max_updates and batch_cursor < len(batches)):
            current_epoch = epoch + 1
            batch_cursor = 0
            epoch_targets_seen = 0
            rng_states = _rng_states(world, device)
            if rank == 0:
                _save(
                    out / f"epoch_{current_epoch:03d}.pt",
                    model,
                    optimizer,
                    step,
                    current_epoch,
                    0,
                    0,
                    args,
                    norm_stats,
                    base_meta,
                    data_contract,
                    world,
                    rng_states,
                )
                _save(
                    out / "last.pt",
                    model,
                    optimizer,
                    step,
                    current_epoch,
                    0,
                    0,
                    args,
                    norm_stats,
                    base_meta,
                    data_contract,
                    world,
                    rng_states,
                )
        else:
            # Preserve the partial cursor when stopping at max-updates.
            rng_states = _rng_states(world, device)
            if rank == 0:
                _save(
                    out / "last.pt",
                    model,
                    optimizer,
                    step,
                    epoch,
                    batch_cursor,
                    epoch_targets_seen,
                    args,
                    norm_stats,
                    base_meta,
                    data_contract,
                    world,
                    rng_states,
                )
        if stop_requested:
            break

    rng_states = _rng_states(world, device)
    if rank == 0:
        _save(
            out / "adapter_final.pt",
            model,
            optimizer,
            step,
            current_epoch,
            batch_cursor,
            epoch_targets_seen,
            args,
            norm_stats,
            base_meta,
            data_contract,
            world,
            rng_states,
        )
        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "epoch": current_epoch,
                    "batch_cursor": batch_cursor,
                    "world_size": world,
                    "fa2": fa2,
                    "architecture_revision": ARCHITECTURE_REVISION,
                },
                indent=2,
            )
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__=='__main__': main()
