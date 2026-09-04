"""Full-parameter trainer for the continuous MOSS-Action stream."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from data import LiberoHDF5Dataset, MossActionCollator
from model import (
    MossActionConfig,
    MossActionVLA,
    checkpoint_fingerprint,
    load_trainable_state_dict,
    load_truncated_moss,
    parameter_count,
    trainable_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "preflight", "train"))
    parser.add_argument("--moss-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/moss_action.pt")
    )
    parser.add_argument("--init-action", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--action-offset", type=int, default=1)
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="raw demo steps between frames"
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.1,
        help="seconds per raw demo step",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8, help="actions per micro-chunk"
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="visual frames kept before each planning time",
    )
    parser.add_argument(
        "--max-visual-age-steps",
        type=int,
        default=2,
        help="delay-aware training: how stale the newest frame may be",
    )
    parser.add_argument("--overfit-one", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attention-backend", default="flash_attention_2")
    args = parser.parse_args()
    if args.command != "audit" and args.data is None:
        parser.error("--data is required for preflight and train")
    positive = (
        args.epochs,
        args.batch_size,
        args.gradient_accumulation,
        args.frame_stride,
        args.frame_interval,
        args.chunk_size,
        args.context_frames,
        args.lr,
        args.backbone_lr,
        args.save_every,
    )
    if (
        any(value <= 0 for value in positive)
        or args.action_offset < 0
        or args.max_visual_age_steps < 0
    ):
        parser.error(
            "training counts/rates must be positive; action-offset and "
            "max-visual-age-steps nonnegative"
        )
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    return args


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _load_action_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "moss_action_v6":
        raise ValueError(f"unsupported action checkpoint: {path}")
    return payload


def _build_policy(args: argparse.Namespace, initial: dict[str, Any] | None):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = _dtype(args.dtype) if device.type == "cuda" else torch.float32
    backend = args.attention_backend if device.type == "cuda" else "eager"
    backbone, audit = load_truncated_moss(
        args.moss_checkpoint,
        dtype=dtype,
        device_map={"": str(device)},
        attention_backend=backend,
    )
    hidden_size = int(backbone.moss.config.text_config.hidden_size)
    config = (
        MossActionConfig(**initial["config"])
        if initial is not None
        else MossActionConfig(
            moss_hidden_size=hidden_size,
            chunk_size=args.chunk_size,
            control_interval=args.frame_interval,
        )
    )
    if config.moss_hidden_size != hidden_size:
        raise ValueError(
            f"action checkpoint expects MOSS width {config.moss_hidden_size}, loaded {hidden_size}"
    )
    policy = MossActionVLA(backbone, config).to(device)
    enable_checkpointing = getattr(
        policy.backbone.moss, "gradient_checkpointing_enable", None
    )
    if callable(enable_checkpointing):
        enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
    if initial is not None:
        load_trainable_state_dict(policy, initial["trainable_state"])
    return policy, audit, device, dtype


def _save(
    path: Path,
    policy: MossActionVLA,
    *,
    args: argparse.Namespace,
    step: int,
    normalization: dict[str, list[float]],
    fingerprint: dict[str, Any],
    audit: Any,
) -> None:
    payload = {
        "format": "moss_action_v6",
        "config": asdict(policy.config),
        "trainable_state": trainable_state_dict(policy),
        "normalization": normalization,
        "global_step": step,
        "moss_checkpoint": fingerprint,
        "checkpoint_audit": audit.to_dict(),
        "training_contract": {
            "stage": "streaming",
            "retained_layers": 24,
            "retained_cross_attention_layers": [2, 6, 10, 14, 18, 22],
            "action_decoder": "ephemeral_action_queries",
            "action_memory": "final_hidden_state",
            "action_history": "none",
            "runtime": "streaming_action_query_decoder_v1",
            "training_context": "single_planning_time",
            "supervision": "one_micro_chunk_per_planning_time",
            "action_generation": "continuous_micro_chunk_l1",
            "delay_aware": "per_query_visual_age",
            "action_offset": args.action_offset,
            "frame_stride": args.frame_stride,
            "frame_interval": args.frame_interval,
            "context_frames": args.context_frames,
            "max_visual_age_steps": args.max_visual_age_steps,
            "backbone_training": "full",
            "backbone_lr": args.backbone_lr,
            "action_lr": args.lr,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    initial = _load_action_payload(args.init_action)
    fingerprint = checkpoint_fingerprint(args.moss_checkpoint)
    if initial is not None:
        if (
            initial["training_contract"].get("backbone_training") != "full"
            or initial["training_contract"].get("training_context")
            != "single_planning_time"
        ):
            raise ValueError("initial checkpoint is not a full-parameter stream model")
        if (
            initial["moss_checkpoint"]["combined_sha256"]
            != fingerprint["combined_sha256"]
        ):
            raise ValueError("initial action checkpoint uses different MOSS-VL weights")
        expected_data_contract = {
            "action_offset": args.action_offset,
            "frame_stride": args.frame_stride,
            "frame_interval": args.frame_interval,
            "context_frames": args.context_frames,
            "max_visual_age_steps": args.max_visual_age_steps,
        }
        mismatched = {
            key: (initial["training_contract"].get(key), value)
            for key, value in expected_data_contract.items()
            if initial["training_contract"].get(key) != value
        }
        if mismatched:
            raise ValueError(f"initial checkpoint data contract mismatch: {mismatched}")
        if initial["config"].get("chunk_size") != args.chunk_size:
            raise ValueError(
                "initial checkpoint chunk_size "
                f"{initial['config'].get('chunk_size')} != {args.chunk_size}"
            )
    policy, audit, device, dtype = _build_policy(args, initial)
    if initial is not None:
        del initial["trainable_state"]
    if initial is not None and initial["checkpoint_audit"] != audit.to_dict():
        raise ValueError("initial action checkpoint has a different MOSS loading audit")
    counts = parameter_count(policy)
    if counts["trainable"] != counts["total"]:
        raise RuntimeError(f"full fine-tuning contract violated: {counts}")
    print(
        json.dumps(
            {
                "parameters": counts,
                "deleted_layers": list(audit.deleted_layer_indices),
                "checkpoint_sha256": fingerprint["combined_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.command == "audit":
        return

    initial_normalization = None if initial is None else initial["normalization"]
    dataset = LiberoHDF5Dataset(
        args.data,
        chunk_size=policy.config.chunk_size,
        action_offset=args.action_offset,
        frame_stride=args.frame_stride,
        frame_interval=args.frame_interval,
        context_frames=args.context_frames,
        max_visual_age_steps=args.max_visual_age_steps,
        seed=args.seed,
        state_low=None
        if initial_normalization is None
        else initial_normalization["state_q01"],
        state_high=None
        if initial_normalization is None
        else initial_normalization["state_q99"],
    )
    processor = AutoProcessor.from_pretrained(
        str(args.moss_checkpoint.expanduser().resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"
    if args.overfit_one:
        training_dataset = Subset(dataset, [0])
    else:
        training_dataset = dataset
    loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=not args.overfit_one,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=MossActionCollator(processor),
    )
    backbone_parameters = list(policy.backbone.parameters())
    action_parameters = list(policy.decoder.parameters())
    parameters = backbone_parameters + action_parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": action_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    policy.train()

    max_steps = 1 if args.command == "preflight" else args.max_steps
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            batch = _move(batch, device)
            autocast = torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            )
            with autocast:
                raw_loss = policy(
                    batch["moss_inputs"],
                    batch["robot_state"],
                    batch["state_velocity"],
                    batch["actions"],
                    batch["visual_age"],
                    query_delays=batch["query_delays"],
                    valid_mask=batch["action_valid_mask"],
                )["loss"]
                loss = raw_loss / args.gradient_accumulation
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite action loss at optimizer step {global_step}"
                )
            loss.backward()
            should_step = (
                batch_index + 1
            ) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(
                parameters, 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(
                f"step={global_step} loss={float(loss) * args.gradient_accumulation:.6f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            if args.command == "train" and global_step % args.save_every == 0:
                _save(
                    args.output,
                    policy,
                    args=args,
                    step=global_step,
                    normalization=dataset.normalization,
                    fingerprint=fingerprint,
                    audit=audit,
                )
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break
    if global_step == 0:
        raise RuntimeError("training produced no optimizer step")
    if args.command == "train":
        _save(
            args.output,
            policy,
            args=args,
            step=global_step,
            normalization=dataset.normalization,
            fingerprint=fingerprint,
            audit=audit,
        )
        print(f"saved={args.output}", flush=True)
    else:
        print("PASS preflight forward/backward/optimizer step", flush=True)


if __name__ == "__main__":
    main()
