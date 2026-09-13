import argparse
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.data import MetaWorldWindows
from fabri_moss.runtime import load_native_checkpoint


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found for sha256 computation: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_flow_kd_loss(
    student_head: torch.nn.Module,
    teacher_head: torch.nn.Module,
    student_deep: torch.Tensor,
    student_shallow: torch.Tensor,
    teacher_deep: torch.Tensor,
    teacher_shallow: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    kd_weight: float = 1.0,
    fixed_noise: Optional[torch.Tensor] = None,
    fixed_t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = action_mask
    if valid.sum() == 0:
        raise ValueError("action_mask has sum 0; cannot compute flow loss.")

    actions_masked = actions * valid

    if fixed_noise is not None:
        noise = fixed_noise.to(device=actions.device, dtype=actions.dtype) * valid
    else:
        noise = (torch.rand_like(actions_masked) * 2.0 - 1.0) * valid

    if fixed_t is not None:
        t = fixed_t.to(device=actions.device, dtype=actions.dtype)
    else:
        b = actions.shape[0]
        alpha = torch.tensor(2.0, device=actions.device)
        beta = torch.tensor(2.0, device=actions.device)
        dist = torch.distributions.Beta(alpha, beta)
        t = dist.sample((b,)).clamp(0.02, 0.98).to(dtype=actions.dtype)

    t_view = t.view(-1, 1, 1)
    x_t = ((1.0 - t_view) * noise + t_view * actions_masked) * valid
    target_velocity = (actions_masked - noise) * valid

    v_student = student_head._predict_velocity(
        fused_tokens=student_deep,
        state=state,
        noisy_actions=x_t,
        t=t,
        shallow_tokens=student_shallow,
    ) * valid

    if not torch.isfinite(v_student).all():
        raise FloatingPointError("Non-finite values detected in student predicted velocity!")

    with torch.no_grad():
        v_teacher = teacher_head._predict_velocity(
            fused_tokens=teacher_deep,
            state=state,
            noisy_actions=x_t,
            t=t,
            shallow_tokens=teacher_shallow,
        ) * valid

    if not torch.isfinite(v_teacher).all():
        raise FloatingPointError("Non-finite values detected in teacher predicted velocity!")

    gt_diff = (v_student - target_velocity) * valid
    gt_loss = (gt_diff ** 2).sum() / valid.sum().clamp_min(1.0)

    kd_diff = (v_student - v_teacher.detach()) * valid
    kd_loss = (kd_diff ** 2).sum() / valid.sum().clamp_min(1.0)

    total_loss = gt_loss + kd_weight * kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite total loss computed in compute_flow_kd_loss!")

    return total_loss, gt_loss, kd_loss, v_student, v_teacher


def train_single_step(
    student_model: torch.nn.Module,
    teacher_policy: torch.nn.Module,
    teacher_head: torch.nn.Module,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    kd_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
    fixed_noise: Optional[torch.Tensor] = None,
    fixed_t: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    student_model.train()
    optimizer.zero_grad()

    images_window = batch["images_window"]
    frame_ids = batch["frame_ids"]
    prompt = batch["prompt"]
    state = batch["state"].to(device)
    state_mask = batch["state_mask"].to(device)
    state = state * state_mask

    actions = batch["actions"].to(device)
    action_mask = batch["action_mask"].to(device)

    student_deep, student_shallow = student_model(
        images_window=images_window,
        frame_ids=frame_ids,
        prompt=prompt,
    )

    current_images = images_window[-1]
    image_mask = torch.ones(len(current_images), dtype=torch.bool, device=device)

    with torch.no_grad():
        teacher_out = teacher_policy.get_vl_embeddings(
            images=current_images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=False,
            shallow_layer_index=6,
        )
        if isinstance(teacher_out, tuple):
            teacher_deep, teacher_shallow = teacher_out
        else:
            teacher_deep = teacher_out
            teacher_shallow = None

    student_head = student_model.policy.action_head
    loss, gt_loss, kd_loss, _, _ = compute_flow_kd_loss(
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
        fixed_noise=fixed_noise,
        fixed_t=fixed_t,
    )

    loss.backward()

    kv_grads = {}
    for name, p in student_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g_norm = float(p.grad.norm().item())
            g_finite = bool(torch.isfinite(p.grad).all().item())
            if "k_proj" in name or "v_proj" in name or "readout" in name:
                kv_grads[name] = {"norm": g_norm, "finite": g_finite}

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in student_model.parameters() if p.requires_grad],
        max_norm=grad_clip_norm,
        error_if_nonfinite=True,
    )

    optimizer.step()

    return {
        "loss": float(loss.item()),
        "gt_loss": float(gt_loss.item()),
        "kd_loss": float(kd_loss.item()),
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        "kv_grads": kv_grads,
    }


def save_adapter_checkpoint(
    output_path: Path,
    student_model: MossInternVL,
    optimizer: torch.optim.Optimizer,
    step: int,
    stage: str,
    args: argparse.Namespace,
    norm_stats: Dict[str, Any],
    base_metadata: Dict[str, Any],
    data_contract: Optional[Dict[str, Any]] = None,
    init_provenance: Optional[Dict[str, Any]] = None,
) -> None:
    config_dict = dataclasses.asdict(student_model.config)
    clean_args = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    adapter_dict = {
        "step": step,
        "stage": stage,
        "config": config_dict,
        "cross_blocks": student_model.cross_blocks.state_dict(),
        "readout_embeddings": student_model.readout_embeddings.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "norm_stats": norm_stats,
        "base_metadata": base_metadata,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "random": random.getstate(),
        },
        "args": clean_args,
    }
    # Save CUDA RNG state only if student parameters are actually on CUDA
    student_device = next(student_model.parameters()).device
    if student_device.type == "cuda" and torch.cuda.is_available():
        adapter_dict["rng_state"]["torch_cuda"] = torch.cuda.get_rng_state_all()

    if data_contract is not None:
        adapter_dict["data_contract"] = data_contract
    if init_provenance is not None:
        adapter_dict["init_provenance"] = init_provenance

    if stage == "expert":
        adapter_dict["action_head"] = student_model.policy.action_head.state_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_dict, str(output_path))


def validate_checkpoint_metadata_and_stats(
    ckpt: Dict[str, Any],
    student_model: MossInternVL,
    expected_norm_stats: Dict[str, Any],
    current_base_metadata: Dict[str, Any],
    allow_max_frames_mismatch: bool = False,
) -> None:
    if "base_metadata" not in ckpt or "checkpoint_sha256" not in ckpt["base_metadata"]:
        raise KeyError("Checkpoint missing required 'base_metadata.checkpoint_sha256'")
    if "checkpoint_sha256" not in current_base_metadata:
        raise KeyError("current_base_metadata missing required 'checkpoint_sha256'")
    resumed_sha = ckpt["base_metadata"]["checkpoint_sha256"]
    curr_sha = current_base_metadata["checkpoint_sha256"]
    if resumed_sha != curr_sha:
        raise ValueError(
            f"Base checkpoint SHA256 mismatch! Resumed: {resumed_sha}, current: {curr_sha}"
        )

    if "config" not in ckpt:
        raise KeyError("Checkpoint missing required field 'config'")
    resumed_cfg = dict(ckpt["config"])
    # Old checkpoints may lack memory_mode, which defaults to 'consume' only and must not be delta
    if "memory_mode" not in resumed_cfg:
        resumed_cfg["memory_mode"] = "consume"
    # Added for joint-stage training; legacy bridge/expert adapters are
    # equivalent to the default frozen-vision setting.
    if "train_vision" not in resumed_cfg:
        resumed_cfg["train_vision"] = False
    curr_cfg = dataclasses.asdict(student_model.config)
    for k, exp_val in curr_cfg.items():
        if allow_max_frames_mismatch and k == "max_frames":
            if k not in resumed_cfg:
                raise KeyError(f"Checkpoint config missing required field '{k}'")
            continue
        if k not in resumed_cfg:
            raise KeyError(f"Checkpoint config missing required field '{k}'")
        res_val = tuple(resumed_cfg[k]) if isinstance(resumed_cfg[k], list) else resumed_cfg[k]
        exp_cmp = tuple(exp_val) if isinstance(exp_val, list) else exp_val
        if res_val != exp_cmp:
            raise ValueError(f"Config mismatch for field '{k}': resumed {res_val} != expected {exp_cmp}")

    if "norm_stats" not in ckpt:
        raise KeyError("Checkpoint missing required field 'norm_stats'")
    resumed_stats = ckpt["norm_stats"]
    for key in ["observation.state", "action"]:
        if key not in resumed_stats or key not in expected_norm_stats:
            raise KeyError(f"norm_stats missing key '{key}'")
        for sub in ["min", "max"]:
            if not torch.equal(torch.as_tensor(resumed_stats[key][sub]), torch.as_tensor(expected_norm_stats[key][sub])):
                raise ValueError(f"norm_stats mismatch for {key}.{sub}!")


def initialize_adapter_weights(
    path: Union[str, Path],
    student_model: MossInternVL,
    expected_norm_stats: Dict[str, Any],
    current_base_metadata: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found at {path}")

    # Compute actual adapter file sha256
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    adapter_file_sha256 = h.hexdigest()

    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    # Validate base SHA, config (ignoring only max_frames), and norm stats
    validate_checkpoint_metadata_and_stats(
        ckpt=ckpt,
        student_model=student_model,
        expected_norm_stats=expected_norm_stats,
        current_base_metadata=current_base_metadata,
        allow_max_frames_mismatch=True,
    )

    if "stage" not in ckpt:
        raise KeyError("Checkpoint missing required field 'stage'")
    source_stage = ckpt["stage"]
    if source_stage not in ("bridge", "expert"):
        raise ValueError(f"Invalid source stage '{source_stage}' in checkpoint")

    target_stage = student_model.training_stage

    # Reject source expert -> target bridge
    if source_stage == "expert" and target_stage == "bridge":
        raise ValueError(
            "Cannot initialize target stage 'bridge' from source stage 'expert' (would drop trained expert action_head)"
        )

    # Check required keys before mutating any weights
    if "cross_blocks" not in ckpt:
        raise KeyError("Checkpoint missing required key 'cross_blocks'")
    if "readout_embeddings" not in ckpt:
        raise KeyError("Checkpoint missing required key 'readout_embeddings'")

    target_cb_state = student_model.cross_blocks.state_dict()
    source_cb_state = ckpt["cross_blocks"]
    if set(target_cb_state.keys()) != set(source_cb_state.keys()):
        raise KeyError("cross_blocks state_dict keys mismatch between student model and source checkpoint")
    for k, v in target_cb_state.items():
        src_v = source_cb_state[k]
        if v.shape != src_v.shape:
            raise ValueError(f"cross_blocks param shape mismatch for {k}: target {v.shape} vs source {src_v.shape}")

    source_readout = ckpt["readout_embeddings"]
    if student_model.readout_embeddings.shape != source_readout.shape:
        raise ValueError(
            f"readout_embeddings shape mismatch: target {student_model.readout_embeddings.shape} vs source {source_readout.shape} (broadcast disallowed)"
        )

    if source_stage == "expert" and target_stage == "expert":
        if "action_head" not in ckpt:
            raise KeyError("Source stage is 'expert' but checkpoint missing required 'action_head' state dict")
        target_head_state = student_model.policy.action_head.state_dict()
        source_head_state = ckpt["action_head"]
        if set(target_head_state.keys()) != set(source_head_state.keys()):
            raise KeyError("action_head state_dict keys mismatch between student model and source checkpoint")
        for k, v in target_head_state.items():
            src_v = source_head_state[k]
            if v.shape != src_v.shape:
                raise ValueError(f"action_head param shape mismatch for {k}: target {v.shape} vs source {src_v.shape}")

    source_window = ckpt['config']['max_frames']
    if type(source_window) is not int or source_window <= 0:
        raise ValueError(f"Source checkpoint config.max_frames must be a positive integer, got {source_window}")
    # Key check complete, now perform load
    changes_record = []
    student_model.cross_blocks.load_state_dict(source_cb_state, strict=True)
    changes_record.append("loaded cross_blocks")

    with torch.no_grad():
        student_model.readout_embeddings.copy_(source_readout.to(student_model.readout_embeddings.device))
    changes_record.append("copied readout_embeddings")

    if source_stage == "expert" and target_stage == "expert":
        student_model.policy.action_head.load_state_dict(ckpt["action_head"], strict=True)
        changes_record.append("loaded action_head from source expert")
    elif source_stage == "bridge" and target_stage == "expert":
        changes_record.append("kept base action_head for target expert (source was bridge)")

    student_model.set_training_stage(target_stage)
    target_window = student_model.config.max_frames
    source_step = int(ckpt.get("step", 0))

    provenance = {
        "source_path": str(path),
        "source_step": source_step,
        "source_stage": source_stage,
        "source_window": source_window,
        "target_stage": target_stage,
        "target_window": target_window,
        "sha256_actual_adapter_file": adapter_file_sha256,
        "changes_record": changes_record,
    }
    return provenance


def load_adapter_checkpoint(
    resume_path: Path,
    student_model: MossInternVL,
    optimizer: Optional[torch.optim.Optimizer],
    expected_stage: str,
    expected_norm_stats: Dict[str, Any],
    current_base_metadata: Dict[str, Any],
    device: str = "cpu",
    expected_data_contract: Optional[Dict[str, Any]] = None,
) -> int:
    resume_path = Path(resume_path).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found at {resume_path}")

    ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)

    if "stage" not in ckpt:
        raise KeyError("Checkpoint missing required field 'stage'")
    resumed_stage = ckpt["stage"]
    if resumed_stage != expected_stage:
        raise ValueError(f"Stage mismatch! Resumed: '{resumed_stage}', expected: '{expected_stage}'")

    validate_checkpoint_metadata_and_stats(
        ckpt=ckpt,
        student_model=student_model,
        expected_norm_stats=expected_norm_stats,
        current_base_metadata=current_base_metadata,
        allow_max_frames_mismatch=False,
    )

    if expected_data_contract is not None:
        if "data_contract" in ckpt:
            ckpt_contract = ckpt["data_contract"]
            if not isinstance(ckpt_contract, dict):
                raise ValueError("Checkpoint 'data_contract' must be a dict")
            # Compare entire dict rather than only a few keys
            if set(ckpt_contract.keys()) != set(expected_data_contract.keys()):
                missing = set(expected_data_contract.keys()) - set(ckpt_contract.keys())
                extra = set(ckpt_contract.keys()) - set(expected_data_contract.keys())
                raise ValueError(
                    f"data_contract keys mismatch! Missing in ckpt: {missing}, unexpected extra: {extra}"
                )
            for k, exp_val in expected_data_contract.items():
                res_val = ckpt_contract[k]
                if res_val != exp_val:
                    raise ValueError(
                        f"data_contract mismatch for '{k}': resumed {res_val} != expected {exp_val}"
                    )
        else:
            # Legacy checkpoint without data_contract:
            # Legacy context mode was always window; cannot resume in consume mode
            if expected_data_contract.get("context_mode") != "window":
                raise ValueError(
                    f"Cannot resume legacy checkpoint without data_contract with context_mode='{expected_data_contract.get('context_mode')}'; legacy checkpoints require context_mode='window'"
                )
            ckpt_args = ckpt.get("args", {})
            legacy_frame_stride = ckpt_args.get("frame_stride", 5)
            legacy_seed = ckpt_args.get("seed", 4042)
            legacy_max_episodes = ckpt_args.get("max_episodes", None)
            legacy_window = ckpt.get("config", {}).get("max_frames", 2)
            legacy_defaults = {
                "context_mode": "window",
                "window": legacy_window,
                "frame_stride": legacy_frame_stride,
                "min_context_frames": 1,
                "seed": legacy_seed,
                "max_episodes": legacy_max_episodes,
            }
            for lk, leg_v in legacy_defaults.items():
                if lk in expected_data_contract:
                    exp_v = expected_data_contract[lk]
                    if exp_v != leg_v:
                        raise ValueError(
                            f"Legacy checkpoint data_contract mismatch for '{lk}': legacy {leg_v} != expected {exp_v}"
                        )

    if expected_stage == "expert":
        if "action_head" not in ckpt:
            raise KeyError("Stage 'expert' requested but checkpoint missing required 'action_head' state dict!")
        student_model.policy.action_head.load_state_dict(ckpt["action_head"], strict=True)

    if "cross_blocks" not in ckpt or "readout_embeddings" not in ckpt:
        raise KeyError("Checkpoint missing 'cross_blocks' or 'readout_embeddings'")
    student_model.cross_blocks.load_state_dict(ckpt["cross_blocks"], strict=True)
    with torch.no_grad():
        student_model.readout_embeddings.copy_(ckpt["readout_embeddings"])

    if optimizer is not None:
        if "optimizer" not in ckpt:
            raise KeyError("Optimizer requested to resume but checkpoint missing 'optimizer' state dict!")
        optimizer.load_state_dict(ckpt["optimizer"])

    if "rng_state" in ckpt:
        rng = ckpt["rng_state"]
        if "torch" in rng:
            rng_t = rng["torch"]
            if isinstance(rng_t, torch.Tensor):
                rng_t = rng_t.cpu()
            torch.set_rng_state(rng_t)
        if "random" in rng:
            random.setstate(rng["random"])
        if "torch_cuda" in rng:
            student_device = next(student_model.parameters()).device
            if student_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])

    # Invalidate session/KV and enforce stage after weight restoration
    student_model.set_training_stage(expected_stage)
    student_model.train()

    return int(ckpt.get("step", 0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabriVLA MOSS Distillation Trainer")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to native FabriVLA repo")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Path to base checkpoint .pt")
    parser.add_argument("--vlm", type=str, default="/root/models/InternVL3_5-1B", help="Path to local InternVL model")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MetaWorld LeRobot dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save adapters and metrics")
    parser.add_argument("--steps", type=int, default=100, help="Number of additional training steps")
    parser.add_argument("--stage", type=str, default="bridge", choices=["bridge", "expert"], help="Training stage")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (positive float)")
    parser.add_argument("--kd-weight", type=float, default=1.0, help="Weight for KD loss term (>= 0)")
    parser.add_argument("--window", type=int, default=2, help="Visual observation window size (>= 1)")
    parser.add_argument("--frame-stride", type=int, default=5, help="Observation frame stride (>= 1)")
    parser.add_argument("--context-mode", type=str, default="window", choices=["window", "consume"], help="Context frame selection mode")
    parser.add_argument("--min-context-frames", type=int, default=1, help="Minimum context frames in consume mode (>= 1)")
    parser.add_argument("--num-readout-tokens", type=int, default=16, help="Number of readout tokens (>= 1)")
    parser.add_argument("--seed", type=int, default=4042, help="Random seed")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to load")
    parser.add_argument("--fixed-sample", action="store_true", help="Diagnostic: repeatedly train on sample")
    parser.add_argument("--sample-index", type=int, default=None, help="Index of sample for fixed-sample training")
    parser.add_argument("--fixed-noise", action="store_true", help="Diagnostic: use deterministic noise and time")
    parser.add_argument("--save-every", type=int, default=100, help="Save interval (>= 1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to adapter checkpoint to resume from")
    parser.add_argument("--init-adapter", type=str, default=None, help="Path to adapter checkpoint to warmstart/initialize from")
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads (>= 1)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Model device; GPU by default, cpu for functional tests")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key in norm stats")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive integer, got {args.steps}")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError(f"--lr must be positive float, got {args.lr}")
    if not math.isfinite(args.kd_weight) or args.kd_weight < 0.0:
        raise ValueError(f"--kd-weight must be non-negative float, got {args.kd_weight}")
    if args.window <= 0:
        raise ValueError(f"--window must be >= 1, got {args.window}")
    if args.frame_stride <= 0:
        raise ValueError(f"--frame-stride must be >= 1, got {args.frame_stride}")
    if args.num_readout_tokens <= 0:
        raise ValueError(f"--num-readout-tokens must be >= 1, got {args.num_readout_tokens}")
    if args.save_every <= 0:
        raise ValueError(f"--save-every must be >= 1, got {args.save_every}")
    if args.threads <= 0:
        raise ValueError(f"--threads must be >= 1, got {args.threads}")

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.resume and args.init_adapter:
        raise ValueError("Cannot specify both --resume and --init-adapter: they are mutually exclusive.")

    if args.min_context_frames < 1 or args.min_context_frames > args.window:
        raise ValueError(
            f"--min-context-frames must be in [1, window ({args.window})], got {args.min_context_frames}"
        )

    output_dir = Path(args.output_dir).resolve()
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory {output_dir} already exists and is not empty! Use --resume or clean it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_log_path = output_dir / "train_metrics.jsonl"
    final_ckpt_path = output_dir / "adapter_final.pt"

    native_policy, raw_config, norm_stats, base_meta = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm,
        device=args.device,
        arm_key=args.arm_key,
    )

    moss_config = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=args.num_readout_tokens,
        max_frames=args.window,
        max_text_tokens=1024,
        shallow_layer=6,
    )
    student_model = MossInternVL(native_policy, moss_config)
    student_model.set_training_stage(args.stage)

    if args.stage == "expert":
        teacher_head = copy.deepcopy(student_model.policy.action_head)
        teacher_head.eval()
        for p in teacher_head.parameters():
            p.requires_grad = False
    else:
        teacher_head = student_model.policy.action_head
        teacher_head.eval()

    init_provenance = None
    if args.init_adapter:
        init_provenance = initialize_adapter_weights(
            path=args.init_adapter,
            student_model=student_model,
            expected_norm_stats=norm_stats,
            current_base_metadata=base_meta,
            device=args.device,
        )
        # Calling set_training_stage enforces stage and invalidates revision (no optimizer/step/RNG mutation)
        student_model.set_training_stage(args.stage)

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    param_names = [n for n, p in student_model.named_parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    dataset = MetaWorldWindows(
        root=args.data_root,
        norm_stats=norm_stats,
        horizon=50,
        window=args.window,
        frame_stride=args.frame_stride,
        split="train",
        seed=args.seed,
        max_episodes=args.max_episodes,
        context_mode=args.context_mode,
        min_context_frames=args.min_context_frames,
    )

    data_contract = {
        "context_mode": args.context_mode,
        "window": args.window,
        "frame_stride": args.frame_stride,
        "min_context_frames": args.min_context_frames,
        "seed": args.seed,
        "max_episodes": args.max_episodes,
        "split": "train",
        "active_episode_ids": sorted([int(ep["episode_index"]) for ep in dataset.active_episodes]),
        "metadata_files_sha256": {
            "info.json": compute_file_sha256(dataset.root / "meta" / "info.json"),
            "tasks.jsonl": compute_file_sha256(dataset.root / "meta" / "tasks.jsonl"),
            "episodes.jsonl": compute_file_sha256(dataset.root / "meta" / "episodes.jsonl"),
        },
    }

    start_step = 0
    if args.resume:
        start_step = load_adapter_checkpoint(
            resume_path=Path(args.resume),
            student_model=student_model,
            optimizer=optimizer,
            expected_stage=args.stage,
            expected_norm_stats=norm_stats,
            current_base_metadata=base_meta,
            device=args.device,
            expected_data_contract=data_contract,
        )

    fixed_noise = None
    fixed_t = None
    if args.fixed_noise:
        generator = torch.Generator().manual_seed(args.seed)
        fixed_noise = torch.rand(1, 50, 24, generator=generator) * 2.0 - 1.0
        fixed_t = torch.tensor([0.5], dtype=torch.float32)

    chosen_sample_idx = args.sample_index
    if args.fixed_sample and chosen_sample_idx is None:
        if args.context_mode == "consume":
            lens = [len(h) for h in dataset._consume_histories]
            max_len = max(lens)
            chosen_sample_idx = lens.index(max_len)
        else:
            chosen_sample_idx = min(args.frame_stride, len(dataset) - 1)

    if chosen_sample_idx is not None and not 0 <= chosen_sample_idx < len(dataset):
        raise ValueError("sample-index must select an existing dataset sample")
    total_target_steps = start_step + args.steps
    step_idx = start_step

    train_log_file = open(metrics_log_path, "a", buffering=1)
    t0 = time.time()

    last_metrics = {}
    while step_idx < total_target_steps:
        sample_idx = chosen_sample_idx if args.fixed_sample else random.randint(0, len(dataset) - 1)
        batch = dataset[sample_idx]

        metrics = train_single_step(
            student_model=student_model,
            teacher_policy=native_policy,
            teacher_head=teacher_head,
            batch=batch,
            optimizer=optimizer,
            kd_weight=args.kd_weight,
            fixed_noise=fixed_noise,
            fixed_t=fixed_t,
            device=args.device,
        )
        last_metrics = metrics
        step_idx += 1
        elapsed = time.time() - t0

        log_entry = {
            "step": step_idx,
            "sample_idx": sample_idx,
            "episode_id": batch.get("episode_id"),
            "frame_ids": batch.get("frame_ids"),
            "context_mode": batch.get("context_mode"),
            "loss": metrics["loss"],
            "gt_loss": metrics["gt_loss"],
            "kd_loss": metrics["kd_loss"],
            "grad_norm": metrics["grad_norm"],
            "elapsed_sec": round(elapsed, 2),
        }
        train_log_file.write(json.dumps(log_entry) + "\n")

        if step_idx % args.save_every == 0 and step_idx < total_target_steps:
            save_adapter_checkpoint(
                output_path=output_dir / f"adapter_step_{step_idx}.pt",
                student_model=student_model,
                optimizer=optimizer,
                step=step_idx,
                stage=args.stage,
                args=args,
                norm_stats=norm_stats,
                base_metadata=base_meta,
                data_contract=data_contract,
                init_provenance=init_provenance,
            )

    train_log_file.close()

    save_adapter_checkpoint(
        output_path=final_ckpt_path,
        student_model=student_model,
        optimizer=optimizer,
        step=step_idx,
        stage=args.stage,
        args=args,
        norm_stats=norm_stats,
        base_metadata=base_meta,
        data_contract=data_contract,
        init_provenance=init_provenance,
    )

    metrics_summary = {
        "start_step": start_step,
        "additional_steps": args.steps,
        "total_steps": step_idx,
        "final_loss": last_metrics.get("loss"),
        "final_gt_loss": last_metrics.get("gt_loss"),
        "final_kd_loss": last_metrics.get("kd_loss"),
        "final_grad_norm": last_metrics.get("grad_norm"),
        "trainable_parameters_count": sum(p.numel() for p in trainable_params),
        "trainable_parameter_names": param_names[:20],
        "device": args.device,
        "stage": args.stage,
        "base_sha256": base_meta.get("checkpoint_sha256"),
        "data_contract": data_contract,
        "init_provenance": init_provenance,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)


if __name__ == "__main__":
    main()
