#!/usr/bin/env python3
"""GPU verification probe for predictive memory policy, causal writer, and future latent head.

Verifies:
1. Loads full FP32 parent checkpoint via load_native_checkpoint(trainable=True).
2. Asserts FlashAttention-2 backend via assert_native_fa2.
3. Wraps policy into PredictiveMemoryPolicy(policy, writer_config=WriterConfig(), shallow_layer=6, gradient_checkpointing=True).
   Enforces all base parameters requires_grad=False and base policy in eval mode, while model is in train mode.
4. Loads PredictiveTrainingDataset with split='train', seed=4042, augmentation=False, history_frames=16, target_frames=8.
   Scans dataset via verify_memory_training.select_layout_segments to select 'memory' segment (mature memory sequence with maximum N).
   Records N, M, future_valid counts, source SHA/step, and FA2 backend diagnostics.
5. Sets up a temporary AdamW optimizer (lr=1e-4) covering new modules (writer + future_head) without saving weights.
   Executes 4 consecutive optimization steps on the exact same sample:
   - Steps 1 & 2: future_weight = 0.0
   - Steps 3 & 4: future_weight = 0.001
   Verifies:
   - All frozen base parameters have grad is None across all steps.
   - Frozen base parameters have exact identical hash/state between initial and post-4-step completion.
   - writer out_proj changes at Step 1.
   - writer upstream parameters receive non-zero gradient by Step 2.
   - future_head parameters have grad is None at Steps 1 & 2, and receive non-zero gradient / positive updates at Steps 3 & 4.
   - Future loss is finite, total loss = action_loss + future_weight * future_loss.
6. Information leakage prevention check:
   - features() and predict_actions() do not read future keys.
   - Substituting future_images (preserving shape) produces exact bit-for-bit identical BF16 features.
   - predict_actions() executes original 50-step flow sampling for the last target, producing shape (1, 50, 24),
     and succeeds even when future keys are deleted.
7. Writes comprehensive JSON report to output directory. Status is 'PASSED' only if all assertions pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fabri_moss.predictive_data import PredictiveTrainingDataset
from fabri_moss.predictive_memory import WriterConfig
from fabri_moss.predictive_policy import FutureLatentHead, PredictiveMemoryPolicy
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint
from fabri_moss.verify_memory_training import select_layout_segments


def write_report(out_dir: Path, data: Dict[str, Any]) -> None:
    """Safely write JSON report via temporary file replace."""
    tmp = out_dir / "report.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(out_dir / "report.json")


def compute_frozen_base_hash(policy: nn.Module) -> str:
    """Compute deterministic SHA-256 hash over all frozen base policy parameters on CPU."""
    hasher = hashlib.sha256()
    for name, param in sorted(policy.named_parameters()):
        # Parameter tensor on CPU in raw contiguous bytes
        data_bytes = param.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(data_bytes)
    return hasher.hexdigest()


def run_verification(args: argparse.Namespace) -> None:
    """Main verification procedure."""
    # Deterministic seed setup
    seed = 4042
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Validate output directory
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise ValueError(f"Output directory {out_dir} exists and is not empty.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but verify_predictive requires GPU execution.")
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise ValueError(f"verify_predictive is GPU-only: device must be cuda, got {args.device!r}")
    torch.cuda.set_device(dev)
    torch.set_num_threads(2)

    report: Dict[str, Any] = {
        "status": "RUNNING",
        "cli_args": {
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
            "device": str(args.device),
            "epoch": int(args.epoch),
            "fabri_root": str(args.fabri_root),
            "vlm_path": str(args.vlm_path),
            "data_root": str(args.data_root),
        },
        "deterministic_seed": seed,
        "config_metadata": {},
        "timings": {},
        "peak_gpu_mib": 0.0,
        "step_results": [],
        "frozen_base_verification": {},
        "leakage_verification": {},
        "action_inference": {},
    }
    write_report(out_dir, report)

    try:
        # 1. Load native checkpoint (trainable=True) to inspect FP32 parent
        t_load_0 = time.perf_counter()
        base_policy, raw_config, norm_stats, src_meta = load_native_checkpoint(
            fabri_root=args.fabri_root,
            checkpoint_path=args.checkpoint,
            vlm_path=args.vlm_path,
            device=args.device,
            trainable=True,
        )
        t_load = time.perf_counter() - t_load_0
        report["timings"]["load_checkpoint_sec"] = t_load

        # Assert FlashAttention-2 backend
        fa2_meta = assert_native_fa2(base_policy)
        report["config_metadata"]["backend_fa2"] = fa2_meta
        report["config_metadata"]["source_checkpoint_sha256"] = src_meta.get("checkpoint_sha256")
        report["config_metadata"]["source_step"] = src_meta.get("step")

        # 2. Wrap into PredictiveMemoryPolicy
        writer_config = WriterConfig()
        model = PredictiveMemoryPolicy(
            policy=base_policy,
            writer_config=writer_config,
            shallow_layer=6,
            gradient_checkpointing=True,
        )
        model.to(dev)
        model.train()  # Model in train mode

        # Verify base policy parameters are frozen, FP32, and in eval mode
        base_params_count = 0
        for n, p in base_policy.named_parameters():
            base_params_count += p.numel()
            if p.requires_grad:
                raise AssertionError(f"Base parameter {n} requires_grad must be False, got True")
            if p.dtype != torch.float32:
                raise AssertionError(f"Base parameter {n} dtype must be float32, got {p.dtype}")
        if base_policy.training:
            raise AssertionError("Base policy must remain in eval mode (training==False)")

        # Verify writer and future_head are in train mode and requires_grad=True
        writer_params_count = 0
        for n, p in model.writer.named_parameters():
            writer_params_count += p.numel()
            if not p.requires_grad:
                raise AssertionError(f"Writer parameter {n} must have requires_grad=True")
            if p.dtype != torch.float32:
                raise AssertionError(f"Writer parameter {n} dtype must be float32, got {p.dtype}")

        head_params_count = 0
        for n, p in model.future_head.named_parameters():
            head_params_count += p.numel()
            if not p.requires_grad:
                raise AssertionError(f"Future head parameter {n} must have requires_grad=True")
            if p.dtype != torch.float32:
                raise AssertionError(f"Future head parameter {n} dtype must be float32, got {p.dtype}")

        report["config_metadata"]["base_params_count"] = base_params_count
        report["config_metadata"]["writer_params_count"] = writer_params_count
        report["config_metadata"]["future_head_params_count"] = head_params_count
        write_report(out_dir, report)

        # Initial CPU hash of frozen base parameters (CPU hash entire frozen state ~3.5GB)
        t_hash_0 = time.perf_counter()
        initial_frozen_hash = compute_frozen_base_hash(base_policy)
        t_hash_init = time.perf_counter() - t_hash_0
        report["frozen_base_verification"]["initial_hash"] = initial_frozen_hash
        report["timings"]["initial_frozen_hash_sec"] = t_hash_init

        # 3. Load dataset and select memory segment
        ds = PredictiveTrainingDataset(
            root=args.data_root,
            norm_stats=norm_stats,
            history_frames=16,
            target_frames=8,
            split="train",
            seed=4042,
            augmentation=False,
        )
        ds.set_epoch(args.epoch)

        selected_segments = select_layout_segments(ds)
        if "memory" not in selected_segments:
            raise RuntimeError("select_layout_segments failed to return 'memory' segment")

        seg_idx, seg_layout = selected_segments["memory"]
        sample = ds[seg_idx]
        sample = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

        N = len(sample["frame_ids"])
        M = len(sample["target_indices"])
        future_valid_tensor = sample.get("future_valid")
        if future_valid_tensor is None:
            raise AssertionError("PredictiveTrainingDataset sample missing 'future_valid'")
        future_valid_count = int(future_valid_tensor.sum().item())
        if not sample.get("memory_replay") or future_valid_count <= 0:
            raise AssertionError("Native probe requires a memory sample with valid future targets")

        report["config_metadata"]["sample_info"] = {
            "segment_index": int(seg_idx),
            "num_observations_N": int(N),
            "num_targets_M": int(M),
            "future_valid_count": int(future_valid_count),
            "future_horizons": [float(h) for h in ds.future_horizons],
        }
        write_report(out_dir, report)

        # 4. Setup temporary AdamW optimizer for writer and future_head only
        opt_params = list(model.writer.parameters()) + list(model.future_head.parameters())
        optimizer = torch.optim.AdamW(opt_params, lr=1e-4)

        # Track representative parameters across steps
        writer_out_proj_params = list(model.writer.out_proj.named_parameters())
        if not writer_out_proj_params:
            raise AssertionError("Could not locate any out_proj parameters in model.writer")
        rep_out_proj_name, rep_out_proj_p = writer_out_proj_params[0]
        init_out_proj_val = rep_out_proj_p.detach().clone()

        writer_upstream_params = [(n, p) for n, p in model.writer.named_parameters() if not n.startswith("out_proj.")]
        if not writer_upstream_params:
            raise AssertionError("Could not locate any upstream parameters in model.writer")

        future_head_params = list(model.future_head.named_parameters())
        if not future_head_params:
            raise AssertionError("model.future_head has no parameters")
        rep_head_name, rep_head_p = future_head_params[0]
        init_head_val = rep_head_p.detach().clone()
        initial_head = {name: p.detach().clone() for name, p in future_head_params}

        # 4 consecutive updates on the same sample
        # Steps 1 & 2: future_weight = 0.0
        # Steps 3 & 4: future_weight = 0.001
        step_weights = [0.0, 0.0, 0.001, 0.001]
        step_records: List[Dict[str, Any]] = []

        # Store baseline values for delta checks
        prev_out_proj_val = init_out_proj_val
        prev_head_val = init_head_val

        for step_idx, fut_weight in enumerate(step_weights, start=1):
            t_step_0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(dev)
            optimizer.zero_grad(set_to_none=True)

            out = model.forward(sample, future_weight=fut_weight)
            loss = out["loss"]
            action_loss = out["action_loss"]
            future_loss = out["future_loss"]

            if not torch.isfinite(loss):
                raise AssertionError(f"Step {step_idx} loss is not finite: {loss.item()}")
            if not torch.isfinite(action_loss):
                raise AssertionError(f"Step {step_idx} action_loss is not finite: {action_loss.item()}")
            if not torch.isfinite(future_loss):
                raise AssertionError(f"Step {step_idx} future_loss is not finite: {future_loss.item()}")

            expected_total = action_loss + fut_weight * future_loss
            if not torch.allclose(loss, expected_total, atol=1e-5):
                raise AssertionError(
                    f"Step {step_idx} total loss {loss.item()} != action_loss {action_loss.item()} + "
                    f"weight {fut_weight} * future_loss {future_loss.item()}"
                )

            # Backward pass + clip + step
            loss.backward()

            # Assert all frozen base policy parameters have grad is None
            for bn, bp in base_policy.named_parameters():
                if bp.grad is not None:
                    raise AssertionError(f"Step {step_idx} frozen base param {bn} has non-None grad!")

            for p in opt_params:
                if p.dtype != torch.float32 or not torch.isfinite(p).all():
                    raise AssertionError("New parameters must be finite FP32")
                if p.grad is not None and (p.grad.dtype != torch.float32 or not torch.isfinite(p.grad).all()):
                    raise AssertionError("New gradients must be finite FP32")

            # Check writer gradients
            out_proj_grad_norm = rep_out_proj_p.grad.norm().item() if rep_out_proj_p.grad is not None else 0.0
            upstream_grads = [
                p.grad.norm().item()
                for _, p in writer_upstream_params
                if p.grad is not None
            ]
            max_upstream_grad = max(upstream_grads) if upstream_grads else 0.0

            # Step 1: writer.out_proj has grad
            if step_idx == 1:
                if rep_out_proj_p.grad is None or out_proj_grad_norm <= 0.0:
                    raise AssertionError(f"Step 1 writer out_proj grad must be > 0, got {out_proj_grad_norm}")

            # Step 2+: upstream writer parameters must receive grad > 0
            if step_idx >= 2:
                if max_upstream_grad <= 0.0:
                    raise AssertionError(f"Step {step_idx} upstream writer gradient must be > 0, got {max_upstream_grad}")

            # Check future_head gradients
            head_grads = [
                p.grad.norm().item()
                for _, p in future_head_params
                if p.grad is not None
            ]

            if fut_weight == 0.0:
                # Steps 1 & 2: future_head must have all grads None
                for hn, hp in future_head_params:
                    if hp.grad is not None:
                        raise AssertionError(f"Step {step_idx} (future_weight=0) future_head param {hn} has non-None grad!")
            else:
                # Steps 3 & 4: future_head must have positive grad
                if not head_grads or max(head_grads) <= 0.0:
                    raise AssertionError(f"Step {step_idx} (future_weight={fut_weight}) future_head must have positive grad")

            # Gradient clip and step
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            for state in optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                        raise AssertionError("AdamW moments must be finite FP32")
            if fut_weight == 0.0:
                for name, p in future_head_params:
                    if not torch.equal(p, initial_head[name]) or p in optimizer.state:
                        raise AssertionError("Stage 1 must not change any future-head weight or create optimizer state")
            torch.cuda.synchronize(dev)

            # Check parameter updates
            curr_out_proj_val = rep_out_proj_p.detach().clone()
            out_proj_delta = (curr_out_proj_val - prev_out_proj_val).abs().max().item()

            curr_head_val = rep_head_p.detach().clone()
            head_delta = (curr_head_val - prev_head_val).abs().max().item()

            if step_idx == 1:
                if out_proj_delta <= 0.0:
                    raise AssertionError(f"Step 1 writer out_proj delta must be > 0, got {out_proj_delta}")

            if fut_weight == 0.0:
                if head_delta != 0.0:
                    raise AssertionError(f"Step {step_idx} future_head changed while future_weight=0.0! delta={head_delta}")
            else:
                if head_delta <= 0.0:
                    raise AssertionError(f"Step {step_idx} future_head must have positive update delta, got {head_delta}")

            prev_out_proj_val = curr_out_proj_val
            prev_head_val = curr_head_val

            step_sec = time.perf_counter() - t_step_0
            peak_mib = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
            report["peak_gpu_mib"] = max(report["peak_gpu_mib"], peak_mib)

            step_info = {
                "step": step_idx,
                "future_weight": fut_weight,
                "loss": float(loss.item()),
                "action_loss": float(action_loss.item()),
                "future_loss": float(future_loss.item()),
                "out_proj_grad_norm": float(out_proj_grad_norm),
                "max_upstream_grad_norm": float(max_upstream_grad),
                "head_grad_norm": float(max(head_grads)) if head_grads else 0.0,
                "out_proj_update_delta": float(out_proj_delta),
                "head_update_delta": float(head_delta),
                "step_duration_sec": float(step_sec),
                "peak_gpu_mib": float(peak_mib),
            }
            step_records.append(step_info)
            report["step_results"] = step_records
            write_report(out_dir, report)

        # Final CPU hash of frozen base policy parameters to assert bit-for-bit invariance
        t_hash_1 = time.perf_counter()
        final_frozen_hash = compute_frozen_base_hash(base_policy)
        t_hash_final = time.perf_counter() - t_hash_1
        report["timings"]["final_frozen_hash_sec"] = t_hash_final
        report["frozen_base_verification"]["final_hash"] = final_frozen_hash

        if initial_frozen_hash != final_frozen_hash:
            raise AssertionError(
                f"Frozen base policy hash changed across 4 optimization steps!\n"
                f"Initial: {initial_frozen_hash}\nFinal:   {final_frozen_hash}"
            )
        report["frozen_base_verification"]["hash_match"] = True

        # 5. Anti-leakage verification
        # Verify features() does not read or depend on future keys / future images
        sample_clean = copy.deepcopy(sample)
        # Create altered future images sample
        sample_altered = copy.deepcopy(sample)
        if "future_images" in sample_altered and len(sample_altered["future_images"]) > 0:
            # Replace future images with solid color / inverted images of same resolution
            from PIL import Image
            sample_altered["future_images"] = [
                [Image.new("RGB", (img[0].width, img[0].height), color=(128, 64, 32))]
                for img in sample_altered["future_images"]
            ]

        with torch.no_grad():
            deep_clean, shallow_clean = model.features(sample_clean)
            deep_alt, shallow_alt = model.features(sample_altered)

        if deep_clean.shape != deep_alt.shape or shallow_clean.shape != shallow_alt.shape:
            raise AssertionError("Features shape mismatch when altering future_images")

        # Must be exact BF16 equal or exact float32 equal
        if not torch.equal(deep_clean, deep_alt):
            diff = (deep_clean - deep_alt).abs().max().item()
            raise AssertionError(f"Information leakage detected! features deep changed with future images: diff={diff}")
        if not torch.equal(shallow_clean, shallow_alt):
            diff = (shallow_clean - shallow_alt).abs().max().item()
            raise AssertionError(f"Information leakage detected! features shallow changed with future images: diff={diff}")

        report["leakage_verification"]["features_future_invariant"] = True
        report["leakage_verification"]["deep_shape"] = list(deep_clean.shape)
        report["leakage_verification"]["shallow_shape"] = list(shallow_clean.shape)

        # 6. Predict actions verification
        # Remove all future keys and ensure predict_actions succeeds and produces shape [1, 50, 24]
        sample_no_future = copy.deepcopy(sample)
        for k in ["future_images", "future_indices", "future_deltas", "future_valid", "future_frame_ids"]:
            sample_no_future.pop(k, None)

        with torch.no_grad():
            actions_pred = model.predict_actions(sample_no_future)

        if actions_pred.ndim != 3 or actions_pred.shape[0] != 1 or actions_pred.shape[1] != 50 or actions_pred.shape[2] != 24:
            raise AssertionError(f"predict_actions output shape expected (1, 50, 24), got {actions_pred.shape}")
        if not torch.isfinite(actions_pred).all():
            raise AssertionError("predict_actions output contains non-finite values")

        report["action_inference"]["shape"] = list(actions_pred.shape)
        report["action_inference"]["finite"] = True
        report["action_inference"]["future_keys_stripped_success"] = True

        report["status"] = "PASSED"
        write_report(out_dir, report)
        print("verify_predictive completed successfully: status PASSED", flush=True)

    except Exception as exc:
        report["status"] = "FAILED"
        report["exception"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_report(out_dir, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native GPU probe verifying frozen base policy, CausalMemoryWriter, and FutureLatentHead."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to parent checkpoint .pt file")
    parser.add_argument("--output-dir", required=True, help="Path to target empty output directory")
    parser.add_argument("--device", default="cuda:0", help="Target CUDA device (default: cuda:0)")
    parser.add_argument("--epoch", type=int, default=5, help="Dataset epoch (default: 5)")
    parser.add_argument("--fabri-root", default="/root/FabriVLA", help="Path to FabriVLA repo")
    parser.add_argument("--vlm-path", default="/root/models/InternVL3_5-1B", help="Path to InternVL model")
    parser.add_argument("--data-root", default="/root/evo1_metaworld_dataset", help="Path to dataset root")
    args = parser.parse_args()

    run_verification(args)


if __name__ == "__main__":
    main()
