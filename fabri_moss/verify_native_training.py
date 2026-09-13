"""Verification probe for FabriVLA / Moss native full-model causal sequence training.

Verifies:
1. Environment & Preflight: CUDA availability, FlashAttention-2 assertion, strict checkpoint
   loading (asserts exactly 676 keys, original sha256 assertion), parameter count assertion
   (zero new parameters), all policy parameters float32 on requested CUDA device and requires_grad=True.
2. Dataset & Target Segment: MetaWorld dataset loading, selecting one segment from first active
   episode with exactly 16 frames (or explicit fail if < 16).
3. Attention Projections Hook & FA2 Tracking: Registers forward hooks on original ViT (24 layers)
   and LLM (14 layers) attention modules to verify input/output dtypes and execution calls.
4. History Gradient Flow & Causality: Forward pass through NativeSequencePolicy with slice to
   last target-only sample, verifies that earliest context visual embeddings receive non-zero,
   finite gradients, confirming loss actually backpropagates to history.
5. Parameter Grouping & Finite Gradients: Groups parameters into vision, projector, llm, head;
   verifies all parameters and gradients are finite and FP32 on the target CUDA device.
6. FP32 Parameter Update & Optimizer Precision: Runs a single probe step using AdamW(lr=1e-5, wd=0);
   verifies actual FP32 delta > 0 for representative parameters and checks all optimizer states
   (exp_avg, exp_avg_sq) are finite float32.
7. Checkpoint Save & Round-trip Load Parity: Saves CPU model state dict, reloads into fresh
   trainable=True policy on CPU and verifies exact torch.equal on all 676 keys; also checks
   reload with trainable=False (inference configuration) and verifies exact parity against
   expected per-key dtype conversions.
8. Generates comprehensive JSON report with memory metrics, execution times, sample metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from fabri_moss.runtime import (
    assert_native_fa2,
    load_native_checkpoint,
)

ORIGINAL_EXPECTED_SHA256 = "360df93088e2ed4d3632d4fd0747680296430910e71797e2e733beac1de81a6d"


def classify_parameter_group(name: str) -> str:
    """Classify policy parameter name into vision, projector, llm, or head.

    Rejects any unclassified parameter names without arbitrary fallbacks.
    """
    if "action_head" in name:
        return "head"
    elif "embedder.model.vision_model" in name:
        return "vision"
    elif "embedder.model.mlp1" in name or "embedder.model.projector" in name:
        return "projector"
    elif "embedder.model.language_model" in name:
        return "llm"
    raise ValueError(
        f"Unclassified parameter name: '{name}'. Must belong to vision_model, mlp1, language_model, or action_head."
    )


def register_attention_dtype_hooks(policy: nn.Module) -> Tuple[List[Any], Dict[str, Any]]:
    """Register forward hooks with with_kwargs=True on ViT and LLM attention projections.

    Inspects args/kwargs and output for tensor dtypes and counts invocations.
    """
    hook_stats: Dict[str, Any] = {
        "vit_attn_calls": 0,
        "vit_input_dtypes": set(),
        "vit_output_dtypes": set(),
        "llm_attn_calls": 0,
        "llm_input_dtypes": set(),
        "llm_output_dtypes": set(),
    }
    handles: List[Any] = []

    model = getattr(getattr(policy, "embedder", None), "model", None)
    if model is None:
        raise AttributeError("policy.embedder.model is required for attention hooks")

    # 1. ViT attention modules (24 layers)
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        v_encoder = getattr(vision_model, "encoder", None)
        if v_encoder is not None and hasattr(v_encoder, "layers"):
            for idx, layer in enumerate(v_encoder.layers):
                attn_mod = getattr(layer, "attn", None)
                if attn_mod is not None:
                    def make_vit_hook(l_idx: int):
                        def hook(module, args, kwargs, output):
                            hook_stats["vit_attn_calls"] += 1
                            for a in args:
                                if isinstance(a, torch.Tensor):
                                    hook_stats["vit_input_dtypes"].add(str(a.dtype))
                            if isinstance(kwargs, dict):
                                for v in kwargs.values():
                                    if isinstance(v, torch.Tensor):
                                        hook_stats["vit_input_dtypes"].add(str(v.dtype))
                            if isinstance(output, torch.Tensor):
                                hook_stats["vit_output_dtypes"].add(str(output.dtype))
                            elif isinstance(output, (tuple, list)):
                                for elem in output:
                                    if isinstance(elem, torch.Tensor):
                                        hook_stats["vit_output_dtypes"].add(str(elem.dtype))
                        return hook
                    handles.append(attn_mod.register_forward_hook(make_vit_hook(idx), with_kwargs=True))

    # 2. LLM attention modules (14 layers)
    lang_model = getattr(model, "language_model", None)
    if lang_model is not None:
        lm_core = getattr(lang_model, "model", lang_model)
        if hasattr(lm_core, "layers"):
            for idx, layer in enumerate(lm_core.layers):
                attn_mod = getattr(layer, "self_attn", None)
                if attn_mod is not None:
                    def make_llm_hook(l_idx: int):
                        def hook(module, args, kwargs, output):
                            hook_stats["llm_attn_calls"] += 1
                            for a in args:
                                if isinstance(a, torch.Tensor):
                                    hook_stats["llm_input_dtypes"].add(str(a.dtype))
                            if isinstance(kwargs, dict):
                                for v in kwargs.values():
                                    if isinstance(v, torch.Tensor):
                                        hook_stats["llm_input_dtypes"].add(str(v.dtype))
                            if isinstance(output, torch.Tensor):
                                hook_stats["llm_output_dtypes"].add(str(output.dtype))
                            elif isinstance(output, (tuple, list)):
                                for elem in output:
                                    if isinstance(elem, torch.Tensor):
                                        hook_stats["llm_output_dtypes"].add(str(elem.dtype))
                        return hook
                    handles.append(attn_mod.register_forward_hook(make_llm_hook(idx), with_kwargs=True))

    return handles, hook_stats


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute complete GPU training verification probe."""
    # Preflight device and CUDA checks BEFORE creating output directory
    device_str = args.device
    target_device = torch.device(device_str)
    if target_device.type != "cuda":
        raise ValueError(f"verify_native_training requires a CUDA device, got {device_str!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists and is non-empty ({len(entries)} items). "
                f"Probe will not overwrite existing results."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "native_training_verification_report.json"

    torch.cuda.set_device(target_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(target_device)
    t0_start = time.perf_counter()

    # Seed current device only; do not call manual_seed_all
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seed": seed,
        "device": device_str,
        "device_name": torch.cuda.get_device_name(target_device),
        "steps_completed": [],
    }

    # 1. Preflight check: load checkpoint directly on target_device with trainable=True
    print(f"[1/7] Loading native checkpoint trainable=True onto {device_str} from {args.checkpoint_path}...")
    t_load_0 = time.perf_counter()
    policy, raw_config, norm_stats, ckpt_meta = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint_path,
        vlm_path=args.vlm_path,
        device=device_str,
        arm_key=args.arm_key,
        trainable=True,
    )
    t_load_sec = time.perf_counter() - t_load_0

    # Explicitly ensure embedder device attribute is in sync with target_device
    if hasattr(policy, "embedder") and hasattr(policy.embedder, "device"):
        policy.embedder.device = device_str

    ckpt_sha = ckpt_meta.get("checkpoint_sha256", "")
    total_keys = ckpt_meta.get("total_state_dict_keys", 0)
    print(f"Loaded checkpoint with {total_keys} keys (sha256: {ckpt_sha}) in {t_load_sec:.2f}s")

    # Strict key count assertion: must be exactly 676
    if total_keys != 676:
        raise AssertionError(
            f"Checkpoint key count mismatch: expected exactly 676 keys, found {total_keys}"
        )

    # SHA256 assertion / verification
    is_original = (ckpt_sha == ORIGINAL_EXPECTED_SHA256)
    if args.expected_sha256 is not None:
        if ckpt_sha != args.expected_sha256:
            raise AssertionError(
                f"Checkpoint SHA256 mismatch! Expected {args.expected_sha256}, got {ckpt_sha}"
            )
    else:
        if not is_original:
            if args.allow_non_original_checkpoint:
                print(f"Notice: Checkpoint SHA256 ({ckpt_sha}) does not match original known hash; allowed by flag.")
            else:
                raise AssertionError(
                    f"Checkpoint SHA256 ({ckpt_sha}) does not match original expected hash "
                    f"({ORIGINAL_EXPECTED_SHA256}). Pass --allow-non-original-checkpoint or --expected-sha256 if intentional."
                )

    report["checkpoint_provenance"] = {
        "sha256": ckpt_sha,
        "is_original_weights": is_original,
        "total_state_dict_keys": total_keys,
    }

    # Assert native FlashAttention-2 configuration (ViT 24 layers, LLM 14 layers)
    print("Checking native FlashAttention-2 assertion...")
    fa2_meta = assert_native_fa2(policy)
    report["fa2_meta"] = fa2_meta

    # Parameter checks: all float32, on target CUDA device, and requires_grad=True
    total_params = 0
    trainable_params = 0
    non_fp32_params = []
    frozen_params = []
    off_device_params = []

    for name, p in policy.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            frozen_params.append(name)
        if p.dtype != torch.float32:
            non_fp32_params.append(name)
        if p.device.type != target_device.type or (target_device.index is not None and p.device.index != target_device.index):
            off_device_params.append((name, str(p.device)))

    if off_device_params:
        raise AssertionError(f"Found parameters on wrong device (expected {device_str}): {off_device_params[:5]}")
    if non_fp32_params:
        raise AssertionError(f"Found non-FP32 parameters: {non_fp32_params[:5]} (total {len(non_fp32_params)})")
    if frozen_params:
        raise AssertionError(f"Found frozen parameters (requires_grad=False): {frozen_params[:5]}")

    report["parameters"] = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "all_fp32": True,
        "all_requires_grad": True,
        "all_on_cuda": True,
    }
    report["steps_completed"].append("checkpoint_loaded_and_verified")

    # 2. Instantiate NativeSequencePolicy wrapper
    print("[2/7] Wrapping in NativeSequencePolicy...")
    try:
        from fabri_moss.native_training import NativeSequencePolicy
    except ImportError as exc:
        raise ImportError(f"Cannot import NativeSequencePolicy from fabri_moss.native_training: {exc}")

    seq_policy = NativeSequencePolicy(
        policy=policy,
        shallow_layer=args.shallow_layer,
        use_timestamps=True,
        gradient_checkpointing=True,
    )
    seq_policy.train()

    # Verify zero NEW trainable parameters in wrapper
    wrapper_params = sum(p.numel() for p in seq_policy.parameters())
    if wrapper_params != total_params:
        raise AssertionError(
            f"NativeSequencePolicy added new parameters! wrapper: {wrapper_params}, original: {total_params}"
        )
    report["steps_completed"].append("wrapper_verified_zero_new_params")

    # 3. Load Dataset & Select 16-frame segment
    print("[3/7] Loading NativeTrainingDataset and finding 16-frame segment...")
    try:
        from fabri_moss.native_data import NativeTrainingDataset
    except ImportError as exc:
        raise ImportError(f"Cannot import NativeTrainingDataset from fabri_moss.native_data: {exc}")

    dataset = NativeTrainingDataset(
        root=args.data_root,
        norm_stats=norm_stats,
        history_frames=16,
        target_frames=8,
        split="train",
        max_episodes=args.max_episodes,
        augmentation=False,
    )

    chosen_sample: Optional[Dict[str, Any]] = None
    chosen_seg_idx = -1
    for seg_idx in range(len(dataset)):
        sample_meta = dataset.segments[seg_idx]
        ep_idx, start_r, end_r = sample_meta
        context_start_r = max(0, end_r - 16)
        if (end_r - context_start_r) == 16:
            chosen_seg_idx = seg_idx
            chosen_sample = dataset[seg_idx]
            break

    if chosen_sample is None:
        raise RuntimeError(
            f"Could not find any segment in dataset with exactly 16 frames (history_frames=16). "
            f"Total segments searched: {len(dataset)}."
        )

    obs_len = len(chosen_sample["images_window"])
    target_count = len(chosen_sample["target_indices"])
    print(
        f"Selected segment index {chosen_seg_idx}: episode {chosen_sample['episode_id']}, "
        f"obs_len={obs_len}, target_count={target_count}, "
        f"frame_ids={chosen_sample['frame_ids']}"
    )

    report["sample_info"] = {
        "segment_index": chosen_seg_idx,
        "episode_id": chosen_sample["episode_id"],
        "obs_frames_count": obs_len,
        "target_count": target_count,
        "frame_ids": chosen_sample["frame_ids"],
        "observation_times": chosen_sample["observation_times"],
        "prompt": chosen_sample["prompt"],
        "target_indices": chosen_sample["target_indices"],
    }
    report["steps_completed"].append("segment_selected_16_frames")

    # Move tensor fields of sample to target device
    sample_gpu: Dict[str, Any] = {}
    for k, v in chosen_sample.items():
        if isinstance(v, torch.Tensor):
            sample_gpu[k] = v.to(target_device)
        else:
            sample_gpu[k] = v

    # 4. Attention Hooks Setup & Forward Check
    print("[4/7] Attaching hooks to ViT and LLM attention projections...")
    handles, hook_stats = register_attention_dtype_hooks(policy)

    # Slice to strictly last target-only sample to verify causal history backprop
    last_target_sample = dict(sample_gpu)
    last_target_sample["target_indices"] = [obs_len - 1]
    last_target_sample["target_count"] = 1
    if "target_frame_ids" in sample_gpu and sample_gpu["target_frame_ids"]:
        last_target_sample["target_frame_ids"] = [sample_gpu["target_frame_ids"][-1]]
    last_target_sample["state"] = sample_gpu["state"][-1:] if sample_gpu.get("state") is not None else None
    last_target_sample["state_mask"] = sample_gpu["state_mask"][-1:] if sample_gpu.get("state_mask") is not None else None
    last_target_sample["actions"] = sample_gpu["actions"][-1:]
    last_target_sample["action_mask"] = sample_gpu["action_mask"][-1:] if sample_gpu.get("action_mask") is not None else None

    embedder_model = getattr(getattr(policy, "embedder", None), "model", None)
    if embedder_model is None:
        raise AttributeError("policy.embedder.model is required")

    orig_extract_feature = embedder_model.extract_feature
    captured_vit_tensors: List[torch.Tensor] = []

    def wrapped_extract_feature(*args, **kwargs):
        out = orig_extract_feature(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            out.retain_grad()
            captured_vit_tensors.append(out)
        return out

    embedder_model.extract_feature = wrapped_extract_feature

    try:
        # Forward pass with proper CUDA synchronization
        print("Executing forward pass on last-target-only sample...")
        torch.cuda.synchronize(target_device)
        t_fwd_0 = time.perf_counter()
        out_dict = seq_policy.forward(last_target_sample)
        torch.cuda.synchronize(target_device)
        t_fwd_sec = time.perf_counter() - t_fwd_0

        loss = out_dict["loss"]
        if not torch.isfinite(loss):
            raise ValueError(f"Forward pass produced non-finite loss: {loss.item()}")
        print(f"Forward completed in {t_fwd_sec:.3f}s. Loss: {loss.item():.6f}")

        # Backward pass with proper CUDA synchronization
        print("[5/7] Executing backward pass...")
        torch.cuda.synchronize(target_device)
        t_bwd_0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(target_device)
        t_bwd_sec = time.perf_counter() - t_bwd_0
        print(f"Backward completed in {t_bwd_sec:.3f}s")
    finally:
        # Always remove attention hooks and restore original extract_feature method
        for h in handles:
            h.remove()
        embedder_model.extract_feature = orig_extract_feature

    # Check that ViT (24 layers) and LLM (14 layers) attention hooks fired
    report["forward_hooks"] = {
        "vit_attn_calls": hook_stats["vit_attn_calls"],
        "vit_input_dtypes": sorted(list(hook_stats["vit_input_dtypes"])),
        "vit_output_dtypes": sorted(list(hook_stats["vit_output_dtypes"])),
        "llm_attn_calls": hook_stats["llm_attn_calls"],
        "llm_input_dtypes": sorted(list(hook_stats["llm_input_dtypes"])),
        "llm_output_dtypes": sorted(list(hook_stats["llm_output_dtypes"])),
    }
    if hook_stats["vit_attn_calls"] < 24:
        raise AssertionError(
            f"ViT attention hook calls ({hook_stats['vit_attn_calls']}) < expected 24 layers!"
        )
    if hook_stats["llm_attn_calls"] < 14:
        raise AssertionError(
            f"LLM attention hook calls ({hook_stats['llm_attn_calls']}) < expected 14 layers!"
        )

    # Verify earliest context visual embedding has non-zero, finite gradient
    if not captured_vit_tensors or captured_vit_tensors[0].grad is None:
        raise AssertionError("Failed to capture grad on visual embeddings!")
    vit_embeds_grad = captured_vit_tensors[0].grad
    # vit_embeds has batch dim of N=16 tiles: index 0 corresponds to the earliest context frame
    first_frame_grad = vit_embeds_grad[0]
    first_frame_grad_norm = float(first_frame_grad.norm().item())
    first_frame_grad_finite = bool(torch.isfinite(first_frame_grad).all().item())
    first_frame_nonzero = bool(first_frame_grad_norm > 0.0)

    print(f"Earliest context frame (t=0) visual grad norm: {first_frame_grad_norm:.6e} (finite={first_frame_grad_finite})")
    if not first_frame_grad_finite or not first_frame_nonzero:
        raise AssertionError(
            f"Earliest visual embedding has zero or non-finite gradient from last-target-only loss! "
            f"norm={first_frame_grad_norm}, finite={first_frame_grad_finite}. Loss is not backpropagating to history!"
        )

    report["causal_history_gradient"] = {
        "first_frame_grad_norm": first_frame_grad_norm,
        "first_frame_grad_finite": first_frame_grad_finite,
        "first_frame_grad_nonzero": first_frame_nonzero,
    }
    report["steps_completed"].append("causal_history_gradient_verified")

    # Check parameter group gradients: vision, projector, llm, head
    group_stats: Dict[str, Dict[str, Any]] = {
        "vision": {"total": 0, "has_grad": 0, "norms": []},
        "projector": {"total": 0, "has_grad": 0, "norms": []},
        "llm": {"total": 0, "has_grad": 0, "norms": []},
        "head": {"total": 0, "has_grad": 0, "norms": []},
    }

    rep_params: Dict[str, Tuple[str, nn.Parameter, torch.Tensor]] = {}

    for name, p in policy.named_parameters():
        grp = classify_parameter_group(name)
        group_stats[grp]["total"] += 1
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise AssertionError(f"Parameter {name} in group {grp} has non-finite gradients!")
            if p.grad.dtype != torch.float32:
                raise AssertionError(f"Parameter {name} grad is not float32: {p.grad.dtype}")
            if p.grad.device.type != target_device.type or (target_device.index is not None and p.grad.device.index != target_device.index):
                raise AssertionError(f"Parameter {name} grad is on {p.grad.device}, expected {target_device}")

            g_norm = float(p.grad.norm().item())
            group_stats[grp]["has_grad"] += 1
            group_stats[grp]["norms"].append(g_norm)
            if grp not in rep_params and g_norm > 0.0:
                rep_params[grp] = (name, p, p.detach().clone())

    report["parameter_groups"] = {}
    for grp, st in group_stats.items():
        norms = st["norms"]
        mean_norm = float(sum(norms) / len(norms)) if norms else 0.0
        max_norm = float(max(norms)) if norms else 0.0
        min_norm = float(min(norms)) if norms else 0.0
        report["parameter_groups"][grp] = {
            "total_params": st["total"],
            "params_with_grad": st["has_grad"],
            "mean_grad_norm": mean_norm,
            "max_grad_norm": max_norm,
            "min_grad_norm": min_norm,
            "all_finite": True,
            "has_nonzero": (max_norm > 0.0),
        }
        print(f"Group '{grp}': {st['has_grad']}/{st['total']} params with grad, max_norm={max_norm:.6e}")
        if st["has_grad"] == 0 or max_norm <= 0.0:
            raise AssertionError(f"Parameter group '{grp}' has no gradients or zero grad norm!")

    report["steps_completed"].append("parameter_group_gradients_verified")

    # 6. AdamW Optimizer Step & FP32 Update Assertion
    print("[6/7] Running probe AdamW step (lr=1e-5, wd=0.0) and asserting FP32 weight delta...")
    if len(rep_params) < 4:
        raise AssertionError(f"Missing representative parameters for all 4 groups! Found {list(rep_params.keys())}")

    optimizer = AdamW(policy.parameters(), lr=1e-5, weight_decay=0.0)
    optimizer.step()

    # Check ALL optimizer states (exp_avg, exp_avg_sq) are float32, finite, and on target device
    for p in policy.parameters():
        if p.grad is not None:
            st = optimizer.state.get(p)
            if st is None:
                raise AssertionError("Optimizer state missing for parameter with grad!")
            exp_avg = st.get("exp_avg")
            exp_avg_sq = st.get("exp_avg_sq")
            if exp_avg is None or exp_avg_sq is None:
                raise AssertionError("exp_avg or exp_avg_sq missing from optimizer state!")
            if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                raise AssertionError(f"Optimizer states are not float32: exp_avg={exp_avg.dtype}, exp_avg_sq={exp_avg_sq.dtype}")
            if not torch.isfinite(exp_avg).all() or not torch.isfinite(exp_avg_sq).all():
                raise AssertionError("Non-finite values detected in optimizer state exp_avg or exp_avg_sq!")
            if exp_avg.device.type != target_device.type or exp_avg_sq.device.type != target_device.type:
                raise AssertionError(f"Optimizer state on wrong device: exp_avg on {exp_avg.device}")

    # Check 4 representative parameter updates: finite and max_delta > 0
    update_stats: Dict[str, Dict[str, Any]] = {}
    for grp in ("vision", "projector", "llm", "head"):
        p_name, param, old_val = rep_params[grp]
        delta = (param.detach() - old_val).abs()
        if not torch.isfinite(delta).all():
            raise AssertionError(f"Parameter {p_name} update delta contains non-finite values!")
        max_delta = float(delta.max().item())
        mean_delta = float(delta.mean().item())
        if max_delta <= 0.0 or not math.isfinite(max_delta):
            raise AssertionError(f"Parameter {p_name} in group '{grp}' did not update! max_delta={max_delta}")

        update_stats[grp] = {
            "parameter_name": p_name,
            "max_fp32_delta": max_delta,
            "mean_fp32_delta": mean_delta,
            "optimizer_state_dtype": "torch.float32",
        }
        print(f"Group '{grp}' ({p_name}): max delta={max_delta:.6e}")

    report["fp32_updates"] = update_stats
    report["steps_completed"].append("optimizer_step_fp32_delta_verified")

    # 7. Checkpoint Save & Round-trip Parity
    print("[7/7] Verifying checkpoint save and reload parity...")
    ckpt_save_path = output_dir / "probe_updated_checkpoint.pt"

    # Save only CPU model state dict and required configs (do not save optimizer in probe)
    saved_model_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    for k, v in saved_model_state.items():
        if not torch.isfinite(v).all():
            raise AssertionError(f"Non-finite values in saved state dict key '{k}'!")

    save_dict = {
        "model": saved_model_state,
        "config": raw_config,
        "norm_stats": norm_stats,
        "step": 1,
    }
    torch.save(save_dict, ckpt_save_path)
    print(f"Saved probe checkpoint to {ckpt_save_path}")

    # Reload Check 1: Fresh reload with trainable=True on CPU (exact FP32 parity)
    print("Reload Check 1: Reloading with trainable=True on CPU and asserting exact state dict equality...")
    reloaded_train_policy, _, _, _ = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=ckpt_save_path,
        vlm_path=args.vlm_path,
        device="cpu",
        arm_key=args.arm_key,
        trainable=True,
    )
    reloaded_train_state = reloaded_train_policy.state_dict()

    if set(saved_model_state.keys()) != set(reloaded_train_state.keys()):
        raise AssertionError(
            f"State dict keys mismatch on trainable reload! "
            f"Missing: {set(saved_model_state.keys()) - set(reloaded_train_state.keys())}, "
            f"Unexpected: {set(reloaded_train_state.keys()) - set(saved_model_state.keys())}"
        )

    for k, v_saved in saved_model_state.items():
        v_reloaded = reloaded_train_state[k]
        if not torch.equal(v_saved, v_reloaded):
            max_d = float((v_saved.float() - v_reloaded.float()).abs().max().item())
            raise AssertionError(
                f"Strict trainable reload parity failed on key '{k}'! max_diff={max_d:.6e}"
            )

    del reloaded_train_policy
    del reloaded_train_state

    # Reload Check 2: Reload with trainable=False (native inference configuration)
    print("Reload Check 2: Reloading with trainable=False on CPU (verifying expected dtype conversions)...")
    reloaded_infer_policy, _, _, _ = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=ckpt_save_path,
        vlm_path=args.vlm_path,
        device="cpu",
        arm_key=args.arm_key,
        trainable=False,
    )
    reloaded_infer_state = reloaded_infer_policy.state_dict()

    max_quantization_diff = 0.0
    for k, v_saved in saved_model_state.items():
        v_infer = reloaded_infer_state[k]
        target_dtype = v_infer.dtype
        expected_v = v_saved.to(target_dtype)
        if not torch.equal(expected_v, v_infer):
            diff = float((expected_v.float() - v_infer.float()).abs().max().item())
            raise AssertionError(
                f"Inference reload mismatch against expected dtype cast on key '{k}'! diff={diff:.6e}"
            )
        q_diff = float((v_saved.float() - v_infer.float()).abs().max().item())
        if q_diff > max_quantization_diff:
            max_quantization_diff = q_diff

    del reloaded_infer_policy
    del reloaded_infer_state

    print(f"Checkpoint reload parity verified! Max quantization diff in inference mode: {max_quantization_diff:.6e}")
    report["checkpoint_reload_parity"] = {
        "saved_path": str(ckpt_save_path),
        "trainable_exact_match": True,
        "inference_expected_cast_match": True,
        "max_quantization_diff": max_quantization_diff,
    }
    report["steps_completed"].append("checkpoint_save_and_reload_verified")

    # Final timing & memory recording
    total_time_sec = time.perf_counter() - t0_start
    peak_allocated_mb = torch.cuda.max_memory_allocated(target_device) / (1024 * 1024)
    peak_reserved_mb = torch.cuda.max_memory_reserved(target_device) / (1024 * 1024)

    report["timing_and_memory"] = {
        "total_elapsed_sec": total_time_sec,
        "last_target_forward_sec": t_fwd_sec,
        "backward_sec": t_bwd_sec,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
    }

    # Only mark status as PASSED after every single assertion has passed
    report["status"] = "PASSED"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Verification completed successfully! Report saved to {report_path}")

    return report


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Verify FabriVLA / Moss native full-model causal sequence training on GPU."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save verification report and artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to run probe on (default: cuda:0).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/root/models/FabriVLA/checkpoint_step_93000.pt",
        help="Path to FabriVLA checkpoint.",
    )
    parser.add_argument(
        "--vlm-path",
        type=str,
        default="/root/models/InternVL3_5-1B",
        help="Path to native InternVL model.",
    )
    parser.add_argument(
        "--fabri-root",
        type=str,
        default="/root/FabriVLA",
        help="Path to FabriVLA codebase root.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/root/evo1_metaworld_dataset",
        help="Path to MetaWorld dataset root.",
    )
    parser.add_argument(
        "--arm-key",
        type=str,
        default="metaworld_sawyer",
        help="Key for normalization stats in checkpoint (default: metaworld_sawyer).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4042,
        help="Random seed for probe (default: 4042).",
    )
    parser.add_argument(
        "--shallow-layer",
        type=int,
        default=6,
        help="Shallow layer index for NativeSequencePolicy (default: 6).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional maximum number of episodes to load for probe (default: None).",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help=f"Explicit expected SHA256 checksum (default expects original: {ORIGINAL_EXPECTED_SHA256}).",
    )
    parser.add_argument(
        "--allow-non-original-checkpoint",
        action="store_true",
        default=False,
        help="Allow checkpoint with non-original SHA256 without raising an error.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_probe(args)
    except Exception as exc:
        print(f"Probe FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
