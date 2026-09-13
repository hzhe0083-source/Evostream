import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

from fabri_moss.core import MossConfig, MossInternVL, VisionSession
from fabri_moss.data import MetaWorldWindows
from fabri_moss.runtime import load_native_checkpoint
from fabri_moss.train import compute_flow_kd_loss, train_single_step


def set_and_verify_cross_gates(student_model: MossInternVL, gate_val: float) -> None:
    for name, block in student_model.cross_blocks.items():
        if not hasattr(block, "attn_gate") or not hasattr(block, "mlp_gate"):
            raise AttributeError(f"CrossBlock {name} missing expected 'attn_gate' or 'mlp_gate' attributes!")
        with torch.no_grad():
            block.attn_gate.fill_(gate_val)
            block.mlp_gate.fill_(gate_val)
        assert abs(block.attn_gate.item() - gate_val) < 1e-6, f"Failed to set attn_gate on {name} to {gate_val}"
        assert abs(block.mlp_gate.item() - gate_val) < 1e-6, f"Failed to set mlp_gate on {name} to {gate_val}"


def compute_tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    diff = (a_f - b_f).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())
    norm_scale = float(torch.sqrt((b_f ** 2).mean()).item() + 1e-8)
    rel_rmse = float(rmse / norm_scale)
    allclose_1e3 = bool(torch.allclose(a_f, b_f, atol=1e-3, rtol=1e-3))
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "allclose_1e3": allclose_1e3,
    }


def verify_zero_gate_parity(
    student_model: MossInternVL,
    sample_images: List[Any],
    prompt: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    set_and_verify_cross_gates(student_model, 0.0)
    student_model.eval()

    with torch.no_grad():
        s_deep, s_shallow = student_model(
            images_window=[sample_images],
            frame_ids=[0],
            prompt=prompt,
        )

        tokenizer = student_model.policy.embedder.tokenizer
        lm = student_model.policy.embedder.model.language_model

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_embeds = lm.get_input_embeddings()(input_ids)

        readout_tokens = student_model.readout_embeddings.unsqueeze(0).to(
            device=device, dtype=input_embeds.dtype
        )
        combined_embeds = torch.cat([input_embeds, readout_tokens], dim=1)

        ref_out = lm(
            inputs_embeds=combined_embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        num_readout = student_model.config.num_readout_tokens
        ref_deep = ref_out.hidden_states[-1][:, -num_readout:, :].float()
        ref_shallow = ref_out.hidden_states[student_model.config.shallow_layer][:, -num_readout:, :].float()

        diff_deep = compute_tensor_diff(s_deep, ref_deep)
        diff_shallow = compute_tensor_diff(s_shallow, ref_shallow)

    passed = diff_deep["allclose_1e3"] and diff_shallow["allclose_1e3"]
    return {
        "passed": bool(passed),
        "deep_diff": diff_deep,
        "shallow_diff": diff_shallow,
    }


def verify_cache_vs_fresh_real_frames(
    student_model: MossInternVL,
    dataset: MetaWorldWindows,
    prompt: str,
) -> Dict[str, Any]:
    set_and_verify_cross_gates(student_model, 0.1)
    student_model.eval()

    if len(dataset) < 11:
        raise ValueError(f"Dataset has only {len(dataset)} samples, need >= 11 for real frame verification")

    sample_0 = dataset[0]
    sample_5 = dataset[5]
    sample_10 = dataset[10]

    ep0 = sample_0["episode_id"]
    ep5 = sample_5["episode_id"]
    ep10 = sample_10["episode_id"]
    if not (ep0 == ep5 == ep10):
        raise ValueError(f"Samples 0, 5, 10 must belong to the same episode! Got {ep0}, {ep5}, {ep10}")

    f0_img = sample_0["images_window"][-1]
    f0_id = sample_0["frame_ids"][-1]

    f1_img = sample_5["images_window"][-1]
    f1_id = sample_5["frame_ids"][-1]

    f2_img = sample_10["images_window"][-1]
    f2_id = sample_10["frame_ids"][-1]

    if f1_id <= f0_id:
        raise ValueError(f"Strict frame ordering violated: f1_id ({f1_id}) <= f0_id ({f0_id})")
    if f2_id <= f1_id:
        raise ValueError(f"Strict frame ordering violated: f2_id ({f2_id}) <= f1_id ({f1_id})")

    session = VisionSession(student_model)
    session.reset(episode_id=f"episode_{ep0:06d}", prompt=prompt)

    test_sequence = [
        (f0_img, f0_id, "step 0: single frame"),
        (f1_img, f1_id, f"step 1: two frames [{f0_id}, {f1_id}]"),
        (f2_img, f2_id, f"step 2: evicted two frames [{f1_id}, {f2_id}]"),
    ]

    accum_images = []
    accum_ids = []
    results = []

    with torch.no_grad():
        for step_i, (img_list, f_id, desc) in enumerate(test_sequence):
            session.append(img_list, f_id)
            accum_images.append(img_list)
            accum_ids.append(f_id)

            pre_cloned_kvs = []
            for f in session.frames:
                pre_cloned_kvs.append({
                    "frame_id": f.frame_id,
                    "keys": tuple(k.clone() for k in f.keys),
                    "values": tuple(v.clone() for v in f.values),
                    "num_tokens": f.num_tokens,
                })

            q_deep, q_shallow = session.query()

            kvs_unmutated = True
            for f, pre in zip(session.frames, pre_cloned_kvs):
                if f.frame_id != pre["frame_id"] or f.num_tokens != pre["num_tokens"]:
                    kvs_unmutated = False
                for k, k_pre in zip(f.keys, pre["keys"]):
                    if not torch.equal(k, k_pre):
                        kvs_unmutated = False
                for v, v_pre in zip(f.values, pre["values"]):
                    if not torch.equal(v, v_pre):
                        kvs_unmutated = False

            active_fids_in_session = [f.frame_id for f in session.frames]
            active_images = [img for img, fid in zip(accum_images, accum_ids) if fid in active_fids_in_session]

            f_deep, f_shallow = student_model(
                images_window=active_images,
                frame_ids=active_fids_in_session,
                prompt=prompt,
            )

            diff_deep = compute_tensor_diff(q_deep, f_deep)
            diff_shallow = compute_tensor_diff(q_shallow, f_shallow)

            step_pass = diff_deep["allclose_1e3"] and diff_shallow["allclose_1e3"] and kvs_unmutated
            results.append({
                "step": step_i,
                "description": desc,
                "active_frame_ids": active_fids_in_session,
                "passed": bool(step_pass),
                "kvs_unmutated_in_query": bool(kvs_unmutated),
                "deep_diff": diff_deep,
                "shallow_diff": diff_shallow,
            })

    eviction_passed = (results[2]["active_frame_ids"] == [f1_id, f2_id])
    all_passed = all(r["passed"] for r in results) and eviction_passed

    return {
        "passed": bool(all_passed),
        "eviction_verified": bool(eviction_passed),
        "steps": results,
    }


def verify_session_hooks_and_reset(
    student_model: MossInternVL,
    sample_images: List[Any],
    prompt1: str,
    prompt2: str,
) -> Dict[str, Any]:
    if prompt1 == prompt2:
        raise ValueError("prompt1 and prompt2 must be strictly distinct text for instruction switch testing")

    session = VisionSession(student_model)
    session.reset(episode_id="episode_000010", prompt=prompt1)

    vit_counts = [0]
    kv_proj_counts = [0]

    def vit_hook(m, i, o):
        vit_counts[0] += 1

    def kv_hook(m, i, o):
        kv_proj_counts[0] += 1

    embedder_model = student_model.policy.embedder.model
    v_hook_h = None
    if hasattr(embedder_model, "vision_model"):
        v_hook_h = embedder_model.vision_model.register_forward_hook(vit_hook)

    k_hooks = []
    for name, block in student_model.cross_blocks.items():
        if hasattr(block, "k_proj"):
            k_hooks.append(block.k_proj.register_forward_hook(kv_hook))
        if hasattr(block, "v_proj"):
            k_hooks.append(block.v_proj.register_forward_hook(kv_hook))

    with torch.no_grad():
        v_0 = vit_counts[0]
        k_0 = kv_proj_counts[0]

        session.append(sample_images, 0)
        v_after_append = vit_counts[0]
        k_after_append = kv_proj_counts[0]

        out1_deep, _ = session.query()
        v_after_q1 = vit_counts[0]
        k_after_q1 = kv_proj_counts[0]

        session.query()
        v_after_q2 = vit_counts[0]
        k_after_q2 = kv_proj_counts[0]

        session.reset(episode_id="episode_000011", prompt=prompt2)
        assert len(session.frames) == 0, "Session.reset failed to clear frames!"

        session.append(sample_images, 0)
        out2_deep, _ = session.query()

    if v_hook_h:
        v_hook_h.remove()
    for h in k_hooks:
        h.remove()

    append_ran_vit = (v_after_append > v_0)
    append_ran_kv = (k_after_append > k_0)
    query_ran_no_vit = (v_after_q1 == v_after_append) and (v_after_q2 == v_after_append)
    query_ran_no_kv = (k_after_q1 == k_after_append) and (k_after_q2 == k_after_append)
    prompt_switch_changed_output = not torch.allclose(out1_deep, out2_deep, atol=1e-3)

    passed = append_ran_vit and append_ran_kv and query_ran_no_vit and query_ran_no_kv and prompt_switch_changed_output

    return {
        "passed": bool(passed),
        "append_ran_vit": bool(append_ran_vit),
        "append_ran_kv_proj": bool(append_ran_kv),
        "query_ran_no_vit": bool(query_ran_no_vit),
        "query_ran_no_kv_proj": bool(query_ran_no_kv),
        "prompt_switch_changed_output": bool(prompt_switch_changed_output),
    }


def verify_synthetic_history_influence_nonzero(
    student_model: MossInternVL,
    prompt: str,
) -> Dict[str, Any]:
    from PIL import Image

    set_and_verify_cross_gates(student_model, 0.1)
    student_model.eval()

    img_black = Image.new("RGB", (448, 448), color=(0, 0, 0))
    img_white = Image.new("RGB", (448, 448), color=(255, 255, 255))

    with torch.no_grad():
        deep_b, _ = student_model([[img_black]], [0], prompt)
        deep_w, _ = student_model([[img_white]], [0], prompt)
        single_frame_diff = float((deep_b - deep_w).abs().max().item())

        deep_2b, _ = student_model([[img_black], [img_black]], [0, 5], prompt)
        deep_2w, _ = student_model([[img_white], [img_black]], [0, 5], prompt)
        history_frame_diff = float((deep_2b - deep_2w).abs().max().item())

    has_influence = (single_frame_diff > 1e-3) and (history_frame_diff > 1e-3)
    return {
        "passed": bool(has_influence),
        "single_frame_max_diff": single_frame_diff,
        "history_frame_max_diff": history_frame_diff,
        "notes": "Synthetic control: changing frame 0 alters readout when gates are 0.1",
    }


def verify_training_smoke_and_param_updates(
    student_model: MossInternVL,
    native_policy: torch.nn.Module,
    dataset: MetaWorldWindows,
    num_steps: int = 3,
    lr: float = 1e-4,
    device: str = "cpu",
) -> Dict[str, Any]:
    if num_steps < 3:
        raise ValueError(f"train_steps must be >= 3, got {num_steps}")

    set_and_verify_cross_gates(student_model, 0.0)
    student_model.set_training_stage("bridge")

    orig_params = {n: p.detach().clone() for n, p in student_model.named_parameters()}

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

    teacher_head = student_model.policy.action_head
    teacher_head.eval()

    step_losses = []
    all_grad_records = []

    for s in range(num_steps):
        batch = dataset[min(5, len(dataset) - 1)]
        metrics = train_single_step(
            student_model=student_model,
            teacher_policy=native_policy,
            teacher_head=teacher_head,
            batch=batch,
            optimizer=optimizer,
            kd_weight=1.0,
            device=device,
        )
        step_losses.append(metrics["loss"])
        all_grad_records.append(metrics.get("kv_grads", {}))

    changed_k_proj = {}
    changed_v_proj = {}
    changed_gates = {}
    changed_readout = 0.0
    frozen_native_unchanged = True

    for name, p in student_model.named_parameters():
        init_p = orig_params[name]
        diff = float((p.detach().cpu() - init_p.cpu()).abs().max().item())

        if name.startswith("policy."):
            if diff != 0.0:
                frozen_native_unchanged = False
        elif "k_proj" in name:
            changed_k_proj[name] = diff
        elif "v_proj" in name:
            changed_v_proj[name] = diff
        elif "attn_gate" in name or "mlp_gate" in name:
            changed_gates[name] = diff
        elif "readout_embeddings" in name:
            changed_readout = diff
        elif not p.requires_grad:
            if diff != 0.0:
                frozen_native_unchanged = False

    at_least_one_k_changed = any(d > 0.0 for d in changed_k_proj.values())
    at_least_one_v_changed = any(d > 0.0 for d in changed_v_proj.values())
    kv_updated = at_least_one_k_changed and at_least_one_v_changed

    gates_updated = any(d > 0.0 for d in changed_gates.values())
    readout_updated = changed_readout > 0.0

    all_grads_finite = True
    any_grad_nonzero = False
    for rec in all_grad_records:
        for g_name, g_info in rec.items():
            if not g_info["finite"]:
                all_grads_finite = False
            if g_info["norm"] > 0.0:
                any_grad_nonzero = True

    passed = (
        kv_updated
        and gates_updated
        and readout_updated
        and frozen_native_unchanged
        and all_grads_finite
        and any_grad_nonzero
        and all(step_losses)
    )

    return {
        "passed": bool(passed),
        "step_losses": step_losses,
        "kv_updated": bool(kv_updated),
        "gates_updated": bool(gates_updated),
        "readout_updated": bool(readout_updated),
        "frozen_native_unchanged": bool(frozen_native_unchanged),
        "all_grads_finite": bool(all_grads_finite),
        "any_grad_nonzero": bool(any_grad_nonzero),
        "gate_deltas": changed_gates,
        "readout_delta": changed_readout,
        "k_proj_deltas": changed_k_proj,
        "v_proj_deltas": changed_v_proj,
    }


def verify_action_sampling_smoke(
    student_model: MossInternVL,
    sample_batch: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    student_model.eval()
    head = student_model.policy.action_head

    orig_timesteps = head.config.num_inference_timesteps
    head.config.num_inference_timesteps = 1

    prompt = sample_batch["prompt"]
    images = sample_batch["images_window"]
    frame_ids = sample_batch["frame_ids"]
    state = sample_batch["state"].to(device)
    action_mask = sample_batch["action_mask"].to(device)
    raw_dim = sample_batch["raw_dim"]

    with torch.no_grad():
        deep, shallow = student_model(images, frame_ids, prompt)
        sampled_actions = head.sample(
            fused_tokens=deep,
            state=state,
            action_mask=action_mask,
            shallow_tokens=shallow,
        )

    head.config.num_inference_timesteps = orig_timesteps

    shape_ok = (sampled_actions.shape == (1, 50, 24))
    finite_ok = bool(torch.isfinite(sampled_actions).all().item())
    padded_is_zero = bool(torch.all(sampled_actions[:, :, raw_dim:] == 0).item())
    valid_nonzero = bool(torch.any(sampled_actions[:, :, :raw_dim] != 0).item())

    passed = shape_ok and finite_ok and padded_is_zero and valid_nonzero
    return {
        "passed": bool(passed),
        "action_shape": list(sampled_actions.shape),
        "is_finite": finite_ok,
        "padded_dims_zero": padded_is_zero,
        "valid_dims_active": valid_nonzero,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabriVLA MOSS Structure Verification")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to native FabriVLA repo")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Path to checkpoint .pt")
    parser.add_argument("--vlm", type=str, default="/root/models/InternVL3_5-1B", help="Path to local InternVL model")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MetaWorld LeRobot dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to write verify_report.json")
    parser.add_argument("--train-steps", type=int, default=3, help="Steps for training smoke check (>= 3)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu)")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key in norm stats")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.train_steps < 3:
        raise ValueError(f"--train-steps must be >= 3, got {args.train_steps}")

    torch.set_num_threads(args.threads)
    torch.manual_seed(4042)

    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "verify_report.json"
    if report_path.exists():
        raise FileExistsError(f"Verification report already exists at {report_path}! Refusing to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": args.device,
        "threads": args.threads,
        "all_passed": False,
        "notes": "Structural diagnostics only: synthetic control for perturbation, real frames for cache, no performance claim",
        "diagnostics": {},
    }

    try:
        native_policy, raw_config, norm_stats, base_meta = load_native_checkpoint(
            fabri_root=args.fabri_root,
            checkpoint_path=args.checkpoint,
            vlm_path=args.vlm,
            device=args.device,
            arm_key=args.arm_key,
        )
        report["base_metadata"] = base_meta

        moss_config = MossConfig(
            cross_layers=(3, 6, 10, 14),
            num_readout_tokens=16,
            max_frames=2,
            shallow_layer=6,
        )
        student_model = MossInternVL(native_policy, moss_config)
        report["moss_config"] = {
            "cross_layers": moss_config.cross_layers,
            "num_readout_tokens": moss_config.num_readout_tokens,
            "max_frames": moss_config.max_frames,
            "shallow_layer": moss_config.shallow_layer,
        }

        dataset = MetaWorldWindows(
            root=args.data_root,
            norm_stats=norm_stats,
            horizon=50,
            window=2,
            frame_stride=5,
            split="all",
            max_episodes=2,
        )

        sample = dataset[0]
        prompt = sample["prompt"]

        report["diagnostics"]["zero_gate_parity"] = verify_zero_gate_parity(
            student_model, sample["images_window"][-1], prompt, device=args.device
        )

        distinct_prompt = None
        for t_idx, t_text in dataset.tasks.items():
            if t_text != prompt:
                distinct_prompt = t_text
                break
        if distinct_prompt is None:
            distinct_prompt = prompt + " then close gripper"

        report["diagnostics"]["session_hooks_reset"] = verify_session_hooks_and_reset(
            student_model, sample["images_window"][-1], prompt, distinct_prompt
        )

        report["diagnostics"]["cache_vs_fresh_real_frames"] = verify_cache_vs_fresh_real_frames(
            student_model, dataset, prompt
        )

        report["diagnostics"]["synthetic_history_influence"] = verify_synthetic_history_influence_nonzero(
            student_model, prompt
        )

        report["diagnostics"]["action_sampling_smoke"] = verify_action_sampling_smoke(
            student_model, sample, device=args.device
        )

        report["diagnostics"]["training_smoke"] = verify_training_smoke_and_param_updates(
            student_model, native_policy, dataset, num_steps=args.train_steps, device=args.device
        )

        all_passed = all(
            d.get("passed", False) for d in report["diagnostics"].values()
        )
        report["all_passed"] = bool(all_passed)

    except Exception as e:
        traceback.print_exc()
        report["all_passed"] = False
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()

    finally:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    if not report.get("all_passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
