#!/usr/bin/env python3
"""Paired offline action diagnostics on held-out expert observations, not success rates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fabri_moss.data import normalize_and_mask
from fabri_moss.delta_data import decode_all_video_frames
from fabri_moss.evaluate_async import denormalize_action
from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.native_training import NativeSequencePolicy, _autocast_context
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint


def case_sample(prefix, case):
    sample = dict(prefix)
    if case in ("official", "current"):
        for key in ("images_window", "frame_ids", "observation_times"):
            sample[key] = prefix[key][-1:]
    elif case in ("history_env", "history_env_decision1"):
        sample["observation_times"] = [i * 0.0125 for i in range(len(prefix["frame_ids"]))]
    last = len(sample["frame_ids"]) - 1
    sample["target_indices"] = [last]
    stride = 1 if case == "history_env_decision1" else 5
    sample["decision_indices"] = sorted(set(range(0, last + 1, stride)) | {last})
    sample["memory_replay"] = True
    return sample


def action_record(prediction, expert, stats, horizon, action_dim):
    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, horizon, action_dim):
        raise ValueError(f"Expected prediction shape {(1, horizon, action_dim)}")
    if not torch.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")
    expert = np.asarray(expert, dtype=np.float32)
    if expert.ndim != 2 or expert.shape[1] != 4 or not 1 <= len(expert) <= 5 or not np.isfinite(expert).all():
        raise ValueError("Expert labels must contain one to five finite four-dimensional actions")
    normalized = prediction[0, :5].detach().float().cpu().numpy()
    commands = np.stack([denormalize_action(a, stats, -np.ones(4), np.ones(4)) for a in normalized])
    if not np.isfinite(commands).all():
        raise ValueError("Denormalized commands contain non-finite values")
    expert_commands = np.clip(expert, -1, 1)
    return {
        "first5_normalized": normalized.tolist(), "first5_commands": commands.tolist(),
        "expert_commands": expert_commands.tolist(), "valid_label_steps": len(expert),
        "first5_command_mae": float(np.abs(commands[:len(expert)] - expert_commands).mean()),
    }


def self_check():
    prefix = {"images_window": [[i] for i in range(8)], "frame_ids": list(range(8)),
              "observation_times": [i / 30 for i in range(8)]}
    current, history = case_sample(prefix, "current"), case_sample(prefix, "history_env")
    assert current["images_window"] == [[7]] and current["decision_indices"] == [0]
    assert history["decision_indices"] == [0, 5, 7] and history["target_indices"] == [7]
    assert history["observation_times"][-1] == 7 * 0.0125 and prefix["observation_times"][-1] == 7 / 30
    dense = case_sample(prefix, "history_env_decision1")
    assert dense["decision_indices"] == list(range(8)) and dense["observation_times"] == history["observation_times"]
    stats = {"min": [-2] * 4, "max": [2] * 4}
    result = action_record(torch.ones(1, 50, 24), [[3] * 4] * 3, stats, 50, 24)
    assert result["first5_command_mae"] == 0 and result["valid_label_steps"] == 3
    try:
        action_record(torch.full((1, 50, 24), float("nan")), [[0] * 4], stats, 50, 24)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite prediction was accepted")
    print("self-check passed: causal samples, decision cadence, clipping, valid tails, finite outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--checkpoint-kind", choices=("original", "compact", "predictive"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/root/evo1_metaworld_dataset"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", default=["Lock the door by rotating the lock clockwise",
                        "Pull a handle up sideways", "Pull a lever down 90 degrees", "Open a drawer"],
                        help="Case-insensitive prompt substrings, each matching exactly one task")
    parser.add_argument("--frame-rows", type=int, nargs="+", default=[0, 20, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4048])
    parser.add_argument("--cases", nargs="+", choices=("official", "current", "history_dataset", "history_env", "history_env_decision1"))
    parser.add_argument("--split-seed", type=int, default=4042)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fabri-root", default="/root/FabriVLA")
    parser.add_argument("--vlm-path", default="/root/models/InternVL3_5-1B")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.checkpoint_kind, args.checkpoint, args.output)):
        parser.error("--checkpoint-kind, --checkpoint, and --output are required")
    if min(args.frame_rows) < 0 or min(args.seeds) < 0 or max(args.seeds) >= 2**63 or not 0 < args.val_fraction < 1 or args.threads < 1:
        parser.error("Rows/seeds must be nonnegative, seeds < 2**63, and val-fraction in (0, 1)")
    cases = args.cases or (["official", "current"] if args.checkpoint_kind == "original" else
                           ["official", "current", "history_dataset", "history_env"])
    if args.checkpoint_kind == "original" and any(c.startswith("history") for c in cases):
        parser.error("Original checkpoint supports only official/current cases")
    if any(not q.strip() for q in args.tasks) or any(len(v) != len(set(v)) for v in (cases, args.frame_rows, args.seeds)):
        parser.error("Task substrings must be nonempty; cases, rows, and seeds must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        def emit(record):
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()

        device = torch.device(args.device)
        torch.set_num_threads(args.threads)
        common = dict(fabri_root=args.fabri_root, checkpoint_path=args.checkpoint,
                      vlm_path=args.vlm_path, device=str(device))
        if args.checkpoint_kind == "original":
            base, config, stats, metadata = load_native_checkpoint(**common, trainable=True)
            wrapper = NativeSequencePolicy(base, shallow_layer=base.action_head.config.shallow_layer_index,
                                           use_timestamps=False, gradient_checkpointing=False)
        else:
            if args.checkpoint_kind == "compact":
                from fabri_moss.compact_inference import load_compact_inference as loader
            else:
                from fabri_moss.predictive_inference import load_predictive_inference as loader
            wrapper, stats, metadata = loader(**common, snapshot_dir=args.output.with_suffix(".snapshots"))
            base = wrapper.policy
        wrapper.requires_grad_(False).eval()
        if device.type == "cuda":
            assert_native_fa2(base)
        cfg = base.action_head.config
        cfg.num_inference_timesteps = 50
        if cfg.horizon < 5 or cfg.per_action_dim < 4 or len(stats["action"]["min"]) != 4:
            raise ValueError("Expected MetaWorld four-dimensional actions and horizon >= 5")
        dataset = NativeTrainingDataset(args.data_root, stats, split="val", seed=args.split_seed,
                                        val_fraction=args.val_fraction, augmentation=False,
                                        horizon=cfg.horizon, state_dim=cfg.state_dim, action_dim=cfg.per_action_dim)
        selected = {}
        for episode in sorted(dataset.active_episodes, key=lambda e: int(e["episode_index"])):
            task = episode["tasks"][0]
            if any(q.casefold() in task.casefold() for q in args.tasks):
                selected.setdefault(task, episode)
        if any(sum(q.casefold() in task.casefold() for task in selected) != 1 for q in args.tasks):
            raise ValueError("Each task substring must match exactly one held-out task; use a full prompt")
        emit({"type": "provenance", "checkpoint_kind": args.checkpoint_kind, "checkpoint": metadata,
              "data_root": str(args.data_root.resolve()), "split_seed": args.split_seed,
              "val_fraction": args.val_fraction, "cases": cases, "frame_rows": args.frame_rows,
              "seeds": args.seeds, "flow_steps": 50, "device": str(device),
              "norm_stats": {k: {b: [float(x) for x in stats[k][b][:4]] for b in ("min", "max")} for k in ("action", "observation.state")},
              "selected_episodes": {task: int(ep["episode_index"]) for task, ep in selected.items()},
              "interpretation": "Expert-state offline command error; not closed-loop success. All base weights preserved FP32."})
        for task, episode in selected.items():
            episode_id = int(episode["episode_index"])
            df = dataset._get_episode_dataframe(episode)
            if max(args.frame_rows) >= len(df):
                raise ValueError(f"Requested row exceeds episode {episode_id} length {len(df)}")
            times, time_source = dataset._validate_and_get_timestamps(df, episode_id)
            frame_ids = [int(f) for f in df["frame_index"].iloc[:max(args.frame_rows) + 1]]
            images = decode_all_video_frames(dataset._locate_video_path(episode_id), frame_ids)
            prompt = dataset.tasks[int(df["task_index"].iloc[0])]
            for row in args.frame_rows:
                state, state_mask = normalize_and_mask(np.asarray(df.iloc[row]["observation.state"]),
                    stats["observation.state"]["min"], stats["observation.state"]["max"], cfg.state_dim)
                action_mask = torch.arange(cfg.per_action_dim, device=device).unsqueeze(0) < 4
                prefix = {"prompt": prompt, "images_window": [[images[f]] for f in frame_ids[:row + 1]],
                          "frame_ids": frame_ids[:row + 1], "observation_times": times[:row + 1],
                          "state": torch.tensor(state, device=device).unsqueeze(0),
                          "state_mask": torch.tensor(state_mask, device=device).bool().unsqueeze(0),
                          "action_mask": action_mask}
                expert = df.iloc[row:row + 5]["action"].tolist()
                image_hash = hashlib.sha256(images[frame_ids[row]].tobytes()).hexdigest()
                prefix_hash = hashlib.sha256(b"".join(images[f].tobytes() for f in frame_ids[:row + 1])).hexdigest()
                for seed in args.seeds:
                    for case in cases:
                        sample = case_sample(prefix, case)
                        torch.manual_seed(seed)
                        started = time.monotonic()
                        with torch.no_grad():
                            if case == "official":
                                with _autocast_context(device):
                                    prediction = base.run_inference(images=sample["images_window"][-1],
                                        image_mask=torch.ones(1, dtype=torch.bool, device=device), prompt=prompt,
                                        state=sample["state"], state_mask=sample["state_mask"], action_mask=action_mask)
                                label = "official_input_fp32_base_bf16_autocast"
                            elif args.checkpoint_kind == "original":
                                sample["memory_replay"] = False
                                deep, shallow = wrapper.features(sample)
                                with _autocast_context(device, enabled=False):
                                    prediction = base.action_head.sample(deep, state=sample["state"],
                                        state_mask=sample["state_mask"], action_mask=action_mask,
                                        shallow_tokens=shallow if cfg.shallow_fusion != "none" else None)
                                label = "native_single_no_text_fp32_head"
                            else:
                                prediction = wrapper.predict_actions(sample)
                                label = case + ("_text_timestamps" if args.checkpoint_kind == "compact" else "_textless")
                        metrics = action_record(prediction, expert, stats["action"], cfg.horizon, cfg.per_action_dim)
                        emit({"type": "case", "task": task, "prompt": prompt, "episode_id": episode_id,
                              "row": row, "frame_id": frame_ids[row], "seed": seed, "case": case,
                              "case_label": label, "frame_ids": sample["frame_ids"],
                              "observation_times": sample["observation_times"],
                              "time_source": "synthetic_row_dt_0.0125" if case.startswith("history_env") else time_source,
                              "current_image_sha256": image_hash, "source_prefix_images_sha256": prefix_hash,
                              "state_normalized": state.tolist(), "image_size": images[frame_ids[row]].size,
                              "decision_indices": sample["decision_indices"],
                              "elapsed_seconds": time.monotonic() - started, **metrics})
                        print(f"{task} row={row} seed={seed} {case}: MAE={metrics['first5_command_mae']:.6f}", flush=True)
        emit({"type": "completed"})


if __name__ == "__main__":
    main()
