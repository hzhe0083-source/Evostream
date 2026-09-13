#!/usr/bin/env python3
"""Paired, fixed-observation action replay for the native and MOSS policies.

This is deliberately not a MuJoCo success benchmark.  It feeds the exact same
expert observations, state, prompt, and diffusion seed to:

* the original FabriVLA checkpoint;
* a MOSS Bridge adapter with only the current frame;
* the same adapter with a causal history prefix.

The output separates action error from planner timing and environment effects.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.data import MetaWorldWindows, normalize_and_mask
from fabri_moss.delta_data import decode_all_video_frames
from fabri_moss.evaluate_async import denormalize_action
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint


DEFAULT_TASKS = (
    "Lock the door by rotating the lock clockwise",
    "Pull a handle up sideways",
    "Pull a lever down 90 degrees",
    "Open a drawer",
)


def _seed(seed: int) -> None:
    """Set only the RNGs used by the flow sampler and data selection."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _timestamps(df: Any, info: Dict[str, Any]) -> Tuple[List[float], str]:
    if "timestamp" in df.columns:
        values = [float(v) for v in df["timestamp"].tolist()]
        source = "parquet_timestamp"
    elif info.get("fps") is not None:
        fps = float(info["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"metadata fps must be positive and finite, got {fps}")
        values = [int(v) / fps for v in df["frame_index"].tolist()]
        source = "metadata_fps"
    else:
        values = [float(v) for v in df["frame_index"].tolist()]
        source = "frame_index_fallback"

    # Cross-attention requires strict temporal order.  Real logs occasionally
    # contain duplicate timestamps; preserve their order with the smallest
    # representable positive increment and report the repair in provenance.
    repaired = False
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + 1e-6
            repaired = True
    return values, source + ("_monotonic_repair" if repaired else "")


def _select_episodes(dataset: MetaWorldWindows, queries: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for query in queries:
        matches = [
            ep for ep in dataset.active_episodes
            if query.casefold() in str(ep["tasks"][0]).casefold()
        ]
        if not matches:
            raise ValueError(
                f"task query {query!r} matched no episodes in split={dataset.split}"
            )
        # Pick one deterministic episode per task; a split normally contains
        # many demonstrations of the same task.
        selected[query] = min(matches, key=lambda ep: int(ep["episode_index"]))
    return selected


def _examples(
    dataset: MetaWorldWindows,
    queries: Sequence[str],
    rows: Sequence[int],
    action_dim: int,
    state_dim: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for query, episode in _select_episodes(dataset, queries).items():
        ep_id = int(episode["episode_index"])
        df = dataset._get_episode_dataframe(episode)
        times, time_source = _timestamps(df, dataset.info)
        valid_rows = [int(r) for r in rows if 0 <= int(r) < len(df)]
        if len(valid_rows) != len(rows):
            raise ValueError(f"episode {ep_id} has {len(df)} rows; requested rows={list(rows)}")
        max_row = max(valid_rows)
        frame_ids = [int(v) for v in df["frame_index"].iloc[: max_row + 1].tolist()]
        images_by_id = decode_all_video_frames(dataset._locate_video_path(ep_id), frame_ids)
        prompt = str(dataset.tasks[int(df.iloc[0]["task_index"])])
        for row in valid_rows:
            state, state_mask = normalize_and_mask(
                np.asarray(df.iloc[row]["observation.state"], dtype=np.float32),
                dataset.state_mins,
                dataset.state_maxs,
                target_dim=state_dim,
            )
            end = min(row + dataset.horizon, len(df))
            expert = np.asarray(df.iloc[row:end]["action"].tolist(), dtype=np.float32)
            if len(expert) == 0:
                raise ValueError(f"episode {ep_id} row {row} has no action label")
            out.append({
                "task": query,
                "prompt": prompt,
                "episode_id": ep_id,
                "row": row,
                "frame_ids": frame_ids,
                "observation_times": times[: max_row + 1],
                "time_source": time_source,
                "images": [[images_by_id[f]] for f in frame_ids],
                "state": torch.from_numpy(state).unsqueeze(0),
                "state_mask": torch.from_numpy(state_mask).bool().unsqueeze(0),
                "expert": expert,
                "action_dim": action_dim,
            })
    return out


def _action_mask(device: torch.device, action_dim: int, raw_dim: int = 4) -> torch.Tensor:
    return (torch.arange(action_dim, device=device).unsqueeze(0) < raw_dim)


def _record_actions(
    prediction: torch.Tensor,
    example: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(f"expected [1,H,D] prediction, got {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    n = min(5, len(example["expert"]))
    norm = prediction[0, :n, :4].detach().float().cpu().numpy()
    commands = np.stack([
        denormalize_action(v, stats, -np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        for v in norm
    ])
    expert = np.clip(example["expert"][:n, :4], -1.0, 1.0)
    return {
        "first5_normalized": norm.tolist(),
        "first5_commands": commands.tolist(),
        "expert_commands": expert.tolist(),
        "command_mae": float(np.abs(commands - expert).mean()),
    }


def _run_original(
    examples: Sequence[Dict[str, Any]],
    checkpoint: Path,
    fabri_root: str,
    vlm: str,
    device: str,
    seed: int,
    stats: Dict[str, Any],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    policy, _, _, _ = load_native_checkpoint(
        fabri_root=fabri_root, checkpoint_path=checkpoint, vlm_path=vlm,
        device=device, trainable=False,
    )
    if str(device).startswith("cuda"):
        assert_native_fa2(policy)
    policy.eval()
    dev = torch.device(device)
    action_mask = _action_mask(dev, policy.action_head.config.per_action_dim)
    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with torch.no_grad():
        for ex in examples:
            _seed(seed)
            prediction = policy.run_inference(
                images=ex["images"][ex["row"]],
                image_mask=torch.ones(1, dtype=torch.bool, device=dev),
                prompt=ex["prompt"],
                state=ex["state"].to(dev),
                state_mask=ex["state_mask"].to(dev),
                action_mask=action_mask,
            )
            results[(ex["episode_id"], ex["row"])] = _record_actions(prediction, ex, stats["action"])
    del policy
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _load_moss(
    checkpoint: Path,
    adapter: Path,
    fabri_root: str,
    vlm: str,
    device: str,
    stage: str,
    seed: int,
) -> MossInternVL:
    _seed(seed)
    policy, _, _, base_meta = load_native_checkpoint(
        fabri_root=fabri_root, checkpoint_path=checkpoint, vlm_path=vlm,
        device=device, trainable=False,
    )
    if str(device).startswith("cuda"):
        assert_native_fa2(policy)
    ckpt = torch.load(str(adapter), map_location="cpu", weights_only=False)
    if ckpt.get("format") != "moss_cross_adapter_v2":
        raise ValueError("paired replay only accepts moss_cross_adapter_v2 adapters")
    if ckpt.get("source_checkpoint_sha256") != base_meta.get("checkpoint_sha256"):
        raise ValueError("adapter/source checkpoint SHA256 mismatch")
    if ckpt.get("stage") != stage:
        raise ValueError(f"adapter stage {ckpt.get('stage')!r} != requested {stage!r}")
    config = MossConfig(**dict(ckpt["config"]))
    model = MossInternVL(policy, config=config)
    model.set_training_stage(stage)
    model.cross_blocks.load_state_dict(ckpt["cross_blocks"], strict=True)
    with torch.no_grad():
        model.readout_embeddings.copy_(ckpt["readout_embeddings"].to(model.readout_embeddings.device))
    if stage in ("expert", "joint"):
        model.policy.action_head.load_state_dict(ckpt["action_head"], strict=True)
    if stage == "joint":
        model.policy.load_state_dict(ckpt["base_policy"], strict=False)
    model.eval()
    return model


def _run_moss(
    model: MossInternVL,
    examples: Sequence[Dict[str, Any]],
    window: int,
    seed: int,
    stats: Dict[str, Any],
) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    dev = next(model.parameters()).device
    action_mask = _action_mask(dev, model.policy.action_head.config.per_action_dim)
    results: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    max_frames = model.config.max_frames
    if max_frames is not None:
        window = min(window, max_frames)
    with torch.no_grad():
        for ex in examples:
            row = ex["row"]
            for label, requested in (("window1", 1), (f"window{window}", window)):
                count = min(requested, row + 1)
                start = row + 1 - count
                imgs = ex["images"][start : row + 1]
                ids = ex["frame_ids"][start : row + 1]
                times = ex["observation_times"][start : row + 1]
                frames = [
                    model.project_frame(
                        model.encode_image(list(image_item)), int(fid), observation_time=float(ts)
                    )
                    for image_item, fid, ts in zip(imgs, ids, times)
                ]
                deep, shallow = model.read_memory(
                    frames, ex["prompt"], frame_ids=ids, observation_times=times
                )
                # Reset immediately before the stochastic flow sampler so the
                # three cases consume identical diffusion noise.
                _seed(seed)
                prediction = model.policy.action_head.sample(
                    deep,
                    state=ex["state"].to(dev),
                    state_mask=ex["state_mask"].to(dev),
                    action_mask=action_mask,
                    shallow_tokens=shallow,
                )
                rec = _record_actions(prediction, ex, stats["action"])
                rec.update({
                    "visible_frame_ids": ids,
                    "visible_frame_count": len(ids),
                    "deep_l2": float(deep.float().norm().item()),
                    "shallow_l2": float(shallow.float().norm().item()),
                })
                results[(ex["episode_id"], row, label)] = rec
    return results


def _summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    for r in records:
        grouped.setdefault(str(r["case"]), []).append(float(r["command_mae"]))
    return {k: {"n": len(v), "mean_command_mae": float(np.mean(v))} for k, v in grouped.items()}


def self_check() -> None:
    assert _action_mask(torch.device("cpu"), 6).tolist() == [[True, True, True, True, False, False]]
    print("self-check passed: fixed seed mask and replay helpers")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=Path("/root/models/FabriVLA/checkpoint_step_93000.pt"))
    p.add_argument("--adapter", type=Path, required=False)
    p.add_argument("--adapter-stage", choices=("bridge", "expert", "joint"), default="bridge")
    p.add_argument("--data-root", type=Path, required=False, default=Path("/root/evo1_metaworld_dataset"))
    p.add_argument("--output", type=Path, required=False)
    p.add_argument("--fabri-root", default="/root/FabriVLA")
    p.add_argument("--vlm", default="/root/models/InternVL3_5-1B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", choices=("val", "train", "all"), default="val")
    p.add_argument("--split-seed", type=int, default=4042)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--rows", type=int, nargs="+", default=[0, 20, 40])
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--seed", type=int, default=4048)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    if args.adapter is None or args.output is None:
        p.error("--adapter and --output are required unless --self-check is used")
    if args.window < 1 or not args.rows or min(args.rows) < 0:
        p.error("window and rows must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load the native checkpoint once to obtain its contract/statistics, then
    # release it before loading the replay model to keep one GPU footprint.
    _seed(args.seed)
    _, _, stats, base_meta = load_native_checkpoint(
        fabri_root=args.fabri_root, checkpoint_path=args.checkpoint,
        vlm_path=args.vlm, device="cpu", trainable=False,
    )
    dataset = MetaWorldWindows(
        args.data_root, stats, horizon=50, state_dim=24, action_dim=24,
        window=max(args.window, 1), frame_stride=1, split=args.split,
        seed=args.split_seed, val_fraction=args.val_fraction,
        context_mode="window",
    )
    examples = _examples(dataset, args.tasks, args.rows, action_dim=24, state_dim=24)

    original = _run_original(
        examples, args.checkpoint, args.fabri_root, args.vlm, args.device,
        args.seed, stats,
    )
    model = _load_moss(
        args.checkpoint, args.adapter, args.fabri_root, args.vlm, args.device,
        args.adapter_stage, args.seed,
    )
    moss = _run_moss(model, examples, args.window, args.seed, stats)
    gate_values = []
    for block in model.cross_blocks.values():
        gate_values.extend(block.attn_gate.detach().float().cpu().flatten().tolist())

    records: List[Dict[str, Any]] = []
    for ex in examples:
        key = (ex["episode_id"], ex["row"])
        base_rec = original[key]
        for case in ("original", "window1", f"window{min(args.window, model.config.max_frames) if model.config.max_frames is not None else args.window}"):
            if case == "original":
                rec = base_rec
            else:
                rec = moss[(ex["episode_id"], ex["row"], case)]
            records.append({
                "task": ex["task"], "episode_id": ex["episode_id"], "row": ex["row"],
                "case": case, "frame_id": ex["frame_ids"][ex["row"]],
                "time_source": ex["time_source"], **rec,
            })

        one = moss[(ex["episode_id"], ex["row"], "window1")]
        hist_case = f"window{min(args.window, model.config.max_frames) if model.config.max_frames is not None else args.window}"
        hist = moss[(ex["episode_id"], ex["row"], hist_case)]
        one_a = np.asarray(one["first5_normalized"])
        hist_a = np.asarray(hist["first5_normalized"])
        records[-1]["history_vs_current_normalized_mae"] = float(np.abs(hist_a - one_a).mean())

    payload = {
        "format": "paired_fixed_observation_replay_v1",
        "checkpoint": str(args.checkpoint),
        "adapter": str(args.adapter),
        "adapter_stage": args.adapter_stage,
        "source_checkpoint_sha256": base_meta.get("checkpoint_sha256"),
        "split": args.split, "split_seed": args.split_seed,
        "rows": args.rows, "window": args.window, "seed": args.seed,
        "num_examples": len(examples),
        "gate_abs_max": float(max(abs(v) for v in gate_values)) if gate_values else 0.0,
        "gate_tanh_abs_max": float(max(abs(np.tanh(v)) for v in gate_values)) if gate_values else 0.0,
        "summary": _summary(records),
        "records": records,
        "interpretation": "Fixed expert-state action error; no MuJoCo rollout or async planner timing.",
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
