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

from fabri_moss.core import MOSS_ARCHITECTURE_REVISION, MossConfig, MossInternVL
from fabri_moss.data import MetaWorldWindows, normalize_and_mask
from fabri_moss.delta_data import decode_all_video_frames
from fabri_moss.evaluate_async import (
    OFFICIAL_EPISODE_HORIZON,
    OFFICIAL_EXEC_HORIZON,
    OFFICIAL_FLOW_STEPS,
    OFFICIAL_IMAGE_SIZE,
    compute_file_sha256,
    denormalize_action,
    derive_plan_seed,
    describe_fa2_runtime,
    fixed_diffusion_seed,
    moss_architecture_contract,
)
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint


DEFAULT_TASKS = (
    "Lock the door by rotating the lock clockwise",
    "Pull a handle up sideways",
    "Pull a lever down 90 degrees",
    "Open a drawer",
)


def _seed(seed: int, device: str | torch.device | None = None) -> None:
    """Set only the RNGs used by the flow sampler and data selection."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        target = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(target):
            torch.cuda.manual_seed(seed)


def _plan_seed(example: Dict[str, Any], master_seed: int, enabled: bool = True) -> int | None:
    """Stable diffusion seed shared by native and every MOSS case."""
    if not enabled:
        return None
    return derive_plan_seed(
        master_seed=master_seed,
        task_slug=str(example["task"]),
        episode_index=int(example["episode_id"]),
        source_frame_id=int(example["frame_ids"][example["row"]]),
    )


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

    if any(not np.isfinite(v) for v in values):
        raise ValueError(f"{source} contains non-finite timestamps")

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
        if any(frame_ids[i] <= frame_ids[i - 1] for i in range(1, len(frame_ids))):
            raise ValueError(f"episode {ep_id} frame_index must be strictly increasing")
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
                "source_frame_id": int(frame_ids[row]),
                "source_time": float(times[row]),
                "frame_offsets": [int(f - frame_ids[row]) for f in frame_ids[: row + 1]],
                "frame_ages": [float(times[row] - t) for t in times[: row + 1]],
                # Offline replay has no queue or realtime fallback.  Keep the
                # explicit zeros so downstream reports cannot confuse this
                # diagnostic with an async rollout.
                "dropped_frame_ids": [],
                "drop_count": 0,
                "fallback": False,
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
    if isinstance(prediction, torch.Tensor) and prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(f"expected [1,H,D] prediction, got {tuple(prediction.shape)}")
    if prediction.shape[2] < 4:
        raise ValueError(f"prediction action dimension must be at least 4, got {prediction.shape[2]}")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    n = min(5, len(example["expert"]), int(prediction.shape[1]))
    if n <= 0:
        raise ValueError("prediction/expert must contain at least one action")
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


def _replay_metadata(
    example: Dict[str, Any],
    visible_frame_ids: Sequence[int],
    visible_times: Sequence[float],
    diffusion_seed: int | None,
) -> Dict[str, Any]:
    """Attach the common frame/latency/drop contract to one replay record."""
    source_id = int(example["source_frame_id"])
    source_time = float(example["source_time"])
    if len(visible_frame_ids) != len(visible_times):
        raise ValueError("visible frame ids and times must have equal length")
    result: Dict[str, Any] = {
        "source_frame_id": source_id,
        "source_time": source_time,
        "visible_frame_ids": [int(v) for v in visible_frame_ids],
        "frame_offsets": [int(v) - source_id for v in visible_frame_ids],
        "frame_ages": [source_time - float(v) for v in visible_times],
        "time_source": example["time_source"],
        "observation_age_sec": 0.0,
        "action_offset": 0,
        "offset": 0,
        "age": 0.0,
        "dropped_frame_ids": [],
        "drop_count": 0,
        "dropped": False,
        "fallback": False,
        "async_timing": False,
    }
    if diffusion_seed is not None:
        result["diffusion_seed"] = int(diffusion_seed)
        result["torch_seed"] = int(diffusion_seed)
    return result


def _run_original(
    examples: Sequence[Dict[str, Any]],
    checkpoint: Path,
    fabri_root: str,
    vlm: str,
    device: str,
    seed: int,
    stats: Dict[str, Any],
    seed_policy: str = "per-plan",
    flow_steps: int = OFFICIAL_FLOW_STEPS,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    policy, _, _, _ = load_native_checkpoint(
        fabri_root=fabri_root, checkpoint_path=checkpoint, vlm_path=vlm,
        device=device, trainable=False,
    )
    if torch.device(device).type == "cuda":
        assert_native_fa2(policy)
    policy.eval()
    if hasattr(policy.action_head, "config"):
        policy.action_head.config.num_inference_timesteps = flow_steps
    dev = torch.device(device)
    action_mask = _action_mask(dev, policy.action_head.config.per_action_dim)
    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with torch.no_grad():
        for ex in examples:
            if seed_policy not in ("per-plan", "once"):
                raise ValueError(f"Unknown seed_policy {seed_policy!r}")
            diffusion_seed = (
                _plan_seed(ex, seed, enabled=True)
                if seed_policy == "per-plan"
                else int(seed)
            )
            with fixed_diffusion_seed(diffusion_seed, dev):
                prediction = policy.run_inference(
                    images=ex["images"][ex["row"]],
                    image_mask=torch.ones(1, dtype=torch.bool, device=dev),
                    prompt=ex["prompt"],
                    state=ex["state"].to(dev),
                    state_mask=ex["state_mask"].to(dev),
                    action_mask=action_mask,
                )
            record = _record_actions(prediction, ex, stats["action"])
            record.update(_replay_metadata(ex, [ex["source_frame_id"]], [ex["source_time"]], diffusion_seed))
            results[(ex["episode_id"], ex["row"])] = record
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
    _seed(seed, device)
    policy, _, _, base_meta = load_native_checkpoint(
        fabri_root=fabri_root, checkpoint_path=checkpoint, vlm_path=vlm,
        device=device, trainable=False,
    )
    if torch.device(device).type == "cuda":
        assert_native_fa2(policy)
    ckpt = torch.load(str(adapter), map_location="cpu", weights_only=False)
    if ckpt.get("format") != "moss_cross_adapter_v2":
        raise ValueError("paired replay only accepts moss_cross_adapter_v2 adapters")
    expected_arch_revision = "moss-cross-qkvo-lora-fp32-r8-a16-d0.1-v1"
    if ckpt.get("architecture_revision") not in (None, expected_arch_revision):
        raise ValueError("MOSS adapter architecture_revision mismatch")
    adapter_source_sha = ckpt.get("source_checkpoint_sha256", ckpt.get("base_sha256"))
    if adapter_source_sha != base_meta.get("checkpoint_sha256"):
        raise ValueError("adapter/source checkpoint SHA256 mismatch")
    if ckpt.get("stage") != stage:
        raise ValueError(f"adapter stage {ckpt.get('stage')!r} != requested {stage!r}")
    config = MossConfig(**dict(ckpt["config"]))
    if getattr(config, "architecture_revision", None) not in (
        MOSS_ARCHITECTURE_REVISION,
        expected_arch_revision,
    ):
        raise ValueError("MOSS config architecture_revision is unsupported")
    model = MossInternVL(policy, config=config)
    if stage == "joint":
        from fabri_moss.lora import inject_lora, load_lora_state_dict, lora_spec

        spec = ckpt.get("lora_spec")
        if not isinstance(spec, dict) or not isinstance(ckpt.get("lora_state"), dict):
            raise ValueError("joint MOSS adapter must contain lora_spec and lora_state")
        paths = []
        for layer_idx in model.config.cross_layers:
            attention = model.native_core.layers[int(layer_idx) - 1].self_attn
            prefix = next((n for n, m in model.policy.named_modules() if m is attention), None)
            if not prefix:
                raise ValueError(f"native attention layer {layer_idx} is not reachable from policy")
            paths.extend(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        inject_lora(
            model.policy,
            paths,
            rank=int(spec.get("rank", 0)),
            alpha=float(spec.get("alpha", 0.0)),
            dropout=float(spec.get("dropout", 0.0)),
        )
        if lora_spec(model.policy) != spec:
            raise ValueError("MOSS adapter LoRA architecture mismatch")
        load_lora_state_dict(model.policy, ckpt["lora_state"], strict=True)
    model.set_training_stage(stage)
    model.cross_blocks.load_state_dict(ckpt["cross_blocks"], strict=True)
    with torch.no_grad():
        model.readout_embeddings.copy_(ckpt["readout_embeddings"].to(model.readout_embeddings.device))
    if stage in ("expert", "joint"):
        model.policy.action_head.load_state_dict(ckpt["action_head"], strict=True)
    # Joint v2 checkpoints restore frozen native weights from the source SHA
    # and LoRA tensors above; they intentionally do not duplicate base_policy.
    model._adapter_architecture_revision = ckpt.get("architecture_revision")  # type: ignore[attr-defined]
    model.eval()
    return model


def _run_moss(
    model: MossInternVL,
    examples: Sequence[Dict[str, Any]],
    window: int,
    seed: int,
    stats: Dict[str, Any],
    seed_policy: str = "per-plan",
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
                if seed_policy not in ("per-plan", "once"):
                    raise ValueError(f"Unknown seed_policy {seed_policy!r}")
                # Native, current-only, and history cases share one seed for
                # this plan so the comparison isolates visible history.
                diffusion_seed = (
                    _plan_seed(ex, seed, enabled=True)
                    if seed_policy == "per-plan"
                    else int(seed)
                )
                with fixed_diffusion_seed(diffusion_seed, dev):
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
                rec.update(_replay_metadata(ex, ids, times, diffusion_seed))
                results[(ex["episode_id"], row, label)] = rec
    return results


def _summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    for r in records:
        grouped.setdefault(str(r["case"]), []).append(float(r["command_mae"]))
    return {k: {"n": len(v), "mean_command_mae": float(np.mean(v))} for k, v in grouped.items()}


def self_check() -> None:
    assert _action_mask(torch.device("cpu"), 6).tolist() == [[True, True, True, True, False, False]]
    seed_a = derive_plan_seed(4048, "task", 0, 5)
    assert seed_a == derive_plan_seed(4048, "task", 0, 5)
    assert seed_a != derive_plan_seed(4048, "task", 0, 6)
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
    p.add_argument("--seed-policy", choices=("per-plan", "once"), default="per-plan")
    p.add_argument("--flow-steps", type=int, default=OFFICIAL_FLOW_STEPS)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    if args.adapter is None or args.output is None:
        p.error("--adapter and --output are required unless --self-check is used")
    if args.window < 1 or not args.rows or min(args.rows) < 0:
        p.error("window and rows must be positive")
    if args.flow_steps <= 0:
        p.error("--flow-steps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load the native checkpoint once to obtain its contract/statistics, then
    # release it before loading the replay model to keep one GPU footprint.
    _seed(args.seed, args.device)
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
    dataset_contract = dataset.get_data_contract() if hasattr(dataset, "get_data_contract") else {}

    original = _run_original(
        examples, args.checkpoint, args.fabri_root, args.vlm, args.device,
        args.seed, stats, seed_policy=args.seed_policy,
        flow_steps=args.flow_steps,
    )
    model = _load_moss(
        args.checkpoint, args.adapter, args.fabri_root, args.vlm, args.device,
        args.adapter_stage, args.seed,
    )
    if hasattr(model.policy.action_head, "config"):
        model.policy.action_head.config.num_inference_timesteps = args.flow_steps
    moss = _run_moss(model, examples, args.window, args.seed, stats, seed_policy=args.seed_policy)
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
            records[-1]["source_checkpoint_sha256"] = base_meta.get("checkpoint_sha256")
            records[-1]["source_sha256"] = base_meta.get("checkpoint_sha256")

        one = moss[(ex["episode_id"], ex["row"], "window1")]
        hist_case = f"window{min(args.window, model.config.max_frames) if model.config.max_frames is not None else args.window}"
        hist = moss[(ex["episode_id"], ex["row"], hist_case)]
        one_a = np.asarray(one["first5_normalized"])
        hist_a = np.asarray(hist["first5_normalized"])
        records[-1]["history_vs_current_normalized_mae"] = float(np.abs(hist_a - one_a).mean())

    adapter_sha = compute_file_sha256(args.adapter)
    for record in records:
        record["adapter_sha256"] = adapter_sha
        record["adapter_sha"] = adapter_sha
        record["architecture_contract"] = "fabri_vla_moss_cross_v2"
        record["data_contract"] = "paired_fixed_observation_replay_v1"
    source_hashes = {
        "paired_moss_replay.py": compute_file_sha256(Path(__file__)),
        "evaluate_async.py": compute_file_sha256(ROOT / "fabri_moss" / "evaluate_async.py"),
    }
    architecture_contract = moss_architecture_contract(
        memory_mode=getattr(model.config, "memory_mode", "consume"),
        cross_layers=getattr(model.config, "cross_layers", None),
        max_frames=getattr(model.config, "max_frames", args.window),
        architecture_revision=getattr(model.config, "architecture_revision", None),
    )
    try:
        from fabri_moss.lora import lora_spec
        architecture_contract["lora"] = lora_spec(model.policy)
    except Exception:
        architecture_contract["lora"] = None
    architecture_contract.update(
        {
            "name": "fabri_vla_moss_cross_v2",
            "adapter_format": "moss_cross_adapter_v2",
            "window": int(args.window),
            "core_architecture_revision": getattr(model.config, "architecture_revision", None),
        }
    )
    data_contract = {
        "name": "paired_fixed_observation_replay_v1",
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": [int(v) for v in args.rows],
        "image_size": OFFICIAL_IMAGE_SIZE,
        "image_preprocess": "dataset_video_frame_same_source; no_env_render",
        "image_source": "official_metaworld_dataset_video",
        "state_dim": 24,
        "action_dim": 24,
        "time_source": "parquet_timestamp_or_metadata_fps_or_frame_index_fallback",
        "action_horizon": 50,
        "official_episode_horizon": OFFICIAL_EPISODE_HORIZON,
        "official_exec_horizon": OFFICIAL_EXEC_HORIZON,
        "official_flow_steps": OFFICIAL_FLOW_STEPS,
        "drop_count": 0,
        "fallback": False,
    }
    data_contract["time_sources_observed"] = sorted({str(ex["time_source"]) for ex in examples})
    try:
        json.dumps(dataset_contract)
    except (TypeError, ValueError):
        dataset_contract = repr(dataset_contract)
    data_contract["dataset_contract"] = dataset_contract
    native_fa2_diag = {}
    if torch.device(args.device).type == "cuda":
        # Assert the base path explicitly; the new MOSS cross branch remains
        # SDPA and must never be reported as full-model FA2.
        native_fa2_diag = assert_native_fa2(model.policy)
    provenance = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": base_meta.get("checkpoint_sha256"),
        "source_sha256": base_meta.get("checkpoint_sha256"),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": adapter_sha,
        "adapter_sha": adapter_sha,
        "architecture_revision": getattr(model.config, "architecture_revision", None),
        "adapter_architecture_revision": getattr(model, "_adapter_architecture_revision", None),
        "architecture_contract": architecture_contract,
        "data_contract": data_contract,
        "fa2_package": describe_fa2_runtime(),
        "native_fa2_diagnostics": native_fa2_diag,
        "source_files": source_hashes,
        "seed_policy": args.seed_policy,
        "master_seed": args.seed,
        "plan_seed_contract": "SHA256([master_seed, task, episode_id, source_frame_id])[:8] mod 2**63",
        "async_success_rate": None,
        "native_sota_success_rate": None,
    }
    provenance["fa2_package"]["native_fa2_verified"] = bool(native_fa2_diag.get("native_fa2_enabled", False))

    payload = {
        "format": "paired_fixed_observation_replay_v1",
        "checkpoint": str(args.checkpoint),
        "adapter": str(args.adapter),
        "adapter_stage": args.adapter_stage,
        "source_checkpoint_sha256": base_meta.get("checkpoint_sha256"),
        "source_sha256": base_meta.get("checkpoint_sha256"),
        "adapter_sha256": adapter_sha,
        "adapter_sha": adapter_sha,
        "architecture_revision": getattr(model.config, "architecture_revision", None),
        "adapter_architecture_revision": getattr(model, "_adapter_architecture_revision", None),
        "split": args.split, "split_seed": args.split_seed,
        "rows": args.rows, "window": args.window, "seed": args.seed,
        "seed_policy": args.seed_policy,
        "flow_steps": args.flow_steps,
        "time_sources": sorted({str(ex["time_source"]) for ex in examples}),
        "drop_count": 0,
        "fallback_count": 0,
        "num_examples": len(examples),
        "gate_abs_max": float(max(abs(v) for v in gate_values)) if gate_values else 0.0,
        "gate_tanh_abs_max": float(max(abs(np.tanh(v)) for v in gate_values)) if gate_values else 0.0,
        "summary": _summary(records),
        "summary_contract": {
            "scope": "fixed_observation_replay",
            "success_rate": None,
            "async_success_rate": None,
            "native_sota_success_rate": None,
            "official_mt50_500": False,
        },
        "provenance": provenance,
        "scope": "fixed_observation_action_error",
        "official_mt50_500": False,
        "configured_full_mt50": False,
        "is_full_mt50_benchmark": False,
        "records": records,
        "interpretation": "Fixed expert-state action error; no MuJoCo rollout or async planner timing. This is not an MT50 success score or native SOTA result.",
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
