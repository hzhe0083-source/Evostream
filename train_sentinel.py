"""Training, calibration, extraction, and offline shadow replay CLI for P2 Sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sentinel import (
    DtBinThreshold,
    FixedFeatureNormalizer,
    LatentDriftMonitor,
    SentinelCalibrationProfile,
    ShallowDynamicsPredictor,
    huber_residual,
)
from sentinel_data import (
    ContextSpec,
    FeatureSpec,
    SentinelTransition,
    SentinelTransitionDataset,
    TimingContract,
    TransitionContractMetadata,
    collate_sentinel_batch,
    extract_transitions_from_recordings,
    verify_disjoint_episodes,
)

__all__ = [
    "compute_dataset_feature_stats",
    "compute_predictor_state_hash",
    "train_sentinel",
    "load_sentinel_predictor",
    "calibrate_sentinel",
    "replay_shadow_transitions",
]


def compute_dataset_feature_stats(
    dataset: SentinelTransitionDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std strictly on training dataset."""
    all_feats = torch.stack([t.start_z for t in dataset.transitions])
    mean = all_feats.mean(dim=(0, 1))
    std = all_feats.std(dim=(0, 1), unbiased=False)
    std = torch.clamp(std, min=1e-5)
    return mean, std


def compute_predictor_state_hash(
    model: ShallowDynamicsPredictor,
    config: dict[str, Any],
) -> str:
    """Compute deterministic hash over model state_dict (including buffers) and config."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    state = model.state_dict()
    for k in sorted(state.keys()):
        hasher.update(k.encode("utf-8"))
        hasher.update(state[k].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def train_sentinel(
    train_dataset: SentinelTransitionDataset,
    val_dataset: SentinelTransitionDataset | None,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    huber_delta: float = 1.0,
    device: str = "cpu",
) -> tuple[ShallowDynamicsPredictor, float | None]:
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    meta = train_dataset.metadata
    if val_dataset is not None:
        verify_disjoint_episodes(train_dataset, val_dataset)
        if train_dataset.metadata.to_dict() != val_dataset.metadata.to_dict():
            raise ValueError("Train and validation datasets have mismatched contract metadata")

    # Fit normalization strictly on train dataset
    mean, std = compute_dataset_feature_stats(train_dataset)
    normalizer = FixedFeatureNormalizer(meta.feature_spec.feature_dim, mean=mean, std=std)

    t0 = train_dataset.transitions[0]
    state_dim = t0.start_state.shape[-1]
    action_dim = t0.action_mean.shape[-1]
    context_dim = meta.context_spec.context_dim
    num_regions = meta.feature_spec.num_regions
    feature_dim = meta.feature_spec.feature_dim
    grid_h = meta.feature_spec.grid_h
    grid_w = meta.feature_spec.grid_w

    model = ShallowDynamicsPredictor(
        num_regions=num_regions,
        feature_dim=feature_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        normalizer=normalizer,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sentinel_batch,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch in loader:
            start_z = batch["start_z"].to(device)
            target_z = batch["target_z"].to(device)
            target_delta = target_z - start_z

            pred_delta = model(
                start_z=start_z,
                start_state=batch["start_state"].to(device),
                start_velocity=batch["start_velocity"].to(device),
                action_mean=batch["action_mean"].to(device),
                action_last=batch["action_last"].to(device),
                dt=batch["dt"].to(device),
                task_context=batch["task_context"].to(device),
                context_age=batch["context_age"].to(device),
            )

            loss = huber_residual(pred_delta, target_delta, delta=huber_delta, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch["episode_id"])
            count += len(batch["episode_id"])

        avg_loss = total_loss / max(1, count)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} - Train Huber Loss: {avg_loss:.6f}")

    # Compute final validation loss if val_dataset provided
    val_loss: float | None = None
    if val_dataset is not None:
        model.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sentinel_batch,
        )
        total_vloss = 0.0
        vcount = 0
        with torch.no_grad():
            for batch in val_loader:
                start_z = batch["start_z"].to(device)
                target_z = batch["target_z"].to(device)
                target_delta = target_z - start_z
                pred_delta = model(
                    start_z=start_z,
                    start_state=batch["start_state"].to(device),
                    start_velocity=batch["start_velocity"].to(device),
                    action_mean=batch["action_mean"].to(device),
                    action_last=batch["action_last"].to(device),
                    dt=batch["dt"].to(device),
                    task_context=batch["task_context"].to(device),
                    context_age=batch["context_age"].to(device),
                )
                loss = huber_residual(pred_delta, target_delta, delta=huber_delta, reduction="mean")
                total_vloss += loss.item() * len(batch["episode_id"])
                vcount += len(batch["episode_id"])
        val_loss = total_vloss / max(1, vcount)
        print(f"Final Validation Huber Loss: {val_loss:.6f}")

    config = {
        "num_regions": num_regions,
        "feature_dim": feature_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "context_dim": context_dim,
        "hidden_dim": hidden_dim,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }
    predictor_hash = compute_predictor_state_hash(model, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "metadata": meta.to_dict(),  # Exact dataset metadata PRESERVED!
        "predictor_state_hash": predictor_hash,
        "train_episodes": list({t.episode_id for t in train_dataset.transitions}),
        "val_loss": val_loss,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved trained sentinel predictor to {output_path} (predictor_state_hash={predictor_hash})")
    return model, val_loss


def load_sentinel_predictor(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[ShallowDynamicsPredictor, TransitionContractMetadata, list[str], str]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    meta = TransitionContractMetadata.from_dict(ckpt["metadata"])
    train_episodes = ckpt.get("train_episodes")
    expected_hash = ckpt.get("predictor_state_hash")
    if (
        not train_episodes
        or isinstance(train_episodes, str)
        or not isinstance(train_episodes, (list, tuple))
        or any(not isinstance(ep, str) or not ep for ep in train_episodes)
        or not expected_hash
    ):
        raise ValueError("checkpoint must bind predictor hash and non-empty list of training episodes")

    predictor = ShallowDynamicsPredictor(
        num_regions=cfg["num_regions"],
        feature_dim=cfg["feature_dim"],
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        context_dim=cfg["context_dim"],
        hidden_dim=cfg["hidden_dim"],
        grid_h=cfg.get("grid_h"),
        grid_w=cfg.get("grid_w"),
    )
    predictor.load_state_dict(ckpt["model_state"])

    # Verify predictor state hash
    computed_hash = compute_predictor_state_hash(predictor, cfg)
    if expected_hash and computed_hash != expected_hash:
        raise ValueError(f"Predictor state hash mismatch! expected {expected_hash}, computed {computed_hash}")

    predictor.to(device)
    predictor.eval()
    return predictor, meta, train_episodes, computed_hash


def calibrate_sentinel(
    checkpoint_path: Path,
    val_dataset: SentinelTransitionDataset,
    output_profile_path: Path,
    quantile: float = 0.90,
    quantile_weight: float = 0.5,
    persistence_seconds: float = 0.5,
    threshold_percentile: float = 95.0,
    device: str = "cpu",
) -> SentinelCalibrationProfile:
    predictor, model_meta, train_episodes, predictor_hash = load_sentinel_predictor(
        checkpoint_path, device=device
    )

    # Disjointness check from train episodes
    if train_episodes:
        val_eps = {t.episode_id for t in val_dataset.transitions}
        overlap = set(train_episodes).intersection(val_eps)
        if overlap:
            raise ValueError(f"Calibration dataset contains training episodes: {overlap}")

    # Strict metadata comparison (including shallow_weights_fingerprint)
    if model_meta.to_dict() != val_dataset.metadata.to_dict():
        raise ValueError(
            f"Model metadata does not match calibration dataset metadata!\n"
            f"Model: {model_meta.to_dict()}\nVal: {val_dataset.metadata.to_dict()}"
        )

    loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_sentinel_batch,
    )

    dt_groups: dict[float, list[dict[str, torch.Tensor]]] = {}
    with torch.no_grad():
        for batch in loader:
            start_z = batch["start_z"].to(device)
            target_z = batch["target_z"].to(device)
            dts = batch["dt"].view(-1).tolist()

            pred_z = predictor.predict_target(
                start_z=start_z,
                start_state=batch["start_state"].to(device),
                start_velocity=batch["start_velocity"].to(device),
                action_mean=batch["action_mean"].to(device),
                action_last=batch["action_last"].to(device),
                dt=batch["dt"].to(device),
                task_context=batch["task_context"].to(device),
                context_age=batch["context_age"].to(device),
            )

            elem_loss = huber_residual(pred_z, target_z, delta=1.0, reduction="none")
            reg_res = elem_loss.mean(dim=-1)
            q_res = torch.quantile(reg_res, q=quantile, dim=-1)
            g_res = reg_res.mean(dim=-1)

            for i, dt_val in enumerate(dts):
                dt_key = round(dt_val, 4)
                if dt_key not in dt_groups:
                    dt_groups[dt_key] = {"q": [], "g": []}
                dt_groups[dt_key]["q"].append(q_res[i].cpu())
                dt_groups[dt_key]["g"].append(g_res[i].cpu())

    sorted_dt = sorted(dt_groups.keys())
    bins: list[DtBinThreshold] = []
    # Half-open intervals [min, max) sorted with adjacent endpoints
    for idx, dt_val in enumerate(sorted_dt):
        all_q = torch.tensor(dt_groups[dt_val]["q"], dtype=torch.float32)
        all_g = torch.tensor(dt_groups[dt_val]["g"], dtype=torch.float32)

        q_thresh = max(0.0, float(torch.quantile(all_q, q=threshold_percentile / 100.0).item()))
        g_thresh = max(0.0, float(torch.quantile(all_g, q=threshold_percentile / 100.0).item()))

        prev_dt = sorted_dt[idx - 1] if idx > 0 else None
        next_dt = sorted_dt[idx + 1] if idx + 1 < len(sorted_dt) else None

        dt_min = (prev_dt + dt_val) / 2.0 if prev_dt is not None else max(1e-4, dt_val * 0.8)
        dt_max = (dt_val + next_dt) / 2.0 if next_dt is not None else dt_val * 1.2

        bins.append(
            DtBinThreshold(
                dt_min=round(dt_min, 5),
                dt_max=round(dt_max, 5),
                region_quantile_threshold=q_thresh,
                global_threshold=g_thresh,
            )
        )

    cal_episodes = tuple(sorted(set(t.episode_id for t in val_dataset.transitions)))
    if not cal_episodes:
        raise ValueError("Calibration dataset has no transitions/episodes")

    profile = SentinelCalibrationProfile(
        bins=tuple(bins),
        persistence_seconds=persistence_seconds,
        quantile=quantile,
        quantile_weight=quantile_weight,
        metadata=model_meta,
        predictor_state_hash=predictor_hash,
        calibration_episodes=cal_episodes,
    )

    output_profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_profile_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)

    print(f"Calibration complete: {len(bins)} bins -> {output_profile_path}")
    return profile


def replay_shadow_transitions(
    dataset: SentinelTransitionDataset,
    checkpoint_path: Path,
    profile_path: Path,
    output_log_path: Path,
    device: str = "cpu",
    allow_data_overlap: bool = False,
) -> list[dict[str, Any]]:
    """Offline shadow evaluation: replay recorded transitions in temporal order.

    Sorted per (episode_id, step_span, end_time). Evaluates drift and logs read-only JSONL.
    Honest offline shadow evaluation, no controller side-effects.
    """
    with open(profile_path) as f:
        profile_data = json.load(f)
    profile = SentinelCalibrationProfile.from_dict(profile_data)

    predictor, model_meta, train_episodes, predictor_hash = load_sentinel_predictor(checkpoint_path, device=device)
    if profile.predictor_state_hash != predictor_hash:
        raise ValueError("Profile predictor_state_hash does not match model checkpoint")
    if not (profile.metadata.to_dict() == dataset.metadata.to_dict() == model_meta.to_dict()):
        raise ValueError("Profile, dataset and model metadata must match")

    dataset_episodes = set(t.episode_id for t in dataset.transitions)
    train_overlap = sorted(dataset_episodes.intersection(set(train_episodes or ())))
    cal_overlap = sorted(dataset_episodes.intersection(set(profile.calibration_episodes)))

    if not allow_data_overlap:
        if train_overlap:
            raise ValueError(f"Shadow dataset contains training episodes in held-out mode: {train_overlap}")
        if cal_overlap:
            raise ValueError(f"Shadow dataset contains calibration episodes in held-out mode: {cal_overlap}")

    mode = "diagnostic" if allow_data_overlap else "held_out"

    monitor = LatentDriftMonitor(predictor, profile)

    # Sort transitions per episode_id, step_span, and true target timestamp (end_time)
    sorted_transitions = sorted(
        dataset.transitions,
        key=lambda t: (t.episode_id, t.step_span, t.end_time),
    )

    records: list[dict[str, Any]] = []
    output_log_path.parent.mkdir(parents=True, exist_ok=True)

    current_ep_span = None

    with open(output_log_path, "w") as log_f:
        for t in sorted_transitions:
            ep_span_key = (t.episode_id, t.step_span)
            if ep_span_key != current_ep_span:
                monitor.reset_monitor()
                current_ep_span = ep_span_key

            dt_val = float(t.dt.item() if isinstance(t.dt, torch.Tensor) else t.dt)
            ctx_age = float(t.context_age.item() if isinstance(t.context_age, torch.Tensor) else t.context_age)

            eval_res = monitor.evaluate_step(
                start_z=t.start_z.to(device),
                target_z=t.target_z.to(device),
                start_state=t.start_state.to(device),
                start_velocity=t.start_velocity.to(device),
                action_mean=t.action_mean.to(device),
                action_last=t.action_last.to(device),
                dt=dt_val,
                task_context=t.task_context.to(device),
                context_age=ctx_age,
                timestamp=t.end_time,
                anchor=f"span_{t.step_span}",
            )

            record = {
                "episode_id": t.episode_id,
                "step_span": t.step_span,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "dt": dt_val,
                "context_age": ctx_age,
                "mode": mode,
                "train_overlap": train_overlap,
                "calibration_overlap": cal_overlap,
                **eval_res,
            }
            records.append(record)
            log_f.write(json.dumps(record) + "\n")

    print(f"Shadow replay finished: {len(records)} steps logged to {output_log_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 Sentinel CLI: extract, train, calibrate, shadow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Extract subcommand
    ext_p = subparsers.add_parser("extract", help="Extract transitions from recordings with metadata")
    ext_p.add_argument("--recordings-path", type=Path, required=True)
    ext_p.add_argument("--output", type=Path, required=True)
    ext_p.add_argument("--spans", type=int, nargs="+", default=[1, 2, 4])

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train dynamics predictor")
    train_p.add_argument("--train-data", type=Path, required=True)
    train_p.add_argument("--val-data", type=Path)
    train_p.add_argument("--output", type=Path, required=True)
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--hidden-dim", type=int, default=256)
    train_p.add_argument("--device", type=str, default="cpu")

    # Calibrate subcommand
    cal_p = subparsers.add_parser("calibrate", help="Calibrate drift thresholds")
    cal_p.add_argument("--model-checkpoint", type=Path, required=True)
    cal_p.add_argument("--val-data", type=Path, required=True)
    cal_p.add_argument("--output-profile", type=Path, required=True)
    cal_p.add_argument("--quantile", type=float, default=0.90)
    cal_p.add_argument("--persistence-seconds", type=float, default=0.5)
    cal_p.add_argument("--percentile", type=float, default=95.0)
    cal_p.add_argument("--device", type=str, default="cpu")

    # Shadow subcommand
    shad_p = subparsers.add_parser("shadow", help="Offline shadow replay of transitions")
    shad_p.add_argument("--dataset", type=Path, required=True)
    shad_p.add_argument("--model-checkpoint", type=Path, required=True)
    shad_p.add_argument("--profile", type=Path, required=True)
    shad_p.add_argument("--output-log", type=Path, required=True)
    shad_p.add_argument(
        "--allow-data-overlap",
        action="store_true",
        help="Explicit diagnostic flag allowing evaluation on data overlapping with train/calibration",
    )
    shad_p.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.command == "extract":
        payload = torch.load(args.recordings_path, map_location="cpu", weights_only=False)
        if "metadata" not in payload or "recordings" not in payload:
            raise ValueError(
                f"Recordings file {args.recordings_path} must contain both 'metadata' and 'recordings'"
            )
        metadata = TransitionContractMetadata.from_dict(payload["metadata"])
        recordings = payload["recordings"]
        transitions = extract_transitions_from_recordings(
            recordings=recordings,
            metadata=metadata,
            spans=args.spans,
        )
        ds = SentinelTransitionDataset(transitions, metadata)
        ds.save(args.output)
        print(f"Extracted {len(transitions)} transitions -> {args.output}")

    elif args.command == "train":
        train_ds = SentinelTransitionDataset.load(args.train_data)
        val_ds = SentinelTransitionDataset.load(args.val_data) if args.val_data else None
        train_sentinel(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )

    elif args.command == "calibrate":
        val_ds = SentinelTransitionDataset.load(args.val_data)
        calibrate_sentinel(
            checkpoint_path=args.model_checkpoint,
            val_dataset=val_ds,
            output_profile_path=args.output_profile,
            quantile=args.quantile,
            persistence_seconds=args.persistence_seconds,
            threshold_percentile=args.percentile,
            device=args.device,
        )

    elif args.command == "shadow":
        ds = SentinelTransitionDataset.load(args.dataset)
        replay_shadow_transitions(
            dataset=ds,
            checkpoint_path=args.model_checkpoint,
            profile_path=args.profile,
            output_log_path=args.output_log,
            device=args.device,
            allow_data_overlap=args.allow_data_overlap,
        )


if __name__ == "__main__":
    main()
