"""Native causal segment training dataset for FabriVLA / Moss native full-model fine-tuning.

Key characteristics:
- Context mode: 'native_causal_segments'.
- Partitions each episode into causal segments of up to target_frames (M <= 8).
- Observation context: context_start = max(0, end - history_frames) through end - 1 (up to 16 frames).
- Supervised rows: start .. end - 1 (M targets).
- Causal language attention ensures no future evidence is available to earlier targets.
- Every original row is supervised exactly once per epoch (all anchor coverage including final tails).
- Opt-in decision_stride replays every observation with per-phase compact memory groups.
- Single sequential video decode per episode with bounded raw frame LRU cache (capacity=2).
- Deterministic per-episode augmentation (same across frames, refreshed per epoch).
- Strict timestamp validation and metadata fps fallback (recording time_source).
- Reuses MetaWorldEpisodes metadata / split / dataframe parsing APIs from delta_data.py.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch

from fabri_moss.data import normalize_and_mask
from fabri_moss.delta_data import (
    MetaWorldEpisodes,
    apply_episode_augmentation,
    decode_all_video_frames,
    sample_episode_augmentation_params,
)
from fabri_moss.runtime import compute_file_sha256
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.memory_protocol import (
    compute_memory_replay_layout,
    get_memory_protocol_contract,
)
from fabri_moss.stream_protocol import (
    compute_stream_replay_v1_layout,
    get_stream_protocol_contract,
)


def _decision_cadence_contract(decision_stride: int) -> Dict[str, Any]:
    return {
        "protocol_name": "deployment_cadence_replay_v1",
        "decision_stride": decision_stride,
        "observation_sampling": "all_episode_rows_from_zero_through_each_groups_last_target",
        "target_coverage": "all_contiguous_target_rows_exactly_once",
        "grouping": "target_row_modulo_decision_stride",
        "decision_indices": "per_group_shared_pool_indices_zero_union_phase_plus_k_times_stride",
        "phase_augmentation": "phase_zero_matches_deployment_other_phases_preserve_expert_target_coverage",
        "bootstrap": "row_zero_always_decides_nonzero_phases_have_one_short_initial_interval",
        "memory_replay": True,
    }


class BoundedFrameCache:
    """Bounded LRU cache for raw decoded video frames per worker."""

    def __init__(self, capacity: int = 2):
        self.capacity = max(1, capacity)
        self.cache: OrderedDict[int, Dict[int, Image.Image]] = OrderedDict()

    def get(self, episode_id: int) -> Optional[Dict[int, Image.Image]]:
        if episode_id in self.cache:
            self.cache.move_to_end(episode_id)
            return self.cache[episode_id]
        return None

    def put(self, episode_id: int, frames: Dict[int, Image.Image]) -> None:
        if episode_id in self.cache:
            self.cache.move_to_end(episode_id)
        self.cache[episode_id] = frames
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()


class NativeTrainingDataset(MetaWorldEpisodes):
    """Dataset delivering causal segments for native full-model fine-tuning.

    Subclasses MetaWorldEpisodes to reuse metadata loading, stratified splitting,
    and parquet/video locating utilities.
    decision_stride emits grouped compact replay for PredictiveMemoryPolicy;
    the legacy CompactNativePolicy consumer does not support those groups.
    """

    def __init__(
        self,
        root: Union[str, Path],
        norm_stats: Dict[str, Any],
        history_frames: int = 16,
        target_frames: int = 8,
        horizon: int = 50,
        state_dim: int = 24,
        action_dim: int = 24,
        split: str = "train",
        seed: int = 4042,
        val_fraction: float = 0.1,
        max_episodes: Optional[int] = None,
        augmentation: bool = True,
        stream_protocol: Optional[str] = None,
        decision_stride: Optional[int] = None,
    ):
        if stream_protocol is not None and stream_protocol not in (
            "stream_replay_v1",
            "memory_replay_v1",
            "compact_memory_replay_v1",
        ):
            raise ValueError(
                f"Unsupported stream_protocol {stream_protocol!r}, must be None, 'stream_replay_v1', 'memory_replay_v1', or 'compact_memory_replay_v1'"
            )
        self.stream_protocol = stream_protocol
        if decision_stride is not None:
            if type(decision_stride) is not int or decision_stride <= 0:
                raise ValueError("decision_stride must be a positive integer or None")
            if stream_protocol != "compact_memory_replay_v1":
                raise ValueError("decision_stride requires stream_protocol='compact_memory_replay_v1'")
        self.decision_stride = decision_stride

        if type(history_frames) is not int or history_frames <= 0:
            raise ValueError(f"history_frames must be a positive integer, got {history_frames}")
        if type(target_frames) is not int or target_frames <= 0:
            raise ValueError(f"target_frames must be a positive integer, got {target_frames}")
        if target_frames > history_frames:
            raise ValueError(
                f"target_frames ({target_frames}) must be <= history_frames ({history_frames})"
            )

        # Validation split must never have data augmentation enabled
        eff_augmentation = False if split == "val" else bool(augmentation)

        # Call MetaWorldEpisodes init with max_chunk_size=1 (metadata only, no chunks needed from parent)
        super().__init__(
            root=root,
            norm_stats=norm_stats,
            horizon=horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            max_chunk_size=1,
            split=split,
            seed=seed,
            val_fraction=val_fraction,
            max_episodes=max_episodes,
            augmentation=eff_augmentation,
        )

        self.history_frames = history_frames
        self.target_frames = target_frames

        # Create deterministic segment index tuples (ep_idx, start_row, end_row)
        # Partition every episode rows [0, L) in chunks of target_frames
        # All every-row anchor coverage including final tails, no dropped/repeated supervision
        self.segments: List[Tuple[int, int, int]] = []
        self._episode_by_id: Dict[int, Dict[str, Any]] = {}
        total_target_count = 0

        for ep in self.active_episodes:
            ep_idx = int(ep["episode_index"])
            self._episode_by_id[ep_idx] = ep
            ep_len = int(ep["length"])
            if ep_len <= 0:
                raise ValueError(f"Episode {ep_idx} has invalid length {ep_len}")

            start = 0
            while start < ep_len:
                end = min(start + self.target_frames, ep_len)
                self.segments.append((ep_idx, start, end))
                total_target_count += (end - start)
                start = end

        self._total_targets = total_target_count

        # Bounded raw frame LRU cache per worker (capacity 2)
        # Note: raw frames only (un-augmented), never caches learned features or epoch augmentations
        self.raw_frame_cache = BoundedFrameCache(capacity=2)

    def _partition_episodes(self) -> List[Dict[str, Any]]:
        episode_ids = [int(ep["episode_index"]) for ep in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("Duplicate episode_index in episode metadata")
        return super()._partition_episodes()

    @property
    def total_targets(self) -> int:
        """Total number of supervised target anchor rows across the active split."""
        return self._total_targets

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for deterministic epoch-dependent augmentation."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.segments)

    def _validate_and_get_timestamps(
        self, df: Any, ep_idx: int
    ) -> Tuple[List[float], str]:
        """Validate entire episode dataframe timestamps and return timestamps list and source."""
        has_timestamp_col = "timestamp" in df.columns
        fps = self.info.get("fps")

        if not has_timestamp_col and fps is not None:
            if isinstance(fps, bool) or not np.isfinite(float(fps)) or float(fps) <= 0:
                raise ValueError(f"metadata fps must be finite and positive, got {fps}")

        if has_timestamp_col:
            time_source = "parquet_timestamp"
            ts_values = df["timestamp"].tolist()
            parsed_times: List[float] = []
            for r_idx, t_val in enumerate(ts_values):
                if (
                    t_val is None
                    or isinstance(t_val, (bool, np.bool_))
                    or not np.isfinite(float(t_val))
                    or float(t_val) < 0
                ):
                    raise ValueError(
                        f"Row {r_idx} in episode {ep_idx} has invalid timestamp: {t_val}"
                    )
                parsed_times.append(float(t_val))

            # Validate timestamps are non-decreasing across entire dataframe
            diffs = np.diff(parsed_times)
            if np.any(diffs < 0):
                raise ValueError(
                    f"Episode {ep_idx} has non-increasing parquet timestamps: {parsed_times}"
                )
            return parsed_times, time_source

        elif fps is not None:
            time_source = "metadata_fps"
            fps_float = float(fps)
            frame_indices = df["frame_index"].values
            parsed_times = [float(int(f) / fps_float) for f in frame_indices]
            return parsed_times, time_source

        else:
            raise ValueError(
                f"Episode {ep_idx} requires either parquet 'timestamp' column or valid metadata 'fps'"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.segments):
            raise IndexError(f"Index {idx} out of range [0, {len(self.segments)})")

        ep_idx, target_start_row, target_end_row = self.segments[idx]
        ep = self._episode_by_id[ep_idx]
        df = self._get_episode_dataframe(ep)
        ep_len = len(df)

        target_rows = list(range(target_start_row, target_end_row))
        target_count = len(target_rows)

        # Validate task prompt
        task_idx = int(df["task_index"].iloc[0])
        prompt = self.tasks[task_idx]
        if not prompt or not prompt.strip():
            raise ValueError(f"Empty prompt for task_index {task_idx}")

        # Extract timestamps and source
        all_timestamps, time_source = self._validate_and_get_timestamps(df, ep_idx)
        all_frame_ids = [int(f) for f in df["frame_index"].values]

        replay_groups: Optional[List[Dict[str, Any]]] = None
        stream_layout: Optional[Dict[str, Any]] = None
        memory_replay: Optional[bool] = None

        if self.decision_stride is not None:
            # Each phase gets its own memory history. Phase zero matches deployment;
            # other phases augment coverage without making adjacent targets decisions.
            if target_end_row > ep_len:
                raise ValueError(f"target_end_row {target_end_row} exceeds episode length {ep_len}")
            if any(fid < 0 for fid in all_frame_ids) or any(
                left >= right for left, right in zip(all_frame_ids, all_frame_ids[1:])
            ):
                raise ValueError("all_frame_ids must be non-negative and strictly increasing")
            obs_rows = list(range(target_end_row))
            target_indices = target_rows
            context_start_row = 0
            replay_groups = []
            for phase in sorted({row % self.decision_stride for row in target_rows}):
                positions = [i for i, row in enumerate(target_rows) if row % self.decision_stride == phase]
                last_target = target_rows[positions[-1]]
                replay_groups.append({
                    "observation_indices": list(range(last_target + 1)),
                    "target_positions": positions,
                    "decision_indices": sorted({0, *range(phase, last_target + 1, self.decision_stride)}),
                })
            stream_layout = {
                "mode": "decision_cadence",
                "decision_stride": self.decision_stride,
                "num_observations": len(obs_rows),
                "target_rows": target_rows,
            }
            memory_replay = True
        elif self.stream_protocol is None:
            # Context observation rows: context_start = max(0, end - history_frames) through end - 1
            context_start_row = max(0, target_end_row - self.history_frames)
            obs_rows = list(range(context_start_row, target_end_row))
            target_indices = [r - context_start_row for r in target_rows]
        elif self.stream_protocol == "stream_replay_v1":
            layout_result = compute_stream_replay_v1_layout(
                seed=self.seed,
                epoch=self.epoch,
                ep_idx=ep_idx,
                target_start_row=target_start_row,
                target_end_row=target_end_row,
                all_timestamps=all_timestamps,
                history_frames=self.history_frames,
                split=self.split,
            )
            obs_rows = layout_result["obs_rows"]
            replay_groups = layout_result["replay_groups"]
            stream_layout = layout_result["stream_layout"]
            obs_row_to_idx = {r: i for i, r in enumerate(obs_rows)}
            target_indices = [obs_row_to_idx[r] for r in target_rows]
            context_start_row = obs_rows[0]
        elif self.stream_protocol in ("memory_replay_v1", "compact_memory_replay_v1"):
            layout_result = compute_memory_replay_layout(
                seed=self.seed,
                epoch=self.epoch,
                ep_idx=ep_idx,
                target_start_row=target_start_row,
                target_end_row=target_end_row,
                all_timestamps=all_timestamps,
                history_frames=self.history_frames,
                split=self.split,
                all_frame_ids=all_frame_ids,
            )
            obs_rows = layout_result["obs_rows"]
            replay_groups = layout_result["replay_groups"]
            stream_layout = layout_result["stream_layout"]
            obs_row_to_idx = {r: i for i, r in enumerate(obs_rows)}
            target_indices = [obs_row_to_idx[r] for r in target_rows]
            context_start_row = obs_rows[0]
            if layout_result.get("memory_replay") is True:
                memory_replay = True
        else:
            raise ValueError(f"Unsupported stream_protocol: {self.stream_protocol}")

        observation_times = [all_timestamps[r] for r in obs_rows]

        # Extract frame IDs
        obs_frame_ids = [all_frame_ids[r] for r in obs_rows]
        target_frame_ids = [all_frame_ids[r] for r in target_rows]

        # Video frames decoding with bounded LRU cache (raw frames only)
        decoded_frames = self.raw_frame_cache.get(ep_idx)
        if decoded_frames is None:
            video_path = self._locate_video_path(ep_idx)
            decoded_frames = decode_all_video_frames(video_path, all_frame_ids)
            self.raw_frame_cache.put(ep_idx, decoded_frames)

        # Sample deterministic episode-level augmentation
        aug_params = None
        if self.augmentation:
            aug_rng = random.Random(self.seed + 100003 * self.epoch + ep_idx * 7919)
            first_img = decoded_frames[all_frame_ids[0]]
            aug_params = sample_episode_augmentation_params(
                rng=aug_rng,
                img_width=first_img.width,
                img_height=first_img.height,
                p=0.5,
            )

        # Prepare images_window: list of [PIL_image] per observation
        images_window: List[List[Image.Image]] = []
        for fid in obs_frame_ids:
            raw_img = decoded_frames[fid]
            aug_img = apply_episode_augmentation(raw_img, aug_params)
            images_window.append([aug_img])

        # Extract supervised state and actions for each target row
        state_list: List[np.ndarray] = []
        state_mask_list: List[np.ndarray] = []
        actions_list: List[np.ndarray] = []
        action_mask_list: List[np.ndarray] = []

        for r in target_rows:
            row_data = df.iloc[r]
            raw_state = row_data["observation.state"]
            if raw_state is None:
                raise ValueError(f"Episode {ep_idx} row {r} has null observation.state")

            st_norm, st_mask = normalize_and_mask(
                np.asarray(raw_state, dtype=np.float32),
                min_val=self.state_mins,
                max_val=self.state_maxs,
                target_dim=self.state_dim,
            )
            state_list.append(st_norm)
            state_mask_list.append(st_mask)

            # Horizon H actions starting from r, repeating last at tail
            end_act = min(r + self.horizon, ep_len)
            act_rows = df.iloc[r:end_act]["action"].tolist()
            if len(act_rows) == 0:
                raise ValueError(f"Episode {ep_idx} row {r} has 0 action rows")

            while len(act_rows) < self.horizon:
                act_rows.append(act_rows[-1])

            act_norm, act_mask = normalize_and_mask(
                np.asarray(act_rows, dtype=np.float32),
                min_val=self.action_mins,
                max_val=self.action_maxs,
                target_dim=self.action_dim,
            )
            actions_list.append(act_norm)
            action_mask_list.append(act_mask)

        state_tensor = torch.from_numpy(np.stack(state_list, axis=0))  # [M, state_dim]
        state_mask_tensor = torch.from_numpy(np.stack(state_mask_list, axis=0))  # [M, state_dim]
        actions_tensor = torch.from_numpy(np.stack(actions_list, axis=0))  # [M, H, action_dim]
        action_mask_tensor = torch.from_numpy(np.stack(action_mask_list, axis=0))  # [M, H, action_dim]

        result = {
            "images_window": images_window,
            "frame_ids": obs_frame_ids,
            "observation_times": observation_times,
            "time_source": time_source,
            "prompt": prompt,
            "target_indices": target_indices,
            "state": state_tensor,
            "state_mask": state_mask_tensor,
            "actions": actions_tensor,
            "action_mask": action_mask_tensor,
            "episode_id": ep_idx,
            "target_frame_ids": target_frame_ids,
            "target_count": target_count,
            "context_start_row": context_start_row,
            "target_start_row": target_start_row,
            "target_end_row": target_end_row,
        }
        if replay_groups is not None:
            result["replay_groups"] = replay_groups
        if stream_layout is not None:
            result["stream_layout"] = stream_layout
        if memory_replay is True:
            result["memory_replay"] = True
            if self.decision_stride is None:
                result["decision_indices"] = layout_result["decision_indices"]
        return result

    def validation_indices(self, per_task: int = 1) -> List[int]:
        """Deterministic first selected episode(s) per task and one middle segment each.

        Allows periodic evaluation on a small, deterministic validation subset (e.g. 50 segments).
        """
        if per_task < 1:
            raise ValueError(f"per_task must be >= 1, got {per_task}")

        # Group active episodes by task name
        task_to_episodes: Dict[str, List[int]] = {}
        for ep in self.active_episodes:
            ep_idx = int(ep["episode_index"])
            task_name = ep["tasks"][0]
            task_to_episodes.setdefault(task_name, []).append(ep_idx)

        # Map episode ID to segment dataset indices
        ep_to_segment_indices: Dict[int, List[int]] = {}
        for seg_idx, (ep_idx, _, _) in enumerate(self.segments):
            ep_to_segment_indices.setdefault(ep_idx, []).append(seg_idx)

        selected_indices: List[int] = []
        for task_name in sorted(task_to_episodes.keys()):
            ep_ids = sorted(task_to_episodes[task_name])
            chosen_eps = ep_ids[:per_task]
            for ep_idx in chosen_eps:
                seg_list = ep_to_segment_indices.get(ep_idx, [])
                if seg_list:
                    # Choose middle segment: len // 2
                    mid_seg_idx = seg_list[len(seg_list) // 2]
                    selected_indices.append(mid_seg_idx)

        selected_indices.sort()
        return selected_indices

    def get_base_data_contract(self) -> Dict[str, Any]:
        """Return the legacy base contract, with cadence metadata only when enabled."""
        info_path = self.root / "meta" / "info.json"
        tasks_path = self.root / "meta" / "tasks.jsonl"
        episodes_path = self.root / "meta" / "episodes.jsonl"

        metadata_shas = {
            "info.json": compute_file_sha256(info_path),
            "tasks.jsonl": compute_file_sha256(tasks_path),
            "episodes.jsonl": compute_file_sha256(episodes_path),
        }

        # Task counts across active episodes
        task_counts: Dict[str, int] = {}
        for ep in self.active_episodes:
            t_name = ep["tasks"][0]
            task_counts[t_name] = task_counts.get(t_name, 0) + 1

        active_ids = [int(ep["episode_index"]) for ep in self.active_episodes]

        # Normalization fingerprint
        norm_summary = {
            "state_mins": [float(x) for x in self.state_mins],
            "state_maxs": [float(x) for x in self.state_maxs],
            "action_mins": [float(x) for x in self.action_mins],
            "action_maxs": [float(x) for x in self.action_maxs],
            "raw_dim": self.raw_dim,
        }
        norm_serialized = json.dumps(norm_summary, sort_keys=True)
        normalization_fingerprint = hashlib.sha256(norm_serialized.encode("utf-8")).hexdigest()

        # Overall data fingerprint
        fingerprint_raw = (
            f"{str(self.root)}:{metadata_shas}:{active_ids}:"
            f"{self.history_frames}:{self.target_frames}:{self.horizon}:"
            f"{self.state_dim}:{self.action_dim}:{normalization_fingerprint}:native_causal_segments_v1"
        )
        data_fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()

        contract = {
            "dataset_class": "NativeTrainingDataset",
            "context_mode": "native_causal_segments",
            "history_frames": self.history_frames,
            "target_frames": self.target_frames,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "seed": self.seed,
            "split": self.split,
            "val_fraction": self.val_fraction,
            "augmentation": self.augmentation,
            "max_episodes": self.max_episodes,
            "active_episode_ids": active_ids,
            "task_counts": task_counts,
            "segment_count": len(self.segments),
            "target_count": self._total_targets,
            "normalization_fingerprint": normalization_fingerprint,
            "metadata_files_sha256": metadata_shas,
            "data_fingerprint": data_fingerprint,
        }
        if self.decision_stride is not None:
            contract["decision_cadence"] = _decision_cadence_contract(self.decision_stride)
            cadence_json = json.dumps(contract["decision_cadence"], sort_keys=True)
            contract["data_fingerprint"] = hashlib.sha256(
                f"{data_fingerprint}:{cadence_json}".encode("utf-8")
            ).hexdigest()
        return contract

    def _get_compact_protocol_contract(self) -> Dict[str, Any]:
        contract = get_compact_protocol_contract()
        if self.decision_stride is not None:
            contract["base_sampling_contract"] = _decision_cadence_contract(self.decision_stride)
        return contract

    def get_data_contract(self) -> Dict[str, Any]:
        """Return JSON-serializable data contract dictionary."""
        base_contract = self.get_base_data_contract()
        if self.stream_protocol is None:
            return base_contract

        if self.stream_protocol == "stream_replay_v1":
            contract = dict(base_contract)
            protocol_contract = get_stream_protocol_contract()
            contract["stream_protocol"] = protocol_contract
            # recomputes data_fingerprint from base fingerprint + sorted JSON protocol
            sorted_protocol_json = json.dumps(protocol_contract, sort_keys=True)
            new_fingerprint_raw = f"{base_contract['data_fingerprint']}:{sorted_protocol_json}"
            contract["data_fingerprint"] = hashlib.sha256(new_fingerprint_raw.encode("utf-8")).hexdigest()
            return contract

        if self.stream_protocol == "memory_replay_v1":
            contract = dict(base_contract)
            protocol_contract = get_memory_protocol_contract()
            contract["stream_protocol"] = protocol_contract
            # recomputes data_fingerprint from base fingerprint + sorted JSON protocol
            sorted_protocol_json = json.dumps(protocol_contract, sort_keys=True)
            new_fingerprint_raw = f"{base_contract['data_fingerprint']}:{sorted_protocol_json}"
            contract["data_fingerprint"] = hashlib.sha256(new_fingerprint_raw.encode("utf-8")).hexdigest()
            return contract

        if self.stream_protocol == "compact_memory_replay_v1":
            contract = dict(base_contract)
            protocol_contract = self._get_compact_protocol_contract()
            contract["stream_protocol"] = protocol_contract
            # recomputes data_fingerprint from base fingerprint + sorted JSON protocol
            sorted_protocol_json = json.dumps(protocol_contract, sort_keys=True)
            new_fingerprint_raw = f"{base_contract['data_fingerprint']}:{sorted_protocol_json}"
            contract["data_fingerprint"] = hashlib.sha256(new_fingerprint_raw.encode("utf-8")).hexdigest()
            return contract

        raise ValueError(f"Unknown stream_protocol: {self.stream_protocol}")
