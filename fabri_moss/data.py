from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def compute_history_row_indices(
    current_row: int,
    window: int = 2,
    stride: int = 5,
) -> List[int]:
    if current_row < 0 or window < 1 or stride < 1:
        raise ValueError("current_row must be nonnegative; window and stride must be positive")
    raw = [max(0, current_row - i * stride) for i in range(window)]
    return sorted(list(set(raw)))


def partition_episode_consume_chunks(
    ep_len: int,
    frame_stride: int,
    window: int,
    min_context_frames: int,
    rng: random.Random,
    decision_stride: Optional[int] = None,
) -> List[List[int]]:
    if ep_len <= 0:
        raise ValueError(f"ep_len must be positive, got {ep_len}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_context_frames < 1:
        raise ValueError(f"min_context_frames must be >= 1, got {min_context_frames}")
    if min_context_frames > window:
        raise ValueError(f"min_context_frames ({min_context_frames}) cannot exceed window ({window})")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if decision_stride is not None and (type(decision_stride) is not int or decision_stride < 1):
        raise ValueError("decision_stride must be a positive integer or None")

    # Legacy consume samples every ``frame_stride``.  Cadence mode keeps every
    # observed row in the chunk; the caller selects decision targets separately
    # using the episode-global decision stride.
    sampled_rows = list(range(ep_len)) if decision_stride is not None else list(range(0, ep_len, frame_stride))
    chunks: List[List[int]] = []
    i = 0
    n = len(sampled_rows)
    while i < n:
        rem = n - i
        chunk_len = rng.randint(min_context_frames, window)
        take = min(chunk_len, rem)
        chunks.append(sampled_rows[i : i + take])
        i += take
    return chunks


def normalize_and_mask(
    tensor_data: np.ndarray,
    min_val: Sequence[float],
    max_val: Sequence[float],
    target_dim: int = 24,
    clamp_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(tensor_data, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Input data contains non-finite values (NaN or Inf)")

    mins = np.asarray(min_val, dtype=np.float32)
    maxs = np.asarray(max_val, dtype=np.float32)
    raw_dim = len(mins)
    if mins.ndim != 1 or maxs.shape != mins.shape or not np.isfinite(mins).all() or not np.isfinite(maxs).all() or np.any(maxs < mins):
        raise ValueError("Normalization statistics must be finite, matching vectors with max >= min")
    if len(maxs) != raw_dim:
        raise ValueError(f"Mins length ({raw_dim}) does not match maxs length ({len(maxs)})")
    if target_dim < raw_dim:
        raise ValueError(f"target_dim ({target_dim}) cannot be smaller than raw_dim ({raw_dim})")

    denom = (maxs - mins) + 1e-8

    if arr.ndim == 1:
        if arr.shape[0] != raw_dim:
            raise ValueError(f"Expected 1D data of dim {raw_dim}, got {arr.shape[0]}")
        norm_valid = 2.0 * (arr - mins) / denom - 1.0
        norm_valid = np.clip(norm_valid, clamp_range[0], clamp_range[1])

        padded = np.zeros((target_dim,), dtype=np.float32)
        mask = np.zeros((target_dim,), dtype=np.float32)
        padded[:raw_dim] = norm_valid
        mask[:raw_dim] = 1.0
        return padded, mask

    elif arr.ndim == 2:
        t, d = arr.shape
        if d != raw_dim:
            raise ValueError(f"Expected 2D data with feature dim {raw_dim}, got {d}")
        norm_valid = 2.0 * (arr - mins) / denom - 1.0
        norm_valid = np.clip(norm_valid, clamp_range[0], clamp_range[1])

        padded = np.zeros((t, target_dim), dtype=np.float32)
        mask = np.zeros((t, target_dim), dtype=np.float32)
        padded[:, :raw_dim] = norm_valid
        mask[:, :raw_dim] = 1.0
        return padded, mask
    else:
        raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")


def decode_exact_video_frame(video_path: Union[str, Path], frame_idx: int) -> Image.Image:
    import av

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                return frame.to_image()
            if i > frame_idx:
                break

    raise IndexError(f"Frame index {frame_idx} exceeds total frames in {video_path}")


_DEFAULT_SINGLE_FRAME_DECODER = decode_exact_video_frame


def decode_exact_video_frames(
    video_path: Union[str, Path], frame_indices: Sequence[int]
) -> Dict[int, Image.Image]:
    """Decode several requested frames with one container scan.

    The previous per-frame helper reopened the video and decoded from frame 0
    for every requested index.  Samples still own their decoded images, but a
    history window now pays for one sequential scan instead of one scan/frame.
    """
    import av

    video_path = Path(video_path)
    requested = sorted(set(int(i) for i in frame_indices))
    if not requested:
        raise ValueError("frame_indices cannot be empty")
    if requested[0] < 0:
        raise ValueError("frame_indices must be non-negative")
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")
        found: Dict[int, Image.Image] = {}
        wanted = set(requested)
        for i, frame in enumerate(container.decode(container.streams.video[0])):
            if i in wanted:
                found[i] = frame.to_image()
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise IndexError(f"Frame indices {sorted(missing)} exceed total frames in {video_path}")
    return found


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_requested_frames(video_path: Path, frame_ids: Sequence[int]) -> Dict[int, Image.Image]:
    """Decode in one scan, while honoring legacy single-frame test hooks."""
    if decode_exact_video_frame is not _DEFAULT_SINGLE_FRAME_DECODER:
        return {int(fid): decode_exact_video_frame(video_path, int(fid)) for fid in frame_ids}
    return decode_exact_video_frames(video_path, frame_ids)


class BoundedParquetCache:
    def __init__(self, capacity: int = 2):
        self.capacity = max(1, capacity)
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class BoundedFrameCache:
    """Small worker-local cache for decoded raw frames.

    Frames are ordinary PIL images; no model tensors or autograd state are
    retained here.  Keeping only a couple of episodes bounds worker memory
    while avoiding repeated video scans for adjacent chunks.
    """

    def __init__(self, capacity: int = 2):
        self.capacity = max(1, capacity)
        self.cache: OrderedDict[int, Dict[int, Image.Image]] = OrderedDict()

    def get(self, episode_id: int) -> Optional[Dict[int, Image.Image]]:
        if episode_id in self.cache:
            self.cache.move_to_end(episode_id)
            return self.cache[episode_id]
        return None

    def put(self, episode_id: int, frames: Dict[int, Image.Image]) -> None:
        self.cache[episode_id] = frames
        self.cache.move_to_end(episode_id)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class MetaWorldWindows(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        norm_stats: Dict[str, Any],
        horizon: int = 50,
        state_dim: int = 24,
        action_dim: int = 24,
        window: int = 2,
        frame_stride: int = 5,
        split: str = "train",
        seed: int = 4042,
        val_fraction: float = 0.1,
        max_episodes: Optional[int] = None,
        context_mode: str = "window",
        min_context_frames: int = 1,
        decision_stride: Optional[int] = None,
        execution_horizon: int = 5,
    ):
        if context_mode not in ("window", "consume", "causal"):
            raise ValueError(f"context_mode must be 'window', 'consume' or 'causal', got '{context_mode}'")
        if window < 1 or min_context_frames < 1:
            raise ValueError(f"window and min_context_frames must be >= 1, got window={window}, min_context_frames={min_context_frames}")
        if min_context_frames > window:
            raise ValueError(f"min_context_frames ({min_context_frames}) cannot exceed window ({window})")
        if min(horizon, state_dim, action_dim, frame_stride) < 1 or (max_episodes is not None and max_episodes < 1):
            raise ValueError("Window, stride, horizon, dimensions and episode limit must be positive")
        if decision_stride is not None and (type(decision_stride) is not int or decision_stride < 1):
            raise ValueError("decision_stride must be a positive integer or None")
        if type(execution_horizon) is not int or execution_horizon < 1:
            raise ValueError("execution_horizon must be a positive integer")
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.norm_stats = norm_stats
        if "observation.state" not in norm_stats or "action" not in norm_stats:
            raise KeyError("norm_stats must contain 'observation.state' and 'action'")

        self.state_mins = norm_stats["observation.state"]["min"]
        self.state_maxs = norm_stats["observation.state"]["max"]
        self.action_mins = norm_stats["action"]["min"]
        self.action_maxs = norm_stats["action"]["max"]
        self.raw_dim = len(self.action_mins)

        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window = window
        self.frame_stride = frame_stride
        self.split = split
        self.seed = seed
        self.val_fraction = val_fraction
        self.max_episodes = max_episodes
        self.context_mode = context_mode
        self.min_context_frames = min_context_frames
        self.decision_stride = decision_stride
        self.execution_horizon = execution_horizon

        self.parquet_cache = BoundedParquetCache(capacity=2)
        self.frame_cache = BoundedFrameCache(capacity=1)

        self.info = self._load_info()
        if "chunks_size" not in self.info:
            raise KeyError("meta/info.json must contain 'chunks_size'")
        self.chunks_size = int(self.info["chunks_size"])
        self.data_template = self.info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        self.video_template = self.info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )

        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self.active_episodes = self._partition_episodes()

        self.anchors: List[Tuple[Dict[str, Any], int]] = []
        self._consume_histories: List[List[int]] = []
        self._causal_chunks: List[List[int]] = []
        if self.context_mode == "window":
            for ep in self.active_episodes:
                ep_len = ep.get("length", 0)
                if ep_len <= 0:
                    raise ValueError(f"Episode {ep.get('episode_index')} has invalid length {ep_len}")
                for row in range(ep_len):
                    # Cadence mode keeps the legacy per-row index geometry for
                    # window callers; non-decision rows are represented as
                    # empty-target samples and can be skipped by the trainer.
                    self.anchors.append((ep, row))
        else:
            for ep in self.active_episodes:
                ep_len = ep.get("length", 0)
                if ep_len <= 0:
                    raise ValueError(f"Episode {ep.get('episode_index')} has invalid length {ep_len}")
                ep_idx = int(ep["episode_index"])
                ep_rng = random.Random(self.seed + ep_idx)
                chunks = partition_episode_consume_chunks(
                    ep_len=ep_len,
                    frame_stride=self.frame_stride,
                    window=self.window,
                    min_context_frames=self.min_context_frames,
                    rng=ep_rng,
                    decision_stride=self.decision_stride,
                )
                for chunk in chunks:
                    # consume supervises only the final row for backwards
                    # compatibility; causal supervises every arrival in the
                    # chunk while exposing only its prefix to each target.
                    last_row = chunk[-1]
                    self.anchors.append((ep, last_row))
                    if self.context_mode == "consume":
                        self._consume_histories.append(chunk)
                    else:
                        self._causal_chunks.append(chunk)

        if not self.anchors:
            raise ValueError(f"No valid frames found across {len(self.active_episodes)} active episodes")

        self.metadata_sha256 = {
            name: _sha256_file(self.root / "meta" / name)
            for name in ("info.json", "tasks.jsonl", "episodes.jsonl")
        }
        partition_payload = {
            "active_episode_ids": [int(ep["episode_index"]) for ep in self.active_episodes],
            "anchors": [(int(ep["episode_index"]), int(row)) for ep, row in self.anchors],
            "consume_histories": self._consume_histories,
            "causal_chunks": self._causal_chunks,
            "decision_stride": self.decision_stride,
            "execution_horizon": self.execution_horizon,
        }
        self.partition_fingerprint = hashlib.sha256(
            json.dumps(partition_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _load_info(self) -> Dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing required metadata file: {info_path}")
        with open(info_path, "r") as f:
            return json.load(f)

    def _load_tasks(self) -> Dict[int, str]:
        tasks_file = self.root / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_file}")

        tasks_map = {}
        with open(tasks_file, "r") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    obj = json.loads(line_s)
                    if "task_index" in obj and "task" in obj:
                        tasks_map[int(obj["task_index"])] = str(obj["task"])
        if not tasks_map:
            raise ValueError(f"No valid tasks found in {tasks_file}")
        return tasks_map

    def _load_episodes(self) -> List[Dict[str, Any]]:
        episodes_file = self.root / "meta" / "episodes.jsonl"
        if not episodes_file.exists():
            raise FileNotFoundError(f"Missing episodes file: {episodes_file}")

        episodes = []
        with open(episodes_file, "r") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    episodes.append(json.loads(line_s))
        if not episodes:
            raise ValueError(f"No episodes in {episodes_file}")
        return episodes

    def _partition_episodes(self) -> List[Dict[str, Any]]:
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in [0.0, 1.0), got {self.val_fraction}")

        sorted_episodes = sorted(self.episodes, key=lambda x: int(x["episode_index"]))
        rng = random.Random(self.seed)
        shuffled = list(sorted_episodes)
        rng.shuffle(shuffled)

        if self.max_episodes is not None and self.max_episodes > 0:
            shuffled = shuffled[: self.max_episodes]

        n_total = len(shuffled)
        if n_total == 0:
            raise ValueError("No episodes available to partition")

        n_val = int(round(n_total * self.val_fraction))
        val_set = shuffled[:n_val]
        train_set = shuffled[n_val:]

        if self.split == "val":
            if len(val_set) == 0:
                raise ValueError(
                    f"Validation split is empty! Total episodes: {n_total}, val_fraction: {self.val_fraction}"
                )
            return val_set
        elif self.split == "train":
            if len(train_set) == 0:
                raise ValueError("Train split is empty! Increase total episodes or reduce val_fraction.")
            return train_set
        elif self.split == "all":
            return shuffled
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _locate_parquet_path(self, ep_idx: int) -> Path:
        chunk_idx = ep_idx // self.chunks_size
        rel_path = self.data_template.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        p = self.root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Parquet file for episode {ep_idx} not found: {p}")
        return p

    def _locate_video_path(self, ep_idx: int) -> Path:
        chunk_idx = ep_idx // self.chunks_size
        rel_path = self.video_template.format(
            episode_chunk=chunk_idx,
            episode_index=ep_idx,
            video_key="observation.images.image",
        )
        p = self.root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Video file for episode {ep_idx} not found: {p}")
        return p

    def _get_episode_dataframe(self, ep: Dict[str, Any]) -> Any:
        import pandas as pd

        ep_idx = int(ep["episode_index"])
        parquet_path = self._locate_parquet_path(ep_idx)

        cached = self.parquet_cache.get(str(parquet_path))
        if cached is not None:
            df = cached
        else:
            df = pd.read_parquet(parquet_path)
            self.parquet_cache.put(str(parquet_path), df)

        if "episode_index" not in df.columns:
            raise KeyError(f"Parquet {parquet_path} missing required column 'episode_index'")
        sub_df = df[df["episode_index"] == ep_idx]
        if len(sub_df) == 0:
            raise ValueError(f"Parquet {parquet_path} does not contain episode_index {ep_idx}")
        df = sub_df

        required_cols = {"frame_index", "episode_index", "task_index", "observation.state", "action"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(f"Parquet {parquet_path} missing required columns: {missing_cols}")

        expected_len = ep.get("length", len(df))
        if len(df) != expected_len:
            raise ValueError(
                f"Episode {ep_idx} length mismatch: metadata {expected_len}, parquet {len(df)}"
            )

        frame_indices = df["frame_index"].values
        diffs = np.diff(frame_indices)
        if not np.all(diffs > 0):
            raise ValueError(f"Episode {ep_idx} has non-strictly-increasing frame_index sequence")

        return df

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep, row_idx = self.anchors[idx]
        ep_idx = int(ep["episode_index"])
        df = self._get_episode_dataframe(ep)
        ep_len = len(df)
        decision_stride = getattr(self, "decision_stride", None)
        execution_horizon = getattr(self, "execution_horizon", 5)

        curr_row = df.iloc[row_idx]

        def is_decision_row(row: int) -> bool:
            return decision_stride is None or int(df.iloc[row]["frame_index"]) % decision_stride == 0

        raw_task_idx = curr_row["task_index"]
        if raw_task_idx is None or (isinstance(raw_task_idx, float) and np.isnan(raw_task_idx)):
            raise ValueError(f"Row {row_idx} of episode {ep_idx} has null or NaN task_index")
        task_idx = int(raw_task_idx)
        if task_idx not in self.tasks:
            raise KeyError(f"task_index {task_idx} not found in meta/tasks.jsonl")
        prompt = self.tasks[task_idx]
        if not prompt or not prompt.strip():
            raise ValueError(f"Empty prompt description for task_index {task_idx}")

        if self.context_mode == "consume":
            history_rows = list(self._consume_histories[idx])
            if decision_stride is None:
                target_rows = [row_idx]
            else:
                target_rows = [r for r in history_rows if is_decision_row(r)]
        elif self.context_mode == "causal":
            history_rows = list(self._causal_chunks[idx])
            if decision_stride is None:
                target_rows = list(history_rows)
            else:
                # Decisions are tied to episode-global rows.  Keep chunks with
                # no phase-zero target so their observed frames remain part of
                # the stream; callers may skip their empty supervision.
                target_rows = [r for r in history_rows if is_decision_row(r)]
        else:
            history_rows = compute_history_row_indices(
                current_row=row_idx,
                window=self.window,
                stride=self.frame_stride,
            )
            target_rows = [row_idx] if is_decision_row(row_idx) else []

        video_path = self._locate_video_path(ep_idx)
        images_window = []
        frame_ids = []
        observation_times: Optional[List[float]] = None
        time_source: Optional[str] = None

        has_timestamp_col = "timestamp" in df.columns
        fps = self.info.get("fps")
        if not has_timestamp_col and fps is not None and (isinstance(fps, bool) or not np.isfinite(float(fps)) or float(fps) <= 0):
            raise ValueError(f"metadata fps must be finite and positive, got {fps}")

        if has_timestamp_col:
            time_source = "parquet_timestamp"
            observation_times = []
            for r_idx in history_rows:
                t_val = df.iloc[r_idx]["timestamp"]
                if t_val is None or isinstance(t_val, (bool, np.bool_)) or not np.isfinite(float(t_val)) or float(t_val) < 0:
                    raise ValueError(f"Row {r_idx} in episode {ep_idx} has invalid timestamp")
                observation_times.append(float(t_val))
        elif fps is not None:
            time_source = "metadata_fps"
            observation_times = []
            for r_idx in history_rows:
                f_idx = int(df.iloc[r_idx]["frame_index"])
                observation_times.append(float(f_idx / float(fps)))
        else:
            time_source = None
            observation_times = None

        requested_frame_ids = [int(df.iloc[r_idx]["frame_index"]) for r_idx in history_rows]
        frame_cache = getattr(self, "frame_cache", None)
        if frame_cache is None:
            frame_cache = BoundedFrameCache(capacity=1)
            self.frame_cache = frame_cache
        decoded_frames = frame_cache.get(ep_idx)
        decode_ids = requested_frame_ids
        if self.context_mode == "causal" and decoded_frames is None:
            # Causal chunks from one episode share one worker-local decode.
            decode_ids = [int(v) for v in df["frame_index"].tolist()]
        if decoded_frames is None or any(fid not in decoded_frames for fid in requested_frame_ids):
            newly_decoded = _decode_requested_frames(video_path, decode_ids)
            if decoded_frames is None:
                decoded_frames = newly_decoded
            else:
                decoded_frames = {**decoded_frames, **newly_decoded}
            frame_cache.put(ep_idx, decoded_frames)
        for r_idx in history_rows:
            r_data = df.iloc[r_idx]
            f_idx = int(r_data["frame_index"])
            frame_ids.append(f_idx)
            frame_img = decoded_frames[f_idx].copy()
            images_window.append([frame_img])

        state_values = []
        state_masks = []
        action_values = []
        action_masks = []
        action_time_masks = []
        action_time_weights = []
        valid_action_lengths: List[int] = []
        for target_row in target_rows:
            raw_state = df.iloc[target_row]["observation.state"]
            if raw_state is None:
                raise ValueError(f"Row {target_row} has null 'observation.state'")
            state_norm, state_mask = normalize_and_mask(
                np.asarray(raw_state, dtype=np.float32),
                min_val=self.state_mins,
                max_val=self.state_maxs,
                target_dim=self.state_dim,
            )
            end_row = min(target_row + self.horizon, ep_len)
            action_rows = df.iloc[target_row:end_row]["action"].tolist()
            if len(action_rows) == 0:
                raise ValueError(f"Episode {ep_idx} row {target_row} has 0 action rows")
            valid_len = len(action_rows)
            while len(action_rows) < self.horizon:
                action_rows.append(action_rows[-1])
            actions_norm, action_mask = normalize_and_mask(
                np.asarray(action_rows, dtype=np.float32),
                min_val=self.action_mins,
                max_val=self.action_maxs,
                target_dim=self.action_dim,
            )
            # Repeat-last values are useful for a fixed-shape head but are not
            # real labels.  Keep their time positions explicitly masked out.
            if valid_len < self.horizon:
                action_mask[valid_len:, :] = 0.0
            action_time_mask = np.zeros((self.horizon,), dtype=np.float32)
            action_time_mask[:valid_len] = 1.0
            action_time_weight = np.ones((self.horizon,), dtype=np.float32)
            action_time_weight[: min(execution_horizon, self.horizon)] = 4.0
            state_values.append(state_norm)
            state_masks.append(state_mask)
            action_values.append(actions_norm)
            action_masks.append(action_mask)
            action_time_masks.append(action_time_mask)
            action_time_weights.append(action_time_weight)
            valid_action_lengths.append(valid_len)

        # Empty-target cadence chunks are valid observations.  Preserve their
        # fixed trailing dimensions so generic collators can still inspect them.
        if target_rows:
            state_tensor = torch.from_numpy(np.stack(state_values, axis=0))
            state_mask_tensor = torch.from_numpy(np.stack(state_masks, axis=0))
            actions_tensor = torch.from_numpy(np.stack(action_values, axis=0))
            action_mask_tensor = torch.from_numpy(np.stack(action_masks, axis=0))
            action_time_mask_tensor = torch.from_numpy(np.stack(action_time_masks, axis=0))
            action_time_weights_tensor = torch.from_numpy(np.stack(action_time_weights, axis=0))
        else:
            state_tensor = torch.empty((0, self.state_dim), dtype=torch.float32)
            state_mask_tensor = torch.empty((0, self.state_dim), dtype=torch.float32)
            actions_tensor = torch.empty((0, self.horizon, self.action_dim), dtype=torch.float32)
            action_mask_tensor = torch.empty((0, self.horizon, self.action_dim), dtype=torch.float32)
            action_time_mask_tensor = torch.empty((0, self.horizon), dtype=torch.float32)
            action_time_weights_tensor = torch.empty((0, self.horizon), dtype=torch.float32)

        context_start = int(df.iloc[history_rows[0]]["frame_index"])
        context_end = int(df.iloc[history_rows[-1]]["frame_index"])

        ret = {
            "images_window": images_window,
            "frame_ids": frame_ids,
            "prompt": prompt,
            "episode_id": ep_idx,
            "state": state_tensor,
            "state_mask": state_mask_tensor,
            "actions": actions_tensor,
            "action_mask": action_mask_tensor,
            "action_time_mask": action_time_mask_tensor,
            "action_time_weights": action_time_weights_tensor,
            "valid_action_lengths": valid_action_lengths,
            "execution_horizon": execution_horizon,
            "raw_dim": self.raw_dim,
            "context_mode": self.context_mode,
            "context_start": context_start,
            "context_end": context_end,
        }
        if observation_times is not None:
            ret["observation_times"] = observation_times
        if time_source is not None or decision_stride is not None:
            ret["time_source"] = time_source
        # Keep the field explicit so checkpoints can distinguish legacy
        # sampling (None) from cadence sampling without inspecting arguments.
        ret["decision_stride"] = decision_stride
        if self.context_mode == "causal":
            target_indices = [history_rows.index(r) for r in target_rows]
            visible_counts = [i + 1 for i in target_indices]
            ret.update({
                "target_rows": list(target_rows),
                "target_indices": target_indices,
                "target_frame_ids": [frame_ids[i] for i in target_indices],
                "target_count": len(target_indices),
                "visible_counts": visible_counts,
                # Alias used by the native memory readers.
                "visible_frame_counts": visible_counts,
                "replay_groups": [
                    {"observation_indices": list(range(i + 1)), "target_positions": [pos]}
                    for pos, i in enumerate(target_indices)
                ],
            })
        elif decision_stride is not None:
            # Keep the same target/visibility contract available to cadence
            # callers in legacy window/consume modes.
            target_indices = [history_rows.index(r) for r in target_rows]
            visible_counts = [i + 1 for i in target_indices]
            ret.update({
                "target_rows": list(target_rows),
                "target_indices": target_indices,
                "target_frame_ids": [frame_ids[i] for i in target_indices],
                "target_count": len(target_indices),
                "visible_counts": visible_counts,
                "visible_frame_counts": visible_counts,
                "replay_groups": [
                    {"observation_indices": list(range(i + 1)), "target_positions": [pos]}
                    for pos, i in enumerate(target_indices)
                ],
            })
        return ret

    def get_data_contract(self) -> Dict[str, Any]:
        meta_info = json.dumps(self.info, sort_keys=True)
        active_ids = [int(ep["episode_index"]) for ep in self.active_episodes]
        metadata_sha256 = getattr(self, "metadata_sha256", {})
        partition_fingerprint = getattr(self, "partition_fingerprint", "")
        decision_stride = getattr(self, "decision_stride", None)
        execution_horizon = getattr(self, "execution_horizon", 5)
        time_source = self._contract_time_source()
        fingerprint_raw = json.dumps({
            "root": str(self.root),
            "meta_info": meta_info,
            "active_episode_ids": active_ids,
            "metadata_sha256": metadata_sha256,
            "partition_fingerprint": partition_fingerprint,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "decision_stride": decision_stride,
            "execution_horizon": execution_horizon,
            "action_time_weighting": "execution=4.0,remaining=1.0",
            "time_source": time_source,
        }, sort_keys=True, separators=(",", ":"))
        data_fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()
        return {
            "context_mode": self.context_mode,
            "window": self.window,
            "frame_stride": self.frame_stride,
            "min_context_frames": self.min_context_frames,
            "seed": self.seed,
            "max_episodes": self.max_episodes,
            "split": self.split,
            "active_episode_ids": active_ids,
            "metadata_sha256": metadata_sha256,
            "metadata_files_sha256": metadata_sha256,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "decision_stride": decision_stride,
            "execution_horizon": execution_horizon,
            "valid_action_lengths": "per_target:min(horizon, episode_length-target_row); repeat_last_tail_masked",
            "action_time_weights": "4.0x first execution_horizon steps, 1.0x remaining steps",
            "time_source": time_source,
            "time_source_policy": "per_sample:parquet_timestamp>metadata_fps>none",
            "partition_fingerprint": partition_fingerprint,
            "data_fingerprint": data_fingerprint,
        }

    def _contract_time_source(self) -> Optional[str]:
        """Resolve the observed timestamp source for the contract when cheap.

        Timestamp provenance is per episode.  A first-episode probe keeps the
        contract useful for homogeneous datasets while avoiding a full parquet
        scan; mixed or unavailable metadata is reported explicitly.
        """
        episodes = getattr(self, "active_episodes", ())
        info = getattr(self, "info", {})
        if not episodes:
            return "metadata_fps" if info.get("fps") is not None else None
        sources = set()
        for ep in episodes[:1]:
            try:
                df = self._get_episode_dataframe(ep)
            except (ImportError, FileNotFoundError, KeyError, ValueError, AttributeError):
                break
            if "timestamp" in df.columns:
                sources.add("parquet_timestamp")
            elif info.get("fps") is not None:
                sources.add("metadata_fps")
            else:
                sources.add(None)
        if not sources:
            return "metadata_fps" if info.get("fps") is not None else None
        return next(iter(sources)) if len(sources) == 1 else "mixed"
