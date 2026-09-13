import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF

from fabri_moss.data import (
    BoundedParquetCache,
    normalize_and_mask,
)
from fabri_moss.runtime import compute_file_sha256


def identity_collate(batch: Any) -> Any:
    """Top-level identity collate keeping raw episode dictionary without PyTorch auto-tensor conversion."""
    return batch


def sample_episode_augmentation_params(
    rng: random.Random,
    img_width: int,
    img_height: int,
    p: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Sample data augmentation parameters shared across an entire episode.

    50% probability to apply augmentation:
    - Random crop: 10 attempts sampling area scale in [0.95, 1.0] and aspect ratio in [3/4, 4/3],
      with center-crop fallback like RandomResizedCrop.
    - Random rotation: angle in [-5.0, 5.0] degrees.
    - Color jitter: brightness, contrast, saturation, hue with application order randomly
      shuffled by episode RNG and shared across all frames of the episode.
    """
    if rng.random() > p:
        return None

    area = img_width * img_height
    crop_params = None
    log_ratio = (math.log(3.0 / 4.0), math.log(4.0 / 3.0))

    # 10 trial attempts like RandomResizedCrop
    for _ in range(10):
        target_area = area * rng.uniform(0.95, 1.0)
        aspect_ratio = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= img_width and 0 < h <= img_height:
            i = rng.randint(0, img_height - h)
            j = rng.randint(0, img_width - w)
            crop_params = (i, j, h, w)
            break

    # Fallback to center crop if 10 attempts failed
    if crop_params is None:
        in_ratio = float(img_width) / float(img_height)
        if in_ratio < 3.0 / 4.0:
            w = img_width
            h = int(round(w / (3.0 / 4.0)))
        elif in_ratio > 4.0 / 3.0:
            h = img_height
            w = int(round(h * (4.0 / 3.0)))
        else:
            w = img_width
            h = img_height
        i = (img_height - h) // 2
        j = (img_width - w) // 2
        crop_params = (i, j, h, w)

    rotation_angle = rng.uniform(-5.0, 5.0)

    # Color jitter parameters: brightness, contrast, saturation, hue
    brightness_factor = rng.uniform(max(0.0, 1.0 - 0.3), 1.0 + 0.3)
    contrast_factor = rng.uniform(max(0.0, 1.0 - 0.4), 1.0 + 0.4)
    saturation_factor = rng.uniform(max(0.0, 1.0 - 0.5), 1.0 + 0.5)
    hue_factor = rng.uniform(-0.08, 0.08)

    # Application order shuffled by episode RNG and shared across all frames in this episode
    jitter_order = ["brightness", "contrast", "saturation", "hue"]
    rng.shuffle(jitter_order)

    return {
        "crop": crop_params,
        "rotation": rotation_angle,
        "brightness": brightness_factor,
        "contrast": contrast_factor,
        "saturation": saturation_factor,
        "hue": hue_factor,
        "jitter_order": jitter_order,
        "orig_size": (img_height, img_width),
    }


def apply_episode_augmentation(
    img: Image.Image,
    params: Optional[Dict[str, Any]],
) -> Image.Image:
    """Apply sampled episode-level augmentation to a single PIL image."""
    if params is None:
        return img

    out = img
    # Crop and resize back to original size
    if params.get("crop") is not None:
        top, left, height, width = params["crop"]
        out = TF.crop(out, top, left, height, width)
        orig_h, orig_w = params["orig_size"]
        out = TF.resize(out, [orig_h, orig_w], interpolation=TF.InterpolationMode.BILINEAR)

    # Rotation
    angle = params.get("rotation", 0.0)
    if abs(angle) > 1e-4:
        out = TF.rotate(out, angle, interpolation=TF.InterpolationMode.BILINEAR)

    # Color jitter applied in episode-shuffled order
    jitter_order = params.get("jitter_order", ["brightness", "contrast", "saturation", "hue"])
    for op in jitter_order:
        if op == "brightness" and "brightness" in params:
            out = TF.adjust_brightness(out, params["brightness"])
        elif op == "contrast" and "contrast" in params:
            out = TF.adjust_contrast(out, params["contrast"])
        elif op == "saturation" and "saturation" in params:
            out = TF.adjust_saturation(out, params["saturation"])
        elif op == "hue" and "hue" in params:
            out = TF.adjust_hue(out, params["hue"])

    return out


def decode_all_video_frames(
    video_path: Union[str, Path],
    target_frame_indices: Sequence[int],
) -> Dict[int, Image.Image]:
    """Single sequential decode of video to extract all requested frame indices."""
    import av

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    needed_indices = set(target_frame_indices)
    frames_dict: Dict[int, Image.Image] = {}

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in needed_indices:
                frames_dict[i] = frame.to_image()
            if len(frames_dict) == len(needed_indices):
                break

    missing = needed_indices - set(frames_dict.keys())
    if missing:
        raise IndexError(f"Frame indices {sorted(list(missing))} not found in video {video_path}")

    return frames_dict


def partition_delta_chunks(
    frame_indices: Sequence[int],
    max_chunk_size: int = 5,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """Partition sequence of frame indices into delta chunks.

    First chunk is strictly singleton [frame_indices[0]].
    Subsequent chunks are randomly sized in 1..max_chunk_size, covering all remaining frames without overlap.
    """
    if not frame_indices:
        raise ValueError("frame_indices cannot be empty")
    if max_chunk_size < 1:
        raise ValueError(f"max_chunk_size must be >= 1, got {max_chunk_size}")

    chunks: List[List[int]] = [[frame_indices[0]]]
    remaining = list(frame_indices[1:])
    idx = 0
    n = len(remaining)

    if rng is None:
        rng = random.Random(4042)

    while idx < n:
        sz = rng.randint(1, max_chunk_size)
        take = min(sz, n - idx)
        chunks.append(remaining[idx : idx + take])
        idx += take

    return chunks


class MetaWorldEpisodes(Dataset):
    """Episode-level sequential dataset for MetaWorld delta memory training.

    - Stratified splitting by task name: default 45 train / 5 val per task (val_fraction=0.1, 2250/250 total).
    - Uses stable sha256(task_key utf8) integer seed across processes.
    - Validates tasks are single non-empty strings matching metadata tasks (no unknown fallback).
    - Episode is the sampling unit.
    - Single sequential decode per video when accessing an episode.
    - Deterministic per-episode augmentation and chunk partitioning.
    """

    def __init__(
        self,
        root: Union[str, Path],
        norm_stats: Dict[str, Any],
        horizon: int = 50,
        state_dim: int = 24,
        action_dim: int = 24,
        max_chunk_size: int = 5,
        split: str = "train",
        seed: int = 4042,
        val_fraction: float = 0.1,
        max_episodes: Optional[int] = None,
        augmentation: bool = True,
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")
        if not (0.0 <= val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in [0.0, 1.0), got {val_fraction}")
        if min(horizon, state_dim, action_dim, max_chunk_size) < 1:
            raise ValueError("horizon, state_dim, action_dim, max_chunk_size must be positive integers")
        if max_episodes is not None and max_episodes < 1:
            raise ValueError(f"max_episodes must be positive if provided, got {max_episodes}")

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
        self.max_chunk_size = max_chunk_size
        self.split = split
        self.seed = seed
        self.val_fraction = val_fraction
        self.max_episodes = max_episodes
        self.augmentation = augmentation

        self.epoch: int = 0
        self.parquet_cache = BoundedParquetCache(capacity=2)

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

        if not self.active_episodes:
            raise ValueError(f"No active episodes found for split '{self.split}'")

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for deterministic epoch-dependent chunking/augmentation."""
        self.epoch = epoch

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
        """Stratified partition by task name with stable sha256 seed.

        - Each episode must have exactly one non-empty task string matching self.tasks values.
        - No unknown fallback allowed.
        - Uses sha256(task_key.encode('utf-8'))[:8] for cross-process reproducibility.
        """
        valid_task_set = set(self.tasks.values())
        task_groups: Dict[str, List[Dict[str, Any]]] = {}

        for ep in self.episodes:
            ep_idx = ep.get("episode_index")
            tasks_list = ep.get("tasks", [])
            if not isinstance(tasks_list, list) or len(tasks_list) != 1:
                raise ValueError(
                    f"Episode {ep_idx} must have exactly one task in 'tasks' list, got {tasks_list}"
                )
            task_key = tasks_list[0]
            if not isinstance(task_key, str) or not task_key.strip():
                raise ValueError(f"Episode {ep_idx} has empty or non-string task: {task_key!r}")
            if task_key not in valid_task_set:
                raise ValueError(
                    f"Episode {ep_idx} task {task_key!r} not found in meta/tasks.jsonl tasks: {sorted(list(valid_task_set))}"
                )
            task_groups.setdefault(task_key, []).append(ep)

        train_set: List[Dict[str, Any]] = []
        val_set: List[Dict[str, Any]] = []

        # Sort task keys for deterministic processing order
        for task_key in sorted(task_groups.keys()):
            group = sorted(task_groups[task_key], key=lambda x: int(x["episode_index"]))

            # Stable cross-process seed via sha256 first 8 bytes
            task_hash = int(hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:8], 16)
            group_seed = (self.seed + task_hash) % (2**31 - 1)
            rng = random.Random(group_seed)

            shuffled = list(group)
            rng.shuffle(shuffled)

            n_total = len(shuffled)
            n_val = int(round(n_total * self.val_fraction))
            if self.val_fraction > 0 and n_val == 0 and n_total > 1:
                n_val = 1

            val_set.extend(shuffled[:n_val])
            train_set.extend(shuffled[n_val:])

        # Sort resulting sets by episode_index
        train_set.sort(key=lambda x: int(x["episode_index"]))
        val_set.sort(key=lambda x: int(x["episode_index"]))

        # Non-overlapping verification
        train_indices = set(int(ep["episode_index"]) for ep in train_set)
        val_indices = set(int(ep["episode_index"]) for ep in val_set)
        intersection = train_indices.intersection(val_indices)
        if intersection:
            raise RuntimeError(f"Train and val splits overlap on episode indices: {sorted(list(intersection))}")

        if self.split == "val":
            selected = val_set
        elif self.split == "train":
            selected = train_set
        elif self.split == "all":
            selected = sorted(train_set + val_set, key=lambda x: int(x["episode_index"]))
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if self.max_episodes is not None and self.max_episodes > 0:
            selected = selected[: self.max_episodes]

        return selected

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

        # Verify task prompt matches metadata task
        meta_tasks = ep.get("tasks", [])
        expected_meta_task = meta_tasks[0]
        task_indices = df["task_index"].values
        if any(t is None or (isinstance(t, float) and np.isnan(t)) for t in task_indices):
            raise ValueError(f"Episode {ep_idx} contains null or NaN task_index")
        unique_tasks = sorted(list(set(int(t) for t in task_indices)))
        if len(unique_tasks) != 1:
            raise ValueError(f"Episode {ep_idx} contains multiple task indices: {unique_tasks}")
        task_idx = unique_tasks[0]
        if task_idx not in self.tasks:
            raise KeyError(f"task_index {task_idx} not found in meta/tasks.jsonl")
        parquet_task = self.tasks[task_idx]
        if parquet_task != expected_meta_task:
            raise ValueError(
                f"Episode {ep_idx} task mismatch: metadata '{expected_meta_task}' vs parquet task_index {task_idx} ('{parquet_task}')"
            )

        return df

    def __len__(self) -> int:
        return len(self.active_episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.active_episodes[idx]
        ep_idx = int(ep["episode_index"])
        df = self._get_episode_dataframe(ep)
        ep_len = len(df)

        if ep_len == 0:
            raise ValueError(f"Episode {ep_idx} has 0 rows")

        task_idx = int(df["task_index"].iloc[0])
        prompt = self.tasks[task_idx]
        if not prompt or not prompt.strip():
            raise ValueError(f"Empty prompt for task_index {task_idx}")

        frame_indices = [int(f) for f in df["frame_index"].values]

        # Chunk partitioning: first frame singleton, then 1..max_chunk_size
        chunk_rng = random.Random(self.seed + self.epoch * 10007 + ep_idx)
        chunks = partition_delta_chunks(
            frame_indices=frame_indices,
            max_chunk_size=self.max_chunk_size,
            rng=chunk_rng,
        )

        # Single sequential decode of all frames in video
        video_path = self._locate_video_path(ep_idx)
        decoded_frames = decode_all_video_frames(video_path, frame_indices)

        # Sample augmentation parameters shared across the whole episode
        aug_params = None
        if self.augmentation:
            # Deterministic RNG per episode and epoch so worker count does not alter RNG
            aug_rng = random.Random(self.seed + 100003 * self.epoch + ep_idx * 7919)
            first_img = decoded_frames[frame_indices[0]]
            aug_params = sample_episode_augmentation_params(
                rng=aug_rng,
                img_width=first_img.width,
                img_height=first_img.height,
                p=0.5,
            )

        # Prepare chunk items
        chunk_data_list: List[Dict[str, Any]] = []
        fidx_to_row = {int(df.iloc[r]["frame_index"]): r for r in range(ep_len)}

        for c_idx, c_frames in enumerate(chunks):
            c_images = []
            for fid in c_frames:
                raw_img = decoded_frames[fid]
                aug_img = apply_episode_augmentation(raw_img, aug_params)
                c_images.append([aug_img])

            # Supervision point is at the last frame of this chunk
            supervision_fid = c_frames[-1]
            supervision_row = fidx_to_row[supervision_fid]
            sup_data = df.iloc[supervision_row]

            raw_state = sup_data["observation.state"]
            if raw_state is None:
                raise ValueError(f"Episode {ep_idx} row {supervision_row} has null state")
            state_norm, state_mask = normalize_and_mask(
                np.asarray(raw_state, dtype=np.float32),
                min_val=self.state_mins,
                max_val=self.state_maxs,
                target_dim=self.state_dim,
            )

            # Actions starting from supervision_row up to horizon
            end_row = min(supervision_row + self.horizon, ep_len)
            action_rows = df.iloc[supervision_row:end_row]["action"].tolist()
            actual_action_len = len(action_rows)
            if actual_action_len == 0:
                raise ValueError(f"Episode {ep_idx} row {supervision_row} has 0 action rows")

            # Pad with repeat-last if needed
            while len(action_rows) < self.horizon:
                action_rows.append(action_rows[-1])

            # FabriVLA contract: repeat-last future action steps are supervised on all valid action dimensions (e.g. first raw_dim=4).
            # normalize_and_mask sets full_mask[:, :raw_dim] = 1.0 and full_mask[:, raw_dim:] = 0.0 across all H steps.
            actions_norm, full_mask = normalize_and_mask(
                np.asarray(action_rows, dtype=np.float32),
                min_val=self.action_mins,
                max_val=self.action_maxs,
                target_dim=self.action_dim,
            )

            chunk_data_list.append({
                "chunk_index": c_idx,
                "images": c_images,
                "frame_ids": list(c_frames),
                "state": torch.from_numpy(state_norm).unsqueeze(0),
                "state_mask": torch.from_numpy(state_mask).unsqueeze(0),
                "actions": torch.from_numpy(actions_norm).unsqueeze(0),
                "action_mask": torch.from_numpy(full_mask).unsqueeze(0),
                "is_first_decision": (c_idx == 0),
            })

        return {
            "episode_id": ep_idx,
            "prompt": prompt,
            "chunks": chunk_data_list,
            "num_decisions": len(chunks),
            "raw_dim": self.raw_dim,
            "frame_count": ep_len,
        }

    def get_data_contract(self) -> Dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        tasks_path = self.root / "meta" / "tasks.jsonl"
        episodes_path = self.root / "meta" / "episodes.jsonl"

        metadata_shas = {
            "info.json": compute_file_sha256(info_path),
            "tasks.jsonl": compute_file_sha256(tasks_path),
            "episodes.jsonl": compute_file_sha256(episodes_path),
        }

        active_ids = [int(ep["episode_index"]) for ep in self.active_episodes]
        fingerprint_raw = f"{str(self.root)}:{metadata_shas}:{active_ids}:{self.horizon}:{self.state_dim}:{self.action_dim}:delta_v1"
        data_fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()

        return {
            "dataset_class": "MetaWorldEpisodes",
            "format": "fabri_delta_v1",
            "max_chunk_size": self.max_chunk_size,
            "seed": self.seed,
            "max_episodes": self.max_episodes,
            "split": self.split,
            "val_fraction": self.val_fraction,
            "augmentation": self.augmentation,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "active_episode_ids": active_ids,
            "metadata_files_sha256": metadata_shas,
            "data_fingerprint": data_fingerprint,
        }


class EpisodePermutationSampler(Sampler[int]):
    """Deterministic permutation sampler over episode indices with start_cursor support.

    Directly yields permutation[start_cursor:] so the DataLoader does not need
    to iterate and decode skipped episodes upon resuming.
    """

    def __init__(
        self,
        num_episodes: int,
        seed: int = 4042,
        start_epoch: int = 0,
        start_cursor: int = 0,
    ):
        self.num_episodes = num_episodes
        self.seed = seed
        self.epoch = start_epoch
        self.start_cursor = start_cursor

    def set_epoch(self, epoch: int, start_cursor: int = 0) -> None:
        self.epoch = epoch
        self.start_cursor = start_cursor

    def get_permutation(self, epoch: Optional[int] = None) -> List[int]:
        ep = self.epoch if epoch is None else epoch
        rng = random.Random(self.seed + ep * 10007)
        indices = list(range(self.num_episodes))
        rng.shuffle(indices)
        return indices

    def __iter__(self):
        full_perm = self.get_permutation()
        if self.start_cursor > 0:
            return iter(full_perm[self.start_cursor :])
        return iter(full_perm)

    def __len__(self) -> int:
        return max(0, self.num_episodes - self.start_cursor)
