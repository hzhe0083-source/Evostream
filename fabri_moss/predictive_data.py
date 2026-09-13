"""Predictive training dataset for two-stage predictive memory fine-tuning.

Subclasses NativeTrainingDataset to provide:
1. Compact memory sampling protocol fixed to 'compact_memory_replay_v1'.
2. Data augmentation strictly disabled (augmentation=False).
3. Future prediction targets sampled at discrete physical time horizons (default 0.1s, 0.3s)
   strictly into the future (first row with time >= target_time + horizon and row > target_row).
4. Out-of-bounds future rows marked invalid (no repeat-last, no cross-episode sampling).
5. Future images deduplicated and isolated from causal context (images_window/frame_ids/decision_indices).
6. Robust data contracts documenting predictive configuration, base contract, and SHA-256 fingerprints.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch

from fabri_moss.native_data import NativeTrainingDataset


class PredictiveTrainingDataset(NativeTrainingDataset):
    """Dataset providing causal compact memory segments with detached future target images."""

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
        augmentation: bool = False,
        future_horizons: Sequence[float] = (0.1, 0.3),
        stream_protocol: str = "compact_memory_replay_v1",
        decision_stride: Optional[int] = None,
    ):
        if augmentation is not False:
            raise ValueError(
                "PredictiveTrainingDataset strictly enforces augmentation=False. "
                "Causal context and future target visual representations require unaugmented images."
            )

        if stream_protocol != "compact_memory_replay_v1":
            raise ValueError(
                f"PredictiveTrainingDataset requires stream_protocol='compact_memory_replay_v1', "
                f"got {stream_protocol!r}"
            )

        # Validate future_horizons: non-empty sequence of positive, finite numbers
        if not isinstance(future_horizons, (list, tuple)):
            raise TypeError(f"future_horizons must be a sequence, got {type(future_horizons).__name__}")
        if len(future_horizons) == 0:
            raise ValueError("future_horizons cannot be empty")

        parsed_horizons: List[float] = []
        for i, h in enumerate(future_horizons):
            if isinstance(h, (bool, np.bool_)):
                raise TypeError(f"future_horizons[{i}] cannot be a boolean: {h!r}")
            try:
                h_float = float(h)
            except (ValueError, TypeError) as err:
                raise TypeError(f"future_horizons[{i}] must be numeric: {h!r}") from err

            if not math.isfinite(h_float) or h_float <= 0.0:
                raise ValueError(
                    f"future_horizons[{i}] must be a positive finite number, got {h_float}"
                )
            parsed_horizons.append(h_float)

        self.future_horizons = tuple(parsed_horizons)

        super().__init__(
            root=root,
            norm_stats=norm_stats,
            history_frames=history_frames,
            target_frames=target_frames,
            horizon=horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            split=split,
            seed=seed,
            val_fraction=val_fraction,
            max_episodes=max_episodes,
            augmentation=False,
            stream_protocol="compact_memory_replay_v1",
            decision_stride=decision_stride,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)

        ep_idx = int(item["episode_id"])
        target_start_row = int(item["target_start_row"])
        target_end_row = int(item["target_end_row"])
        target_rows = list(range(target_start_row, target_end_row))
        M = len(target_rows)
        K = len(self.future_horizons)

        ep = self._episode_by_id[ep_idx]
        df = self._get_episode_dataframe(ep)
        ep_len = len(df)

        all_timestamps, _ = self._validate_and_get_timestamps(df, ep_idx)
        all_frame_ids = [int(f) for f in df["frame_index"].values]

        # Ensure raw frames are in cache
        decoded_frames = self.raw_frame_cache.get(ep_idx)
        if decoded_frames is None:
            video_path = self._locate_video_path(ep_idx)
            from fabri_moss.delta_data import decode_all_video_frames
            decoded_frames = decode_all_video_frames(video_path, all_frame_ids)
            self.raw_frame_cache.put(ep_idx, decoded_frames)

        future_indices = torch.full((M, K), -1, dtype=torch.long)
        future_deltas = torch.zeros((M, K), dtype=torch.float32)
        future_valid = torch.zeros((M, K), dtype=torch.bool)
        future_frame_ids_tensor = torch.full((M, K), -1, dtype=torch.long)

        # Map unique selected future row -> index in future_images
        future_row_to_img_idx: Dict[int, int] = {}
        future_images: List[List[Image.Image]] = []

        # Find future targets
        # Condition: first row with row > r and all_timestamps[cand_r] >= t_r + horizon
        for m_idx, r in enumerate(target_rows):
            t_r = all_timestamps[r]
            for k_idx, h in enumerate(self.future_horizons):
                cand_target_time = t_r + h
                selected_r: Optional[int] = None
                for cand_r in range(r + 1, ep_len):
                    if all_timestamps[cand_r] >= cand_target_time:
                        selected_r = cand_r
                        break

                if selected_r is not None:
                    # Valid future target
                    if selected_r not in future_row_to_img_idx:
                        img_idx = len(future_images)
                        future_row_to_img_idx[selected_r] = img_idx
                        fid = all_frame_ids[selected_r]
                        raw_img = decoded_frames[fid]
                        future_images.append([raw_img])
                    else:
                        img_idx = future_row_to_img_idx[selected_r]

                    future_indices[m_idx, k_idx] = img_idx
                    future_deltas[m_idx, k_idx] = float(all_timestamps[selected_r] - t_r)
                    future_valid[m_idx, k_idx] = True
                    future_frame_ids_tensor[m_idx, k_idx] = all_frame_ids[selected_r]
                else:
                    # Beyond end of episode: invalid
                    future_indices[m_idx, k_idx] = -1
                    future_deltas[m_idx, k_idx] = 0.0
                    future_valid[m_idx, k_idx] = False
                    future_frame_ids_tensor[m_idx, k_idx] = -1

        item["future_images"] = future_images
        item["future_indices"] = future_indices
        item["future_deltas"] = future_deltas
        item["future_valid"] = future_valid
        item["future_frame_ids"] = future_frame_ids_tensor

        return item

    def get_data_contract(self) -> Dict[str, Any]:
        """Return JSON-serializable stable contract specification for PredictiveTrainingDataset."""
        base_contract = self.get_base_data_contract()
        compact_contract = self._get_compact_protocol_contract()

        predictive_spec: Dict[str, Any] = {
            "protocol_name": "predictive_compact_replay_v1",
            "version": "1.0",
            "future_horizons": [float(h) for h in self.future_horizons],
            "augmentation_policy": "strictly_disabled",
            "text_encoding": False,
            "expected_time_input": "numeric_seconds",
            "compact_memory_contract": compact_contract,
        }

        contract = copy.deepcopy(base_contract)
        contract["dataset_class"] = "PredictiveTrainingDataset"
        contract["stream_protocol"] = compact_contract
        contract["predictive_protocol"] = predictive_spec

        # Compute deterministic fingerprint including base contract and predictive specification
        sorted_predictive_json = json.dumps(predictive_spec, sort_keys=True)
        new_fingerprint_raw = f"{base_contract['data_fingerprint']}:{sorted_predictive_json}"
        contract["data_fingerprint"] = hashlib.sha256(new_fingerprint_raw.encode("utf-8")).hexdigest()

        return contract
