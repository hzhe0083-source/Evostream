"""Tests for PredictiveTrainingDataset in fabri_moss/predictive_data.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.predictive_data import PredictiveTrainingDataset
from fabri_moss.tests.test_native_data import _create_mock_metaworld_root


@pytest.fixture
def mock_norm_stats() -> Dict[str, Any]:
    return {
        "observation.state": {"min": [-1.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }


def test_predictive_irregular_timestamps_and_tail_invalidation(
    tmp_path: Path, mock_norm_stats: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    """Verify physical time selection on non-uniform timestamps, deduplication, and invalidation at tail."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1, lengths=[6])
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16), color=(int(f * 20) % 255, 0, 0)) for f in fids},
    )
    monkeypatch.setattr(
        "fabri_moss.delta_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16), color=(int(f * 20) % 255, 0, 0)) for f in fids},
    )

    timestamps = [0.00, 0.05, 0.25, 0.35, 0.45, 0.50]
    pq_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(pq_path)
    df["timestamp"] = timestamps
    df.to_parquet(pq_path)

    ds = PredictiveTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        future_horizons=(0.1, 0.3),
        split="train",
    )

    sample = ds[0]
    M = sample["target_count"]
    assert M == 6
    future_indices = sample["future_indices"]
    future_deltas = sample["future_deltas"]
    future_valid = sample["future_valid"]
    future_frame_ids = sample["future_frame_ids"]
    future_images = sample["future_images"]

    assert future_indices.shape == (6, 2)
    assert future_deltas.shape == (6, 2)
    assert future_valid.shape == (6, 2)
    assert future_frame_ids.shape == (6, 2)

    # Physical time horizons:
    # Row 0 (0.00): +0.1->0.10 picks row 2 (0.25); +0.3->0.30 picks row 3 (0.35)
    # Row 1 (0.05): +0.1->0.15 picks row 2 (0.25); +0.3->0.35 picks row 3 (0.35)
    # Row 2 (0.25): +0.1->0.35 picks row 3 (0.35); +0.3->0.55 invalid
    # Row 3 (0.35): +0.1->0.45 picks row 4 (0.45); +0.3->0.65 invalid
    # Row 4 (0.45): +0.1->0.55 invalid;            +0.3->0.75 invalid
    # Row 5 (0.50): invalid;                       invalid
    expected_rows = [
        [2, 3],
        [2, 3],
        [3, -1],
        [4, -1],
        [-1, -1],
        [-1, -1],
    ]

    for m in range(6):
        for k in range(2):
            exp_r = expected_rows[m][k]
            if exp_r != -1:
                assert future_valid[m, k].item() is True
                assert future_frame_ids[m, k].item() == exp_r
                assert future_deltas[m, k].item() == pytest.approx(timestamps[exp_r] - timestamps[m])
                img_idx = future_indices[m, k].item()
                assert 0 <= img_idx < len(future_images)
            else:
                assert future_valid[m, k].item() is False
                assert future_indices[m, k].item() == -1
                assert future_deltas[m, k].item() == 0.0
                assert future_frame_ids[m, k].item() == -1

    assert len(future_images) == 3
    assert future_indices[0, 0] == future_indices[1, 0]
    assert future_indices[0, 1] == future_indices[1, 1]


def test_predictive_vs_compact_base_equivalence(
    tmp_path: Path, mock_norm_stats: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    """Verify that under same seed and epoch, compact base and predictive causal keys match exactly."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=2, lengths=[12, 18])
    mock_frames = {f: Image.new("RGB", (16, 16), color=(f * 10, 50, 100)) for f in range(30)}
    monkeypatch.setattr("fabri_moss.native_data.decode_all_video_frames", lambda path, fids: mock_frames)
    monkeypatch.setattr("fabri_moss.delta_data.decode_all_video_frames", lambda path, fids: mock_frames)

    for epoch in (0, 1):
        base_ds = NativeTrainingDataset(
            root=root,
            norm_stats=mock_norm_stats,
            stream_protocol="compact_memory_replay_v1",
            augmentation=False,
            split="train",
            seed=4042,
        )
        base_ds.set_epoch(epoch)

        pred_ds = PredictiveTrainingDataset(
            root=root,
            norm_stats=mock_norm_stats,
            augmentation=False,
            split="train",
            seed=4042,
        )
        pred_ds.set_epoch(epoch)

        predictive_keys = {"future_images", "future_indices", "future_deltas", "future_valid", "future_frame_ids"}
        saw_decision_indices = False

        for idx in range(len(base_ds)):
            item_base = base_ds[idx]
            item_pred = pred_ds[idx]

            assert set(item_pred.keys()) - set(item_base.keys()) == predictive_keys
            for pk in predictive_keys:
                assert pk not in item_base

            for k, v_base in item_base.items():
                v_pred = item_pred[k]
                if isinstance(v_base, torch.Tensor):
                    torch.testing.assert_close(v_pred, v_base)
                elif isinstance(v_base, list) and len(v_base) > 0 and isinstance(v_base[0], Image.Image):
                    assert len(v_pred) == len(v_base)
                    for img_p, img_b in zip(v_pred, v_base):
                        assert np.array_equal(np.array(img_p), np.array(img_b))
                else:
                    assert v_pred == v_base, f"Mismatch on key {k} at index {idx}"

            if "decision_indices" in item_base:
                saw_decision_indices = True
                assert item_pred["decision_indices"] == item_base["decision_indices"]

    assert saw_decision_indices, "decision_indices should be present in memory replay segments and matched"


def test_predictive_data_contract_equality_and_spec(tmp_path: Path, mock_norm_stats: Dict[str, Any]):
    """Verify get_base_data_contract equality and get_data_contract predictive extensions."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=2, lengths=[10])

    base_ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="compact_memory_replay_v1",
        augmentation=False,
        split="train",
    )
    pred_ds = PredictiveTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        future_horizons=(0.1, 0.3),
        augmentation=False,
        split="train",
    )

    base_contract = base_ds.get_base_data_contract()
    pred_base_contract = pred_ds.get_base_data_contract()
    assert pred_base_contract == base_contract

    full_pred_contract = pred_ds.get_data_contract()
    assert json.dumps(full_pred_contract) is not None
    assert full_pred_contract["dataset_class"] == "PredictiveTrainingDataset"
    assert "predictive_protocol" in full_pred_contract
    assert full_pred_contract["predictive_protocol"]["protocol_name"] == "predictive_compact_replay_v1"
    assert full_pred_contract["predictive_protocol"]["future_horizons"] == [0.1, 0.3]
    assert full_pred_contract["data_fingerprint"] != base_contract["data_fingerprint"]


@pytest.mark.parametrize(
    "invalid_kwargs,err_type,match_msg",
    [
        ({"augmentation": True}, ValueError, "augmentation=False"),
        ({"future_horizons": (True, 0.3)}, TypeError, "cannot be a boolean"),
        ({"future_horizons": (0.1, False)}, TypeError, "cannot be a boolean"),
        ({"future_horizons": (0.0, 0.3)}, ValueError, "positive finite number"),
        ({"future_horizons": (-0.2, 0.3)}, ValueError, "positive finite number"),
        ({"future_horizons": (float("nan"), 0.3)}, ValueError, "positive finite number"),
        ({"future_horizons": (float("inf"), 0.3)}, ValueError, "positive finite number"),
        ({"future_horizons": ()}, ValueError, "cannot be empty"),
        ({"future_horizons": "0.1,0.3"}, TypeError, "must be a sequence"),
        ({"stream_protocol": "stream_replay_v1"}, ValueError, "compact_memory_replay_v1"),
    ],
)
def test_predictive_parameter_validation_rejections(
    tmp_path: Path, mock_norm_stats: Dict[str, Any], invalid_kwargs: Dict[str, Any], err_type: type, match_msg: str
):
    """Verify rejection of invalid augmentation, horizons, and stream protocol."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1, lengths=[5])
    with pytest.raises(err_type, match=match_msg):
        PredictiveTrainingDataset(root=root, norm_stats=mock_norm_stats, split="train", **invalid_kwargs)
