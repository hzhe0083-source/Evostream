"""Deployment cadence retains every expert target without dense decision histories."""

import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.native_training import NativeSequencePolicy
from fabri_moss.predictive_data import PredictiveTrainingDataset
from fabri_moss.tests.test_native_data import _create_mock_metaworld_root, mock_norm_stats


@pytest.fixture
def episode_root(tmp_path, monkeypatch):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1, lengths=[53])
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {fid: Image.new("RGB", (8, 8), color=(fid % 255, 0, 0)) for fid in fids},
    )
    return root


def test_cadence_groups_preserve_all_labels_and_causal_prefixes(episode_root, mock_norm_stats):
    kwargs = dict(root=episode_root, norm_stats=mock_norm_stats, augmentation=False)
    dense = NativeTrainingDataset(**kwargs)
    cadence = NativeTrainingDataset(**kwargs, stream_protocol="compact_memory_replay_v1", decision_stride=5)
    covered = []
    for idx in range(len(cadence)):
        sample, reference = cadence[idx], dense[idx]
        assert NativeSequencePolicy._validate_sample(None, sample) == (sample["target_end_row"], sample["target_count"])
        assert sample["memory_replay"] is True
        assert "decision_indices" not in sample
        assert sample["frame_ids"] == list(range(sample["target_end_row"]))
        assert sample["target_indices"] == reference["target_frame_ids"]
        for key in ("state", "state_mask", "actions", "action_mask"):
            assert torch.equal(sample[key], reference[key])
        positions_seen = []
        for group in sample["replay_groups"]:
            positions = group["target_positions"]
            rows = [sample["target_indices"][pos] for pos in positions]
            decisions = group["decision_indices"]
            phase = rows[0] % 5
            assert all(row % 5 == phase for row in rows)
            assert decisions == sorted({0, *range(phase, rows[-1] + 1, 5)})
            assert set(rows) <= set(decisions)
            assert group["observation_indices"] == list(range(rows[-1] + 1))
            assert decisions[-1] == rows[-1]
            # Row zero bootstraps every history; nonzero phases then have one short gap.
            gaps = np.diff(decisions)
            if len(gaps):
                assert gaps[0] == (phase or 5)
                assert all(gap == 5 for gap in gaps[1:])
            positions_seen.extend(positions)
            covered.extend(rows)
        assert sorted(positions_seen) == list(range(sample["target_count"]))
    assert sorted(covered) == list(range(53))
    assert len(covered) == cadence.total_targets == 53
    late = cadence[5]
    assert [[late["target_indices"][pos] for pos in group["target_positions"]]
            for group in late["replay_groups"]] == [[40, 45], [41, 46], [42, 47], [43], [44]]
    early = cadence[0]
    assert [group["decision_indices"] for group in early["replay_groups"]] == [
        [0, 5], [0, 1, 6], [0, 2, 7], [0, 3], [0, 4],
    ]


def test_cadence_uses_row_positions_and_keeps_future_labels(episode_root, mock_norm_stats):
    parquet = episode_root / "data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(parquet)
    df["frame_index"] = [10 + row * 3 for row in range(len(df))]
    df.to_parquet(parquet)
    kwargs = dict(root=episode_root, norm_stats=mock_norm_stats, future_horizons=(0.15, 0.35))
    legacy = PredictiveTrainingDataset(**kwargs)
    cadence = PredictiveTrainingDataset(**kwargs, decision_stride=5)
    for idx in range(len(cadence)):
        sample, reference = cadence[idx], legacy[idx]
        for key in ("actions", "state", "future_indices", "future_deltas", "future_valid", "future_frame_ids"):
            assert torch.equal(sample[key], reference[key])
        assert sample["target_frame_ids"] == reference["target_frame_ids"]
        for expected, actual in zip(reference["future_images"], sample["future_images"]):
            assert np.array_equal(np.asarray(expected[0]), np.asarray(actual[0]))
        for group in sample["replay_groups"]:
            assert group["decision_indices"][0] == 0
            assert group["observation_indices"][-1] < sample["target_end_row"]
    late = cadence[5]
    assert late["replay_groups"][0]["decision_indices"] == list(range(0, 46, 5))
    assert late["frame_ids"] == [10 + row * 3 for row in range(48)]
    assert late["future_frame_ids"][-1].tolist() == [10 + 49 * 3, 10 + 51 * 3]
    assert all(fid > late["frame_ids"][-1] for fid in late["future_frame_ids"][-1].tolist())


@pytest.mark.parametrize("dataset_class", [NativeTrainingDataset, PredictiveTrainingDataset])
def test_cadence_contract_opt_in_and_default_compatibility(episode_root, mock_norm_stats, dataset_class):
    kwargs = dict(root=episode_root, norm_stats=mock_norm_stats, augmentation=False,
                  stream_protocol="compact_memory_replay_v1")
    default = dataset_class(**kwargs)
    explicit_none = dataset_class(**kwargs, decision_stride=None)
    cadence = dataset_class(**kwargs, decision_stride=5)
    assert json.dumps(default.get_data_contract()) == json.dumps(explicit_none.get_data_contract())
    assert "decision_cadence" not in default.get_base_data_contract()
    assert default.get_data_contract()["stream_protocol"] == get_compact_protocol_contract()
    for idx in range(len(default)):
        expected, actual = default[idx], explicit_none[idx]
        assert expected.keys() == actual.keys()
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                assert torch.equal(expected[key], actual[key])
            else:
                assert expected[key] == actual[key]
    base = cadence.get_base_data_contract()
    contract = cadence.get_data_contract()
    assert base["data_fingerprint"] != default.get_base_data_contract()["data_fingerprint"]
    assert contract["data_fingerprint"] != default.get_data_contract()["data_fingerprint"]
    spec = contract["stream_protocol"]["base_sampling_contract"]
    assert spec == base["decision_cadence"]
    assert spec["decision_stride"] == 5
    assert "all_episode_rows" in spec["observation_sampling"]
    assert "other_phases" in spec["phase_augmentation"]
    assert contract["data_fingerprint"] != dataset_class(**kwargs, decision_stride=4).get_data_contract()["data_fingerprint"]


@pytest.mark.parametrize("stride", [0, -1, True, 1.5, "5"])
def test_cadence_rejects_invalid_stride(episode_root, mock_norm_stats, stride):
    with pytest.raises(ValueError, match="decision_stride must be a positive integer"):
        PredictiveTrainingDataset(root=episode_root, norm_stats=mock_norm_stats, decision_stride=stride)


def test_cadence_requires_compact_protocol(episode_root, mock_norm_stats):
    with pytest.raises(ValueError, match="decision_stride requires"):
        NativeTrainingDataset(root=episode_root, norm_stats=mock_norm_stats, decision_stride=5)
