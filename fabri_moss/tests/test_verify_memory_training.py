"""Unit tests for fabri_moss.verify_memory_training probe and selection algorithms (CPU only)."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from fabri_moss.verify_memory_training import (
    build_parser,
    compute_structural_token_score,
    select_layout_segments,
)


class FakeDataset:
    """Lightweight in-memory dataset fixture for CPU metadata scan verification."""

    def __init__(
        self,
        segments: List[Tuple[int, int, int]],
        timestamps_by_ep: Dict[int, List[float]],
        frame_ids_by_ep: Dict[int, List[int]],
        seed: int = 42,
        epoch: int = 1,
        history_frames: int = 16,
        split: str = "train",
    ) -> None:
        self.segments = segments
        self.seed = seed
        self.epoch = epoch
        self.history_frames = history_frames
        self.split = split
        self._episode_by_id = {ep_idx: {"episode_index": ep_idx} for ep_idx in timestamps_by_ep}
        self._timestamps_by_ep = timestamps_by_ep
        self._frame_ids_by_ep = frame_ids_by_ep

    def _get_episode_dataframe(self, ep: Dict[str, Any]) -> Any:
        import pandas as pd

        ep_idx = ep["episode_index"]
        return pd.DataFrame({
            "frame_index": self._frame_ids_by_ep[ep_idx],
            "timestamp": self._timestamps_by_ep[ep_idx],
        })

    def _validate_and_get_timestamps(self, df: Any, ep_idx: int) -> Tuple[List[float], float]:
        return list(df["timestamp"].values), 0.1


def test_import_no_side_effects() -> None:
    """Verify importing verify_memory_training is side-effect free and parser works."""
    import fabri_moss.verify_memory_training as vmt

    assert callable(vmt.select_layout_segments)
    assert callable(vmt.run_probe)
    assert callable(vmt.build_parser)

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--output-dir" in help_text
    assert "--checkpoint" in help_text
    assert "--data-root" in help_text


def test_structural_token_score_formula() -> None:
    """Verify structural token score formula explicitly excludes text headers."""
    # Score = recent_frames_count * 1024 + retired_anchor_count * 256 + retired_summary_bin_count * 16
    queries = [
        {
            "recent_frames_count": 4,
            "retired_anchor_count": 0,
            "retired_summary_bin_count": 1,
        },
        {
            "recent_frames_count": 4,
            "retired_anchor_count": 3,
            "retired_summary_bin_count": 2,
        },
    ]
    # Query 0: 4 * 1024 + 0 * 256 + 1 * 16 = 4096 + 16 = 4112
    # Query 1: 4 * 1024 + 3 * 256 + 2 * 16 = 4096 + 768 + 32 = 4896
    max_score = compute_structural_token_score(queries)
    assert max_score == 4896

    # Empty queries return 0
    assert compute_structural_token_score([]) == 0


def test_dense_and_current_only_priority() -> None:
    """Verify scan prioritizes target count 8 and dense obs 16 for dense/current_only."""
    segments = [
        (0, 0, 4),    # seg 0: dense, target_count=4, obs=10
        (0, 10, 18),  # seg 1: dense, target_count=8, obs=12
        (0, 20, 28),  # seg 2: dense, target_count=8, obs=16 (optimal!)
        (0, 30, 34),  # seg 3: current_only, target_count=4, obs=4
        (0, 40, 48),  # seg 4: current_only, target_count=8, obs=8 (optimal!)
        (0, 50, 58),  # seg 5: memory, target_count=8, obs=80
    ]
    timestamps = [float(i) * 0.1 for i in range(100)]
    frame_ids = list(range(100))
    dataset = FakeDataset(segments, {0: timestamps}, {0: frame_ids})

    layouts = [
        # seg 0: dense, M=4, N=10
        {"stream_layout": {"mode": "dense", "num_observations": 10}},
        # seg 1: dense, M=8, N=12
        {"stream_layout": {"mode": "dense", "num_observations": 12}},
        # seg 2: dense, M=8, N=16 (ideal)
        {"stream_layout": {"mode": "dense", "num_observations": 16}},
        # seg 3: current_only, M=4, N=4
        {"stream_layout": {"mode": "current_only", "num_observations": 4}},
        # seg 4: current_only, M=8, N=8 (ideal)
        {"stream_layout": {"mode": "current_only", "num_observations": 8}},
        # seg 5: memory, M=8, N=80
        {
            "stream_layout": {
                "mode": "memory",
                "num_observations": 80,
                "calendar_queries": [
                    {"recent_frames_count": 4, "retired_anchor_count": 2, "retired_summary_bin_count": 1}
                ],
            }
        },
    ]

    def mock_layout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        tgt_start = kwargs.get("target_start_row", args[3] if len(args) > 3 else 0)
        tgt_end = kwargs.get("target_end_row", args[4] if len(args) > 4 else 0)
        for idx, (ep, s, e) in enumerate(segments):
            if s == tgt_start and e == tgt_end:
                return layouts[idx]
        return layouts[0]

    with patch("fabri_moss.verify_memory_training.compute_memory_replay_layout", side_effect=mock_layout):
        selected = select_layout_segments(dataset)

    # dense should prefer seg 2 (target_count=8 and obs=16) over seg 0 (target_count=4) and seg 1 (obs=12)
    assert selected["dense"][0] == 2
    # current_only should prefer seg 4 (target_count=8) over seg 3 (target_count=4)
    assert selected["current_only"][0] == 4
    # memory and memory_context selected
    assert selected["memory"][0] == 5
    assert selected["memory_context"][0] == 5


def test_max_n_and_max_structural_select_different_samples() -> None:
    """Verify maxN memory and max structural score memory select different samples when distinct."""
    segments = [
        (0, 0, 8),    # seg 0: dense, target_count=8, obs=16
        (0, 8, 16),   # seg 1: current_only, target_count=8, obs=8
        (0, 20, 28),  # seg 2: memory A -> high N (85), low structural score (4112)
        (0, 30, 38),  # seg 3: memory B -> moderate N (75 > 72), high structural score (6224)
    ]
    timestamps = [float(i) * 0.1 for i in range(120)]
    frame_ids = list(range(120))
    dataset = FakeDataset(segments, {0: timestamps}, {0: frame_ids})

    layouts = [
        {"stream_layout": {"mode": "dense", "num_observations": 16}},
        {"stream_layout": {"mode": "current_only", "num_observations": 8}},
        {
            "stream_layout": {
                "mode": "memory",
                "num_observations": 85,  # Higher N
                "calendar_queries": [
                    {"recent_frames_count": 4, "retired_anchor_count": 0, "retired_summary_bin_count": 1}
                    # 4 * 1024 + 0 * 256 + 1 * 16 = 4112
                ],
            }
        },
        {
            "stream_layout": {
                "mode": "memory",
                "num_observations": 75,  # > 72, but lower N than seg 2
                "calendar_queries": [
                    {"recent_frames_count": 4, "retired_anchor_count": 8, "retired_summary_bin_count": 5}
                    # 4 * 1024 + 8 * 256 + 5 * 16 = 4096 + 2048 + 80 = 6224 (higher structural pressure!)
                ],
            }
        },
    ]

    def mock_layout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        tgt_start = kwargs.get("target_start_row", args[3] if len(args) > 3 else 0)
        tgt_end = kwargs.get("target_end_row", args[4] if len(args) > 4 else 0)
        for idx, (ep, s, e) in enumerate(segments):
            if s == tgt_start and e == tgt_end:
                return layouts[idx]
        return layouts[0]

    with patch("fabri_moss.verify_memory_training.compute_memory_replay_layout", side_effect=mock_layout):
        selected = select_layout_segments(dataset)

    # memory (max N) should pick seg 2
    assert selected["memory"][0] == 2
    assert selected["memory"][1]["stream_layout"]["num_observations"] == 85

    # memory_context should pick seg 3
    assert selected["memory_context"][0] == 3
    assert selected["memory_context"][1]["stream_layout"]["structural_token_score"] == 6224

    # They must be different segments
    assert selected["memory"][0] != selected["memory_context"][0]


def test_max_n_and_max_structural_same_sample_handled() -> None:
    """Verify that when max N and max structural score share the same sample, both are returned."""
    segments = [
        (0, 0, 8),    # seg 0: dense
        (0, 8, 16),   # seg 1: current_only
        (0, 20, 28),  # seg 2: memory A -> high N (85) AND high score (6224)
        (0, 30, 38),  # seg 3: memory B -> lower N (75), lower score (4112)
    ]
    timestamps = [float(i) * 0.1 for i in range(120)]
    frame_ids = list(range(120))
    dataset = FakeDataset(segments, {0: timestamps}, {0: frame_ids})

    layouts = [
        {"stream_layout": {"mode": "dense", "num_observations": 16}},
        {"stream_layout": {"mode": "current_only", "num_observations": 8}},
        {
            "stream_layout": {
                "mode": "memory",
                "num_observations": 85,
                "calendar_queries": [
                    {"recent_frames_count": 4, "retired_anchor_count": 8, "retired_summary_bin_count": 5}
                ],
            }
        },
        {
            "stream_layout": {
                "mode": "memory",
                "num_observations": 75,
                "calendar_queries": [
                    {"recent_frames_count": 4, "retired_anchor_count": 0, "retired_summary_bin_count": 1}
                ],
            }
        },
    ]

    def mock_layout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        tgt_start = kwargs.get("target_start_row", args[3] if len(args) > 3 else 0)
        tgt_end = kwargs.get("target_end_row", args[4] if len(args) > 4 else 0)
        for idx, (ep, s, e) in enumerate(segments):
            if s == tgt_start and e == tgt_end:
                return layouts[idx]
        return layouts[0]

    with patch("fabri_moss.verify_memory_training.compute_memory_replay_layout", side_effect=mock_layout):
        selected = select_layout_segments(dataset)

    assert selected["memory"][0] == 2
    assert selected["memory_context"][0] == 2
    assert len(selected) == 4


def test_scan_raises_on_max_n_under_threshold() -> None:
    """Verify scan raises RuntimeError when max memory observation count <= 72."""
    segments = [
        (0, 0, 8),    # dense
        (0, 8, 16),   # current_only
        (0, 20, 28),  # memory with N=50 <= 72
    ]
    timestamps = [float(i) * 0.1 for i in range(50)]
    frame_ids = list(range(50))
    dataset = FakeDataset(segments, {0: timestamps}, {0: frame_ids})

    layouts = [
        {"stream_layout": {"mode": "dense", "num_observations": 16}},
        {"stream_layout": {"mode": "current_only", "num_observations": 8}},
        {
            "stream_layout": {
                "mode": "memory",
                "num_observations": 50,
                "calendar_queries": [
                    {"recent_frames_count": 4, "retired_anchor_count": 1, "retired_summary_bin_count": 1}
                ],
            }
        },
    ]

    def mock_layout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        tgt_start = kwargs.get("target_start_row", args[3] if len(args) > 3 else 0)
        tgt_end = kwargs.get("target_end_row", args[4] if len(args) > 4 else 0)
        for idx, (ep, s, e) in enumerate(segments):
            if s == tgt_start and e == tgt_end:
                return layouts[idx]
        return layouts[0]

    with patch("fabri_moss.verify_memory_training.compute_memory_replay_layout", side_effect=mock_layout):
        with pytest.raises(RuntimeError, match="Maximum memory observations found was 50 <= 72"):
            select_layout_segments(dataset)
