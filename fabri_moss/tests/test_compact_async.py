"""Integration and async pipeline tests for compact intermediate memory with temporal RoPE."""

from __future__ import annotations

import threading
import time
from typing import Any
import pytest
import torch

from fabri_moss.async_pipeline import AsyncVisualPlanner, Observation
from fabri_moss.compact_cache import (
    CompactKVState,
    validate_compact_memory,
)
from fabri_moss.evaluate_async import build_parser, main
from fabri_moss.native_async import make_native_cache_callbacks
from fabri_moss.tests.test_compact_cache import make_tiny_compact_adapter


def make_obs(frame_id: int, state_val: float = 0.0) -> Observation:
    """Create a valid deterministic CPU observation matching FabriVLA contract."""
    return Observation(
        frame_id=frame_id,
        capture_time=float(frame_id),
        images=[torch.randn(4, 16)],
        state=torch.full((1, 4), state_val),
        state_mask=torch.ones((1, 4), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
        observation_time=float(frame_id) * 0.1,
    )


def test_compact_async_stream_and_decision_rebuild() -> None:
    """Test streaming multiple ready frames per snapshot, decision protection, and background rebuild."""
    adapter = make_tiny_compact_adapter(
        max_frames=16,
        shallow_layer=1,
        recent_frames=2,
        consolidate_every=2,
        intermediate_grid=1,
    )
    prompt = "reach red cup"

    captured: dict[str, Any] = {}
    orig_sample = adapter.policy.action_head.sample

    def spy_sample(deep: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        captured["deep_shape"] = deep.shape
        captured["state"] = kwargs.get("state")
        captured["shallow"] = kwargs.get("shallow_tokens")
        return orig_sample(deep, *args, **kwargs)

    adapter.policy.action_head.sample = spy_sample

    encode_fn, plan_fn, val_fn = make_native_cache_callbacks(adapter, prompt=prompt)
    planner = AsyncVisualPlanner(
        encode=encode_fn,
        plan=plan_fn,
        validate=val_fn,
        max_frames=None,
        stateful=True,
        memory_validator=validate_compact_memory,
        overflow_policy="error",
    )

    try:
        planner.reset("ep1", prompt)

        # Snapshot 1: Submit 3 frames (0, 1, 2)
        for i in range(3):
            planner.submit(make_obs(i, state_val=float(i)))

        assert planner.wait_ready(min_frames=3, timeout=10.0)
        assert planner.request_plan()
        plan1 = planner.wait_plan(timeout=10.0)
        assert plan1 is not None

        # Verify spy captured full 16-token deep representation of the decision frame (frame 2)
        assert captured["deep_shape"] == (1, 16, 64)
        assert captured["shallow"] is not None and captured["shallow"].shape == (1, 16, 64)
        assert torch.allclose(captured["state"], torch.full((1, 4), 2.0))

        mem1 = planner._memory
        assert isinstance(mem1, CompactKVState)
        assert mem1.frame_count == 3
        assert mem1.last_frame_id == 2
        assert len(mem1.memory.recent) == 3
        # In snapshot 1, frames 0 and 1 are non-decision, frame 2 is decision
        assert not mem1.memory.recent[0].is_decision
        assert not mem1.memory.recent[1].is_decision
        assert mem1.memory.recent[2].is_decision
        # len(recent) is 3 < R+K (2+2=4), so no rebuild pending yet
        assert mem1.pending is None

        # Snapshot 2: Submit 3 more frames (3, 4, 5)
        # Total recent will be 3 + 3 = 6 >= 4, which triggers rebuild
        for i in range(3, 6):
            planner.submit(make_obs(i, state_val=float(i)))

        assert planner.wait_ready(min_frames=3, timeout=10.0)
        assert planner.request_plan()
        plan2 = planner.wait_plan(timeout=10.0)
        assert plan2 is not None

        mem2 = planner._memory
        assert isinstance(mem2, CompactKVState)
        assert mem2.frame_count == 6
        assert mem2.last_frame_id == 5
        assert mem2.memory.recent[-1].is_decision
        assert mem2.pending is not None  # Rebuild triggered!

        # Snapshot 3: Submit 2 more frames (6, 7)
        # Next query will resolve the pending rebuild before appending frames 6 and 7
        for i in range(6, 8):
            planner.submit(make_obs(i, state_val=float(i)))

        assert planner.wait_ready(min_frames=2, timeout=10.0)
        assert planner.request_plan()
        plan3 = planner.wait_plan(timeout=10.0)
        assert plan3 is not None

        mem3 = planner._memory
        assert isinstance(mem3, CompactKVState)
        assert mem3.frame_count == 8
        assert mem3.last_frame_id == 7
        assert mem3.rebuild_count > 0  # Rebuild applied
        # PlanResult returned to caller should not expose internal pending futures or cache
        assert plan3.computation.next_memory is None
    finally:
        planner.close()
        adapter.close()


def test_compact_failure_isolation_retry_and_reset() -> None:
    """Test retry of the same reservation doesn't corrupt old state, and reset isolates futures."""
    adapter = make_tiny_compact_adapter(
        max_frames=16,
        shallow_layer=1,
        recent_frames=2,
        consolidate_every=2,
        intermediate_grid=1,
    )
    prompt = "retry task"
    encode_fn, raw_plan, val_fn = make_native_cache_callbacks(adapter, prompt=prompt)

    fail_after_pending = [False]
    observed_pending = threading.Event()

    def flaky_plan(*args: Any, **kwargs: Any) -> Any:
        comp = raw_plan(*args, **kwargs)
        if fail_after_pending[0]:
            fail_after_pending[0] = False
            # Verify failure occurs strictly AFTER raw_plan produces candidate with pending rebuild
            assert comp.next_memory is not None
            assert comp.next_memory.pending is not None
            observed_pending.set()
            raise RuntimeError("simulated compact planner failure after pending rebuild creation")
        return comp

    planner = AsyncVisualPlanner(
        encode=encode_fn,
        plan=flaky_plan,
        validate=val_fn,
        max_frames=None,
        stateful=True,
        memory_validator=validate_compact_memory,
        overflow_policy="error",
    )

    try:
        planner.reset("ep1", prompt)

        # Snapshot 1: Submit 2 frames
        for i in range(2):
            planner.submit(make_obs(i))
        assert planner.wait_ready(min_frames=2, timeout=10.0)
        assert planner.request_plan()
        plan1 = planner.wait_plan(timeout=10.0)
        assert plan1 is not None
        mem_committed = planner._memory
        assert mem_committed is not None and mem_committed.frame_count == 2
        assert mem_committed.pending is None
        cached_kv_copy = mem_committed.layer_kv[0][0].clone()

        # Snapshot 2: Submit 2 frames (2, 3) -> recent len = 2 + 2 = 4 >= R + K -> triggers rebuild
        for i in range(2, 4):
            planner.submit(make_obs(i))
        assert planner.wait_ready(min_frames=2, timeout=10.0)
        fail_after_pending[0] = True
        assert planner.request_plan()

        with pytest.raises(RuntimeError, match="simulated compact planner failure after pending rebuild creation"):
            planner.wait_plan(timeout=10.0)

        assert observed_pending.is_set(), "Failure did not occur after pending rebuild was generated"

        # Committed state must remain completely uncorrupted (no double-write, same memory instance)
        assert planner._memory is mem_committed
        assert planner._memory.frame_count == 2
        assert planner._memory.pending is None
        assert torch.equal(planner._memory.layer_kv[0][0], cached_kv_copy)

        # Retry with identical reservation
        assert planner.request_plan(retry=True)
        plan_retry = planner.wait_plan(timeout=10.0)
        assert plan_retry is not None
        assert planner._memory.frame_count == 4
        assert planner._memory.last_frame_id == 3
        assert planner._memory.pending is not None

        # Episode reset isolates previous episode token and resets frame0 / time0 state
        planner.reset("ep2", prompt)
        assert planner._memory is None
        assert planner.stats()["consumed_count"] == 0

        # Episode 2 frame 0 / time 0 isolation
        planner.submit(make_obs(0))
        assert planner.wait_ready(min_frames=1, timeout=10.0)
        assert planner.request_plan()
        plan_ep2 = planner.wait_plan(timeout=10.0)
        assert plan_ep2 is not None
        assert planner._memory.frame_count == 1
        assert planner._memory.last_frame_id == 0
        assert planner._memory.memory.recent[0].observation_time == 0.0
    finally:
        planner.close()
        adapter.close()


def test_compact_cli_parser_and_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Test CLI argument parsing and validation for --native-memory compact-temporal."""
    out_dir = tmp_path / "cli_test"

    # 1. compact-temporal requires --mode=native-cache
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_async",
            "--mode",
            "moss",
            "--adapter",
            "a.pt",
            "--output-dir",
            str(out_dir),
            "--native-memory",
            "compact-temporal",
        ],
    )
    with pytest.raises(ValueError, match="only allowed when --mode=native-cache"):
        main()

    # 2. compact-temporal requires --timestamp-mode=text
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_async",
            "--mode",
            "native-cache",
            "--output-dir",
            str(out_dir),
            "--timestamp-mode",
            "none",
            "--native-memory",
            "compact-temporal",
        ],
    )
    with pytest.raises(ValueError, match="requires --timestamp-mode=text"):
        main()

    # 3. Valid CLI options parse correctly
    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "native-cache",
            "--output-dir",
            str(out_dir),
            "--timestamp-mode",
            "text",
            "--native-memory",
            "compact-temporal",
        ]
    )
    assert args.mode == "native-cache"
    assert args.timestamp_mode == "text"
    assert args.native_memory == "compact-temporal"
