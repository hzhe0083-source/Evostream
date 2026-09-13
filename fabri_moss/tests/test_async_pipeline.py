"""Deterministic concurrency and behavior unit tests for fabri_moss.async_pipeline."""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, List, Optional, Tuple
import weakref

import pytest
import torch
import torch.nn as nn

from fabri_moss.async_pipeline import (
    AsyncVisualPlanner,
    ConsumedFrame,
    EncodedFrame,
    Observation,
    ObservationMetadata,
    PlanComputation,
    PlanResult,
    make_moss_callbacks,
)
from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.tests.test_core import TinyFabriVLAPolicy


class TinyActionExpertWithSample(nn.Module):
    def __init__(self, hidden_size: int = 128, action_dim: int = 4, horizon: int = 10):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.proj = nn.Linear(hidden_size, horizon * action_dim)

    def sample(
        self,
        deep_tokens: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        shallow_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B = deep_tokens.shape[0]
        rep = deep_tokens.mean(dim=1) + shallow_tokens.mean(dim=1)
        out = self.proj(rep).view(B, self.horizon, self.action_dim)
        return out


def make_obs(
    frame_id: int,
    capture_time: float = 0.0,
    state_val: float = 1.0,
    observation_time: Optional[float] = None,
) -> Observation:
    return Observation(
        frame_id=frame_id,
        capture_time=capture_time,
        images=[torch.zeros((1, 3, 16, 16))],
        state=torch.full((1, 10), state_val, dtype=torch.float32),
        state_mask=torch.ones((1, 10), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
        observation_time=observation_time,
    )


def test_vision_encodes_extra_frames_while_planner_blocked():
    planner_entered = threading.Event()
    planner_release = threading.Event()

    def fake_encode(obs: Observation) -> str:
        return f"payload_{obs.frame_id}"

    def fake_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        planner_entered.set()
        if not planner_release.wait(timeout=10.0):
            raise TimeoutError("Planner was not released in time")
        return PlanComputation(actions=torch.zeros((1, 10, 4)))

    try:
        with AsyncVisualPlanner(fake_encode, fake_plan, max_frames=5, max_pending=8) as planner:
            planner.reset(episode_id="ep0", prompt="reach goal")

            planner.submit(make_obs(0, capture_time=0.0))
            planner.submit(make_obs(1, capture_time=0.1))

            assert planner.wait_ready(min_frames=2, timeout=5.0)
            assert planner.request_plan()

            assert planner_entered.wait(timeout=5.0)
            assert planner.stats()["planner_busy"] is True

            planner.submit(make_obs(2, capture_time=0.2))
            planner.submit(make_obs(3, capture_time=0.3))
            planner.submit(make_obs(4, capture_time=0.4))

            assert planner.wait_ready(min_frames=3, timeout=5.0)

            st = planner.stats()
            assert st["planner_busy"] is True
            assert st["ready_frame_ids"] == [2, 3, 4]

            planner_release.set()

            res = planner.wait_plan(timeout=5.0)
            assert res is not None
            assert res.source_frame_id == 1
            assert [f.observation.frame_id for f in res.frames] == [0, 1]
            assert all(not hasattr(f, "payload") for f in res.frames)
            assert all(not hasattr(f.observation, "images") for f in res.frames)
            assert all(not hasattr(f.observation, "state") for f in res.frames)
            assert res.computation.actions.shape == (1, 10, 4)

            st_after = planner.stats()
            assert st_after["last_consumed_id"] == 1
            assert st_after["consumed_count"] == 2
            assert st_after["ready_frame_ids"] == [2, 3, 4]
            assert st_after["plans_completed"] == 1
    finally:
        planner_release.set()


def test_retry_failure_preserves_snapshot_and_no_consume_on_failure():
    attempts = 0
    fail_first = True

    def fake_encode(obs: Observation) -> str:
        return f"payload_{obs.frame_id}"

    def fake_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        nonlocal attempts
        attempts += 1
        if fail_first and attempts == 1:
            raise RuntimeError("Temporary model computation error")
        return PlanComputation(actions=torch.ones((1, 10, 4)))

    with AsyncVisualPlanner(fake_encode, fake_plan, max_frames=5) as planner:
        planner.reset(episode_id="ep1", prompt="pick cup")
        planner.submit(make_obs(0, capture_time=0.0))
        planner.submit(make_obs(1, capture_time=0.1))

        assert planner.wait_ready(min_frames=2, timeout=5.0)
        assert planner.request_plan()

        with pytest.raises(RuntimeError, match="Temporary model computation error"):
            planner.wait_plan(timeout=5.0)

        st = planner.stats()
        assert st["last_consumed_id"] == -1
        assert st["consumed_count"] == 0
        assert st["errors_count"] == 1

        with pytest.raises(RuntimeError, match="Planner error pending retry"):
            planner.request_plan(retry=False)

        planner.submit(make_obs(2, capture_time=0.2))
        assert planner.wait_ready(min_frames=1, timeout=5.0)

        assert planner.request_plan(retry=True)
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert [f.observation.frame_id for f in res.frames] == [0, 1]
        assert res.source_frame_id == 1

        st_after = planner.stats()
        assert st_after["last_consumed_id"] == 1
        assert st_after["ready_frame_ids"] == [2]


def test_pending_and_ready_overflow_drops_oldest():
    encode_gate = threading.Event()

    def blocked_encode(obs: Observation) -> str:
        if not encode_gate.wait(timeout=10.0):
            raise TimeoutError("Encode gate wait timed out")
        return f"p_{obs.frame_id}"

    def fake_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        return PlanComputation(actions=torch.zeros((1, 10, 4)))

    try:
        with AsyncVisualPlanner(blocked_encode, fake_plan, max_frames=3, max_pending=3) as planner:
            planner.reset(episode_id="ep_overflow", prompt="test overflow")

            planner.submit(make_obs(0))
            while not planner.stats()["vision_busy"]:
                pass

            for i in range(1, 6):
                planner.submit(make_obs(i))

            st = planner.stats()
            assert st["dropped_pending"] == 2
            assert st["pending_count"] == 3

            encode_gate.set()

            assert planner.wait_ready(min_frames=3, timeout=5.0)
            while planner.stats()["pending_count"] > 0:
                pass

            st_ready = planner.stats()
            assert st_ready["dropped_pending"] >= 2
            assert len(st_ready["ready_frame_ids"]) == 3
            assert st_ready["dropped_ready"] >= 1
            assert st_ready["ready_frame_ids"][-1] == 5
    finally:
        encode_gate.set()


def test_reset_during_vision_and_planner_prevents_contamination():
    vision_entered = threading.Event()
    vision_release = threading.Event()
    planner_entered = threading.Event()
    planner_release = threading.Event()

    def gating_encode(obs: Observation) -> str:
        if obs.frame_id == 0:
            vision_entered.set()
            if not vision_release.wait(timeout=10.0):
                raise TimeoutError("Vision release timed out")
        return f"enc_{obs.frame_id}"

    def gating_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        planner_entered.set()
        if not planner_release.wait(timeout=10.0):
            raise TimeoutError("Planner release timed out")
        return PlanComputation(actions=torch.zeros((1, 10, 4)))

    try:
        with AsyncVisualPlanner(gating_encode, gating_plan, max_frames=5) as planner:
            planner.reset(episode_id="ep_old", prompt="old prompt")

            planner.submit(make_obs(0))
            assert vision_entered.wait(timeout=5.0)

            planner.reset(episode_id="ep_new", prompt="new prompt")
            vision_release.set()

            st = planner.stats()
            assert st["generation"] == 2
            assert st["episode_id"] == "ep_new"
            assert st["ready_frame_ids"] == []

            planner.submit(make_obs(0))
            planner.submit(make_obs(1))
            assert planner.wait_ready(min_frames=2, timeout=5.0)

            assert planner.request_plan()
            assert planner_entered.wait(timeout=5.0)

            planner.reset(episode_id="ep_new2", prompt="new2 prompt")
            planner_release.set()

            assert planner.poll_plan() is None
            st2 = planner.stats()
            assert st2["generation"] == 3
            assert st2["plans_completed"] == 0
    finally:
        vision_release.set()
        planner_release.set()


def test_cannot_request_plan_while_planner_running():
    gate = threading.Event()

    def slow_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        if not gate.wait(timeout=10.0):
            raise TimeoutError("Gate wait timed out")
        return PlanComputation(actions=torch.zeros((1, 10, 4)))

    try:
        with AsyncVisualPlanner(lambda obs: "p", slow_plan, max_frames=5) as planner:
            planner.reset(episode_id="ep", prompt="prompt")
            planner.submit(make_obs(0))
            assert planner.wait_ready(min_frames=1, timeout=5.0)

            assert planner.request_plan() is True
            assert planner.request_plan() is False

            gate.set()
            res = planner.wait_plan(timeout=5.0)
            assert res is not None

            planner.submit(make_obs(1))
            assert planner.wait_ready(min_frames=1, timeout=5.0)
            assert planner.request_plan() is True

            while planner.stats()["plans_completed"] < 2:
                pass
            assert planner.request_plan() is False
            popped = planner.poll_plan()
            assert popped is not None
    finally:
        gate.set()


def test_submit_isolates_observation_tensors_from_caller_mutation():
    captured_obs: List[Observation] = []

    def mock_encode(obs: Observation) -> str:
        captured_obs.append(obs)
        return "ok"

    with AsyncVisualPlanner(mock_encode, lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1)))) as planner:
        planner.reset(episode_id="ep", prompt="prompt")

        raw_state = torch.ones((1, 5), dtype=torch.float32)
        raw_mask = torch.ones((1, 5), dtype=torch.bool)
        raw_action_mask = torch.ones((1, 4), dtype=torch.bool)
        img_tensor = torch.zeros((1, 3, 8, 8))

        obs = Observation(
            frame_id=0,
            capture_time=1.0,
            images=[img_tensor],
            state=raw_state,
            state_mask=raw_mask,
            action_mask=raw_action_mask,
        )
        planner.submit(obs)

        raw_state.zero_()
        raw_mask.zero_()
        raw_action_mask.zero_()
        img_tensor.fill_(99.0)

        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert len(captured_obs) == 1
        stored_obs = captured_obs[0]
        assert (stored_obs.state == 1.0).all()
        assert stored_obs.state_mask.all()
        assert stored_obs.action_mask.all()
        assert (stored_obs.images[0] == 0.0).all()


def test_make_moss_callbacks_with_tiny_moss():
    policy = TinyFabriVLAPolicy(hidden_size=128, num_layers=6)
    policy.action_head = TinyActionExpertWithSample(hidden_size=128, action_dim=4, horizon=10)
    config = MossConfig(cross_layers=(2, 4, 6), num_readout_tokens=8, max_frames=2, shallow_layer=4)
    moss = MossInternVL(policy, config=config)
    moss.eval()

    encode_fn_temp, _, validate_fn_temp = make_moss_callbacks(moss)
    validate_fn_temp()

    moss.train()
    with pytest.raises(RuntimeError, match="must be in eval mode"):
        validate_fn_temp()
    moss.eval()

    encode_fn, plan_fn, validate_fn = make_moss_callbacks(moss)

    with AsyncVisualPlanner(encode_fn, plan_fn, max_frames=2, validate=validate_fn) as planner:
        planner.reset(episode_id="ep_real", prompt="move arm")

        obs0 = Observation(
            frame_id=0,
            capture_time=0.0,
            images=[torch.randn(1, 64)],
            state=torch.randn(1, 10),
            state_mask=torch.ones((1, 10), dtype=torch.bool),
            action_mask=torch.ones((1, 4), dtype=torch.bool),
        )
        planner.submit(obs0)
        assert planner.wait_ready(min_frames=1, timeout=10.0)

        assert planner.request_plan()
        res = planner.wait_plan(timeout=10.0)
        assert res is not None
        assert res.source_frame_id == 0
        assert res.computation.actions.shape == (1, 10, 4)
        assert res.computation.deep is not None
        assert res.computation.shallow is not None


def test_close_cleanly_terminates_threads():
    planner = AsyncVisualPlanner(lambda obs: "p", lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))))
    planner.reset(episode_id="ep", prompt="prompt")
    planner.submit(make_obs(0))
    planner.close(timeout=5.0)
    assert not planner._vision_thread.is_alive()
    assert not planner._planner_thread.is_alive()


def test_frame_id_strict_monotonicity_and_validation():
    planner = AsyncVisualPlanner(lambda obs: "p", lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))))
    planner.reset(episode_id="ep", prompt="prompt")

    planner.submit(make_obs(5))

    with pytest.raises(ValueError, match="strictly greater"):
        planner.submit(make_obs(5))

    with pytest.raises(ValueError, match="strictly greater"):
        planner.submit(make_obs(4))

    with pytest.raises(ValueError, match="contains non-finite values"):
        Observation(
            frame_id=6,
            capture_time=0.0,
            images=[torch.zeros((1, 3, 4, 4))],
            state=torch.tensor([[float("nan")]]),
            state_mask=torch.ones((1, 1), dtype=torch.bool),
            action_mask=torch.ones((1, 1), dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="capture_time must be finite non-negative float"):
        Observation(
            frame_id=7,
            capture_time=True,  # bool should be rejected
            images=[torch.zeros((1, 3, 4, 4))],
            state=torch.ones((1, 1)),
            state_mask=torch.ones((1, 1), dtype=torch.bool),
            action_mask=torch.ones((1, 1), dtype=torch.bool),
        )

    planner.close()


def test_fatal_vision_error_propagation_and_poll_closed():
    """Fatal vision error surfaces across all methods until reset."""
    def error_encode(obs: Observation):
        raise ValueError("Corrupt camera stream")

    with AsyncVisualPlanner(error_encode, lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1)))) as planner:
        planner.reset(episode_id="ep_fatal", prompt="p")
        planner.submit(make_obs(0))

        with pytest.raises(RuntimeError, match="Vision worker fatal error"):
            planner.wait_ready(timeout=5.0)

        with pytest.raises(RuntimeError, match="Vision worker fatal error"):
            planner.submit(make_obs(1))

        with pytest.raises(RuntimeError, match="Vision worker fatal error"):
            planner.request_plan()

        with pytest.raises(RuntimeError, match="Vision worker fatal error"):
            planner.poll_plan()

        # Reset clears fatal error
        planner.reset(episode_id="ep_recovered", prompt="p")
        assert planner.stats()["errors_count"] == 0

    # poll_plan on closed planner raises RuntimeError
    with pytest.raises(RuntimeError, match="AsyncVisualPlanner is closed"):
        planner.poll_plan()


def test_malformed_plan_computation_rejected_by_worker():
    """Worker catches and surfaces malformed or mutated NaN plan computation."""
    def bad_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        comp = PlanComputation(actions=torch.zeros((1, 10, 4)))
        comp.actions.fill_(float("nan"))  # Mutate after creation
        return comp

    with AsyncVisualPlanner(lambda obs: "p", bad_plan) as planner:
        planner.reset(episode_id="ep_bad", prompt="p")
        planner.submit(make_obs(0))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()

        with pytest.raises(RuntimeError, match="actions tensor contains non-finite values"):
            planner.wait_plan(timeout=5.0)


def test_weakref_consumed_frames_gc_after_result_released():
    """Consumed frames and payloads can be garbage collected once plan is consumed even while res is held."""
    class BigPayload:
        pass

    payload_ref = None

    def make_payload_encode(obs: Observation):
        nonlocal payload_ref
        p = BigPayload()
        payload_ref = weakref.ref(p)
        return p

    with AsyncVisualPlanner(make_payload_encode, lambda f, p: PlanComputation(actions=torch.zeros((1, 10, 4)))) as planner:
        planner.reset(episode_id="ep_gc", prompt="p")
        planner.submit(make_obs(0))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()

        res = planner.wait_plan(timeout=5.0)
        assert res is not None

        # After wait_plan, consumed frames are transformed to ConsumedFrame metadata in res.frames,
        # and reservation/snapshot in planner are cleared.
        gc.collect()
        assert payload_ref() is None

        # res still held, frame_ids still correct and metadata has no payload/images/state
        assert len(res.frames) == 1
        f0 = res.frames[0]
        assert isinstance(f0, ConsumedFrame)
        assert isinstance(f0.observation, ObservationMetadata)
        assert f0.observation.frame_id == 0
        assert not hasattr(f0, "payload")
        assert not hasattr(f0.observation, "images")
        assert not hasattr(f0.observation, "state")
        assert not hasattr(f0.observation, "state_mask")
        assert not hasattr(f0.observation, "action_mask")


def test_stateful_commit_and_rollback_on_retry():
    """Stateful AsyncVisualPlanner tests:
    - Candidate state returned by plan callback
    - Atomic commit on success
    - Retry does not commit failed candidate and reuses untouched old memory
    - Subsequent plan sees committed state
    - PlanResult.computation.next_memory is stripped (None)
    """
    from fabri_moss.delta import DeltaMemoryState

    attempt = 0

    def fake_encode(obs: Observation) -> str:
        return f"payload_{obs.frame_id}"

    def fake_stateful_plan(
        frames: Tuple[EncodedFrame, ...], prompt: str, previous_memory: Optional[DeltaMemoryState]
    ) -> PlanComputation:
        nonlocal attempt
        attempt += 1
        last_id = frames[-1].observation.frame_id
        prev_cnt = previous_memory.frame_count if previous_memory is not None else 0

        if attempt == 1:
            # First attempt fails after generating candidate
            raise RuntimeError("Transient network/GPU fail")

        # Create dummy candidate DeltaMemoryState
        cand_mat = (torch.full((1, 2, 8, 8), float(attempt)),)
        cand_state = DeltaMemoryState(
            matrices=cand_mat,
            last_frame_id=last_id,
            frame_count=prev_cnt + len(frames),
            prompt=prompt,
            owner=None,
            revision=1,
        )
        return PlanComputation(
            actions=torch.ones((1, 5, 4)),
            deep=None,
            shallow=None,
            next_memory=cand_state,
        )

    with AsyncVisualPlanner(
        fake_encode, fake_stateful_plan, max_frames=5, stateful=True
    ) as planner:
        planner.reset(episode_id="ep_stateful", prompt="turn knob")

        # Initial memory is None
        st0 = planner.stats()
        assert st0["memory_frame_count"] == 0
        assert st0["last_memory_frame_id"] == -1
        assert st0["memory_bytes"] == 0

        # Submit frames 0, 1
        planner.submit(make_obs(0))
        planner.submit(make_obs(1))
        assert planner.wait_ready(min_frames=2, timeout=5.0)

        # First plan request -> attempt 1 fails
        assert planner.request_plan()
        with pytest.raises(RuntimeError, match="Transient network/GPU fail"):
            planner.wait_plan(timeout=5.0)

        # On failure, memory remains None (no premature update)
        st_fail = planner.stats()
        assert st_fail["memory_frame_count"] == 0
        assert st_fail["last_memory_frame_id"] == -1
        assert st_fail["memory_bytes"] == 0

        # Retry -> attempt 2 succeeds with previous_memory=None
        assert planner.request_plan(retry=True)
        res1 = planner.wait_plan(timeout=5.0)
        assert res1 is not None
        assert res1.source_frame_id == 1
        # Result computation must have next_memory stripped to avoid holding state
        assert res1.computation.next_memory is None

        # Planner now has committed memory
        st1 = planner.stats()
        assert st1["memory_frame_count"] == 2
        assert st1["last_memory_frame_id"] == 1
        assert st1["memory_bytes"] > 0
        assert planner._memory is not None
        assert planner._memory.frame_count == 2
        assert planner._memory.last_frame_id == 1

        # Second decision: submit frame 2
        planner.submit(make_obs(2))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()

        res2 = planner.wait_plan(timeout=5.0)
        assert res2 is not None
        assert res2.source_frame_id == 2
        assert res2.computation.next_memory is None

        st2 = planner.stats()
        assert st2["memory_frame_count"] == 3
        assert st2["last_memory_frame_id"] == 2

        # Reset clears memory
        planner.reset(episode_id="ep_new", prompt="turn knob")
        st_reset = planner.stats()
        assert st_reset["memory_frame_count"] == 0
        assert st_reset["last_memory_frame_id"] == -1
        assert st_reset["memory_bytes"] == 0
        assert planner._memory is None


def test_reset_during_planner_discards_next_memory():
    """If reset occurs while planner is running, completed candidate next_memory is not committed."""
    from fabri_moss.delta import DeltaMemoryState

    entered = threading.Event()
    gate = threading.Event()

    def fake_encode(obs: Observation) -> str:
        return "p"

    def slow_plan(frames: Tuple[EncodedFrame, ...], prompt: str, prev_mem: Optional[Any]) -> PlanComputation:
        entered.set()
        gate.wait(timeout=5.0)
        cand_state = DeltaMemoryState(
            matrices=(torch.zeros((1, 2, 4, 4), dtype=torch.float32),),
            last_frame_id=frames[-1].observation.frame_id,
            frame_count=1,
            prompt=prompt,
            owner=None,
            revision=1,
        )
        return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=cand_state)

    try:
        with AsyncVisualPlanner(fake_encode, slow_plan, max_frames=2, stateful=True) as planner:
            planner.reset(episode_id="ep_race1", prompt="p1")
            planner.submit(make_obs(0))
            assert planner.wait_ready(min_frames=1, timeout=5.0)
            assert planner.request_plan()

            # Confirm worker truly entered callback before triggering reset
            assert entered.wait(timeout=5.0)

            # Trigger reset while worker is blocked in callback
            planner.reset(episode_id="ep_race2", prompt="p2")
            # Release worker thread
            gate.set()

            # Wait for planner to finish discarding or becoming idle
            with planner._cv:
                assert planner._cv.wait_for(lambda: not planner._planner_busy, timeout=5.0)

            # Memory should still be None because generation changed
            assert planner._memory is None
            st = planner.stats()
            assert st["memory_frame_count"] == 0
            assert st["generation"] == 2
            assert st["stale_outputs_discarded"] == 1
    finally:
        gate.set()


def test_close_clears_memory_and_releases_state():
    """Closing planner clears memory reference immediately."""
    from fabri_moss.delta import DeltaMemoryState

    cand_state = DeltaMemoryState(
        matrices=(torch.zeros((1, 2, 4, 4), dtype=torch.float32),),
        last_frame_id=0,
        frame_count=1,
        prompt="p",
        owner=None,
        revision=1,
    )

    planner = AsyncVisualPlanner(
        lambda o: "p",
        lambda f, p, m: PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=cand_state),
        stateful=True,
    )
    planner.reset(episode_id="ep_close", prompt="p")
    planner.submit(make_obs(0))
    planner.wait_ready(min_frames=1, timeout=5.0)
    planner.request_plan()
    res = planner.wait_plan(timeout=5.0)
    assert res is not None
    assert planner._memory is not None

    planner.close(timeout=5.0)
    assert planner._memory is None


def test_weakref_matrix_and_state_gc_reclaimable():
    """Delta memory matrices and state are fully reclaimable by GC after reset/close, even while holding PlanResult."""
    from fabri_moss.delta import DeltaMemoryState

    matrix_ref = None
    state_ref = None

    def plan_with_isolated_state(frames: Tuple[EncodedFrame, ...], prompt: str, prev_mem: Optional[Any]) -> PlanComputation:
        nonlocal matrix_ref, state_ref
        mat = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        matrix_ref = weakref.ref(mat)
        st = DeltaMemoryState(
            matrices=(mat,),
            last_frame_id=frames[-1].observation.frame_id,
            frame_count=1,
            prompt=prompt,
            owner=None,
            revision=1,
        )
        state_ref = weakref.ref(st)
        return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=st)

    with AsyncVisualPlanner(lambda o: "p", plan_with_isolated_state, stateful=True) as planner:
        planner.reset(episode_id="ep_gc_mem", prompt="p_gc")
        planner.submit(make_obs(0))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()

        # Hold PlanResult reference
        plan_result = planner.wait_plan(timeout=5.0)
        assert plan_result is not None
        assert plan_result.computation.next_memory is None

        # Reset clears memory in planner
        planner.reset(episode_id="ep_gc_next", prompt="p_gc2")

        # After reset, no strong references to candidate/committed state or matrices should remain,
        # even though plan_result is still held!
        gc.collect()
        assert matrix_ref() is None, "Delta memory matrix leaked after reset while holding PlanResult"
        assert state_ref() is None, "DeltaMemoryState leaked after reset while holding PlanResult"
        assert plan_result.source_frame_id == 0


def test_weakref_discarded_candidate_memory_gc_reclaimable():
    """Candidate next_memory discarded due to reset during planning is collected by GC."""
    from fabri_moss.delta import DeltaMemoryState

    entered = threading.Event()
    gate = threading.Event()
    matrix_ref = None
    state_ref = None

    def slow_plan(frames: Tuple[EncodedFrame, ...], prompt: str, prev_mem: Optional[Any]) -> PlanComputation:
        nonlocal matrix_ref, state_ref
        mat = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        matrix_ref = weakref.ref(mat)
        st = DeltaMemoryState(
            matrices=(mat,),
            last_frame_id=frames[-1].observation.frame_id,
            frame_count=1,
            prompt=prompt,
            owner=None,
            revision=1,
        )
        state_ref = weakref.ref(st)
        entered.set()
        gate.wait(timeout=5.0)
        return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=st)

    try:
        with AsyncVisualPlanner(lambda o: "p", slow_plan, stateful=True) as planner:
            planner.reset(episode_id="ep_discard1", prompt="p1")
            planner.submit(make_obs(0))
            assert planner.wait_ready(min_frames=1, timeout=5.0)
            assert planner.request_plan()

            assert entered.wait(timeout=5.0)
            planner.reset(episode_id="ep_discard2", prompt="p2")
            gate.set()

            with planner._cv:
                assert planner._cv.wait_for(lambda: not planner._planner_busy, timeout=5.0)

            gc.collect()
            assert matrix_ref() is None, "Discarded candidate matrix leaked after stale generation reset"
            assert state_ref() is None, "Discarded candidate state leaked after stale generation reset"
    finally:
        gate.set()


def test_stateful_validation_failures_surface_errors_and_allow_retry():
    """Candidate memory invalidations (None, future frame, wrong frame_count, nonfinite) cause error and do not commit."""
    from fabri_moss.delta import DeltaMemoryState

    mode = "valid"

    def dynamic_plan(frames: Tuple[EncodedFrame, ...], prompt: str, prev_mem: Optional[Any]) -> PlanComputation:
        last_id = frames[-1].observation.frame_id
        prev_cnt = prev_mem.frame_count if prev_mem is not None else 0

        if mode == "none":
            # Statefull planner returns None next_memory
            return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=None)
        elif mode == "future_frame":
            st = DeltaMemoryState(
                matrices=(torch.zeros((1, 2, 4, 4), dtype=torch.float32),),
                last_frame_id=last_id + 10,  # in future
                frame_count=prev_cnt + len(frames),
                prompt=prompt,
                owner=None,
                revision=1,
            )
            return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=st)
        elif mode == "wrong_count":
            st = DeltaMemoryState(
                matrices=(torch.zeros((1, 2, 4, 4), dtype=torch.float32),),
                last_frame_id=last_id,
                frame_count=prev_cnt + 999,  # wrong count
                prompt=prompt,
                owner=None,
                revision=1,
            )
            return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=st)
        elif mode == "nonfinite":
            bad_mat = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
            bad_mat[0, 0, 0, 0] = float("nan")
            st = DeltaMemoryState(
                matrices=(bad_mat,),
                last_frame_id=last_id,
                frame_count=prev_cnt + len(frames),
                prompt=prompt,
                owner=None,
                revision=1,
            )
            return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=st)
        else:
            st = DeltaMemoryState(
                matrices=(torch.zeros((1, 2, 4, 4), dtype=torch.float32),),
                last_frame_id=last_id,
                frame_count=prev_cnt + len(frames),
                prompt=prompt,
                owner=None,
                revision=1,
            )
            return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory=st)

    with AsyncVisualPlanner(lambda o: "p", dynamic_plan, stateful=True) as planner:
        # Case 1: mode == "none"
        planner.reset(episode_id="ep_val", prompt="p")
        planner.submit(make_obs(0))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        mode = "none"
        assert planner.request_plan()
        with pytest.raises(RuntimeError, match="next_memory must be DeltaMemoryState"):
            planner.wait_plan(timeout=5.0)
        assert planner._memory is None
        # Retry with valid succeeds
        mode = "valid"
        assert planner.request_plan(retry=True)
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert planner._memory is not None
        assert planner._memory.frame_count == 1

        # Case 2: mode == "future_frame"
        planner.submit(make_obs(1))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        mode = "future_frame"
        assert planner.request_plan()
        with pytest.raises(RuntimeError, match="next_memory last_frame_id must equal snapshot latest frame"):
            planner.wait_plan(timeout=5.0)
        # Memory unchanged from previous valid commit
        assert planner._memory.frame_count == 1
        assert planner._memory.last_frame_id == 0
        # Retry with valid
        mode = "valid"
        assert planner.request_plan(retry=True)
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert planner._memory.frame_count == 2
        assert planner._memory.last_frame_id == 1

        # Case 3: mode == "wrong_count"
        planner.submit(make_obs(2))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        mode = "wrong_count"
        assert planner.request_plan()
        with pytest.raises(RuntimeError, match="next_memory frame_count mismatch"):
            planner.wait_plan(timeout=5.0)
        assert planner._memory.frame_count == 2
        # Retry with valid
        mode = "valid"
        assert planner.request_plan(retry=True)
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert planner._memory.frame_count == 3

        # Case 4: mode == "nonfinite"
        planner.submit(make_obs(3))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        mode = "nonfinite"
        assert planner.request_plan()
        with pytest.raises(RuntimeError, match="contains non-finite values"):
            planner.wait_plan(timeout=5.0)
        assert planner._memory.frame_count == 3
        mode = "valid"
        assert planner.request_plan(retry=True)
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert planner._memory.frame_count == 4
        assert planner._memory.last_frame_id == 3


def test_custom_memory_validator_in_async_visual_planner():
    """Verify AsyncVisualPlanner uses custom memory_validator when provided."""
    custom_called = []

    def my_validator(cand, prev, snap, prompt):
        custom_called.append((cand, prev, len(snap), prompt))
        if cand == "BAD":
            raise ValueError("custom validation rejected")
        return f"validated_{cand}"

    def fake_plan(frames, prompt, prev_mem):
        return PlanComputation(actions=torch.zeros((1, 2, 2)), next_memory="candidate_state")

    with AsyncVisualPlanner(
        lambda o: "p",
        fake_plan,
        stateful=True,
        memory_validator=my_validator,
    ) as planner:
        planner.reset(episode_id="ep_custom_val", prompt="turn knob")
        planner.submit(make_obs(0))
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert len(custom_called) == 1
        assert custom_called[0][0] == "candidate_state"
        assert custom_called[0][1] is None
        assert custom_called[0][2] == 1
        assert custom_called[0][3] == "turn knob"
        assert planner._memory == "validated_candidate_state"


def test_dynamic_mode_all_ready_snapshots_arbitrary_frames():
    """Dynamic mode (max_frames=None, max_pending=None, max_ready=None) takes all ready frames (e.g. 13 and 23) without truncating."""
    with AsyncVisualPlanner(
        lambda obs: f"p_{obs.frame_id}",
        lambda frames, prompt: PlanComputation(actions=torch.zeros((1, 1, 4))),
        max_frames=None,
        max_pending=None,
        max_ready=None,
        overflow_policy="error",
    ) as planner:
        planner.reset(episode_id="ep_dyn1", prompt="test dynamic")
        st = planner.stats()
        assert st["snapshot_mode"] == "all_ready"
        assert st["max_ready"] is None
        assert st["max_pending"] is None
        assert st["overflow_policy"] == "error"

        # Submit 13 frames
        for i in range(13):
            planner.submit(make_obs(i, capture_time=i * 0.1))

        assert planner.wait_ready(min_frames=13, timeout=10.0)
        assert planner.request_plan()
        res = planner.wait_plan(timeout=10.0)
        assert res is not None
        assert len(res.frames) == 13
        assert [f.observation.frame_id for f in res.frames] == list(range(13))
        assert res.source_frame_id == 12

        # Submit 23 more frames
        for i in range(13, 13 + 23):
            planner.submit(make_obs(i, capture_time=i * 0.1))

        assert planner.wait_ready(min_frames=23, timeout=10.0)
        assert planner.request_plan()
        res2 = planner.wait_plan(timeout=10.0)
        assert res2 is not None
        assert len(res2.frames) == 23
        assert [f.observation.frame_id for f in res2.frames] == list(range(13, 36))
        assert res2.source_frame_id == 35


def test_dynamic_mode_planner_blocked_new_frames_held_for_next_round():
    """In dynamic mode, frames arriving while planner is blocked stay ready for the next round."""
    planner_entered = threading.Event()
    planner_release = threading.Event()

    def fake_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        planner_entered.set()
        if not planner_release.wait(timeout=10.0):
            raise TimeoutError("Planner release timed out")
        return PlanComputation(actions=torch.zeros((1, 2, 4)))

    try:
        with AsyncVisualPlanner(
            lambda obs: f"p_{obs.frame_id}",
            fake_plan,
            max_frames=None,
            max_pending=None,
            max_ready=None,
            overflow_policy="error",
        ) as planner:
            planner.reset(episode_id="ep_dyn_block", prompt="test block")

            for i in range(5):
                planner.submit(make_obs(i, capture_time=i * 0.1))
            assert planner.wait_ready(min_frames=5, timeout=5.0)
            assert planner.request_plan()
            assert planner_entered.wait(timeout=5.0)

            # Submit 8 new frames while planner is busy
            for i in range(5, 13):
                planner.submit(make_obs(i, capture_time=i * 0.1))
            assert planner.wait_ready(min_frames=8, timeout=5.0)

            # Release first plan
            planner_release.set()
            res1 = planner.wait_plan(timeout=5.0)
            assert res1 is not None
            assert len(res1.frames) == 5
            assert [f.observation.frame_id for f in res1.frames] == list(range(5))

            # The 8 frames are ready for the next round
            st = planner.stats()
            assert st["ready_frame_ids"] == list(range(5, 13))

            # Request second plan without new submits
            assert planner.request_plan()
            res2 = planner.wait_plan(timeout=5.0)
            assert res2 is not None
            assert len(res2.frames) == 8
            assert [f.observation.frame_id for f in res2.frames] == list(range(5, 13))
    finally:
        planner_release.set()


def test_finite_pending_error_policy_rejects_and_allows_retry_without_corrupting_state():
    """With finite max_pending and overflow_policy='error', submit raises BufferError when full,
    does not advance last_submitted_id or time anchor, and allows successful retry with same id."""
    vision_gate = threading.Event()

    def blocked_encode(obs: Observation) -> str:
        if not vision_gate.wait(timeout=10.0):
            raise TimeoutError("Vision gate timed out")
        return f"p_{obs.frame_id}"

    try:
        with AsyncVisualPlanner(
            blocked_encode,
            lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
            max_frames=None,
            max_pending=2,
            max_ready=None,
            overflow_policy="error",
        ) as planner:
            planner.reset(episode_id="ep_pen_err", prompt="test pending error")

            # Submit frame 0 -> vision worker picks it up
            planner.submit(make_obs(0, capture_time=100.0))
            with planner._cv:
                assert planner._cv.wait_for(lambda: planner._vision_busy, timeout=5.0)

            # Submit frames 1 and 2 to fill pending queue (capacity 2)
            planner.submit(make_obs(1, capture_time=100.1))
            planner.submit(make_obs(2, capture_time=100.2))

            # Frame 3 exceeds pending queue capacity -> raises BufferError
            with pytest.raises(BufferError, match="Pending queue full"):
                planner.submit(make_obs(3, capture_time=100.3))

            st = planner.stats()
            assert st["rejected_pending"] == 1
            assert st["last_submitted_id"] == 2
            assert st["last_observation_time"] == pytest.approx(0.2)
            assert st["episode_capture_origin"] == 100.0

            # Release vision gate to drain pending queue
            vision_gate.set()
            assert planner.wait_ready(min_frames=3, timeout=5.0)

            # Now submit frame 3 (same frame_id) succeeds!
            assert planner.submit(make_obs(3, capture_time=100.3))
            st2 = planner.stats()
            assert st2["last_submitted_id"] == 3
            assert st2["last_observation_time"] == pytest.approx(0.3)
            assert st2["episode_capture_origin"] == 100.0
    finally:
        vision_gate.set()


def test_finite_ready_error_policy_propagates_fatal_and_exits_cleanly():
    """With finite max_ready and overflow_policy='error', when ready queue fills,
    vision worker triggers fatal BufferError, existing ready frames are preserved,
    wait/poll/submit surface fatal error, and close/reset exit cleanly."""
    with AsyncVisualPlanner(
        lambda obs: f"p_{obs.frame_id}",
        lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
        max_frames=None,
        max_pending=None,
        max_ready=2,
        overflow_policy="error",
    ) as planner:
        planner.reset(episode_id="ep_ready_err", prompt="test ready error")

        planner.submit(make_obs(0, capture_time=0.0))
        planner.submit(make_obs(1, capture_time=0.1))
        assert planner.wait_ready(min_frames=2, timeout=5.0)

        # Submit 3rd frame which will cause ready queue overflow in worker
        planner.submit(make_obs(2, capture_time=0.2))

        # Wait for vision worker to encounter error and surface via cv
        with planner._cv:
            assert planner._cv.wait_for(lambda: planner._vision_error is not None, timeout=5.0)

        with pytest.raises(RuntimeError, match="Vision worker fatal error.*Ready queue full"):
            planner.wait_ready(min_frames=2, timeout=5.0)

        # Existing ready frames are preserved
        st = planner.stats()
        assert st["ready_frame_ids"] == [0, 1]
        assert st["errors_count"] == 1

        # Surface in submit and poll_plan
        with pytest.raises(RuntimeError, match="Vision worker fatal error"):
            planner.submit(make_obs(3, capture_time=0.3))
        with pytest.raises(RuntimeError, match="Vision worker fatal error"):
            planner.poll_plan()

        # Reset clears fatal error cleanly
        planner.reset(episode_id="ep_ready_recovered", prompt="recovered")
        assert planner.stats()["errors_count"] == 0

    # Thread joins cleanly in context manager exit


def test_wait_ready_allows_arbitrary_positive_in_dynamic_mode():
    """wait_ready accepts any positive integer when max_ready/max_frames is None,
    but rejects non-positive or when min_frames exceeds effective capacity."""
    with AsyncVisualPlanner(
        lambda obs: "p",
        lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
        max_frames=None,
        max_ready=None,
    ) as planner:
        planner.reset(episode_id="ep", prompt="p")

        # Invalid min_frames
        with pytest.raises(ValueError, match="positive int"):
            planner.wait_ready(min_frames=0)
        with pytest.raises(ValueError, match="positive int"):
            planner.wait_ready(min_frames=-1)
        with pytest.raises(ValueError, match="positive int"):
            planner.wait_ready(min_frames=True)  # bool rejected

        # Large min_frames is valid syntax in dynamic unlimited mode
        # Timeout quickly since no frames submitted
        assert planner.wait_ready(min_frames=100, timeout=0.01) is False

    # When capacity is bounded (e.g. max_frames=5), min_frames > 5 is rejected
    with AsyncVisualPlanner(
        lambda obs: "p",
        lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
        max_frames=5,
    ) as planner5:
        with pytest.raises(ValueError, match="exceeds effective ready capacity"):
            planner5.wait_ready(min_frames=6)


def test_constructor_capacity_and_overflow_policy_validation():
    """Constructor validates capacities, rejection of bool/non-positive, and max_ready constraints."""
    # bool rejected for max_frames, max_pending, max_ready
    with pytest.raises(ValueError, match="max_frames must be positive int or None"):
        AsyncVisualPlanner(lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_frames=True)
    with pytest.raises(ValueError, match="max_pending must be positive int or None"):
        AsyncVisualPlanner(lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_pending=False)
    with pytest.raises(ValueError, match="max_ready must be positive int or None"):
        AsyncVisualPlanner(
            lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_frames=None, max_ready=True
        )

    # Negative/zero rejected
    with pytest.raises(ValueError, match="max_frames must be positive int or None"):
        AsyncVisualPlanner(lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_frames=0)
    with pytest.raises(ValueError, match="max_pending must be positive int or None"):
        AsyncVisualPlanner(lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_pending=-2)
    with pytest.raises(ValueError, match="max_ready must be positive int or None"):
        AsyncVisualPlanner(
            lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_frames=None, max_ready=0
        )

    # max_ready rejected if max_frames is not None
    with pytest.raises(ValueError, match="max_ready is only allowed when max_frames is None"):
        AsyncVisualPlanner(
            lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), max_frames=5, max_ready=10
        )

    # Invalid overflow_policy rejected
    with pytest.raises(ValueError, match="overflow_policy must be 'drop_oldest' or 'error'"):
        AsyncVisualPlanner(
            lambda o: "p", lambda f, p: PlanComputation(torch.zeros(1, 1, 1)), overflow_policy="drop_newest"
        )


def test_observation_time_fallback_and_provided_semantics():
    """Test observation_time:
    - Fallback: capture_time relative to first frame (0.0, 0.2, 0.9)
    - Equal observation_times allowed
    - Provided mode with non-uniform intervals
    - Strict non-decreasing check: backward time rejected without advancing state
    - NaN/Inf/bool/negative rejected
    - Mixing provided and fallback rejected
    """
    # 1. Fallback capture_relative
    with AsyncVisualPlanner(
        lambda o: "p",
        lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
        max_frames=None,
    ) as planner:
        planner.reset(episode_id="ep_time1", prompt="p")
        # Submit with capture_time 10.0, 10.2, 10.9
        planner.submit(make_obs(0, capture_time=10.0))
        planner.submit(make_obs(1, capture_time=10.2))
        planner.submit(make_obs(2, capture_time=10.9))

        st = planner.stats()
        assert st["time_source"] == "capture_relative"
        assert st["episode_capture_origin"] == 10.0
        assert st["last_observation_time"] == pytest.approx(0.9)

        # Equal capture_time (and thus equal observation_time) is allowed
        planner.submit(make_obs(3, capture_time=10.9))
        assert planner.stats()["last_observation_time"] == pytest.approx(0.9)

        # Backward capture_time in fallback mode is rejected
        with pytest.raises(ValueError, match="non-decreasing"):
            planner.submit(make_obs(4, capture_time=10.5))

        # Failed submit did not advance last_submitted_id
        assert planner.stats()["last_submitted_id"] == 3

        # Mixing provided into capture_relative episode is rejected
        with pytest.raises(ValueError, match="Cannot mix observation_time sources"):
            planner.submit(make_obs(4, capture_time=11.0, observation_time=1.0))

        assert planner.wait_ready(min_frames=4, timeout=5.0)
        assert planner.request_plan()
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        obs_times = [f.observation.observation_time for f in res.frames]
        assert obs_times == [pytest.approx(0.0), pytest.approx(0.2), pytest.approx(0.9), pytest.approx(0.9)]
        # Public metadata has observation_time
        assert all(isinstance(f.observation, ObservationMetadata) for f in res.frames)

    # 2. Explicit provided observation_time
    with AsyncVisualPlanner(
        lambda o: "p",
        lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
        max_frames=None,
    ) as planner:
        planner.reset(episode_id="ep_time2", prompt="p")
        planner.submit(make_obs(0, capture_time=50.0, observation_time=0.0))
        planner.submit(make_obs(1, capture_time=50.1, observation_time=0.35))
        planner.submit(make_obs(2, capture_time=50.2, observation_time=1.20))

        st = planner.stats()
        assert st["time_source"] == "provided"
        assert st["episode_capture_origin"] is None
        assert st["last_observation_time"] == pytest.approx(1.20)

        # Backward provided time is rejected
        with pytest.raises(ValueError, match="non-decreasing"):
            planner.submit(make_obs(3, capture_time=50.3, observation_time=1.10))
        assert planner.stats()["last_submitted_id"] == 2

        # Mixing None into provided episode is rejected
        with pytest.raises(ValueError, match="Cannot mix observation_time sources"):
            planner.submit(make_obs(3, capture_time=50.3, observation_time=None))

        # Bad observation_time values rejected
        with pytest.raises(ValueError, match="observation_time must be finite non-negative float"):
            planner.submit(make_obs(3, capture_time=50.3, observation_time=float("nan")))
        with pytest.raises(ValueError, match="observation_time must be finite non-negative float"):
            planner.submit(make_obs(3, capture_time=50.3, observation_time=True))
        with pytest.raises(ValueError, match="observation_time must be finite non-negative float"):
            planner.submit(make_obs(3, capture_time=50.3, observation_time=-0.1))


def test_reset_clears_time_origin_and_late_vision_does_not_pollute_new_origin():
    """Reset clears time origin; slow vision encode from previous generation does not pollute new origin."""
    vision_entered = threading.Event()
    vision_release = threading.Event()

    def gating_encode(obs: Observation) -> str:
        if obs.frame_id == 0:
            vision_entered.set()
            if not vision_release.wait(timeout=10.0):
                raise TimeoutError("Vision release timed out")
        return f"enc_{obs.frame_id}"

    try:
        with AsyncVisualPlanner(
            gating_encode,
            lambda f, p: PlanComputation(actions=torch.zeros((1, 1, 1))),
            max_frames=None,
        ) as planner:
            planner.reset(episode_id="ep_gen1", prompt="p1")
            # Old generation frame 0 has capture_time 1000.0
            planner.submit(make_obs(0, capture_time=1000.0))
            assert vision_entered.wait(timeout=5.0)

            # Reset to gen 2 while frame 0 is encoding
            planner.reset(episode_id="ep_gen2", prompt="p2")
            st_reset = planner.stats()
            assert st_reset["time_source"] is None
            assert st_reset["episode_capture_origin"] is None
            assert st_reset["last_observation_time"] is None

            # New generation frame 0 has capture_time 50.0
            planner.submit(make_obs(0, capture_time=50.0))
            assert planner.stats()["episode_capture_origin"] == 50.0

            # Release old vision worker
            vision_release.set()

            planner.submit(make_obs(1, capture_time=50.5))
            assert planner.wait_ready(min_frames=2, timeout=5.0)
            assert planner.request_plan()
            res = planner.wait_plan(timeout=5.0)
            assert res is not None
            assert len(res.frames) == 2
            # Origin was not polluted by 1000.0
            assert res.frames[0].observation.observation_time == pytest.approx(0.0)
            assert res.frames[1].observation.observation_time == pytest.approx(0.5)
            assert planner.stats()["episode_capture_origin"] == 50.0
    finally:
        vision_release.set()


def test_retry_preserves_exact_snapshot_and_observation_times():
    """Planner retry retains exact snapshot including observation_times, and new frames are not mixed in."""
    attempt = 0

    def fail_first_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("Temporary planner error")
        return PlanComputation(actions=torch.zeros((1, 1, 1)))

    with AsyncVisualPlanner(
        lambda o: "p",
        fail_first_plan,
        max_frames=None,
    ) as planner:
        planner.reset(episode_id="ep_retry_time", prompt="p")
        planner.submit(make_obs(0, capture_time=10.0))
        planner.submit(make_obs(1, capture_time=10.3))
        assert planner.wait_ready(min_frames=2, timeout=5.0)

        assert planner.request_plan()
        with pytest.raises(RuntimeError, match="Temporary planner error"):
            planner.wait_plan(timeout=5.0)

        # Submit frame 2 while retry is pending
        planner.submit(make_obs(2, capture_time=10.7))
        assert planner.wait_ready(min_frames=1, timeout=5.0)

        # Retry first plan
        assert planner.request_plan(retry=True)
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        # Must only contain frames 0 and 1
        assert [f.observation.frame_id for f in res.frames] == [0, 1]
        assert [f.observation.observation_time for f in res.frames] == [pytest.approx(0.0), pytest.approx(0.3)]
        # Frame 2 is still ready for next request
        assert planner.stats()["ready_frame_ids"] == [2]
