from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from fabri_moss.async_pipeline import (
    AsyncVisualPlanner,
    EncodedFrame,
    Observation,
    PlanComputation,
    PlanResult,
)
from fabri_moss.evaluate_async import (
    action_for_step,
    capture_observation,
    denormalize_action,
    recover_starvation_plan,
    resolve_task_prompt,
    run_episode,
    should_accept_plan,
)


# ============================================================================
# Test Fake Environment & Utilities
# ============================================================================


class BoxSpace:
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low.astype(np.float32)
        self.high = high.astype(np.float32)
        self.shape = self.low.shape


class DummyEnv:
    """Fake environment with asymmetric bounds to detect denormalization fallbacks."""

    def __init__(self, action_dim: int = 4, render_h: int = 60, render_w: int = 80) -> None:
        self.action_space = BoxSpace(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
        )
        self.render_h = render_h
        self.render_w = render_w
        self.step_actions = []
        self.render_calls = 0
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self.step_actions.clear()
        self.render_calls = 0
        return np.ones(39, dtype=np.float32) * 0.1, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.step_count += 1
        obs = np.ones(39, dtype=np.float32) * 0.1
        return obs, 0.0, False, False, {}

    def render(self) -> np.ndarray:
        self.render_calls += 1
        return np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8)


def make_dummy_stats() -> Dict[str, Any]:
    # Asymmetric action stats min/max so denormalized 0 != raw 0
    # For min=-2, max=6:
    # (0 + 1) / 2 * (6 - (-2)) + (-2) = 1/2 * 8 - 2 = +2.0 != 0.0
    return {
        "action": {
            "min": [-2.0, -2.0, -2.0, -2.0],
            "max": [6.0, 6.0, 6.0, 6.0],
        },
        "observation.state": {
            "min": [-1.0] * 4,
            "max": [1.0] * 4,
        },
    }


def make_dummy_plan(
    source_frame_id: int,
    horizon: int = 4,
    dim: int = 4,
    generation: int = 0,
) -> PlanResult:
    # Action tensor [1, H, D] with values (t + 1) * 0.1
    actions = torch.empty((1, horizon, dim), dtype=torch.float32)
    for t in range(horizon):
        actions[0, t, :] = float(t + 1) * 0.1

    comp = PlanComputation(actions=actions)
    obs = Observation(
        frame_id=source_frame_id,
        capture_time=time.monotonic(),
        images=[Image.new("RGB", (16, 16))],
        state=torch.zeros((1, 24), dtype=torch.float32),
        state_mask=torch.ones((1, 24), dtype=torch.bool),
        action_mask=torch.ones((1, 24), dtype=torch.bool),
    )
    enc = EncodedFrame(
        observation=obs,
        payload=None,
        encode_started=time.monotonic(),
        encode_finished=time.monotonic(),
    )
    now = time.monotonic()
    return PlanResult(
        episode_id="ep0",
        generation=generation,
        frames=(enc,),
        source_frame_id=source_frame_id,
        capture_time=now,
        started=now,
        finished=now,
        computation=comp,
    )


# ============================================================================
# 1. Action Ledger & Acceptance Unit Tests
# ============================================================================


def test_action_for_step_ledger_alignment() -> None:
    # source_frame_id = 5, horizon = 4 (valid for steps 5, 6, 7, 8)
    plan = make_dummy_plan(source_frame_id=5, horizon=4, dim=4)

    # current_step = 7 -> index = 7 - 5 = 2 -> value 0.3
    act_7 = action_for_step(plan, current_step=7)
    assert act_7 is not None
    np.testing.assert_allclose(act_7, np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32), rtol=1e-5)

    # expired step: current_step = 9 -> index = 4 >= horizon -> None
    assert action_for_step(plan, current_step=9) is None

    # future step: current_step = 4 -> index = -1 < 0 -> None
    assert action_for_step(plan, current_step=4) is None


def test_should_accept_plan_invariants() -> None:
    # Base plan at source_frame_id=5, horizon=4
    curr_gen = 1
    plan = make_dummy_plan(source_frame_id=5, horizon=4, generation=curr_gen)

    # Valid acceptance: current_step=7, accepted_source=3
    assert should_accept_plan(plan, current_generation=curr_gen, accepted_source_frame_id=3, current_step=7)

    # Invariant 1: Generation mismatch rejected
    assert not should_accept_plan(plan, current_generation=curr_gen + 1, accepted_source_frame_id=3, current_step=7)

    # Invariant 2: Candidate source_frame_id <= accepted_source_frame_id rejected
    assert not should_accept_plan(plan, current_generation=curr_gen, accepted_source_frame_id=5, current_step=7)
    assert not should_accept_plan(plan, current_generation=curr_gen, accepted_source_frame_id=6, current_step=7)

    # Invariant 3: Candidate source in future beyond current_step rejected
    assert not should_accept_plan(plan, current_generation=curr_gen, accepted_source_frame_id=3, current_step=4)

    # Invariant 4: Plan already expired at current_step (current_step - source >= horizon) rejected
    # horizon is 4, so at step 5+4=9 it has expired
    assert not should_accept_plan(plan, current_generation=curr_gen, accepted_source_frame_id=3, current_step=9)


# ============================================================================
# 2. Task Prompt Resolution & Missing Metadata Tests
# ============================================================================


def test_resolve_task_prompt_official_metadata(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evaluations" / "metaworld"
    eval_dir.mkdir(parents=True)

    order_path = eval_dir / "mt50_order.json"
    tasks_path = eval_dir / "tasks.jsonl"

    order_data = {"idx_to_slug": {"0": "custom-task-v1", "1": "reach-v3"}}
    order_path.write_text(json.dumps(order_data), encoding="utf-8")

    tasks_lines = [
        json.dumps({"task_index": 0, "task": "Do custom operation"}),
        json.dumps({"task_index": 1, "task": "Reach goal official prompt"}),
    ]
    tasks_path.write_text("\n".join(tasks_lines), encoding="utf-8")

    prompt = resolve_task_prompt("custom-task-v1", fabri_root=tmp_path)
    assert prompt == "Do custom operation"

    # Missing task in metadata and not reach-v3 fallback must raise RuntimeError
    with pytest.raises(RuntimeError, match="Could not resolve official prompt"):
        resolve_task_prompt("unknown-nonexistent-task", fabri_root=tmp_path)


# ============================================================================
# 3. Step-Wait Starvation Recovery Without Deadlock
# ============================================================================


def test_recover_starvation_plan_inflight_no_deadlock() -> None:
    # Test recover_starvation_plan when planner is already busy (inflight/pending)
    # Background planner block is released via Event (no arbitrary sleep)
    gate = threading.Event()

    def dummy_encode(obs: Observation) -> Any:
        return None

    def dummy_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        gate.wait(timeout=5.0)
        return PlanComputation(actions=torch.ones((1, 3, 4), dtype=torch.float32))

    planner = AsyncVisualPlanner(
        encode=dummy_encode,
        plan=dummy_plan,
        max_frames=3,
        max_pending=4,
    )
    env = DummyEnv()
    stats = make_dummy_stats()

    try:
        planner.reset(episode_id="rec_test", prompt="test")
        obs0 = capture_observation(env, 0, stats, np.zeros(39, dtype=np.float32))
        planner.submit(obs0)
        assert planner.wait_ready(min_frames=1, timeout=2.0)

        # Trigger plan so it becomes inflight and blocked on gate
        assert planner.request_plan()
        st = planner.stats()
        assert st["planner_busy"] or st["plan_pending"] or len(st["planner_inflight_frame_ids"]) > 0

        # Run recover in a separate thread because it will wait for the inflight plan
        recovered = [None]
        rec_err = [None]

        def run_recover():
            try:
                plan, _ = recover_starvation_plan(
                    planner=planner,
                    env=env,
                    current_step=0,
                    real_env_obs=np.zeros(39, dtype=np.float32),
                    norm_stats=stats,
                    current_generation=st["generation"],
                    accepted_source_id=-1,
                    wait_timeout=4.0,
                )
                recovered[0] = plan
            except Exception as e:
                rec_err[0] = e

        t = threading.Thread(target=run_recover)
        t.start()

        # Release background worker
        gate.set()
        t.join(timeout=3.0)

        assert not t.is_alive(), "Starvation recovery deadlocked"
        assert rec_err[0] is None, f"Starvation recovery raised: {rec_err[0]}"
        assert recovered[0] is not None
        assert recovered[0].source_frame_id == 0
    finally:
        gate.set()
        planner.close()


# ============================================================================
# 4. Realtime Mode Zero Fallback
# ============================================================================


def test_realtime_zero_fallback_on_starvation() -> None:
    # First plan finishes immediately with H=1.
    # Second plan is blocked by gate so step 1 exhausts actions and enters fallback.
    gate = threading.Event()
    call_count = 0

    def dummy_encode(obs: Observation) -> Any:
        return None

    def dummy_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            gate.wait(timeout=2.0)
        # H=1 action
        return PlanComputation(actions=torch.ones((1, 1, 4), dtype=torch.float32))

    planner = AsyncVisualPlanner(
        encode=dummy_encode,
        plan=dummy_plan,
        max_frames=2,
        max_pending=4,
    )
    env = DummyEnv(action_dim=4)
    stats = make_dummy_stats()

    try:
        res = run_episode(
            env=env,
            planner=planner,
            episode_id="rt_ep",
            prompt="reach",
            norm_stats=stats,
            control_mode="realtime",
            episode_horizon=3,
            exec_horizon=1,
            control_hz=100.0,
            wait_timeout=2.0,
        )

        assert res["realtime_zero_fallbacks"] > 0
        # Warmup step is index 0; subsequent steps are from rollout
        # Rollout steps that experienced starvation must be exact zeros
        # not denormalized zero (which would be +2.0 under dummy_stats)
        assert len(env.step_actions) >= 3  # 1 warmup + 2 or 3 steps
        rollout_actions = env.step_actions[1:]  # skip warmup

        # At least one step was a fallback
        has_exact_zero_fallback = False
        for act in rollout_actions:
            if np.all(act == 0.0):
                has_exact_zero_fallback = True
                assert not np.allclose(act, 2.0), "Action denormalized to 2.0 instead of raw 0"

        assert has_exact_zero_fallback, f"No exact zero fallback found in {rollout_actions}"
    finally:
        gate.set()
        planner.close()


# ============================================================================
# 5. Short Step-Wait Rollout with Recovery & Render Verification
# ============================================================================


def test_short_step_wait_rollout_with_recovery() -> None:
    # Plans return H=2 actions. Episode horizon=4 so starvation recovery triggers.
    def dummy_encode(obs: Observation) -> Any:
        return None

    def dummy_plan(frames: Tuple[EncodedFrame, ...], prompt: str) -> PlanComputation:
        return PlanComputation(actions=torch.ones((1, 2, 4), dtype=torch.float32) * 0.5)

    planner = AsyncVisualPlanner(
        encode=dummy_encode,
        plan=dummy_plan,
        max_frames=3,
        max_pending=4,
    )
    env = DummyEnv(action_dim=4)
    stats = make_dummy_stats()

    try:
        res = run_episode(
            env=env,
            planner=planner,
            episode_id="step_wait_ep",
            prompt="reach",
            norm_stats=stats,
            control_mode="step_wait",
            episode_horizon=4,
            exec_horizon=2,
            wait_timeout=3.0,
        )

        assert res["steps"] == 4
        assert res["executed_actions_count"] == 4
        assert len(res["plans"]) >= 2, "Should have accepted at least 2 plans"
        # Render was invoked for initial frame + subsequent frames in rollout
        assert env.render_calls > 0
        # Action offsets must be appropriate (0 <= offset < horizon 2)
        for offset in res["action_offsets"]:
            assert 0 <= offset < 2
    finally:
        planner.close()


# ============================================================================
# 6. Capture Observation Validation Tests
# ============================================================================


def test_capture_observation_validation() -> None:
    env = DummyEnv()
    stats = make_dummy_stats()

    # Valid capture
    obs = capture_observation(
        env=env,
        step_id=0,
        norm_stats=stats,
        real_env_obs=np.zeros(39, dtype=np.float32),
    )
    assert obs.frame_id == 0
    assert obs.state.shape == (1, 24)
    assert len(obs.images) == 1

    # Missing state dimensions (shorter than stats length 39)
    with pytest.raises(ValueError, match="is shorter than stats length"):
        capture_observation(
            env=env,
            step_id=0,
            norm_stats=stats,
            real_env_obs=np.zeros(3, dtype=np.float32),
        )

    # Bad render shape from env
    class BadRenderEnv(DummyEnv):
        def render(self) -> np.ndarray:
            return np.zeros((48, 48), dtype=np.uint8)  # Missing 3rd channel

    bad_env = BadRenderEnv()
    with pytest.raises(ValueError, match="raw_rgb must have shape"):
        capture_observation(
            env=bad_env,
            step_id=0,
            norm_stats=stats,
            real_env_obs=np.zeros(39, dtype=np.float32),
        )


def test_evaluate_async_cli_parser_memory_mode():
    from fabri_moss.evaluate_async import build_parser

    parser = build_parser()
    args_default = parser.parse_args(["--mode", "moss", "--output-dir", "/tmp/test"])
    assert args_default.memory_mode == "consume"
    assert args_default.window is None

    args_delta = parser.parse_args(["--mode", "moss", "--output-dir", "/tmp/test", "--memory-mode", "delta", "--window", "4"])
    assert args_delta.memory_mode == "delta"
    assert args_delta.window == 4

    args_native_cache = parser.parse_args(["--mode", "native-cache", "--output-dir", "/tmp/test"])
    assert args_native_cache.mode == "native-cache"
    assert args_native_cache.adapter is None  # native-cache does not require adapter checkpoint
    assert args_native_cache.history_frames == 16
    assert args_native_cache.window is None
    assert args_native_cache.max_pending is None
    assert args_native_cache.max_ready is None
    assert args_native_cache.timestamp_mode == "text"

    args_native_custom = parser.parse_args([
        "--mode", "native-cache",
        "--output-dir", "/tmp/test",
        "--history-frames", "20",
        "--timestamp-mode", "none",
    ])
    assert args_native_custom.history_frames == 20
    assert args_native_custom.timestamp_mode == "none"

    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "moss", "--output-dir", "/tmp/test", "--memory-mode", "invalid_mode"])


def test_evaluate_async_cli_native_cache_adapter_guard(monkeypatch, tmp_path: Path):
    from fabri_moss.evaluate_async import main
    import sys

    output_dir = tmp_path / "test_adapter_guard"
    test_args = [
        "evaluate_async",
        "--mode", "native-cache",
        "--output-dir", str(output_dir),
        "--adapter", "/path/to/adapter.pt",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(ValueError, match="--adapter is forbidden and unsupported when --mode=native-cache"):
        main()

    # Output directory must not be created since validation fails prior to directory creation
    assert not output_dir.exists()


def test_evaluate_async_cli_native_cache_window_guard(monkeypatch, tmp_path: Path):
    from fabri_moss.evaluate_async import main
    import sys

    output_dir = tmp_path / "test_guard"
    test_args = [
        "evaluate_async",
        "--mode", "native-cache",
        "--output-dir", str(output_dir),
        "--history-frames", "3",
        "--window", "5",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(ValueError, match="--window is forbidden and unsupported when --mode=native-cache"):
        main()

    # Output directory must not be created since validation fails prior to directory creation
    assert not output_dir.exists()


def test_evaluate_async_cli_native_cache_history_frames_small_allowed(monkeypatch, tmp_path: Path):
    from fabri_moss.evaluate_async import build_parser, main
    import sys
    import fabri_moss.evaluate_async

    output_dir = tmp_path / "test_history_3"
    test_args = [
        "evaluate_async",
        "--mode", "native-cache",
        "--output-dir", str(output_dir),
        "--history-frames", "3",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr(fabri_moss.evaluate_async, "resolve_task_prompt", lambda **kwargs: "reach task")

    def mock_load_native_checkpoint(**kwargs):
        raise RuntimeError("loader reached")

    monkeypatch.setattr(fabri_moss.evaluate_async, "load_native_checkpoint", mock_load_native_checkpoint)

    with pytest.raises(RuntimeError, match="loader reached"):
        main()

    # Output directory was created as validation passed before loader reached
    assert output_dir.exists()


@pytest.mark.parametrize(
    "invalid_cli_args,expected_error",
    [
        (
            ["--mode", "native-cache", "--history-frames", "0"],
            "--history-frames must be positive",
        ),
        (
            ["--mode", "native-cache", "--max-ready", "0"],
            "--max-ready must be positive",
        ),
        (
            ["--mode", "native-cache", "--max-pending", "-1"],
            "--max-pending must be positive",
        ),
    ],
)
def test_evaluate_async_cli_early_guards_prevent_output_dir(monkeypatch, tmp_path: Path, invalid_cli_args, expected_error):
    from fabri_moss.evaluate_async import main
    import sys

    output_dir = tmp_path / "test_early_guard"
    test_args = ["evaluate_async", *invalid_cli_args, "--output-dir", str(output_dir)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(ValueError, match=expected_error):
        main()

    assert not output_dir.exists()


def test_evaluate_async_simulated_env_dt_independent_of_step_wait():
    """Verify capture_observation computes observation_time = step_id * env.dt regardless of wall-clock delay."""
    class MockEnvWithDt(DummyEnv):
        def __init__(self):
            super().__init__()
            self.dt = 0.0125

    env = MockEnvWithDt()
    stats = make_dummy_stats()
    real_env_obs = np.zeros(39, dtype=np.float32)

    # capture_time passed as keyword argument (mock time values 100.0 and 900.0)
    obs0 = capture_observation(
        env,
        step_id=0,
        norm_stats=stats,
        real_env_obs=real_env_obs,
        capture_time=100.0,
    )
    assert obs0.observation_time == 0.0
    assert obs0.capture_time == 100.0

    obs1 = capture_observation(
        env,
        step_id=5,
        norm_stats=stats,
        real_env_obs=real_env_obs,
        capture_time=900.0,
    )
    # 5 * 0.0125 = 0.0625 s, step 5 computed without needing sleep
    assert abs(obs1.observation_time - 0.0625) < 1e-6
    assert obs1.capture_time == 900.0

    # Explicit observation_time overrides env.dt
    obs_explicit = capture_observation(
        env,
        step_id=5,
        norm_stats=stats,
        real_env_obs=real_env_obs,
        observation_time=1.234,
    )
    assert obs_explicit.observation_time == 1.234

    # When env has no dt (None), observation_time falls back to None
    env_no_dt = DummyEnv()
    obs_none_dt = capture_observation(
        env_no_dt,
        step_id=5,
        norm_stats=stats,
        real_env_obs=real_env_obs,
    )
    assert obs_none_dt.observation_time is None
