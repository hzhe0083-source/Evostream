"""CPU Unit and integration tests for fabri_moss.evaluate_predictive.

Tests cover:
- Real tiny PredictiveMemoryPolicy running run_episode with real CausalMemoryWriter activation
  and verifying future_head.forward is never called.
- Action ledger, stride, execution horizon, planning steps, decision mappings, and offset progression.
- Exact normalized denormalization clipping and asymmetric stats.
- Success, terminal/truncated, horizon cap termination conditions.
- Step seconds requirement when env.dt is missing and strict simulated dt behavior.
- NaN / non-finite action detection and hook cleanup guarantees.
- Orchestrator run_evaluation end-to-end with metadata manifest, incremental writes,
  summary reporting, error handling, and coverage checks.
"""

from __future__ import annotations

import json
from pathlib import Path
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from fabri_moss.evaluate_predictive import (
    build_parser,
    compute_evaluation_summary,
    load_mt50_metadata,
    resolve_task_manifest,
    run_episode,
    run_evaluation,
)
import fabri_moss.predictive_inference as pi
from fabri_moss.tests.test_predictive_inference import _make_toy_policy


class FakeBoxSpace:
    def __init__(self, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0) -> None:
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class FakeGymnasiumEnv:
    def __init__(
        self,
        dt: Optional[float] = 0.1,
        success_at_step: Optional[int] = None,
        terminal_at_step: Optional[int] = None,
        fail_on_reset: bool = False,
    ) -> None:
        self.dt = dt
        self.action_space = FakeBoxSpace((4,), low=-1.0, high=1.0)
        self.calls = 0
        self.obs = np.zeros(39, dtype=np.float32)
        self.last_action: Optional[np.ndarray] = None
        self.success_at_step = success_at_step
        self.terminal_at_step = terminal_at_step
        self.fail_on_reset = fail_on_reset
        self.reset_seeds: List[Optional[int]] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.fail_on_reset:
            raise RuntimeError("FakeEnv simulated reset failure")
        self.reset_seeds.append(seed)
        self.calls = 0
        self.obs = np.zeros(39, dtype=np.float32)
        return self.obs.copy(), {}

    def step(self, action: np.ndarray) -> Any:
        self.calls += 1
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        dim = min(len(self.obs), len(action))
        self.obs[:dim] = action[:dim]

        formal_step = self.calls - 2  # warmup is calls=1, first episode step is calls=2 => step 0
        success = bool(self.success_at_step is not None and formal_step >= self.success_at_step)
        info = {"success": 1.0 if success else 0.0}
        terminated = bool(self.terminal_at_step is not None and formal_step >= self.terminal_at_step)
        return self.obs.copy(), 0.0, terminated, False, info

    def render(self) -> np.ndarray:
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        img[..., 0] = (self.calls * 17) % 256
        return img


class FakeGymWrapper:
    def __init__(self, envs: List[Any]) -> None:
        self.envs = envs
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeScriptedPolicy(nn.Module):
    def __init__(self, horizon: int = 5, per_action_dim: int = 24, return_nan: bool = False) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.writer = nn.Identity()
        self.policy = types.SimpleNamespace(
            action_head=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    horizon=horizon,
                    state_dim=24,
                    per_action_dim=per_action_dim,
                    num_inference_timesteps=50,
                )
            )
        )
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.return_nan = return_nan
        self.saved_samples: List[Any] = []

    def predict_actions(self, sample: Dict[str, Any]) -> torch.Tensor:
        self.saved_samples.append(sample)
        _ = self.writer(torch.zeros(1, device=self.param.device))
        assert "future_images" not in sample, "future_images leaked into sample!"
        assert "future_actions" not in sample, "future_actions leaked into sample!"

        if self.return_nan:
            return torch.full((1, self.horizon, self.per_action_dim), float("nan"), device=self.param.device)

        out = torch.empty((1, self.horizon, self.per_action_dim), dtype=torch.float32, device=self.param.device)
        for h in range(self.horizon):
            out[0, h, :] = 0.1 * (h + 1)
        return out


class FakeNoisyPolicy(FakeScriptedPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.noises: List[torch.Tensor] = []

    def predict_actions(self, sample: Dict[str, Any]) -> torch.Tensor:
        self.noises.append(torch.randn(8, device=self.param.device).cpu())
        return super().predict_actions(sample)


def _make_toy_metadata_workspace(tmp_path: Path) -> Tuple[Path, Path]:
    eval_dir = tmp_path / "evaluations" / "metaworld"
    eval_dir.mkdir(parents=True, exist_ok=True)

    order_file = eval_dir / "mt50_order.json"
    order_data = {
        "ordered_indices": [0, 1],
        "idx_to_slug": {"0": "reach-v2", "1": "push-v2"},
        "groups": {"easy": ["reach-v2"], "hard": ["push-v2"]},
    }
    with order_file.open("w", encoding="utf-8") as f:
        json.dump(order_data, f)

    tasks_file = eval_dir / "tasks.jsonl"
    with tasks_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": "reach target point", "task_slug": "reach-v2"}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": "push puck to target", "task_slug": "push-v2"}) + "\n")

    return tmp_path, eval_dir


def test_real_tiny_predictive_memory_policy_writer_activation() -> None:
    """Verify real PredictiveMemoryPolicy runs in run_episode, activates writer, and omits future_head."""
    policy = _make_toy_policy(shallow_layer=1)
    policy.policy.action_head.config.horizon = 2
    policy.policy.action_head.config.state_dim = 24
    policy.policy.action_head.config.per_action_dim = 4

    def _throw_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("future_head.forward must not be invoked during inference!")

    policy.future_head.forward = _throw_if_called

    norm_stats = {
        "action": {"min": np.array([-1.0] * 4), "max": np.array([1.0] * 4)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }
    env = FakeGymnasiumEnv(dt=0.1)

    result = run_episode(
        env=env,
        model=policy,
        prompt="reach red cube",
        norm_stats=norm_stats,
        episode_horizon=12,
        exec_horizon=2,
        observation_stride=1,
        seed=1001,
        state_dim=24,
        action_dim=4,
    )

    assert result["steps"] == 12
    assert result["executed_actions_count"] == 12
    assert result["total_writer_calls"] > 0
    assert len(result["plans"]) == 6
    assert env.reset_seeds == [1001]


def test_stride_execution_horizon_action_ledger_and_decision_mapping() -> None:
    """Verify stride=2, exec=3: frame IDs, planning steps, decision index tracking, and offsets."""
    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    env = FakeGymnasiumEnv(dt=0.1)
    norm_stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }

    result = run_episode(
        env=env,
        model=policy,
        prompt="push puck",
        norm_stats=norm_stats,
        episode_horizon=10,
        exec_horizon=3,
        observation_stride=2,
        seed=42,
    )

    expected_frames = [0, 2, 3, 4, 6, 8, 9]
    assert result["history_final_frames"] == expected_frames
    assert [p["source_step"] for p in result["plans"]] == [0, 3, 6, 9]
    assert [c["offset"] for c in result["command_log"]] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    assert result["history_final_decisions"] == [0, 2, 4, 6]
    assert result["decision_frame_ids"] == [0, 3, 6, 9]
    np.testing.assert_allclose(result["history_final_observation_times"], np.asarray(expected_frames) * 0.1)
    np.testing.assert_allclose(policy.saved_samples[1]["state"][0, :4], [0.3] * 4, atol=1e-6)
    assert all(sample["memory_replay"] and "actions" not in sample for sample in policy.saved_samples)


def test_asymmetric_denormalization_and_action_clipping() -> None:
    """Test asymmetric min/max denormalization and environment action clipping bounds."""
    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    env = FakeGymnasiumEnv(dt=0.1)
    env.action_space.low = np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32)
    env.action_space.high = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    norm_stats = {
        "action": {
            "min": np.array([-2.0, 0.0, -10.0, -1.0] + [0.0] * 20, dtype=np.float32),
            "max": np.array([2.0, 4.0, 10.0, 1.0] + [1.0] * 20, dtype=np.float32),
        },
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }

    result = run_episode(
        env=env,
        model=policy,
        prompt="test action bounds",
        norm_stats=norm_stats,
        episode_horizon=2,
        exec_horizon=2,
    )

    for cmd in result["command_log"]:
        act = np.array(cmd["action"])
        assert len(act) == 4
        assert np.all(act >= -0.5)
        assert np.all(act <= 0.5)


def test_termination_success_terminal_and_horizon_cap() -> None:
    """Verify termination under success flag, terminal/truncated flag, and episode horizon."""
    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    norm_stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }

    # Success before terminal
    env_succ = FakeGymnasiumEnv(dt=0.1, success_at_step=2)
    res_succ = run_episode(env_succ, policy, "succ", norm_stats, episode_horizon=10)
    assert res_succ["success"] is True
    assert res_succ["stop_reason"] == "env_success"
    assert res_succ["steps"] == 3

    # Terminal without success
    env_term = FakeGymnasiumEnv(dt=0.1, terminal_at_step=3)
    res_term = run_episode(env_term, policy, "term", norm_stats, episode_horizon=10)
    assert res_term["success"] is False
    assert res_term["stop_reason"] == "env_terminal"
    assert res_term["steps"] == 4

    # Horizon cap reached
    env_cap = FakeGymnasiumEnv(dt=0.1)
    res_cap = run_episode(env_cap, policy, "cap", norm_stats, episode_horizon=5)
    assert res_cap["success"] is False
    assert res_cap["stop_reason"] == "horizon_reached"
    assert res_cap["steps"] == 5


def test_missing_dt_requires_step_seconds() -> None:
    """Verify environment without dt fails unless step_seconds is explicitly provided."""
    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    norm_stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }
    env_no_dt = FakeGymnasiumEnv(dt=None)

    with pytest.raises(RuntimeError, match="Explicit --step-seconds is strictly required"):
        run_episode(env_no_dt, policy, "test dt", norm_stats, episode_horizon=2)

    res = run_episode(env_no_dt, policy, "test dt", norm_stats, episode_horizon=2, step_seconds=0.05)
    assert res["steps"] == 2
    assert res["plans"][0]["source_time"] == 0.0


def test_invalid_shape_nan_handling_and_hook_cleanup() -> None:
    """Verify NaN actions trigger ValueError and writer hook is reliably removed."""
    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24, return_nan=True)
    env = FakeGymnasiumEnv(dt=0.1)
    norm_stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }

    initial_hook_count = len(policy.writer._forward_hooks)
    with pytest.raises(ValueError, match="non-finite action values"):
        run_episode(env, policy, "test nan", norm_stats, episode_horizon=3)

    assert len(policy.writer._forward_hooks) == initial_hook_count


@pytest.mark.parametrize("seed_policy", ["once", "per-plan"])
def test_run_evaluation_orchestrator_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seed_policy: str) -> None:
    """Verify run_evaluation end-to-end with manifest resolution, incremental output, and summary."""
    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    out_dir = tmp_path / "output_eval"

    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }
    ckpt_meta = {"horizon": 5, "state_dim": 24, "action_dim": 24, "image_size": 448}
    monkeypatch.setattr(pi, "load_predictive_inference", lambda **kwargs: (policy, stats, ckpt_meta))

    wrapper = FakeGymWrapper([FakeGymnasiumEnv(dt=0.1), FakeGymnasiumEnv(dt=0.1)])

    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", str(tmp_path / "dummy.pt"),
        "--output-dir", str(out_dir),
        "--fabri-root", str(fabri_root),
        "--tasks", "all",
        "--episodes", "2",
        "--episode-horizon", "3",
        "--exec-horizon", "2",
        "--device", "cpu",
        "--seed-policy", seed_policy,
    ])
    if seed_policy == "per-plan":
        monkeypatch.setattr(torch, "manual_seed", lambda *_: pytest.fail("Do not seed all CUDA devices"))

    summary = run_evaluation(args, device="cpu", env_factory=lambda seed: wrapper)

    assert summary["status"] == "completed"
    assert summary["counts"]["planned_episodes"] == 4
    assert summary["counts"]["completed_episodes"] == 4
    assert wrapper.closed is True

    manifest_path = out_dir / "manifest.json"
    episodes_path = out_dir / "episodes.jsonl"
    summary_path = out_dir / "summary_report.json"
    assert manifest_path.exists()
    assert episodes_path.exists()
    assert summary_path.exists()

    with episodes_path.open("r", encoding="utf-8") as f:
        ep_lines = [json.loads(line) for line in f if line.strip()]
    assert len(ep_lines) == 4
    assert all(env.reset_seeds == [4048, 4049] for env in wrapper.envs)
    if seed_policy == "per-plan":
        assert summary["provenance"]["seed_policy"] == "per_plan_torch_sha256_v1"
        assert "master_seed + episode_index" in summary["provenance"]["plan_rng_scope"]
        assert len({ep["plans"][0]["torch_seed"] for ep in ep_lines}) == 4
    else:
        assert summary["provenance"]["seed_policy"] == "explicit_after_load_once"
        assert all("torch_seed" not in plan for ep in ep_lines for plan in ep["plans"])

    assert "overall" in summary
    assert "difficulty_groups" in summary
    assert "easy" in summary["difficulty_groups"]
    assert "hard" in summary["difficulty_groups"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))])
def test_per_plan_noise_pairing_and_rng_restoration(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    stats = {"action": {"min": [-1.] * 24, "max": [1.] * 24},
             "observation.state": {"min": [-1.] * 24, "max": [1.] * 24}}
    original_fork_rng = torch.random.fork_rng

    def selected_device_fork(*, devices, enabled):
        assert devices == ([torch.device(device).index] if device.startswith("cuda") else [])
        assert enabled is True
        return original_fork_rng(devices=devices)

    monkeypatch.setattr(torch.random, "fork_rng", selected_device_fork)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: pytest.fail("Do not seed other CUDA devices"))

    def episode(task_slug="reach-v2", episode_index=2, success_at_step=None):
        policy = FakeNoisyPolicy().to(device)
        env = FakeGymnasiumEnv(success_at_step=success_at_step)
        cpu_rng = torch.random.get_rng_state()
        cuda_rngs = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        result = run_episode(
            env, policy, "test", stats, episode_horizon=11, exec_horizon=5, image_size=16,
            seed=2468, seed_policy="per-plan", master_seed=4048,
            task_slug=task_slug, episode_index=episode_index,
        )
        assert env.reset_seeds == [2468]
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        for i, before in enumerate(cuda_rngs):
            assert torch.equal(before, torch.cuda.get_rng_state(i))
        return result, policy.noises

    reference, noises = episode()
    assert reference["plans"][0]["torch_seed"] == 7911832927264572434
    assert [p["source_step"] for p in reference["plans"]] == [0, 5, 10]
    assert [c["offset"] for c in reference["command_log"]] == list(range(5)) * 2 + [0]
    assert len({p["torch_seed"] for p in reference["plans"]}) == 3
    assert not torch.equal(noises[0], noises[1])

    for success_at_step in (1, 8):
        # Earlier tasks can finish at different times and unrelated code can
        # consume arbitrary noise without moving the paired target's stream.
        earlier, _ = episode("push-v2", 0, success_at_step)
        assert earlier["steps"] == success_at_step + 1
        torch.randn(17 * earlier["steps"], device=device)
        repeated, repeated_noises = episode()
        assert [p["torch_seed"] for p in repeated["plans"]] == [p["torch_seed"] for p in reference["plans"]]
        assert all(torch.equal(a, b) for a, b in zip(noises, repeated_noises))

    for task_slug, episode_index in (("push-v2", 2), ("reach-v2", 3)):
        changed, changed_noises = episode(task_slug, episode_index)
        assert changed["plans"][0]["torch_seed"] != reference["plans"][0]["torch_seed"]
        assert not torch.equal(noises[0], changed_noises[0])


def test_default_once_keeps_consuming_the_existing_torch_stream() -> None:
    assert build_parser().parse_args(["--checkpoint", "unused", "--output-dir", "unused"]).seed_policy == "once"
    policy = FakeNoisyPolicy()
    state = torch.random.get_rng_state()
    expected = [torch.randn(8) for _ in range(2)]
    expected_final = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    stats = {"action": {"min": [-1.] * 24, "max": [1.] * 24},
             "observation.state": {"min": [-1.] * 24, "max": [1.] * 24}}
    result = run_episode(FakeGymnasiumEnv(), policy, "test", stats, episode_horizon=6, image_size=16)
    assert all(torch.equal(a, b) for a, b in zip(policy.noises, expected))
    assert torch.equal(torch.random.get_rng_state(), expected_final)
    assert all("torch_seed" not in plan for plan in result["plans"])


def test_per_plan_requires_explicit_episode_identity() -> None:
    env = FakeGymnasiumEnv()
    with pytest.raises(ValueError, match="master_seed"):
        run_episode(env, FakeScriptedPolicy(), "test", {}, seed_policy="per-plan")
    with pytest.raises(ValueError, match="episode_index"):
        run_episode(env, FakeScriptedPolicy(), "test", {}, seed_policy="per-plan", master_seed=1, task_slug="reach-v2")
    assert env.reset_seeds == []


def test_partial_failure_flushes_failed_summary_and_closes_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that an episode failure saves previous entries, writes failed summary, and closes env."""
    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    out_dir = tmp_path / "output_fail"

    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }
    monkeypatch.setattr(pi, "load_predictive_inference", lambda **kwargs: (policy, stats, {}))

    env1 = FakeGymnasiumEnv(dt=0.1)
    env2 = FakeGymnasiumEnv(dt=0.1, fail_on_reset=True)
    wrapper = FakeGymWrapper([env1, env2])

    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", str(tmp_path / "dummy.pt"),
        "--output-dir", str(out_dir),
        "--fabri-root", str(fabri_root),
        "--tasks", "reach-v2", "push-v2",
        "--episodes", "1",
        "--episode-horizon", "2",
        "--device", "cpu",
    ])

    with pytest.raises(RuntimeError, match="FakeEnv simulated reset failure"):
        run_evaluation(args, device="cpu", env_factory=lambda seed: wrapper)

    assert wrapper.closed is True
    summary_path = out_dir / "summary_report.json"
    with summary_path.open("r", encoding="utf-8") as f:
        fail_summary = json.load(f)

    assert fail_summary["status"] == "failed"
    assert fail_summary["counts"]["completed_episodes"] == 1
    assert "FakeEnv simulated reset failure" in fail_summary["error"]


def test_refuse_overwrite_nonempty_directory(tmp_path: Path) -> None:
    """Verify run_evaluation strictly refuses to overwrite an existing non-empty output directory."""
    non_empty_dir = tmp_path / "existing_dir"
    non_empty_dir.mkdir(parents=True)
    (non_empty_dir / "stray_file.txt").write_text("content")

    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", str(tmp_path / "dummy.pt"),
        "--output-dir", str(non_empty_dir),
        "--fabri-root", str(fabri_root),
        "--tasks", "reach-v2",
    ])

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_evaluation(args, device="cpu")


def test_manifest_validation_rejects_duplicate_or_unknown_tasks(tmp_path: Path) -> None:
    """Verify task validation rejects duplicates, unknown slugs, and 'all' mixed with slugs."""
    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")

    with pytest.raises(ValueError, match="Duplicate task slug"):
        resolve_task_manifest(["reach-v2", "reach-v2"], fabri_root=fabri_root)

    with pytest.raises(ValueError, match="Unknown task slug"):
        resolve_task_manifest(["non-existent-v2"], fabri_root=fabri_root)

    with pytest.raises(ValueError, match="'all' cannot be combined"):
        resolve_task_manifest(["all", "reach-v2"], fabri_root=fabri_root)


def test_summary_distinguishes_partial_and_rejects_duplicate_episodes():
    manifest = [{"task_slug": "a", "env_task_index": 0, "group": "easy"},
                {"task_slug": "b", "env_task_index": 1, "group": "hard"}]
    episodes = [{"task_slug": "a", "episode_index": 0, "success": True},
                {"task_slug": "a", "episode_index": 1, "success": False},
                {"task_slug": "b", "episode_index": 0, "success": True}]
    summary = compute_evaluation_summary(manifest, episodes, 2, {"easy": ["a"], "hard": ["b"]}, status="failed", is_full_mt50=True)
    assert summary["complete"] is False and summary["is_full_mt50_benchmark"] is False
    assert summary["counts"]["planned_episodes"] == 4
    assert summary["overall"]["success_rate"] == pytest.approx(2 / 3)
    assert summary["overall"]["task_macro_success_rate"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="duplicate or invalid"):
        compute_evaluation_summary(manifest, episodes + [episodes[0]], 2, {})
    episodes.append({"task_slug": "b", "episode_index": 1, "success": False})
    assert compute_evaluation_summary(manifest, episodes, 2, {}, status="completed")["complete"] is True


def test_episode_reset_no_seed_fallback_and_dict_observations():
    stats = {"action": {"min": [-1.] * 4, "max": [1.] * 4},
             "observation.state": {"min": [-1.] * 4, "max": [1.] * 4}}
    policy = FakeScriptedPolicy()
    env = FakeGymnasiumEnv()
    for _ in range(2):
        result = run_episode(env, policy, "task", stats, episode_horizon=6, exec_horizon=3)
        assert result["plans"][0]["history_frame_ids"] == [0]
        assert result["plans"][0]["decision_frame_ids"] == [0]
    calls = []
    def invalid_reset(seed=None):
        calls.append(seed)
        raise TypeError("seeded reset failed")
    env.reset = invalid_reset
    with pytest.raises(TypeError, match="seeded reset failed"):
        run_episode(env, policy, "task", stats)
    assert len(calls) == 1
    env = FakeGymnasiumEnv()
    old_step = env.step
    def dict_step(action):
        obs, reward, term, trunc, info = old_step(action)
        return {"observation": obs}, reward, term, trunc, info
    env.step = dict_step
    result = run_episode(env, policy, "task", stats, episode_horizon=4, exec_horizon=2)
    assert result["steps"] == 4


def test_loader_failure_is_reported(monkeypatch, tmp_path):
    root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    def failed_loader(**kwargs):
        raise ValueError("checkpoint mismatch")
    monkeypatch.setattr(pi, "load_predictive_inference", failed_loader)
    out = tmp_path / "results"
    args = build_parser().parse_args(["--checkpoint", str(tmp_path / "unused.pt"), "--output-dir", str(out), "--fabri-root", str(root)])
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        run_evaluation(args, device="cpu", env_factory=lambda seed: pytest.fail("Environment must not start after load failure"))
    summary = json.loads((out / "summary_report.json").read_text())
    assert summary["status"] == "failed"
    assert summary["counts"]["completed_episodes"] == 0


def test_env_coverage_failure_closes_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An invalid environment task mapping must not leak environment resources."""
    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    out_dir = tmp_path / "output_coverage"

    policy = FakeScriptedPolicy(horizon=5, per_action_dim=24)
    stats = {
        "action": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
        "observation.state": {"min": np.array([-1.0] * 24), "max": np.array([1.0] * 24)},
    }
    monkeypatch.setattr(pi, "load_predictive_inference", lambda **kwargs: (policy, stats, {}))

    wrapper = FakeGymWrapper([FakeGymnasiumEnv(dt=0.1)])

    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", str(tmp_path / "dummy.pt"),
        "--output-dir", str(out_dir),
        "--fabri-root", str(fabri_root),
        "--tasks", "all",
        "--device", "cpu",
    ])

    with pytest.raises(RuntimeError, match="cannot cover required task index"):
        run_evaluation(args, device="cpu", env_factory=lambda seed: wrapper)

    assert wrapper.closed is True
    summary = json.loads((out_dir / "summary_report.json").read_text())
    assert summary["status"] == "failed"
    assert summary["counts"]["completed_episodes"] == 0
