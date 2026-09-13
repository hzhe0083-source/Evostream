"""CPU contract checks for the official MOSS evaluation entrypoint."""

from __future__ import annotations

import torch

from fabri_moss.evaluate_async import (
    OFFICIAL_EPISODE_HORIZON,
    OFFICIAL_EXEC_HORIZON,
    OFFICIAL_FLOW_STEPS,
    OFFICIAL_MT50_EPISODES_PER_TASK,
    OFFICIAL_MT50_TASK_COUNT,
    derive_plan_seed,
    fixed_diffusion_seed,
    official_mt50_contract,
)
from fabri_moss.evaluate_moss import build_parser, compute_moss_evaluation_summary


def test_plan_seed_is_stable_and_plan_scoped() -> None:
    a = derive_plan_seed(4048, "reach-v3", 0, 5)
    assert a == derive_plan_seed(4048, "reach-v3", 0, 5)
    assert a != derive_plan_seed(4048, "reach-v3", 0, 10)
    assert a != derive_plan_seed(4048, "push-v3", 0, 5)


def test_fixed_diffusion_seed_restores_cpu_rng() -> None:
    state = torch.random.get_rng_state()
    with fixed_diffusion_seed(1234, "cpu"):
        first = torch.randn(4)
    assert torch.equal(state, torch.random.get_rng_state())
    with fixed_diffusion_seed(1234, "cpu"):
        second = torch.randn(4)
    assert torch.equal(first, second)


def test_official_contract_requires_all_500_episodes() -> None:
    contract = official_mt50_contract(
        task_count=OFFICIAL_MT50_TASK_COUNT,
        episodes_per_task=OFFICIAL_MT50_EPISODES_PER_TASK,
        episode_horizon=OFFICIAL_EPISODE_HORIZON,
        exec_horizon=OFFICIAL_EXEC_HORIZON,
        flow_steps=OFFICIAL_FLOW_STEPS,
    )
    assert contract["official_mt50_500"] is True
    partial = official_mt50_contract(
        task_count=1,
        episodes_per_task=1,
        episode_horizon=OFFICIAL_EPISODE_HORIZON,
        exec_horizon=OFFICIAL_EXEC_HORIZON,
        flow_steps=OFFICIAL_FLOW_STEPS,
    )
    assert partial["official_mt50_500"] is False


def test_summary_keeps_partial_moss_scope_explicit() -> None:
    manifest = [{"task_slug": "reach-v3", "env_task_index": 0, "group": "easy"}]
    summary = compute_moss_evaluation_summary(
        manifest,
        [{"task_slug": "reach-v3", "episode_index": 0, "success": True}],
        2,
        status="completed",
        contract={"official_mt50_500": False},
        provenance={"source_checkpoint_sha256": "source"},
    )
    assert summary["official_mt50_500"] is False
    assert summary["evaluation_kind"] == "moss_sync_partial_or_single_task"
    assert summary["comparison_boundary"]["async_success_rate"] is None


def test_parser_uses_official_defaults() -> None:
    args = build_parser().parse_args(["--checkpoint", "base.pt", "--adapter", "adapter.pt", "--output-dir", "out"])
    assert (args.episodes, args.episode_horizon, args.exec_horizon, args.num_inference_timesteps) == (10, 400, 5, 50)
    assert args.seed_policy == "per-plan"
