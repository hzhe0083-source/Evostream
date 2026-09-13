"""Deployment-cadence replay matches independent schedules without re-encoding images."""

import copy

import pytest
import torch

from fabri_moss.tests.test_joint_predictive_policy import _build_toy_joint_policy
from fabri_moss.tests.test_native_training import make_sample
from fabri_moss.tests.test_predictive_policy import _build_toy_policy


def _grouped_sample():
    sample = make_sample(N=23, target_indices=list(range(10, 23)))
    sample["memory_replay"] = True
    sample["replay_groups"] = []
    for phase in range(5):
        positions = [p for p, row in enumerate(sample["target_indices"]) if row % 5 == phase]
        last_target = sample["target_indices"][positions[-1]]
        sample["replay_groups"].append({
            "observation_indices": list(range(last_target + 1)),
            "target_positions": positions,
            "decision_indices": sorted({0, *range(phase, last_target + 1, 5)}),
        })
    return sample


def _standalone(sample, group):
    result = dict(sample)
    result.pop("replay_groups")
    for key in ("images_window", "frame_ids", "observation_times"):
        result[key] = [sample[key][i] for i in group["observation_indices"]]
    result["target_indices"] = [sample["target_indices"][p] for p in group["target_positions"]]
    result["decision_indices"] = group["decision_indices"]
    for key in ("actions", "action_mask"):
        result[key] = sample[key][group["target_positions"]]
    return result


@pytest.mark.parametrize("builder", [_build_toy_policy, _build_toy_joint_policy])
@pytest.mark.parametrize("tbptt_decisions", [1, 2])
def test_grouped_features_and_gradients_match_independent_replays(builder, tbptt_decisions, monkeypatch):
    torch.manual_seed(17)
    policy = builder(tbptt_decisions=tbptt_decisions).eval()
    # Exercise upstream writer gradients too, beyond its zero-initialized output projection.
    with torch.no_grad():
        policy.writer.out_proj.weight.normal_(std=0.02)
    sample = _grouped_sample()
    reference = copy.deepcopy(policy)

    encoded_batches = []
    original_extract = policy.policy.embedder.model.extract_feature

    def record_extract(pixels):
        encoded_batches.append((len(pixels), torch.is_grad_enabled()))
        return original_extract(pixels)

    monkeypatch.setattr(policy.policy.embedder.model, "extract_feature", record_extract)
    deep, shallow = policy.features(sample)
    assert sum(size for size, _ in encoded_batches) == len(sample["images_window"])
    expected_grad_start = min(
        policy._observation_grad_start(dict(
            sample,
            target_indices=[sample["target_indices"][p] for p in group["target_positions"]],
            decision_indices=group["decision_indices"],
        ))
        for group in sample["replay_groups"]
    )
    assert sum(size for size, has_grad in encoded_batches if not has_grad) == expected_grad_start

    deep_weights, shallow_weights = torch.randn_like(deep), torch.randn_like(shallow)
    ((deep * deep_weights).sum() + (shallow * shallow_weights).sum()).backward()
    for group in sample["replay_groups"]:
        positions = group["target_positions"]
        group_deep, group_shallow = reference.features(_standalone(sample, group))
        torch.testing.assert_close(deep[positions], group_deep, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(shallow[positions], group_shallow, atol=2e-6, rtol=2e-5)
        ((group_deep * deep_weights[positions]).sum()
         + (group_shallow * shallow_weights[positions]).sum()).backward()

    assert policy.writer.out_proj.weight.grad.abs().sum() > 0
    assert policy.writer.input_proj.weight.grad.abs().sum() > 0
    for name, parameter in policy.named_parameters():
        expected = dict(reference.named_parameters())[name].grad
        if expected is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=2e-4, rtol=2e-4, msg=name)


def test_group_order_and_future_observations_do_not_change_earlier_targets():
    torch.manual_seed(19)
    policy = _build_toy_policy().eval()
    sample = _grouped_sample()
    with torch.no_grad():
        expected = policy.features(sample)
        sample["replay_groups"].reverse()
        reordered = policy.features(sample)
        for actual, original in zip(reordered, expected):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
        cutoff = 15
        for row in range(cutoff + 1, len(sample["images_window"])):
            sample["images_window"][row] = torch.randn_like(sample["images_window"][row]) * 50
        perturbed = policy.features(sample)
    earlier = [p for p, row in enumerate(sample["target_indices"]) if row <= cutoff]
    for actual, original in zip(perturbed, expected):
        torch.testing.assert_close(actual[earlier], original[earlier], atol=0, rtol=0)
    assert not torch.equal(perturbed[0][-1], expected[0][-1])


@pytest.mark.parametrize("decisions, error", [
    (None, ValueError),
    ([], ValueError),
    ([False, 5, 10, 15, 20], TypeError),
    ([0, 5, 5, 10, 15, 20], ValueError),
    ([0, 5, 15, 20], ValueError),
    ([0, 5, 10, 15, 20, 21], ValueError),
    ([0, 5, 10, 15, 20, 23], IndexError),
])
def test_group_decisions_are_required_and_validated(decisions, error):
    sample = _grouped_sample()
    if decisions is None:
        sample["replay_groups"][0].pop("decision_indices")
        # Global metadata must never substitute for a group's own decision schedule.
        sample["decision_indices"] = list(range(23))
    else:
        sample["replay_groups"][0]["decision_indices"] = decisions
    with pytest.raises(error, match="decision"):
        _build_toy_policy().features(sample)


def test_group_target_coverage_and_no_post_target_observations():
    policy = _build_toy_policy()
    sample = _grouped_sample()
    sample["replay_groups"].append(copy.deepcopy(sample["replay_groups"][0]))
    with pytest.raises(ValueError, match="exactly once"):
        policy.features(sample)
    sample = _grouped_sample()
    sample["replay_groups"][0]["observation_indices"].append(21)
    with pytest.raises(ValueError, match="trailing"):
        policy.features(sample)


def test_grouped_expert_action_loss_reaches_writer_and_counts_every_target():
    policy = _build_toy_policy().train()
    sample = _grouped_sample()
    output = policy(sample, future_weight=0.0)
    assert output["target_count"] == len(sample["target_indices"])
    assert torch.isfinite(output["action_loss"])
    output["action_loss"].backward()
    assert policy.writer.out_proj.weight.grad.abs().sum() > 0
