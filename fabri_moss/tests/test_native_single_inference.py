"""Original FabriVLA API baseline, checked without loading the large model."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fabri_moss.native_single_inference as native
from fabri_moss.evaluate_predictive import build_parser, run_episode, run_evaluation
from fabri_moss.runtime import compute_file_sha256
from fabri_moss.tests.test_evaluate_predictive import (
    FakeGymnasiumEnv, FakeGymWrapper, _make_toy_metadata_workspace,
)


class FakeOriginalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.000123], dtype=torch.float32))
        self.action_head = SimpleNamespace(config=SimpleNamespace(
            horizon=5, state_dim=24, per_action_dim=24, action_dim=120,
            num_inference_timesteps=50,
        ))
        self.calls = []

    def run_inference(self, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        if self.weight.device.type == "cuda":
            assert torch.is_autocast_enabled("cuda")
            assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        else:
            assert not torch.is_autocast_enabled("cpu")
        self.calls.append(kwargs)
        return torch.randn(1, 5, 24, device=self.weight.device) * 0.1


def make_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "original93k.pt"
    config = {"horizon": 5, "state_dim": 24, "action_dim": 120, "image_size": 16}
    stats = {
        "action": {"min": [-1.0] * 24, "max": [1.0] * 24},
        "observation.state": {"min": [-1.0] * 24, "max": [1.0] * 24},
    }
    original = FakeOriginalPolicy()
    torch.save({"model": original.state_dict(), "config": config, "norm_stats": stats, "step": 93000}, source)
    loads = []

    def load_base(**kwargs):
        loads.append(kwargs)
        assert kwargs["trainable"] is True
        assert kwargs["checkpoint_path"] != source
        payload = torch.load(kwargs["checkpoint_path"], weights_only=False)
        policy = FakeOriginalPolicy()
        policy.load_state_dict(payload["model"], strict=True)
        return policy, config, stats, {"step": payload["step"]}

    monkeypatch.setattr(native, "load_native_checkpoint", load_base)
    return source, original, stats, loads


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable",
))])
def test_original_wrapper_passes_only_latest_observation(device):
    base = FakeOriginalPolicy().to(device)
    policy = native.NativeSingleInferencePolicy(base)
    images = [[object()], [object(), object()]]
    sample = {
        "images_window": images, "prompt": "open the drawer",
        "state": torch.arange(48).reshape(2, 24).float(),
        "state_mask": torch.arange(48).reshape(2, 24) > 10,
        "action_mask": torch.arange(48).reshape(2, 24) > 30,
        "observation_times": [0.1, 0.2], "frame_ids": [1, 2],
        "memory_replay": True, "decision_indices": [0, 1],
    }
    assert policy.predict_actions(sample).shape == (1, 5, 24)
    called = base.calls[0]
    assert set(called) == {"images", "image_mask", "prompt", "state", "state_mask", "action_mask"}
    assert called["images"] == images[-1]
    assert called["prompt"] == sample["prompt"]
    assert called["image_mask"].dtype == torch.bool and called["image_mask"].tolist() == [True, True]
    for key in ("state", "state_mask", "action_mask"):
        torch.testing.assert_close(called[key], sample[key][-1:].to(device))
    assert list(policy.parameters()) == list(base.parameters())
    assert all(not p.requires_grad for p in policy.parameters())
    assert not hasattr(policy, "writer") and not hasattr(policy, "future_head")


def test_original_loader_snapshots_preserves_weights_and_records_dimensions(tmp_path, monkeypatch):
    source, original, stats, loads = make_checkpoint(tmp_path, monkeypatch)
    sha256 = compute_file_sha256(source)
    model, loaded_stats, metadata = native.load_native_single_inference(
        source, snapshot_dir=tmp_path / "snapshots", expected_sha256=sha256,
    )
    assert loads and metadata["checkpoint_sha256"] == sha256
    assert loaded_stats == stats
    assert torch.equal(model.policy.weight, original.weight)
    assert model.policy.weight.dtype == torch.float32
    assert not model.training and not model.policy.weight.requires_grad
    assert metadata["global_step"] == 93000
    assert (metadata["horizon"], metadata["state_dim"], metadata["action_dim"]) == (5, 24, 24)
    assert metadata["image_size"] == 16
    assert metadata["memory_kind"] == "native_single_frame"
    assert metadata["learned_writer_enabled"] is False
    assert metadata["new_trainable_params"] == metadata["writer_params_loaded"] == 0
    assert metadata["historical_benchmark_precision_reproduced"] is False


@pytest.mark.parametrize("extra", [{"writer": {}}, {"format": "predictive_memory_joint_v1"}])
def test_original_loader_rejects_memory_checkpoint(tmp_path, monkeypatch, extra):
    source, _, _, loads = make_checkpoint(tmp_path, monkeypatch)
    payload = torch.load(source, weights_only=False)
    torch.save(dict(payload, **extra), source)
    with pytest.raises(ValueError, match="original bare"):
        native.load_native_single_inference(source, snapshot_dir=tmp_path / "snapshots")
    assert not loads


def test_original_loader_rejects_wrong_sha_before_loading(tmp_path, monkeypatch):
    source, _, _, loads = make_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        native.load_native_single_inference(source, snapshot_dir=tmp_path / "snapshots", expected_sha256="wrong")
    assert not loads


@pytest.mark.parametrize("seed_policy", ["once", "per-plan"])
def test_original_evaluation_uses_latest_image_and_original_api(tmp_path, monkeypatch, seed_policy):
    source, _, _, loads = make_checkpoint(tmp_path, monkeypatch)
    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    output = tmp_path / "evaluation"
    wrapper = FakeGymWrapper([FakeGymnasiumEnv(), FakeGymnasiumEnv()])
    loaded_models = []
    load_base = native.load_native_checkpoint

    def remember_model(**kwargs):
        result = load_base(**kwargs)
        loaded_models.append(result[0])
        return result

    monkeypatch.setattr(native, "load_native_checkpoint", remember_model)
    args = build_parser().parse_args([
        "--policy-kind", "native-single", "--checkpoint", str(source),
        "--output-dir", str(output), "--fabri-root", str(fabri_root),
        "--episodes", "1", "--episode-horizon", "6", "--exec-horizon", "5",
        "--seed-policy", seed_policy,
    ])
    summary = run_evaluation(args, device="cpu", env_factory=lambda seed: wrapper)
    assert summary["status"] == "completed" and wrapper.closed and loads
    provenance = summary["provenance"]
    assert provenance["memory_kind"] == "native_single_frame"
    assert provenance["learned_writer_enabled"] is False
    assert "native_single_inference.py" in provenance["source_hashes"]
    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    for episode in episodes:
        assert episode["total_writer_calls"] == 0
        assert episode["history_final_frames"] == list(range(6))
        assert [plan["history_frames_count"] for plan in episode["plans"]] == [1, 6]
        assert all(plan["model_input_frames_count"] == 1 for plan in episode["plans"])
        assert all(("torch_seed" in plan) == (seed_policy == "per-plan") for plan in episode["plans"])
    calls = loaded_models[0].calls
    assert len(calls) == 4
    for call, env_steps in zip(calls, (1, 6, 1, 6)):
        assert len(call["images"]) == 1
        assert np.asarray(call["images"][0])[0, 0, 0] == env_steps * 17


def test_native_episode_forbids_writer():
    policy = native.NativeSingleInferencePolicy(FakeOriginalPolicy())
    policy.writer = nn.Identity()
    with pytest.raises(ValueError, match="forbids learned writer"):
        run_episode(FakeGymnasiumEnv(), policy, "reach", {}, policy_kind="native-single")
