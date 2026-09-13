"""Integration test for CPU tinyQwen compact memory training and resume lifecycle."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict
import pytest
import torch

from fabri_moss.compact_memory import CompactMemoryConfig
from fabri_moss.compact_protocol import get_compact_protocol_contract
import fabri_moss.compact_training as compact_training
from fabri_moss.periodic_memory import PeriodicMemoryConfig
from fabri_moss.tests.test_memory_training import MockTokenizerForMemory
from fabri_moss.tests.test_native_training import make_sample, make_tiny_training_policy
from fabri_moss.tests.test_stream_transition import MockStreamTrainingDataset
import fabri_moss.train_native as tn


class ToyCompactTrainingDataset(MockStreamTrainingDataset):
    """Toy dataset emitting N=6 target=[4, 5] samples with deterministic local fork_rng."""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + idx)
            sample = make_sample(N=6, target_indices=[4, 5], horizon=2, action_dim=4)
            sample["state"] = torch.zeros(2, 4)
            sample["state_mask"] = torch.ones(2, 4, dtype=torch.bool)
            sample["action_mask"] = torch.ones(2, 2, 4, dtype=torch.bool)
            sample["episode_id"] = 0
            sample["target_frame_ids"] = [4, 5]
            sample["target_count"] = 2
            sample["frame_ids"] = list(range(6))
            if self.stream_protocol == "compact_memory_replay_v1":
                sample["memory_replay"] = True
                sample["decision_indices"] = [1, 3, 4, 5]
            return sample

    def get_data_contract(self) -> Dict[str, Any]:
        contract = self.get_base_data_contract()
        if self.stream_protocol == "compact_memory_replay_v1":
            contract["stream_protocol"] = get_compact_protocol_contract()
            contract["data_fingerprint"] = f"compact_dfp_{self.split}"
        return contract


def test_compact_main_cpu_training_and_resume_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify train_native.main compact memory lifecycle: dispatch, real training update, and resume continuation."""
    output_dir = tmp_path / "compact_run"
    construction_count = 0
    orig_compact_cls = compact_training.NativeCompactMemorySequencePolicy

    def mock_compact_factory(*args: Any, **kwargs: Any):
        nonlocal construction_count
        construction_count += 1
        kwargs.pop("shallow_layer", None)
        kwargs["shallow_layer"] = 1
        kwargs["compact_config"] = CompactMemoryConfig(
            memory=PeriodicMemoryConfig(
                recent_frames=2,
                consolidate_every=2,
                memory_slots=2,
                spatial_grid=1,
                protect_decision_frames=True,
            ),
            intermediate_grid=1,
        )
        return orig_compact_cls(*args, **kwargs)

    def mock_load_checkpoint(**kw: Any):
        seq_policy = make_tiny_training_policy(shallow_layer=1, horizon=2, action_dim=4)
        policy = seq_policy.policy
        policy.action_head.config.shallow_layer_index = 1
        policy.action_head.config.state_dim = 4
        policy.action_head.config.per_action_dim = 4
        policy.action_head.config.horizon = 2
        policy.embedder.tokenizer = MockTokenizerForMemory(policy.embedder.tokenizer)
        policy.embedder.model.mlp1 = policy.embedder.model.projector
        del policy.embedder.model.projector
        policy.embedder.model.extract_feature = lambda pv: policy.embedder.model.mlp1(
            policy.embedder.model.vision_model.encoder(pv)
        )
        return policy, {"dim": 64}, {}, {"checkpoint_sha256": "toy_sha256_compact"}

    # Pure CPU test skips device and FA2 guards
    monkeypatch.setattr(tn, "require_training_device", lambda p, d: None)
    monkeypatch.setattr("fabri_moss.runtime.assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr("fabri_moss.runtime.load_native_checkpoint", mock_load_checkpoint)
    monkeypatch.setattr("fabri_moss.compact_training.NativeCompactMemorySequencePolicy", mock_compact_factory)
    monkeypatch.setattr("fabri_moss.native_data.NativeTrainingDataset", ToyCompactTrainingDataset)

    base_args = [
        "--output-dir", str(output_dir),
        "--checkpoint", "dummy_base.pt",
        "--data-root", "dummy_root",
        "--stream-protocol", "compact_memory_replay_v1",
        "--epochs", "2",
        "--global-batch-size", "4",
        "--workers", "0",
        "--seed", "42",
        "--val-fraction", "0.1",
        "--lr-vision", "1e-4",
        "--lr-projector", "1e-4",
        "--lr-llm", "1e-4",
        "--lr-head", "1e-4",
        "--warmup-updates", "2",
        "--save-every", "10",
        "--eval-every", "10",
        "--val-per-task", "1",
        "--grad-clip", "1.0",
    ]

    # Run 1: Initial training to max_updates=1
    tn.main(base_args + ["--max-updates", "1"], device=torch.device("cpu"))

    assert construction_count == 1
    last_path = output_dir / "last.pt"
    assert last_path.exists()
    ckpt1 = torch.load(str(last_path), weights_only=False)

    assert ckpt1["global_step"] == 1
    assert ckpt1["batch_cursor"] == 4
    assert ckpt1["training_contract"]["format"] == "native_compact_memory_replay_v1"
    assert ckpt1["training_contract"]["stream_protocol"]["protocol_name"] == "compact_memory_replay_v1"
    assert ckpt1["train_data_contract"]["stream_protocol"]["protocol_name"] == "compact_memory_replay_v1"

    # Optimizer state verification for update 1 across parameter groups
    opt_states1 = ckpt1["optimizer"]["state"]
    assert len(opt_states1) > 0
    assert all(s["step"] == 1 for s in opt_states1.values())
    weights1 = copy.deepcopy(ckpt1["model"])

    # Run 2: Resume to max_updates=2
    resume_args = base_args + ["--max-updates", "2", "--resume", str(last_path)]
    tn.main(resume_args, device=torch.device("cpu"))

    assert construction_count == 2
    ckpt2 = torch.load(str(last_path), weights_only=False)

    assert ckpt2["global_step"] == 2
    assert ckpt2["batch_cursor"] == 8
    assert "rng_states_per_rank" in ckpt2 and len(ckpt2["rng_states_per_rank"]) >= 1
    assert "torch_cpu" in ckpt2["rng_states_per_rank"][0]
    assert "source_metadata" in ckpt2

    # Optimizer state verification for update 2
    opt_states2 = ckpt2["optimizer"]["state"]
    assert len(opt_states2) == len(opt_states1)
    assert all(s["step"] == 2 for s in opt_states2.values())

    # Weights actually updated across steps
    weight_diffs = [
        k for k, w1 in weights1.items()
        if not torch.equal(w1, ckpt2["model"][k])
    ]
    assert len(weight_diffs) > 0
