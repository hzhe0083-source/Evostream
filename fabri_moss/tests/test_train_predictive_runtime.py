"""Runtime and full lifecycle integration tests for train_predictive.run_training.

Tests:
1. test_predictive_training_uninterrupted_vs_resumed_exactness:
   Uninterrupted 4 updates vs max-updates 2 then resume to 4 updates under same budget:
   - final writer / future_head / optimizer / scheduler exact bitwise match.
   - python / numpy / torch CPU RNG states exact match across ranks.
   - final epoch, batch_cursor, and epoch_targets_seen identical.
   - stage1.pt exists and stage2.pt absent at update 2; stage2.pt exists after update 4.
   - future_head has no optimizer state at update 2, and optimizer state exists at update 4.
   - base backbone parameters remain strictly unchanged throughout.
2. test_completed_resume_idempotent_hashes:
   Resuming an already-completed run (completed_updates >= total_target_updates)
   leaves run_config.json, metrics.jsonl, and last.pt completely untouched.
3. test_immediate_stopfile_preserves_cursor_and_stage:
   Resuming with an immediate stop-file triggered on the first check:
   - cleanly exits without crashing on unbound stage_name.
   - saves checkpoint with exact same cursor and update count.
4. test_skip_batch_advances_cursor_and_metrics:
   Dataset returns dense first batch (no writer graph, requires_grad=False)
   followed by memory batch:
   - skip batch advances cursor and epoch_targets_seen without optimizer step.
   - skipped_batch metric logged to metrics.jsonl with epoch, cursor, and targets.
   - update 1 consumes the skipped batch and advances cursor correctly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from fabri_moss.compact_protocol import get_compact_protocol_contract
import fabri_moss.predictive_data as predictive_data
from fabri_moss.predictive_memory import WriterConfig
import fabri_moss.predictive_policy as predictive_policy
import fabri_moss.runtime as runtime
from fabri_moss.tests.test_native_training import make_sample, make_tiny_training_policy
from fabri_moss.train_native import compute_epoch_batches
import fabri_moss.train_predictive as train_predictive
from fabri_moss.train_predictive import parse_args, run_training


# ============================================================================
# Deterministic Mock Dataset & Checkpoint Fixtures
# ============================================================================


class FakePredictiveTrainingDataset:
    """Deterministic mock PredictiveTrainingDataset with contract enforcement."""

    first_batch_dense: bool = False

    def __init__(
        self,
        root: str = "",
        norm_stats: Optional[Dict[str, Any]] = None,
        split: str = "train",
        history_frames: int = 16,
        target_frames: int = 8,
        augmentation: bool = False,
        seed: int = 4042,
        val_fraction: float = 0.1,
        max_episodes: Optional[int] = None,
        **kwargs: Any,
    ):
        self.root = root
        self.norm_stats = norm_stats or {}
        self.split = split
        self.history_frames = history_frames
        self.target_frames = target_frames
        self.augmentation = augmentation
        self.seed = seed
        self.val_fraction = val_fraction
        self.max_episodes = max_episodes
        # 16 triples (epid=0, s=i*2, e=s+2) -> total 32 targets
        self.segments = [(0, i * 2, (i + 1) * 2) for i in range(16)]
        self._total_targets = len(self.segments) * 2
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.segments)

    def validation_indices(self) -> List[int]:
        return [0, 1]

    def get_base_data_contract(self) -> Dict[str, Any]:
        base_fp = hashlib.sha256(
            f"{self.split}_root_data_fingerprint_seed{self.seed}".encode("utf-8")
        ).hexdigest()
        return {
            "dataset_class": "NativeTrainingDataset",
            "split": self.split,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "augmentation": self.augmentation,
            "segment_count": len(self.segments),
            "target_count": self._total_targets,
            "data_fingerprint": base_fp,
        }

    def get_data_contract(self) -> Dict[str, Any]:
        base = self.get_base_data_contract()
        compact_proto = get_compact_protocol_contract()
        contract = dict(base)
        contract["dataset_class"] = "PredictiveTrainingDataset"
        contract["stream_protocol"] = compact_proto
        contract["predictive_protocol"] = {
            "label": "predictive",
            "segment_count": len(self.segments),
            "target_count": self._total_targets,
            "root_fingerprint": base["data_fingerprint"],
        }
        raw = f"{base['data_fingerprint']}:{json.dumps(contract['predictive_protocol'], sort_keys=True)}"
        contract["data_fingerprint"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return contract

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + idx * 100 + self.epoch * 1000)
            sample = make_sample(N=8, target_indices=[5, 7], horizon=2, action_dim=4)
            sample["target_count"] = 2
            sample["episode_id"] = 0
            # If first_batch_dense enabled and segment is in the first batch after cursor 4
            # epoch 5 batches with seed 4042: [[0, 11], [6, 7], [5, 12], ...]
            # cursor 4 corresponds to batch index 2: [5, 12]
            if self.first_batch_dense and idx in (5, 12):
                sample["memory_replay"] = False
            else:
                sample["memory_replay"] = True
                sample["decision_indices"] = [1, 3, 5, 7]
            sample["future_images"] = [torch.randn(4, 16), torch.randn(4, 16)]
            sample["future_indices"] = torch.tensor([[0, 1], [0, -1]], dtype=torch.long)
            sample["future_valid"] = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
            sample["future_deltas"] = torch.tensor([[0.1, 0.3], [0.1, 0.0]], dtype=torch.float32)
            return sample


def _create_mock_parent_checkpoint(path: Path) -> None:
    """Create tiny trusted parent checkpoint conforming strictly to runtime contract requirements."""
    train_ds = FakePredictiveTrainingDataset(split="train")
    val_ds = FakePredictiveTrainingDataset(split="val")
    compact_proto = get_compact_protocol_contract()
    compact_proto_sorted_json = json.dumps(compact_proto, sort_keys=True)

    parent_train_dc = dict(train_ds.get_base_data_contract())
    parent_train_dc["augmentation"] = False
    parent_train_dc["stream_protocol"] = compact_proto
    exp_dfp_train = f"{parent_train_dc['data_fingerprint']}:{compact_proto_sorted_json}"
    parent_train_dc["data_fingerprint"] = hashlib.sha256(exp_dfp_train.encode("utf-8")).hexdigest()

    parent_val_dc = dict(val_ds.get_base_data_contract())
    parent_val_dc["augmentation"] = False
    parent_val_dc["stream_protocol"] = compact_proto
    exp_dfp_val = f"{parent_val_dc['data_fingerprint']}:{compact_proto_sorted_json}"
    parent_val_dc["data_fingerprint"] = hashlib.sha256(exp_dfp_val.encode("utf-8")).hexdigest()

    payload = {
        "training_contract": {
            "format": "native_compact_memory_replay_v1",
            "stream_protocol": compact_proto,
            "seed": 4042,
            "global_batch_size": 2,
            "world_size": 1,
        },
        "global_step": 14000,
        "epoch": 5,
        "batch_cursor": 4,
        "epoch_targets_seen": 8,
        "train_data_contract": parent_train_dc,
        "val_data_contract": parent_val_dc,
    }
    torch.save(payload, str(path))


# ============================================================================
# Helpers
# ============================================================================


def _assert_equal_recursive(obj1: Any, obj2: Any, path: str = "") -> None:
    """Strictly assert equality recursively for nested dicts, lists, and tensors."""
    if isinstance(obj1, dict):
        assert isinstance(obj2, dict), f"{path}: type mismatch {type(obj1)} vs {type(obj2)}"
        assert set(obj1.keys()) == set(obj2.keys()), f"{path}: keys mismatch {set(obj1.keys()) ^ set(obj2.keys())}"
        for k in obj1:
            _assert_equal_recursive(obj1[k], obj2[k], f"{path}.{k}")
    elif isinstance(obj1, (list, tuple)):
        assert isinstance(obj2, (list, tuple)), f"{path}: type mismatch {type(obj1)} vs {type(obj2)}"
        assert len(obj1) == len(obj2), f"{path}: length mismatch {len(obj1)} vs {len(obj2)}"
        for idx, (v1, v2) in enumerate(zip(obj1, obj2)):
            _assert_equal_recursive(v1, v2, f"{path}[{idx}]")
    elif isinstance(obj1, torch.Tensor):
        assert isinstance(obj2, torch.Tensor), f"{path}: type mismatch {type(obj1)} vs {type(obj2)}"
        assert torch.equal(obj1, obj2), f"{path}: tensor values mismatch max_diff={(obj1 - obj2).abs().max()}"
    elif isinstance(obj1, np.ndarray):
        assert isinstance(obj2, np.ndarray), f"{path}: type mismatch {type(obj1)} vs {type(obj2)}"
        assert np.array_equal(obj1, obj2), f"{path}: numpy array values mismatch"
    else:
        assert obj1 == obj2, f"{path}: scalar mismatch {obj1!r} != {obj2!r}"


# ============================================================================
# Tests
# ============================================================================


def test_predictive_training_uninterrupted_vs_resumed_exactness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test uninterrupted 4 updates vs max-updates 2 then resume to 4 updates."""
    parent_path = tmp_path / "parent_compact.pt"
    _create_mock_parent_checkpoint(parent_path)

    # Deterministic fixed base policy weights
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(234)
        base_seq = make_tiny_training_policy()
        fixed_base_policy = base_seq.policy

    captured_models: List[Tuple[nn.Module, Dict[str, str]]] = []
    orig_policy_cls = predictive_policy.PredictiveMemoryPolicy

    def mock_policy_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["writer_config"] = WriterConfig(
            input_dim=64,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            grid=1,
            intermediate_grid=1,
            recent_frames=2,
            consolidate_every=2,
            tbptt_decisions=4,
        )
        kwargs["shallow_layer"] = 1
        kwargs["gradient_checkpointing"] = True
        inst = orig_policy_cls(*args, **kwargs)
        base_hashes = {
            n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n, p in inst.policy.named_parameters()
        }
        captured_models.append((inst, base_hashes))
        return inst

    def mock_load_native_checkpoint(*args: Any, **kwargs: Any) -> Any:
        return copy.deepcopy(fixed_base_policy), {}, {"norm": True}, {}

    monkeypatch.setattr(predictive_policy, "PredictiveMemoryPolicy", mock_policy_factory)
    monkeypatch.setattr(runtime, "load_native_checkpoint", mock_load_native_checkpoint)
    FakePredictiveTrainingDataset.first_batch_dense = False
    monkeypatch.setattr(predictive_data, "PredictiveTrainingDataset", FakePredictiveTrainingDataset)

    # 1. Uninterrupted 4 updates
    out_dir_4 = tmp_path / "run_uninterrupted"
    args_4 = parse_args([
        "--init-from", str(parent_path),
        "--output-dir", str(out_dir_4),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "4",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    run_training(args_4, device=torch.device("cpu"))

    # 2. Max-updates 2 (stage 1 boundary)
    out_dir_split = tmp_path / "run_split"
    args_split_2 = parse_args([
        "--init-from", str(parent_path),
        "--output-dir", str(out_dir_split),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "2",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    run_training(args_split_2, device=torch.device("cpu"))

    # Check stage files at update 2
    assert (out_dir_split / "stage1.pt").exists(), "stage1.pt must exist after 2 writer updates"
    assert not (out_dir_split / "stage2.pt").exists(), "stage2.pt must be absent after 2 updates"

    # Inspect optimizer states in checkpoint at update 2: future_head should have no optimizer states
    ckpt_at_2 = torch.load(str(out_dir_split / "last.pt"), map_location="cpu", weights_only=False)
    opt_state_keys_2 = set(ckpt_at_2["optimizer"]["state"].keys())
    # Trainable params: writer has 24 params (indices 0..8, 17..31), future_head has 17 params (indices 9..16, 32..40)
    # At update 2 (phase 1: future_weight=0.0), future_head has received 0 gradients -> 0 optimizer moments
    for fh_idx in list(range(9, 17)) + list(range(32, 41)):
        assert fh_idx not in opt_state_keys_2, f"future_head parameter index {fh_idx} must not have optimizer state at update 2"

    # 3. Resume from update 2 to 4 updates
    args_split_res = parse_args([
        "--resume", str(out_dir_split / "last.pt"),
        "--output-dir", str(out_dir_split),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "4",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    run_training(args_split_res, device=torch.device("cpu"))

    # Check stage files after update 4
    assert (out_dir_split / "stage2.pt").exists(), "stage2.pt must exist after completing 4 updates"

    # Checkpoint comparisons between uninterrupted 4 and resumed 4
    ckpt_uninterrupted = torch.load(str(out_dir_4 / "last.pt"), map_location="cpu", weights_only=False)
    ckpt_resumed = torch.load(str(out_dir_split / "last.pt"), map_location="cpu", weights_only=False)

    # Writer, future_head, optimizer, scheduler exact bitwise match
    _assert_equal_recursive(ckpt_uninterrupted["writer"], ckpt_resumed["writer"], "writer")
    _assert_equal_recursive(ckpt_uninterrupted["future_head"], ckpt_resumed["future_head"], "future_head")
    _assert_equal_recursive(ckpt_uninterrupted["optimizer"], ckpt_resumed["optimizer"], "optimizer")
    _assert_equal_recursive(ckpt_uninterrupted["scheduler"], ckpt_resumed["scheduler"], "scheduler")

    # Cursors and targets equal
    assert ckpt_uninterrupted["epoch"] == ckpt_resumed["epoch"]
    assert ckpt_uninterrupted["batch_cursor"] == ckpt_resumed["batch_cursor"]
    assert ckpt_uninterrupted["epoch_targets_seen"] == ckpt_resumed["epoch_targets_seen"]
    assert ckpt_uninterrupted["update"] == 4 and ckpt_resumed["update"] == 4

    # RNG states bitwise equal
    rng_unint = ckpt_uninterrupted["rng_states_per_rank"][0]
    rng_res = ckpt_resumed["rng_states_per_rank"][0]
    assert rng_unint["python"] == rng_res["python"], "Python RNG states mismatch"
    assert np.array_equal(rng_unint["numpy"][1], rng_res["numpy"][1]), "Numpy RNG states mismatch"
    assert torch.equal(rng_unint["torch_cpu"], rng_res["torch_cpu"]), "Torch CPU RNG states mismatch"

    # future_head optimizer states exist at update 4
    opt_state_keys_4 = set(ckpt_resumed["optimizer"]["state"].keys())
    for fh_idx in list(range(9, 17)) + list(range(32, 41)):
        assert fh_idx in opt_state_keys_4, f"future_head parameter index {fh_idx} must have optimizer state at update 4"

    # Base parameters strictly unchanged across all policy instances
    for inst, initial_hashes in captured_models:
        for name, param in inst.policy.named_parameters():
            curr_h = hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
            assert curr_h == initial_hashes[name], f"Base parameter {name} was unexpectedly modified"


def test_completed_resume_idempotent_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test resuming an already completed budget exits immediately without modifying artifacts."""
    parent_path = tmp_path / "parent_compact.pt"
    _create_mock_parent_checkpoint(parent_path)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(234)
        base_seq = make_tiny_training_policy()
        fixed_base_policy = base_seq.policy

    orig_policy_cls = predictive_policy.PredictiveMemoryPolicy

    def mock_policy_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["writer_config"] = WriterConfig(
            input_dim=64,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            grid=1,
            intermediate_grid=1,
            recent_frames=2,
            consolidate_every=2,
            tbptt_decisions=4,
        )
        kwargs["shallow_layer"] = 1
        kwargs["gradient_checkpointing"] = True
        return orig_policy_cls(*args, **kwargs)

    monkeypatch.setattr(predictive_policy, "PredictiveMemoryPolicy", mock_policy_factory)
    monkeypatch.setattr(runtime, "load_native_checkpoint", lambda *args, **kwargs: (copy.deepcopy(fixed_base_policy), {}, {"norm": True}, {}))
    FakePredictiveTrainingDataset.first_batch_dense = False
    monkeypatch.setattr(predictive_data, "PredictiveTrainingDataset", FakePredictiveTrainingDataset)

    out_dir = tmp_path / "run_completed"
    args_init = parse_args([
        "--init-from", str(parent_path),
        "--output-dir", str(out_dir),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "2",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    run_training(args_init, device=torch.device("cpu"))

    run_config_file = out_dir / "run_config.json"
    metrics_file = out_dir / "metrics.jsonl"
    last_file = out_dir / "last.pt"

    h_rc_before = hashlib.sha256(run_config_file.read_bytes()).hexdigest()
    h_met_before = hashlib.sha256(metrics_file.read_bytes()).hexdigest()
    h_last_before = hashlib.sha256(last_file.read_bytes()).hexdigest()

    # Re-run with resume at max-updates=2
    args_resume = parse_args([
        "--resume", str(last_file),
        "--output-dir", str(out_dir),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "2",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    run_training(args_resume, device=torch.device("cpu"))

    h_rc_after = hashlib.sha256(run_config_file.read_bytes()).hexdigest()
    h_met_after = hashlib.sha256(metrics_file.read_bytes()).hexdigest()
    h_last_after = hashlib.sha256(last_file.read_bytes()).hexdigest()

    assert h_rc_before == h_rc_after, "run_config.json changed on completed resume"
    assert h_met_before == h_met_after, "metrics.jsonl changed on completed resume"
    assert h_last_before == h_last_after, "last.pt changed on completed resume"


def test_immediate_stopfile_preserves_cursor_and_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test stopfile on resumed max2 saves same cursor/update2 without unbound stage_name."""
    parent_path = tmp_path / "parent_compact.pt"
    _create_mock_parent_checkpoint(parent_path)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(234)
        base_seq = make_tiny_training_policy()
        fixed_base_policy = base_seq.policy

    orig_policy_cls = predictive_policy.PredictiveMemoryPolicy

    def mock_policy_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["writer_config"] = WriterConfig(
            input_dim=64,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            grid=1,
            intermediate_grid=1,
            recent_frames=2,
            consolidate_every=2,
            tbptt_decisions=4,
        )
        kwargs["shallow_layer"] = 1
        kwargs["gradient_checkpointing"] = True
        return orig_policy_cls(*args, **kwargs)

    monkeypatch.setattr(predictive_policy, "PredictiveMemoryPolicy", mock_policy_factory)
    monkeypatch.setattr(runtime, "load_native_checkpoint", lambda *args, **kwargs: (copy.deepcopy(fixed_base_policy), {}, {"norm": True}, {}))
    FakePredictiveTrainingDataset.first_batch_dense = False
    monkeypatch.setattr(predictive_data, "PredictiveTrainingDataset", FakePredictiveTrainingDataset)

    out_dir = tmp_path / "run_stopfile"
    args_2 = parse_args([
        "--init-from", str(parent_path),
        "--output-dir", str(out_dir),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "2",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    run_training(args_2, device=torch.device("cpu"))

    ckpt_before = torch.load(str(out_dir / "last.pt"), map_location="cpu", weights_only=False)

    stop_file = tmp_path / "STOP_SENTINEL"
    stop_file.touch()

    args_resumed_stop = parse_args([
        "--resume", str(out_dir / "last.pt"),
        "--output-dir", str(out_dir),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "4",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
        "--stop-file", str(stop_file),
    ])

    # Trainer should cleanly handle immediate stopfile without UnboundLocalError: 'stage_name'
    run_training(args_resumed_stop, device=torch.device("cpu"))

    ckpt_after = torch.load(str(out_dir / "last.pt"), map_location="cpu", weights_only=False)
    assert ckpt_after["update"] == ckpt_before["update"]
    assert ckpt_after["batch_cursor"] == ckpt_before["batch_cursor"]
    assert ckpt_after["epoch_targets_seen"] == ckpt_before["epoch_targets_seen"]
    assert ckpt_after["stage"] == ckpt_before["stage"]


def test_skip_batch_advances_cursor_and_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test skip batch (no grad) advances cursor and epoch_targets_seen without optimizer update."""
    parent_path = tmp_path / "parent_compact.pt"
    _create_mock_parent_checkpoint(parent_path)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(234)
        base_seq = make_tiny_training_policy()
        fixed_base_policy = base_seq.policy

    orig_policy_cls = predictive_policy.PredictiveMemoryPolicy

    def mock_policy_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["writer_config"] = WriterConfig(
            input_dim=64,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            grid=1,
            intermediate_grid=1,
            recent_frames=2,
            consolidate_every=2,
            tbptt_decisions=4,
        )
        kwargs["shallow_layer"] = 1
        kwargs["gradient_checkpointing"] = True
        return orig_policy_cls(*args, **kwargs)

    monkeypatch.setattr(predictive_policy, "PredictiveMemoryPolicy", mock_policy_factory)
    monkeypatch.setattr(runtime, "load_native_checkpoint", lambda *args, **kwargs: (copy.deepcopy(fixed_base_policy), {}, {"norm": True}, {}))
    # Enable first batch dense so the first consumed batch after cursor=4 has no writer graph (no grad)
    FakePredictiveTrainingDataset.first_batch_dense = True
    monkeypatch.setattr(predictive_data, "PredictiveTrainingDataset", FakePredictiveTrainingDataset)

    out_dir = tmp_path / "run_skip"
    args_skip = parse_args([
        "--init-from", str(parent_path),
        "--output-dir", str(out_dir),
        "--writer-updates", "2",
        "--joint-updates", "2",
        "--max-updates", "1",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])

    run_training(args_skip, device=torch.device("cpu"))

    ckpt = torch.load(str(out_dir / "last.pt"), map_location="cpu", weights_only=False)
    # Start cursor was 4, start targets was 8.
    # Batch 1 (segments [5, 12]) had no grad -> skipped: cursor advanced by 2 to 6, targets to 12.
    # Batch 2 had grad -> update 1: cursor advanced by 2 to 8, targets to 16.
    assert ckpt["update"] == 1, f"Expected 1 update, got {ckpt['update']}"
    assert ckpt["batch_cursor"] == 8, f"Expected batch_cursor 8, got {ckpt['batch_cursor']}"
    assert ckpt["epoch_targets_seen"] == 16, f"Expected epoch_targets_seen 16, got {ckpt['epoch_targets_seen']}"

    # Verify metrics.jsonl records skipped_batch
    metrics_path = out_dir / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics_path.read_text().strip().split("\n")]
    skipped_entries = [m for m in lines if m.get("type") == "skipped_batch"]
    assert len(skipped_entries) == 1, f"Expected 1 skipped_batch entry in metrics, got {len(skipped_entries)}"
    skip_metric = skipped_entries[0]
    assert skip_metric["epoch"] == 5
    assert skip_metric["batch_cursor"] == 6
    assert skip_metric["epoch_targets_seen"] == 12
    assert skip_metric["targets"] == 4
