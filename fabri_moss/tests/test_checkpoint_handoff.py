"""Unit tests for fabri_moss/scripts/stop_at_native_checkpoint.py."""

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
import torch

from fabri_moss.scripts.stop_at_native_checkpoint import (
    CheckpointValidationError,
    ProcessIdentityError,
    compute_file_sha256,
    exclusive_archive_checkpoint,
    main,
    validate_checkpoint_structure,
    write_exclusive_initial_report,
)


class FakeProcEnv:
    """Reusable fixture to simulate /proc entries and signal recording without OS signals."""

    def __init__(self, root: Path, run_dir: Path):
        self.root = root / "proc"
        self.run_dir = run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.signals: List[tuple] = []
        self.active_pids: Dict[int, Dict[str, Any]] = {}

    def add_process(self, pid: int, ticks: int, ppid: int, state: str, role: str):
        p_dir = self.root / str(pid)
        p_dir.mkdir(parents=True, exist_ok=True)
        self.active_pids[pid] = {"ticks": ticks, "ppid": ppid, "state": state, "role": role}
        self.sync_proc(pid)

    def sync_proc(self, pid: int):
        info = self.active_pids[pid]
        p_dir = self.root / str(pid)
        tokens = [str(pid), f"({info['role']})", info["state"], str(info["ppid"])] + ["0"] * 17 + [str(info["ticks"])] + ["0"] * 10
        (p_dir / "stat").write_text(" ".join(tokens) + "\n", encoding="utf-8")
        if info["role"] == "launcher":
            args = ["python", "-m", "torch.distributed.run", "-m", "fabri_moss.train_native", "--output-dir", str(self.run_dir)]
        else:
            args = ["python", "-m", "fabri_moss.train_native", "--output-dir", str(self.run_dir)]
        (p_dir / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")

    def mock_kill(self, pid: int, sig: int):
        self.signals.append((pid, sig))
        if pid in self.active_pids:
            if sig == signal.SIGTERM:
                self.active_pids[pid]["terminated"] = True
                self.active_pids[pid]["state"] = "Z"
            elif sig == signal.SIGSTOP:
                self.active_pids[pid]["state"] = "T"
            elif sig == signal.SIGCONT:
                # If already terminated by SIGTERM, state remains Z (terminated)
                if not self.active_pids[pid].get("terminated"):
                    self.active_pids[pid]["state"] = "S"
            self.sync_proc(pid)


def make_valid_checkpoint(step: int = 10, epoch: int = 0, cursor: int = 10, targets: int = 80) -> Dict[str, Any]:
    w = torch.randn(4, 4, dtype=torch.float32)
    b = torch.zeros(4, dtype=torch.float32)
    return {
        "model": {"weight": w, "bias": b},
        "optimizer": {
            "state": {0: {"momentum": torch.zeros(4, 4, dtype=torch.float32)}},
            "param_groups": [{"lr": 0.001, "params": [0]}],
        },
        "scheduler": {"last_epoch": step},
        "global_step": step,
        "epoch": epoch,
        "batch_cursor": cursor,
        "epoch_targets_seen": targets,
        "training_contract": {"global_batch_size": 1},
        "train_data_contract": {"segment_count": 100},
        "val_data_contract": {"val": 1},
        "config": {"batch_size": 1},
        "norm_stats": {"action": {"mean": [0.0], "std": [1.0]}},
        "source_metadata": {"checkpoint_sha256": "fake_sha_source"},
        "rng_states_per_rank": [
            {"python": (3, (0,), None), "numpy": ["MT19937", [0], 0, 0, 0.0], "torch_cpu": torch.ByteTensor([0]), "torch_cuda": torch.ByteTensor([0])},
            {"python": (3, (0,), None), "numpy": ["MT19937", [0], 0, 0, 0.0], "torch_cpu": torch.ByteTensor([0]), "torch_cuda": torch.ByteTensor([0])},
        ],
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proc = FakeProcEnv(tmp_path, run_dir)
    monkeypatch.setattr(os, "kill", proc.mock_kill)
    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.parse_proc_stat", lambda p, proc_root=None: proc.sync_proc(p) or (proc.active_pids[p]["state"], proc.active_pids[p]["ppid"], proc.active_pids[p]["ticks"]))
    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.parse_proc_cmdline", lambda p, proc_root=None: ["python", "-m", "torch.distributed.run", "-m", "fabri_moss.train_native", "--output-dir", str(run_dir)] if proc.active_pids[p]["role"] == "launcher" else ["python", "-m", "fabri_moss.train_native", "--output-dir", str(run_dir)])
    return proc


def get_base_args(tmp_path: Path, run_dir: Path) -> List[str]:
    return [
        "--run-dir", str(run_dir),
        "--launcher-pid", "100",
        "--launcher-start-ticks", "1000",
        "--worker-pids", "201", "202",
        "--archive-path", str(tmp_path / "arch.pt"),
        "--report-path", str(tmp_path / "rep.json"),
        "--timeout-seconds", "1.0",
        "--poll-seconds", "0.01",
    ]


def test_noexecute_zero_effects(env: FakeProcEnv, tmp_path: Path):
    env.add_process(100, 1000, 1, "S", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")
    args = get_base_args(tmp_path, env.run_dir)
    ret = main(args)
    assert ret == 0
    assert len(env.signals) == 0
    assert not (tmp_path / "arch.pt").exists()
    assert not (tmp_path / "rep.json").exists()


def test_initial_stopped_reject(env: FakeProcEnv, tmp_path: Path):
    env.add_process(100, 1000, 1, "T", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")
    ret = main(get_base_args(tmp_path, env.run_dir) + ["--execute"])
    assert ret == 1
    assert len(env.signals) == 0


def test_wrong_module_substring_reject(env: FakeProcEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env.add_process(100, 1000, 1, "S", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")
    # Substring matches "fabri_moss.train_native" but not exact token pair
    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.parse_proc_cmdline", lambda p, proc_root=None: ["python", "fabri_moss.train_native_not_module.py", "--output-dir", str(env.run_dir)])
    ret = main(get_base_args(tmp_path, env.run_dir) + ["--execute"])
    assert ret == 1
    assert len(env.signals) == 0


def test_archive_destination_exists_reject(env: FakeProcEnv, tmp_path: Path):
    env.add_process(100, 1000, 1, "S", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")
    arch = tmp_path / "arch.pt"
    arch.write_text("busy")
    ret = main(get_base_args(tmp_path, env.run_dir) + ["--execute"])
    assert ret == 1
    assert len(env.signals) == 0


def test_schema_validations(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.pt"
    metric = {"global_step": 10, "epoch": 0, "epoch_targets_seen": 80, "loss": 0.1}

    # 1. Non-empty model / optimizer
    bad_ckpt = make_valid_checkpoint(10)
    bad_ckpt["model"] = {}
    torch.save(bad_ckpt, str(ckpt_path))
    with pytest.raises(CheckpointValidationError, match="model state dict must be a non-empty"):
        validate_checkpoint_structure(ckpt_path, metric)

    # 2. Incomplete RNG states (must have 2 ranks, each with all 4 keys)
    bad_ckpt = make_valid_checkpoint(10)
    bad_ckpt["rng_states_per_rank"][0].pop("torch_cuda")
    torch.save(bad_ckpt, str(ckpt_path))
    with pytest.raises(CheckpointValidationError, match="missing torch_cuda"):
        validate_checkpoint_structure(ckpt_path, metric)

    # 3. Bad geometry
    bad_ckpt = make_valid_checkpoint(10)
    bad_ckpt["batch_cursor"] = 5  # calculated step = (0*100 + 5) // 1 = 5 != 10
    torch.save(bad_ckpt, str(ckpt_path))
    with pytest.raises(CheckpointValidationError, match="Resume geometry mismatch"):
        validate_checkpoint_structure(ckpt_path, metric)

    # 4. FP32 floating tensors enforced
    bad_ckpt = make_valid_checkpoint(10)
    bad_ckpt["model"]["weight"] = bad_ckpt["model"]["weight"].to(torch.float16)
    torch.save(bad_ckpt, str(ckpt_path))
    with pytest.raises(CheckpointValidationError, match="Model param weight dtype torch.float16 != torch.float32"):
        validate_checkpoint_structure(ckpt_path, metric)


@pytest.mark.parametrize("failure_point", ["archive", "load", "hash", "report", "interrupt", "mid_stop"])
def test_exception_during_archive_ensures_sigcont(env: FakeProcEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point):
    env.add_process(100, 1000, 1, "S", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")

    last_pt = env.run_dir / "last.pt"
    torch.save(make_valid_checkpoint(10), str(last_pt))
    (env.run_dir / "train_metrics.jsonl").write_text(json.dumps({"global_step": 10, "epoch": 0, "epoch_targets_seen": 80, "loss": 0.1}) + "\n")

    # Initial stat before loop is (1, 100). Then mock stat returns (2, 200) to trigger update.
    stat_seq = [(1, 100), (2, 200)]
    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.get_last_checkpoint_stat", lambda p: stat_seq.pop(0) if stat_seq else (2, 200))

    def blow_up(*args, **kwargs):
        if failure_point == "interrupt":
            raise KeyboardInterrupt()
        raise RuntimeError("Injected failure")

    module = "fabri_moss.scripts.stop_at_native_checkpoint."
    targets = {
        "archive": "exclusive_archive_checkpoint", "load": "torch.load",
        "hash": "compute_file_sha256", "report": "write_exclusive_initial_report",
        "interrupt": "exclusive_archive_checkpoint",
    }
    if failure_point == "mid_stop":
        def fail_second_worker(pid, sig):
            if pid == 202 and sig == signal.SIGSTOP:
                raise OSError("Injected signal failure")
            env.mock_kill(pid, sig)
        monkeypatch.setattr(os, "kill", fail_second_worker)
    else:
        monkeypatch.setattr(module + targets[failure_point], blow_up)

    ret = main(get_base_args(tmp_path, env.run_dir) + ["--execute"])
    assert ret == 1
    # Check that SIGCONT was sent to all stopped processes and NO SIGTERM was sent
    cont_pids = [pid for pid, sig in env.signals if sig == signal.SIGCONT]
    expected = {100, 201} if failure_point == "mid_stop" else {100, 201, 202}
    assert expected <= set(cont_pids)
    assert not any(sig == signal.SIGTERM for _, sig in env.signals)


def test_mismatch_resumes_and_later_succeeds(env: FakeProcEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env.add_process(100, 1000, 1, "S", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")

    last_pt = env.run_dir / "last.pt"
    metrics_path = env.run_dir / "train_metrics.jsonl"

    stats_call = [0]

    def mock_stat(p: Path):
        stats_call[0] += 1
        if stats_call[0] == 1:
            return (1, 100)
        elif stats_call[0] == 2:
            # First update: step mismatch (ckpt 20, metric 19)
            torch.save(make_valid_checkpoint(20, epoch=0, cursor=20), str(last_pt))
            metrics_path.write_text(json.dumps({"global_step": 19, "epoch": 0, "epoch_targets_seen": 80, "loss": 0.1}) + "\n")
            return (2, 200)
        elif stats_call[0] == 3:
            # Polling before next save returns same stat (2, 200)
            return (2, 200)
        else:
            # Second update: step match (ckpt 20, metric 20)
            metrics_path.write_text(json.dumps({"global_step": 20, "epoch": 0, "epoch_targets_seen": 80, "loss": 0.1}) + "\n")
            return (3, 300)

    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.get_last_checkpoint_stat", mock_stat)

    ret = main(get_base_args(tmp_path, env.run_dir) + ["--execute"])
    assert ret == 0
    # Verified resume happened after step 1, then termination succeeded on step 2
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["status"] == "stopped"


def test_geometry_with_nondivisible_epoch_and_validation_records(tmp_path):
    from fabri_moss.scripts.stop_at_native_checkpoint import read_latest_train_metric
    ckpt = make_valid_checkpoint(step=6, epoch=1, cursor=9, targets=80)
    ckpt["training_contract"]["global_batch_size"] = 4
    ckpt["train_data_contract"]["segment_count"] = 9
    path = tmp_path / "last.pt"
    torch.save(ckpt, path)
    metric = {"global_step": 6, "epoch": 1, "loss": 0.1, "epoch_targets_seen": 80}
    assert validate_checkpoint_structure(path, metric)["global_step"] == 6
    log = tmp_path / "train_metrics.jsonl"
    log.write_text(json.dumps(metric) + "\n" + json.dumps({"global_step": 6, "epoch": 2, "val_loss": 0.2}) + "\n")
    assert read_latest_train_metric(log) == metric


def test_lingering_process_reports_stop_failed(env: FakeProcEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env.add_process(100, 1000, 1, "S", "launcher")
    env.add_process(201, 2001, 100, "S", "worker")
    env.add_process(202, 2002, 100, "S", "worker")

    last_pt = env.run_dir / "last.pt"
    torch.save(make_valid_checkpoint(10), str(last_pt))
    (env.run_dir / "train_metrics.jsonl").write_text(json.dumps({"global_step": 10, "epoch": 0, "epoch_targets_seen": 80, "loss": 0.1}) + "\n")

    stat_seq = [(1, 100), (2, 200)]
    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.get_last_checkpoint_stat", lambda p: stat_seq.pop(0) if stat_seq else (2, 200))
    # Processes linger: override mock_kill so SIGTERM does NOT make them Z
    def lingering_kill(pid: int, sig: int):
        env.signals.append((pid, sig))
        if sig == signal.SIGSTOP:
            env.active_pids[pid]["state"] = "T"
        elif sig == signal.SIGCONT:
            env.active_pids[pid]["state"] = "S"
        # On SIGTERM, remains alive (S)
        env.sync_proc(pid)
    monkeypatch.setattr(os, "kill", lingering_kill)
    monkeypatch.setattr("fabri_moss.scripts.stop_at_native_checkpoint.poll_brief_deadline", lambda fn, timeout, step=0.01: False if timeout > 5 else True)

    ret = main(get_base_args(tmp_path, env.run_dir) + ["--execute"])
    assert ret == 1
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["status"] == "stop_failed"
    # Ensure running (SIGCONT)
    cont_pids = [pid for pid, sig in env.signals if sig == signal.SIGCONT]
    assert 100 in cont_pids and 201 in cont_pids and 202 in cont_pids
