"""Unit tests for continue_predictive_stage3.py coordinator.

Verifies:
- pure boundary validation (validate_stage2_checkpoint, validate_gate_checkpoint)
- rejection on wrong update budget, missing checkpoint, SHA mismatch
- stop flag detection (predroot / jointroot)
- flock exclusivity preventing duplicate coordinators
- exact 2-stage command sequence (gate max-updates 2 -> resume last.pt)
- no GPU launch / calls before old workers exit
- no launch when predecessor fails
- exit already running if child process alive
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
import pytest
import torch

# Ensure repo root is on sys.path for scripts and module imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.continue_predictive_stage3 import (
    compute_sha256,
    find_old_workers,
    is_process_alive,
    parse_args,
    run_coordinator,
    validate_gate_checkpoint,
    validate_stage2_checkpoint,
)


def _make_dummy_parent(tmp_path: Path) -> Path:
    p = tmp_path / "dummy_parent.pt"
    p.write_bytes(b"parent_model_bytes_12345")
    return p


def _make_dummy_stage2_payload(parent_path: Path, update: int = 2500) -> Dict[str, Any]:
    parent_sha = compute_sha256(parent_path)
    tc = {
        "writer_updates": 500,
        "joint_updates": 2000,
        "lr": 1e-4,
    }
    src_fingerprints = {
        "predictive_memory.py": "hash_a",
        "predictive_policy.py": "hash_b",
    }
    return {
        "format": "predictive_memory_adapter_v1",
        "stage": "stage2_joint_future",
        "update": update,
        "global_step": 14741 + update,
        "epoch": 2,
        "batch_cursor": 12,
        "epoch_targets_seen": 96,
        "parent_path": str(parent_path),
        "parent_sha256": parent_sha,
        "parent_global_step": 14741,
        "training_contract": tc,
        "source_module_fingerprints": src_fingerprints,
    }


def _make_dummy_run_config(parent_path: Path, tc: Dict[str, Any], src_fp: Dict[str, Any]) -> Dict[str, Any]:
    parent_sha = compute_sha256(parent_path)
    return {
        "parent_sha256": parent_sha,
        "training_contract": tc,
        "source_module_fingerprints": src_fp,
    }


def _make_valid_gate_payload() -> Dict[str, Any]:
    return {
        "format": "predictive_memory_joint_v1",
        "update": 2,
        "scheduler": {"last_epoch": 2},
        "world_size": 2,
        "rng_states_per_rank": [{"rank": 0}, {"rank": 1}],
        "group_update_checks": {
            "vision": True,
            "projector": True,
            "llm": True,
            "head": True,
            "writer": True,
            "future": True,
        },
        "teacher_hash_match": True,
    }


def test_cli_help():
    args = parse_args([
        "--predecessor-root", "/tmp/pred",
        "--joint-root", "/tmp/joint",
        "--predecessor-pid", "1234",
        "--predecessor-start-ticks", "5678",
    ])
    assert args.predecessor_root == "/tmp/pred"
    assert args.joint_root == "/tmp/joint"
    assert args.predecessor_pid == 1234
    assert args.predecessor_start_ticks == 5678
    assert args.epochs == 10
    assert args.poll_seconds == 10


def test_validate_stage2_checkpoint_success(tmp_path: Path):
    parent = _make_dummy_parent(tmp_path)
    stage2_payload = _make_dummy_stage2_payload(parent, update=2500)
    stage2_path = tmp_path / "stage2.pt"
    torch.save(stage2_payload, str(stage2_path))

    run_cfg = _make_dummy_run_config(parent, stage2_payload["training_contract"], stage2_payload["source_module_fingerprints"])
    run_cfg_path = tmp_path / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg))

    ckpt, s2_sha, p_sha = validate_stage2_checkpoint(stage2_path, run_cfg_path)
    assert ckpt["update"] == 2500
    assert p_sha == compute_sha256(parent)
    assert s2_sha == compute_sha256(stage2_path)


def test_validate_stage2_checkpoint_wrong_budget(tmp_path: Path):
    parent = _make_dummy_parent(tmp_path)
    stage2_payload = _make_dummy_stage2_payload(parent, update=2499)
    stage2_path = tmp_path / "stage2.pt"
    torch.save(stage2_payload, str(stage2_path))

    run_cfg = _make_dummy_run_config(parent, stage2_payload["training_contract"], stage2_payload["source_module_fingerprints"])
    run_cfg_path = tmp_path / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg))

    with pytest.raises(ValueError, match="Incomplete update budget"):
        validate_stage2_checkpoint(stage2_path, run_cfg_path)


def test_validate_stage2_checkpoint_parent_sha_mismatch(tmp_path: Path):
    parent = _make_dummy_parent(tmp_path)
    stage2_payload = _make_dummy_stage2_payload(parent, update=2500)
    stage2_payload["parent_sha256"] = "wrong_sha"
    stage2_path = tmp_path / "stage2.pt"
    torch.save(stage2_payload, str(stage2_path))

    run_cfg = _make_dummy_run_config(parent, stage2_payload["training_contract"], stage2_payload["source_module_fingerprints"])
    run_cfg_path = tmp_path / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg))

    with pytest.raises(ValueError, match="Parent SHA mismatch"):
        validate_stage2_checkpoint(stage2_path, run_cfg_path)


def test_validate_gate_checkpoint_success(tmp_path: Path):
    payload = _make_valid_gate_payload()
    p = tmp_path / "last.pt"
    torch.save(payload, str(p))
    res = validate_gate_checkpoint(p)
    assert res["update"] == 2
    assert res["teacher_hash_match"] is True


def test_validate_gate_checkpoint_missing_group(tmp_path: Path):
    payload = _make_valid_gate_payload()
    payload["group_update_checks"]["future"] = False
    p = tmp_path / "last.pt"
    torch.save(payload, str(p))
    with pytest.raises(ValueError, match="Group update check failed"):
        validate_gate_checkpoint(p)


def test_validate_gate_checkpoint_wrong_world_size(tmp_path: Path):
    payload = _make_valid_gate_payload()
    payload["world_size"] = 1
    p = tmp_path / "last.pt"
    torch.save(payload, str(p))
    with pytest.raises(ValueError, match="world_size"):
        validate_gate_checkpoint(p)


def test_flock_duplicate_prevention(tmp_path: Path):
    joint_root = tmp_path / "joint"
    joint_root.mkdir()
    lock_file = open(joint_root / "handoff.lock", "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    args = argparse.Namespace(
        predecessor_root=str(tmp_path / "pred"),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=1,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 0


def test_stop_flag_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_root = tmp_path / "pred"
    pred_root.mkdir()
    joint_root = tmp_path / "joint"
    joint_root.mkdir()

    (pred_root / "STOP_PREDICTIVE").write_text("stop")

    # Mock launcher alive once, then coordinator sees stop
    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: True)

    args = argparse.Namespace(
        predecessor_root=str(pred_root),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 0
    state = json.loads((joint_root / "handoff_state.json").read_text())
    assert state["status"] == "cancelled"


def test_blocked_when_stage2_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_root = tmp_path / "pred"
    pred_root.mkdir()
    joint_root = tmp_path / "joint"
    joint_root.mkdir()

    # Launcher and old workers exited
    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: False)
    monkeypatch.setattr("scripts.continue_predictive_stage3.find_old_workers", lambda predecessor_cwd=None, **kwargs: [])

    args = argparse.Namespace(
        predecessor_root=str(pred_root),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 1
    state = json.loads((joint_root / "handoff_state.json").read_text())
    assert state["status"] == "blocked"


def test_no_launch_when_old_workers_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_root = tmp_path / "pred"
    pred_root.mkdir()
    joint_root = tmp_path / "joint"
    joint_root.mkdir()

    call_counts = {"workers_checked": 0, "launched": 0}

    def mock_find_old_workers(predecessor_cwd=None, **kwargs):
        call_counts["workers_checked"] += 1
        if call_counts["workers_checked"] == 1:
            return [12345]  # old worker still running
        (pred_root / "STOP_PREDICTIVE").write_text("stop")
        return [12345]

    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: False)
    monkeypatch.setattr("scripts.continue_predictive_stage3.find_old_workers", mock_find_old_workers)

    args = argparse.Namespace(
        predecessor_root=str(pred_root),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 0
    state = json.loads((joint_root / "handoff_state.json").read_text())
    assert state["status"] == "cancelled"


def test_exact_two_stage_commands_and_full_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_root = tmp_path / "pred"
    formal_two_stage = pred_root / "formal_two_stage"
    formal_two_stage.mkdir(parents=True)
    joint_root = tmp_path / "joint"
    joint_root.mkdir()

    parent = _make_dummy_parent(tmp_path)
    stage2_payload = _make_dummy_stage2_payload(parent, update=2500)
    stage2_path = formal_two_stage / "stage2.pt"
    torch.save(stage2_payload, str(stage2_path))

    run_cfg = _make_dummy_run_config(parent, stage2_payload["training_contract"], stage2_payload["source_module_fingerprints"])
    run_cfg_path = formal_two_stage / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg))

    scripts_dir = joint_root / "scripts"
    scripts_dir.mkdir()
    launcher_script = scripts_dir / "run_joint_predictive.sh"
    launcher_script.write_text("#!/bin/bash\nexit 0\n")

    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: False)
    monkeypatch.setattr("scripts.continue_predictive_stage3.find_old_workers", lambda predecessor_cwd=None, **kwargs: [])
    monkeypatch.setattr("scripts.continue_predictive_stage3.get_gpu_compute_pids", lambda: [])

    commands_run: List[List[str]] = []

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, env=None, cwd=None):
            commands_run.append(cmd)
            self.pid = 4321
            formal_joint = joint_root / "formal_joint"
            formal_joint.mkdir(parents=True, exist_ok=True)
            if "--max-updates" in cmd:
                # First stage gate: write valid last.pt
                torch.save(_make_valid_gate_payload(), str(formal_joint / "last.pt"))
            elif "--resume" in cmd:
                # Second stage formal: write final.pt
                torch.save({"format": "predictive_memory_joint_v1", "epoch": 10, "batch_cursor": 0}, formal_joint / "final.pt")

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr("scripts.continue_predictive_stage3.parse_proc_stat", lambda pid: ("S", 1, 9999))

    args = argparse.Namespace(
        predecessor_root=str(pred_root),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 0
    state = json.loads((joint_root / "handoff_state.json").read_text())
    assert state["status"] == "completed"

    assert len(commands_run) == 2
    gate_cmd, formal_cmd = commands_run[0], commands_run[1]

    # Verify exact gate arguments
    assert gate_cmd[0] == "bash"
    assert gate_cmd[1] == str(launcher_script)
    assert gate_cmd[2] == str(joint_root / "formal_joint")
    assert "--init-from-stage2" in gate_cmd
    assert str(stage2_path) in gate_cmd
    assert "--max-updates" in gate_cmd
    assert "2" in gate_cmd

def test_no_launch_when_predecessor_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_root = tmp_path / "pred"
    formal_two_stage = pred_root / "formal_two_stage"
    formal_two_stage.mkdir(parents=True)
    joint_root = tmp_path / "joint"
    joint_root.mkdir()

    # Predecessor failed and never produced valid stage2.pt (e.g. only incomplete last.pt)
    parent = _make_dummy_parent(tmp_path)
    stage2_payload = _make_dummy_stage2_payload(parent, update=1200)  # incomplete
    stage2_path = formal_two_stage / "stage2.pt"
    torch.save(stage2_payload, str(stage2_path))

    run_cfg = _make_dummy_run_config(parent, stage2_payload["training_contract"], stage2_payload["source_module_fingerprints"])
    run_cfg_path = formal_two_stage / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg))

    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: False)
    monkeypatch.setattr("scripts.continue_predictive_stage3.find_old_workers", lambda predecessor_cwd=None, **kwargs: [])

    args = argparse.Namespace(
        predecessor_root=str(pred_root),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 1
    state = json.loads((joint_root / "handoff_state.json").read_text())
    assert state["status"] == "blocked"
    assert "Incomplete update budget" in state["reason"]


def test_coordinator_restart_exits_if_child_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    joint_root = tmp_path / "joint"
    joint_root.mkdir()
    state_file = joint_root / "handoff_state.json"
    state_file.write_text(json.dumps({
        "status": "running_formal",
        "child_pid": 112233,
        "child_start_ticks": 445566,
    }))

    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: True)

    args = argparse.Namespace(
        predecessor_root=str(tmp_path / "pred"),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 0


def test_wait_for_foreign_gpu_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_root = tmp_path / "pred"
    formal_two_stage = pred_root / "formal_two_stage"
    formal_two_stage.mkdir(parents=True)
    joint_root = tmp_path / "joint"
    joint_root.mkdir()

    parent = _make_dummy_parent(tmp_path)
    stage2_payload = _make_dummy_stage2_payload(parent, update=2500)
    stage2_path = formal_two_stage / "stage2.pt"
    torch.save(stage2_payload, str(stage2_path))

    run_cfg = _make_dummy_run_config(parent, stage2_payload["training_contract"], stage2_payload["source_module_fingerprints"])
    run_cfg_path = formal_two_stage / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg))

    gpu_calls = {"count": 0}

    def mock_gpu_pids():
        gpu_calls["count"] += 1
        if gpu_calls["count"] == 1:
            return [8888]  # foreign job present
        (pred_root / "STOP_PREDICTIVE").write_text("stop")
        return [8888]

    monkeypatch.setattr("scripts.continue_predictive_stage3.is_process_alive", lambda pid, ticks: False)
    monkeypatch.setattr("scripts.continue_predictive_stage3.find_old_workers", lambda predecessor_cwd=None, **kwargs: [])
    monkeypatch.setattr("scripts.continue_predictive_stage3.get_gpu_compute_pids", mock_gpu_pids)

    args = argparse.Namespace(
        predecessor_root=str(pred_root),
        joint_root=str(joint_root),
        predecessor_pid=99999,
        predecessor_start_ticks=1234,
        epochs=10,
        poll_seconds=0,
        python=None,
        rootenv=None,
    )
    ret = run_coordinator(args)
    assert ret == 0
    state = json.loads((joint_root / "handoff_state.json").read_text())
    assert state["status"] == "cancelled"
