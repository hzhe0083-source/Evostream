#!/usr/bin/env python3
"""continue_predictive_stage3.py - One-shot background coordinator for Predictive Stage 3 fine-tuning."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_proc_stat(pid: int, proc_root: str = "/proc") -> Tuple[str, int, int]:
    stat_path = Path(proc_root) / str(pid) / "stat"
    if not stat_path.exists():
        raise ProcessLookupError(f"PID {pid} stat not found: {stat_path}")
    content = stat_path.read_text(encoding="utf-8", errors="replace").strip()
    rparen = content.rfind(")")
    if rparen == -1 or rparen + 2 >= len(content):
        raise ValueError(f"Malformed stat line for PID {pid}")
    fields = content[rparen + 2:].split()
    if len(fields) < 20:
        raise ValueError(f"Insufficient fields in stat for PID {pid}")
    return fields[0], int(fields[1]), int(fields[19])


def parse_proc_cmdline(pid: int, proc_root: str = "/proc") -> List[str]:
    cmd_path = Path(proc_root) / str(pid) / "cmdline"
    if not cmd_path.exists():
        return []
    return [p.decode("utf-8", errors="replace") for p in cmd_path.read_bytes().split(b"\x00") if p]


def parse_proc_cwd(pid: int, proc_root: str = "/proc") -> Optional[Path]:
    try:
        return (Path(proc_root) / str(pid) / "cwd").resolve()
    except Exception:
        return None


def is_process_alive(pid: int, expected_ticks: Optional[int], proc_root: str = "/proc") -> bool:
    try:
        state, _, ticks = parse_proc_stat(pid, proc_root=proc_root)
        if state == "Z":
            return False
        return expected_ticks is None or ticks == expected_ticks
    except (ProcessLookupError, FileNotFoundError, ValueError):
        return False


def find_old_workers(predecessor_cwd: Path, expected_script_sub: str = "train_predictive", proc_root: str = "/proc") -> List[int]:
    pids: List[int] = []
    root = Path(proc_root)
    if not root.exists():
        return pids
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cwd = parse_proc_cwd(pid, proc_root=proc_root)
            if cwd is None or cwd != predecessor_cwd.resolve():
                continue
            if expected_script_sub in " ".join(parse_proc_cmdline(pid, proc_root=proc_root)):
                state, _, _ = parse_proc_stat(pid, proc_root=proc_root)
                if state != "Z":
                    pids.append(pid)
        except Exception:
            continue
    return pids


def get_gpu_compute_pids() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        return [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    except Exception as exc:
        raise RuntimeError("Cannot verify that GPUs are free") from exc


def validate_stage2_checkpoint(stage2_path: Path, run_config_path: Path, expected_writer_updates: int = 500, expected_joint_updates: int = 2000) -> Tuple[Dict[str, Any], str, str]:
    if not stage2_path.exists():
        raise ValueError(f"stage2.pt not found at {stage2_path}")
    if not run_config_path.exists():
        raise ValueError(f"run_config.json not found at {run_config_path}")

    actual_stage2_sha = compute_sha256(stage2_path)
    run_cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
    ckpt = torch.load(str(stage2_path), map_location="cpu", weights_only=False)

    for k in ("format", "stage", "update", "global_step", "epoch", "batch_cursor", "epoch_targets_seen", "parent_path", "parent_sha256", "parent_global_step", "training_contract", "source_module_fingerprints"):
        if k not in ckpt:
            raise ValueError(f"Missing required checkpoint key: {k}")

    if ckpt.get("format") != "predictive_memory_adapter_v1" or ckpt.get("stage") != "stage2_joint_future":
        raise ValueError(f"Invalid format or stage: {ckpt.get('format')}, {ckpt.get('stage')}")

    total_expected = expected_writer_updates + expected_joint_updates
    if ckpt.get("update") != total_expected:
        raise ValueError(f"Incomplete update budget: {ckpt.get('update')} != {total_expected}")

    tc = ckpt["training_contract"]
    if tc.get("writer_updates") != expected_writer_updates or tc.get("joint_updates") != expected_joint_updates:
        raise ValueError("Stage 2 phase budgets mismatch")
    if ckpt["global_step"] != ckpt["parent_global_step"] + total_expected:
        raise ValueError("Stage 2 lineage step mismatch")
    parent_path = Path(ckpt["parent_path"])
    if not parent_path.exists():
        raise ValueError(f"Parent model path does not exist: {parent_path}")
    actual_parent_sha = compute_sha256(parent_path)
    if ckpt["parent_sha256"] != actual_parent_sha or run_cfg.get("parent_sha256") != actual_parent_sha:
        raise ValueError("Parent SHA mismatch with actual parent file")

    if ckpt["training_contract"] != run_cfg.get("training_contract") or ckpt["source_module_fingerprints"] != run_cfg.get("source_module_fingerprints"):
        raise ValueError("training_contract or source_module_fingerprints mismatch")

    return ckpt, actual_stage2_sha, actual_parent_sha


def validate_gate_checkpoint(gate_ckpt_path: Path) -> Dict[str, Any]:
    if not gate_ckpt_path.exists():
        raise ValueError(f"Gate checkpoint not found at {gate_ckpt_path}")
    ckpt = torch.load(str(gate_ckpt_path), map_location="cpu", weights_only=False, mmap=True)

    for k in ("format", "update", "scheduler", "world_size", "rng_states_per_rank", "group_update_checks", "teacher_hash_match"):
        if k not in ckpt:
            raise ValueError(f"Gate checkpoint missing required key: {k}")

    if ckpt["format"] != "predictive_memory_joint_v1" or ckpt["update"] != 2:
        raise ValueError(f"Gate format/update invalid: {ckpt.get('format')}, update={ckpt.get('update')}")

    sched = ckpt.get("scheduler")
    sched_updates = sched.get("last_epoch", sched.get("update")) if isinstance(sched, dict) else getattr(sched, "last_epoch", None)
    if sched_updates != 2:
        raise ValueError(f"Gate scheduler step {sched_updates} != 2")

    if ckpt.get("world_size") != 2 or len(ckpt.get("rng_states_per_rank", [])) != 2:
        raise ValueError("Gate world_size or rng_states_per_rank count != 2")

    checks = ckpt.get("group_update_checks", {})
    for group in ("vision", "projector", "llm", "head", "writer", "future"):
        if not checks.get(group):
            raise ValueError(f"Group update check failed or missing for: {group}")

    if not ckpt.get("teacher_hash_match"):
        raise ValueError("Teacher hash match check failed")

    return ckpt


def parse_args(raw_args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictive Stage 3 Coordinator")
    parser.add_argument("--predecessor-root", type=str, required=True, help="Predecessor run root dir")
    parser.add_argument("--joint-root", type=str, required=True, help="Joint stage 3 run root dir")
    parser.add_argument("--predecessor-pid", type=int, required=True, help="Launcher PID of predecessor")
    parser.add_argument("--predecessor-start-ticks", type=int, required=True, help="Start ticks of predecessor launcher")
    parser.add_argument("--epochs", type=int, default=10, help="Target epochs")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval in seconds")
    parser.add_argument("--python", type=str, default=None, help="Python binary override")
    args = parser.parse_args(raw_args)
    if min(args.epochs, args.poll_seconds, args.predecessor_pid, args.predecessor_start_ticks) <= 0:
        parser.error("epochs, poll interval and predecessor identity must be positive")
    return args


def write_state(joint_root: Path, state_data: Dict[str, Any]) -> None:
    state_file = joint_root / "handoff_state.json"
    tmp_file = joint_root / "handoff_state.json.tmp"
    state_data["updated_at"] = time.time()
    tmp_file.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
    tmp_file.replace(state_file)


def run_coordinator(args: argparse.Namespace) -> int:
    pred_root = Path(args.predecessor_root).resolve()
    joint_root = Path(args.joint_root).resolve()
    joint_root.mkdir(parents=True, exist_ok=True)
    logs_dir = joint_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    lock_file = open(joint_root / "handoff.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        print(f"[Coordinator] Another coordinator holds lock on {joint_root}. Exiting.")
        return 0

    state_file = joint_root / "handoff_state.json"
    existing_state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    child_pid = existing_state.get("child_pid")
    child_ticks = existing_state.get("child_start_ticks")
    if child_pid and is_process_alive(child_pid, child_ticks):
        print(f"[Coordinator] Child process {child_pid} is already running. Exiting.")
        return 0

    state: Dict[str, Any] = {
        "status": "waiting_predecessor", "predecessor_pid": args.predecessor_pid,
        "predecessor_start_ticks": args.predecessor_start_ticks, "predecessor_root": str(pred_root),
        "joint_root": str(joint_root), "epochs": args.epochs,
    }
    write_state(joint_root, state)

    stop_pred, stop_joint = pred_root / "STOP_PREDICTIVE", joint_root / "STOP_JOINT"

    while is_process_alive(args.predecessor_pid, args.predecessor_start_ticks):
        if stop_pred.exists() or stop_joint.exists():
            state["status"], state["reason"] = "cancelled", "Stop flag detected while waiting for predecessor launcher"
            write_state(joint_root, state)
            return 0
        time.sleep(args.poll_seconds)

    if stop_pred.exists() or stop_joint.exists():
        state["status"], state["reason"] = "cancelled", "Stop flag detected after predecessor launcher exited"
        write_state(joint_root, state)
        return 0

    while find_old_workers(predecessor_cwd=pred_root):
        if stop_pred.exists() or stop_joint.exists():
            state["status"], state["reason"] = "cancelled", "Stop flag detected while waiting for old workers"
            write_state(joint_root, state)
            return 0
        time.sleep(args.poll_seconds)

    stage2_path = pred_root / "formal_two_stage" / "stage2.pt"
    run_cfg_path = pred_root / "formal_two_stage" / "run_config.json"
    try:
        ckpt, stage2_sha, parent_sha = validate_stage2_checkpoint(stage2_path, run_cfg_path)
    except Exception as e:
        state["status"], state["reason"] = "blocked", f"Stage 2 checkpoint validation failed: {e}"
        write_state(joint_root, state)
        return 1

    while True:
        try:
            gpu_pids = get_gpu_compute_pids()
        except RuntimeError as exc:
            state["status"], state["reason"] = "blocked", str(exc)
            write_state(joint_root, state)
            return 1
        if not gpu_pids:
            break
        state["status"], state["gpu_active_pids"] = "waiting_gpu", gpu_pids
        write_state(joint_root, state)
        if stop_pred.exists() or stop_joint.exists():
            state["status"], state["reason"] = "cancelled", "Stop flag detected while waiting for GPU"
            write_state(joint_root, state)
            return 0
        time.sleep(args.poll_seconds)

    if stop_pred.exists() or stop_joint.exists():
        state["status"], state["reason"] = "cancelled", "Stop flag detected before launch"
        write_state(joint_root, state)
        return 0

    formal_joint_dir = joint_root / "formal_joint"
    if formal_joint_dir.exists():
        state["status"], state["reason"] = "blocked", f"formal_joint target directory {formal_joint_dir} already exists"
        write_state(joint_root, state)
        return 1

    launcher_script = joint_root / "scripts" / "run_joint_predictive.sh"
    cmd_gate = [
        "bash", str(launcher_script), str(formal_joint_dir),
        "--init-from-stage2", str(stage2_path), "--epochs", str(args.epochs),
        "--max-updates", "2", "--stop-file", str(stop_joint),
    ]
    state["status"], state["gate_command"] = "running_gate", cmd_gate
    write_state(joint_root, state)

    env = os.environ.copy()
    if args.python:
        env["PYTHON"] = args.python

    with open(logs_dir / "gate.log", "wb") as log_f:
        p_gate = subprocess.Popen(cmd_gate, cwd=joint_root, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        try:
            _, _, ticks = parse_proc_stat(p_gate.pid)
            state["child_pid"], state["child_start_ticks"] = p_gate.pid, ticks
            write_state(joint_root, state)
        except Exception:
            pass
        ret_gate = p_gate.wait()

    if ret_gate != 0:
        state["status"], state["reason"] = "failed", f"Gate process failed with exit code {ret_gate}"
        write_state(joint_root, state)
        return 1

    gate_last_pt = formal_joint_dir / "last.pt"
    try:
        validate_gate_checkpoint(gate_last_pt)
    except Exception as e:
        state["status"], state["reason"] = "failed", f"Gate checkpoint validation failed: {e}"
        write_state(joint_root, state)
        return 1

    if stop_pred.exists() or stop_joint.exists():
        state["status"], state["reason"] = "cancelled", "Stop flag detected after gate"
        write_state(joint_root, state)
        return 0

    cmd_formal = [
        "bash", str(launcher_script), str(formal_joint_dir),
        "--resume", str(gate_last_pt), "--epochs", str(args.epochs),
        "--stop-file", str(stop_joint),
    ]
    state["status"], state["formal_command"] = "running_formal", cmd_formal
    write_state(joint_root, state)

    with open(logs_dir / "formal_joint.log", "wb") as log_f:
        p_formal = subprocess.Popen(cmd_formal, cwd=joint_root, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        try:
            _, _, ticks = parse_proc_stat(p_formal.pid)
            state["child_pid"], state["child_start_ticks"] = p_formal.pid, ticks
            state["child_cmd"] = cmd_formal
            state["stage2_sha256"], state["parent_sha256"] = stage2_sha, parent_sha
            write_state(joint_root, state)
        except Exception:
            pass
        ret_formal = p_formal.wait()

    final_pt = formal_joint_dir / "final.pt"
    if ret_formal == 0 and final_pt.exists():
        final = torch.load(final_pt, map_location="cpu", weights_only=False, mmap=True)
        if final.get("format") != "predictive_memory_joint_v1" or final.get("epoch") != args.epochs or final.get("batch_cursor") != 0:
            state["status"], state["reason"] = "failed", "Final checkpoint has not completed the epoch budget"
        else:
            state["status"], state["reason"] = "completed", "Stage 3 completed successfully, final.pt verified"
    elif stop_joint.exists():
        state["status"], state["reason"] = "stopped", "Stopped cleanly by sentinel file"
    else:
        state["status"], state["reason"] = "failed", f"Formal training exited with code {ret_formal} without final.pt"

    write_state(joint_root, state)
    return 0 if state["status"] == "completed" else 1


def main() -> None:
    sys.exit(run_coordinator(parse_args()))


if __name__ == "__main__":
    main()
