#!/usr/bin/env python3
"""stop_at_native_checkpoint.py - Operator CLI to stop legacy training at next atomic checkpoint."""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class HandOffError(Exception):
    """Base exception for handoff errors."""
    pass


class ProcessIdentityError(HandOffError):
    """Raised when process identity, relationship, or cmdline validation fails."""
    pass


class CheckpointValidationError(HandOffError):
    """Raised when checkpoint contract or state validation fails."""
    pass


def parse_proc_stat(pid: int, proc_root: str = "/proc") -> Tuple[str, int, int]:
    stat_path = Path(proc_root) / str(pid) / "stat"
    if not stat_path.exists():
        raise ProcessIdentityError(f"Process {pid} stat not found: {stat_path}")
    content = stat_path.read_text(encoding="utf-8", errors="replace").strip()
    rparen = content.rfind(")")
    if rparen == -1 or rparen + 2 >= len(content):
        raise ProcessIdentityError(f"Malformed stat line for PID {pid}")
    fields = content[rparen + 2:].split()
    if len(fields) < 20:
        raise ProcessIdentityError(f"Insufficient fields in stat for PID {pid}")
    return fields[0], int(fields[1]), int(fields[19])


def parse_proc_cmdline(pid: int, proc_root: str = "/proc") -> List[str]:
    cmd_path = Path(proc_root) / str(pid) / "cmdline"
    if not cmd_path.exists():
        raise ProcessIdentityError(f"Process {pid} cmdline not found: {cmd_path}")
    raw = cmd_path.read_bytes()
    return [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]


def has_token_pair(args: List[str], flag: str, val: str) -> bool:
    for i in range(len(args) - 1):
        if args[i] == flag and args[i + 1] == val:
            return True
    return False


def get_exact_output_dir(args: List[str]) -> Optional[Path]:
    for i, a in enumerate(args):
        if a == "--output-dir" and i + 1 < len(args):
            return Path(args[i + 1]).resolve()
        if a.startswith("--output-dir="):
            return Path(a.split("=", 1)[1]).resolve()
    return None


def validate_launcher_cmdline(args: List[str], expected_run_dir: Path) -> None:
    if not has_token_pair(args, "-m", "torch.distributed.run"):
        raise ProcessIdentityError("Launcher missing exact '-m torch.distributed.run'")
    if not has_token_pair(args, "-m", "fabri_moss.train_native"):
        raise ProcessIdentityError("Launcher missing exact '-m fabri_moss.train_native'")
    out_dir = get_exact_output_dir(args)
    if out_dir is None or out_dir != expected_run_dir.resolve():
        raise ProcessIdentityError(f"Launcher output dir {out_dir} != expected {expected_run_dir.resolve()}")


def validate_worker_cmdline(args: List[str], expected_run_dir: Path) -> None:
    if not has_token_pair(args, "-m", "fabri_moss.train_native"):
        raise ProcessIdentityError("Worker missing exact '-m fabri_moss.train_native'")
    out_dir = get_exact_output_dir(args)
    if out_dir is None or out_dir != expected_run_dir.resolve():
        raise ProcessIdentityError(f"Worker output dir {out_dir} != expected {expected_run_dir.resolve()}")


def verify_process_identity(
    pid: int,
    expected_ticks: Optional[int],
    expected_ppid: Optional[int],
    expected_run_dir: Path,
    role: str,
    allow_ppid_1: bool = False,
    proc_root: str = "/proc",
) -> Tuple[str, int, int]:
    state, ppid, ticks = parse_proc_stat(pid, proc_root=proc_root)
    if expected_ticks is not None and ticks != expected_ticks:
        raise ProcessIdentityError(f"PID {pid} start_ticks {ticks} != expected {expected_ticks} (PID reuse)")
    if expected_ppid is not None and ppid != expected_ppid:
        if not (allow_ppid_1 and ppid == 1):
            raise ProcessIdentityError(f"PID {pid} PPID {ppid} != expected {expected_ppid}")
    cmdline = parse_proc_cmdline(pid, proc_root=proc_root)
    if role == "launcher":
        validate_launcher_cmdline(cmdline, expected_run_dir)
    else:
        validate_worker_cmdline(cmdline, expected_run_dir)
    return state, ppid, ticks


def safe_send_signal(
    pid: int,
    sig: int,
    expected_ticks: int,
    expected_ppid: Optional[int],
    expected_run_dir: Path,
    role: str,
    allow_ppid_1: bool = False,
    proc_root: str = "/proc",
) -> None:
    verify_process_identity(pid, expected_ticks, expected_ppid, expected_run_dir, role, allow_ppid_1, proc_root)
    os.kill(pid, sig)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def exclusive_archive_checkpoint(src: Path, dst: Path) -> Tuple[str, str, str]:
    if dst.is_symlink() or dst.exists() or os.path.lexists(dst):
        raise FileExistsError(f"Archive destination exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_sha = compute_file_sha256(src)
    try:
        os.link(str(src), str(dst))
        method = "hardlink"
        if src.stat().st_ino == dst.stat().st_ino:
            dst_sha = src_sha
        else:
            dst_sha = compute_file_sha256(dst)
    except OSError:
        method = "exclusive_copy"
        fd = os.open(str(dst), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with open(src, "rb") as sf, open(fd, "wb", closefd=False) as df:
                while buf := sf.read(1024 * 1024):
                    df.write(buf)
        finally:
            os.close(fd)
        dst_sha = compute_file_sha256(dst)
    if src_sha != dst_sha:
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointValidationError(f"Archive SHA256 mismatch: {src_sha} != {dst_sha}")
    return method, src_sha, dst_sha


def write_exclusive_initial_report(path: Path, data: Dict[str, Any]) -> Tuple[int, int]:
    if path.is_symlink() or path.exists() or os.path.lexists(path):
        raise FileExistsError(f"Report path exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".tmp_{path.name}_{os.getpid()}_{int(time.time()*1e6)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(str(tmp_path), str(path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    st = path.stat()
    return st.st_ino, st.st_dev


def update_owned_report(path: Path, data: Dict[str, Any], expected_ino: int, expected_dev: int) -> Tuple[int, int]:
    cur_st = path.stat()
    if cur_st.st_ino != expected_ino or cur_st.st_dev != expected_dev:
        raise HandOffError(f"Report inode changed: expected ({expected_dev},{expected_ino}), got ({cur_st.st_dev},{cur_st.st_ino})")
    tmp_path = path.parent / f".tmp_{path.name}_{os.getpid()}_{int(time.time()*1e6)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))
    st = path.stat()
    return st.st_ino, st.st_dev


def read_latest_train_metric(metrics_path: Path) -> Optional[Dict[str, Any]]:
    if not metrics_path.exists():
        return None
    raw = metrics_path.read_bytes()
    if not raw:
        return None
    last_metric = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
            if isinstance(rec, dict) and "global_step" in rec and "epoch" in rec and "loss" in rec:
                last_metric = rec
        except Exception:
            return None
    return last_metric


def validate_checkpoint_structure(ckpt_path: Path, expected_metric: Dict[str, Any]) -> Dict[str, Any]:
    orig_th = torch.get_num_threads()
    torch.set_num_threads(min(orig_th, 2))
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:
        raise CheckpointValidationError(f"Raw torch.load mmap failed: {e}") from e
    finally:
        torch.set_num_threads(orig_th)

    if not isinstance(ckpt, dict):
        raise CheckpointValidationError("Checkpoint root is not a dictionary")

    for k in ("model", "optimizer", "scheduler", "global_step", "epoch", "batch_cursor",
              "rng_states_per_rank", "training_contract", "train_data_contract",
              "val_data_contract", "source_metadata"):
        if k not in ckpt:
            raise CheckpointValidationError(f"Checkpoint missing required key: {k}")

    # Core non-empty requirements
    model_state = ckpt["model"]
    if not isinstance(model_state, dict) or not model_state:
        raise CheckpointValidationError("model state dict must be a non-empty dictionary")

    opt_state = ckpt["optimizer"]
    if not isinstance(opt_state, dict):
        raise CheckpointValidationError("optimizer state must be a dictionary")
    param_groups = opt_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise CheckpointValidationError("optimizer param_groups must be a non-empty list")
    if not isinstance(opt_state.get("state"), dict) or not opt_state["state"]:
        raise CheckpointValidationError("optimizer inner state must be a dictionary")

    for k, v in model_state.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32:
            raise CheckpointValidationError(f"Model param {k} dtype {v.dtype} != torch.float32")

    for p_id, p_dict in opt_state.get("state", {}).items():
        if not isinstance(p_dict, dict):
            continue
        for mk, mv in p_dict.items():
            if torch.is_tensor(mv) and mv.is_floating_point() and mv.dtype != torch.float32:
                raise CheckpointValidationError(f"Optimizer {p_id} {mk} dtype {mv.dtype} != torch.float32")

    # RNG states: 2 ranks, each non-empty dict with python, numpy, torch_cpu, torch_cuda
    rng_list = ckpt["rng_states_per_rank"]
    if not isinstance(rng_list, list) or len(rng_list) != 2:
        raise CheckpointValidationError(f"Expected 2 RNG states, got {len(rng_list) if isinstance(rng_list, list) else type(rng_list)}")
    for i, r in enumerate(rng_list):
        if not isinstance(r, dict) or not r:
            raise CheckpointValidationError(f"RNG rank {i} must be a non-empty dict")
        for rk in ("python", "numpy", "torch_cpu", "torch_cuda"):
            if rk not in r or r[rk] is None:
                raise CheckpointValidationError(f"RNG rank {i} missing {rk}")

    # Source SHA and contracts
    src_meta = ckpt["source_metadata"]
    if not isinstance(src_meta, dict) or "checkpoint_sha256" not in src_meta or not src_meta["checkpoint_sha256"]:
        raise CheckpointValidationError("source_metadata missing non-empty checkpoint_sha256")
    for ck in ("config", "norm_stats"):
        if ck not in ckpt or not isinstance(ckpt[ck], dict) or not ckpt[ck]:
            raise CheckpointValidationError(f"Checkpoint missing required contract {ck}")

    ckpt_step = int(ckpt["global_step"])
    sched = ckpt.get("scheduler", {})
    sched_step = sched.get("last_epoch") if isinstance(sched, dict) else None
    if sched_step != ckpt_step:
        raise CheckpointValidationError(f"Scheduler last_epoch {sched_step} != checkpoint global_step {ckpt_step}")

    ckpt_epoch = int(ckpt["epoch"])
    ckpt_cursor = int(ckpt["batch_cursor"])
    ckpt_targets = int(ckpt.get("epoch_targets_seen", 0))

    # Metric step consistency
    metric_step = int(expected_metric["global_step"])
    if ckpt_step != metric_step:
        raise CheckpointValidationError(f"STEP_MISMATCH: ckpt {ckpt_step} != metric {metric_step}")

    # Resume geometry verification
    train_contract = ckpt.get("training_contract", {})
    train_data_contract = ckpt.get("train_data_contract", {})
    seg_count = train_data_contract.get("segment_count")
    g_batch = train_contract.get("global_batch_size")
    if type(seg_count) is not int or seg_count <= 0 or type(g_batch) is not int or g_batch <= 0:
        raise CheckpointValidationError("Resume geometry requires positive segment_count and global_batch_size")
    if ckpt_epoch < 0 or not 0 <= ckpt_cursor <= seg_count or (ckpt_cursor % g_batch and ckpt_cursor != seg_count):
        raise CheckpointValidationError("Resume geometry has an invalid epoch or batch cursor")
    exp_step = ckpt_epoch * math.ceil(seg_count / g_batch) + math.ceil(ckpt_cursor / g_batch)
    if ckpt_step != exp_step:
        raise CheckpointValidationError(f"Resume geometry mismatch: step {ckpt_step} != calculated {exp_step}")

    metric_epoch = int(expected_metric["epoch"])
    metric_targets = int(expected_metric.get("epoch_targets_seen", 0))
    if ckpt_epoch == metric_epoch:
        if ckpt_targets != metric_targets:
            raise CheckpointValidationError(f"Targets mismatch: ckpt {ckpt_targets} != metric {metric_targets}")
    elif ckpt_epoch == metric_epoch + 1:
        if ckpt_cursor != 0 or ckpt_targets != 0:
            raise CheckpointValidationError(f"Epoch boundary ckpt cursor/targets non-zero: ({ckpt_cursor}, {ckpt_targets})")
    else:
        raise CheckpointValidationError(f"Epoch mismatch: ckpt {ckpt_epoch} vs metric {metric_epoch}")

    return {"global_step": ckpt_step, "epoch": ckpt_epoch, "batch_cursor": ckpt_cursor, "epoch_targets_seen": ckpt_targets}


def get_last_checkpoint_stat(ckpt_path: Path) -> Optional[Tuple[int, int]]:
    if not ckpt_path.exists():
        return None
    st = ckpt_path.stat()
    return st.st_ino, st.st_mtime_ns


def poll_brief_deadline(check_fn, timeout: float = 0.5, step: float = 0.01) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if check_fn():
            return True
        time.sleep(step)
    return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stop native training at next atomic checkpoint.")
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--launcher-pid", required=True, type=int)
    p.add_argument("--launcher-start-ticks", required=True, type=int)
    p.add_argument("--worker-pids", required=True, nargs=2, type=int)
    p.add_argument("--archive-path", required=True, type=str)
    p.add_argument("--report-path", required=True, type=str)
    p.add_argument("--timeout-seconds", type=float, default=3600.0)
    p.add_argument("--poll-seconds", type=float, default=0.05)
    p.add_argument("--execute", action="store_true", default=False)
    return p


def main(raw_args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(raw_args)

    run_dir = Path(args.run_dir).resolve()
    archive_path = Path(args.archive_path).resolve()
    report_path = Path(args.report_path).resolve()
    launcher_pid = args.launcher_pid
    launcher_ticks = args.launcher_start_ticks
    worker_pids = args.worker_pids

    # Validate arguments
    all_pids = [launcher_pid] + worker_pids
    if any(p <= 0 for p in all_pids) or len(all_pids) != len(set(all_pids)):
        print("Error: PIDs must be positive and unique", file=sys.stderr)
        return 1
    if not all(math.isfinite(v) and v > 0 for v in (args.timeout_seconds, args.poll_seconds)):
        print("Error: timeout and poll seconds must be positive", file=sys.stderr)
        return 1
    if archive_path == report_path:
        print("Error: archive_path and report_path must be different", file=sys.stderr)
        return 1
    if archive_path.is_symlink() or archive_path.exists() or os.path.lexists(archive_path):
        print(f"Error: archive_path exists: {archive_path}", file=sys.stderr)
        return 1
    if report_path.is_symlink() or report_path.exists() or os.path.lexists(report_path):
        print(f"Error: report_path exists: {report_path}", file=sys.stderr)
        return 1
    last_pt = run_dir / "last.pt"
    if archive_path == last_pt.resolve():
        print("Error: archive_path cannot be source last.pt", file=sys.stderr)
        return 1

    # Immutable identities: (pid, start_ticks, role, expected_ppid)
    identities: List[Tuple[int, int, str, Optional[int]]] = []
    worker_ticks: Dict[int, int] = {}
    try:
        l_state, _, l_ticks = verify_process_identity(launcher_pid, launcher_ticks, None, run_dir, "launcher")
        if l_state in ("T", "t", "Z"):
            print(f"Error: Launcher initial state invalid: {l_state}", file=sys.stderr)
            return 1
        identities.append((launcher_pid, launcher_ticks, "launcher", None))
        for wp in worker_pids:
            w_state, _, w_ticks = verify_process_identity(wp, None, launcher_pid, run_dir, "worker")
            if w_state in ("T", "t", "Z"):
                print(f"Error: Worker {wp} initial state invalid: {w_state}", file=sys.stderr)
                return 1
            worker_ticks[wp] = w_ticks
            identities.append((wp, w_ticks, "worker", launcher_pid))
    except HandOffError as e:
        print(f"Error: Initial identity verification failed: {e}", file=sys.stderr)
        return 1

    metrics_path = run_dir / "train_metrics.jsonl"
    init_stat = get_last_checkpoint_stat(last_pt)
    if not args.execute:
        print(json.dumps({
            "mode": "dry_run",
            "launcher": {"pid": launcher_pid, "ticks": launcher_ticks},
            "workers": [{"pid": wp, "ticks": worker_ticks[wp]} for wp in worker_pids],
            "checkpoint_stat": init_stat,
        }, indent=2))
        return 0

    owned_stops: Set[int] = set()
    report_meta: Optional[Tuple[int, int]] = None
    exit_code = 1

    def resume_all_owned():
        for pid, ticks, role, ppid in reversed(identities):
            if pid in owned_stops:
                try:
                    safe_send_signal(pid, signal.SIGCONT, ticks, ppid, run_dir, role, allow_ppid_1=True)
                except Exception as ex:
                    print(f"Warning: SIGCONT failed for PID {pid}: {ex}", file=sys.stderr)
                owned_stops.discard(pid)

    try:
        deadline = time.time() + args.timeout_seconds
        last_seen_stat = init_stat
        while time.time() < deadline:
            # Periodic poll: verify PPID launcher
            for pid, ticks, role, ppid in identities:
                verify_process_identity(pid, ticks, ppid, run_dir, role, allow_ppid_1=False)

            cur_stat = get_last_checkpoint_stat(last_pt)
            if cur_stat is not None and cur_stat != last_seen_stat:
                # Pause launcher first, then workers
                for pid, ticks, role, ppid in identities:
                    safe_send_signal(pid, signal.SIGSTOP, ticks, ppid, run_dir, role, allow_ppid_1=False)
                    owned_stops.add(pid)

                # Wait for state 'T' via brief deadline
                all_stopped = poll_brief_deadline(
                    lambda: all(parse_proc_stat(p)[0] == "T" for p, _, _, _ in identities),
                    timeout=0.5,
                )
                if not all_stopped:
                    resume_all_owned()
                    print("Error: Processes failed to enter stopped state 'T'", file=sys.stderr)
                    return 1

                # Read metrics
                metric = read_latest_train_metric(metrics_path)
                if metric is None:
                    # Incomplete metrics -> resume and wait next save
                    resume_all_owned()
                    last_seen_stat = cur_stat
                    time.sleep(args.poll_seconds)
                    continue

                # Validate checkpoint
                try:
                    ckpt_info = validate_checkpoint_structure(last_pt, metric)
                except CheckpointValidationError as cve:
                    if "STEP_MISMATCH" in str(cve):
                        resume_all_owned()
                        last_seen_stat = cur_stat
                        time.sleep(args.poll_seconds)
                        continue
                    else:
                        resume_all_owned()
                        raise

                # Archive checkpoint
                method, src_sha, dst_sha = exclusive_archive_checkpoint(last_pt, archive_path)

                # Reread archive contract
                validate_checkpoint_structure(archive_path, metric)

                # Write exclusive initial report: checkpoint_preserved
                rep_data = {
                    "status": "checkpoint_preserved",
                    "timestamp": time.time(),
                    "run_dir": str(run_dir),
                    "archive_path": str(archive_path),
                    "archive_method": method,
                    "sha256": dst_sha,
                    "checkpoint_info": ckpt_info,
                    "metric": metric,
                    "launcher": {"pid": launcher_pid, "ticks": launcher_ticks},
                    "workers": [{"pid": wp, "ticks": worker_ticks[wp]} for wp in worker_pids],
                }
                report_meta = write_exclusive_initial_report(report_path, rep_data)

                # Controlled termination:
                # 1. Send SIGTERM to exact workers while still stopped
                for wp in worker_pids:
                    safe_send_signal(wp, signal.SIGTERM, worker_ticks[wp], launcher_pid, run_dir, "worker", allow_ppid_1=True)
                # 2. Send SIGTERM to launcher
                safe_send_signal(launcher_pid, signal.SIGTERM, launcher_ticks, None, run_dir, "launcher", allow_ppid_1=False)
                # 3. Send SIGCONT to all owned (workers + launcher) so they process SIGTERM
                resume_all_owned()

                # Wait up to 30s for all known processes to terminate
                def all_gone():
                    for pid, ticks, _, _ in identities:
                        try:
                            st, _, cur_ticks = parse_proc_stat(pid)
                            if cur_ticks == ticks and st != "Z":
                                return False
                        except ProcessIdentityError:
                            pass
                    return True

                term_ok = poll_brief_deadline(all_gone, timeout=30.0, step=0.1)
                final_status = "stopped" if term_ok else "stop_failed"
                rep_data["status"] = final_status
                rep_data["final_timestamp"] = time.time()
                update_owned_report(report_path, rep_data, report_meta[0], report_meta[1])

                if term_ok:
                    exit_code = 0
                else:
                    print("Error: Processes still alive after 30s SIGTERM wait", file=sys.stderr)
                    exit_code = 1
                break

            time.sleep(args.poll_seconds)

    except BaseException as e:
        resume_all_owned()
        if not isinstance(e, SystemExit):
            print(f"Error during execution: {e}", file=sys.stderr)
        return 1
    finally:
        resume_all_owned()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
