"""Asynchronous MOSS visual streaming and action-chunk planning workers."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Observation",
    "PlannedChunk",
    "LatestObservation",
    "ChunkExecutor",
    "AsyncPerception",
    "ChunkPlanner",
    "ActionRecord",
    "ExecutedActionLedger",
]


@dataclass(frozen=True)
class ActionRecord:
    """Immutable record of an executed action passed to env.step."""

    action: np.ndarray
    timestamp: float
    duration: float
    step_index: int
    command_id: int
    plan_id: str | int | None = None
    event_epoch: int | None = None
    fallback: bool = False

    def copy(self) -> ActionRecord:
        """Return an isolated copy where the action array is fresh."""
        action = self.action.copy()
        action.flags.writeable = False
        return ActionRecord(
            action=action,
            timestamp=self.timestamp,
            duration=self.duration,
            step_index=self.step_index,
            command_id=self.command_id,
            plan_id=self.plan_id,
            event_epoch=self.event_epoch,
            fallback=self.fallback,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.tolist(),
            "timestamp": float(self.timestamp),
            "duration": float(self.duration),
            "step_index": int(self.step_index),
            "command_id": int(self.command_id),
            "plan_id": self.plan_id,
            "event_epoch": self.event_epoch,
            "fallback": bool(self.fallback),
        }


class ExecutedActionLedger:
    """Bounded, thread-safe ledger of executed robot actions."""

    def __init__(self, capacity: int = 10000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._records: deque[ActionRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._total_recorded = 0

    def record(
        self,
        action: np.ndarray,
        timestamp: float | None = None,
        duration: float = 0.0,
        *,
        step_index: int | None = None,
        plan_id: str | int | None = None,
        event_epoch: int | None = None,
        fallback: bool = False,
    ) -> ActionRecord:
        """Record an executed action.

        Validates finite actions, finite timestamps, nonnegative duration,
        and monotonic, non-overlapping intervals against existing records.
        """
        arr = np.asarray(action, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("action must be finite")
        t = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(t):
            raise ValueError("timestamp must be finite")
        dur = float(duration)
        if not math.isfinite(dur) or dur < 0.0:
            raise ValueError("duration must be finite and nonnegative")

        # Copy array and set writeable=False for immutability
        isolated_action = arr.copy()
        isolated_action.flags.writeable = False

        with self._lock:
            cmd_id = self._total_recorded
            idx = cmd_id if step_index is None else int(step_index)
            if self._records:
                last = self._records[-1]
                if t < last.timestamp:
                    raise ValueError(
                        f"monotonicity violation: timestamp {t} < last {last.timestamp}"
                    )
                if t < last.timestamp + last.duration:
                    raise ValueError(
                        f"overlapping boundary: timestamp {t} < last interval end {last.timestamp + last.duration}"
                    )
            entry = ActionRecord(
                action=isolated_action,
                timestamp=t,
                duration=dur,
                step_index=idx,
                command_id=cmd_id,
                plan_id=plan_id,
                event_epoch=event_epoch,
                fallback=fallback,
            )
            self._records.append(entry)
            self._total_recorded += 1
            return entry.copy()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def total_recorded(self) -> int:
        with self._lock:
            return self._total_recorded

    def get_records(self) -> list[ActionRecord]:
        """Return isolated copies of records to prevent external mutation."""
        with self._lock:
            return [r.copy() for r in self._records]

    def query_interval(
        self, start_time: float, end_time: float
    ) -> dict[str, Any]:
        """Query actions active or starting within [start_time, end_time].

        Validates intervals and detects internal gaps via union scan over
        half-open intervals [t_i, t_i + duration).
        """
        start = float(start_time)
        end = float(end_time)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("query intervals must be finite")
        if end < start:
            raise ValueError("end_time must be >= start_time")
        with self._lock:
            records = list(self._records)
            total = self._total_recorded

        if not records:
            return {
                "records": [],
                "covered": False,
                "truncated": total > 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
                "earliest_step": None,
                "latest_step": None,
            }

        earliest_t = records[0].timestamp
        latest_t = records[-1].timestamp + records[-1].duration
        truncated = (total > len(records)) and (start < earliest_t)

        matching: list[ActionRecord] = []
        matching_intervals: list[tuple[float, float]] = []
        for r in records:
            r_start = r.timestamp
            r_end = r.timestamp + r.duration
            if r.duration > 0.0:
                overlaps = (r_start < end) and (r_end > start)
            else:
                overlaps = (start <= r_start <= end)
            if overlaps:
                matching.append(r.copy())
                matching_intervals.append((r_start, r_end))

        # Check coverage over [start, end]
        covered = False
        if not truncated and start >= earliest_t and end <= latest_t:
            if start == end:
                # Point query
                covered = any(iv[0] <= start <= iv[1] for iv in matching_intervals)
            elif matching_intervals:
                # Union scan: merge half-open segments and ensure [start, end] is contiguous without gaps
                matching_intervals.sort(key=lambda iv: iv[0])
                cur_cov_start = matching_intervals[0][0]
                cur_cov_end = matching_intervals[0][1]
                has_internal_gap = False
                for iv_s, iv_e in matching_intervals[1:]:
                    if iv_s > cur_cov_end + 1e-9:  # internal gap found
                        has_internal_gap = True
                        break
                    cur_cov_end = max(cur_cov_end, iv_e)
                if not has_internal_gap and cur_cov_start <= start + 1e-9 and cur_cov_end >= end - 1e-9:
                    covered = True

        return {
            "records": matching,
            "covered": covered,
            "truncated": truncated,
            "earliest_timestamp": records[0].timestamp,
            "latest_timestamp": records[-1].timestamp,
            "earliest_step": records[0].step_index,
            "latest_step": records[-1].step_index,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            count = len(self._records)
            total = self._total_recorded
            if count == 0:
                return {
                    "total_recorded": total,
                    "retained_count": 0,
                    "fallback_count": 0,
                    "fallback_fraction": 0.0,
                    "mean_duration": 0.0,
                }
            fallback_count = sum(1 for r in self._records if r.fallback)
            durations = [r.duration for r in self._records]
            mean_dur = float(np.mean(durations)) if durations else 0.0
            return {
                "total_recorded": total,
                "retained_count": count,
                "fallback_count": fallback_count,
                "fallback_fraction": fallback_count / count,
                "mean_duration": mean_dur,
            }

    def export(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._records]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._total_recorded = 0


@dataclass(frozen=True)
class Observation:
    version: int
    image: Any
    robot_state: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class PlannedChunk:
    """A micro-chunk placed on the wall clock: action j executes at start + j*interval."""

    version: int
    actions: np.ndarray
    start_time: float
    interval: float
    visual_age: float

    def index_at(self, moment: float) -> int:
        if moment < self.start_time:
            return -1
        return int((moment - self.start_time) // self.interval)

    def action_at(self, moment: float) -> np.ndarray | None:
        index = self.index_at(moment)
        if index < 0 or index >= len(self.actions):
            return None
        return self.actions[index]


class ChunkExecutor:
    """Turns overlapping micro-chunks into one action per control tick.

    With `ensemble_lambda` unset the newest chunk simply wins, which is the
    honest measurement baseline: it exposes whatever discontinuity exists at
    chunk boundaries instead of hiding it. Setting a decay enables ACT-style
    temporal ensembling, where older plans keep a vote weighted by
    `exp(-lambda * age)`. Ensembling smooths seams but slows the response to a
    scene change, so the two modes are meant to be compared, not stacked.
    """

    def __init__(
        self,
        action_dim: int = 7,
        *,
        ensemble_lambda: float | None = None,
        max_chunks: int = 4,
    ) -> None:
        if action_dim < 1 or max_chunks < 1:
            raise ValueError("action_dim and max_chunks must be positive")
        if ensemble_lambda is not None and ensemble_lambda < 0:
            raise ValueError("ensemble_lambda must be nonnegative")
        self.action_dim = action_dim
        self.ensemble_lambda = ensemble_lambda
        self.max_chunks = max_chunks
        self._chunks: list[PlannedChunk] = []
        self._version = 0

    def submit(
        self,
        actions: np.ndarray,
        start_time: float,
        interval: float,
        visual_age: float = 0.0,
    ) -> PlannedChunk:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.action_dim:
            raise ValueError(f"chunk must have shape [steps, {self.action_dim}]")
        if not np.isfinite(values).all():
            raise ValueError("planned chunk must be finite")
        if interval <= 0 or not math.isfinite(start_time):
            raise ValueError("chunk interval must be positive and start finite")
        self._version += 1
        chunk = PlannedChunk(
            self._version, values.copy(), float(start_time), float(interval),
            float(visual_age),
        )
        self._chunks.append(chunk)
        del self._chunks[: -self.max_chunks]
        return chunk

    def action_at(self, moment: float) -> np.ndarray:
        live = [
            (chunk, value)
            for chunk in self._chunks
            if (value := chunk.action_at(moment)) is not None
        ]
        if not live:
            raise RuntimeError("no planned chunk covers the requested control tick")
        if self.ensemble_lambda is None:
            return live[-1][1].copy()
        weights = np.asarray(
            [
                math.exp(-self.ensemble_lambda * max(0.0, moment - chunk.start_time))
                for chunk, _ in live
            ],
            dtype=np.float64,
        )
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return live[-1][1].copy()
        stacked = np.stack([value for _, value in live]).astype(np.float64)
        return (stacked * weights[:, None]).sum(axis=0).astype(np.float32) / total

    def clear(self) -> None:
        self._chunks.clear()


class LatestObservation:
    """A bounded one-item stream; slow consumers skip directly to the newest value."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: Observation | None = None
        self._version = 0

    def publish(
        self, image: Any, robot_state: np.ndarray, timestamp: float | None = None
    ) -> int:
        state = np.asarray(robot_state, dtype=np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("robot_state must be one finite vector")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("observation timestamp must be finite")
        with self._condition:
            self._version += 1
            self._value = Observation(
                self._version, image, state.copy(), timestamp
            )
            self._condition.notify_all()
            return self._version

    def wait_for_new(
        self, after_version: int, timeout: float | None = None
    ) -> Observation | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._value is not None and self._value.version > after_version,
                timeout=timeout,
            )
            if self._value is None or self._value.version <= after_version:
                return None
            return self._value

    def latest(self) -> Observation | None:
        with self._condition:
            return self._value

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class AsyncPerception:
    """Continuously appends camera frames to MOSS and publishes its latest memory."""

    def __init__(
        self,
        observations: LatestObservation,
        features: LatestObservation,
        encode_fn: Callable[[Observation], Any],
    ) -> None:
        self.observations = observations
        self.features = features
        self.encode_fn = encode_fn
        self.last_latency = 0.0
        self.frames_encoded = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("perception is already started")
        self._thread = threading.Thread(
            target=self._run, name="moss-stream-perception", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.observations.wake()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous perception failed") from self._error

    def _run(self) -> None:
        version = 0
        try:
            while not self._stop.is_set():
                observation = self.observations.wait_for_new(version, timeout=0.25)
                if observation is None:
                    continue
                version = observation.version
                started = time.monotonic()
                memory = self.encode_fn(observation)
                self.last_latency = time.monotonic() - started
                self.frames_encoded += 1
                self.features.publish(
                    memory, observation.robot_state, observation.timestamp
                )
        except Exception as error:  # noqa: BLE001 - surfaced by the control thread.
            self._error = error
            self._stop.set()


class ChunkPlanner:
    """Replans a micro-chunk whenever a frame lands, using the freshest state.

    Runs slower than the control loop on purpose: the executor interpolates the
    chunk across control ticks, so the backbone only has to keep up with the
    planning rate.
    """

    def __init__(
        self,
        features: LatestObservation,
        executor: ChunkExecutor,
        plan_fn: Callable[[Observation], dict[str, Any]],
        *,
        period: float,
        states: LatestObservation | None = None,
    ) -> None:
        if period <= 0:
            raise ValueError("planning period must be positive")
        self.features = features
        self.executor = executor
        self.plan_fn = plan_fn
        self.period = period
        self.states = states
        self.last_latency = 0.0
        self.chunks_planned = 0
        self.last_visual_age = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("chunk planner is already started")
        self._thread = threading.Thread(
            target=self._run, name="moss-chunk-planner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.features.wake()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def wait_for_first_chunk(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("chunk planner failed") from self._error

    def _run(self) -> None:
        version = 0
        latest_feature = None
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                timeout = (
                    0.25
                    if latest_feature is None
                    else max(0.0, deadline - time.monotonic())
                )
                fresh = self.features.wait_for_new(version, timeout=timeout)
                if fresh is not None:
                    latest_feature = fresh
                    version = fresh.version
                if latest_feature is None or time.monotonic() < deadline:
                    continue
                state = None if self.states is None else self.states.latest()
                condition = (
                    latest_feature
                    if state is None
                    else Observation(
                        latest_feature.version,
                        latest_feature.image,
                        state.robot_state,
                        state.timestamp,
                    )
                )
                started = time.monotonic()
                plan = self.plan_fn(condition)
                created = time.monotonic()
                self.last_latency = created - started
                self.last_visual_age = float(plan.get("visual_age", 0.0))
                self.executor.submit(
                    plan["actions"],
                    plan.get("start_time", created),
                    plan["interval"],
                    self.last_visual_age,
                )
                self.chunks_planned += 1
                self._ready.set()
                deadline = max(deadline + self.period, created)
        except Exception as error:  # noqa: BLE001 - surfaced by the control thread.
            self._error = error
            self._stop.set()
            self._ready.set()
