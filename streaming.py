"""Continuous MOSS perception and action-space Flow workers."""

from __future__ import annotations

import math
import threading
import time
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
]


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
