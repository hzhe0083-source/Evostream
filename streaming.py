"""Continuous MOSS perception and action-space Flow workers."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Observation:
    version: int
    image: Any
    robot_state: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class StreamAction:
    version: int
    value: np.ndarray
    observation_time: float
    created_time: float


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


class LatestAction:
    """A bounded one-item stream of continuously generated robot actions."""

    def __init__(self, action_dim: int = 7) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self._condition = threading.Condition()
        self._value: StreamAction | None = None
        self._version = 0

    def publish(
        self,
        action: np.ndarray,
        observation_time: float,
        created_time: float | None = None,
    ) -> int:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (self.action_dim,) or not np.isfinite(value).all():
            raise ValueError(f"action must be finite with shape [{self.action_dim}]")
        created_time = time.monotonic() if created_time is None else float(created_time)
        if not math.isfinite(observation_time) or not math.isfinite(created_time):
            raise ValueError("action timestamps must be finite")
        with self._condition:
            self._version += 1
            self._value = StreamAction(
                self._version,
                value.copy(),
                float(observation_time),
                created_time,
            )
            self._condition.notify_all()
            return self._version

    def wait_for_new(
        self, after_version: int, timeout: float | None = None
    ) -> StreamAction | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._value is not None and self._value.version > after_version,
                timeout=timeout,
            )
            if self._value is None or self._value.version <= after_version:
                return None
            return self._value


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


class StreamingActionWorker:
    """Advances persistent action flow at control rate using the freshest MOSS memory."""

    def __init__(
        self,
        features: LatestObservation,
        actions: LatestAction,
        step_fn: Callable[[Observation], np.ndarray],
        *,
        period: float,
        states: LatestObservation | None = None,
    ) -> None:
        if period <= 0:
            raise ValueError("action stream period must be positive")
        self.features = features
        self.actions = actions
        self.step_fn = step_fn
        self.period = period
        self.states = states
        self.last_latency = 0.0
        self.actions_generated = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("action worker is already started")
        self._thread = threading.Thread(
            target=self._run, name="streaming-action-flow", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.features.wake()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("streaming action worker failed") from self._error

    def _run(self) -> None:
        version = 0
        memory = None
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                timeout = (
                    0.25
                    if memory is None
                    else max(0.0, deadline - time.monotonic())
                )
                latest = self.features.wait_for_new(version, timeout=timeout)
                if latest is not None:
                    memory = latest
                    version = latest.version
                if memory is None or time.monotonic() < deadline:
                    continue
                state = None if self.states is None else self.states.latest()
                condition = (
                    memory
                    if state is None
                    else Observation(
                        memory.version,
                        memory.image,
                        state.robot_state,
                        state.timestamp,
                    )
                )
                started = time.monotonic()
                action = self.step_fn(condition)
                created = time.monotonic()
                self.last_latency = created - started
                self.actions_generated += 1
                self.actions.publish(action, condition.timestamp, created)
                deadline = max(deadline + self.period, created)
        except Exception as error:  # noqa: BLE001 - surfaced by the control thread.
            self._error = error
            self._stop.set()
