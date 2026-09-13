from __future__ import annotations

import collections
import dataclasses
import math
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import torch
import numpy as np
from PIL import Image

from fabri_moss.core import FrameKV, FrameKVSession, MossInternVL
from fabri_moss.delta import DeltaMemoryState


@dataclasses.dataclass(frozen=True)
class Observation:
    frame_id: int
    capture_time: float
    images: Union[Tuple[Any, ...], List[Any]]
    state: torch.Tensor
    state_mask: torch.Tensor
    action_mask: torch.Tensor
    observation_time: Optional[float] = None

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative int, got {self.frame_id}")
        if type(self.capture_time) is bool or not (
            isinstance(self.capture_time, (int, float))
            and math.isfinite(self.capture_time)
            and self.capture_time >= 0.0
        ):
            raise ValueError(f"capture_time must be finite non-negative float, got {self.capture_time}")
        if self.observation_time is not None:
            if type(self.observation_time) is bool or not (
                isinstance(self.observation_time, (int, float))
                and math.isfinite(self.observation_time)
                and self.observation_time >= 0.0
            ):
                raise ValueError(
                    f"observation_time must be finite non-negative float or None, got {self.observation_time}"
                )
        if not isinstance(self.images, (tuple, list)) or len(self.images) == 0:
            raise ValueError("images must be non-empty tuple or list")
        for name, t in (("state", self.state), ("state_mask", self.state_mask), ("action_mask", self.action_mask)):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} must be torch.Tensor, got {type(t).__name__}")
            if t.device.type != 'cpu':
                raise ValueError(f"{name} must be on CPU at the observation handoff")
            if not torch.isfinite(t).all():
                raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        if self.state.ndim != 2 or self.state.shape[0] != 1:
            raise ValueError(f"state must be 2D [1, D], got shape {tuple(self.state.shape)}")
        if self.state_mask.ndim != 2 or self.state_mask.shape[0] != 1 or self.state_mask.shape[1] != self.state.shape[1]:
            raise ValueError(
                f"state_mask must match state shape [1, D], got {tuple(self.state_mask.shape)} vs {tuple(self.state.shape)}"
            )
        if not (self.action_mask.ndim in (2, 3) and self.action_mask.shape[0] == 1):
            raise ValueError(f"action_mask must be [1, D] or [1, H, D], got {tuple(self.action_mask.shape)}")


@dataclasses.dataclass(frozen=True)
class EncodedFrame:
    observation: Observation
    payload: Any
    encode_started: float
    encode_finished: float


@dataclasses.dataclass(frozen=True)
class PlanComputation:
    actions: torch.Tensor
    deep: Optional[torch.Tensor] = None
    shallow: Optional[torch.Tensor] = None
    next_memory: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.actions, torch.Tensor):
            raise TypeError(f"actions must be torch.Tensor, got {type(self.actions).__name__}")
        if self.actions.ndim != 3 or self.actions.shape[0] != 1 or self.actions.shape[1] == 0 or self.actions.shape[2] == 0:
            raise ValueError(f"actions must be 3D [1, H, D] non-empty, got shape {tuple(self.actions.shape)}")
        if not torch.isfinite(self.actions).all():
            raise ValueError("actions tensor contains non-finite values (NaN or Inf)")


@dataclasses.dataclass(frozen=True)
class ObservationMetadata:
    frame_id: int
    capture_time: float
    observation_time: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class ConsumedFrame:
    observation: ObservationMetadata
    encode_started: float
    encode_finished: float


@dataclasses.dataclass(frozen=True)
class PlanResult:
    episode_id: str
    generation: int
    frames: Tuple[ConsumedFrame, ...]
    source_frame_id: int
    capture_time: float
    started: float
    finished: float
    computation: PlanComputation


def _copy_image_item(img: Any) -> Any:
    if isinstance(img, torch.Tensor):
        if img.device.type != 'cpu':
            raise ValueError("Images must be on CPU at the observation handoff")
        return img.detach().clone()
    if isinstance(img, (Image.Image, np.ndarray)):
        return img.copy()
    raise TypeError("Images must be PIL images, NumPy arrays, or CPU tensors")


def _copy_observation(obs: Observation) -> Observation:
    copied_images = tuple(_copy_image_item(img) for img in obs.images)
    return Observation(
        frame_id=obs.frame_id,
        capture_time=obs.capture_time,
        images=copied_images,
        state=obs.state.detach().clone(),
        state_mask=obs.state_mask.detach().clone(),
        action_mask=obs.action_mask.detach().clone(),
        observation_time=obs.observation_time,
    )


def _validate_next_memory(
    candidate: Any,
    previous: Optional[Any],
    snapshot: Tuple[EncodedFrame, ...],
    prompt: str,
) -> DeltaMemoryState:
    """Validate candidate next_memory strictly and return its detached state.

    All temporary local references are confined within this helper.
    """
    if not isinstance(candidate, DeltaMemoryState):
        raise TypeError(f"next_memory must be DeltaMemoryState, got {type(candidate).__name__}")

    if not isinstance(candidate.matrices, (tuple, list)) or len(candidate.matrices) == 0:
        raise ValueError("next_memory matrices must be a non-empty tuple or list")

    for i, m in enumerate(candidate.matrices):
        if not isinstance(m, torch.Tensor):
            raise TypeError(f"next_memory matrix {i} must be a torch.Tensor, got {type(m).__name__}")
        if m.dtype != torch.float32:
            raise TypeError(f"next_memory matrix {i} must be float32, got {m.dtype}")
        if m.ndim != 4:
            raise ValueError(f"next_memory matrix {i} must be 4D [B, Hkv, d, d], got shape {tuple(m.shape)}")
        if m.shape[-2] != m.shape[-1]:
            raise ValueError(
                f"next_memory matrix {i} must be square in last two dims, got shape {tuple(m.shape)}"
            )
        if not torch.isfinite(m).all():
            raise ValueError(f"next_memory matrix {i} contains non-finite values (NaN or Inf)")

    old_count = previous.frame_count if previous is not None else 0
    expected_count = old_count + len(snapshot)
    if candidate.frame_count != expected_count:
        raise ValueError(
            f"next_memory frame_count mismatch: expected {expected_count}, got {candidate.frame_count}"
        )

    last_snapshot_id = snapshot[-1].observation.frame_id
    if candidate.last_frame_id != last_snapshot_id:
        raise ValueError(
            f"next_memory last_frame_id must equal snapshot latest frame {last_snapshot_id}, got {candidate.last_frame_id}"
        )

    if candidate.prompt != prompt:
        raise ValueError(
            f"next_memory prompt '{candidate.prompt}' does not match active prompt '{prompt}'"
        )

    if previous is not None:
        if candidate.owner is not previous.owner:
            raise ValueError("next_memory owner mismatch with previous memory")
        if candidate.revision != previous.revision:
            raise ValueError(
                f"next_memory revision {candidate.revision} does not match previous {previous.revision}"
            )
        if len(candidate.matrices) != len(previous.matrices):
            raise ValueError(
                f"next_memory layers count {len(candidate.matrices)} mismatch with previous {len(previous.matrices)}"
            )
        for i, (m_cand, m_prev) in enumerate(zip(candidate.matrices, previous.matrices)):
            if m_cand.shape != m_prev.shape:
                raise ValueError(
                    f"next_memory matrix {i} shape {tuple(m_cand.shape)} mismatch with previous {tuple(m_prev.shape)}"
                )

    detached_state = candidate.detached()
    return detached_state


class AsyncVisualPlanner:
    def __init__(
        self,
        encode: Callable[[Observation], Any],
        plan: Union[
            Callable[[Tuple[EncodedFrame, ...], str], PlanComputation],
            Callable[[Tuple[EncodedFrame, ...], str, Optional[Any]], PlanComputation],
        ],
        max_frames: Optional[int] = 5,
        max_pending: Optional[int] = 8,
        validate: Optional[Callable[[], None]] = None,
        stateful: bool = False,
        memory_validator: Optional[Callable[[Any, Optional[Any], Tuple[EncodedFrame, ...], str], Any]] = None,
        max_ready: Optional[int] = None,
        overflow_policy: str = "drop_oldest",
        reset_hook: Optional[Callable[[str, str], None]] = None,
        frame_commit_hook: Optional[Callable[[EncodedFrame], None]] = None,
    ) -> None:
        if max_frames is not None:
            if type(max_frames) is bool or not (isinstance(max_frames, int) and max_frames > 0):
                raise ValueError(f"max_frames must be positive int or None, got {max_frames}")
            if max_ready is not None:
                raise ValueError("max_ready is only allowed when max_frames is None (dynamic mode)")
            effective_max_ready = max_frames
        else:
            if max_ready is not None:
                if type(max_ready) is bool or not (isinstance(max_ready, int) and max_ready > 0):
                    raise ValueError(f"max_ready must be positive int or None, got {max_ready}")
            effective_max_ready = max_ready

        if max_pending is not None:
            if type(max_pending) is bool or not (isinstance(max_pending, int) and max_pending > 0):
                raise ValueError(f"max_pending must be positive int or None, got {max_pending}")

        if overflow_policy not in ("drop_oldest", "error"):
            raise ValueError(f"overflow_policy must be 'drop_oldest' or 'error', got {overflow_policy}")

        self.encode_fn = encode
        self.plan_fn = plan
        self.max_frames = max_frames
        self.max_pending = max_pending
        self.max_ready = max_ready
        self.effective_max_ready = effective_max_ready
        self.overflow_policy = overflow_policy
        self.validate_fn = validate
        self.stateful = bool(stateful)
        self.memory_validator = memory_validator
        # Optional lifecycle hooks are intentionally tiny: they let a consume
        # callback own an episode session without changing the old callback
        # signatures.  Function attributes provide automatic wiring from
        # ``make_moss_callbacks``; explicit arguments win for custom callers.
        self.reset_hook = reset_hook or getattr(plan, "reset_session", None) or getattr(plan, "reset", None)
        self.frame_commit_hook = (
            frame_commit_hook
            or getattr(plan, "commit_frame", None)
            or getattr(plan, "append_frame", None)
        )
        self._frame_session = getattr(plan, "frame_session", None)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._closed = False
        self._generation = 0
        self._episode_id: Optional[str] = None
        self._prompt: Optional[str] = None

        self._memory: Optional[Any] = None

        self._time_source: Optional[str] = None
        self._episode_capture_origin: Optional[float] = None
        self._last_observation_time: Optional[float] = None

        self._pending_raw: Deque[Tuple[Observation, int]] = collections.deque()
        self._ready: Deque[EncodedFrame] = collections.deque()
        self._reservation: Optional[Tuple[EncodedFrame, ...]] = None
        self._completed: Optional[PlanResult] = None
        self._vision_error: Optional[BaseException] = None
        self._planner_error: Optional[BaseException] = None

        self._vision_busy = False
        self._planner_busy = False
        self._vision_inflight_id: Optional[int] = None
        self._planner_inflight_ids: List[int] = []

        self._last_submitted_id = -1
        self._last_encoded_id = -1
        self._last_consumed_id = -1
        self._encoded_count = 0
        self._consumed_count = 0
        self._dropped_pending = 0
        self._dropped_ready = 0
        self._rejected_pending = 0
        self._plans_completed = 0
        self._stale_outputs_discarded = 0
        self._errors_count = 0
        self._events: Deque[Dict[str, Any]] = collections.deque(maxlen=256)

        self._planner_task: Optional[Tuple[int, str, Tuple[EncodedFrame, ...], Optional[Any]]] = None

        self._vision_thread = threading.Thread(
            target=self._vision_worker_loop, name="AsyncVisionWorker", daemon=False
        )
        self._planner_thread = threading.Thread(
            target=self._planner_worker_loop, name="AsyncPlannerWorker", daemon=False
        )
        self._vision_thread.start()
        self._planner_thread.start()

    def __enter__(self) -> "AsyncVisualPlanner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def reset(self, episode_id: str, prompt: str) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty string")

        if self.validate_fn is not None:
            self.validate_fn()

        with self._cv:
            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")

            self._generation += 1
            gen = self._generation
            self._episode_id = episode_id.strip()
            self._prompt = prompt.strip()

            self._memory = None
            self._time_source = None
            self._episode_capture_origin = None
            self._last_observation_time = None

            self._pending_raw.clear()
            self._ready.clear()
            self._reservation = None
            self._completed = None
            self._vision_error = None
            self._planner_error = None

            self._last_submitted_id = -1
            self._last_encoded_id = -1
            self._last_consumed_id = -1
            self._encoded_count = 0
            self._consumed_count = 0
            self._dropped_pending = 0
            self._dropped_ready = 0
            self._rejected_pending = 0
            self._plans_completed = 0
            self._stale_outputs_discarded = 0
            self._errors_count = 0

            if self._planner_task is not None and self._planner_task[0] != gen:
                self._planner_task = None

            self._events.append(
                {
                    "type": "reset",
                    "generation": gen,
                    "episode_id": self._episode_id,
                    "timestamp": time.monotonic(),
                }
            )
            if self.reset_hook is not None:
                # Clear callback-owned episode state before any new-generation
                # observation can be committed.
                self.reset_hook(self._episode_id, self._prompt)
            self._cv.notify_all()

    def submit(self, observation: Observation) -> bool:
        if not isinstance(observation, Observation):
            raise TypeError(f"observation must be an Observation instance, got {type(observation).__name__}")

        if self.validate_fn is not None:
            self.validate_fn()

        with self._cv:
            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")
            if self._episode_id is None:
                raise RuntimeError("AsyncVisualPlanner has not been reset for an episode")
            if self._vision_error is not None:
                raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error

            if observation.frame_id <= self._last_submitted_id:
                raise ValueError(
                    f"frame_id {observation.frame_id} must be strictly greater than last submitted {self._last_submitted_id}"
                )

            # Check capacity in error policy before deep copying
            if (
                self.max_pending is not None
                and self.overflow_policy == "error"
                and len(self._pending_raw) >= self.max_pending
            ):
                self._rejected_pending += 1
                self._events.append(
                    {
                        "type": "rejected_pending",
                        "frame_id": observation.frame_id,
                        "generation": self._generation,
                        "timestamp": time.monotonic(),
                    }
                )
                raise BufferError(
                    f"Pending queue full (capacity {self.max_pending}) with overflow_policy='error'"
                )

            # Validate time source and ordering before deep copying
            is_provided = observation.observation_time is not None
            if self._time_source is not None:
                if is_provided and self._time_source != "provided":
                    raise ValueError(
                        f"Cannot mix observation_time sources: episode initialized as {self._time_source}, got provided observation_time"
                    )
                if not is_provided and self._time_source != "capture_relative":
                    raise ValueError(
                        f"Cannot mix observation_time sources: episode initialized as {self._time_source}, got None observation_time"
                    )

            if is_provided:
                cand_obs_time = observation.observation_time
                assert cand_obs_time is not None
            else:
                cand_origin = (
                    observation.capture_time
                    if self._episode_capture_origin is None
                    else self._episode_capture_origin
                )
                cand_obs_time = observation.capture_time - cand_origin

            if type(cand_obs_time) is bool or not (
                isinstance(cand_obs_time, (int, float))
                and math.isfinite(cand_obs_time)
                and cand_obs_time >= 0.0
            ):
                raise ValueError(f"observation_time must be finite non-negative float, got {cand_obs_time}")

            if self._last_observation_time is not None and cand_obs_time < self._last_observation_time:
                raise ValueError(
                    f"observation_time must be non-decreasing: got {cand_obs_time} < last {self._last_observation_time}"
                )

            gen = self._generation

        copied_obs = _copy_observation(observation)

        with self._cv:
            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")
            if self._generation != gen:
                return False
            if self._vision_error is not None:
                raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error
            if copied_obs.frame_id <= self._last_submitted_id:
                raise ValueError(
                    f"frame_id {copied_obs.frame_id} must be strictly greater than last submitted {self._last_submitted_id}"
                )

            # Re-check capacity if error policy
            if (
                self.max_pending is not None
                and self.overflow_policy == "error"
                and len(self._pending_raw) >= self.max_pending
            ):
                self._rejected_pending += 1
                self._events.append(
                    {
                        "type": "rejected_pending",
                        "frame_id": copied_obs.frame_id,
                        "generation": gen,
                        "timestamp": time.monotonic(),
                    }
                )
                raise BufferError(
                    f"Pending queue full (capacity {self.max_pending}) with overflow_policy='error'"
                )

            # Commit time source, origin, and last_observation_time
            if self._time_source is None:
                self._time_source = "provided" if is_provided else "capture_relative"
            else:
                if is_provided and self._time_source != "provided":
                    raise ValueError(
                        f"Cannot mix observation_time sources: episode initialized as {self._time_source}, got provided observation_time"
                    )
                if not is_provided and self._time_source != "capture_relative":
                    raise ValueError(
                        f"Cannot mix observation_time sources: episode initialized as {self._time_source}, got None observation_time"
                    )

            if not is_provided and self._episode_capture_origin is None:
                self._episode_capture_origin = copied_obs.capture_time

            assigned_obs_time = (
                copied_obs.observation_time
                if is_provided
                else copied_obs.capture_time - self._episode_capture_origin  # type: ignore[operator]
            )

            if type(assigned_obs_time) is bool or not (
                isinstance(assigned_obs_time, (int, float))
                and math.isfinite(assigned_obs_time)
                and assigned_obs_time >= 0.0
            ):
                raise ValueError(f"observation_time must be finite non-negative float, got {assigned_obs_time}")

            if (
                self._last_observation_time is not None
                and assigned_obs_time < self._last_observation_time
            ):
                raise ValueError(
                    f"observation_time must be non-decreasing: got {assigned_obs_time} < last {self._last_observation_time}"
                )

            final_obs = dataclasses.replace(copied_obs, observation_time=assigned_obs_time)
            self._last_submitted_id = final_obs.frame_id
            self._last_observation_time = assigned_obs_time

            if self.max_pending is not None and len(self._pending_raw) >= self.max_pending:
                self._pending_raw.popleft()
                self._dropped_pending += 1

            self._pending_raw.append((final_obs, gen))
            self._events.append(
                {
                    "type": "submit",
                    "frame_id": final_obs.frame_id,
                    "generation": gen,
                    "observation_time": assigned_obs_time,
                    "timestamp": time.monotonic(),
                }
            )
            self._cv.notify_all()
            return True

    def request_plan(self, retry: bool = False) -> bool:
        if self.validate_fn is not None:
            self.validate_fn()

        with self._cv:
            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")
            if self._episode_id is None:
                raise RuntimeError("AsyncVisualPlanner has not been reset for an episode")
            if self._vision_error is not None:
                raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error

            if retry:
                if self._reservation is None:
                    return False
                if self._planner_task is not None or self._planner_busy or self._completed is not None:
                    return False
                self._planner_error = None
                snapshot = self._reservation
            else:
                if self._planner_error is not None:
                    raise RuntimeError(f"Planner error pending retry: {self._planner_error}") from self._planner_error
                if self._planner_task is not None or self._planner_busy or self._completed is not None:
                    return False
                if len(self._ready) == 0:
                    return False
                ready_list = list(self._ready)
                if self.max_frames is not None:
                    if len(ready_list) > self.max_frames:
                        snapshot = tuple(ready_list[-self.max_frames :])
                    else:
                        snapshot = tuple(ready_list)
                else:
                    # Dynamic mode: snapshot all ready frames
                    snapshot = tuple(ready_list)
                self._ready.clear()
                self._reservation = snapshot

            gen = self._generation
            prompt = self._prompt
            assert prompt is not None
            current_memory = self._memory

            self._planner_task = (gen, prompt, snapshot, current_memory)
            self._events.append(
                {
                    "type": "request_plan",
                    "generation": gen,
                    "retry": retry,
                    "snapshot_frame_ids": [f.observation.frame_id for f in snapshot],
                    "snapshot_observation_times": [f.observation.observation_time for f in snapshot],
                    "timestamp": time.monotonic(),
                }
            )
            self._cv.notify_all()
            return True

    def poll_plan(self) -> Optional[PlanResult]:
        if self.validate_fn is not None:
            self.validate_fn()

        with self._cv:
            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")
            if self._vision_error is not None:
                raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error
            if self._planner_error is not None:
                err = self._planner_error
                raise RuntimeError(f"Planner worker failed: {err}") from err
            res = self._completed
            self._completed = None
            return res

    def wait_ready(self, min_frames: int = 1, timeout: float = 30.0) -> bool:
        if type(min_frames) is not int or min_frames <= 0:
            raise ValueError(f"min_frames must be positive int, got {min_frames}")
        if self.effective_max_ready is not None and min_frames > self.effective_max_ready:
            raise ValueError(
                f"min_frames ({min_frames}) exceeds effective ready capacity ({self.effective_max_ready})"
            )
        if type(timeout) is bool or not (isinstance(timeout, (int, float)) and math.isfinite(timeout) and timeout >= 0.0):
            raise ValueError(f"timeout must be finite non-negative float, got {timeout}")

        if self.validate_fn is not None:
            self.validate_fn()

        deadline = time.monotonic() + timeout
        with self._cv:
            while len(self._ready) < min_frames:
                if self._closed:
                    raise RuntimeError("AsyncVisualPlanner is closed")
                if self._vision_error is not None:
                    raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False
                self._cv.wait(timeout=rem)

            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")
            if self._vision_error is not None:
                raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error
            return True

    def wait_plan(self, timeout: float = 30.0) -> Optional[PlanResult]:
        if type(timeout) is bool or not (isinstance(timeout, (int, float)) and math.isfinite(timeout) and timeout >= 0.0):
            raise ValueError(f"timeout must be finite non-negative float, got {timeout}")

        if self.validate_fn is not None:
            self.validate_fn()

        deadline = time.monotonic() + timeout
        with self._cv:
            while self._completed is None:
                if self._closed:
                    raise RuntimeError("AsyncVisualPlanner is closed")
                if self._vision_error is not None:
                    raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error
                if self._planner_error is not None:
                    err = self._planner_error
                    raise RuntimeError(f"Planner worker failed: {err}") from err
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return None
                self._cv.wait(timeout=rem)

            if self._closed:
                raise RuntimeError("AsyncVisualPlanner is closed")
            if self._vision_error is not None:
                raise RuntimeError(f"Vision worker fatal error: {self._vision_error}") from self._vision_error
            if self._planner_error is not None:
                err = self._planner_error
                raise RuntimeError(f"Planner worker failed: {err}") from err

            res = self._completed
            self._completed = None
            return res

    def stats(self) -> Dict[str, Any]:
        with self._cv:
            inflight = []
            if self._vision_inflight_id is not None:
                inflight.append(self._vision_inflight_id)
            res_ids = [f.observation.frame_id for f in self._reservation] if self._reservation is not None else []

            mem_frame_count = getattr(self._memory, "frame_count", 0)
            mem_last_id = getattr(self._memory, "last_frame_id", -1)
            mem_bytes = getattr(self._memory, "nbytes", 0)
            if self._frame_session is not None:
                session_lock = getattr(self._frame_session, "_lock", None)
                if session_lock is None:
                    session_frames = tuple(getattr(self._frame_session, "frames", ()))
                else:
                    with session_lock:
                        session_frames = tuple(getattr(self._frame_session, "frames", ()))
                mem_frame_count = len(session_frames)
                mem_last_id = session_frames[-1].frame_id if session_frames else -1
                mem_bytes = sum(
                    int(t.numel() * t.element_size())
                    for f in session_frames
                    for t in (*f.keys, *f.values)
                )

            return {
                "generation": self._generation,
                "episode_id": self._episode_id,
                "last_submitted_id": self._last_submitted_id,
                "last_encoded_id": self._last_encoded_id,
                "last_consumed_id": self._last_consumed_id,
                "ready_frame_ids": [f.observation.frame_id for f in self._ready],
                "pending_count": len(self._pending_raw),
                "inflight_frame_ids": inflight,
                "planner_inflight_frame_ids": list(self._planner_inflight_ids),
                "vision_busy": self._vision_busy,
                "planner_busy": self._planner_busy,
                "plan_pending": self._planner_task is not None,
                "result_ready": self._completed is not None,
                "reservation_frame_ids": res_ids,
                "encoded_count": self._encoded_count,
                "consumed_count": self._consumed_count,
                "dropped_pending": self._dropped_pending,
                "dropped_ready": self._dropped_ready,
                "rejected_pending": self._rejected_pending,
                "plans_completed": self._plans_completed,
                "stale_outputs_discarded": self._stale_outputs_discarded,
                "errors_count": self._errors_count,
                "memory_frame_count": mem_frame_count,
                "last_memory_frame_id": mem_last_id,
                "memory_bytes": mem_bytes,
                "max_ready": self.effective_max_ready,
                "max_pending": self.max_pending,
                "snapshot_mode": "latest_bounded" if self.max_frames is not None else "all_ready",
                "overflow_policy": self.overflow_policy,
                "time_source": self._time_source,
                "episode_capture_origin": self._episode_capture_origin,
                "last_observation_time": self._last_observation_time,
                "events": list(self._events),
            }

    def close(self, timeout: float = 30.0) -> None:
        with self._cv:
            self._closed = True
            self._memory = None
            if self._frame_session is not None and hasattr(self._frame_session, "clear"):
                session_lock = getattr(self._frame_session, "_lock", None)
                if session_lock is None:
                    self._frame_session.clear()
                else:
                    with session_lock:
                        self._frame_session.clear()
            self._pending_raw.clear()
            self._ready.clear()
            self._reservation = None
            self._completed = None
            self._vision_error = None
            self._planner_error = None
            self._planner_task = None
            self._cv.notify_all()

        self._vision_thread.join(timeout=timeout)
        self._planner_thread.join(timeout=timeout)

        if self._vision_thread.is_alive():
            raise RuntimeError("Vision worker thread failed to exit within timeout")
        if self._planner_thread.is_alive():
            raise RuntimeError("Planner worker thread failed to exit within timeout")

    def _vision_worker_loop(self) -> None:
        th_id = threading.get_ident()
        while True:
            with self._cv:
                while not self._closed and (len(self._pending_raw) == 0 or self._vision_error is not None):
                    self._cv.wait()
                if self._closed:
                    return
                obs, gen = self._pending_raw.popleft()
                self._vision_busy = True
                self._vision_inflight_id = obs.frame_id

            t0 = time.monotonic()
            encoded_frame: Optional[EncodedFrame] = None
            exc: Optional[BaseException] = None
            try:
                if self.validate_fn is not None:
                    self.validate_fn()
                with torch.no_grad():
                    payload = self.encode_fn(obs)
                if self.validate_fn is not None:
                    self.validate_fn()
                t1 = time.monotonic()
                encoded_frame = EncodedFrame(
                    observation=obs,
                    payload=payload,
                    encode_started=t0,
                    encode_finished=t1,
                )
            except BaseException as e:
                exc = e
            finally:
                obs = None
                payload = None

            with self._cv:
                self._vision_busy = False
                self._vision_inflight_id = None

                if self._closed:
                    encoded_frame = None
                    return

                if exc is not None:
                    if gen == self._generation:
                        self._vision_error = exc
                        self._errors_count += 1
                        self._events.append(
                            {
                                "type": "vision_error",
                                "thread_id": th_id,
                                "frame_id": encoded_frame.observation.frame_id if encoded_frame else None,
                                "generation": gen,
                                "error": str(exc),
                                "timestamp": time.monotonic(),
                            }
                        )
                        self._cv.notify_all()
                    encoded_frame = None
                    continue

                assert encoded_frame is not None
                if gen != self._generation:
                    self._stale_outputs_discarded += 1
                    encoded_frame = None
                    self._cv.notify_all()
                    continue

                if self.frame_commit_hook is not None:
                    try:
                        # Commit exactly once, after generation fencing and
                        # before publishing the frame as ready.  A retry then
                        # reuses the reservation instead of appending again.
                        with torch.no_grad():
                            self.frame_commit_hook(encoded_frame)
                    except BaseException as hook_exc:
                        self._vision_error = hook_exc
                        self._errors_count += 1
                        self._events.append(
                            {
                                "type": "vision_commit_error",
                                "thread_id": th_id,
                                "frame_id": encoded_frame.observation.frame_id,
                                "generation": gen,
                                "error": str(hook_exc),
                                "timestamp": time.monotonic(),
                            }
                        )
                        encoded_frame = None
                        self._cv.notify_all()
                        continue

                self._last_encoded_id = encoded_frame.observation.frame_id
                self._encoded_count += 1

                if self.effective_max_ready is not None and len(self._ready) >= self.effective_max_ready:
                    if self.overflow_policy == "error":
                        err = BufferError(
                            f"Ready queue full (capacity {self.effective_max_ready}) with overflow_policy='error'"
                        )
                        self._vision_error = err
                        self._errors_count += 1
                        self._events.append(
                            {
                                "type": "vision_error",
                                "thread_id": th_id,
                                "frame_id": encoded_frame.observation.frame_id,
                                "generation": gen,
                                "error": str(err),
                                "timestamp": time.monotonic(),
                            }
                        )
                        encoded_frame = None
                        self._cv.notify_all()
                        continue
                    else:
                        self._ready.popleft()
                        self._dropped_ready += 1

                self._ready.append(encoded_frame)
                self._events.append(
                    {
                        "type": "encoded",
                        "thread_id": th_id,
                        "frame_id": encoded_frame.observation.frame_id,
                        "generation": gen,
                        "start": t0,
                        "finish": encoded_frame.encode_finished,
                        "observation_time": encoded_frame.observation.observation_time,
                    }
                )
                encoded_frame = None
                self._cv.notify_all()

    def _planner_worker_loop(self) -> None:
        th_id = threading.get_ident()
        while True:
            with self._cv:
                while not self._closed and self._planner_task is None:
                    self._cv.wait()
                if self._closed:
                    return
                gen, prompt, snapshot, current_mem = self._planner_task
                self._planner_task = None
                self._planner_busy = True
                self._planner_inflight_ids = [f.observation.frame_id for f in snapshot]

            t0 = time.monotonic()
            plan_comp: Optional[PlanComputation] = None
            prepared_memory: Optional[Any] = None
            exc: Optional[BaseException] = None
            try:
                if self.validate_fn is not None:
                    self.validate_fn()
                with torch.no_grad():
                    if self.stateful:
                        plan_comp = self.plan_fn(snapshot, prompt, current_mem)
                    else:
                        plan_comp = self.plan_fn(snapshot, prompt)
                if not isinstance(plan_comp, PlanComputation):
                    raise TypeError(f"plan callback must return PlanComputation, got {type(plan_comp).__name__}")
                plan_comp.__post_init__()

                # In stateful mode, candidate next_memory is prepared via custom validator or default Delta validator
                if self.stateful:
                    if self._frame_session is not None and plan_comp.next_memory is None:
                        # Consume sessions are committed at encode time and do
                        # not return a DeltaMemoryState candidate.
                        prepared_memory = current_mem
                    else:
                        validator = self.memory_validator or _validate_next_memory
                        prepared_memory = validator(
                            plan_comp.next_memory,
                            current_mem,
                            snapshot,
                            prompt,
                        )

                if self.validate_fn is not None:
                    self.validate_fn()
            except BaseException as e:
                exc = e
            t1 = time.monotonic()

            with self._cv:
                self._planner_busy = False
                self._planner_inflight_ids = []

                if self._closed:
                    snapshot = None
                    plan_comp = None
                    prepared_memory = None
                    current_mem = None
                    return

                if exc is not None:
                    if gen == self._generation:
                        self._planner_error = exc
                        self._errors_count += 1
                        self._events.append(
                            {
                                "type": "planner_error",
                                "thread_id": th_id,
                                "generation": gen,
                                "planning_frame_ids": [f.observation.frame_id for f in snapshot],
                                "error": str(exc),
                                "timestamp": t1,
                            }
                        )
                        self._cv.notify_all()
                    snapshot = None
                    plan_comp = None
                    prepared_memory = None
                    current_mem = None
                    continue

                assert plan_comp is not None
                if gen != self._generation:
                    self._stale_outputs_discarded += 1
                    snapshot = None
                    plan_comp = None
                    prepared_memory = None
                    current_mem = None
                    self._cv.notify_all()
                    continue

                source_frame = snapshot[-1]
                source_frame_id = source_frame.observation.frame_id
                capture_time = source_frame.observation.capture_time

                consumed_frames = tuple(
                    ConsumedFrame(
                        observation=ObservationMetadata(
                            frame_id=f.observation.frame_id,
                            capture_time=f.observation.capture_time,
                            observation_time=f.observation.observation_time,
                        ),
                        encode_started=f.encode_started,
                        encode_finished=f.encode_finished,
                    )
                    for f in snapshot
                )

                # Strip next_memory from public PlanComputation in PlanResult
                public_comp = PlanComputation(
                    actions=plan_comp.actions,
                    deep=plan_comp.deep,
                    shallow=plan_comp.shallow,
                    next_memory=None,
                )

                result = PlanResult(
                    episode_id=self._episode_id or "",
                    generation=gen,
                    frames=consumed_frames,
                    source_frame_id=source_frame_id,
                    capture_time=capture_time,
                    started=t0,
                    finished=t1,
                    computation=public_comp,
                )

                # Commit next_memory only after public_comp and result construction succeed
                if self.stateful:
                    self._memory = prepared_memory

                self._last_consumed_id = max(self._last_consumed_id, source_frame_id)
                self._consumed_count += len(snapshot)
                self._plans_completed += 1
                self._completed = result
                self._reservation = None

                self._events.append(
                    {
                        "type": "planned",
                        "thread_id": th_id,
                        "generation": gen,
                        "planning_frame_ids": [f.observation.frame_id for f in snapshot],
                        "snapshot_observation_times": [f.observation.observation_time for f in snapshot],
                        "source_frame_id": source_frame_id,
                        "start": t0,
                        "finish": t1,
                    }
                )
                source_frame = None
                snapshot = None
                plan_comp = None
                prepared_memory = None
                public_comp = None
                current_mem = None
                result = None
                self._cv.notify_all()



def make_moss_callbacks(
    model: MossInternVL,
) -> Tuple[
    Callable[[Observation], Any],
    Callable[..., PlanComputation],
    Callable[[], None],
]:
    initial_revision = model._revision

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    use_cuda = (device.type == "cuda") and torch.cuda.is_available()

    if use_cuda:
        vision_stream = torch.cuda.Stream(device=device)
        planner_stream = torch.cuda.Stream(device=device)
        init_event = torch.cuda.Event()
        init_event.record(torch.cuda.current_stream(device=device))
        vision_stream.wait_event(init_event)
        planner_stream.wait_event(init_event)
    else:
        vision_stream = None
        planner_stream = None

    memory_mode = getattr(model.config, "memory_mode", "consume")
    consume_session = FrameKVSession(model) if memory_mode == "consume" else None
    consume_lock = threading.RLock()

    def reset_session(episode_id: str, prompt: str) -> None:
        if consume_session is not None:
            with consume_lock:
                consume_session.reset(episode_id, prompt)

    def commit_frame(encoded: EncodedFrame) -> None:
        if consume_session is None:
            return
        if not isinstance(encoded.payload, FrameKV):
            raise TypeError(
                f"consume callback payload must be FrameKV, got {type(encoded.payload).__name__}"
            )
        with consume_lock:
            consume_session.append_frame(encoded.payload)

    def consume_snapshot(frames: Tuple[EncodedFrame, ...], prompt: str) -> Tuple[FrameKV, ...]:
        if consume_session is None:
            return tuple(f.payload for f in frames)
        cutoff = frames[-1].observation.frame_id
        with consume_lock:
            direct_mode = False
            if consume_session.episode_id is None:
                # Direct callback callers do not have an AsyncVisualPlanner to
                # invoke the lifecycle hook; initialize a private episode.
                consume_session.reset("direct", prompt)
                direct_mode = True
            elif consume_session.prompt != prompt.strip():
                raise ValueError(
                    f"consume session prompt {consume_session.prompt!r} does not match active prompt {prompt.strip()!r}"
                )
            # A direct invocation may bypass the commit hook.  Append only
            # unseen payloads, preserving the exactly-once invariant.
            known_ids = {f.frame_id for f in consume_session.frames}
            for encoded in frames:
                if not isinstance(encoded.payload, FrameKV):
                    raise TypeError(
                        f"consume callback payload must be FrameKV, got {type(encoded.payload).__name__}"
                    )
                if encoded.payload.frame_id not in known_ids:
                    if not direct_mode:
                        raise RuntimeError(
                            f"frame_id={encoded.payload.frame_id} was not committed to the active consume session"
                        )
                    consume_session.append_frame(encoded.payload)
                    known_ids.add(encoded.payload.frame_id)
            history = consume_session.snapshot(cutoff)
        if not history or history[-1].frame_id != cutoff:
            raise RuntimeError(
                f"consume session has no complete history through frame_id={cutoff}"
            )
        return history

    def validate() -> None:
        if model.training:
            raise RuntimeError("MossInternVL model must be in eval mode for AsyncVisualPlanner")
        if model._revision != initial_revision:
            raise RuntimeError(
                f"Model revision changed from {initial_revision} to {model._revision}; pipeline invalidated"
            )

    def encode(obs: Observation) -> FrameKV:
        validate()
        with torch.no_grad():
            if use_cuda and vision_stream is not None:
                with torch.cuda.stream(vision_stream):
                    features = model.encode_image(list(obs.images))
                    frame_kv = model.project_frame(
                        features, frame_id=obs.frame_id,
                        # Delta's historical contract never included the
                        # optional metadata token; retain it for consume while
                        # still recording the observation time on the frame.
                        observation_time=obs.observation_time if memory_mode == "consume" else None,
                    )
                    if memory_mode == "delta" and obs.observation_time is not None:
                        frame_kv = dataclasses.replace(frame_kv, observation_time=obs.observation_time)
                    completion_event = torch.cuda.Event()
                    completion_event.record(vision_stream)
                    completion_event.synchronize()
            else:
                features = model.encode_image(list(obs.images))
                frame_kv = model.project_frame(
                    features, frame_id=obs.frame_id,
                    observation_time=obs.observation_time if memory_mode == "consume" else None,
                )
                if memory_mode == "delta" and obs.observation_time is not None:
                    frame_kv = dataclasses.replace(frame_kv, observation_time=obs.observation_time)
        return frame_kv

    def plan(
        frames: Tuple[EncodedFrame, ...],
        prompt: str,
        previous_memory: Optional[Any] = None,
    ) -> PlanComputation:
        validate()
        latest_obs = frames[-1].observation
        if memory_mode == "consume":
            read_frames = consume_snapshot(frames, prompt)
            payloads = list(read_frames)
        else:
            payloads = [f.payload for f in frames]
        next_mem_candidate: Optional[Any] = None

        with torch.no_grad():
            if use_cuda and planner_stream is not None:
                with torch.cuda.stream(planner_stream):
                    for p in payloads:
                        if isinstance(p, FrameKV):
                            for k in p.keys:
                                k.record_stream(planner_stream)
                            for v in p.values:
                                v.record_stream(planner_stream)

                    if previous_memory is not None and hasattr(previous_memory, "matrices"):
                        for mat in previous_memory.matrices:
                            if isinstance(mat, torch.Tensor):
                                mat.record_stream(planner_stream)

                    state_dev = latest_obs.state.to(device)
                    state_mask_dev = latest_obs.state_mask.to(device)
                    action_mask_dev = latest_obs.action_mask.to(device)

                    if memory_mode == "delta":
                        deep, shallow, cand_state = model.read_delta(
                            payloads, prompt, previous=previous_memory
                        )
                        next_mem_candidate = cand_state.detached()
                    else:
                        deep, shallow = model.read_memory(
                            payloads,
                            prompt,
                            frame_ids=[f.frame_id for f in payloads],
                            observation_times=[
                                float(f.observation_time)
                                if f.observation_time is not None
                                else float(f.frame_id)
                                for f in payloads
                            ],
                        )
                        next_mem_candidate = None

                    actions = model.policy.action_head.sample(
                        deep,
                        state=state_dev,
                        state_mask=state_mask_dev,
                        action_mask=action_mask_dev,
                        shallow_tokens=shallow,
                    )
                    planner_stream.synchronize()
                    actions_cpu = actions.detach().cpu()
                    deep_cpu = deep.detach().cpu() if deep is not None else None
                    shallow_cpu = shallow.detach().cpu() if shallow is not None else None
            else:
                state_cpu = latest_obs.state.detach().clone()
                state_mask_cpu = latest_obs.state_mask.detach().clone()
                action_mask_cpu = latest_obs.action_mask.detach().clone()

                if memory_mode == "delta":
                    deep, shallow, cand_state = model.read_delta(
                        payloads, prompt, previous=previous_memory
                    )
                    next_mem_candidate = cand_state.detached()
                else:
                    deep, shallow = model.read_memory(
                        payloads,
                        prompt,
                        frame_ids=[f.frame_id for f in payloads],
                        observation_times=[
                            float(f.observation_time)
                            if f.observation_time is not None
                            else float(f.frame_id)
                            for f in payloads
                        ],
                    )
                    next_mem_candidate = None

                actions = model.policy.action_head.sample(
                    deep,
                    state=state_cpu,
                    state_mask=state_mask_cpu,
                    action_mask=action_mask_cpu,
                    shallow_tokens=shallow,
                )
                actions_cpu = actions.detach().cpu()
                deep_cpu = deep.detach().cpu() if deep is not None else None
                shallow_cpu = shallow.detach().cpu() if shallow is not None else None

        return PlanComputation(
            actions=actions_cpu,
            deep=deep_cpu,
            shallow=shallow_cpu,
            next_memory=next_mem_candidate,
        )

    # AsyncVisualPlanner discovers these optional hooks while retaining the
    # historical three-callback return shape.
    if consume_session is not None:
        plan.reset_session = reset_session  # type: ignore[attr-defined]
        plan.reset = reset_session  # type: ignore[attr-defined]
        plan.commit_frame = commit_frame  # type: ignore[attr-defined]
        plan.append_frame = commit_frame  # type: ignore[attr-defined]
        plan.frame_session = consume_session  # type: ignore[attr-defined]

    return encode, plan, validate
