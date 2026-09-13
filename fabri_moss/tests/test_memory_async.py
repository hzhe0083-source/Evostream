"""Integration and pipeline tests for real asynchronous periodic memory cache."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import pytest
import torch

from fabri_moss.async_pipeline import AsyncVisualPlanner, Observation
from fabri_moss.evaluate_async import build_parser, main
from fabri_moss.memory_cache import NativeMemoryCacheAdapter, validate_memory_cache
from fabri_moss.native_async import make_native_cache_callbacks
from fabri_moss.native_cache import NativeCacheConfig
from fabri_moss.periodic_memory import PeriodicMemoryConfig
from fabri_moss.tests.test_native_training import make_tiny_training_policy


def make_tiny_adapter() -> NativeMemoryCacheAdapter:
    seq = make_tiny_training_policy(shallow_layer=1)
    policy = seq.policy
    embedder = policy.embedder
    embedder._preprocess_images = lambda imgs: (embedder._preprocess_images_on_cpu(imgs)[0], [1])
    def _prepare_and_fuse(prompt: str, vit_embeds: torch.Tensor, image_mask: torch.Tensor, num_tiles_list: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return embedder._prepare_batch_and_fuse_embeddings([prompt], [vit_embeds], [image_mask], [num_tiles_list])
    embedder._prepare_and_fuse_embeddings = _prepare_and_fuse
    orig_tok = embedder.tokenizer
    embedder.tokenizer = lambda text, return_tensors="pt", add_special_tokens=False: orig_tok(
        text, return_tensors=return_tensors
    )
    c_cfg = NativeCacheConfig(shallow_layer=1, use_timestamps=True)
    m_cfg = PeriodicMemoryConfig(recent_frames=2, consolidate_every=2, memory_slots=2, spatial_grid=1)
    adapter = NativeMemoryCacheAdapter(policy, config=c_cfg, memory_config=m_cfg)
    adapter.eval()
    return adapter


def make_obs(frame_id: int, state_val: float = 0.0) -> Observation:
    return Observation(
        frame_id=frame_id,
        capture_time=float(frame_id),
        images=[torch.randn(4, 16)],
        state=torch.full((1, 4), state_val),
        state_mask=torch.ones((1, 4), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
        observation_time=float(frame_id) * 0.1,
    )


def test_1_stream_20_frames_and_head_sample_spy() -> None:
    adapter = make_tiny_adapter()
    captured: dict[str, Any] = {}
    orig_sample = adapter.policy.action_head.sample

    def spy_sample(*args: Any, **kwargs: Any) -> torch.Tensor:
        captured["state"] = kwargs.get("state")
        return orig_sample(*args, **kwargs)

    adapter.policy.action_head.sample = spy_sample
    prompt = "reach goal"
    encode_fn, plan_fn, val_fn = make_native_cache_callbacks(adapter, prompt)
    planner = AsyncVisualPlanner(
        encode=encode_fn, plan=plan_fn, validate=val_fn, max_frames=None,
        stateful=True, memory_validator=validate_memory_cache, overflow_policy="error",
    )
    planner.reset("ep1", prompt)

    for i in range(20):
        planner.submit(make_obs(i, state_val=float(i)))

    assert planner.wait_ready(min_frames=20, timeout=10.0)
    assert planner.request_plan()
    plan_res = planner.wait_plan(timeout=10.0)
    assert plan_res is not None

    # Assert spy captured state == 19
    assert captured["state"] is not None
    assert torch.allclose(captured["state"], torch.full((1, 4), 19.0))

    # Assert validator passed and state invariants hold
    mem_state = planner._memory
    assert mem_state is not None
    assert mem_state.frame_count == 20
    assert len(mem_state.memory.entries) == 1
    assert mem_state.memory.recent[-1].is_decision
    assert not mem_state.memory.anchors
    assert mem_state.memory.merges > 0
    assert mem_state.memory.recent[-1].frame_id == 19
    assert plan_res.computation.deep.shape == (1, 16, 64)
    assert planner.stats()["consumed_count"] == 20
    planner.close()


def test_2_failure_isolation_and_retry() -> None:
    adapter = make_tiny_adapter()
    prompt = "retry task"
    encode_fn, raw_plan, val_fn = make_native_cache_callbacks(adapter, prompt)

    fail_flag = [False]

    def flaky_plan(*args: Any, **kwargs: Any) -> Any:
        if fail_flag[0]:
            fail_flag[0] = False
            raise RuntimeError("simulated planner failure")
        return raw_plan(*args, **kwargs)

    planner = AsyncVisualPlanner(
        encode=encode_fn, plan=flaky_plan, validate=val_fn, max_frames=None,
        stateful=True, memory_validator=validate_memory_cache, overflow_policy="error",
    )
    planner.reset("ep1", prompt)

    # Commit 8 frames
    for i in range(8):
        planner.submit(make_obs(i))
    assert planner.wait_ready(min_frames=8, timeout=10.0)
    assert planner.request_plan()
    plan1 = planner.wait_plan(timeout=10.0)
    assert plan1 is not None
    mem8 = planner._memory
    assert mem8 is not None and mem8.frame_count == 8
    cached_kv = mem8.layer_kv[0][0].clone()

    # Submit 4 more frames and trigger planner error
    for i in range(8, 12):
        planner.submit(make_obs(i))
    assert planner.wait_ready(min_frames=4, timeout=10.0)
    fail_flag[0] = True
    assert planner.request_plan()

    with pytest.raises(RuntimeError, match="simulated planner failure"):
        planner.wait_plan(timeout=10.0)

    # Invariants: memory unchanged
    assert planner._memory is mem8
    assert planner._memory.frame_count == 8
    assert torch.equal(planner._memory.layer_kv[0][0], cached_kv)

    # Retry same ready reservation
    assert planner.request_plan(retry=True)
    plan2 = planner.wait_plan(timeout=10.0)
    assert plan2 is not None
    assert planner._memory.frame_count == 12
    assert [a.frame_id for a in planner._memory.memory.anchors] == [7]
    assert planner._memory.memory.recent[-1].frame_id == 11
    assert planner._memory.memory.recent[-1].is_decision
    assert planner.stats()["consumed_count"] == 12
    planner.close()


def test_3_episode_reset_and_cli_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = make_tiny_adapter()
    prompt = "reset task"
    encode_fn, plan_fn, val_fn = make_native_cache_callbacks(adapter, prompt)
    planner = AsyncVisualPlanner(
        encode=encode_fn, plan=plan_fn, validate=val_fn, max_frames=None,
        stateful=True, memory_validator=validate_memory_cache, overflow_policy="error",
    )

    planner.reset("ep1", prompt)
    planner.submit(make_obs(0))
    assert planner.wait_ready(min_frames=1, timeout=10.0)
    assert planner.request_plan()
    assert planner.wait_plan(timeout=10.0) is not None
    assert planner._memory is not None

    # Reset clears memory
    planner.reset("ep2", prompt)
    assert planner._memory is None
    assert planner.stats()["consumed_count"] == 0

    # New single frame episode
    planner.submit(make_obs(0))
    assert planner.wait_ready(min_frames=1, timeout=10.0)
    assert planner.request_plan()
    assert planner.wait_plan(timeout=10.0) is not None

    mem = planner._memory
    assert mem is not None
    assert mem.last_frame_id == 0
    assert mem.frame_count == 1
    assert len(mem.memory.entries) == 0
    assert len(mem.memory.recent) == 1
    planner.close()

    # CLI validation: periodic rejects non native-cache or non text timestamps
    out_dir = tmp_path / "cli_out"
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_async", "--mode", "moss", "--adapter", "a.pt", "--output-dir", str(out_dir), "--native-memory", "periodic"],
    )
    with pytest.raises(ValueError, match="only allowed when --mode=native-cache"):
        main()

    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_async", "--mode", "native-cache", "--output-dir", str(out_dir), "--timestamp-mode", "none", "--native-memory", "periodic"],
    )
    with pytest.raises(ValueError, match="requires --timestamp-mode=text"):
        main()

    # Valid CLI parses properly
    parser = build_parser()
    args = parser.parse_args(
        ["--mode", "native-cache", "--output-dir", str(out_dir), "--timestamp-mode", "text", "--native-memory", "periodic"]
    )
    assert args.mode == "native-cache" and args.timestamp_mode == "text" and args.native_memory == "periodic"
