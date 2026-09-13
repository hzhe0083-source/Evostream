"""Regression and parity tests for compact cache adapter and dataset protocol."""

from __future__ import annotations

import concurrent.futures
from dataclasses import replace
from pathlib import Path
import threading
from typing import Callable, Tuple

from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from fabri_moss.compact_cache import (
    CompactRebuildPayload,
    NativeCompactMemoryCacheAdapter,
    validate_compact_memory,
)
from fabri_moss.native_data import NativeTrainingDataset
import fabri_moss.native_data as nd
from fabri_moss.periodic_memory import PeriodicMemoryConfig
from fabri_moss.tests.test_compact_cache import make_tiny_compact_adapter
from fabri_moss.tests.test_native_data import _create_mock_metaworld_root


def test_stale_future_cancellation_and_exception_recovery():
    adapter = make_tiny_compact_adapter()
    prompt = "regression_stale_future"
    event_start = threading.Event()
    event_done = threading.Event()
    th = None

    try:
        b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
        b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
        _, _, s0 = adapter.read_blocks([b0], prompt=prompt)
        assert s0.pending is None

        # 1. Stale running future fails in a second thread
        fut_running = concurrent.futures.Future()
        fut_running.set_running_or_notify_cancel()

        def worker():
            event_start.wait(timeout=5.0)
            fut_running.set_exception(RuntimeError("worker failed"))
            event_done.set()

        th = threading.Thread(target=worker)
        th.start()

        with adapter._lock:
            adapter._active_future = fut_running

        event_start.set()
        with pytest.raises(RuntimeError, match="worker failed"):
            adapter.read_blocks([b1], prompt=prompt, previous=s0)

        assert event_done.wait(timeout=5.0)
        with adapter._lock:
            assert adapter._active_future is None

        # Retry with the same snapshot succeeds and only advances frame count
        _, _, s1 = adapter.read_blocks([b1], prompt=prompt, previous=s0)
        assert s1.frame_count == s0.frame_count + 1
        assert s1.rebuild_count == s0.rebuild_count

        # 2. Stale cancelled future (cancel() == True) should not raise CancelledError
        fut_cancel = concurrent.futures.Future()
        with adapter._lock:
            adapter._active_future = fut_cancel

        b2 = adapter.encode_frame(images=["img2"], frame_id=2, prompt=prompt, observation_time=0.2)
        _, _, s2 = adapter.read_blocks([b2], prompt=prompt, previous=s1)
        assert fut_cancel.cancelled()
        with adapter._lock:
            assert adapter._active_future is None
        assert s2.frame_count == s1.frame_count + 1
    finally:
        event_start.set()
        if th is not None and th.is_alive():
            th.join(timeout=5.0)
        adapter.close()


def test_single_frame_validator_closed_adapter_and_shallow_layer():
    adapter = make_tiny_compact_adapter(shallow_layer=3)
    prompt = "regression_adapter_state"
    try:
        b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt, observation_time=0.0)
        b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
        deep0, shallow0, s0 = adapter.read_blocks([b0], prompt=prompt)

        # Single frame validator returns the exact state instance
        assert validate_compact_memory(s0, None, [b0], prompt=prompt) is s0
        # shallow_layer == 3 matches len(tiny layers), so normalized shallow == deep
        assert torch.equal(deep0, shallow0)

        # Config change: old state with no pending is rejected
        s0_bad_cfg = replace(s0, config=replace(s0.config, memory=PeriodicMemoryConfig(recent_frames=99)))
        with pytest.raises(ValueError, match="config mismatch"):
            adapter.read_blocks([b1], prompt=prompt, previous=s0_bad_cfg)
        with pytest.raises(ValueError, match="config mismatch"):
            validate_compact_memory(s0_bad_cfg, None, [b0], prompt=prompt)

        # Closed adapter rejects encode and read
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt, observation_time=0.1)
        with pytest.raises(RuntimeError, match="closed"):
            adapter.read_blocks([b0], prompt=prompt)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("corrupt_fn", "exc_type"),
    [
        (lambda pl: replace(pl, rebuild_count=pl.rebuild_count + 10), ValueError),
        (lambda pl: replace(pl, memory=replace(pl.memory, frame_count=pl.memory.frame_count + 1)), ValueError),
        (lambda pl: replace(pl, attention_mask=torch.zeros_like(pl.attention_mask)), ValueError),
        (lambda pl: replace(pl, layer_kv=tuple((k.to(torch.float64), v.to(torch.float64)) for k, v in pl.layer_kv)), TypeError),
    ],
)
def test_rebuild_payload_corruption_rejection(
    corrupt_fn: Callable[[CompactRebuildPayload], CompactRebuildPayload],
    exc_type: type[Exception],
):
    adapter = make_tiny_compact_adapter(
        recent_frames=2,
        consolidate_every=2,
        background_rebuild=False,
    )
    prompt = "regression_payload_corruption"
    try:
        blocks = [
            adapter.encode_frame(images=[f"img{i}"], frame_id=i, prompt=prompt, observation_time=0.1 * i)
            for i in range(5)
        ]
        _, _, s0 = adapter.read_blocks([blocks[0]], prompt=prompt)
        _, _, s1 = adapter.read_blocks([blocks[1]], prompt=prompt, previous=s0)
        _, _, s2 = adapter.read_blocks([blocks[2]], prompt=prompt, previous=s1)
        _, _, s3 = adapter.read_blocks([blocks[3]], prompt=prompt, previous=s2)
        assert s3.pending is not None

        # Build genuine next state for validator target
        _, _, s4 = adapter.read_blocks([blocks[4]], prompt=prompt, previous=s3)

        orig_payload = s3.pending.future.result()
        bad_payload = corrupt_fn(orig_payload)

        fake_fut = concurrent.futures.Future()
        fake_fut.set_result(bad_payload)
        corrupted_s3 = replace(s3, pending=replace(s3.pending, future=fake_fut))

        with pytest.raises(exc_type):
            adapter.read_blocks([blocks[4]], prompt=prompt, previous=corrupted_s3)
        with pytest.raises(exc_type):
            validate_compact_memory(s4, corrupted_s3, [blocks[4]], prompt=prompt)
    finally:
        adapter.close()


def test_native_dataset_compact_memory_replay_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=4, lengths=[25, 40])
    monkeypatch.setattr(
        nd,
        "decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (32, 32), color=(f % 255, (f * 7) % 255, 120)) for f in fids},
    )

    mock_norm = {
        "observation.state": {"min": [-1.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }

    ds_mem = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm,
        stream_protocol="memory_replay_v1",
        split="train",
        seed=4042,
        augmentation=False,
    )
    ds_mem.set_epoch(1)

    ds_compact = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm,
        stream_protocol="compact_memory_replay_v1",
        split="train",
        seed=4042,
        augmentation=False,
    )
    ds_compact.set_epoch(1)

    assert len(ds_mem) == len(ds_compact)
    assert ds_mem.get_base_data_contract() == ds_compact.get_base_data_contract()
    assert ds_mem.get_data_contract() != ds_compact.get_data_contract()
    assert ds_mem.get_data_contract()["data_fingerprint"] != ds_compact.get_data_contract()["data_fingerprint"]

    supervised_per_ep = {}
    found_memory_replay = False

    for i in range(len(ds_compact)):
        s_mem = ds_mem[i]
        s_comp = ds_compact[i]

        assert s_mem["frame_ids"] == s_comp["frame_ids"]
        assert s_mem["observation_times"] == s_comp["observation_times"]
        assert s_mem["target_indices"] == s_comp["target_indices"]
        assert s_mem.get("decision_indices") == s_comp.get("decision_indices")
        assert torch.equal(s_mem["actions"], s_comp["actions"])
        assert torch.equal(s_mem["state"], s_comp["state"])
        assert torch.equal(s_mem["state_mask"], s_comp["state_mask"])
        assert torch.equal(s_mem["action_mask"], s_comp["action_mask"])

        if s_comp.get("memory_replay") is True:
            found_memory_replay = True

        ep_id = s_comp["episode_id"]
        t_start = s_comp["target_start_row"]
        t_end = s_comp["target_end_row"]
        supervised_per_ep.setdefault(ep_id, []).extend(range(t_start, t_end))

    assert found_memory_replay is True
    for ep in ds_compact.active_episodes:
        ep_id = ep["episode_index"]
        assert supervised_per_ep[ep_id] == list(range(ep["length"]))


def test_concurrent_fast_tokenizers_render_rebuild_isolation():
    """Verify that source tokenizer, foreground render, and background rebuild tokenizer

    instances are truly isolated, avoiding HuggingFace/Rust 'RuntimeError: Already borrowed'.
    """
    base_adapter = make_tiny_compact_adapter()
    vocab = {"<unk>": 0, "test": 1, "observation": 2, "frame": 3}
    rust_tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=rust_tok)
    fast_tok.pad_token = "<unk>"

    base_adapter.policy.embedder.tokenizer = fast_tok

    params_before = [id(p) for p in base_adapter.policy.parameters()]

    adapter = NativeCompactMemoryCacheAdapter(
        policy=base_adapter.policy,
        config=base_adapter.config,
        compact_config=base_adapter.compact_config,
        background_rebuild=base_adapter.background_rebuild,
    )

    params_after = [id(p) for p in adapter.policy.parameters()]
    assert params_before == params_after

    assert adapter._render_embedder is not None
    assert adapter._rebuild_embedder is not None
    assert adapter._render_embedder.model is base_adapter.policy.embedder.model
    assert adapter._rebuild_embedder.model is base_adapter.policy.embedder.model

    source_tok = adapter.policy.embedder.tokenizer
    render_tok = adapter._render_embedder.tokenizer
    rebuild_tok = adapter._rebuild_embedder.tokenizer

    assert source_tok._tokenizer is not render_tok._tokenizer
    assert source_tok._tokenizer is not rebuild_tok._tokenizer
    assert render_tok._tokenizer is not rebuild_tok._tokenizer

    # Precompute serial baselines for each configuration
    baseline_source = source_tok("test frame", padding="max_length", max_length=1024, truncation=True)
    baseline_render = render_tok("test frame", add_special_tokens=False)
    baseline_rebuild = rebuild_tok("observation 2", add_special_tokens=False)

    num_iters = 100
    barrier = threading.Barrier(3)
    errors: list[Tuple[str, BaseException]] = []
    source_results: list[dict] = []
    render_results: list[dict] = []
    rebuild_results: list[dict] = []

    def run_source():
        try:
            barrier.wait(timeout=10.0)
            for _ in range(num_iters):
                res = source_tok("test frame", padding="max_length", max_length=1024, truncation=True)
                source_results.append(res)
        except Exception as exc:
            errors.append(("source", exc))

    def run_render():
        try:
            barrier.wait(timeout=10.0)
            for _ in range(num_iters):
                res = render_tok("test frame", add_special_tokens=False)
                render_results.append(res)
        except Exception as exc:
            errors.append(("render", exc))

    def run_rebuild():
        try:
            barrier.wait(timeout=10.0)
            for _ in range(num_iters):
                res = rebuild_tok("observation 2", add_special_tokens=False)
                rebuild_results.append(res)
        except Exception as exc:
            errors.append(("rebuild", exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_source), pool.submit(run_render), pool.submit(run_rebuild)]
        concurrent.futures.wait(futures)

    assert errors == [], f"Encountered concurrency errors: {errors}"
    assert len(source_results) == num_iters
    assert len(render_results) == num_iters
    assert len(rebuild_results) == num_iters

    for res in source_results:
        assert res["input_ids"] == baseline_source["input_ids"]
    for res in render_results:
        assert res["input_ids"] == baseline_render["input_ids"]
    for res in rebuild_results:
        assert res["input_ids"] == baseline_rebuild["input_ids"]
