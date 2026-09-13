"""Unit tests for fabri_moss.native_async: validator and callback contracts."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from fabri_moss.async_pipeline import (
    AsyncVisualPlanner,
    EncodedFrame,
    Observation,
    PlanComputation,
)
from fabri_moss.native_async import make_native_cache_callbacks, validate_native_memory
from fabri_moss.native_cache import (
    NativeCacheAdapter,
    NativeCacheConfig,
    NativeEmbeddingBlock,
    NativeKVState,
)
from fabri_moss.tests.test_native_cache import make_tiny_native_adapter


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int = 64, num_kv_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int = 64, num_kv_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.self_attn = TinyAttention(hidden_size, num_kv_heads, head_dim)


class TinyCore(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 64, num_kv_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            TinyLayer(hidden_size, num_kv_heads, head_dim) for _ in range(num_layers)
        ])
        self.dummy_param = nn.Parameter(torch.zeros(hidden_size))


class DummyActionHead(nn.Module):
    def sample(self, deep, state, state_mask, action_mask, shallow_tokens=None):
        return torch.ones((1, 5, 4))


class TinyPolicy(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 64, num_kv_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.core = TinyCore(num_layers, hidden_size, num_kv_heads, head_dim)
        self.embedder = nn.Module()
        self.embedder.model = nn.Module()
        self.embedder.model.language_model = nn.Module()
        self.embedder.model.language_model.model = self.core
        self.action_head = DummyActionHead()


def make_tiny_adapter(max_frames: int = 4, num_layers: int = 2, head_dim: int = 32, num_kv_heads: int = 2):
    policy = TinyPolicy(num_layers=num_layers, hidden_size=64, num_kv_heads=num_kv_heads, head_dim=head_dim)
    cfg = NativeCacheConfig(max_frames=max_frames, shallow_layer=1, use_timestamps=False)
    adapter = NativeCacheAdapter(policy, config=cfg)
    adapter.eval()
    return adapter


def make_dummy_encoded_frame(frame_id: int, block: Optional[NativeEmbeddingBlock] = None) -> EncodedFrame:
    obs = Observation(
        frame_id=frame_id,
        capture_time=0.0,
        images=[torch.zeros((1, 3, 16, 16))],
        state=torch.zeros((1, 10)),
        state_mask=torch.ones((1, 10), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    return EncodedFrame(
        observation=obs,
        payload=block,
        encode_started=0.0,
        encode_finished=0.1,
    )


def test_validate_native_memory_accepts_valid_state():
    adapter = make_tiny_adapter(max_frames=4, num_layers=2, head_dim=32, num_kv_heads=2)
    seq_len = 16
    blk0 = NativeEmbeddingBlock(
        frame_id=0,
        inputs_embeds=torch.zeros((1, seq_len, 64)),
        attention_mask=torch.ones((1, seq_len), dtype=torch.bool),
        owner=adapter,
        revision=adapter.revision,
        prompt="reach goal",
    )
    blk1 = NativeEmbeddingBlock(
        frame_id=1,
        inputs_embeds=torch.zeros((1, seq_len, 64)),
        attention_mask=torch.ones((1, seq_len), dtype=torch.bool),
        owner=adapter,
        revision=adapter.revision,
        prompt="reach goal",
    )
    total_seq = seq_len * 2
    layer_kv = tuple(
        (torch.zeros((1, 2, total_seq, 32)), torch.zeros((1, 2, total_seq, 32)))
        for _ in range(2)
    )
    state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk0, blk1),
        attention_mask=torch.ones((1, total_seq), dtype=torch.bool),
        last_frame_id=1,
        frame_count=2,
        prompt="reach goal",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )

    snapshot = (make_dummy_encoded_frame(0, blk0), make_dummy_encoded_frame(1, blk1))
    validated = validate_native_memory(
        candidate=state,
        previous=None,
        snapshot=snapshot,
        prompt="reach goal",
    )
    assert validated.last_frame_id == 1
    assert validated.frame_count == 2
    assert len(validated.layer_kv) == 2
    assert validated.kv_nbytes > 0


def test_validate_native_memory_seq_equals_head_dim_is_valid():
    adapter = make_tiny_adapter(max_frames=4, num_layers=2, head_dim=32, num_kv_heads=2)
    seq_len = 32
    blk0 = NativeEmbeddingBlock(
        frame_id=0,
        inputs_embeds=torch.zeros((1, seq_len, 64)),
        attention_mask=torch.ones((1, seq_len), dtype=torch.bool),
        owner=adapter,
        revision=adapter.revision,
        prompt="reach goal",
    )
    layer_kv = tuple(
        (torch.zeros((1, 2, 32, 32)), torch.zeros((1, 2, 32, 32)))
        for _ in range(2)
    )
    state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk0,),
        attention_mask=torch.ones((1, 32), dtype=torch.bool),
        last_frame_id=0,
        frame_count=1,
        prompt="reach goal",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )
    snapshot = (make_dummy_encoded_frame(0, blk0),)
    validated = validate_native_memory(
        candidate=state,
        previous=None,
        snapshot=snapshot,
        prompt="reach goal",
    )
    assert validated.last_frame_id == 0
    assert validated.frame_count == 1


def test_validate_native_memory_shape_seq_mismatch_rejected():
    adapter = make_tiny_adapter(max_frames=4, num_layers=2, head_dim=32, num_kv_heads=2)
    blk0 = NativeEmbeddingBlock(
        frame_id=0,
        inputs_embeds=torch.zeros((1, 16, 64)),
        attention_mask=torch.ones((1, 16), dtype=torch.bool),
        owner=adapter,
        revision=adapter.revision,
        prompt="reach goal",
    )
    layer_kv = tuple(
        (torch.zeros((1, 2, 32, 32)), torch.zeros((1, 2, 32, 32)))
        for _ in range(2)
    )
    state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk0,),
        attention_mask=torch.ones((1, 16), dtype=torch.bool),
        last_frame_id=0,
        frame_count=1,
        prompt="reach goal",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )
    snapshot = (make_dummy_encoded_frame(0, blk0),)
    with pytest.raises(ValueError, match="mismatch with expected"):
        validate_native_memory(state, None, snapshot, "reach goal")


def test_validate_native_memory_rejections():
    adapter = make_tiny_adapter(max_frames=4, num_layers=2, head_dim=32, num_kv_heads=2)
    blk0 = NativeEmbeddingBlock(
        frame_id=0,
        inputs_embeds=torch.zeros((1, 16, 64)),
        attention_mask=torch.ones((1, 16), dtype=torch.bool),
        owner=adapter,
        revision=adapter.revision,
        prompt="reach goal",
    )
    blk1 = NativeEmbeddingBlock(
        frame_id=1,
        inputs_embeds=torch.zeros((1, 16, 64)),
        attention_mask=torch.ones((1, 16), dtype=torch.bool),
        owner=adapter,
        revision=adapter.revision,
        prompt="reach goal",
    )
    snapshot = (make_dummy_encoded_frame(0, blk0), make_dummy_encoded_frame(1, blk1))

    layer_kv = tuple(
        (torch.zeros((1, 2, 32, 32)), torch.zeros((1, 2, 32, 32)))
        for _ in range(2)
    )

    with pytest.raises(TypeError, match="must be NativeKVState"):
        validate_native_memory("not_a_state", None, snapshot, "reach goal")

    bad_prompt_state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk0, blk1),
        attention_mask=torch.ones((1, 32), dtype=torch.bool),
        last_frame_id=1,
        frame_count=2,
        prompt="wrong prompt",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )
    with pytest.raises(ValueError, match="prompt 'wrong prompt' does not match active prompt"):
        validate_native_memory(bad_prompt_state, None, snapshot, "reach goal")

    bad_cnt_state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk0, blk1),
        attention_mask=torch.ones((1, 32), dtype=torch.bool),
        last_frame_id=1,
        frame_count=99,
        prompt="reach goal",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )
    with pytest.raises(ValueError, match="frame_count mismatch"):
        validate_native_memory(bad_cnt_state, None, snapshot, "reach goal")

    bad_id_state = NativeKVState(
        layer_kv=layer_kv,
        blocks=(blk0,),
        attention_mask=torch.ones((1, 16), dtype=torch.bool),
        last_frame_id=0,
        frame_count=2,
        prompt="reach goal",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )
    with pytest.raises(ValueError, match="last_frame_id must equal snapshot latest frame"):
        validate_native_memory(bad_id_state, None, snapshot, "reach goal")

    nan_k = torch.zeros((1, 2, 32, 32))
    nan_k[0, 0, 0, 0] = float("nan")
    nan_kv = ((nan_k, torch.zeros((1, 2, 32, 32))), (torch.zeros((1, 2, 32, 32)), torch.zeros((1, 2, 32, 32))))
    nan_state = NativeKVState(
        layer_kv=nan_kv,
        blocks=(blk0, blk1),
        attention_mask=torch.ones((1, 32), dtype=torch.bool),
        last_frame_id=1,
        frame_count=2,
        prompt="reach goal",
        owner=adapter,
        revision=adapter.revision,
        rebuild_count=0,
    )
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_native_memory(nan_state, None, snapshot, "reach goal")


def test_make_native_cache_callbacks_contract():
    adapter = make_tiny_adapter(max_frames=4, num_layers=2, head_dim=32, num_kv_heads=2)

    def mock_encode_frame(images, frame_id, prompt, capture_time=None, observation_time=None):
        return NativeEmbeddingBlock(
            frame_id=frame_id,
            inputs_embeds=torch.zeros((1, 16, 64)),
            attention_mask=torch.ones((1, 16), dtype=torch.bool),
            owner=adapter,
            revision=adapter.revision,
            prompt=prompt,
        )

    def mock_read_blocks(new_blocks, prompt, previous=None):
        prev_cnt = previous.frame_count if previous is not None else 0
        prev_blocks = previous.blocks if previous is not None else ()
        combined_blocks = (prev_blocks + tuple(new_blocks))[-4:]
        total_seq = sum(b.seq_len for b in combined_blocks)
        layer_kv = tuple(
            (torch.zeros((1, 2, total_seq, 32)), torch.zeros((1, 2, total_seq, 32)))
            for _ in range(2)
        )
        state = NativeKVState(
            layer_kv=layer_kv,
            blocks=combined_blocks,
            attention_mask=torch.ones((1, total_seq), dtype=torch.bool),
            last_frame_id=new_blocks[-1].frame_id,
            frame_count=prev_cnt + len(new_blocks),
            prompt=prompt,
            owner=adapter,
            revision=adapter.revision,
            rebuild_count=0,
        )
        deep = torch.zeros((1, 16, 64))
        shallow = torch.zeros((1, 16, 64))
        return deep, shallow, state

    adapter.encode_frame = mock_encode_frame
    adapter.read_blocks = mock_read_blocks

    encode, plan, validate = make_native_cache_callbacks(adapter, prompt="reach goal")
    validate()

    obs0 = Observation(
        frame_id=0,
        capture_time=0.0,
        images=[torch.zeros((1, 3, 16, 16))],
        state=torch.zeros((1, 10)),
        state_mask=torch.ones((1, 10), dtype=torch.bool),
        action_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    payload0 = adapter.encode_frame(list(obs0.images), 0, "reach goal")
    enc_frame0 = EncodedFrame(
        observation=obs0,
        payload=payload0,
        encode_started=0.0,
        encode_finished=0.1,
    )
    with pytest.raises(ValueError, match="does not match bound factory prompt"):
        plan((enc_frame0,), plan_prompt="different prompt", previous_memory=None)

    comp = plan((enc_frame0,), plan_prompt="reach goal", previous_memory=None)
    assert isinstance(comp, PlanComputation)
    assert comp.actions.shape == (1, 5, 4)
    assert comp.next_memory is not None
    assert comp.next_memory.last_frame_id == 0
    assert comp.next_memory.frame_count == 1

    with AsyncVisualPlanner(
        encode=encode,
        plan=plan,
        max_frames=4,
        stateful=True,
        memory_validator=validate_native_memory,
        validate=validate,
    ) as planner:
        planner.reset(episode_id="ep0", prompt="reach goal")
        planner.submit(obs0)
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()
        res = planner.wait_plan(timeout=5.0)
        assert res is not None
        assert res.source_frame_id == 0
        assert res.computation.actions.shape == (1, 5, 4)
        assert res.computation.next_memory is None

        st = planner.stats()
        assert st["memory_frame_count"] == 1
        assert st["last_memory_frame_id"] == 0
        assert st["memory_bytes"] > 0


def test_two_calls_batchsize2_history4_and_failure_isolation():
    adapter = make_tiny_adapter(max_frames=4, num_layers=2, head_dim=32, num_kv_heads=2)

    def mock_encode_frame(images, frame_id, prompt, capture_time=None, observation_time=None):
        return NativeEmbeddingBlock(
            frame_id=frame_id,
            inputs_embeds=torch.zeros((1, 16, 64)),
            attention_mask=torch.ones((1, 16), dtype=torch.bool),
            owner=adapter,
            revision=adapter.revision,
            prompt=prompt,
        )

    def mock_read_blocks(new_blocks, prompt, previous=None):
        prev_cnt = previous.frame_count if previous is not None else 0
        prev_blocks = previous.blocks if previous is not None else ()
        combined_blocks = (prev_blocks + tuple(new_blocks))[-4:]
        total_seq = sum(b.seq_len for b in combined_blocks)
        layer_kv = tuple(
            (torch.zeros((1, 2, total_seq, 32)), torch.zeros((1, 2, total_seq, 32)))
            for _ in range(2)
        )
        state = NativeKVState(
            layer_kv=layer_kv,
            blocks=combined_blocks,
            attention_mask=torch.ones((1, total_seq), dtype=torch.bool),
            last_frame_id=new_blocks[-1].frame_id,
            frame_count=prev_cnt + len(new_blocks),
            prompt=prompt,
            owner=adapter,
            revision=adapter.revision,
            rebuild_count=0,
        )
        deep = torch.zeros((1, 16, 64))
        shallow = torch.zeros((1, 16, 64))
        return deep, shallow, state

    adapter.encode_frame = mock_encode_frame
    adapter.read_blocks = mock_read_blocks

    encode, plan, validate = make_native_cache_callbacks(adapter, prompt="reach goal")

    with AsyncVisualPlanner(
        encode=encode,
        plan=plan,
        max_frames=4,
        stateful=True,
        memory_validator=validate_native_memory,
        validate=validate,
    ) as planner:
        planner.reset(episode_id="ep0", prompt="reach goal")

        for fid in (0, 1):
            obs = Observation(
                frame_id=fid,
                capture_time=0.0,
                images=[torch.zeros((1, 3, 16, 16))],
                state=torch.zeros((1, 10)),
                state_mask=torch.ones((1, 10), dtype=torch.bool),
                action_mask=torch.ones((1, 4), dtype=torch.bool),
            )
            planner.submit(obs)

        assert planner.wait_ready(min_frames=2, timeout=5.0)
        assert planner.request_plan()
        res1 = planner.wait_plan(timeout=5.0)
        assert res1 is not None
        assert res1.source_frame_id == 1
        assert res1.computation.next_memory is None

        st1 = planner.stats()
        assert st1["memory_frame_count"] == 2
        assert st1["last_memory_frame_id"] == 1

        for fid in (2, 3):
            obs = Observation(
                frame_id=fid,
                capture_time=0.0,
                images=[torch.zeros((1, 3, 16, 16))],
                state=torch.zeros((1, 10)),
                state_mask=torch.ones((1, 10), dtype=torch.bool),
                action_mask=torch.ones((1, 4), dtype=torch.bool),
            )
            planner.submit(obs)

        assert planner.wait_ready(min_frames=2, timeout=5.0)
        assert planner.request_plan()
        res2 = planner.wait_plan(timeout=5.0)
        assert res2 is not None
        assert res2.source_frame_id == 3
        assert res2.computation.next_memory is None

        st2 = planner.stats()
        assert st2["memory_frame_count"] == 4
        assert st2["last_memory_frame_id"] == 3
        assert len(planner._memory.blocks) == 4

        old_mem = planner._memory

        def failing_read_blocks(new_blocks, prompt, previous=None):
            deep, shallow, state = mock_read_blocks(new_blocks, prompt, previous)
            bad_state = NativeKVState(
                layer_kv=state.layer_kv,
                blocks=state.blocks,
                attention_mask=state.attention_mask,
                last_frame_id=state.last_frame_id,
                frame_count=999,
                prompt=prompt,
                owner=state.owner,
                revision=state.revision,
                rebuild_count=state.rebuild_count,
            )
            return deep, shallow, bad_state

        adapter.read_blocks = failing_read_blocks

        obs4 = Observation(
            frame_id=4,
            capture_time=0.0,
            images=[torch.zeros((1, 3, 16, 16))],
            state=torch.zeros((1, 10)),
            state_mask=torch.ones((1, 10), dtype=torch.bool),
            action_mask=torch.ones((1, 4), dtype=torch.bool),
        )
        planner.submit(obs4)
        assert planner.wait_ready(min_frames=1, timeout=5.0)
        assert planner.request_plan()

        with pytest.raises(RuntimeError, match="frame_count mismatch"):
            planner.wait_plan(timeout=5.0)

        assert planner._memory is old_mem
        assert planner._memory.frame_count == 4
        assert planner._memory.last_frame_id == 3


def test_actual_qwen3_planner_pipeline_and_negative_guards():
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16)
    prompt = "execute task"
    encode, plan, validate = make_native_cache_callbacks(adapter, prompt=prompt)

    with AsyncVisualPlanner(
        encode=encode,
        plan=plan,
        max_frames=4,
        stateful=True,
        memory_validator=validate_native_memory,
        validate=validate,
    ) as planner:
        planner.reset(episode_id="qwen3_ep", prompt=prompt)

        for fid in (0, 1):
            obs = Observation(
                frame_id=fid,
                capture_time=0.0,
                images=[torch.zeros((1, 3, 16, 16))],
                state=torch.zeros((1, 10)),
                state_mask=torch.ones((1, 10), dtype=torch.bool),
                action_mask=torch.ones((1, 24), dtype=torch.bool),
            )
            planner.submit(obs)

        assert planner.wait_ready(min_frames=2, timeout=5.0)
        assert planner.request_plan()
        res1 = planner.wait_plan(timeout=5.0)
        assert res1 is not None
        assert res1.source_frame_id == 1
        assert res1.computation.actions.shape == (1, 50, 24)
        assert planner._memory.rebuild_count == 0
        assert len(planner._memory.blocks) == 2

        for fid in (2, 3):
            obs = Observation(
                frame_id=fid,
                capture_time=0.0,
                images=[torch.zeros((1, 3, 16, 16))],
                state=torch.zeros((1, 10)),
                state_mask=torch.ones((1, 10), dtype=torch.bool),
                action_mask=torch.ones((1, 24), dtype=torch.bool),
            )
            planner.submit(obs)

        assert planner.wait_ready(min_frames=2, timeout=5.0)
        assert planner.request_plan()
        res2 = planner.wait_plan(timeout=5.0)
        assert res2 is not None
        assert res2.source_frame_id == 3
        assert planner._memory.rebuild_count == 0
        assert len(planner._memory.blocks) == 4

        for fid in (4, 5):
            obs = Observation(
                frame_id=fid,
                capture_time=0.0,
                images=[torch.zeros((1, 3, 16, 16))],
                state=torch.zeros((1, 10)),
                state_mask=torch.ones((1, 10), dtype=torch.bool),
                action_mask=torch.ones((1, 24), dtype=torch.bool),
            )
            planner.submit(obs)

        assert planner.wait_ready(min_frames=2, timeout=5.0)
        assert planner.request_plan()
        res3 = planner.wait_plan(timeout=5.0)
        assert res3 is not None
        assert res3.source_frame_id == 5
        assert planner._memory.rebuild_count == 1
        assert len(planner._memory.blocks) == 4
        assert [b.frame_id for b in planner._memory.blocks] == [2, 3, 4, 5]


def test_validator_negative_guards():
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16)
    prompt = "execute task"

    b0 = adapter.encode_frame(images=["img0"], frame_id=0, prompt=prompt)
    b1 = adapter.encode_frame(images=["img1"], frame_id=1, prompt=prompt)
    b2 = adapter.encode_frame(images=["img2"], frame_id=2, prompt=prompt)

    _, _, state01 = adapter.read_blocks([b0, b1], prompt=prompt)
    snap01 = (
        make_dummy_encoded_frame(0, b0),
        make_dummy_encoded_frame(1, b1),
    )
    validated01 = validate_native_memory(state01, None, snap01, prompt)
    assert validated01.frame_count == 2

    snap2 = (make_dummy_encoded_frame(2, b2),)
    _, _, valid_state2 = adapter.read_blocks([b2], prompt=prompt, previous=validated01)

    # 1. Dropping an old block from retained history
    dropped_blocks_state = NativeKVState(
        layer_kv=valid_state2.layer_kv,
        blocks=valid_state2.blocks[1:],
        attention_mask=valid_state2.attention_mask,
        last_frame_id=valid_state2.last_frame_id,
        frame_count=valid_state2.frame_count,
        prompt=valid_state2.prompt,
        owner=valid_state2.owner,
        revision=valid_state2.revision,
        rebuild_count=valid_state2.rebuild_count,
    )
    with pytest.raises(ValueError, match="blocks length"):
        validate_native_memory(dropped_blocks_state, validated01, snap2, prompt)

    # 2. Candidate stale revision
    stale_rev_state = NativeKVState(
        layer_kv=valid_state2.layer_kv,
        blocks=valid_state2.blocks,
        attention_mask=valid_state2.attention_mask,
        last_frame_id=valid_state2.last_frame_id,
        frame_count=valid_state2.frame_count,
        prompt=valid_state2.prompt,
        owner=valid_state2.owner,
        revision=valid_state2.revision + 99,
        rebuild_count=valid_state2.rebuild_count,
    )
    with pytest.raises(ValueError, match="revision .* does not match owner"):
        validate_native_memory(stale_rev_state, validated01, snap2, prompt)

    # 3. V dtype mismatch
    bad_v_kv = tuple(
        (k, v.to(torch.float64))
        for k, v in valid_state2.layer_kv
    )
    bad_v_state = NativeKVState(
        layer_kv=bad_v_kv,
        blocks=valid_state2.blocks,
        attention_mask=valid_state2.attention_mask,
        last_frame_id=valid_state2.last_frame_id,
        frame_count=valid_state2.frame_count,
        prompt=valid_state2.prompt,
        owner=valid_state2.owner,
        revision=valid_state2.revision,
        rebuild_count=valid_state2.rebuild_count,
    )
    with pytest.raises(TypeError, match="dtype must match native_core"):
        validate_native_memory(bad_v_state, validated01, snap2, prompt)

    # 4. Payload replaced with different block (non-shared storage)
    b2_replaced = NativeEmbeddingBlock(
        frame_id=2,
        inputs_embeds=b2.inputs_embeds.clone(),
        attention_mask=b2.attention_mask.clone(),
        owner=adapter,
        revision=adapter.revision,
        prompt=prompt,
    )
    snap2_replaced = (make_dummy_encoded_frame(2, b2_replaced),)
    with pytest.raises(ValueError, match="must share storage with expected block"):
        validate_native_memory(valid_state2, validated01, snap2_replaced, prompt)

    # 5. Mask corruption
    corrupted_mask = valid_state2.attention_mask.clone()
    corrupted_mask[0, 0] = not corrupted_mask[0, 0]
    corrupted_mask_state = NativeKVState(
        layer_kv=valid_state2.layer_kv,
        blocks=valid_state2.blocks,
        attention_mask=corrupted_mask,
        last_frame_id=valid_state2.last_frame_id,
        frame_count=valid_state2.frame_count,
        prompt=valid_state2.prompt,
        owner=valid_state2.owner,
        revision=valid_state2.revision,
        rebuild_count=valid_state2.rebuild_count,
    )
    with pytest.raises(ValueError, match="content does not match concatenated block masks"):
        validate_native_memory(corrupted_mask_state, validated01, snap2, prompt)
