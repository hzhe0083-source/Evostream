"""Regression tests for dynamic timestamps in native KV cache and async pipeline."""
from __future__ import annotations
import dataclasses
import pytest
import torch
from fabri_moss.async_pipeline import AsyncVisualPlanner, EncodedFrame, Observation
from fabri_moss.native_async import make_native_cache_callbacks, validate_native_memory
from fabri_moss.native_cache import NativeKVState, format_frame_timestamp
from fabri_moss.tests.test_native_cache import make_tiny_native_adapter


def _obs(fid: int, obs_t: float, cap_t: float = 0.0) -> Observation:
    return Observation(
        frame_id=fid, capture_time=cap_t, images=[torch.zeros((1, 3, 16, 16)) + fid],
        state=torch.zeros((1, 10)), state_mask=torch.ones((1, 10), dtype=torch.bool),
        action_mask=torch.ones((1, 24), dtype=torch.bool), observation_time=obs_t,
    )


def test_1_prefix_modelvisibility_and_invariance(monkeypatch):
    ad = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16, use_timestamps=True)
    emb = ad.policy.embedder
    monkeypatch.setattr(emb, "_build_multimodal_prompt", lambda tiles, p: f"Image-1: <image>\n{p}")
    prompts, orig_fuse = [], emb._prepare_and_fuse_embeddings

    def spy_fuse(prompt, vit_embeds, image_mask, num_tiles_list):
        prompts.append(prompt)
        return orig_fuse(prompt, vit_embeds, image_mask, num_tiles_list)

    monkeypatch.setattr(emb, "_prepare_and_fuse_embeddings", spy_fuse)
    b_t1 = ad.encode_frame(["img"], frame_id=0, prompt="task", capture_time=10.0, observation_time=0.1)
    b_t9 = ad.encode_frame(["img"], frame_id=0, prompt="task", capture_time=10.0, observation_time=0.9)
    assert not torch.equal(b_t1.inputs_embeds, b_t9.inputs_embeds)

    b_c100 = ad.encode_frame(["img"], frame_id=0, prompt="task", capture_time=100.0, observation_time=0.1)
    b_c900 = ad.encode_frame(["img"], frame_id=0, prompt="task", capture_time=900.0, observation_time=0.1)
    assert torch.equal(b_c100.inputs_embeds, b_c900.inputs_embeds)

    pfx = format_frame_timestamp(0, 0.1)
    assert prompts[0] == pfx + "Image-1: <image>\ntask" and prompts[0].index(pfx) < prompts[0].index("Image-1")
    assert b_t1.prompt == "task"
    _, _, st = ad.read_blocks([b_t1], prompt="task")
    det = st.detached()
    assert det.blocks[0].observation_time == 0.1 and det.blocks[0].capture_time == 10.0
    assert sum(p.numel() for p in ad.parameters()) == sum(p.numel() for p in ad.policy.parameters())
    assert all(not p.requires_grad for p in ad.parameters())


@pytest.mark.parametrize("W", [4, 16])
def test_2_window_read_and_rebuild_equivalence(monkeypatch, W):
    ad = make_tiny_native_adapter(max_frames=W, shallow_layer=1, max_text_length=16, use_timestamps=True)
    blks = [
        ad.encode_frame([f"frame{i}"], frame_id=i, prompt="task", observation_time=i * 0.125)
        for i in range(23)
    ]
    seen_ids, orig_exec = set(), ad._execute_native_layers

    def spy_exec(inputs_embeds, attention_mask_2d, cache, start_pos):
        for t in range(0, inputs_embeds.shape[1], 16):
            c = inputs_embeds[:, t : t + 16, :]
            for idx, b in enumerate(blks):
                if torch.equal(c, b.inputs_embeds):
                    seen_ids.add(idx)
        return orig_exec(inputs_embeds, attention_mask_2d, cache, start_pos)

    monkeypatch.setattr(ad, "_execute_native_layers", spy_exec)
    vit_calls = ad.policy.embedder.extract_feature_call_count
    deep, shallow, st = ad.read_blocks(blks, prompt="task")
    assert seen_ids == set(range(23)) and st.frame_count == 23 and len(st.blocks) == W
    assert st.rebuild_count == (23 - 1) // W and ad.policy.embedder.extract_feature_call_count == vit_calls

    mst = None
    for s in range(0, len(blks), W):
        mdeep, mshallow, mst = ad.read_blocks(blks[s : s + W], prompt="task", previous=mst)

    assert torch.allclose(deep, mdeep, atol=1e-5, rtol=0) and torch.allclose(shallow, mshallow, atol=1e-5, rtol=0)
    for (k1, v1), (k2, v2) in zip(st.layer_kv, mst.layer_kv):
        assert torch.allclose(k1, k2, atol=1e-5, rtol=0) and torch.allclose(v1, v2, atol=1e-5, rtol=0)


def test_3_invalid_timestamps_and_mode_guards(monkeypatch):
    ad = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16, use_timestamps=True)
    for bad in (None, float("nan"), -1.0, True):
        with pytest.raises(ValueError):
            ad.encode_frame(["img"], frame_id=0, prompt="task", observation_time=bad)

    b0 = ad.encode_frame(["img0"], frame_id=0, prompt="task", observation_time=1.0)
    _, _, st0 = ad.read_blocks([b0], prompt="task")
    old_kv = [(k.clone(), v.clone()) for k, v in st0.layer_kv]

    next_blks = [
        ad.encode_frame([f"img{i}"], frame_id=i, prompt="task", observation_time=1.0 + i * 0.1)
        for i in range(1, 9)
    ]
    next_blks[-1] = dataclasses.replace(next_blks[-1], observation_time=0.5)
    orig_exec, spy_calls = ad._execute_native_layers, 0

    def spy_no_call(*args, **kwargs):
        nonlocal spy_calls
        spy_calls += 1
        return orig_exec(*args, **kwargs)

    monkeypatch.setattr(ad, "_execute_native_layers", spy_no_call)
    with pytest.raises(ValueError, match="non-decreasing"):
        ad.read_blocks(next_blks, prompt="task", previous=st0)
    assert spy_calls == 0
    for (ok, ov), (ck, cv) in zip(old_kv, st0.layer_kv):
        assert torch.equal(ok, ck) and torch.equal(ov, cv)

    b1 = ad.encode_frame(["img1"], frame_id=1, prompt="task", capture_time=10.0, observation_time=2.0)
    b2 = ad.encode_frame(["img2"], frame_id=2, prompt="task", capture_time=20.0, observation_time=2.0)
    ad.read_blocks([b1, b2], prompt="task", previous=st0)

    ad_legacy = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16, use_timestamps=False)
    l0 = ad_legacy.encode_frame(["img0"], frame_id=0, prompt="task", observation_time=None)
    l1 = ad_legacy.encode_frame(["img0"], frame_id=0, prompt="task", observation_time=5.0)
    assert torch.equal(l0.inputs_embeds, l1.inputs_embeds)


def test_4_real_async_pipeline_batches_and_reset():
    ad = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16, use_timestamps=True)
    prompt = "execute task"
    enc, plan, val = make_native_cache_callbacks(ad, prompt=prompt)

    with AsyncVisualPlanner(
        encode=enc, plan=plan, max_frames=None, max_pending=None, overflow_policy="error",
        stateful=True, memory_validator=validate_native_memory, validate=val,
    ) as pl:
        pl.reset(episode_id="ep_async", prompt=prompt)
        for i in range(13):
            pl.submit(_obs(i, i * 0.125))
        assert pl.wait_ready(min_frames=13, timeout=10.0) and pl.request_plan()
        r1 = pl.wait_plan(timeout=10.0)
        assert r1.source_frame_id == 12 and r1.computation.next_memory is None
        assert [f.observation.frame_id for f in r1.frames] == list(range(13))
        assert [f.observation.observation_time for f in r1.frames] == [i * 0.125 for i in range(13)]

        for i in (13, 14):
            pl.submit(_obs(i, i * 0.125))
        assert pl.wait_ready(min_frames=2, timeout=10.0) and pl.request_plan()
        r2 = pl.wait_plan(timeout=10.0)
        assert r2.source_frame_id == 14 and r2.computation.next_memory is None
        assert [f.observation.frame_id for f in r2.frames] == [13, 14]
        assert pl._memory.frame_count == 15 and len(pl._memory.blocks) == 4
        assert [b.frame_id for b in pl._memory.blocks] == [11, 12, 13, 14]

        pl.reset(episode_id="ep_reset", prompt=prompt)
        assert pl._memory is None and pl._episode_capture_origin is None
        pl.submit(_obs(0, 0.0))
        assert pl.wait_ready(min_frames=1, timeout=10.0) and pl.request_plan()
        r_next = pl.wait_plan(timeout=10.0)
        assert r_next.source_frame_id == 0 and pl._memory.frame_count == 1


def test_5_validator_tamper_and_detached_contract():
    ad = make_tiny_native_adapter(max_frames=4, shallow_layer=1, max_text_length=16, use_timestamps=True)
    prompt = "execute task"
    blks = [
        ad.encode_frame(["img"], frame_id=i, prompt=prompt, capture_time=0.0, observation_time=i * 0.1)
        for i in range(8)
    ]
    snap = tuple(
        EncodedFrame(observation=_obs(i, i * 0.1, 0.0), payload=b, encode_started=0.0, encode_finished=0.1)
        for i, b in enumerate(blks)
    )
    _, _, st = ad.read_blocks(blks, prompt=prompt)
    val_st = validate_native_memory(st, None, snap, prompt)
    assert isinstance(val_st, NativeKVState) and val_st.frame_count == 8 and val_st is not st

    tampered_b = dataclasses.replace(st.blocks[-1], observation_time=st.blocks[-1].observation_time + 1.0)
    tampered_cand = NativeKVState(
        layer_kv=st.layer_kv, blocks=st.blocks[:-1] + (tampered_b,), attention_mask=st.attention_mask,
        last_frame_id=st.last_frame_id, frame_count=st.frame_count, prompt=st.prompt,
        owner=st.owner, revision=st.revision, rebuild_count=st.rebuild_count,
    )
    with pytest.raises(ValueError, match="observation_time"):
        validate_native_memory(tampered_cand, None, snap, prompt)
