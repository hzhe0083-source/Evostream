"""Verification probe for FabriVLA native LLM KV-cache without modifying attention structures.

Sections:
  A. Single frame parity: native policy vs NativeCacheAdapter(read_blocks).
     Zero new trainable parameters, identical layer identities, action head sampling with fixed noise.
  B. History incremental caching vs fresh replay:
     3-segment caching (1 + 4 + 2 blocks) vs fresh replay, immutability, diagnostic one-shot comparison.
  C. Eviction & rebuild:
     Sliding window (W=2) eviction, block rebinding/rebuilding, exact parity with fresh adapter.
  D. Asynchronous pipeline integration:
     AsyncVisualPlanner with validate_native_memory across two batches (5 + 2 frames).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fabri_moss.async_pipeline import (
    AsyncVisualPlanner,
    Observation,
)
from fabri_moss.data import MetaWorldWindows
from fabri_moss.native_async import make_native_cache_callbacks, validate_native_memory
from fabri_moss.native_cache import (
    NativeCacheAdapter,
    NativeCacheConfig,
    NativeEmbeddingBlock,
    NativeKVState,
    format_frame_timestamp,
)
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint


def compute_tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """Compute numerical differences between two tensors in float32."""
    if a.shape != b.shape or a.numel() == 0:
        raise AssertionError(f"Comparison shapes differ or are empty: {a.shape}, {b.shape}")
    a_f = a.float().detach().cpu()
    b_f = b.float().detach().cpu()
    if not torch.isfinite(a_f).all() or not torch.isfinite(b_f).all():
        raise AssertionError("Comparison contains non-finite values")
    diff = (a_f - b_f).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())
    norm_scale = float(torch.sqrt((b_f ** 2).mean()).item() + 1e-8)
    rel_rmse = float(rmse / norm_scale)
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "allclose_1e5": bool(max_diff <= 1e-5),
        "allclose_1e4": bool(max_diff <= 1e-4),
    }


def find_episode_samples(
    dataset: MetaWorldWindows,
    target_episode_id: int = 148,
    required_frames: int = 7,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Locate first required_frames of target_episode_id using anchors and index exactly 7 times."""
    if not hasattr(dataset, "anchors"):
        raise AttributeError("Dataset does not have 'anchors' metadata attribute")

    matching_indices: List[int] = []
    for idx, anchor in enumerate(dataset.anchors):
        ep_dict, _ = anchor
        ep_idx = int(ep_dict.get("episode_index", ep_dict.get("episode_id", -1)))
        if ep_idx == target_episode_id:
            matching_indices.append(idx)
            if len(matching_indices) == required_frames:
                break

    if len(matching_indices) < required_frames:
        raise ValueError(
            f"Episode {target_episode_id} not found or insufficient frames in anchors: "
            f"found {len(matching_indices)}, required {required_frames}."
        )

    chosen_samples = [dataset[i] for i in matching_indices]
    fids = [int(s["frame_ids"][-1]) for s in chosen_samples]
    expected_fids = list(range(fids[0], fids[0] + required_frames))
    if fids != expected_fids:
        raise ValueError(f"Episode {target_episode_id} frames not consecutive: {fids} vs {expected_fids}")
    return target_episode_id, chosen_samples


def convert_samples_to_observations(samples: Sequence[Dict[str, Any]]) -> List[Observation]:
    """Convert dataset samples to strict CPU Observation objects with monotonic timestamps."""
    observations: List[Observation] = []
    t_prev = -1.0
    for s in samples:
        t_cap = time.monotonic()
        if t_cap <= t_prev:
            t_cap = t_prev + 1e-4
        t_prev = t_cap

        # Observation.images expects tuple/list of CPU tensors, PIL images, or numpy arrays
        # Use samples's last image from images_window (or images)
        img_item = s["images_window"][-1] if "images_window" in s else s["images"]
        if isinstance(img_item, (list, tuple)):
            images_list = list(img_item)
        else:
            images_list = [img_item]

        # Ensure all image tensors are on CPU
        cpu_images = [img.detach().cpu() if isinstance(img, torch.Tensor) else img for img in images_list]

        state_tensor = s["state"].clone().float().cpu()
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        state_mask_tensor = s["state_mask"].clone().bool().cpu()
        if state_mask_tensor.ndim == 1:
            state_mask_tensor = state_mask_tensor.unsqueeze(0)

        action_mask_tensor = s["action_mask"].clone().bool().cpu()
        if action_mask_tensor.ndim == 2:
            action_mask_tensor = action_mask_tensor[0:1, :]
        elif action_mask_tensor.ndim == 1:
            action_mask_tensor = action_mask_tensor.unsqueeze(0)

        obs_time = None
        if "observation_times" in s and s["observation_times"]:
            obs_time = float(s["observation_times"][-1])

        obs = Observation(
            frame_id=int(s["frame_ids"][-1]),
            capture_time=t_cap,
            images=cpu_images,
            state=state_tensor,
            state_mask=state_mask_tensor,
            action_mask=action_mask_tensor,
            observation_time=obs_time,
        )
        observations.append(obs)
    return observations


@contextmanager
def count_vit_calls(adapter: NativeCacheAdapter) -> Iterator[Dict[str, int]]:
    """Count calls to ViT feature extraction and fusion."""
    counts = {"vit_extract": 0, "fuse_embeddings": 0}
    embedder = adapter.policy.embedder
    orig_extract = embedder.model.extract_feature
    orig_fuse = embedder._prepare_and_fuse_embeddings

    def wrapped_extract(*args: Any, **kwargs: Any) -> Any:
        counts["vit_extract"] += 1
        return orig_extract(*args, **kwargs)

    def wrapped_fuse(*args: Any, **kwargs: Any) -> Any:
        counts["fuse_embeddings"] += 1
        return orig_fuse(*args, **kwargs)

    embedder.model.extract_feature = wrapped_extract
    embedder._prepare_and_fuse_embeddings = wrapped_fuse
    try:
        yield counts
    finally:
        embedder.model.extract_feature = orig_extract
        embedder._prepare_and_fuse_embeddings = orig_fuse


def rebind_block_owner(block: NativeEmbeddingBlock, new_owner: Any) -> NativeEmbeddingBlock:
    """Create a shallow clone of NativeEmbeddingBlock with updated owner and revision, preserving timestamps."""
    return NativeEmbeddingBlock(
        frame_id=block.frame_id,
        inputs_embeds=block.inputs_embeds,
        attention_mask=block.attention_mask,
        owner=new_owner,
        revision=getattr(new_owner, "revision", getattr(new_owner, "_revision", 0)),
        prompt=block.prompt,
        capture_time=block.capture_time,
        observation_time=block.observation_time,
    )


# ---------------------------------------------------------------------------
# Section A: Single Frame Parity & Zero Parameter Invariance
# ---------------------------------------------------------------------------

def run_section_a_single_frame(
    policy: Any,
    adapter: NativeCacheAdapter,
    block0: NativeEmbeddingBlock,
    obs0: Observation,
    prompt: str,
    seed: int,
    flow_steps: int = 50,
) -> Dict[str, Any]:
    """Verify single frame deep/shallow/action parity <= 1e-5 and exact parameter identity equality."""
    policy_params = dict(policy.named_parameters())
    adapter_params = dict(adapter.named_parameters())

    # Verify adapter has exact same parameters with identical identities in both directions
    for name, p in policy_params.items():
        adapter_p_name = f"policy.{name}"
        if adapter_p_name not in adapter_params:
            raise AssertionError(f"Parameter {adapter_p_name} missing from adapter")
        if adapter_params[adapter_p_name] is not p:
            raise AssertionError(f"Parameter identity mismatch for {name}")

    for a_name, a_p in adapter_params.items():
        if not a_name.startswith("policy."):
            raise AssertionError(f"Unexpected non-policy parameter {a_name} found in adapter")
        orig_name = a_name[len("policy."):]
        if orig_name not in policy_params or policy_params[orig_name] is not a_p:
            raise AssertionError(f"Adapter parameter {a_name} does not match policy parameter")

    total_adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    if total_adapter_trainable != 0:
        raise AssertionError(f"Expected 0 trainable parameters, got {total_adapter_trainable}")

    core = adapter.native_core
    layer0 = core.layers[0]
    attn0 = getattr(layer0, "self_attn", None)
    mlp0 = getattr(layer0, "mlp", None)

    device = next(policy.parameters()).device

    # 1. Native policy forward on frame 0 (strictly under torch.no_grad)
    with torch.no_grad():
        image_mask = torch.ones((1, len(obs0.images)), dtype=torch.bool, device=device)
        orig_deep, orig_shallow = policy.get_vl_embeddings(
            images=list(obs0.images),
            image_mask=image_mask,
            prompt=prompt,
            shallow_layer_index=adapter.config.shallow_layer,
        )
    orig_deep = orig_deep.float()
    orig_shallow = orig_shallow.float()

    # 2. Adapter forward on block0
    deep0, shallow0, state0 = adapter.read_blocks([block0], prompt=prompt, previous=None)

    diff_deep = compute_tensor_diff(deep0, orig_deep)
    diff_shallow = compute_tensor_diff(shallow0, orig_shallow)

    if diff_deep["max_diff"] > 1e-5:
        raise AssertionError(f"Section A deep diff {diff_deep['max_diff']} exceeds 1e-5")
    if diff_shallow["max_diff"] > 1e-5:
        raise AssertionError(f"Section A shallow diff {diff_shallow['max_diff']} exceeds 1e-5")

    # 3. Action head sampling with fixed noise
    head = policy.action_head
    state_t = obs0.state.to(device)
    state_mask_t = obs0.state_mask.to(device)
    action_mask_t = obs0.action_mask.to(device)

    # Set flow steps on head.config
    orig_timesteps = getattr(head.config, "num_inference_timesteps", 50)
    head.config.num_inference_timesteps = flow_steps

    try:
        # Policy path sample with fixed seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        orig_action = head.sample(
            orig_deep,
            state=state_t,
            action_mask=action_mask_t,
            state_mask=state_mask_t,
            shallow_tokens=orig_shallow,
        )

        # Adapter path sample with identical seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        adapter_action = head.sample(
            deep0,
            state=state_t,
            action_mask=action_mask_t,
            state_mask=state_mask_t,
            shallow_tokens=shallow0,
        )
    finally:
        head.config.num_inference_timesteps = orig_timesteps

    action_diff = compute_tensor_diff(adapter_action, orig_action)
    if action_diff["max_diff"] > 1e-5:
        raise AssertionError(f"Section A action diff {action_diff['max_diff']} exceeds 1e-5")

    if not torch.isfinite(adapter_action).all():
        raise AssertionError("Action contains non-finite values")

    expected_action_shape = (1, head.config.horizon, head.config.per_action_dim)
    if adapter_action.shape != expected_action_shape:
        raise AssertionError(f"Unexpected action shape: {tuple(adapter_action.shape)}, expected {expected_action_shape}")

    # Check padding dims >= 4 are strictly zero
    padding_max = float(adapter_action[:, :, 4:].abs().max().item())
    if padding_max > 0.0:
        raise AssertionError(f"Action padding dims >= 4 non-zero: {padding_max}")

    return {
        "passed": True,
        "timestamps_mode": "disabled_baseline",
        "note": "Section A single frame parity is evaluated with timestamps disabled (use_timestamps=False baseline). Contract explicitly forbids claiming timestamp-enabled prompt matches untimestamped baseline.",
        "trainable_parameters": total_adapter_trainable,
        "deep_diff": diff_deep,
        "shallow_diff": diff_shallow,
        "action_diff": action_diff,
        "action_shape": list(adapter_action.shape),
        "action_padding_max": padding_max,
        "layer0_attn_id": id(attn0),
        "layer0_mlp_id": id(mlp0),
    }


# ---------------------------------------------------------------------------
# Section B: History Incremental vs Fresh Replay & Diagnostic One-Shot
# ---------------------------------------------------------------------------

def run_section_b_history(
    adapter: NativeCacheAdapter,
    blocks: List[NativeEmbeddingBlock],
    prompt: str,
) -> Dict[str, Any]:
    """Verify incremental history caching [0], [1..4], [5..6] against fresh replay."""
    b0 = [blocks[0]]
    b1_4 = blocks[1:5]
    b5_6 = blocks[5:7]
    all_7 = blocks[:7]

    # --- Phase 1: Incremental online processing ---
    with count_vit_calls(adapter) as vit_counts:
        # Segment 1: block 0
        d0, s0, state1 = adapter.read_blocks(b0, prompt=prompt, previous=None)

        # Immutability snapshot of state1 KV
        state1_kv_clones = tuple((k.clone(), v.clone()) for k, v in state1.layer_kv)

        # Segment 2: blocks 1..4 -> state5
        d1, s1, state5 = adapter.read_blocks(b1_4, prompt=prompt, previous=state1)

        # Assert state1 was NOT modified in place
        for idx, (k_orig, v_orig) in enumerate(state1.layer_kv):
            k_clone, v_clone = state1_kv_clones[idx]
            if not torch.equal(k_orig, k_clone) or not torch.equal(v_orig, v_clone):
                raise AssertionError(f"state1 KV was mutated during incremental append at layer {idx}")

        state5_kv_clones = tuple((k.clone(), v.clone()) for k, v in state5.layer_kv)

        # Segment 3: blocks 5..6 -> state7
        d2, s2, state7 = adapter.read_blocks(b5_6, prompt=prompt, previous=state5)

        for idx, (k_orig, v_orig) in enumerate(state5.layer_kv):
            k_clone, v_clone = state5_kv_clones[idx]
            if not torch.equal(k_orig, k_clone) or not torch.equal(v_orig, v_clone):
                raise AssertionError(f"state5 KV was mutated during incremental append at layer {idx}")

    # ViT and project calls must be exactly 0 during read_blocks
    if vit_counts["vit_extract"] != 0 or vit_counts["fuse_embeddings"] != 0:
        raise AssertionError(f"ViT/embedder was invoked during read_blocks: {vit_counts}")

    # --- Phase 2: Fresh replay with exact same partitions ---
    _, _, f_state1 = adapter.read_blocks(b0, prompt=prompt, previous=None)
    _, _, f_state5 = adapter.read_blocks(b1_4, prompt=prompt, previous=f_state1)
    f_d2, f_s2, f_state7 = adapter.read_blocks(b5_6, prompt=prompt, previous=f_state5)

    # Parity check between incremental state7 and fresh replay f_state7
    diff_deep = compute_tensor_diff(d2, f_d2)
    diff_shallow = compute_tensor_diff(s2, f_s2)

    max_kv_diff = 0.0
    for l_idx in range(len(state7.layer_kv)):
        k_diff = (state7.layer_kv[l_idx][0].float() - f_state7.layer_kv[l_idx][0].float()).abs().max().item()
        v_diff = (state7.layer_kv[l_idx][1].float() - f_state7.layer_kv[l_idx][1].float()).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)

    if diff_deep["max_diff"] > 1e-5:
        raise AssertionError(f"Section B fresh replay deep diff {diff_deep['max_diff']} > 1e-5")
    if diff_shallow["max_diff"] > 1e-5:
        raise AssertionError(f"Section B fresh replay shallow diff {diff_shallow['max_diff']} > 1e-5")
    if max_kv_diff > 1e-5:
        raise AssertionError(f"Section B fresh replay KV diff {max_kv_diff} > 1e-5")

    # Latest-only difference check (history vs latest block alone)
    latest_d, latest_s, _ = adapter.read_blocks([blocks[6]], prompt=prompt, previous=None)
    history_vs_latest_deep = compute_tensor_diff(d2, latest_d)
    history_vs_latest_shallow = compute_tensor_diff(s2, latest_s)

    # Diagnostic one-shot full 7 blocks comparison (diagnostic report only, not strict assertion)
    one_shot_d, one_shot_s, one_shot_state = adapter.read_blocks(all_7, prompt=prompt, previous=None)
    oneshot_diff_deep = compute_tensor_diff(d2, one_shot_d)
    oneshot_diff_shallow = compute_tensor_diff(s2, one_shot_s)

    mask_last = blocks[6].attention_mask
    padding_indices = (mask_last == 0).nonzero(as_tuple=True)[1]
    valid_indices = (mask_last != 0).nonzero(as_tuple=True)[1]
    if len(padding_indices) > 0:
        pad_deep_diff = (d2[:, padding_indices, :].float() - one_shot_d[:, padding_indices, :].float()).abs().max().item()
    else:
        pad_deep_diff = 0.0

    if len(valid_indices) > 0:
        valid_deep_diff = (d2[:, valid_indices, :].float() - one_shot_d[:, valid_indices, :].float()).abs().max().item()
    else:
        valid_deep_diff = 0.0

    total_tokens = sum(b.seq_len for b in state7.blocks)
    kv_bytes = state7.kv_nbytes

    return {
        "passed": True,
        "fresh_replay_deep_diff": diff_deep,
        "fresh_replay_shallow_diff": diff_shallow,
        "fresh_replay_max_kv_diff": max_kv_diff,
        "history_vs_latest_deep": history_vs_latest_deep,
        "history_vs_latest_shallow": history_vs_latest_shallow,
        "diagnostic_oneshot_deep": oneshot_diff_deep,
        "diagnostic_oneshot_shallow": oneshot_diff_shallow,
        "diagnostic_oneshot_valid_diff_max": valid_deep_diff,
        "diagnostic_oneshot_padding_diff_max": pad_deep_diff,
        "frame_count": state7.frame_count,
        "retained_blocks": len(state7.blocks),
        "total_seq_len": total_tokens,
        "kv_bytes": kv_bytes,
        "kv_megabytes": round(kv_bytes / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Section C: Eviction & Rebuild (W=2)
# ---------------------------------------------------------------------------

def run_section_c_eviction(
    policy: Any,
    blocks: List[NativeEmbeddingBlock],
    prompt: str,
) -> Dict[str, Any]:
    """Verify window eviction and rebuild with W=2 without extra ViT calls."""
    adapter_w2 = NativeCacheAdapter(policy, NativeCacheConfig(max_frames=2, shallow_layer=adapter_shallow_layer(policy), use_timestamps=False))

    # Rebind blocks 0, 1, 2 to adapter_w2
    w2_b0 = rebind_block_owner(blocks[0], adapter_w2)
    w2_b1 = rebind_block_owner(blocks[1], adapter_w2)
    w2_b2 = rebind_block_owner(blocks[2], adapter_w2)

    with count_vit_calls(adapter_w2) as vit_counts:
        # Step 1: feed block 0 -> state1
        _, _, s1 = adapter_w2.read_blocks([w2_b0], prompt=prompt, previous=None)
        # Step 2: feed block 1 -> state2 (window full: blocks 0, 1)
        _, _, s2 = adapter_w2.read_blocks([w2_b1], prompt=prompt, previous=s1)
        if s2.rebuild_count != 0:
            raise AssertionError("Rebuild count should be 0 before overflow")

        # Step 3: feed block 2 -> state3 (overflows W=2, evicts block 0, triggers rebuild with [1, 2])
        d3, sh3, s3 = adapter_w2.read_blocks([w2_b2], prompt=prompt, previous=s2)

    if vit_counts["vit_extract"] != 0 or vit_counts["fuse_embeddings"] != 0:
        raise AssertionError("ViT was called during eviction/rebuild!")

    if s3.rebuild_count != 1:
        raise AssertionError(f"Expected rebuild_count == 1, got {s3.rebuild_count}")
    retained_fids = [b.frame_id for b in s3.blocks]
    if retained_fids != [w2_b1.frame_id, w2_b2.frame_id]:
        raise AssertionError(f"Expected retained frame IDs {[w2_b1.frame_id, w2_b2.frame_id]}, got {retained_fids}")

    # Compare s3 directly against a fresh adapter running [w2_b1, w2_b2]
    adapter_fresh_w2 = NativeCacheAdapter(policy, NativeCacheConfig(max_frames=2, shallow_layer=adapter_shallow_layer(policy), use_timestamps=False))
    fw2_b1 = rebind_block_owner(blocks[1], adapter_fresh_w2)
    fw2_b2 = rebind_block_owner(blocks[2], adapter_fresh_w2)

    fresh_d, fresh_sh, fresh_s = adapter_fresh_w2.read_blocks([fw2_b1, fw2_b2], prompt=prompt, previous=None)

    diff_deep = compute_tensor_diff(d3, fresh_d)
    diff_shallow = compute_tensor_diff(sh3, fresh_sh)

    max_kv_diff = 0.0
    for l_idx in range(len(s3.layer_kv)):
        k_diff = (s3.layer_kv[l_idx][0].float() - fresh_s.layer_kv[l_idx][0].float()).abs().max().item()
        v_diff = (s3.layer_kv[l_idx][1].float() - fresh_s.layer_kv[l_idx][1].float()).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)

    if diff_deep["max_diff"] > 1e-5:
        raise AssertionError(f"Section C rebuild deep diff {diff_deep['max_diff']} > 1e-5")
    if diff_shallow["max_diff"] > 1e-5:
        raise AssertionError(f"Section C rebuild shallow diff {diff_shallow['max_diff']} > 1e-5")
    if max_kv_diff > 1e-5:
        raise AssertionError(f"Section C rebuild KV diff {max_kv_diff} > 1e-5")

    return {
        "passed": True,
        "rebuild_count": s3.rebuild_count,
        "retained_frame_ids": retained_fids,
        "rebuild_vs_fresh_deep_diff": diff_deep,
        "rebuild_vs_fresh_shallow_diff": diff_shallow,
        "rebuild_vs_fresh_max_kv_diff": max_kv_diff,
        "vit_calls_count": vit_counts["vit_extract"],
    }


def adapter_shallow_layer(policy: Any) -> int:
    """Helper to detect safe shallow layer index for test policy vs real checkpoint."""
    num_layers = len(policy.embedder.model.language_model.model.layers)
    return min(6, max(1, num_layers // 2))


# ---------------------------------------------------------------------------
# Section D: Asynchronous Pipeline (5 + 2 frames)
# ---------------------------------------------------------------------------

def run_section_d_async(
    adapter: NativeCacheAdapter,
    observations: List[Observation],
    prompt: str,
    flow_steps: int = 50,
) -> Dict[str, Any]:
    """Verify AsyncVisualPlanner with 5 + 2 frames and validate_native_memory."""
    # Ensure policy action_head has configured flow_steps
    head = adapter.policy.action_head
    orig_timesteps = getattr(head.config, "num_inference_timesteps", 50)
    head.config.num_inference_timesteps = flow_steps

    planner: Optional[AsyncVisualPlanner] = None
    try:
        encode_cb, plan_cb, validate_cb = make_native_cache_callbacks(adapter, prompt=prompt)

        planner = AsyncVisualPlanner(
            encode=encode_cb,
            plan=plan_cb,
            max_frames=5,
            max_pending=8,
            validate=validate_cb,
            stateful=True,
            memory_validator=validate_native_memory,
        )

        planner.reset(episode_id="ep_probe_1", prompt=prompt)

        # Batch 1: Submit frames 0..4 (5 frames)
        for obs in observations[:5]:
            if not planner.submit(obs):
                raise RuntimeError(f"Failed to submit observation frame_id={obs.frame_id}")

        if not planner.wait_ready(min_frames=5, timeout=30.0):
            raise TimeoutError("Timed out waiting for 5 frames ready in planner batch 1")

        if not planner.request_plan():
            raise RuntimeError("Failed to request plan #1")

        # Submit batch 2 frames (5..6) immediately to allow host overlap between vision and planning
        for obs in observations[5:7]:
            if not planner.submit(obs):
                raise RuntimeError(f"Failed to submit observation frame_id={obs.frame_id}")

        # Wait for plan #1 completion
        plan_res1 = planner.wait_plan(timeout=30.0)
        if plan_res1 is None:
            raise TimeoutError("Timed out waiting for plan #1")

        stats_after_plan1 = copy.deepcopy(planner.stats())

        # Wait for batch 2 to be ready, then request and wait for plan #2
        if not planner.wait_ready(min_frames=2, timeout=30.0):
            raise TimeoutError("Timed out waiting for 2 frames ready in planner batch 2")

        if not planner.request_plan():
            raise RuntimeError("Failed to request plan #2")

        plan_res2 = planner.wait_plan(timeout=30.0)
        if plan_res2 is None:
            raise TimeoutError("Timed out waiting for plan #2")

        stats_after_plan2 = copy.deepcopy(planner.stats())

        # Verify public PlanResult does not leak next_memory
        if plan_res1.computation.next_memory is not None:
            raise AssertionError("plan_res1 leaked next_memory in computation")
        if plan_res2.computation.next_memory is not None:
            raise AssertionError("plan_res2 leaked next_memory in computation")

        # Verify consumed frames
        p1_fids = [f.observation.frame_id for f in plan_res1.frames]
        p2_fids = [f.observation.frame_id for f in plan_res2.frames]
        expected_p1_ids = [obs.frame_id for obs in observations[:5]]
        expected_p2_ids = [obs.frame_id for obs in observations[5:7]]

        if p1_fids != expected_p1_ids:
            raise AssertionError(f"plan 1 consumed frames {p1_fids} != expected {expected_p1_ids}")
        if p2_fids != expected_p2_ids:
            raise AssertionError(f"plan 2 consumed frames {p2_fids} != expected {expected_p2_ids}")

        # In-context memory check after plan 2
        pipe_mem = planner._memory
        if not isinstance(pipe_mem, NativeKVState):
            raise AssertionError("planner._memory is not NativeKVState")
        if pipe_mem.frame_count != 7:
            raise AssertionError(f"Expected frame_count 7, got {pipe_mem.frame_count}")
        if len(pipe_mem.blocks) != 7:
            raise AssertionError(f"Expected 7 retained blocks, got {len(pipe_mem.blocks)}")

        kv_bytes = pipe_mem.kv_nbytes
        total_tokens = sum(b.seq_len for b in pipe_mem.blocks)

        # Check action tensor validity
        act1 = plan_res1.computation.actions
        act2 = plan_res2.computation.actions
        if not (torch.isfinite(act1).all() and torch.isfinite(act2).all()):
            raise AssertionError("Action tensor contains non-finite values")

        expected_act_shape = (1, head.config.horizon, head.config.per_action_dim)
        if act1.shape != expected_act_shape or act2.shape != expected_act_shape:
            raise AssertionError(f"Action shapes invalid: {act1.shape} / {act2.shape}, expected {expected_act_shape}")

        pad_max1 = float(act1[:, :, 4:].abs().max().item())
        pad_max2 = float(act2[:, :, 4:].abs().max().item())
        if pad_max1 > 0.0 or pad_max2 > 0.0:
            raise AssertionError(f"Action padding non-zero: {pad_max1}, {pad_max2}")

        # Independent direct 5+2 execution to compare plan2 deep/shallow features
        direct_blocks = [
            adapter.encode_frame(images=list(obs.images), frame_id=obs.frame_id, prompt=prompt)
            for obs in observations[:7]
        ]
        _, _, d_s1 = adapter.read_blocks(direct_blocks[:5], prompt=prompt, previous=None)
        d_deep2, d_sh2, d_s2 = adapter.read_blocks(direct_blocks[5:7], prompt=prompt, previous=d_s1)

        diff_plan2_deep = compute_tensor_diff(plan_res2.computation.deep, d_deep2)
        diff_plan2_shallow = compute_tensor_diff(plan_res2.computation.shallow, d_sh2)

        if diff_plan2_deep["max_diff"] > 1e-5:
            raise AssertionError(f"Async plan2 deep diff {diff_plan2_deep['max_diff']} > 1e-5")
        if diff_plan2_shallow["max_diff"] > 1e-5:
            raise AssertionError(f"Async plan2 shallow diff {diff_plan2_shallow['max_diff']} > 1e-5")

        # Quantify host intervals for vision encode vs plan
        all_events = stats_after_plan2.get("events", [])
        encode_events = [e for e in all_events if e.get("type") == "encoded"]
        plan1_start = plan_res1.started
        plan1_finish = plan_res1.finished
        overlap_seconds = 0.0
        for enc in encode_events:
            if enc.get("frame_id") in expected_p2_ids:
                s_e, f_e = enc["start"], enc["finish"]
                overlap = max(0.0, min(plan1_finish, f_e) - max(plan1_start, s_e))
                overlap_seconds += overlap

        # Test reset: reset planner to new episode with same prompt and verify memory clearing
        planner.reset(episode_id="ep_probe_2", prompt=prompt)
        stats_reset = copy.deepcopy(planner.stats())
        if stats_reset.get("memory_frame_count", 0) != 0 or stats_reset.get("memory_bytes", 0) != 0:
            raise AssertionError("Planner memory was not cleared on reset")

        # Process frame 0 once in new episode to verify count becomes 1, not 8
        obs0_fresh = Observation(
            frame_id=0,
            capture_time=time.monotonic(),
            images=observations[0].images,
            state=observations[0].state.clone(),
            state_mask=observations[0].state_mask.clone(),
            action_mask=observations[0].action_mask.clone(),
        )
        planner.submit(obs0_fresh)
        planner.wait_ready(min_frames=1, timeout=30.0)
        planner.request_plan()
        res_fresh = planner.wait_plan(timeout=30.0)
        stats_fresh = copy.deepcopy(planner.stats())
        if stats_fresh.get("memory_frame_count") != 1:
            raise AssertionError(f"Expected post-reset frame_count == 1, got {stats_fresh.get('memory_frame_count')}")

        return {
            "passed": True,
            "plan1_actions_shape": list(act1.shape),
            "plan2_actions_shape": list(act2.shape),
            "plan2_deep_diff": diff_plan2_deep,
            "plan2_shallow_diff": diff_plan2_shallow,
            "total_frames_committed": pipe_mem.frame_count,
            "retained_blocks_count": len(pipe_mem.blocks),
            "total_seq_len": total_tokens,
            "kv_bytes": kv_bytes,
            "kv_megabytes": round(kv_bytes / (1024 * 1024), 2),
            "host_concurrency_overlap_seconds": overlap_seconds,
            "stats_after_plan2": stats_after_plan2,
            "stats_after_reset": stats_reset,
            "stats_post_reset_one_frame": stats_fresh,
        }
    finally:
        head.config.num_inference_timesteps = orig_timesteps
        if planner is not None:
            planner.close()


# ---------------------------------------------------------------------------
# Section E: Dynamic Snapshot & Timestamped Execution (1 + 8 + 2 frames)
# ---------------------------------------------------------------------------

def run_section_e_dynamic_timestamps(
    adapter: NativeCacheAdapter,
    observations: List[Observation],
    prompt: str,
    flow_steps: int = 50,
) -> Dict[str, Any]:
    """Verify Section E: Dynamic unconstrained ready snapshots and visible timestamps.

    Configuration:
      - NativeCacheAdapter max_frames=4, use_timestamps=True
      - AsyncVisualPlanner(max_frames=None, max_pending=None, max_ready=None, overflow_policy="error")
      - 3 batches: 1 + 8 + 2 frames (total 11 frames)
      - Direct comparison with same 3 batches and timestamps on direct adapter.read_blocks
      - Hook / instrument _execute_native_layers to record all new tokens entering the LLM.
    """
    if len(observations) < 11:
        raise ValueError(f"Section E requires at least 11 observations, got {len(observations)}")

    obs11 = observations[:11]
    # Ensure all 11 have valid non-None observation_time
    for idx, obs in enumerate(obs11):
        if obs.observation_time is None:
            raise ValueError(f"Observation[{idx}] has observation_time=None; Section E strictly requires timestamps")

    head = adapter.policy.action_head
    orig_timesteps = getattr(head.config, "num_inference_timesteps", 50)
    head.config.num_inference_timesteps = flow_steps

    phase = "async"
    current_group: List[NativeEmbeddingBlock] = []
    async_exec_groups: List[List[int]] = []
    sample_counts: Dict[str, int] = {"async": 0, "direct": 0, "reset": 0}
    token_counts_by_phase: Dict[str, List[int]] = {"async": [], "direct": [], "reset": []}

    orig_exec = adapter._execute_native_layers
    orig_group = adapter._read_block_group
    orig_head_sample = head.sample

    def tracked_group(new_blocks: Sequence[NativeEmbeddingBlock], prompt_arg: str, previous: Optional[NativeKVState] = None):
        current_group.clear()
        current_group.extend(new_blocks)
        try:
            return orig_group(new_blocks, prompt_arg, previous=previous)
        finally:
            current_group.clear()

    def tracked_execute(inputs_embeds: torch.Tensor, attention_mask_2d: torch.Tensor, cache: Any, start_pos: int):
        token_counts_by_phase[phase].append(inputs_embeds.shape[1])
        new_fids: List[int] = []
        if current_group:
            block_len = current_group[0].seq_len
            total_tokens = inputs_embeds.shape[1]
            slices = [inputs_embeds[:, i : i + block_len, :] for i in range(0, total_tokens, block_len)]
            for blk in current_group:
                matched = any(torch.equal(blk.inputs_embeds, s) for s in slices)
                if not matched:
                    raise AssertionError(f"Frame {blk.frame_id} input embedding not found in actual LLM execution slices")
                new_fids.append(blk.frame_id)
        if phase == "async" and new_fids:
            async_exec_groups.append(new_fids)
        return orig_exec(inputs_embeds, attention_mask_2d, cache, start_pos)

    def tracked_head_sample(*args: Any, **kwargs: Any):
        sample_counts[phase] += 1
        return orig_head_sample(*args, **kwargs)

    adapter._execute_native_layers = tracked_execute
    adapter._read_block_group = tracked_group
    head.sample = tracked_head_sample

    planner: Optional[AsyncVisualPlanner] = None
    try:
        encode_cb, plan_cb, validate_cb = make_native_cache_callbacks(adapter, prompt=prompt)

        planner = AsyncVisualPlanner(
            encode=encode_cb,
            plan=plan_cb,
            max_frames=None,  # Dynamic mode: snapshot all ready frames
            max_pending=None,
            max_ready=None,
            overflow_policy="error",
            validate=validate_cb,
            stateful=True,
            memory_validator=validate_native_memory,
        )

        planner.reset(episode_id="ep_probe_e", prompt=prompt)

        # Batch 1: Submit 1 frame (frame 0)
        if not planner.submit(obs11[0]):
            raise RuntimeError("Failed to submit batch 1 (frame 0)")
        if not planner.wait_ready(min_frames=1, timeout=30.0):
            raise TimeoutError("Batch 1 (1 frame) timed out waiting for ready")
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan 1")
        plan_res1 = planner.wait_plan(timeout=30.0)
        if plan_res1 is None:
            raise TimeoutError("Plan 1 timed out")

        # Batch 2: Submit 8 frames (frames 1..8, where 8 > 5 and 8 > history 4)
        for obs in obs11[1:9]:
            if not planner.submit(obs):
                raise RuntimeError(f"Failed to submit batch 2 frame {obs.frame_id}")
        if not planner.wait_ready(min_frames=8, timeout=30.0):
            raise TimeoutError("Batch 2 (8 frames) timed out waiting for ready")
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan 2")
        plan_res2 = planner.wait_plan(timeout=30.0)
        if plan_res2 is None:
            raise TimeoutError("Plan 2 timed out")

        # Batch 3: Submit 2 frames (frames 9..10)
        for obs in obs11[9:11]:
            if not planner.submit(obs):
                raise RuntimeError(f"Failed to submit batch 3 frame {obs.frame_id}")
        if not planner.wait_ready(min_frames=2, timeout=30.0):
            raise TimeoutError("Batch 3 (2 frames) timed out waiting for ready")
        if not planner.request_plan():
            raise RuntimeError("Failed to request plan 3")
        plan_res3 = planner.wait_plan(timeout=30.0)
        if plan_res3 is None:
            raise TimeoutError("Plan 3 timed out")

        # Validate snapshot frames
        p1_fids = [f.observation.frame_id for f in plan_res1.frames]
        p2_fids = [f.observation.frame_id for f in plan_res2.frames]
        p3_fids = [f.observation.frame_id for f in plan_res3.frames]

        expected_p1_fids = [obs11[0].frame_id]
        expected_p2_fids = [obs.frame_id for obs in obs11[1:9]]
        expected_p3_fids = [obs.frame_id for obs in obs11[9:11]]

        if p1_fids != expected_p1_fids:
            raise AssertionError(f"Plan 1 fids mismatch: {p1_fids} vs {expected_p1_fids}")
        if p2_fids != expected_p2_fids:
            raise AssertionError(f"Plan 2 fids mismatch (must contain all 8 frames): {p2_fids} vs {expected_p2_fids}")
        if p3_fids != expected_p3_fids:
            raise AssertionError(f"Plan 3 fids mismatch: {p3_fids} vs {expected_p3_fids}")

        p2_obs_times = [f.observation.observation_time for f in plan_res2.frames]
        expected_p2_times = [obs.observation_time for obs in obs11[1:9]]
        if p2_obs_times != expected_p2_times:
            raise AssertionError(f"Plan 2 observation_times mismatch: {p2_obs_times} vs {expected_p2_times}")

        # Validate async plan groups and head sample counts
        expected_async_exec_groups = [
            [obs11[0].frame_id],
            [obs.frame_id for obs in obs11[1:5]],
            [obs.frame_id for obs in obs11[5:9]],
            [obs.frame_id for obs in obs11[9:11]],
        ]
        if async_exec_groups != expected_async_exec_groups:
            raise AssertionError(
                f"async_exec_groups mismatch: got {async_exec_groups}, expected {expected_async_exec_groups}"
            )
        flat_async_fids = [fid for grp in async_exec_groups for fid in grp]
        all_11_fids = [obs.frame_id for obs in obs11]
        if flat_async_fids != all_11_fids:
            raise AssertionError(f"Flattened async executed frame IDs {flat_async_fids} != {all_11_fids}")

        if sample_counts["async"] != 3:
            raise AssertionError(f"Expected exactly 3 action head sample calls in async phase, got {sample_counts['async']}")

        for p_idx, plan_res in enumerate((plan_res1, plan_res2, plan_res3), start=1):
            if plan_res.computation.next_memory is not None:
                raise AssertionError(f"Plan {p_idx} public next_memory is not None")

        pipe_mem = planner._memory
        if not isinstance(pipe_mem, NativeKVState):
            raise AssertionError("planner._memory is not NativeKVState")
        if pipe_mem.frame_count != 11:
            raise AssertionError(f"Expected cumulative frame_count=11, got {pipe_mem.frame_count}")
        if len(pipe_mem.blocks) != 4:
            raise AssertionError(f"Expected only last 4 frames retained in window, got {len(pipe_mem.blocks)}")
        if pipe_mem.rebuild_count != 3:
            raise AssertionError(f"Expected rebuild_count=3, got {pipe_mem.rebuild_count}")

        retained_fids = [b.frame_id for b in pipe_mem.blocks]
        expected_retained = [obs.frame_id for obs in obs11[-4:]]
        if retained_fids != expected_retained:
            raise AssertionError(f"Retained frame ids {retained_fids} != expected {expected_retained}")

        retained_times = [b.observation_time for b in pipe_mem.blocks]
        expected_retained_times = [obs.observation_time for obs in obs11[-4:]]
        if retained_times != expected_retained_times:
            raise AssertionError(f"Retained observation times {retained_times} != {expected_retained_times}")

        final_stats = copy.deepcopy(planner.stats())
        if final_stats["dropped_pending"] != 0 or final_stats["dropped_ready"] != 0 or final_stats["rejected_pending"] != 0:
            raise AssertionError(f"Zero drop contract violated: {final_stats}")

        # Switch phase to direct for parity check
        phase = "direct"
        direct_blocks = [
            adapter.encode_frame(
                images=list(obs.images),
                frame_id=obs.frame_id,
                prompt=prompt,
                capture_time=obs.capture_time,
                observation_time=obs.observation_time,
            )
            for obs in obs11
        ]

        d_deep1, d_sh1, d_s1 = adapter.read_blocks(direct_blocks[:1], prompt=prompt, previous=None)
        d_deep2, d_sh2, d_s2 = adapter.read_blocks(direct_blocks[1:9], prompt=prompt, previous=d_s1)
        d_deep3, d_sh3, d_s3 = adapter.read_blocks(direct_blocks[9:11], prompt=prompt, previous=d_s2)

        diff_deep1 = compute_tensor_diff(plan_res1.computation.deep, d_deep1)
        diff_deep2 = compute_tensor_diff(plan_res2.computation.deep, d_deep2)
        diff_deep3 = compute_tensor_diff(plan_res3.computation.deep, d_deep3)
        diff_sh1 = compute_tensor_diff(plan_res1.computation.shallow, d_sh1)
        diff_sh2 = compute_tensor_diff(plan_res2.computation.shallow, d_sh2)
        diff_sh3 = compute_tensor_diff(plan_res3.computation.shallow, d_sh3)

        if not (diff_deep1["allclose_1e5"] and diff_deep2["allclose_1e5"] and diff_deep3["allclose_1e5"]):
            raise AssertionError(f"Direct deep diff > 1e-5: p1={diff_deep1['max_diff']}, p2={diff_deep2['max_diff']}, p3={diff_deep3['max_diff']}")
        if not (diff_sh1["allclose_1e5"] and diff_sh2["allclose_1e5"] and diff_sh3["allclose_1e5"]):
            raise AssertionError(f"Direct shallow diff > 1e-5: p1={diff_sh1['max_diff']}, p2={diff_sh2['max_diff']}, p3={diff_sh3['max_diff']}")

        # Compare pipe_mem and d_s3 layer_kv shape, dtype, and max abs diff <= 1e-5
        if len(pipe_mem.layer_kv) != len(d_s3.layer_kv):
            raise AssertionError(f"Layer KV count mismatch: pipe {len(pipe_mem.layer_kv)} vs direct {len(d_s3.layer_kv)}")
        max_kv_diff = 0.0
        for l_idx, ((k_pipe, v_pipe), (k_dir, v_dir)) in enumerate(zip(pipe_mem.layer_kv, d_s3.layer_kv)):
            if k_pipe.shape != k_dir.shape or v_pipe.shape != v_dir.shape:
                raise AssertionError(f"Layer {l_idx} KV shape mismatch: K {k_pipe.shape} vs {k_dir.shape}, V {v_pipe.shape} vs {v_dir.shape}")
            if k_pipe.dtype != k_dir.dtype or v_pipe.dtype != v_dir.dtype:
                raise AssertionError(f"Layer {l_idx} KV dtype mismatch: K {k_pipe.dtype} vs {k_dir.dtype}, V {v_pipe.dtype} vs {v_dir.dtype}")
            diff_k = compute_tensor_diff(k_pipe, k_dir)
            diff_v = compute_tensor_diff(v_pipe, v_dir)
            if not diff_k["allclose_1e5"] or not diff_v["allclose_1e5"]:
                raise AssertionError(f"Layer {l_idx} KV max diff > 1e-5: K={diff_k['max_diff']}, V={diff_v['max_diff']}")
            max_kv_diff = max(max_kv_diff, diff_k["max_diff"], diff_v["max_diff"])

        # Verify action head shapes and finite padding
        for plan_res in (plan_res1, plan_res2, plan_res3):
            act = plan_res.computation.actions
            if not torch.isfinite(act).all():
                raise AssertionError("Action tensor contains non-finite values")
            if act.shape != (1, head.config.horizon, head.config.per_action_dim):
                raise AssertionError(f"Invalid action shape: {tuple(act.shape)}")
            pad_max = float(act[:, :, 4:].abs().max().item())
            if pad_max > 0.0:
                raise AssertionError(f"Action padding non-zero: {pad_max}")

        # Verification of timestamp visibility and prefix verification
        captured_prompts: List[str] = []
        embedder = adapter.policy.embedder
        orig_fuse = embedder._prepare_and_fuse_embeddings

        def spy_fuse(*args: Any, **kwargs: Any):
            p = kwargs.get("prompt", args[0] if args else "")
            captured_prompts.append(p)
            return orig_fuse(*args, **kwargs)

        embedder._prepare_and_fuse_embeddings = spy_fuse
        try:
            blk_t1 = adapter.encode_frame(
                images=list(obs11[0].images),
                frame_id=0,
                prompt=prompt,
                capture_time=100.0,
                observation_time=0.1,
            )
            blk_t2 = adapter.encode_frame(
                images=list(obs11[0].images),
                frame_id=0,
                prompt=prompt,
                capture_time=900.0,
                observation_time=0.9,
            )
            blk_c100 = blk_t1
            blk_c900 = adapter.encode_frame(
                images=list(obs11[0].images),
                frame_id=0,
                prompt=prompt,
                capture_time=900.0,
                observation_time=0.1,
            )
        finally:
            embedder._prepare_and_fuse_embeddings = orig_fuse

        embed_diff = float((blk_t1.inputs_embeds - blk_t2.inputs_embeds).abs().max().item())
        if embed_diff == 0.0:
            raise AssertionError("Timestamp is not visible: inputs_embeds identical for obs_time 0.1 vs 0.9")

        # Capture time must not enter model: embeddings with capture 100 vs 900 must be equal
        if not torch.equal(blk_c100.inputs_embeds, blk_c900.inputs_embeds):
            raise AssertionError("Capture time entered model embeddings: blk_c100 != blk_c900")

        # Verify timestamp prefix format and ordering relative to image marker
        expected_prefix = format_frame_timestamp(0, 0.1)
        if not captured_prompts or not captured_prompts[0].startswith(expected_prefix):
            raise AssertionError(f"Prompt does not start with expected prefix: '{expected_prefix}', got '{captured_prompts[0] if captured_prompts else None}'")
        actual_prefix = expected_prefix
        if "Image-1" in captured_prompts[0]:
            if captured_prompts[0].index(expected_prefix) >= captured_prompts[0].index("Image-1"):
                raise AssertionError("Timestamp prefix must precede image marker Image-1")

        # Verify adapter has zero trainable parameters and matches policy parameter IDs
        adapter_params = dict(adapter.named_parameters())
        policy_params = dict(adapter.policy.named_parameters())
        for p_name, p_tensor in policy_params.items():
            ad_p_name = f"policy.{p_name}"
            if ad_p_name not in adapter_params or adapter_params[ad_p_name] is not p_tensor:
                raise AssertionError(f"Parameter identity mismatch between adapter and policy for {p_name}")
        if any(p.requires_grad for p in adapter.parameters()):
            raise AssertionError("Adapter parameters must have requires_grad=False")

        # Test reset: same prompt, frame 0, observation_time 0.0
        phase = "reset"
        planner.reset(episode_id="ep_probe_e2", prompt=prompt)
        st_reset = copy.deepcopy(planner.stats())
        if st_reset.get("memory_frame_count", 0) != 0:
            raise AssertionError("Memory frame count not 0 after reset")

        obs_fresh = Observation(
            frame_id=0,
            capture_time=time.monotonic(),
            images=obs11[0].images,
            state=obs11[0].state.clone(),
            state_mask=obs11[0].state_mask.clone(),
            action_mask=obs11[0].action_mask.clone(),
            observation_time=0.0,
        )
        planner.submit(obs_fresh)
        planner.wait_ready(min_frames=1, timeout=30.0)
        planner.request_plan()
        res_fresh = planner.wait_plan(timeout=30.0)
        if res_fresh is None:
            raise TimeoutError("Fresh plan after reset timed out")
        st_fresh = copy.deepcopy(planner.stats())
        if st_fresh.get("memory_frame_count") != 1:
            raise AssertionError(f"Expected frame_count 1 after reset bootstrap, got {st_fresh.get('memory_frame_count')}")

        if sample_counts["reset"] != 1:
            raise AssertionError(f"Expected exactly 1 action head sample call in reset phase, got {sample_counts['reset']}")

        return {
            "passed": True,
            "batches_processed": [1, 8, 2],
            "async_execution_groups": async_exec_groups,
            "sample_counts_by_phase": sample_counts,
            "total_frames_committed": pipe_mem.frame_count,
            "retained_blocks_count": len(pipe_mem.blocks),
            "retained_frame_ids": retained_fids,
            "retained_observation_times": retained_times,
            "rebuild_count": pipe_mem.rebuild_count,
            "plan2_snapshot_len": len(plan_res2.frames),
            "plan2_frame_ids": p2_fids,
            "plan2_observation_times": p2_obs_times,
            "diff_deep2": diff_deep2,
            "diff_shallow2": diff_sh2,
            "max_kv_diff": max_kv_diff,
            "llm_execution_token_counts": token_counts_by_phase["async"],
            "llm_execution_token_counts_by_phase": token_counts_by_phase,
            "timestamp_prefix": actual_prefix,
            "timestamp_sensitivity_embed_diff": embed_diff,
            "final_planner_stats": final_stats,
            "stats_post_reset": st_fresh,
        }
    finally:
        if planner is not None:
            planner.close()
        adapter._execute_native_layers = orig_exec
        adapter._read_block_group = orig_group
        head.sample = orig_head_sample
        head.config.num_inference_timesteps = orig_timesteps


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabriVLA Native KV Cache Verification Probe")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to 93k checkpoint")
    parser.add_argument("--vlm", required=True, type=Path, help="Path to InternVL model directory")
    parser.add_argument("--fabri-root", required=True, type=Path, help="Path to FabriVLA repository root")
    parser.add_argument("--data-root", required=True, type=Path, help="Path to MetaWorld data directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save verification report")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to use (must be cuda)")
    parser.add_argument("--episode-id", default=148, type=int, help="Target episode id")
    parser.add_argument("--threads", default=4, type=int, help="DataLoader threads (must be > 0)")
    parser.add_argument("--seed", default=4042, type=int, help="Random seed for verification")
    parser.add_argument("--history-frames", default=16, type=int, help="Native cache window capacity (>= 7)")
    parser.add_argument("--flow-steps", default=50, type=int, help="Action flow matching steps (> 0)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Pre-checks on args before any CUDA/loader init
    if not (args.device.startswith("cuda") and torch.cuda.is_available()):
        sys.exit(f"Error: main probe strictly requires a functional CUDA device, got '{args.device}'")
    if args.threads <= 0:
        sys.exit(f"Error: threads must be > 0, got {args.threads}")
    if args.history_frames < 7:
        sys.exit(f"Error: history_frames must be >= 7, got {args.history_frames}")
    if args.flow_steps <= 0:
        sys.exit(f"Error: flow_steps must be > 0, got {args.flow_steps}")

    torch.set_num_threads(args.threads)

    # Reject non-empty output directory
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        contents = [f for f in out_dir.iterdir() if f.name != ".git"]
        if contents:
            sys.exit(f"Error: Output directory {out_dir} is not empty. Execution aborted.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CUDA and reset peak memory tracking
    dev_idx = torch.device(args.device).index or 0
    torch.cuda.set_device(dev_idx)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(dev_idx)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(dev_idx),
            "threads": args.threads,
        },
        "sections": {},
        "summary": {},
    }

    try:
        # Set seeds
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # 1. Load native checkpoint
        policy, raw_config, norm_stats, meta = load_native_checkpoint(
            fabri_root=args.fabri_root,
            checkpoint_path=args.checkpoint,
            vlm_path=args.vlm,
            device=args.device,
        )

        report["checkpoint_metadata"] = meta
        import transformers
        report["environment"]["transformers_version"] = transformers.__version__
        report["parameter_dtypes"] = {
            "vision": str(next(policy.embedder.model.vision_model.parameters()).dtype),
            "language": str(next(policy.embedder.model.language_model.parameters()).dtype),
            "action_head": str(next(policy.action_head.parameters()).dtype),
        }
        if any(p.device.type != "cuda" for p in policy.parameters()):
            raise AssertionError("All policy parameters must be on CUDA")

        # 2. Strict native FA2 assertion on GPU
        fa2_diag = assert_native_fa2(policy)
        report["fa2_diagnostics"] = fa2_diag

        # 3. Load dataset using root & norm_stats and locate 11 frames of target episode
        dataset = MetaWorldWindows(
            root=args.data_root,
            norm_stats=norm_stats,
            horizon=50,
            state_dim=24,
            action_dim=24,
            window=1,
            frame_stride=1,
            split="all",
            seed=args.seed,
        )

        _, raw_samples_11 = find_episode_samples(
            dataset=dataset,
            target_episode_id=args.episode_id,
            required_frames=11,
        )
        raw_samples = raw_samples_11[:7]
        observations = convert_samples_to_observations(raw_samples)
        observations_11 = convert_samples_to_observations(raw_samples_11)

        for idx, obs in enumerate(observations_11):
            if obs.observation_time is None:
                raise ValueError(f"Sample {idx} has no recorded timestamp or metadata fps; cannot verify scene time")
        report["observation_time_sources"] = sorted({s["time_source"] for s in raw_samples_11})
        report["observation_times"] = [obs.observation_time for obs in observations_11]

        prompt = raw_samples_11[0]["prompt"]
        for idx, s in enumerate(raw_samples_11):
            if s["prompt"] != prompt:
                raise ValueError(f"Sample {idx} prompt mismatch with sample 0")

        # 4. Create main NativeCacheAdapter with use_timestamps=False for legacy A/B/C/D
        config = NativeCacheConfig(
            max_frames=args.history_frames,
            shallow_layer=adapter_shallow_layer(policy),
            use_timestamps=False,
        )
        adapter = NativeCacheAdapter(policy, config=config)

        # Encode 7 frames into blocks (once via ViT) without timestamps
        blocks: List[NativeEmbeddingBlock] = []
        with count_vit_calls(adapter) as vit_counts:
            for obs in observations:
                blk = adapter.encode_frame(images=list(obs.images), frame_id=obs.frame_id, prompt=prompt)
                blocks.append(blk)

        if vit_counts["vit_extract"] != 7:
            raise AssertionError(f"Expected exactly 7 ViT calls during initial encoding, got {vit_counts['vit_extract']}")

        # Track GPU memory after encoding
        mem_after_encode = {
            "allocated_bytes": torch.cuda.memory_allocated(dev_idx),
            "reserved_bytes": torch.cuda.memory_reserved(dev_idx),
        }

        # --- Section A: Single frame parity ---
        sec_a = run_section_a_single_frame(
            policy=policy,
            adapter=adapter,
            block0=blocks[0],
            obs0=observations[0],
            prompt=prompt,
            seed=args.seed,
            flow_steps=args.flow_steps,
        )
        report["sections"]["section_a_single_frame"] = sec_a

        # --- Section B: History incremental vs fresh replay ---
        sec_b = run_section_b_history(
            adapter=adapter,
            blocks=blocks,
            prompt=prompt,
        )
        report["sections"]["section_b_history"] = sec_b

        # --- Section C: Eviction & rebuild (W=2) ---
        sec_c = run_section_c_eviction(
            policy=policy,
            blocks=blocks,
            prompt=prompt,
        )
        report["sections"]["section_c_eviction"] = sec_c

        # --- Section D: Asynchronous pipeline ---
        sec_d = run_section_d_async(
            adapter=adapter,
            observations=observations,
            prompt=prompt,
            flow_steps=args.flow_steps,
        )
        report["sections"]["section_d_async"] = sec_d
        for section, count_key in ((sec_b, "frame_count"), (sec_d, "total_frames_committed")):
            if (section[count_key], section["total_seq_len"], section["kv_bytes"]) != (7, 7168, 392 * 1024**2):
                raise AssertionError("Expected seven native 1024-token blocks with 392 MiB of BF16 KV")
        report["initial_encode_counts"] = vit_counts

        # --- Section E: Dynamic Snapshot & Timestamps (1 + 8 + 2 frames) ---
        config_e = NativeCacheConfig(
            max_frames=4,
            shallow_layer=adapter_shallow_layer(policy),
            use_timestamps=True,
        )
        adapter_e = NativeCacheAdapter(policy, config=config_e)
        sec_e = run_section_e_dynamic_timestamps(
            adapter=adapter_e,
            observations=observations_11,
            prompt=prompt,
            flow_steps=args.flow_steps,
        )
        report["sections"]["section_e_dynamic_timestamps"] = sec_e

        peak_allocated = torch.cuda.max_memory_allocated(dev_idx)
        peak_reserved = torch.cuda.max_memory_reserved(dev_idx)

        all_passed = bool(
            sec_a.get("passed", False)
            and sec_b.get("passed", False)
            and sec_c.get("passed", False)
            and sec_d.get("passed", False)
            and sec_e.get("passed", False)
        )

        report["gpu_memory"] = {
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_mb": round(peak_allocated / (1024 * 1024), 2),
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_mb": round(peak_reserved / (1024 * 1024), 2),
            "post_encode": mem_after_encode,
        }

        report["summary"] = {
            "all_sections_passed": all_passed,
            "section_a_passed": sec_a["passed"],
            "section_b_passed": sec_b["passed"],
            "section_c_passed": sec_c["passed"],
            "section_d_passed": sec_d["passed"],
            "section_e_passed": sec_e["passed"],
            "trainable_parameters": 0,
            "history_window_capacity": args.history_frames,
            "tested_frames_count": 11,
            "pure_kv_megabytes": sec_b.get("kv_megabytes"),
        }

        report_file = out_dir / "report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Verification completed successfully. Report saved to {report_file}")

    except Exception as e:
        err_report = {
            "error": str(e),
            "partial_report": report,
        }
        err_file = out_dir / "error.json"
        with open(err_file, "w") as f:
            json.dump(err_report, f, indent=2, default=str)
        raise


if __name__ == "__main__":
    main()
