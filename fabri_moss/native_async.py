"""Native KV Cache Asynchronous Execution and Pipeline Integration for FabriVLA.

Provides:
- validate_native_memory: Strict validator for NativeKVState in AsyncVisualPlanner.
- make_native_cache_callbacks: Factory returning (encode, plan, validate) bound to prompt and model.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from fabri_moss.async_pipeline import EncodedFrame, Observation, PlanComputation
from fabri_moss.native_cache import NativeCacheAdapter, NativeEmbeddingBlock, NativeKVState


def validate_native_memory(
    candidate: Any,
    previous: Optional[Any],
    snapshot: Tuple[EncodedFrame, ...],
    prompt: str,
) -> Any:
    if candidate is None:
        raise TypeError("candidate next_memory cannot be None for native cache stateful planner")
    if not isinstance(candidate, NativeKVState):
        raise TypeError(f"next_memory must be NativeKVState, got {type(candidate).__name__}")
    if previous is not None and not isinstance(previous, NativeKVState):
        raise TypeError(f"previous state must be NativeKVState, got {type(previous).__name__}")

    owner = candidate.owner
    if not isinstance(owner, NativeCacheAdapter):
        raise TypeError(f"candidate.owner must be NativeCacheAdapter, got {type(owner).__name__}")
    if candidate.revision != owner.revision:
        raise ValueError(f"candidate revision {candidate.revision} does not match owner revision {owner.revision}")
    if candidate.prompt != prompt:
        raise ValueError(f"next_memory prompt '{candidate.prompt}' does not match active prompt '{prompt}'")

    if not snapshot:
        raise ValueError("snapshot cannot be empty")

    for idx, frame in enumerate(snapshot):
        blk = frame.payload
        if not isinstance(blk, NativeEmbeddingBlock):
            raise TypeError(f"snapshot[{idx}].payload must be NativeEmbeddingBlock, got {type(blk).__name__}")
        if blk.frame_id != frame.observation.frame_id:
            raise ValueError(
                f"snapshot[{idx}].payload frame_id {blk.frame_id} != observation frame_id {frame.observation.frame_id}"
            )
        if idx > 0 and blk.frame_id <= snapshot[idx - 1].payload.frame_id:
            raise ValueError("snapshot payload frame IDs must be strictly increasing")
        if blk.owner is not owner:
            raise ValueError(f"snapshot[{idx}].payload owner does not match candidate owner")
        if blk.revision != owner.revision:
            raise ValueError(f"snapshot[{idx}].payload revision {blk.revision} does not match owner revision {owner.revision}")
        if blk.prompt != prompt:
            raise ValueError(f"snapshot[{idx}].payload prompt '{blk.prompt}' does not match active prompt '{prompt}'")

    if previous is not None:
        if previous.owner is not owner:
            raise ValueError("previous owner does not match candidate owner")
        if previous.revision != owner.revision:
            raise ValueError("previous revision does not match owner revision")
        if previous.prompt != prompt:
            raise ValueError("previous prompt does not match active prompt")
        if snapshot[0].payload.frame_id <= previous.last_frame_id:
            raise ValueError(
                f"snapshot[0] frame_id {snapshot[0].payload.frame_id} must be > previous last_frame_id {previous.last_frame_id}"
            )

    old_count = previous.frame_count if previous is not None else 0
    expected_count = old_count + len(snapshot)
    if candidate.frame_count != expected_count:
        raise ValueError(f"next_memory frame_count mismatch: expected {expected_count}, got {candidate.frame_count}")

    last_snapshot_id = snapshot[-1].observation.frame_id
    if candidate.last_frame_id != last_snapshot_id:
        raise ValueError(
            f"next_memory last_frame_id must equal snapshot latest frame {last_snapshot_id}, got {candidate.last_frame_id}"
        )

    prev_blocks = previous.blocks if previous is not None else ()
    snapshot_payloads = tuple(f.payload for f in snapshot)
    all_blocks = prev_blocks + snapshot_payloads
    max_w = owner.config.max_frames
    expected_blocks = all_blocks[-max_w:]

    # Compute expected rebuild count across consecutive groups of size <= max_w
    n = len(snapshot)
    cur_len = len(prev_blocks)
    present = previous is not None
    computed_rebuild = previous.rebuild_count if previous is not None else 0
    for i in range(0, n, max_w):
        g = min(max_w, n - i)
        if present and cur_len + g > max_w:
            computed_rebuild += 1
        cur_len = min(max_w, cur_len + g)
        present = True

    expected_rebuild = computed_rebuild
    if candidate.rebuild_count != expected_rebuild:
        raise ValueError(
            f"candidate rebuild_count {candidate.rebuild_count} mismatch with expected {expected_rebuild}"
        )

    if len(candidate.blocks) != len(expected_blocks):
        raise ValueError(
            f"candidate blocks length {len(candidate.blocks)} != expected {len(expected_blocks)}"
        )

    # Validate timestamps
    use_timestamps = getattr(owner.config, "use_timestamps", True)
    if use_timestamps:
        last_time = previous.blocks[-1].observation_time if (previous is not None and previous.blocks) else None
        for idx, frame in enumerate(snapshot):
            obs = frame.observation
            blk = frame.payload
            if obs.observation_time is None or blk.observation_time is None:
                raise ValueError(f"snapshot[{idx}] observation_time cannot be None when use_timestamps=True")
            if obs.observation_time != blk.observation_time:
                raise ValueError(
                    f"snapshot[{idx}] payload observation_time {blk.observation_time} != obs {obs.observation_time}"
                )
            if obs.capture_time != blk.capture_time:
                raise ValueError(
                    f"snapshot[{idx}] payload capture_time {blk.capture_time} != obs {obs.capture_time}"
                )
            if last_time is not None and obs.observation_time < last_time:
                raise ValueError(
                    f"snapshot[{idx}] observation_time {obs.observation_time} must be non-decreasing from previous {last_time}"
                )
            last_time = obs.observation_time
    else:
        for idx, frame in enumerate(snapshot):
            obs = frame.observation
            blk = frame.payload
            if blk.observation_time is not None:
                if obs.observation_time != blk.observation_time:
                    raise ValueError(
                        f"snapshot[{idx}] payload observation_time {blk.observation_time} != obs {obs.observation_time}"
                    )
            if blk.capture_time is not None:
                if obs.capture_time != blk.capture_time:
                    raise ValueError(
                        f"snapshot[{idx}] payload capture_time {blk.capture_time} != obs {obs.capture_time}"
                    )

    for idx, (cand_blk, exp_blk) in enumerate(zip(candidate.blocks, expected_blocks)):
        if not isinstance(cand_blk, NativeEmbeddingBlock):
            raise TypeError(f"candidate.blocks[{idx}] must be NativeEmbeddingBlock, got {type(cand_blk).__name__}")
        if cand_blk.frame_id != exp_blk.frame_id:
            raise ValueError(f"candidate.blocks[{idx}] frame_id {cand_blk.frame_id} != expected {exp_blk.frame_id}")
        if cand_blk.capture_time != exp_blk.capture_time:
            raise ValueError(
                f"candidate.blocks[{idx}] capture_time {cand_blk.capture_time} != expected {exp_blk.capture_time}"
            )
        if cand_blk.observation_time != exp_blk.observation_time:
            raise ValueError(
                f"candidate.blocks[{idx}] observation_time {cand_blk.observation_time} != expected {exp_blk.observation_time}"
            )
        if cand_blk.owner is not owner:
            raise ValueError(f"candidate.blocks[{idx}] owner does not match candidate owner")
        if cand_blk.revision != owner.revision:
            raise ValueError(f"candidate.blocks[{idx}] revision mismatch with owner")
        if cand_blk.prompt != prompt:
            raise ValueError(f"candidate.blocks[{idx}] prompt mismatch with active prompt")
        if cand_blk.inputs_embeds.shape != exp_blk.inputs_embeds.shape:
            raise ValueError(f"candidate.blocks[{idx}] inputs_embeds shape mismatch")
        if cand_blk.inputs_embeds.dtype != exp_blk.inputs_embeds.dtype:
            raise TypeError(f"candidate.blocks[{idx}] inputs_embeds dtype mismatch")
        if cand_blk.inputs_embeds.device != exp_blk.inputs_embeds.device:
            raise ValueError(f"candidate.blocks[{idx}] inputs_embeds device mismatch")
        if cand_blk.inputs_embeds.data_ptr() != exp_blk.inputs_embeds.data_ptr():
            raise ValueError(f"candidate.blocks[{idx}] inputs_embeds must share storage with expected block")
        if cand_blk.inputs_embeds.stride() != exp_blk.inputs_embeds.stride():
            raise ValueError(f"candidate.blocks[{idx}] inputs_embeds stride mismatch")

        if cand_blk.attention_mask.shape != exp_blk.attention_mask.shape:
            raise ValueError(f"candidate.blocks[{idx}] attention_mask shape mismatch")
        if cand_blk.attention_mask.dtype != exp_blk.attention_mask.dtype:
            raise TypeError(f"candidate.blocks[{idx}] attention_mask dtype mismatch")
        if cand_blk.attention_mask.device != exp_blk.attention_mask.device:
            raise ValueError(f"candidate.blocks[{idx}] attention_mask device mismatch")
        if cand_blk.attention_mask.data_ptr() != exp_blk.attention_mask.data_ptr():
            raise ValueError(f"candidate.blocks[{idx}] attention_mask must share storage with expected block")
        if cand_blk.attention_mask.stride() != exp_blk.attention_mask.stride():
            raise ValueError(f"candidate.blocks[{idx}] attention_mask stride mismatch")

    total_seq_len = sum(b.seq_len for b in candidate.blocks)
    expected_mask = torch.cat([b.attention_mask for b in candidate.blocks], dim=1)
    if candidate.attention_mask.shape != (1, total_seq_len):
        raise ValueError(
            f"candidate attention_mask shape {tuple(candidate.attention_mask.shape)} != (1, {total_seq_len})"
        )
    if candidate.attention_mask.device != expected_mask.device:
        raise ValueError("candidate attention_mask device mismatch")
    if candidate.attention_mask.dtype != expected_mask.dtype:
        raise TypeError("candidate attention_mask dtype mismatch")
    if not torch.equal(candidate.attention_mask, expected_mask):
        raise ValueError("candidate attention_mask content does not match concatenated block masks")

    native_core = owner.native_core
    layers = getattr(native_core, "layers", None)
    if layers is None:
        raise AttributeError("owner.native_core must have 'layers' attribute")
    if len(candidate.layer_kv) != len(layers):
        raise ValueError(f"candidate layer_kv count {len(candidate.layer_kv)} != layers count {len(layers)}")

    core_param = next(native_core.parameters())
    for block in candidate.blocks:
        if block.inputs_embeds.dtype != core_param.dtype or block.inputs_embeds.device != core_param.device:
            raise ValueError("Block embeddings must match native_core dtype and device")
        if block.attention_mask.device != core_param.device:
            raise ValueError("Block masks must match native_core device")
    for idx, (kv, layer) in enumerate(zip(candidate.layer_kv, layers)):
        if not isinstance(kv, (tuple, list)) or len(kv) != 2:
            raise ValueError(f"layer_kv[{idx}] must be a (K, V) pair")
        k, v = kv
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError(f"layer_kv[{idx}] K and V must be torch.Tensor")
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"layer_kv[{idx}] K and V must be 4D [1, H, S, D]")

        attn = getattr(layer, "self_attn", layer)
        k_proj = getattr(attn, "k_proj", None)
        if k_proj is None:
            raise AttributeError(f"layer[{idx}] attention has no k_proj")
        head_dim = getattr(attn, "head_dim", 128)
        if k_proj.out_features % head_dim != 0:
            raise ValueError(f"layer[{idx}] k_proj out_features not divisible by head_dim")
        expected_heads = k_proj.out_features // head_dim
        expected_shape = (1, expected_heads, total_seq_len, head_dim)

        if k.shape != expected_shape or v.shape != expected_shape:
            raise ValueError(
                f"layer_kv[{idx}] shape K:{tuple(k.shape)}, V:{tuple(v.shape)} mismatch with expected {expected_shape}"
            )
        if k.dtype != core_param.dtype or v.dtype != core_param.dtype:
            raise TypeError(f"layer_kv[{idx}] dtype must match native_core {core_param.dtype}")
        if k.device != core_param.device or v.device != core_param.device:
            raise ValueError(f"layer_kv[{idx}] device must match native_core {core_param.device}")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError(f"layer_kv[{idx}] contains non-finite values")

    return candidate.detached()


def make_native_cache_callbacks(
    model: NativeCacheAdapter,
    prompt: str,
) -> Tuple[
    Callable[[Observation], Any],
    Callable[..., PlanComputation],
    Callable[[], None],
]:
    """Factory creating (encode, plan, validate) callbacks bound to an official prompt and model."""
    if not isinstance(model, NativeCacheAdapter):
        raise TypeError(f"model must be NativeCacheAdapter, got {type(model).__name__}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    bound_prompt = prompt.strip()

    initial_revision = model.revision
    device = next(model.parameters()).device
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

    def validate() -> None:
        if model.training or getattr(model.policy, "training", False):
            raise RuntimeError("Model and policy must be in eval mode for native cache async pipeline")
        current_rev = model.revision
        if current_rev != initial_revision:
            raise RuntimeError(
                f"Model revision changed from {initial_revision} to {current_rev}; pipeline invalidated"
            )

    def encode(obs: Observation) -> Any:
        validate()
        with torch.no_grad():
            if use_cuda and vision_stream is not None:
                with torch.cuda.stream(vision_stream):
                    block = model.encode_frame(
                        images=list(obs.images),
                        frame_id=obs.frame_id,
                        prompt=bound_prompt,
                        capture_time=obs.capture_time,
                        observation_time=obs.observation_time,
                    )
                    completion_event = torch.cuda.Event()
                    completion_event.record(vision_stream)
                    completion_event.synchronize()
            else:
                block = model.encode_frame(
                    images=list(obs.images),
                    frame_id=obs.frame_id,
                    prompt=bound_prompt,
                    capture_time=obs.capture_time,
                    observation_time=obs.observation_time,
                )
        return block

    def plan(
        frames: Tuple[EncodedFrame, ...],
        plan_prompt: str,
        previous_memory: Optional[Any] = None,
    ) -> PlanComputation:
        validate()
        if plan_prompt != bound_prompt:
            raise ValueError(
                f"Plan prompt '{plan_prompt}' does not match bound factory prompt '{bound_prompt}'"
            )

        latest_obs = frames[-1].observation
        new_blocks = [f.payload for f in frames]

        with torch.no_grad():
            if use_cuda and planner_stream is not None:
                with torch.cuda.stream(planner_stream):
                    for blk in new_blocks:
                        blk.inputs_embeds.record_stream(planner_stream)
                        blk.attention_mask.record_stream(planner_stream)
                        if getattr(blk, "visual_tokens", None) is not None:
                            blk.visual_tokens.record_stream(planner_stream)

                    if previous_memory is not None:
                        from fabri_moss.compact_cache import CompactKVState
                        from fabri_moss.memory_cache import NativeMemoryKVState

                        if isinstance(previous_memory, CompactKVState):
                            previous_memory.record_stream(planner_stream)
                        else:
                            for blk in previous_memory.blocks:
                                blk.inputs_embeds.record_stream(planner_stream)
                                blk.attention_mask.record_stream(planner_stream)
                                if getattr(blk, "visual_tokens", None) is not None:
                                    blk.visual_tokens.record_stream(planner_stream)
                            for k, v in previous_memory.layer_kv:
                                k.record_stream(planner_stream)
                                v.record_stream(planner_stream)

                            if isinstance(previous_memory, NativeMemoryKVState):
                                for entry in previous_memory.memory.entries:
                                    entry.visual_tokens.record_stream(planner_stream)
                                for anchor in previous_memory.memory.anchors:
                                    anchor.inputs_embeds.record_stream(planner_stream)
                                    anchor.attention_mask.record_stream(planner_stream)
                                    anchor.visual_tokens.record_stream(planner_stream)

                    state_dev = latest_obs.state.to(device)
                    state_mask_dev = latest_obs.state_mask.to(device)
                    action_mask_dev = latest_obs.action_mask.to(device)

                    deep, shallow, next_state = model.read_blocks(
                        new_blocks=new_blocks,
                        prompt=bound_prompt,
                        previous=previous_memory,
                    )

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
                    next_memory_candidate = next_state.detached() if next_state is not None else None
            else:
                state_cpu = latest_obs.state
                state_mask_cpu = latest_obs.state_mask
                action_mask_cpu = latest_obs.action_mask

                deep, shallow, next_state = model.read_blocks(
                    new_blocks=new_blocks,
                    prompt=bound_prompt,
                    previous=previous_memory,
                )

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
                next_memory_candidate = next_state.detached() if next_state is not None else None

        return PlanComputation(
            actions=actions_cpu,
            deep=deep_cpu,
            shallow=shallow_cpu,
            next_memory=next_memory_candidate,
        )

    return encode, plan, validate
