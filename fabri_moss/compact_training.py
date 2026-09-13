"""Compact memory-augmented sequence policy for FabriVLA / Moss.

Implements NativeCompactMemorySequencePolicy inheriting from NativeMemorySequencePolicy,
supporting compact non-decision historical observation pooling, temporal RoPE position embeddings,
post-query consolidation lifecycle, and differentiable execution while maintaining
backward compatibility with dense replay and standard samples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fabri_moss.compact_memory import (
    CompactMemoryConfig,
    RenderedMemory,
    cold_prefix_end,
    render_compact_memory,
)
from fabri_moss.memory_training import NativeMemorySequencePolicy
from fabri_moss.native_cache import execute_native_layers
from fabri_moss.native_training import _autocast_context
from fabri_moss.periodic_memory import (
    MemoryFrame,
    PeriodicMemoryState,
    append_memory_frame,
    consolidate_memory,
)


class NativeCompactMemorySequencePolicy(NativeMemorySequencePolicy):
    """Sequence policy with compact intermediate memory, temporal RoPE, and post-query consolidation.

    Inherits from NativeMemorySequencePolicy.
    - If memory_replay is False/absent, delegates directly to superclass (which handles dense / current-only).
    - Under memory_replay:
        - Strict validation of decision_indices and target_indices before running ViT.
        - Cold prefix (before first target's retired threshold) is encoded in no_grad mini-batches (size 8).
        - Remaining frames are differentiably encoded with gradients.
        - Frames are added via append_memory_frame.
        - If a frame is a decision:
            - If it is a target, first render_compact_memory, execute_native_layers with token_times
              and temporal_config, slice the full current block for action head, THEN consolidate_memory.
            - If it is an un-supervised prefix decision, consolidate_memory without running LLM.
        - No added parameters or state.
    """

    def __init__(
        self,
        policy: nn.Module,
        shallow_layer: int = 6,
        use_timestamps: bool = True,
        gradient_checkpointing: bool = True,
        compact_config: Optional[CompactMemoryConfig] = None,
    ):
        if not use_timestamps:
            raise ValueError("use_timestamps=False is strictly forbidden for NativeCompactMemorySequencePolicy; deployment requires timestamps")

        compact_cfg = compact_config if compact_config is not None else CompactMemoryConfig()
        if not isinstance(compact_cfg, CompactMemoryConfig):
            raise TypeError(f"compact_config must be a CompactMemoryConfig, got {type(compact_cfg).__name__}")

        super().__init__(
            policy=policy,
            shallow_layer=shallow_layer,
            use_timestamps=use_timestamps,
            gradient_checkpointing=gradient_checkpointing,
            memory_config=compact_cfg.memory,
        )
        self.compact_config = compact_cfg

    def features(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute deep and shallow features for target frames in sample."""
        # Delegate non-memory_replay samples directly to superclass (preserves dense / current-only)
        if not sample.get("memory_replay", False):
            return super().features(sample)

        # Under memory_replay, replay_groups is forbidden
        if sample.get("replay_groups", None) is not None:
            raise ValueError("memory_replay cannot be combined with replay_groups")

        # Strict validation of decision_indices before ViT encoding
        if "decision_indices" not in sample:
            raise ValueError("sample must contain 'decision_indices' when memory_replay is True")
        decision_indices = sample["decision_indices"]
        if not isinstance(decision_indices, (list, tuple)):
            raise TypeError(
                f"decision_indices must be a list or tuple of integers, got {type(decision_indices).__name__}"
            )
        if len(decision_indices) == 0:
            raise ValueError("decision_indices cannot be empty")

        # Validate sample structure using parent validation
        N, M = self._validate_sample(sample)

        # Verify decision_indices: non-bool ints, strictly increasing, in [0, N)
        for idx, d_idx in enumerate(decision_indices):
            if isinstance(d_idx, bool) or not isinstance(d_idx, int):
                raise TypeError(
                    f"decision_indices must contain non-bool integers, got {type(d_idx).__name__} at index {idx}"
                )
            if d_idx < 0 or d_idx >= N:
                raise IndexError(f"decision index {d_idx} out of bounds [0, {N})")
            if idx > 0 and d_idx <= decision_indices[idx - 1]:
                raise ValueError(
                    f"decision_indices must be strictly increasing: index {idx} ({d_idx}) <= previous ({decision_indices[idx - 1]})"
                )

        target_indices = sample["target_indices"]
        decision_set = set(decision_indices)
        for t_idx in target_indices:
            if t_idx not in decision_set:
                raise ValueError(
                    f"target_indices must be a subset of decision_indices, but target {t_idx} is missing from decision_indices"
                )

        if decision_indices[-1] != target_indices[-1]:
            raise ValueError(
                f"last decision index ({decision_indices[-1]}) must equal last target index ({target_indices[-1]}), no post-target decisions allowed"
            )

        images_window = sample["images_window"]
        frame_ids = sample["frame_ids"]
        observation_times = sample["observation_times"]
        prompt = sample["prompt"]
        image_masks = sample.get("image_masks", None)

        tokens_per_frame = getattr(getattr(self.policy, "embedder", None), "max_text_length", 1024)
        core = self.native_core
        num_layers = len(core.layers)
        do_grad_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()

        # Compute cold prefix boundary using contract: cold_prefix_end
        first_target = target_indices[0]
        cold_end = cold_prefix_end(
            decision_indices=decision_indices,
            first_target=first_target,
            memory_config=self.compact_config.memory,
        )

        state = PeriodicMemoryState()
        embedder = getattr(self.policy, "embedder", None)
        if embedder is None:
            raise AttributeError("policy must have an 'embedder' attribute")

        # 1. Process cold frames in mini-batches of 8 under torch.no_grad()
        if cold_end > 0:
            batch_size = 8
            with torch.no_grad():
                for start_idx in range(0, cold_end, batch_size):
                    end_idx = min(start_idx + batch_size, cold_end)
                    b_imgs = images_window[start_idx:end_idx]
                    b_fids = frame_ids[start_idx:end_idx]
                    b_times = observation_times[start_idx:end_idx]
                    b_masks = image_masks[start_idx:end_idx] if image_masks is not None else None

                    b_seq_embeds, b_seq_mask, b_vit_embeds = self._encode_sequence_embeddings(
                        images_window=b_imgs,
                        frame_ids=b_fids,
                        observation_times=b_times,
                        prompt=prompt,
                        image_masks=b_masks,
                        return_visual_tokens=True,
                    )
                    b_len = end_idx - start_idx
                    h_dim = b_seq_embeds.shape[-1]
                    b_embeds_split = b_seq_embeds.view(b_len, tokens_per_frame, h_dim)
                    b_mask_split = b_seq_mask.view(b_len, tokens_per_frame)
                    b_vit_split = b_vit_embeds.to(dtype=b_seq_embeds.dtype)

                    for i in range(b_len):
                        f_idx = start_idx + i
                        is_dec = f_idx in decision_set
                        frame = MemoryFrame(
                            frame_id=b_fids[i],
                            observation_time=b_times[i],
                            inputs_embeds=b_embeds_split[i : i + 1].detach(),
                            attention_mask=b_mask_split[i : i + 1].detach(),
                            visual_tokens=b_vit_split[i : i + 1].detach(),
                            is_decision=is_dec,
                            boundary_frame_id=None,
                        )
                        state = append_memory_frame(state, frame)
                        if is_dec:
                            state = consolidate_memory(state, self.compact_config.memory)

        # 2. Differentiably encode remaining frames from cold_end to N
        if cold_end < N:
            rem_imgs = images_window[cold_end:N]
            rem_fids = frame_ids[cold_end:N]
            rem_times = observation_times[cold_end:N]
            rem_masks = image_masks[cold_end:N] if image_masks is not None else None

            rem_seq_embeds, rem_seq_mask, rem_vit_embeds = self._encode_sequence_embeddings(
                images_window=rem_imgs,
                frame_ids=rem_fids,
                observation_times=rem_times,
                prompt=prompt,
                image_masks=rem_masks,
                return_visual_tokens=True,
            )
            rem_len = N - cold_end
            h_dim = rem_seq_embeds.shape[-1]
            rem_embeds_split = rem_seq_embeds.view(rem_len, tokens_per_frame, h_dim)
            rem_mask_split = rem_seq_mask.view(rem_len, tokens_per_frame)
            rem_vit_split = rem_vit_embeds.to(dtype=rem_seq_embeds.dtype)

            target_set = set(target_indices)
            deep_targets: List[torch.Tensor] = []
            shallow_targets: List[torch.Tensor] = []

            for i in range(rem_len):
                f_idx = cold_end + i
                is_dec = f_idx in decision_set
                is_tgt = f_idx in target_set

                frame = MemoryFrame(
                    frame_id=rem_fids[i],
                    observation_time=rem_times[i],
                    inputs_embeds=rem_embeds_split[i : i + 1],
                    attention_mask=rem_mask_split[i : i + 1],
                    visual_tokens=rem_vit_split[i : i + 1],
                    is_decision=is_dec,
                    boundary_frame_id=None,
                )
                state = append_memory_frame(state, frame)

                if is_tgt:
                    # Target decision frame: render compact memory BEFORE consolidation
                    rendered: RenderedMemory = render_compact_memory(
                        state=state,
                        embedder=embedder,
                        config=self.compact_config,
                    )
                    device = rendered.inputs_embeds.device

                    with _autocast_context(device, enabled=True):
                        final_norm_h, inter_h = execute_native_layers(
                            core=core,
                            inputs_embeds=rendered.inputs_embeds,
                            attention_mask_2d=rendered.attention_mask,
                            cache=None,
                            start_pos=0,
                            gradient_checkpointing=do_grad_checkpoint,
                            token_times=rendered.token_times,
                            temporal_config=self.compact_config.temporal,
                        )

                    deep_full = final_norm_h
                    if self.shallow_layer == num_layers:
                        shallow_full = final_norm_h
                    else:
                        shallow_full = inter_h[self.shallow_layer]
                    del inter_h

                    # Slicing full current block (tokens_per_frame=1024)
                    c_start = rendered.current_start
                    deep_target = deep_full[:, c_start : c_start + tokens_per_frame, :]
                    shallow_target = shallow_full[:, c_start : c_start + tokens_per_frame, :]

                    deep_targets.append(deep_target.to(torch.float32))
                    shallow_targets.append(shallow_target.to(torch.float32))

                # Consolidate after decision
                if is_dec:
                    state = consolidate_memory(state, self.compact_config.memory)

            deep_selected = torch.cat(deep_targets, dim=0)
            shallow_selected = torch.cat(shallow_targets, dim=0)
            return deep_selected, shallow_selected
        else:
            raise ValueError(f"cold_end {cold_end} >= N {N}, no target frames to evaluate")
