"""Predictive Memory Policy for FabriVLA.

Implements PredictiveMemoryPolicy inheriting from NativeSequencePolicy,
with frozen backbone, differentiable CausalMemoryWriter, and FutureLatentHead.
Consumes predictive_memory contract:
  WriterConfig, CausalMemoryWriter, consolidate_predictive, render_predictive_memory,
  append_predictive_frame
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.native_cache import execute_native_layers
from fabri_moss.native_training import NativeSequencePolicy, _autocast_context
from fabri_moss.periodic_memory import MemoryFrame, PeriodicMemoryState, detached_state
from fabri_moss.predictive_memory import (
    CausalMemoryWriter,
    WriterConfig,
    append_predictive_frame,
    consolidate_predictive,
    render_predictive_memory,
)
from fabri_moss.temporal_rope import TemporalRoPEConfig


class FutureLatentHead(nn.Module):
    """Lightweight future latent prediction head.

    Fuses deep and shallow features from the current context, adds learned queries
    modulated by scalar dt sinusoidal embeddings, queries context via MHA,
    and predicts future pooled visual latents.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_queries: int = 16,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Linear projection fusing deep and shallow features
        self.context_proj = nn.Linear(input_dim * 2, hidden_dim)

        # Learned spatial queries [num_queries, hidden_dim]
        self.learned_queries = nn.Parameter(
            torch.empty(num_queries, hidden_dim).normal_(mean=0.0, std=0.02)
        )

        # Scalar dt embedding: sin/cos (half and half) + MLP
        # dt is projected to hidden_dim
        self.dt_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Single MHA where queries read context
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output MLP predicting [num_queries, input_dim]
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def _embed_dt(self, dt: torch.Tensor) -> torch.Tensor:
        """Embed scalar dt tensor [*shape] into [*shape, hidden_dim]."""
        # dt: [*shape] float tensor
        half_dim = self.hidden_dim // 2
        # inv_freq: 10000^(-2i / hidden_dim)
        device = dt.device
        dtype = dt.dtype
        indices = torch.arange(half_dim, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000.0 ** (2.0 * indices / self.hidden_dim))

        # outer product with dt
        dt_expanded = dt.unsqueeze(-1)  # [*shape, 1]
        angles = dt_expanded * inv_freq.view(*([1] * (dt.ndim)), half_dim)  # [*shape, half_dim]
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # [*shape, 2 * half_dim]
        if emb.shape[-1] < self.hidden_dim:
            # In case hidden_dim is odd
            pad = torch.zeros(*dt.shape, self.hidden_dim - emb.shape[-1], device=device, dtype=dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return self.dt_mlp(emb)

    def forward(
        self,
        current_deep: torch.Tensor,
        current_shallow: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for FutureLatentHead.

        Args:
            current_deep: [M, context_tokens, input_dim]
            current_shallow: [M, context_tokens, input_dim]
            deltas: [M, K] or [M] float tensor of time offsets

        Returns:
            predicted_latents: [M, K, num_queries, input_dim] (or [M, num_queries, input_dim] if K is 1D)
            normalized along input_dim
        """
        # Ensure FP32
        current_deep = current_deep.float()
        current_shallow = current_shallow.float()
        deltas = deltas.float()

        M, L, D = current_deep.shape
        # Fuse deep and shallow: [M, L, 2*D] -> [M, L, hidden_dim]
        context = torch.cat([current_deep, current_shallow], dim=-1)
        context_emb = self.context_proj(context)

        # Handle deltas shape: [M, K] or [M]
        has_k = deltas.ndim == 2
        if not has_k:
            deltas = deltas.unsqueeze(1)  # [M, 1]
        K = deltas.shape[1]

        # Flatten M and K for batch MHA processing: [M * K, ...]
        dt_flat = deltas.reshape(M * K)  # [M * K]
        dt_emb = self._embed_dt(dt_flat)  # [M * K, hidden_dim]

        # Queries: learned [num_queries, hidden_dim] broadcast to [M * K, num_queries, hidden_dim]
        queries = self.learned_queries.unsqueeze(0).expand(M * K, self.num_queries, self.hidden_dim)
        queries = queries + dt_emb.unsqueeze(1)  # Add time embedding to queries

        # Expand context for M * K: [M, 1, L, hidden_dim] -> [M * K, L, hidden_dim]
        context_expanded = (
            context_emb.unsqueeze(1)
            .expand(M, K, L, self.hidden_dim)
            .reshape(M * K, L, self.hidden_dim)
        )

        # MHA: queries attend to context
        # attn_out: [M * K, num_queries, hidden_dim]
        attn_out, _ = self.mha(query=queries, key=context_expanded, value=context_expanded, need_weights=False)

        # Output MLP and F.layer_norm (no learnable scale/shift to prevent gamma->0 collapse)
        out = self.out_mlp(attn_out)  # [M * K, num_queries, input_dim]
        out_norm = F.layer_norm(out, (self.input_dim,))  # [M * K, num_queries, input_dim]

        # Reshape back
        if has_k:
            pred = out_norm.view(M, K, self.num_queries, self.input_dim)
        else:
            pred = out_norm.view(M, self.num_queries, self.input_dim)
        return pred


class PredictiveMemoryPolicy(NativeSequencePolicy):
    """Predictive memory-augmented sequence policy with frozen base, causal writer, and future latent head."""

    def __init__(
        self,
        policy: nn.Module,
        writer_config: Optional[WriterConfig] = None,
        shallow_layer: int = 6,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            shallow_layer=shallow_layer,
            use_timestamps=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Freeze base policy and set to eval
        self.policy.requires_grad_(False)
        self.policy.eval()

        # Turn off vision gradient checkpointing since encoder is frozen / no_grad
        self._set_vision_gradient_checkpointing(False)

        # Resolve hidden / input dimension from core
        core = self.native_core
        model_dim = getattr(core.config, "hidden_size", 1024)

        if writer_config is None:
            self.writer_config = WriterConfig(input_dim=model_dim)
        else:
            self.writer_config = writer_config

        # Instantiate CausalMemoryWriter and FutureLatentHead on policy device in FP32
        embedder = getattr(self.policy, "embedder", None)
        policy_device = getattr(embedder, "device", next(self.policy.parameters()).device)

        self.writer = CausalMemoryWriter(self.writer_config).to(device=policy_device, dtype=torch.float32)
        for p in self.writer.parameters():
            p.requires_grad = True

        # Target grid and query count from writer_config
        target_grid = self.writer_config.grid
        num_queries = target_grid * target_grid
        writer_hidden = self.writer_config.hidden_dim
        num_heads = self.writer_config.num_heads

        # Instantiate FutureLatentHead on policy device in FP32
        self.future_head = FutureLatentHead(
            input_dim=self.writer_config.input_dim,
            hidden_dim=writer_hidden,
            num_queries=num_queries,
            num_heads=num_heads,
        ).to(device=policy_device, dtype=torch.float32)
        for p in self.future_head.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True) -> PredictiveMemoryPolicy:
        """Override train so base policy always stays in eval mode, while new modules follow mode."""
        super().train(mode)
        # Force base policy to always stay in eval mode
        self.policy.eval()
        # Keep writer and future_head following mode
        self.writer.train(mode)
        self.future_head.train(mode)
        return self

    def _observation_grad_start(self, sample: Dict[str, Any]) -> int:
        """Return frame index where observation gradients start. Defaults to N (all no_grad)."""
        images_window = sample.get("images_window", [])
        return len(images_window)

    def _extract_future_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual features for future targets. Defaults to policy embedder model."""
        embedder = getattr(self.policy, "embedder", None)
        return embedder.model.extract_feature(pixel_values)

    def _encode_observation_frames(
        self,
        sample: Dict[str, Any],
        indices: Sequence[int],
        grad_start: int,
        tokens_per_frame: int,
        return_visual_tokens: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Encode observation frames in chunks of 8, splitting at grad_start for no_grad vs grad."""
        images_window = sample["images_window"]
        frame_ids = sample["frame_ids"]
        observation_times = sample["observation_times"]
        prompt = sample["prompt"]
        image_masks = sample.get("image_masks", None)

        N_sub = len(indices)
        if N_sub == 0:
            return []

        results: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []

        # Find partition index in indices where orig_idx >= grad_start
        split_pos = 0
        while split_pos < N_sub and indices[split_pos] < grad_start:
            split_pos += 1

        # Two chunks: [0:split_pos] (cold / no_grad) and [split_pos:N_sub] (warm / grad if externally enabled)
        partitions = []
        if split_pos > 0:
            partitions.append((0, split_pos, False))
        if split_pos < N_sub:
            partitions.append((split_pos, N_sub, torch.is_grad_enabled()))

        batch_size = 8
        for p_start, p_end, enable_grad in partitions:
            with torch.set_grad_enabled(enable_grad):
                for start_idx in range(p_start, p_end, batch_size):
                    end_idx = min(start_idx + batch_size, p_end)
                    b_sub_indices = indices[start_idx:end_idx]
                    b_imgs = [images_window[i] for i in b_sub_indices]
                    b_fids = [frame_ids[i] for i in b_sub_indices]
                    b_times = [observation_times[i] for i in b_sub_indices]
                    b_masks = [image_masks[i] if image_masks is not None else None for i in b_sub_indices]

                    if return_visual_tokens:
                        b_seq_embeds, b_seq_mask, b_vit_embeds = self._encode_sequence_embeddings(
                            images_window=b_imgs,
                            frame_ids=b_fids,
                            observation_times=b_times,
                            prompt=prompt,
                            image_masks=b_masks,
                            return_visual_tokens=True,
                        )
                    else:
                        b_seq_embeds, b_seq_mask = self._encode_sequence_embeddings(
                            images_window=b_imgs,
                            frame_ids=b_fids,
                            observation_times=b_times,
                            prompt=prompt,
                            image_masks=b_masks,
                            return_visual_tokens=False,
                        )
                        b_vit_embeds = None

                    b_len = end_idx - start_idx
                    h_dim = b_seq_embeds.shape[-1]
                    b_embeds_split = b_seq_embeds.view(b_len, tokens_per_frame, h_dim)
                    b_mask_split = b_seq_mask.view(b_len, tokens_per_frame)
                    if b_vit_embeds is not None:
                        b_vit_split = b_vit_embeds.to(dtype=b_seq_embeds.dtype)
                    else:
                        b_vit_split = None

                    for i in range(b_len):
                        v_tok = b_vit_split[i : i + 1] if b_vit_split is not None else None
                        results.append((b_embeds_split[i : i + 1], b_mask_split[i : i + 1], v_tok))

        return results

    def _render_full_group(
        self,
        group_obs_indices: Sequence[int],
        group_target_positions: Sequence[int],
        sample: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a dense replay group of decisions with textless full frames and temporal RoPE.

        Returns (deep_targets, shallow_targets) for the target positions in this group.
        """
        observation_times = sample["observation_times"]
        target_indices = sample["target_indices"]

        core = self.native_core
        num_layers = len(core.layers)
        tokens_per_frame = getattr(getattr(self.policy, "embedder", None), "max_text_length", 1024)

        g_times = [observation_times[i] for i in group_obs_indices]
        g_N = len(group_obs_indices)

        # Batch 8 sequence embeddings with grad hook
        grad_start = self._observation_grad_start(sample)
        encoded_frames = self._encode_observation_frames(
            sample=sample,
            indices=group_obs_indices,
            grad_start=grad_start,
            tokens_per_frame=tokens_per_frame,
            return_visual_tokens=False,
        )

        batch_embeds_list = [f[0] for f in encoded_frames]
        batch_masks_list = [f[1] for f in encoded_frames]

        seq_inputs_embeds = torch.cat(batch_embeds_list, dim=1)
        seq_attention_mask = torch.cat(batch_masks_list, dim=1)

        # Build token_times tensor
        time_tensors = []
        for t in g_times:
            time_tensors.append(torch.full((1, tokens_per_frame), float(t), dtype=torch.float32))
        token_times = torch.cat(time_tensors, dim=1).to(device=seq_inputs_embeds.device)

        # Execute native layers with temporal RoPE under autocast
        temporal_config = TemporalRoPEConfig()
        device = seq_inputs_embeds.device
        do_grad_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        with _autocast_context(device, enabled=True):
            final_norm_h, inter_h = execute_native_layers(
                core=core,
                inputs_embeds=seq_inputs_embeds,
                attention_mask_2d=seq_attention_mask,
                cache=None,
                start_pos=0,
                gradient_checkpointing=do_grad_checkpoint,
                token_times=token_times,
                temporal_config=temporal_config,
            )

        deep_full = final_norm_h
        if self.shallow_layer == num_layers:
            shallow_full = final_norm_h
        else:
            shallow_full = inter_h[self.shallow_layer]

        deep_frames = deep_full.view(g_N, tokens_per_frame, -1)
        shallow_frames = shallow_full.view(g_N, tokens_per_frame, -1)

        obs_to_rel = {pool_idx: rel_idx for rel_idx, pool_idx in enumerate(group_obs_indices)}
        g_rel_target_indices = [obs_to_rel[target_indices[t_pos]] for t_pos in group_target_positions]
        g_rel_tensor = torch.tensor(g_rel_target_indices, dtype=torch.long, device=deep_full.device)

        g_deep_targets = torch.index_select(deep_frames, 0, g_rel_tensor).to(torch.float32)
        g_shallow_targets = torch.index_select(shallow_frames, 0, g_rel_tensor).to(torch.float32)
        return g_deep_targets, g_shallow_targets

    def _features_dense(self, sample: Dict[str, Any], N: int, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle dense / current-only samples without memory writer using full textless frames + temporal RoPE."""
        replay_groups = sample.get("replay_groups", None)
        if replay_groups is None:
            # Single group with all frames as decisions, querying targets
            group_obs_indices = list(range(N))
            group_target_positions = list(range(M))
            return self._render_full_group(group_obs_indices, group_target_positions, sample)

        # Process each replay group
        deep_target_list: List[Optional[torch.Tensor]] = [None] * M
        shallow_target_list: List[Optional[torch.Tensor]] = [None] * M

        for group in replay_groups:
            obs_indices = group["observation_indices"]
            target_positions = group["target_positions"]
            g_deep, g_shallow = self._render_full_group(obs_indices, target_positions, sample)
            for ti, t_pos in enumerate(target_positions):
                deep_target_list[t_pos] = g_deep[ti]
                shallow_target_list[t_pos] = g_shallow[ti]

        deep_selected = torch.stack(deep_target_list, dim=0)
        shallow_selected = torch.stack(shallow_target_list, dim=0)
        return deep_selected, shallow_selected

    def features(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations once, then replay each decision schedule independently."""
        N, M = self._validate_sample(sample)

        # If not memory_replay, use dense helper
        if not sample.get("memory_replay", False):
            return self._features_dense(sample, N, M)

        groups = sample.get("replay_groups")
        if groups is None:
            if "decision_indices" not in sample:
                raise ValueError("sample must contain 'decision_indices' when memory_replay is True")
            groups = [{
                "observation_indices": list(range(N)),
                "target_positions": list(range(M)),
                "decision_indices": sample["decision_indices"],
            }]

        group_samples = []
        grad_starts = []
        for group in groups:
            if "decision_indices" not in group:
                raise ValueError("memory replay group must contain 'decision_indices'")
            group_sample = dict(
                sample,
                target_indices=[sample["target_indices"][p] for p in group["target_positions"]],
                decision_indices=group["decision_indices"],
            )
            group_sample.pop("replay_groups", None)
            self._validate_memory_decisions(group_sample, group["observation_indices"], N)
            group_samples.append(group_sample)
            grad_starts.append(self._observation_grad_start(group_sample))

        tokens_per_frame = getattr(getattr(self.policy, "embedder", None), "max_text_length", 1024)
        encoded_obs = self._encode_observation_frames(
            sample=sample,
            indices=list(range(N)),
            grad_start=min(grad_starts),
            tokens_per_frame=tokens_per_frame,
            return_visual_tokens=True,
        )

        deep_targets: List[Optional[torch.Tensor]] = [None] * M
        shallow_targets: List[Optional[torch.Tensor]] = [None] * M
        for group, group_sample, grad_start in zip(groups, group_samples, grad_starts):
            deep, shallow = self._render_memory_group(
                group_sample, group["observation_indices"], encoded_obs, grad_start,
            )
            for i, target_position in enumerate(group["target_positions"]):
                deep_targets[target_position] = deep[i]
                shallow_targets[target_position] = shallow[i]
        return torch.stack(deep_targets), torch.stack(shallow_targets)

    @staticmethod
    def _validate_memory_decisions(
        sample: Dict[str, Any], observation_indices: Sequence[int], N: int,
    ) -> None:
        decision_indices = sample["decision_indices"]
        if not isinstance(decision_indices, (list, tuple)):
            raise TypeError(
                f"decision_indices must be a list or tuple of integers, got {type(decision_indices).__name__}"
            )
        if len(decision_indices) == 0:
            raise ValueError("decision_indices cannot be empty")

        observation_set = set(observation_indices)
        for idx, d_idx in enumerate(decision_indices):
            if isinstance(d_idx, bool) or not isinstance(d_idx, int):
                raise TypeError(
                    f"decision_indices must contain non-bool integers, got {type(d_idx).__name__} at index {idx}"
                )
            if d_idx < 0 or d_idx >= N:
                raise IndexError(f"decision index {d_idx} out of bounds [0, {N})")
            if d_idx not in observation_set:
                raise ValueError(f"decision index {d_idx} is missing from group's observation_indices")
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

    def _render_memory_group(
        self,
        sample: Dict[str, Any],
        observation_indices: Sequence[int],
        encoded_obs: Sequence[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        grad_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replay one schedule with its own cold prefix, writer state, and TBPTT boundaries."""
        frame_ids = sample["frame_ids"]
        observation_times = sample["observation_times"]
        decision_indices = sample["decision_indices"]
        target_indices = sample["target_indices"]
        decision_set = set(decision_indices)
        core = self.native_core
        num_layers = len(core.layers)

        # Determine warm_start decision boundary:
        # Prior decisions before the first target:
        first_target = target_indices[0]
        prior_decisions = [d for d in decision_indices if d < first_target]
        num_prior = len(prior_decisions)
        tbptt_k = self.writer_config.tbptt_decisions
        if num_prior <= tbptt_k:
            warm_decision_start = 0
        else:
            warm_decision_start = prior_decisions[num_prior - tbptt_k]

        target_set = set(target_indices)
        deep_target_dict: Dict[int, torch.Tensor] = {}
        shallow_target_dict: Dict[int, torch.Tensor] = {}

        state = PeriodicMemoryState()
        warm_decisions_since_first_target = 0
        do_grad_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        temporal_config = TemporalRoPEConfig()

        # Step through frames
        for f_idx in observation_indices:
            b_embed, b_mask, b_vit = encoded_obs[f_idx]
            # Shared encoding can retain gradients for another group's earlier warm prefix.
            if f_idx < grad_start:
                b_embed = b_embed.detach()
                b_mask = b_mask.detach()
                b_vit = b_vit.detach() if b_vit is not None else None
            frame = MemoryFrame(
                frame_id=frame_ids[f_idx],
                observation_time=observation_times[f_idx],
                inputs_embeds=b_embed,
                attention_mask=b_mask,
                visual_tokens=b_vit,
                is_decision=f_idx in decision_set,
                boundary_frame_id=None,
            )
            state = append_predictive_frame(state, frame)

            if frame.is_decision:
                is_warm = (f_idx >= warm_decision_start)

                if f_idx in target_set:
                    # Target decision: render and execute native layers
                    rendered = render_predictive_memory(state, self.writer_config)
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
                            temporal_config=temporal_config,
                        )

                    # Slice current full frame tokens [1, tokens_per_frame, H]
                    cur_deep = final_norm_h[:, rendered.current_start :, :].to(torch.float32)
                    if self.shallow_layer == num_layers:
                        cur_shallow = final_norm_h[:, rendered.current_start :, :].to(torch.float32)
                    else:
                        cur_shallow = inter_h[self.shallow_layer][:, rendered.current_start :, :].to(torch.float32)

                    deep_target_dict[f_idx] = cur_deep.squeeze(0)  # [tokens_per_frame, H]
                    shallow_target_dict[f_idx] = cur_shallow.squeeze(0)

                # Consolidate memory
                if not is_warm:
                    with torch.no_grad():
                        state = consolidate_predictive(state, self.writer)
                else:
                    state = consolidate_predictive(state, self.writer)
                    if f_idx >= first_target:
                        warm_decisions_since_first_target += 1
                        if warm_decisions_since_first_target >= tbptt_k:
                            state = detached_state(state)
                            warm_decisions_since_first_target = 0

        deep_selected = torch.stack([deep_target_dict[t] for t in target_indices], dim=0)
        shallow_selected = torch.stack([shallow_target_dict[t] for t in target_indices], dim=0)
        return deep_selected, shallow_selected

    @torch.no_grad()
    def _future_targets(
        self,
        sample: Dict[str, Any],
        future_indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and pool normalized future targets from frozen visual encoder under torch.no_grad().

        Returns:
            targets: [M, K, num_queries, input_dim] normalized along input_dim
        """
        if "future_images" not in sample:
            raise KeyError("sample missing 'future_images' when valid future prediction horizons exist")

        future_images = sample["future_images"]
        M, K = future_indices.shape
        total_future = len(future_images)

        target_grid = self.writer_config.grid
        num_queries = target_grid * target_grid
        model_dim = self.writer_config.input_dim
        device = future_indices.device

        targets = torch.zeros(M, K, num_queries, model_dim, dtype=torch.float32, device=device)
        if total_future == 0 or not valid_mask.any():
            return targets

        embedder = getattr(self.policy, "embedder", None)
        emb_device = getattr(embedder, "device", next(self.policy.parameters()).device)

        # Preprocess future images on CPU
        all_frames = []
        for img_elem in future_images:
            if isinstance(img_elem, (list, tuple)):
                all_frames.append(img_elem[0])
            else:
                all_frames.append(img_elem)

        pixel_values, num_tiles_list = embedder._preprocess_images_on_cpu(all_frames)
        if num_tiles_list != [1] * total_future:
            raise ValueError("Future targets require exactly one image tile per frame")

        is_cuda = (emb_device.type == "cuda") if isinstance(emb_device, torch.device) else str(emb_device).startswith("cuda")
        pixel_dtype = torch.bfloat16 if is_cuda else torch.float32
        pixel_values = pixel_values.to(device=emb_device, dtype=pixel_dtype)

        # Batch 8 no_grad feature extraction through frozen visual encoder
        batch_size = 8
        vit_embeds_list = []
        with _autocast_context(emb_device, enabled=True):
            for start_idx in range(0, total_future, batch_size):
                end_idx = min(start_idx + batch_size, total_future)
                b_pixels = pixel_values[start_idx:end_idx]
                b_vit = self._extract_future_features(b_pixels)
                vit_embeds_list.append(b_vit)

        # vit_embeds: [total_future, P, H]
        vit_embeds = torch.cat(vit_embeds_list, dim=0).to(torch.float32)

        # Spatial average pooling to grid x grid (e.g. 4x4 = 16)
        _, p, h = vit_embeds.shape
        if h != model_dim:
            raise ValueError(f"Extracted feature dim {h} does not match writer input_dim {model_dim}")
        s = int(math.isqrt(p))
        if s * s != p:
            raise ValueError(f"visual tokens count P={p} must be a square")

        x = vit_embeds.transpose(1, 2).view(total_future, h, s, s)
        pooled = F.adaptive_avg_pool2d(x, (target_grid, target_grid))
        pooled = pooled.view(total_future, h, num_queries).transpose(1, 2)  # [total_future, num_queries, H]

        # F.layer_norm along hidden dimension
        pooled_norm = F.layer_norm(pooled, (h,)).to(device=device)

        # Assign pooled targets strictly for valid positions
        valid_indices = future_indices[valid_mask]  # 1D tensor
        targets[valid_mask] = pooled_norm[valid_indices]

        return targets

    def forward(
        self,
        sample: Dict[str, Any],
        future_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute action loss and optional future latent prediction loss.

        Args:
            sample: dict with inputs
            future_weight: float weight for future prediction loss. If 0.0, strictly skips futurehead/encoder.

        Returns:
            dict with:
                loss: total mean loss
                loss_sum: total loss sum across targets (action + w_fut) * M
                action_loss: mean action loss
                future_loss: mean future loss (or 0.0 tensor)
                target_count: M
                future_valid_count: count of valid future prediction horizons
        """
        N, M = self._validate_sample(sample)
        actions_gt = sample.get("actions", None)
        if actions_gt is None:
            raise KeyError("sample missing 'actions' for training forward")

        state = sample.get("state", None)
        state_mask = sample.get("state_mask", None)
        action_mask = sample.get("action_mask", None)

        if actions_gt.shape[0] != M:
            raise ValueError(f"actions batch size {actions_gt.shape[0]} != target_indices count M={M}")
        if state is not None and state.shape[0] != M:
            raise ValueError(f"state batch size {state.shape[0]} != target_indices count M={M}")
        if state_mask is not None and state_mask.shape[0] != M:
            raise ValueError(f"state_mask batch size {state_mask.shape[0]} != target_indices count M={M}")
        if action_mask is not None and action_mask.shape[0] != M:
            raise ValueError(f"action_mask batch size {action_mask.shape[0]} != target_indices count M={M}")

        # Deep and shallow features for targets
        deep, shallow = self.features(sample)
        device = deep.device

        # Compute action loss via policy.action_head in FP32 without autocast
        ah_config = getattr(self.policy.action_head, "config", None)
        shallow_fusion = getattr(ah_config, "shallow_fusion", "none")
        pass_shallow = shallow if shallow_fusion != "none" else None

        with _autocast_context(device, enabled=False):
            action_out = self.policy.action_head(
                fused_tokens=deep,
                state=state,
                actions_gt=actions_gt,
                action_mask=action_mask,
                state_mask=state_mask,
                shallow_tokens=pass_shallow,
            )

        if not isinstance(action_out, dict) or "loss" not in action_out:
            raise TypeError(f"Expected action_head to return dict with 'loss', got {type(action_out)}")

        action_loss = action_out["loss"]

        # Validate future_weight: strictly non-negative finite float/int, not bool
        if isinstance(future_weight, bool) or not isinstance(future_weight, (int, float)):
            raise TypeError(f"future_weight must be a finite float or int >= 0, got {type(future_weight).__name__}")
        if not math.isfinite(future_weight) or future_weight < 0.0:
            raise ValueError(f"future_weight must be finite and >= 0.0, got {future_weight}")

        # Future loss computation: strictly forbidden when future_weight == 0.0
        future_loss = torch.zeros((), dtype=torch.float32, device=device)
        future_valid_count = 0

        if future_weight > 0.0:
            if "future_indices" not in sample or "future_deltas" not in sample or "future_valid" not in sample:
                raise KeyError(
                    "sample must contain 'future_indices', 'future_deltas', and 'future_valid' when future_weight > 0"
                )

            future_indices = sample["future_indices"]
            future_deltas = sample["future_deltas"]
            future_valid = sample["future_valid"]

            if not isinstance(future_indices, torch.Tensor):
                future_indices = torch.tensor(future_indices, dtype=torch.long, device=device)
            else:
                if future_indices.dtype != torch.long:
                    raise TypeError(f"future_indices must have dtype torch.long, got {future_indices.dtype}")
                future_indices = future_indices.to(device=device)

            if not isinstance(future_deltas, torch.Tensor):
                future_deltas = torch.tensor(future_deltas, dtype=torch.float32, device=device)
            else:
                if future_deltas.dtype not in (torch.float32, torch.float64):
                    raise TypeError(f"future_deltas must have float dtype, got {future_deltas.dtype}")
                future_deltas = future_deltas.to(device=device, dtype=torch.float32)

            if not isinstance(future_valid, torch.Tensor):
                future_valid = torch.tensor(future_valid, dtype=torch.bool, device=device)
            else:
                if future_valid.dtype != torch.bool:
                    raise TypeError(f"future_valid must have dtype torch.bool, got {future_valid.dtype}")
                future_valid = future_valid.to(device=device)

            # Shapes must match [M, K]
            if future_indices.ndim != 2 or future_indices.shape[0] != M:
                raise ValueError(f"future_indices shape {tuple(future_indices.shape)} does not match (M={M}, K)")
            K = future_indices.shape[1]
            if future_deltas.shape != (M, K):
                raise ValueError(f"future_deltas shape {tuple(future_deltas.shape)} does not match (M={M}, K={K})")
            if future_valid.shape != (M, K):
                raise ValueError(f"future_valid shape {tuple(future_valid.shape)} does not match (M={M}, K={K})")

            # Check future_images
            future_images = sample.get("future_images", [])
            total_future = len(future_images)

            # Validate valid positions and invalid positions
            # Valid positions must satisfy: indices >= 0, index < total_future, delta > 0 and finite
            # Invalid positions must satisfy: index == -1
            for m in range(M):
                for k in range(K):
                    is_v = bool(future_valid[m, k].item())
                    idx = int(future_indices[m, k].item())
                    dt = float(future_deltas[m, k].item())

                    if is_v:
                        if idx < 0 or idx >= total_future:
                            raise ValueError(
                                f"Valid future target at [{m}, {k}] has out-of-range index {idx} (total_future={total_future})"
                            )
                        if not math.isfinite(dt) or dt <= 0.0:
                            raise ValueError(
                                f"Valid future target at [{m}, {k}] has non-positive/non-finite delta {dt}"
                            )
                    else:
                        if idx != -1:
                            raise ValueError(
                                f"Invalid future target at [{m}, {k}] must have index -1, got {idx}"
                            )
                        if not math.isfinite(dt) or dt != 0.0:
                            raise ValueError("Invalid future target must have finite zero delta")

            future_valid_count = int(future_valid.sum().item())

            if future_valid_count > 0:
                # Extract target representations from frozen encoder under torch.no_grad()
                targets = self._future_targets(sample, future_indices, future_valid)

                # Predict future latents via future_head
                # deep: [M, L, D], shallow: [M, L, D], deltas: [M, K]
                pred_latents = self.future_head(deep, shallow, future_deltas)

                # MSE per horizon: averaged over queries and hidden dimension
                # pred_latents: [M, K, queries, D], targets: [M, K, queries, D]
                diff_sq = (pred_latents - targets) ** 2
                mse_per_target_k = diff_sq.mean(dim=(-1, -2))  # [M, K]

                # Per-target: average over valid horizons, 0 if target has no valid horizon
                target_losses = []
                for m in range(M):
                    m_valid = future_valid[m]
                    if m_valid.any():
                        m_loss = mse_per_target_k[m][m_valid].mean()
                    else:
                        m_loss = torch.zeros((), dtype=torch.float32, device=device)
                    target_losses.append(m_loss)

                # Average across all M targets
                future_loss = torch.stack(target_losses, dim=0).mean()

        total_loss = action_loss + float(future_weight) * future_loss
        loss_sum = total_loss * float(M)

        result = dict(action_out)
        result["loss"] = total_loss
        result["loss_sum"] = loss_sum
        result["action_loss"] = action_loss
        result["future_loss"] = future_loss
        result["target_count"] = M
        result["future_valid_count"] = future_valid_count
        return result

    @torch.no_grad()
    def predict_actions(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Offline evaluation inference for the last target frame in sample.

        Does not require future images or future keys.
        """
        deep, shallow = self.features(sample)

        # Slice the last target
        last_deep = deep[-1:]  # [1, L, D]
        last_shallow = shallow[-1:]  # [1, L, D]

        state = sample.get("state", None)
        state_mask = sample.get("state_mask", None)
        action_mask = sample.get("action_mask", None)

        last_state = state[-1:] if state is not None else None
        last_state_mask = state_mask[-1:] if state_mask is not None else None
        last_action_mask = action_mask[-1:] if action_mask is not None else None

        ah_config = getattr(self.policy.action_head, "config", None)
        shallow_fusion = getattr(ah_config, "shallow_fusion", "none")
        pass_shallow = last_shallow if shallow_fusion != "none" else None

        with _autocast_context(last_deep.device, enabled=False):
            return self.policy.action_head.sample(
                last_deep,
                state=last_state,
                state_mask=last_state_mask,
                action_mask=last_action_mask,
                shallow_tokens=pass_shallow,
            )
