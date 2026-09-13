"""Differentiable bounded causal sequence training module for FabriVLA."""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fabri_moss.native_cache import execute_native_layers, format_frame_timestamp


def _autocast_context(device: torch.device, enabled: bool = True) -> Any:
    """Control CUDA autocast explicitly, including inside an outer mixed-precision context."""
    is_cuda = (device.type == "cuda") if isinstance(device, torch.device) else str(device).startswith("cuda")
    if is_cuda:
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)
    return contextlib.nullcontext()


class NativeSequencePolicy(nn.Module):
    """Sequence-level differentiable training wrapper for FabriVLA.

    Flattens an episode segment of N frames into a single temporal sequence [1, N * 1024, hidden_dim].
    Causal attention ensures that earlier queries cannot attend to later keys.
    """

    def __init__(
        self,
        policy: nn.Module,
        shallow_layer: int = 6,
        use_timestamps: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.policy = policy
        self.shallow_layer = shallow_layer
        self.use_timestamps = use_timestamps
        self.gradient_checkpointing = gradient_checkpointing

        core = self.native_core
        if not hasattr(core, "layers"):
            raise AttributeError("native_core must have 'layers' attribute")
        num_layers = len(core.layers)
        if not (1 <= self.shallow_layer <= num_layers):
            raise ValueError(
                f"shallow_layer must be in [1, {num_layers}], got {self.shallow_layer}"
            )

        self.policy.float()
        for p in self.policy.parameters():
            p.requires_grad = True

        self._set_vision_gradient_checkpointing(self.gradient_checkpointing)

    @property
    def native_core(self) -> nn.Module:
        embedder = getattr(self.policy, "embedder", None)
        if embedder is None:
            raise AttributeError("policy must have an 'embedder' attribute")
        model = getattr(embedder, "model", None)
        if model is None:
            raise AttributeError("policy.embedder must have a 'model' attribute")
        lm = getattr(model, "language_model", None)
        if lm is None:
            raise AttributeError("policy.embedder.model must have a 'language_model' attribute")

        if hasattr(lm, "model"):
            return lm.model
        return lm

    def _set_vision_gradient_checkpointing(self, enable: bool) -> None:
        embedder = getattr(self.policy, "embedder", None)
        if embedder is None:
            return
        model = getattr(embedder, "model", None)
        if model is None:
            return
        vision_model = getattr(model, "vision_model", None)
        if vision_model is None:
            return
        encoder = getattr(vision_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "gradient_checkpointing"):
            encoder.gradient_checkpointing = enable

    def _validate_sample(self, sample: Dict[str, Any]) -> Tuple[int, int]:
        required_keys = ("images_window", "frame_ids", "observation_times", "prompt", "target_indices")
        for k in required_keys:
            if k not in sample:
                raise KeyError(f"Sample missing required key: '{k}'")

        images_window = sample["images_window"]
        frame_ids = sample["frame_ids"]
        observation_times = sample["observation_times"]
        prompt = sample["prompt"]
        target_indices = sample["target_indices"]

        if not isinstance(images_window, (list, tuple)):
            raise TypeError(f"images_window must be list or tuple, got {type(images_window)}")
        N = len(images_window)
        if N == 0:
            raise ValueError("images_window must not be empty")

        if len(frame_ids) != N:
            raise ValueError(f"frame_ids length {len(frame_ids)} != images_window length {N}")
        if len(observation_times) != N:
            raise ValueError(f"observation_times length {len(observation_times)} != images_window length {N}")

        for i, fid in enumerate(frame_ids):
            if type(fid) is not int or fid < 0:
                raise ValueError(f"frame_ids[{i}] must be non-negative int, got {fid}")
            if i > 0 and fid <= frame_ids[i - 1]:
                raise ValueError(
                    f"frame_ids must be strictly increasing: frame_ids[{i}]={fid} <= frame_ids[{i-1}]={frame_ids[i-1]}"
                )

        for i, t in enumerate(observation_times):
            if type(t) is bool or not (isinstance(t, (int, float)) and math.isfinite(t) and t >= 0.0):
                raise ValueError(f"observation_times[{i}] must be finite non-negative float, got {t}")
            if i > 0 and t < observation_times[i - 1]:
                raise ValueError(
                    f"observation_times must be non-decreasing: t[{i}]={t} < t[{i-1}]={observation_times[i-1]}"
                )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if not isinstance(target_indices, (list, tuple)) or len(target_indices) == 0:
            raise ValueError("target_indices must be a non-empty list or tuple")

        M = len(target_indices)
        for i, idx in enumerate(target_indices):
            if type(idx) is not int or not (0 <= idx < N):
                raise ValueError(f"target_indices contains invalid relative frame index {idx} for sequence of len {N}")
            if i > 0 and idx <= target_indices[i - 1]:
                raise ValueError(
                    f"target_indices must be strictly increasing: idx[{i}]={idx} <= idx[{i-1}]={target_indices[i-1]}"
                )

        image_masks = sample.get("image_masks", None)
        if image_masks is not None:
            if not isinstance(image_masks, (list, tuple)):
                raise TypeError(f"image_masks must be a list or tuple if provided, got {type(image_masks)}")
            if len(image_masks) != N:
                raise ValueError(f"image_masks length {len(image_masks)} != images_window length {N}")
            for i, m in enumerate(image_masks):
                if m is not None and not isinstance(m, torch.Tensor):
                    raise TypeError(f"image_masks[{i}] must be a torch.Tensor or None, got {type(m)}")

        # Validate replay_groups if provided
        replay_groups = sample.get("replay_groups", None)
        if replay_groups is not None:
            if not isinstance(replay_groups, (list, tuple)):
                raise TypeError(f"replay_groups must be list or tuple, got {type(replay_groups)}")
            if len(replay_groups) == 0:
                raise ValueError("replay_groups must not be empty if provided")

            all_target_positions: List[int] = []
            all_group_obs_indices: set[int] = set()

            for g_idx, group in enumerate(replay_groups):
                if not isinstance(group, dict):
                    raise TypeError(f"replay_groups[{g_idx}] must be a dict, got {type(group)}")
                if "observation_indices" not in group:
                    raise KeyError(f"replay_groups[{g_idx}] missing required key 'observation_indices'")
                if "target_positions" not in group:
                    raise KeyError(f"replay_groups[{g_idx}] missing required key 'target_positions'")

                obs_indices = group["observation_indices"]
                target_positions = group["target_positions"]

                if not isinstance(obs_indices, (list, tuple)):
                    raise TypeError(f"replay_groups[{g_idx}]['observation_indices'] must be list or tuple")
                if len(obs_indices) == 0:
                    raise ValueError(f"replay_groups[{g_idx}]['observation_indices'] must not be empty")

                if not isinstance(target_positions, (list, tuple)):
                    raise TypeError(f"replay_groups[{g_idx}]['target_positions'] must be list or tuple")
                if len(target_positions) == 0:
                    raise ValueError(f"replay_groups[{g_idx}]['target_positions'] must not be empty")

                # Validate observation_indices: strictly increasing ints in [0, N-1]
                for oi, o_idx in enumerate(obs_indices):
                    if type(o_idx) is not int or not (0 <= o_idx < N):
                        raise ValueError(
                            f"replay_groups[{g_idx}]['observation_indices'][{oi}]={o_idx} must be int in [0, {N-1}]"
                        )
                    if oi > 0 and o_idx <= obs_indices[oi - 1]:
                        raise ValueError(
                            f"replay_groups[{g_idx}]['observation_indices'] must be strictly increasing: "
                            f"{o_idx} <= {obs_indices[oi - 1]}"
                        )
                    all_group_obs_indices.add(o_idx)

                # Validate target_positions: strictly increasing ints in [0, M-1]
                obs_set = set(obs_indices)
                group_target_pool_frames: List[int] = []
                for ti, t_pos in enumerate(target_positions):
                    if type(t_pos) is not int or not (0 <= t_pos < M):
                        raise ValueError(
                            f"replay_groups[{g_idx}]['target_positions'][{ti}]={t_pos} must be int in [0, {M-1}]"
                        )
                    if ti > 0 and t_pos <= target_positions[ti - 1]:
                        raise ValueError(
                            f"replay_groups[{g_idx}]['target_positions'] must be strictly increasing: "
                            f"{t_pos} <= {target_positions[ti - 1]}"
                        )
                    all_target_positions.append(t_pos)

                    target_pool_frame = target_indices[t_pos]
                    # Check each target's frame is in its group's observations
                    if target_pool_frame not in obs_set:
                        raise ValueError(
                            f"replay_groups[{g_idx}] target at outer position {t_pos} (frame {target_pool_frame}) "
                            f"is not in group's observation_indices {obs_indices}"
                        )
                    group_target_pool_frames.append(target_pool_frame)

                # Check group's final observation equals its latest target frame (no unsupervised trailing frames)
                latest_target_frame = max(group_target_pool_frames)
                final_obs_frame = obs_indices[-1]
                if final_obs_frame != latest_target_frame:
                    raise ValueError(
                        f"replay_groups[{g_idx}] final observation frame ({final_obs_frame}) does not equal "
                        f"its latest target frame ({latest_target_frame}); unsupervised trailing frames are forbidden"
                    )

            # Every outer target position must appear exactly once across groups
            if len(all_target_positions) != M or set(all_target_positions) != set(range(M)):
                raise ValueError(
                    f"Every outer target position 0..{M-1} must appear exactly once across replay_groups; "
                    f"got count {len(all_target_positions)}, unique count {len(set(all_target_positions))}"
                )

            # Strict check: unused pool frames should reject (union of group observations must exactly equal 0..N-1)
            # to catch sampler inconsistency and prevent silent data waste or mismatch
            if all_group_obs_indices != set(range(N)):
                unused = sorted(list(set(range(N)) - all_group_obs_indices))
                raise ValueError(
                    f"Union of replay_groups observation_indices does not cover entire pool of frames 0..{N-1}. "
                    f"Unused pool frames: {unused}. Sampler inconsistency detected."
                )

        return N, M

    def _encode_sequence_embeddings(
        self,
        images_window: Sequence[Any],
        frame_ids: Sequence[int],
        observation_times: Sequence[float],
        prompt: str,
        image_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        return_visual_tokens: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        embedder = getattr(self.policy, "embedder", None)
        if embedder is None:
            raise AttributeError("policy must have an 'embedder' attribute")
        if not hasattr(embedder, "_preprocess_images_on_cpu"):
            raise AttributeError("embedder must implement _preprocess_images_on_cpu")
        if not hasattr(embedder, "_prepare_batch_and_fuse_embeddings"):
            raise AttributeError("embedder must implement _prepare_batch_and_fuse_embeddings")

        N = len(images_window)
        device = getattr(embedder, "device", next(self.policy.parameters()).device)

        # 1. Build and validate all frame prompts before visual feature extraction
        frame_prompts: List[str] = []
        for i in range(N):
            base_prompt = embedder._build_multimodal_prompt([1], prompt)
            if self.use_timestamps:
                ts_prefix = format_frame_timestamp(frame_ids[i], observation_times[i])
                full_frame_prompt = ts_prefix + base_prompt
            else:
                full_frame_prompt = base_prompt
            frame_prompts.append(full_frame_prompt)

        max_text_len = getattr(embedder, "max_text_length", 1024)
        if hasattr(embedder, "tokenizer"):
            for i, p in enumerate(frame_prompts):
                tok_out = embedder.tokenizer(p, return_tensors="pt")
                input_ids = tok_out.input_ids if hasattr(tok_out, "input_ids") else tok_out
                if input_ids.shape[1] > max_text_len:
                    raise ValueError(
                        f"Frame {i} prompt token length {input_ids.shape[1]} exceeds max_text_length {max_text_len}"
                    )

        # 2. Flatten and preprocess images
        all_frames: List[Any] = []
        for i, img_elem in enumerate(images_window):
            if isinstance(img_elem, (list, tuple)):
                if len(img_elem) != 1:
                    raise ValueError(
                        f"Each frame in images_window must contain exactly 1 image/tile, got {len(img_elem)}"
                    )
                all_frames.append(img_elem[0])
            else:
                all_frames.append(img_elem)

        pixel_values, num_tiles_list = embedder._preprocess_images_on_cpu(all_frames)
        if num_tiles_list != [1] * N:
            raise ValueError(f"num_tiles_list must be {[1] * N}, got {num_tiles_list}")

        pixel_dtype = torch.bfloat16 if (device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda")) else torch.float32
        pixel_values = pixel_values.to(device=device, dtype=pixel_dtype)

        # 3. Visual feature extraction
        with _autocast_context(device, enabled=True):
            vit_embeds = embedder.model.extract_feature(pixel_values)

        if vit_embeds.ndim != 3 or vit_embeds.shape[0] != N:
            raise ValueError(f"vit_embeds must have batch shape [N={N}, tokens, dim], got {tuple(vit_embeds.shape)}")

        vit_embeds_list = [vit_embeds[i : i + 1] for i in range(N)]

        # 4. Prepare masks
        masks: List[torch.Tensor] = []
        for i in range(N):
            if image_masks is not None and image_masks[i] is not None:
                masks.append(image_masks[i].to(device=device, dtype=torch.bool))
            else:
                masks.append(torch.ones(1, dtype=torch.bool, device=device))

        # 5. Batch embedding fusion
        batch_num_tiles = [[1] for _ in range(N)]
        with _autocast_context(device, enabled=True):
            batch_inputs_embeds, batch_attention_mask = embedder._prepare_batch_and_fuse_embeddings(
                prompts=frame_prompts,
                vit_embeds_batch=vit_embeds_list,
                image_masks=masks,
                batch_num_tiles_list=batch_num_tiles,
            )

        if batch_inputs_embeds.shape[:2] != (N, max_text_len):
            raise ValueError(
                f"batch_inputs_embeds shape prefix must be {(N, max_text_len)}, got {tuple(batch_inputs_embeds.shape)}"
            )

        seq_inputs_embeds = batch_inputs_embeds.reshape(1, N * max_text_len, -1)
        seq_attention_mask = batch_attention_mask.reshape(1, N * max_text_len)

        core = self.native_core
        core_dtype = next(core.parameters()).dtype
        core_device = next(core.parameters()).device
        seq_inputs_embeds = seq_inputs_embeds.to(dtype=core_dtype, device=core_device)
        seq_attention_mask = seq_attention_mask.to(device=core_device)

        if return_visual_tokens:
            return seq_inputs_embeds, seq_attention_mask, vit_embeds
        return seq_inputs_embeds, seq_attention_mask

    def features(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute deep and shallow features for the target frames in the sample."""
        N, M = self._validate_sample(sample)
        images_window = sample["images_window"]
        frame_ids = sample["frame_ids"]
        observation_times = sample["observation_times"]
        prompt = sample["prompt"]
        target_indices = sample["target_indices"]
        image_masks = sample.get("image_masks", None)
        replay_groups = sample.get("replay_groups", None)

        seq_inputs_embeds, seq_attention_mask = self._encode_sequence_embeddings(
            images_window=images_window,
            frame_ids=frame_ids,
            observation_times=observation_times,
            prompt=prompt,
            image_masks=image_masks,
        )

        core = self.native_core
        num_layers = len(core.layers)
        tokens_per_frame = getattr(getattr(self.policy, "embedder", None), "max_text_length", 1024)

        device = seq_inputs_embeds.device
        do_grad_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()

        if replay_groups is None:
            # Original sequential execution path over all N frames
            with _autocast_context(device, enabled=True):
                final_norm_h, inter_h = execute_native_layers(
                    core=core,
                    inputs_embeds=seq_inputs_embeds,
                    attention_mask_2d=seq_attention_mask,
                    cache=None,
                    start_pos=0,
                    gradient_checkpointing=do_grad_checkpoint,
                )

            deep_full = final_norm_h
            if self.shallow_layer == num_layers:
                shallow_full = final_norm_h
            else:
                shallow_full = inter_h[self.shallow_layer]

            deep_frames = deep_full.view(N, tokens_per_frame, -1)
            shallow_frames = shallow_full.view(N, tokens_per_frame, -1)

            target_indices_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
            deep_selected = torch.index_select(deep_frames, 0, target_indices_tensor).to(torch.float32)
            shallow_selected = torch.index_select(shallow_frames, 0, target_indices_tensor).to(torch.float32)

            return deep_selected, shallow_selected

        # Replay groups path:
        # Reshape [1, N * L, H] -> [N, L, H] and [1, N * L] -> [N, L]
        hidden_dim = seq_inputs_embeds.shape[-1]
        pool_embeds = seq_inputs_embeds.view(N, tokens_per_frame, hidden_dim)
        pool_masks = seq_attention_mask.view(N, tokens_per_frame)

        # Pre-allocate lists/tensors to gather target features in original outer target order (0..M-1)
        deep_target_list: List[Optional[torch.Tensor]] = [None] * M
        shallow_target_list: List[Optional[torch.Tensor]] = [None] * M

        for group in replay_groups:
            obs_indices = group["observation_indices"]
            target_positions = group["target_positions"]
            g_N = len(obs_indices)

            # Select full frames with index_select from reshaped pool [N, L, H]
            obs_indices_tensor = torch.tensor(obs_indices, dtype=torch.long, device=device)
            g_embeds_frames = torch.index_select(pool_embeds, 0, obs_indices_tensor)  # [g_N, L, H]
            g_masks_frames = torch.index_select(pool_masks, 0, obs_indices_tensor)    # [g_N, L]

            # Flatten independently: [1, g_N * L, H] and [1, g_N * L]
            g_inputs_embeds = g_embeds_frames.reshape(1, g_N * tokens_per_frame, hidden_dim)
            g_attention_mask = g_masks_frames.reshape(1, g_N * tokens_per_frame)

            # Run execute_native_layers independently per group, no crossgroup LLM state
            with _autocast_context(device, enabled=True):
                g_final_norm_h, g_inter_h = execute_native_layers(
                    core=core,
                    inputs_embeds=g_inputs_embeds,
                    attention_mask_2d=g_attention_mask,
                    cache=None,
                    start_pos=0,
                    gradient_checkpointing=do_grad_checkpoint,
                )

            g_deep_full = g_final_norm_h
            if self.shallow_layer == num_layers:
                g_shallow_full = g_final_norm_h
            else:
                g_shallow_full = g_inter_h[self.shallow_layer]

            g_deep_frames = g_deep_full.view(g_N, tokens_per_frame, -1)
            g_shallow_frames = g_shallow_full.view(g_N, tokens_per_frame, -1)

            # Map each target in group to its group-relative frame index in obs_indices
            obs_to_rel = {pool_idx: rel_idx for rel_idx, pool_idx in enumerate(obs_indices)}
            g_rel_target_indices = [obs_to_rel[target_indices[t_pos]] for t_pos in target_positions]
            g_rel_tensor = torch.tensor(g_rel_target_indices, dtype=torch.long, device=device)

            # Extract group's target blocks
            g_deep_targets = torch.index_select(g_deep_frames, 0, g_rel_tensor).to(torch.float32)
            g_shallow_targets = torch.index_select(g_shallow_frames, 0, g_rel_tensor).to(torch.float32)

            for ti, t_pos in enumerate(target_positions):
                deep_target_list[t_pos] = g_deep_targets[ti]
                shallow_target_list[t_pos] = g_shallow_targets[ti]

        # Gather in original outer target order preserving gradients
        deep_selected = torch.stack(deep_target_list, dim=0)
        shallow_selected = torch.stack(shallow_target_list, dim=0)

        return deep_selected, shallow_selected

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute action loss across target frames in sample with autocast disabled for action_head."""
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

        deep, shallow = self.features(sample)

        device = deep.device
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

        loss = action_out["loss"]
        loss_sum = loss * float(M)

        result = dict(action_out)
        result["loss_sum"] = loss_sum
        result["target_count"] = M
        return result
