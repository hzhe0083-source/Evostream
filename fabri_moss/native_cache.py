"""Native LLM KV-cache management and adapter for FabriVLA.

Implements native joint vision-language KV-cache without modifying
attention structures, matrix compression, or adding new trainable parameters.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers.cache_utils import DynamicCache


def execute_native_layers(
    core: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    cache: Optional[DynamicCache] = None,
    start_pos: int = 0,
    gradient_checkpointing: bool = False,
    token_times: Optional[torch.Tensor] = None,
    temporal_config: Optional[Any] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Directly loop over core.layers bypassing model-level causal mask guards.

    Args:
        core: native LLM core module with .layers, .rotary_emb, and .norm
        inputs_embeds: [1, seq_len, hidden_dim]
        attention_mask_2d: [1, total_key_len] combined binary mask (past + new)
        cache: DynamicCache populated with prior KV, or None
        start_pos: absolute token position index where inputs_embeds begins
        gradient_checkpointing: whether to use non-reentrant torch checkpoint per layer
        token_times: optional [1, seq_len] tensor with physical seconds
        temporal_config: optional TemporalRoPEConfig

    Returns:
        final_norm_hidden: [1, seq_len, hidden_dim] (norm applied to last layer)
        intermediate_hiddens: dict mapping 1-based layer index s -> [1, seq_len, hidden_dim]
    """
    if gradient_checkpointing and cache is not None:
        raise ValueError("Cannot enable gradient_checkpointing with mutable cache; cache must be None.")

    # Validate pairing of token_times and temporal_config
    if (token_times is None) != (temporal_config is None):
        raise ValueError("token_times and temporal_config must be provided together or both be None.")

    seq_len = inputs_embeds.shape[1]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    total_key_len = attention_mask_2d.shape[1]

    # 1. Continuous physical token indices
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_position = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device)

    # 2. Rotary embeddings from core.rotary_emb or temporal_position_embeddings
    if token_times is not None and temporal_config is not None:
        from fabri_moss.temporal_rope import temporal_position_embeddings
        position_embeddings = temporal_position_embeddings(
            core=core,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_times=token_times,
            config=temporal_config,
        )
    else:
        position_embeddings = core.rotary_emb(inputs_embeds, position_ids)

    # 3. Attention mask construction:
    attn_impl = getattr(core.config, "_attn_implementation", "eager")
    if attn_impl == "flash_attention_2":
        # FA2 native attention unpads queries itself using 2D mask
        layer_attention_mask = attention_mask_2d
    else:
        # 4D additive causal mask for eager / sdpa:
        # shape: [1, 1, seq_len, total_key_len]
        # key_abs_pos <= query_abs_pos AND attention_mask_2d[key] == 1 -> 0.0, else finfo(dtype).min
        min_dtype = torch.finfo(dtype).min
        q_abs = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        k_abs = torch.arange(0, total_key_len, device=device).unsqueeze(0)                 # [1, total_key_len]
        causal_allowed = k_abs <= q_abs                                                    # [seq_len, total_key_len]
        k_valid = attention_mask_2d.squeeze(0).unsqueeze(0).bool()                          # [1, total_key_len]
        attend = causal_allowed & k_valid                                                  # [seq_len, total_key_len]

        causal_mask_4d = torch.full((1, 1, seq_len, total_key_len), min_dtype, dtype=dtype, device=device)
        causal_mask_4d[0, 0][attend] = 0.0
        layer_attention_mask = causal_mask_4d

    # 4. Determine layer argument name for past_key_value(s) via inspect.signature
    layer0 = core.layers[0]
    sig = inspect.signature(layer0.forward)
    kv_param_name = "past_key_values" if "past_key_values" in sig.parameters else "past_key_value"

    use_cache = cache is not None

    h = inputs_embeds
    intermediate_hiddens: Dict[int, torch.Tensor] = {}

    for layer_idx, layer in enumerate(core.layers, start=1):
        kwargs = {
            "hidden_states": h,
            "attention_mask": layer_attention_mask,
            "position_ids": position_ids,
            kv_param_name: cache,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        # Only pass parameters accepted by layer.forward
        layer_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        if gradient_checkpointing and torch.is_grad_enabled():
            def make_custom_forward(mod, pass_kwargs):
                def custom_forward(hidden_states):
                    kw = dict(pass_kwargs)
                    kw["hidden_states"] = hidden_states
                    out = mod(**kw)
                    return out[0] if isinstance(out, tuple) else out
                return custom_forward

            layer_fn = make_custom_forward(layer, layer_kwargs)
            h = torch_checkpoint(layer_fn, h, use_reentrant=False)
        else:
            layer_outputs = layer(**layer_kwargs)
            h = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        intermediate_hiddens[layer_idx] = h

    # Apply final norm
    final_norm_hidden = core.norm(h)
    return final_norm_hidden, intermediate_hiddens


def format_frame_timestamp(frame_id: int, observation_time: float) -> str:
    """Format frame timestamp prefix for multimodal prompt."""
    if type(frame_id) is not int or frame_id < 0:
        raise ValueError(f"frame_id must be non-negative int (not bool), got {frame_id} ({type(frame_id)})")
    if type(observation_time) is bool or not (
        isinstance(observation_time, (int, float))
        and math.isfinite(observation_time)
        and observation_time >= 0.0
    ):
        raise ValueError(f"observation_time must be finite non-negative float (not bool), got {observation_time}")
    return f"Frame {frame_id}, time {observation_time:.6f} s.\n"


@dataclass(frozen=True)
class NativeCacheConfig:
    """Configuration for native KV cache."""
    max_frames: int = 5
    shallow_layer: int = 6
    use_timestamps: bool = True

    def __post_init__(self):
        if type(self.max_frames) is not int or self.max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {self.max_frames}")
        if type(self.shallow_layer) is not int or self.shallow_layer <= 0:
            raise ValueError(f"shallow_layer must be > 0, got {self.shallow_layer}")
        if type(self.use_timestamps) is not bool:
            raise TypeError(f"use_timestamps must be strictly bool, got {type(self.use_timestamps)}")


@dataclass(frozen=True)
class NativeEmbeddingBlock:
    """Single-frame fused multimodal embedding block."""
    frame_id: int
    inputs_embeds: torch.Tensor  # shape [1, seq_len, hidden_dim]
    attention_mask: torch.Tensor  # shape [1, seq_len]
    owner: Any
    revision: int
    prompt: str
    capture_time: Optional[float] = None
    observation_time: Optional[float] = None
    visual_tokens: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Strict validation: frame_id must be int, not bool, non-negative
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative int (not bool), got {self.frame_id} ({type(self.frame_id)})")
        if self.capture_time is not None:
            if type(self.capture_time) is bool or not (
                isinstance(self.capture_time, (int, float))
                and math.isfinite(self.capture_time)
                and self.capture_time >= 0.0
            ):
                raise ValueError(f"capture_time must be finite non-negative float (not bool), got {self.capture_time}")
        if self.observation_time is not None:
            if type(self.observation_time) is bool or not (
                isinstance(self.observation_time, (int, float))
                and math.isfinite(self.observation_time)
                and self.observation_time >= 0.0
            ):
                raise ValueError(f"observation_time must be finite non-negative float (not bool), got {self.observation_time}")
        if self.inputs_embeds.ndim != 3 or self.inputs_embeds.shape[0] != 1:
            raise ValueError(
                f"inputs_embeds must have shape [1, seq_len, hidden_dim], got {tuple(self.inputs_embeds.shape)}"
            )
        if self.attention_mask.ndim != 2 or self.attention_mask.shape[0] != 1:
            raise ValueError(
                f"attention_mask must have shape [1, seq_len], got {tuple(self.attention_mask.shape)}"
            )
        if self.inputs_embeds.shape[1] != self.attention_mask.shape[1]:
            raise ValueError(
                f"inputs_embeds seq_len ({self.inputs_embeds.shape[1]}) must match "
                f"attention_mask seq_len ({self.attention_mask.shape[1]})"
            )
        # Binary mask validation: values must be in {0, 1}
        mask_vals = torch.unique(self.attention_mask)
        for v in mask_vals:
            if v.item() not in (0, 1, 0.0, 1.0):
                raise ValueError(f"attention_mask must be binary (0 or 1), found {v.item()}")
        if not isinstance(self.prompt, str):
            raise TypeError(f"prompt must be str, got {type(self.prompt)}")
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty after stripping")
        if self.visual_tokens is not None:
            if not isinstance(self.visual_tokens, torch.Tensor):
                raise TypeError(f"visual_tokens must be a torch.Tensor, got {type(self.visual_tokens).__name__}")
            if self.visual_tokens.ndim != 3 or self.visual_tokens.shape[0] != 1:
                raise ValueError(
                    f"visual_tokens must have shape [1, P, H], got {tuple(self.visual_tokens.shape)}"
                )
            if self.visual_tokens.shape[2] != self.inputs_embeds.shape[2]:
                raise ValueError(
                    f"visual_tokens hidden_dim ({self.visual_tokens.shape[2]}) must match inputs_embeds hidden_dim ({self.inputs_embeds.shape[2]})"
                )
            if self.visual_tokens.device != self.inputs_embeds.device:
                raise ValueError(
                    f"visual_tokens device ({self.visual_tokens.device}) must match inputs_embeds device ({self.inputs_embeds.device})"
                )
            if not torch.isfinite(self.visual_tokens).all():
                raise ValueError("visual_tokens contains non-finite values")

    @property
    def seq_len(self) -> int:
        return self.inputs_embeds.shape[1]


@dataclass(frozen=True)
class NativeKVState:
    """Persistent native joint KV state over a sliding window of frames."""
    layer_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    blocks: Tuple[NativeEmbeddingBlock, ...]
    attention_mask: torch.Tensor  # shape [1, total_seq_len]
    last_frame_id: int
    frame_count: int
    prompt: str
    owner: Any
    revision: int
    rebuild_count: int

    def __post_init__(self):
        if self.frame_count < 0:
            raise ValueError(f"frame_count must be non-negative, got {self.frame_count}")
        if self.last_frame_id < -1:
            raise ValueError(f"last_frame_id must be >= -1, got {self.last_frame_id}")
        if self.rebuild_count < 0:
            raise ValueError(f"rebuild_count must be non-negative, got {self.rebuild_count}")
        if not isinstance(self.layer_kv, tuple):
            raise TypeError("layer_kv must be a tuple of (K, V) pairs")
        if not isinstance(self.blocks, tuple):
            raise TypeError("blocks must be a tuple of NativeEmbeddingBlock")
        if self.attention_mask.ndim != 2 or self.attention_mask.shape[0] != 1:
            raise ValueError(
                f"attention_mask must have shape [1, total_seq_len], got {tuple(self.attention_mask.shape)}"
            )

    def detached(self) -> NativeKVState:
        """Return a detached copy of this state."""
        detached_kv = tuple(
            (k.detach(), v.detach())
            for k, v in self.layer_kv
        )
        detached_blocks = tuple(
            NativeEmbeddingBlock(
                frame_id=b.frame_id,
                inputs_embeds=b.inputs_embeds.detach(),
                attention_mask=b.attention_mask.detach(),
                owner=b.owner,
                revision=b.revision,
                prompt=b.prompt,
                capture_time=b.capture_time,
                observation_time=b.observation_time,
                visual_tokens=b.visual_tokens.detach() if b.visual_tokens is not None else None,
            )
            for b in self.blocks
        )
        return NativeKVState(
            layer_kv=detached_kv,
            blocks=detached_blocks,
            attention_mask=self.attention_mask.detach(),
            last_frame_id=self.last_frame_id,
            frame_count=self.frame_count,
            prompt=self.prompt,
            owner=self.owner,
            revision=self.revision,
            rebuild_count=self.rebuild_count,
        )

    @property
    def kv_nbytes(self) -> int:
        """Memory in bytes occupied strictly by layer_kv tensors."""
        total = 0
        for k, v in self.layer_kv:
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total

    @property
    def nbytes(self) -> int:
        """Total memory in bytes including layer_kv, blocks embeddings, and masks."""
        total = self.kv_nbytes
        for b in self.blocks:
            total += b.inputs_embeds.numel() * b.inputs_embeds.element_size()
            total += b.attention_mask.numel() * b.attention_mask.element_size()
            if b.visual_tokens is not None:
                total += b.visual_tokens.numel() * b.visual_tokens.element_size()
        total += self.attention_mask.numel() * self.attention_mask.element_size()
        return total


def extract_layer_kv(cache: DynamicCache) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Extract (K, V) tuples across layers from DynamicCache.

    Compatible with both transformers 4.x (key_cache/value_cache) and
    transformers 5.x (layers[i].keys / layers[i].values).
    """
    layers_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if hasattr(cache, "layers") and cache.layers:
        for layer in cache.layers:
            layers_kv.append((layer.keys, layer.values))
    elif hasattr(cache, "key_cache") and cache.key_cache:
        for k, v in zip(cache.key_cache, cache.value_cache):
            layers_kv.append((k, v))
    else:
        raise ValueError("Cannot extract key/value pairs from cache: unsupported DynamicCache structure.")
    return tuple(layers_kv)


def populate_cache_from_layer_kv(layer_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> DynamicCache:
    """Create a fresh DynamicCache and populate it with layer_kv references.

    Does not modify old tensors; DynamicCache.update appends/concatenates
    new tensors when subsequent tokens are fed.
    """
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(layer_kv):
        cache.update(k, v, layer_idx)
    return cache


class NativeCacheAdapter(nn.Module):
    """Adapter wrapping an existing FabriVLA policy with native sliding-window KV cache."""

    def __init__(self, policy: nn.Module, config: Optional[NativeCacheConfig] = None):
        super().__init__()
        self.policy = policy
        self.config = config if config is not None else NativeCacheConfig()
        self._revision = 0

        # Validate shallow_layer strictly in [1, len(layers)]
        core = self.native_core
        if not hasattr(core, "layers"):
            raise AttributeError("native_core must have 'layers' attribute")
        num_layers = len(core.layers)
        if not (1 <= self.config.shallow_layer <= num_layers):
            raise ValueError(
                f"shallow_layer must be in [1, {num_layers}], got {self.config.shallow_layer}"
            )

        # Freeze all policy parameters
        for param in self.policy.parameters():
            param.requires_grad = False
        self.policy.eval()
        self.eval()

    @property
    def revision(self) -> int:
        """Read-only property returning current revision."""
        return self._revision

    def train(self, mode: bool = True) -> NativeCacheAdapter:
        """If transitioning to train mode (or mode=True), bump revision.

        Keeps parameters frozen to satisfy zero-trainable requirement.
        """
        if mode:
            self._revision += 1
        return super().train(mode)

    @property
    def native_core(self) -> nn.Module:
        """Resolve the native LLM core module.

        Strictly resolves policy.embedder.model.language_model.model
        (or language_model itself if it doesn't have .model).
        """
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

    def _get_kv_heads_and_dim(self) -> Tuple[int, int]:
        """Extract num_key_value_heads and head_dim from native layer projection dims."""
        core = self.native_core
        layer0 = core.layers[0]
        attn = getattr(layer0, "self_attn", None)
        if attn is None:
            raise AttributeError("layer 0 must have 'self_attn' attribute")
        k_proj = getattr(attn, "k_proj", None)
        if k_proj is None or not hasattr(k_proj, "out_features"):
            raise AttributeError("self_attn must have 'k_proj' with out_features")

        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None and hasattr(core, "config") and hasattr(core.config, "head_dim"):
            head_dim = core.config.head_dim
        if head_dim is None:
            raise AttributeError("Could not determine head_dim from layer or config")

        num_kv_heads = k_proj.out_features // head_dim
        return num_kv_heads, head_dim

    @torch.no_grad()
    def encode_frame(
        self,
        images: List[Any],
        frame_id: int,
        prompt: str,
        image_mask: Optional[torch.Tensor] = None,
        capture_time: Optional[float] = None,
        observation_time: Optional[float] = None,
        preserve_visual_tokens: bool = False,
    ) -> NativeEmbeddingBlock:
        """Encode a single frame into a NativeEmbeddingBlock without calling LLM.

        Strictly adheres to single-view/tile semantics, 1024 token padding,
        prompt non-truncation verification, and mask-embed dimension validation.
        """
        if self.training:
            raise RuntimeError("encode_frame is only allowed in eval mode.")

        # Strict validation before any feature extraction
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"frame_id must be non-negative int (not bool), got {frame_id} ({type(frame_id)})")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt must be a non-empty string after stripping, got '{prompt}'")
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise ValueError(f"encode_frame strictly requires len(images) == 1, got {len(images) if isinstance(images, (list, tuple)) else type(images)}")

        if capture_time is not None:
            if type(capture_time) is bool or not (
                isinstance(capture_time, (int, float))
                and math.isfinite(capture_time)
                and capture_time >= 0.0
            ):
                raise ValueError(f"capture_time must be finite non-negative float (not bool), got {capture_time}")
        if observation_time is not None:
            if type(observation_time) is bool or not (
                isinstance(observation_time, (int, float))
                and math.isfinite(observation_time)
                and observation_time >= 0.0
            ):
                raise ValueError(f"observation_time must be finite non-negative float (not bool), got {observation_time}")
        if self.config.use_timestamps and observation_time is None:
            raise ValueError("observation_time must not be None when use_timestamps is True.")

        embedder = getattr(self.policy, "embedder", None)
        if embedder is None:
            raise AttributeError("policy must have an 'embedder' attribute for encode_frame.")

        # Ensure image_mask is provided or default all ones
        device = getattr(embedder, "device", next(self.policy.parameters()).device)
        if image_mask is None:
            image_mask = torch.ones(1, dtype=torch.bool, device=device)

        # Preprocess images
        pixel_values, num_tiles_list = embedder._preprocess_images(images)

        # Validate tiles == [1] strictly before feature extraction
        if num_tiles_list != [1]:
            raise ValueError(f"encode_frame strictly requires single tile [1], got {num_tiles_list}")

        # Build prompt string
        full_prompt = embedder._build_multimodal_prompt(num_tiles_list, prompt)
        if self.config.use_timestamps:
            full_prompt = format_frame_timestamp(frame_id, observation_time) + full_prompt

        # Validate prompt is not truncated
        max_text_len = getattr(embedder, "max_text_length", 1024)
        if hasattr(embedder, "tokenizer"):
            untruncated_ids = embedder.tokenizer(full_prompt, return_tensors="pt").input_ids
            if untruncated_ids.shape[1] > max_text_len:
                raise ValueError(
                    f"Prompt length {untruncated_ids.shape[1]} exceeds max_text_length {max_text_len}; "
                    "contract forbids truncated prompt in encode_frame."
                )

        # Extract ViT features (only after prompt validation)
        vit_embeds = embedder.model.extract_feature(pixel_values)

        # Fuse embeddings
        inputs_embeds, attention_mask = embedder._prepare_and_fuse_embeddings(
            prompt=full_prompt,
            vit_embeds=vit_embeds,
            image_mask=image_mask,
            num_tiles_list=num_tiles_list,
        )

        # Match dtype and device with native_core
        core = self.native_core
        core_dtype = next(core.parameters()).dtype
        core_device = next(core.parameters()).device
        inputs_embeds = inputs_embeds.to(dtype=core_dtype, device=core_device)
        attention_mask = attention_mask.to(device=core_device)

        # Validate shape [1, max_text_length, hidden]
        expected_seq_len = getattr(embedder, "max_text_length", 1024)
        if inputs_embeds.shape != (1, expected_seq_len, core.config.hidden_size):
            raise ValueError(
                f"inputs_embeds shape must be [1, {expected_seq_len}, {core.config.hidden_size}], got {tuple(inputs_embeds.shape)}"
            )

        saved_visual_tokens = None
        if preserve_visual_tokens:
            saved_vis = vit_embeds.to(dtype=core_dtype, device=core_device)
            if saved_vis.ndim == 2:
                saved_vis = saved_vis.unsqueeze(0)
            saved_visual_tokens = saved_vis

        return NativeEmbeddingBlock(
            frame_id=frame_id,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            owner=self,
            revision=self._revision,
            prompt=prompt,
            capture_time=capture_time,
            observation_time=observation_time,
            visual_tokens=saved_visual_tokens,
        )

    def _execute_native_layers(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        cache: DynamicCache,
        start_pos: int,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Directly loop over native_core.layers bypassing model-level causal mask guards.

        Delegates to module-level execute_native_layers preserving exact signature and behavior.
        """
        return execute_native_layers(
            core=self.native_core,
            inputs_embeds=inputs_embeds,
            attention_mask_2d=attention_mask_2d,
            cache=cache,
            start_pos=start_pos,
            gradient_checkpointing=False,
        )

    def _validate_previous_state(self, previous: NativeKVState, prompt: str) -> None:
        """Validate previous NativeKVState against adapter and model config."""
        if not isinstance(previous, NativeKVState):
            raise TypeError(f"previous must be NativeKVState, got {type(previous)}")
        if previous.owner is not self:
            raise ValueError(f"previous owner {previous.owner} does not match adapter {self}")
        if previous.revision != self._revision:
            raise ValueError(
                f"previous revision {previous.revision} does not match adapter revision {self._revision}"
            )
        if previous.prompt != prompt:
            raise ValueError(
                f"previous prompt '{previous.prompt}' does not match requested prompt '{prompt}'"
            )

        core = self.native_core
        num_layers = len(core.layers)
        if len(previous.layer_kv) != num_layers:
            raise ValueError(
                f"previous layer_kv count {len(previous.layer_kv)} does not match core layers count {num_layers}"
            )

        # Validate sequence length consistency with blocks
        expected_total_seq_len = sum(b.seq_len for b in previous.blocks)
        if previous.attention_mask.shape[1] != expected_total_seq_len:
            raise ValueError(
                f"previous attention_mask seq_len {previous.attention_mask.shape[1]} does not match "
                f"sum of blocks seq_len {expected_total_seq_len}"
            )

        num_kv_heads, head_dim = self._get_kv_heads_and_dim()
        for idx, (k, v) in enumerate(previous.layer_kv):
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError(f"layer_kv[{idx}] K/V must be 4D [1, Hkv, S, D]")
            if k.shape[0] != 1 or k.shape[1] != num_kv_heads or k.shape[3] != head_dim:
                raise ValueError(
                    f"layer_kv[{idx}] K shape {tuple(k.shape)} does not match expected [1, {num_kv_heads}, {expected_total_seq_len}, {head_dim}]"
                )
            if v.shape != k.shape:
                raise ValueError(f"layer_kv[{idx}] V shape {tuple(v.shape)} != K shape {tuple(k.shape)}")
            if k.shape[2] != expected_total_seq_len:
                raise ValueError(
                    f"layer_kv[{idx}] seq_len {k.shape[2]} does not match blocks seq_len {expected_total_seq_len}"
                )
            if not torch.isfinite(k).all() or not torch.isfinite(v).all():
                raise ValueError(f"layer_kv[{idx}] contains non-finite values")

        if not previous.blocks or len(previous.blocks) > self.config.max_frames:
            raise ValueError("previous blocks must be non-empty and within the history window")
        for block in previous.blocks:
            if block.owner is not self or block.revision != self.revision or block.prompt != prompt:
                raise ValueError("previous block metadata does not match adapter and prompt")
        if previous.blocks:
            if previous.last_frame_id != previous.blocks[-1].frame_id:
                raise ValueError(
                    f"previous last_frame_id {previous.last_frame_id} does not match last block frame_id {previous.blocks[-1].frame_id}"
                )

    def _read_block_group(
        self,
        new_blocks: Sequence[NativeEmbeddingBlock],
        prompt: str,
        previous: Optional[NativeKVState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, NativeKVState]:
        """Process a single chunk (<= max_frames) of new blocks."""
        w = self.config.max_frames
        core = self.native_core
        num_layers = len(core.layers)
        shallow_idx = self.config.shallow_layer
        last_block_len = new_blocks[-1].seq_len
        last_frame_id = new_blocks[-1].frame_id
        old_frame_count = previous.frame_count if previous is not None else 0
        new_frame_count = old_frame_count + len(new_blocks)

        old_blocks = previous.blocks if previous is not None else ()
        total_blocks_count = len(old_blocks) + len(new_blocks)

        if total_blocks_count <= w and previous is not None:
            # Incremental append: all new_blocks processed jointly in single forward pass
            combined_blocks = old_blocks + tuple(new_blocks)
            cache = populate_cache_from_layer_kv(previous.layer_kv)

            old_seq_len = sum(b.seq_len for b in old_blocks)
            new_inputs_embeds = torch.cat([b.inputs_embeds for b in new_blocks], dim=1)
            new_attention_mask = torch.cat([b.attention_mask for b in new_blocks], dim=1)
            combined_attention_mask = torch.cat([previous.attention_mask, new_attention_mask], dim=1)

            final_norm_h, inter_h = self._execute_native_layers(
                inputs_embeds=new_inputs_embeds,
                attention_mask_2d=combined_attention_mask,
                cache=cache,
                start_pos=old_seq_len,
            )

            updated_layer_kv = extract_layer_kv(cache)
            rebuild_count = previous.rebuild_count

        else:
            # Full prefill from position 0 (either cold start or window overflow rebuild)
            if previous is not None:
                # Window overflow: retain embeddings of last W blocks without re-running ViT
                combined_all_blocks = old_blocks + tuple(new_blocks)
                combined_blocks = combined_all_blocks[-w:]
                rebuild_count = previous.rebuild_count + 1
            else:
                # Cold start
                combined_blocks = tuple(new_blocks)[-w:]
                rebuild_count = 0

            cache = DynamicCache()
            all_inputs_embeds = torch.cat([b.inputs_embeds for b in combined_blocks], dim=1)
            combined_attention_mask = torch.cat([b.attention_mask for b in combined_blocks], dim=1)

            final_norm_h, inter_h = self._execute_native_layers(
                inputs_embeds=all_inputs_embeds,
                attention_mask_2d=combined_attention_mask,
                cache=cache,
                start_pos=0,
            )

            updated_layer_kv = extract_layer_kv(cache)

        # Features for the last block tokens in float32
        deep = final_norm_h[:, -last_block_len:, :].to(torch.float32)
        if shallow_idx == num_layers:
            # Equivalent to HF hidden_states[-1] which is final norm
            shallow = final_norm_h[:, -last_block_len:, :].to(torch.float32)
        else:
            # Output of layer shallow_idx before final norm
            shallow = inter_h[shallow_idx][:, -last_block_len:, :].to(torch.float32)

        next_state = NativeKVState(
            layer_kv=updated_layer_kv,
            blocks=combined_blocks,
            attention_mask=combined_attention_mask,
            last_frame_id=last_frame_id,
            frame_count=new_frame_count,
            prompt=prompt,
            owner=self,
            revision=self._revision,
            rebuild_count=rebuild_count,
        )

        return deep, shallow, next_state

    @torch.no_grad()
    def read_blocks(
        self,
        new_blocks: Sequence[NativeEmbeddingBlock],
        prompt: str,
        previous: Optional[NativeKVState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, NativeKVState]:
        """Process new embedding blocks with native incremental or rebuilt KV cache.

        Returns:
            deep_features: [1, last_block_len, hidden_dim] float32
            shallow_features: [1, last_block_len, hidden_dim] float32
            next_state: NativeKVState
        """
        if self.training:
            raise RuntimeError("read_blocks is only allowed in eval mode.")
        if not new_blocks:
            raise ValueError("new_blocks must not be empty.")

        w = self.config.max_frames

        # Validate owner, revision, prompt, and frame_id monotonicity across new_blocks
        expected_owner = self
        expected_rev = self._revision
        expected_prompt = prompt
        expected_dtype = new_blocks[0].inputs_embeds.dtype
        expected_device = new_blocks[0].inputs_embeds.device
        expected_seq_len = new_blocks[0].seq_len

        for i, block in enumerate(new_blocks):
            if block.owner is not expected_owner:
                raise ValueError(f"Block[{i}] owner {block.owner} does not match adapter {self}")
            if block.revision != expected_rev:
                raise ValueError(
                    f"Block[{i}] revision {block.revision} does not match adapter revision {expected_rev}"
                )
            if block.prompt != expected_prompt:
                raise ValueError(
                    f"Block[{i}] prompt '{block.prompt}' does not match requested prompt '{expected_prompt}'"
                )
            if block.inputs_embeds.dtype != expected_dtype or block.inputs_embeds.device != expected_device:
                raise ValueError("All new_blocks must share the same dtype and device")
            if block.seq_len != expected_seq_len:
                raise ValueError("All new_blocks must have identical seq_len")
            if i > 0 and block.frame_id <= new_blocks[i - 1].frame_id:
                raise ValueError(
                    f"new_blocks frame_ids must be strictly increasing: "
                    f"block[{i}].frame_id={block.frame_id} <= block[{i-1}].frame_id={new_blocks[i-1].frame_id}"
                )

        if previous is not None:
            self._validate_previous_state(previous, prompt)
            if new_blocks[0].frame_id <= previous.last_frame_id:
                raise ValueError(
                    f"First new block frame_id {new_blocks[0].frame_id} must be > "
                    f"previous last_frame_id {previous.last_frame_id}"
                )

        # Time checks
        if self.config.use_timestamps:
            if any(b.observation_time is None for b in new_blocks):
                raise ValueError("All new_blocks must have observation_time when use_timestamps is True")
            if previous is not None and any(b.observation_time is None for b in previous.blocks):
                raise ValueError("All previous blocks must have observation_time when use_timestamps is True")

        all_blocks_to_check = (previous.blocks if previous is not None else ()) + tuple(new_blocks)
        has_time_flags = [b.observation_time is not None for b in all_blocks_to_check]
        if any(has_time_flags):
            if not all(has_time_flags):
                raise ValueError("All blocks (previous and new) must have observation_time if any block has it")
            # Non-decreasing check
            for i in range(1, len(all_blocks_to_check)):
                if all_blocks_to_check[i].observation_time < all_blocks_to_check[i - 1].observation_time:
                    raise ValueError(
                        f"observation_time must be non-decreasing: "
                        f"block[{i}] ({all_blocks_to_check[i].observation_time}) < "
                        f"block[{i-1}] ({all_blocks_to_check[i-1].observation_time})"
                    )

        state = previous
        deep = None
        shallow = None
        for start in range(0, len(new_blocks), w):
            chunk = new_blocks[start : start + w]
            deep, shallow, state = self._read_block_group(chunk, prompt, state)

        return deep, shallow, state

    @torch.no_grad()
    def forward(
        self,
        images_window: Sequence[List[Any]],
        frame_ids: Sequence[int],
        prompt: str,
        previous: Optional[NativeKVState] = None,
        image_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        capture_times: Optional[Sequence[Optional[float]]] = None,
        observation_times: Optional[Sequence[Optional[float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, NativeKVState]:
        """Convenience forward: encode each new image frame and read_blocks."""
        if len(images_window) != len(frame_ids):
            raise ValueError(
                f"images_window length ({len(images_window)}) must match frame_ids length ({len(frame_ids)})"
            )
        if image_masks is not None and len(image_masks) != len(frame_ids):
            raise ValueError(
                f"image_masks length ({len(image_masks)}) must match frame_ids length ({len(frame_ids)})"
            )
        if capture_times is not None and len(capture_times) != len(frame_ids):
            raise ValueError(
                f"capture_times length ({len(capture_times)}) must match frame_ids length ({len(frame_ids)})"
            )
        if observation_times is not None and len(observation_times) != len(frame_ids):
            raise ValueError(
                f"observation_times length ({len(observation_times)}) must match frame_ids length ({len(frame_ids)})"
            )

        new_blocks: List[NativeEmbeddingBlock] = []
        for i, (imgs, fid) in enumerate(zip(images_window, frame_ids)):
            im_mask = image_masks[i] if image_masks is not None else None
            cap_time = capture_times[i] if capture_times is not None else None
            obs_time = observation_times[i] if observation_times is not None else None
            block = self.encode_frame(
                images=imgs,
                frame_id=fid,
                prompt=prompt,
                image_mask=im_mask,
                capture_time=cap_time,
                observation_time=obs_time,
            )
            new_blocks.append(block)

        return self.read_blocks(new_blocks=new_blocks, prompt=prompt, previous=previous)
