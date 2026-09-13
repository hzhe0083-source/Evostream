"""MOSS-style decoupled visual cross-attention memory for FabriVLA."""

from __future__ import annotations

import copy
from dataclasses import replace as dataclass_replace
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.delta import DeltaMemoryState, delta_read, delta_update


# This is a data-contract marker, not the mutable model ``_revision``.  A
# FrameKV made by one architecture must never be consumed by another one.
MOSS_ARCHITECTURE_REVISION = "moss_native_consume_v2"
FRAME_INDEX_COORDINATE = "frame_index"


@dataclass(frozen=True)
class MossConfig:
    cross_layers: Tuple[int, ...] = (3, 6, 10, 14)
    num_readout_tokens: int = 16
    max_frames: Optional[int] = 2
    shallow_layer: int = 6
    max_text_tokens: int = 256
    rope_base: float = 1e6
    memory_mode: str = "consume"
    train_vision: bool = False
    temporal_coordinate: str = FRAME_INDEX_COORDINATE
    architecture_revision: str = MOSS_ARCHITECTURE_REVISION

    def validate(self, num_native_layers: int) -> None:
        if self.memory_mode not in ("consume", "delta"):
            raise ValueError(f"memory_mode must be 'consume' or 'delta', got {self.memory_mode!r}")
        if not isinstance(self.architecture_revision, str) or not self.architecture_revision.strip():
            raise ValueError("architecture_revision must be a non-empty string")
        if self.temporal_coordinate != FRAME_INDEX_COORDINATE:
            raise ValueError(f"temporal_coordinate must be '{FRAME_INDEX_COORDINATE}'")
        if self.num_readout_tokens <= 0:
            raise ValueError(f"num_readout_tokens must be > 0, got {self.num_readout_tokens}")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {self.max_frames}")
        if self.max_text_tokens <= 0:
            raise ValueError(f"max_text_tokens must be > 0, got {self.max_text_tokens}")
        if not (math.isfinite(self.rope_base) and self.rope_base > 0):
            raise ValueError(f"rope_base must be finite and > 0, got {self.rope_base}")
        if not self.cross_layers:
            raise ValueError("cross_layers cannot be empty")
        prev = 0
        for lay in self.cross_layers:
            if not (1 <= lay <= num_native_layers):
                raise ValueError(f"cross layer index {lay} out of native range [1, {num_native_layers}]")
            if lay <= prev:
                raise ValueError(f"cross_layers must be strictly increasing and unique: {self.cross_layers}")
            prev = lay
        if not (1 <= self.shallow_layer <= num_native_layers):
            raise ValueError(
                f"shallow_layer index {self.shallow_layer} out of native range [1, {num_native_layers}]"
            )


@dataclass
class FrameKV:
    frame_id: int
    keys: Tuple[torch.Tensor, ...]
    values: Tuple[torch.Tensor, ...]
    owner: Any
    revision: int
    num_tokens: int
    # Original per-patch feature for the newest frame.  Keeping this allows
    # the native FabriVLA query path to remain intact while older frames live
    # only in independent K/V memory.
    native_features: Optional[torch.Tensor] = None
    architecture_revision: str = MOSS_ARCHITECTURE_REVISION
    temporal_coordinate: str = FRAME_INDEX_COORDINATE
    observation_time: Optional[float] = None


def build_causal_cross_mask(
    key_times: Sequence[float],
    key_token_counts: Sequence[int],
    query_times: Sequence[float],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive causal mask for cross-attention.

    ``key_times`` identifies each frame in the concatenated visual memory and
    ``key_token_counts`` expands it to patch-token positions.  A query can
    attend only to keys whose frame time is at most its own time.  The return
    shape is ``[1, 1, num_queries, num_key_tokens]`` and is broadcastable over
    batch and attention heads.
    """
    if len(key_times) != len(key_token_counts):
        raise ValueError(
            f"key_times ({len(key_times)}) and key_token_counts ({len(key_token_counts)}) must have equal length"
        )
    if not key_times:
        raise ValueError("key_times must be non-empty")
    if not query_times:
        raise ValueError("query_times must be non-empty")
    if any(int(n) != n or int(n) <= 0 for n in key_token_counts):
        raise ValueError(f"key_token_counts must contain positive integers, got {list(key_token_counts)}")

    key_time_tensor = torch.repeat_interleave(
        torch.as_tensor(key_times, dtype=torch.float32, device=device),
        torch.as_tensor(key_token_counts, dtype=torch.long, device=device),
    )
    query_time_tensor = torch.as_tensor(query_times, dtype=torch.float32, device=device)
    if not torch.isfinite(key_time_tensor).all() or not torch.isfinite(query_time_tensor).all():
        raise ValueError("key_times and query_times must be finite")

    allowed = key_time_tensor.unsqueeze(0) <= query_time_tensor.unsqueeze(1)
    if not allowed.any(dim=1).all():
        raise ValueError("every query time must have at least one visible key frame")
    mask = torch.zeros(allowed.shape, dtype=dtype, device=key_time_tensor.device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_3d_rope(
    head_dim: int,
    t_pos: torch.Tensor,
    row_pos: torch.Tensor,
    col_pos: torch.Tensor,
    base: float = 1e6,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_pairs = head_dim // 2
    dim_t_pairs = total_pairs // 4
    dim_spatial_pairs = total_pairs - dim_t_pairs
    dim_row_pairs = dim_spatial_pairs // 2
    dim_col_pairs = dim_spatial_pairs - dim_row_pairs

    inv_freq_t = 1.0 / (base ** (torch.arange(0, dim_t_pairs, dtype=torch.float32, device=device) / dim_t_pairs))
    inv_freq_row = 1.0 / (base ** (torch.arange(0, dim_row_pairs, dtype=torch.float32, device=device) / dim_row_pairs))
    inv_freq_col = 1.0 / (base ** (torch.arange(0, dim_col_pairs, dtype=torch.float32, device=device) / dim_col_pairs))

    freqs_t = torch.outer(t_pos.float(), inv_freq_t)
    freqs_row = torch.outer(row_pos.float(), inv_freq_row)
    freqs_col = torch.outer(col_pos.float(), inv_freq_col)

    freqs = torch.cat([freqs_t, freqs_row, freqs_col], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_3d_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    if cos.ndim == x_f.ndim - 1 and x_f.ndim >= 3:
        # [B,S,D] positions for [B,H,S,D] queries.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        while cos.ndim < x_f.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
    out = (x_f * cos) + (rotate_half(x_f) * sin)
    return out.to(orig_dtype)


class MossCrossAttentionBlock(nn.Module):
    def __init__(self, native_layer: nn.Module, memory_mode: str = "consume"):
        super().__init__()
        self.memory_mode = memory_mode
        self.q_proj = copy.deepcopy(native_layer.self_attn.q_proj).float()
        self.k_proj = copy.deepcopy(native_layer.self_attn.k_proj).float()
        self.v_proj = copy.deepcopy(native_layer.self_attn.v_proj).float()
        self.o_proj = copy.deepcopy(native_layer.self_attn.o_proj).float()

        self.q_norm = copy.deepcopy(native_layer.self_attn.q_norm).float()
        self.k_norm = copy.deepcopy(native_layer.self_attn.k_norm).float()

        self.input_layernorm = copy.deepcopy(native_layer.input_layernorm).float()
        self.post_attention_layernorm = copy.deepcopy(native_layer.post_attention_layernorm).float()
        self.mlp = copy.deepcopy(native_layer.mlp).float()

        dev = self.q_proj.weight.device
        self.attn_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=dev))
        self.mlp_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=dev))

        self.head_dim = getattr(native_layer.self_attn, "head_dim", 128)
        if self.head_dim < 8 or self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be >= 8 and even, got {self.head_dim}")
        if self.q_proj.out_features % self.head_dim != 0:
            raise ValueError(f"q_proj out_features {self.q_proj.out_features} not divisible by head_dim {self.head_dim}")
        if self.k_proj.out_features % self.head_dim != 0:
            raise ValueError(f"k_proj out_features {self.k_proj.out_features} not divisible by head_dim {self.head_dim}")

        self.num_heads = self.q_proj.out_features // self.head_dim
        self.num_kv_heads = self.k_proj.out_features // self.head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}")
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        if self.memory_mode == "delta":
            self.write_logits = nn.Parameter(torch.zeros(self.num_kv_heads, dtype=torch.float32, device=dev))
            self.memory_gate = nn.Parameter(torch.full((self.num_kv_heads,), 0.1, dtype=torch.float32, device=dev))

    def project_kv(self, visual_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = visual_features.shape
        vf_f = visual_features.float()

        k = self.k_proj(vf_f).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(vf_f).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)
        return k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        cross_attention_mask: Optional[torch.Tensor] = None,
        memory_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        residual = hidden_states

        normed = self.input_layernorm(hidden_states.float())
        B, S, _ = normed.shape
        q = self.q_proj(normed).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q_rot = apply_3d_rotary_pos_emb(self.q_norm(q), cos_q, sin_q)

        k = key_states.float()
        v = value_states.float()
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = F.scaled_dot_product_attention(
            q_rot, k, v, attn_mask=cross_attention_mask, dropout_p=0.0, is_causal=False, scale=scale
        )
        if memory_matrix is not None:
            R = delta_read(q_rot, memory_matrix)
            attn_out = attn_out + torch.tanh(self.memory_gate).repeat_interleave(self.num_kv_groups).view(1, self.num_heads, 1, 1) * R
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        attn_out = self.o_proj(attn_out)

        hidden_states = residual + (self.attn_gate.tanh() * attn_out).to(orig_dtype)

        residual = hidden_states
        normed_mlp = self.post_attention_layernorm(hidden_states.float())
        mlp_out = self.mlp(normed_mlp)
        hidden_states = residual + (self.mlp_gate.tanh() * mlp_out).to(orig_dtype)
        return hidden_states


class MossInternVL(nn.Module):
    def __init__(self, policy: Any, config: Optional[MossConfig] = None):
        super().__init__()
        self.policy = policy
        self.config = config or MossConfig()
        self.training_stage = "bridge"
        self._revision = 0

        for p in self.policy.parameters():
            p.requires_grad = False
        self.policy.eval()

        if hasattr(self.policy, "action_head") and self.policy.action_head is not None:
            self.policy.action_head.float()

        core = self.native_core
        self.config.validate(len(core.layers))

        self.cross_blocks = nn.ModuleDict()
        for layer_idx in self.config.cross_layers:
            native_layer = core.layers[layer_idx - 1]
            self.cross_blocks[str(layer_idx)] = MossCrossAttentionBlock(
                native_layer, memory_mode=self.config.memory_mode
            )

        dev = core.embed_tokens.weight.device
        hidden_size = core.config.hidden_size
        q_tokens = self.config.num_readout_tokens
        base_embed = None
        if hasattr(self.policy.embedder, "img_context_token_id"):
            img_ctx_id = self.policy.embedder.img_context_token_id
            if isinstance(img_ctx_id, int) and 0 <= img_ctx_id < core.embed_tokens.weight.shape[0]:
                with torch.no_grad():
                    base_embed = core.embed_tokens.weight[img_ctx_id].clone().float()

        if base_embed is not None:
            initial_readout = base_embed.unsqueeze(0).repeat(q_tokens, 1)
            initial_readout = initial_readout + torch.randn_like(initial_readout) * 0.01
        else:
            initial_readout = torch.randn(q_tokens, hidden_size, dtype=torch.float32, device=dev) * 0.01

        self.readout_embeddings = nn.Parameter(initial_readout.to(dev))

        self.set_training_stage("bridge")

    @property
    def native_core(self) -> Any:
        embedder = self.policy.embedder
        lm = embedder.model.language_model
        return lm.model if hasattr(lm, "model") else lm

    @property
    def architecture_revision(self) -> str:
        return self.config.architecture_revision

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._revision += 1
        self.policy.eval()
        if mode and self.training_stage in ("expert", "joint") and hasattr(self.policy, "action_head"):
            self.policy.action_head.train()
        return self

    def set_training_stage(self, stage: str, *, train_vision: Optional[bool] = None):
        if stage not in ("bridge", "expert", "joint"):
            raise ValueError(f"Unknown stage {stage}, choices: 'bridge', 'expert', 'joint'")
        if train_vision is not None:
            self.config = dataclass_replace(self.config, train_vision=bool(train_vision))
        self.training_stage = stage
        self._revision += 1
        for p in self.policy.parameters():
            p.requires_grad = False
        for p in self.bridge_parameters():
            p.requires_grad = True
        if stage in ("expert", "joint"):
            for p in self.action_parameters():
                p.requires_grad = True
            if hasattr(self.policy, "action_head") and self.policy.action_head is not None:
                self.policy.action_head.train() if self.training else self.policy.action_head.eval()
        if stage == "joint":
            for p in self.base_parameters(include_vision=self.config.train_vision):
                p.requires_grad = True

    def bridge_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.cross_blocks.parameters():
            yield p
        yield self.readout_embeddings

    def action_parameters(self) -> Iterator[nn.Parameter]:
        head = getattr(self.policy, "action_head", None)
        if head is not None:
            yield from head.parameters()

    @staticmethod
    def _is_vision_name(name: str) -> bool:
        lowered = name.lower()
        # Freeze the complete visual path by default.  The projector (mlp1)
        # must stay with the ViT so joint FP32 language updates do not create a
        # BF16/FP32 boundary inside extract_feature().
        return (
            "vision_model" in lowered
            or ".mlp1." in lowered
            or lowered.endswith(".mlp1.weight")
            or lowered.endswith(".mlp1.bias")
            or "vision_proj" in lowered
        )

    def base_parameters(self, *, include_vision: Optional[bool] = None) -> Iterator[nn.Parameter]:
        if include_vision is None:
            include_vision = self.config.train_vision
        action_ids = {id(p) for p in self.action_parameters()}
        for name, p in self.policy.named_parameters():
            if id(p) in action_ids:
                continue
            if not include_vision and self._is_vision_name(name):
                continue
            yield p

    def encode_image(self, images: List[Any]) -> torch.Tensor:
        def _encode():
            pixel_values, num_tiles_list = self.policy.embedder._preprocess_images(images)
            if len(num_tiles_list) != 1 or num_tiles_list[0] != 1:
                raise ValueError(
                    f"Strict single-view / single-tile required: got tiles {num_tiles_list} for {len(images)} images"
                )
            features = self.policy.embedder.model.extract_feature(pixel_values)
            if features.ndim == 2:
                features = features.unsqueeze(0)
            if features.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 from single-view image, got {features.shape[0]}")
            return features if self.config.train_vision else features.detach()
        if self.config.train_vision:
            return _encode()
        with torch.no_grad():
            return _encode()

    def encode_images_batch(self, images_batch: Sequence[List[Any]]) -> torch.Tensor:
        """Encode one single-tile image per item without re-running ViT per item."""
        if not images_batch:
            raise ValueError("images_batch must be non-empty")
        embedder = self.policy.embedder
        flat_images: List[Any] = []
        for images in images_batch:
            if not isinstance(images, (list, tuple)) or len(images) != 1:
                raise ValueError("each batch item must contain exactly one image/tile")
            flat_images.append(images[0])
        with torch.no_grad():
            if hasattr(embedder, "_preprocess_images_on_cpu"):
                pixel_values, num_tiles_list = embedder._preprocess_images_on_cpu(flat_images)
            else:
                processed = [embedder._preprocess_images([img]) for img in flat_images]
                pixel_values = torch.cat([x[0] for x in processed], dim=0)
                num_tiles_list = [n for x in processed for n in x[1]]
            if num_tiles_list != [1] * len(flat_images):
                raise ValueError(f"strict single-tile batch required, got {num_tiles_list}")
            device = self.native_core.embed_tokens.weight.device
            pixel_values = pixel_values.to(device=device)
            features = self.policy.embedder.model.extract_feature(pixel_values)
            if features.ndim == 2:
                features = features.unsqueeze(0)
            if features.ndim != 3 or features.shape[0] != len(flat_images):
                raise ValueError(f"expected batched visual features [B,P,H], got {tuple(features.shape)}")
            return features.detach()

    def _tokenize_native_prompt(self, prompt: str, seq_len: int) -> torch.Tensor:
        """Return padded ids used to locate native image context tokens."""
        embedder = self.policy.embedder
        tokenizer = getattr(embedder, "tokenizer", None)
        if tokenizer is None or not callable(tokenizer):
            raise AttributeError(
                "native multimodal path requires policy.embedder.tokenizer to locate image tokens"
            )
        if type(seq_len) is not int or seq_len <= 0:
            raise ValueError(f"native sequence length must be positive int, got {seq_len}")

        try:
            token_out = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
        except TypeError as exc:
            # Tiny/legacy tokenizers may only accept ``return_tensors``.  Keep
            # this narrow so errors raised inside a tokenizer are not hidden.
            msg = str(exc).lower()
            if "unexpected keyword" not in msg and "keyword argument" not in msg:
                raise
            token_out = tokenizer(prompt, return_tensors="pt")

        if isinstance(token_out, dict):
            input_ids = token_out.get("input_ids")
        else:
            input_ids = getattr(token_out, "input_ids", None)
        if input_ids is None:
            raise AttributeError("native tokenizer output must provide input_ids")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"native tokenizer input_ids must have shape [1,S], got {tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] > seq_len:
            input_ids = input_ids[:, :seq_len]
        elif input_ids.shape[1] < seq_len:
            pad_id = getattr(tokenizer, "pad_token_id", 0)
            pad_id = 0 if pad_id is None else int(pad_id)
            pad = torch.full(
                (1, seq_len - input_ids.shape[1]),
                pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            if getattr(tokenizer, "padding_side", "right") == "left":
                input_ids = torch.cat([pad, input_ids], dim=1)
            else:
                input_ids = torch.cat([input_ids, pad], dim=1)
        return input_ids

    def _prepare_native_queries_details(
        self, features: torch.Tensor, prompts: Sequence[str], *, require_image_mask: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Fuse native queries and resolve exact image positions when needed."""
        if features.ndim != 3 or features.shape[0] != len(prompts):
            raise ValueError("features must have shape [B,P,H] and match prompts")
        embedder = self.policy.embedder
        prompts_clean = [p.strip() for p in prompts]
        if any(not p for p in prompts_clean):
            raise ValueError("prompts must be non-empty")
        if not hasattr(embedder, "_build_multimodal_prompt"):
            raise AttributeError("native multimodal path requires embedder._build_multimodal_prompt")
        frame_prompts = []
        for p in prompts_clean:
            try:
                built = embedder._build_multimodal_prompt([1], p)
            except TypeError as exc:
                # A few native embedders expose the batch-shaped builder
                # ``([[tiles]], [prompts])`` instead of the single-frame
                # signature.  Fall back only for a signature mismatch.
                msg = str(exc).lower()
                if "argument" not in msg and "iterable" not in msg:
                    raise
                batch_built = embedder._build_multimodal_prompt([[1]], [p])
                if not isinstance(batch_built, (list, tuple)) or len(batch_built) != 1:
                    raise ValueError("batch multimodal prompt builder must return one prompt")
                built = batch_built[0]
            if isinstance(built, (list, tuple)):
                if len(built) != 1:
                    raise ValueError("_build_multimodal_prompt([1], prompt) must return one prompt")
                built = built[0]
            if not isinstance(built, str):
                raise TypeError(
                    f"_build_multimodal_prompt must return str for one frame, got {type(built).__name__}"
                )
            frame_prompts.append(built)
        masks = [torch.ones(1, dtype=torch.bool, device=features.device) for _ in prompts_clean]
        feature_list = [features[i : i + 1] for i in range(features.shape[0])]
        provided_image_mask: Optional[torch.Tensor] = None
        if hasattr(embedder, "_prepare_batch_and_fuse_embeddings"):
            fused_result = embedder._prepare_batch_and_fuse_embeddings(
                prompts=frame_prompts,
                vit_embeds_batch=feature_list,
                image_masks=masks,
                batch_num_tiles_list=[[1] for _ in prompts_clean],
            )
            if isinstance(fused_result, (tuple, list)) and len(fused_result) == 3:
                fused, attn, provided_image_mask = fused_result
            else:
                fused, attn = fused_result
        else:
            fused_parts, mask_parts = [], []
            for p, f, m in zip(frame_prompts, feature_list, masks):
                fused_result = embedder._prepare_and_fuse_embeddings(
                    prompt=p, vit_embeds=f, image_mask=m, num_tiles_list=[1]
                )
                if isinstance(fused_result, (tuple, list)) and len(fused_result) == 3:
                    fused_i, mask_i, image_mask_i = fused_result
                    if provided_image_mask is None:
                        provided_image_mask = []
                    if not isinstance(provided_image_mask, list):
                        raise ValueError("native fuser returned mixed image-mask formats")
                    provided_image_mask.append(image_mask_i)
                else:
                    fused_i, mask_i = fused_result
                fused_parts.append(fused_i)
                mask_parts.append(mask_i)
            fused, attn = torch.cat(fused_parts, dim=0), torch.cat(mask_parts, dim=0)
        core = self.native_core
        if not isinstance(fused, torch.Tensor) or fused.ndim != 3:
            raise ValueError(f"native fuser must return fused [B,S,H], got {type(fused).__name__}")
        if fused.shape[0] != len(prompts_clean) or fused.shape[2] != core.config.hidden_size:
            raise ValueError(
                f"native fused shape {tuple(fused.shape)} does not match "
                f"({len(prompts_clean)}, S, {core.config.hidden_size})"
            )
        if not isinstance(attn, torch.Tensor) or attn.shape != fused.shape[:2]:
            raise ValueError(
                f"native fuser attention mask must have shape {tuple(fused.shape[:2])}, "
                f"got {tuple(attn.shape) if isinstance(attn, torch.Tensor) else type(attn).__name__}"
            )

        image_mask: Optional[torch.Tensor] = None
        if require_image_mask:
            if isinstance(provided_image_mask, list):
                provided_image_mask = torch.cat(provided_image_mask, dim=0)
            if provided_image_mask is not None:
                if not isinstance(provided_image_mask, torch.Tensor) or provided_image_mask.shape != fused.shape[:2]:
                    raise ValueError(
                        f"native fuser image mask must have shape {tuple(fused.shape[:2])}"
                    )
                image_mask = provided_image_mask.to(device=fused.device, dtype=torch.bool)
                expected_counts = [int(x.shape[0]) for x in features]
                actual_counts = [int(x) for x in image_mask.sum(dim=1).tolist()]
                if actual_counts != expected_counts:
                    raise ValueError(
                        f"native fuser image-token counts {actual_counts} do not match visual token counts {expected_counts}"
                    )
            else:
                image_token_id = getattr(embedder, "img_context_token_id", None)
                if image_token_id is None:
                    tokenizer = getattr(embedder, "tokenizer", None)
                    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
                    if callable(convert):
                        image_token_id = convert("<IMG_CONTEXT>")
                if isinstance(image_token_id, torch.Tensor) and image_token_id.numel() == 1:
                    image_token_id = int(image_token_id.item())
                if type(image_token_id) is not int:
                    raise AttributeError(
                        "native multimodal path requires a valid embedder.img_context_token_id "
                        "or tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')"
                    )
                masks_by_sample = []
                for i, p in enumerate(frame_prompts):
                    ids = self._tokenize_native_prompt(p, int(fused.shape[1])).to(device=fused.device)
                    mask_i = ids.eq(image_token_id)
                    expected = int(features[i].shape[0])
                    actual = int(mask_i.sum().item())
                    if actual != expected:
                        raise ValueError(
                            f"native image-token count {actual} for sample {i} does not match "
                            f"visual token count {expected}"
                        )
                    masks_by_sample.append(mask_i)
                image_mask = torch.cat(masks_by_sample, dim=0).to(device=fused.device)

        return (
            fused.to(device=core.embed_tokens.weight.device, dtype=core.embed_tokens.weight.dtype),
            attn.to(device=core.embed_tokens.weight.device),
            image_mask.to(device=core.embed_tokens.weight.device) if image_mask is not None else None,
        )

    def prepare_native_queries(
        self, features: torch.Tensor, prompts: Sequence[str], *, return_image_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse precomputed visual features with unchanged native prompts.

        ``return_image_mask`` is opt-in to preserve the existing two-value API.
        """
        fused, attn, image_mask = self._prepare_native_queries_details(
            features, prompts, require_image_mask=return_image_mask
        )
        if return_image_mask:
            return fused, attn, image_mask  # type: ignore[return-value]
        return fused, attn

    def project_frame(
        self,
        features: torch.Tensor,
        frame_id: int,
        observation_time: Optional[float] = None,
        metadata: Optional[str] = None,
    ) -> FrameKV:
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"frame_id must be non-negative integer, got {frame_id} (type: {type(frame_id).__name__})")
        if features.ndim != 3:
            raise ValueError(f"visual features must have 3 dimensions [1, P, D], got shape {tuple(features.shape)}")
        B, P, D = features.shape
        if B != 1:
            raise ValueError(f"Expected batch size 1 for visual features, got {B}")
        if P <= 0:
            raise ValueError(f"Token count P must be > 0, got {P}")
        hidden_dim = self.native_core.config.hidden_size
        if D != hidden_dim:
            raise ValueError(f"Feature dim D={D} must match model hidden_size={hidden_dim}")

        grid_side = int(math.isqrt(P))
        if grid_side * grid_side != P:
            raise ValueError(f"Visual tokens P={P} must be a square number for grid coordinates")

        if observation_time is not None and (
            isinstance(observation_time, bool)
            or not isinstance(observation_time, (int, float))
            or not math.isfinite(float(observation_time))
            or float(observation_time) < 0.0
        ):
            raise ValueError(f"observation_time must be a finite non-negative number, got {observation_time!r}")
        if metadata is not None and (not isinstance(metadata, str) or not metadata.strip()):
            raise ValueError("metadata must be a non-empty string when provided")

        device = features.device
        row_indices = torch.arange(grid_side, device=device).repeat_interleave(grid_side)
        col_indices = torch.arange(grid_side, device=device).repeat(grid_side)
        # Use the same physical-time coordinate as the reader/cross mask.
        # Falling back to frame_id preserves callers that do not provide time.
        # Use frame order as the internal temporal coordinate.  Dataset video
        # timestamps (30 Hz) and online MuJoCo steps (80 Hz) have different
        # physical scales; frame_id keeps K/V RoPE consistent across them.
        frame_time = float(frame_id)
        time_indices = torch.full((P,), frame_time, dtype=torch.float32, device=device)

        native_features = features

        # Keep every original patch and optionally append one metadata token to
        # the independent visual memory only.  The language prompt is untouched,
        # so a zero cross gate remains exactly the native FabriVLA path.
        if observation_time is not None or metadata is not None:
            # Keep the textual marker on the same discrete coordinate used by
            # RoPE; physical seconds are recorded separately in the sample.
            metadata_text = metadata or f"Frame {frame_id}."
            tokenizer = getattr(self.policy.embedder, "tokenizer", None)
            if tokenizer is None:
                raise AttributeError("policy.embedder.tokenizer is required for frame metadata")
            token_out = tokenizer(metadata_text, return_tensors="pt")
            token_ids = token_out.input_ids.to(device=device)
            metadata_embed = self.native_core.embed_tokens(token_ids).float().mean(dim=1, keepdim=True)
            img_context_id = getattr(self.policy.embedder, "img_context_token_id", None)
            vocab_size = getattr(self.native_core.embed_tokens, "num_embeddings", self.native_core.embed_tokens.weight.shape[0])
            if isinstance(img_context_id, int) and 0 <= img_context_id < vocab_size:
                metadata_embed = metadata_embed + self.native_core.embed_tokens(
                    torch.tensor([[img_context_id]], dtype=torch.long, device=device)
                ).float()
            feature_dtype = features.dtype
            features = torch.cat([features.float(), metadata_embed.to(device=device)], dim=1).to(dtype=feature_dtype)
            P = P + 1
            row_indices = torch.cat([row_indices, torch.zeros(1, dtype=torch.long, device=device)])
            col_indices = torch.cat([col_indices, torch.zeros(1, dtype=torch.long, device=device)])
            time_indices = torch.cat([time_indices, torch.full((1,), frame_time, dtype=torch.float32, device=device)])

        first_block = next(iter(self.cross_blocks.values()))
        head_dim = first_block.head_dim

        cos_k, sin_k = compute_3d_rope(
            head_dim=head_dim,
            t_pos=time_indices,
            row_pos=row_indices,
            col_pos=col_indices,
            base=self.config.rope_base,
            device=device,
        )

        keys_list = []
        values_list = []
        for layer_idx in self.config.cross_layers:
            block = self.cross_blocks[str(layer_idx)]
            k, v = block.project_kv(features)
            keys_list.append(apply_3d_rotary_pos_emb(k, cos_k, sin_k))
            values_list.append(v)

        return FrameKV(
            frame_id=frame_id,
            keys=tuple(keys_list),
            values=tuple(values_list),
            owner=self,
            revision=self._revision,
            num_tokens=P,
            native_features=native_features,
            architecture_revision=self.config.architecture_revision,
            temporal_coordinate=self.config.temporal_coordinate,
            observation_time=float(observation_time) if observation_time is not None else None,
        )

    def _validate_frames(
        self, frames: Sequence[FrameKV], *, enforce_max_frames: bool = True
    ) -> None:
        if not frames:
            raise ValueError("frames sequence must be non-empty")
        if enforce_max_frames and self.config.max_frames is not None and len(frames) > self.config.max_frames:
            raise ValueError(f"frames length {len(frames)} exceeds max_frames {self.config.max_frames}")

        num_layers = len(self.config.cross_layers)
        for i, f in enumerate(frames):
            if f.owner is not self:
                raise ValueError(f"FrameKV at index {i} was created by foreign owner")
            if f.revision != self._revision:
                raise ValueError(f"FrameKV at index {i} has stale revision {f.revision} (current {self._revision})")
            if f.architecture_revision != self.config.architecture_revision:
                raise ValueError(
                    f"FrameKV at index {i} has architecture_revision {f.architecture_revision!r}; "
                    f"expected {self.config.architecture_revision!r}"
                )
            if f.temporal_coordinate != self.config.temporal_coordinate:
                raise ValueError(
                    f"FrameKV at index {i} has temporal_coordinate {f.temporal_coordinate!r}; "
                    f"expected {self.config.temporal_coordinate!r}"
                )
            if f.observation_time is not None and (
                isinstance(f.observation_time, bool)
                or not isinstance(f.observation_time, (int, float))
                or not math.isfinite(float(f.observation_time))
                or float(f.observation_time) < 0.0
            ):
                raise ValueError(f"FrameKV at index {i} has invalid observation_time {f.observation_time!r}")
            if type(f.frame_id) is not int or f.frame_id < 0:
                raise ValueError(f"FrameKV at index {i} has invalid non-negative frame_id {f.frame_id}")
            if i > 0 and f.frame_id <= frames[i - 1].frame_id:
                raise ValueError(f"FrameKV frame_ids must be strictly increasing: {frames[i-1].frame_id} -> {f.frame_id}")
            if f.num_tokens <= 0:
                raise ValueError(f"FrameKV at index {i} has invalid num_tokens={f.num_tokens}")
            if len(f.keys) != num_layers or len(f.values) != num_layers:
                raise ValueError(f"FrameKV at index {i} has mismatched layer count")

            for lay_pos, lay_idx in enumerate(self.config.cross_layers):
                block = self.cross_blocks[str(lay_idx)]
                k, v = f.keys[lay_pos], f.values[lay_pos]
                expected_shape = (1, block.num_kv_heads, f.num_tokens, block.head_dim)
                if k.shape != expected_shape or v.shape != expected_shape:
                    raise ValueError(
                        f"FrameKV {i} layer {lay_idx} shape mismatch: expected {expected_shape}, "
                        f"got k={tuple(k.shape)}, v={tuple(v.shape)}"
                    )

    def _execute_language_layers(
        self,
        frames: Sequence[FrameKV],
        prompt: str,
        frame_ids: Optional[Sequence[int]] = None,
        observation_times: Optional[Sequence[float]] = None,
        memory_matrices: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt must be non-empty")

        # In stream mode, add one lightweight placeholder per visual frame.
        # Patch tokens remain exclusively in FrameKV; these markers only give
        # the text-side reader the frame order and timestamp context.
        prompt_text = clean_prompt
        if frame_ids is not None:
            if len(frame_ids) != len(frames):
                raise ValueError(f"frame_ids length {len(frame_ids)} must match frames length {len(frames)}")
            times = observation_times if observation_times is not None else [float(x) for x in frame_ids]
            if len(times) != len(frames):
                raise ValueError(f"observation_times length {len(times)} must match frames length {len(frames)}")
            markers = "".join(
                f"Frame {int(fid)}, time {float(ts):.6f} s. <IMG_CONTEXT>\n"
                for fid, ts in zip(frame_ids, times)
            )
            prompt_text = f"{clean_prompt}\n{markers}"

        core = self.native_core
        base_dtype = core.embed_tokens.weight.dtype
        device = core.embed_tokens.weight.device

        # Preserve the native current-frame visual query sequence.  Historical
        # frames are represented only by FrameKV; the newest frame keeps the
        # original multimodal placeholder layout used by FabriVLA's expert.
        native_features = getattr(frames[-1], "native_features", None)
        can_fuse_native = hasattr(self.policy.embedder, "_build_multimodal_prompt")
        native_path = native_features is not None and can_fuse_native
        if native_path:
            require_image_mask = memory_matrices is not None or len(frames) > 1
            native_inputs, native_attention_mask, native_visual_query_mask = self._prepare_native_queries_details(
                native_features.to(device=device),
                [clean_prompt],
                require_image_mask=require_image_mask,
            )
            hidden_states = native_inputs.to(device=device, dtype=base_dtype)
            # Preserve the exact native multimodal sequence and padding mask.
            # Synthetic readout tokens would break gate=0 FabriVLA parity.
            attention_mask_2d = native_attention_mask.to(device=device)
        else:
            tokenizer = self.policy.embedder.tokenizer
            # Synthetic/legacy embedders do not understand multimodal frame
            # markers.  Keep their prompt unchanged; the native path above
            # handles exact image-token placement and timestamps.
            tokens = tokenizer(clean_prompt, return_tensors="pt")
            input_ids = tokens.input_ids.to(device)
            text_len = input_ids.shape[1]
            if text_len > self.config.max_text_tokens:
                raise ValueError(
                    f"Prompt token length {text_len} exceeds max_text_tokens {self.config.max_text_tokens}"
                )
            hidden_states = core.embed_tokens(input_ids)
            readout_embeds = self.readout_embeddings.unsqueeze(0).to(dtype=base_dtype, device=device)
            hidden_states = torch.cat([hidden_states, readout_embeds], dim=1)
            native_visual_query_mask = None
            attention_mask_2d = torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.bool, device=device
            )
        total_seq_len = hidden_states.shape[1]

        # Match native padding behavior for FA2; eager/SDPA also needs causal
        # visibility combined with the 2-D padding mask.
        attn_impl = getattr(core.config, "_attn_implementation", None)
        if attn_impl == "flash_attention_2":
            causal_mask = attention_mask_2d
        else:
            q_abs = torch.arange(total_seq_len, device=device).view(total_seq_len, 1)
            k_abs = torch.arange(total_seq_len, device=device).view(1, total_seq_len)
            allowed = (k_abs <= q_abs) & attention_mask_2d.bool().view(1, total_seq_len)
            if native_visual_query_mask is not None:
                padding_queries = ~attention_mask_2d.bool()
                allowed = allowed & ~padding_queries.unsqueeze(-1)
                diagonal = torch.eye(total_seq_len, dtype=torch.bool, device=device).unsqueeze(0)
                allowed = allowed | (padding_queries.unsqueeze(-1) & diagonal)
            causal_mask = torch.full(
                (1, 1, total_seq_len, total_seq_len), torch.finfo(base_dtype).min,
                device=device, dtype=base_dtype,
            )
            causal_mask.masked_fill_(allowed.view(1, 1, total_seq_len, total_seq_len), 0.0)

        position_ids = torch.arange(total_seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = core.rotary_emb(hidden_states, position_ids)

        if frame_ids is None:
            effective_frame_ids = [f.frame_id for f in frames]
        else:
            if len(frame_ids) != len(frames):
                raise ValueError(
                    f"frame_ids length {len(frame_ids)} must match frames length {len(frames)}"
                )
            effective_frame_ids = list(frame_ids)
            if any(type(fid) is not int or fid < 0 for fid in effective_frame_ids):
                raise ValueError(f"frame_ids must contain non-negative integers, got {effective_frame_ids}")
            if any(effective_frame_ids[i] <= effective_frame_ids[i - 1] for i in range(1, len(effective_frame_ids))):
                raise ValueError(f"frame_ids must be strictly increasing, got {effective_frame_ids}")
            frame_ids_from_payload = [f.frame_id for f in frames]
            if effective_frame_ids != frame_ids_from_payload:
                raise ValueError(
                    f"frame_ids {effective_frame_ids} do not match FrameKV ids {frame_ids_from_payload}"
                )

        if observation_times is None:
            provided_observation_times = [float(fid) for fid in effective_frame_ids]
        else:
            if len(observation_times) != len(frames):
                raise ValueError(
                    f"observation_times length {len(observation_times)} must match frames length {len(frames)}"
                )
            provided_observation_times = [float(t) for t in observation_times]
            if not all(math.isfinite(t) for t in provided_observation_times):
                raise ValueError("observation_times must contain finite values")
            if any(
                provided_observation_times[i] < provided_observation_times[i - 1]
                for i in range(1, len(provided_observation_times))
            ):
                raise ValueError(
                    "observation_times must be non-decreasing to preserve causal ordering"
                )

        # All internal temporal operations use the discrete frame coordinate;
        # the provided physical timestamps are validated above for provenance.
        effective_observation_times = [float(fid) for fid in effective_frame_ids]

        # Native consume reads the current frame through its original
        # multimodal sequence and historical frames through FrameKV.  Route it
        # through the batch reader so masking/deduplication stay identical for
        # single and batched calls.  Delta keeps the legacy matrix path below.
        if native_path and memory_matrices is None and len(frames) > 1:
            if native_visual_query_mask is None:
                raise RuntimeError("native image-token mask is required when historical memory is read")
            return self.read_native_queries_batch(
                [frames],
                hidden_states,
                attention_mask_2d,
                [effective_frame_ids[-1]],
                image_token_mask=native_visual_query_mask,
                exclude_current=True,
            )

        # The newest frame is already present in the native query sequence.
        # Only older frames are auxiliary memory on that path; this avoids
        # counting the current image twice when the gate opens.
        memory_frames = (
            frames
            if memory_matrices is not None
            else (frames[:-1] if native_path else frames)
        )
        memory_times = (
            effective_observation_times
            if memory_matrices is not None
            else (effective_observation_times[:-1] if native_path else effective_observation_times)
        )
        first_block = next(iter(self.cross_blocks.values()))
        expected_head_dim = first_block.head_dim
        curr_observation_time = effective_observation_times[-1]
        q_time_pos = torch.full((total_seq_len,), curr_observation_time, dtype=torch.float32, device=device)
        q_row_pos = torch.zeros((total_seq_len,), dtype=torch.long, device=device)
        q_col_pos = torch.zeros((total_seq_len,), dtype=torch.long, device=device)

        cos_q, sin_q = compute_3d_rope(
            head_dim=expected_head_dim,
            t_pos=q_time_pos,
            row_pos=q_row_pos,
            col_pos=q_col_pos,
            base=self.config.rope_base,
            device=device,
        )

        mem_keys: Dict[int, torch.Tensor] = {}
        mem_values: Dict[int, torch.Tensor] = {}
        cross_mask = None
        if memory_frames:
            cross_mask = build_causal_cross_mask(
                memory_times,
                [f.num_tokens for f in memory_frames],
                [curr_observation_time] * total_seq_len,
                device=device,
                dtype=torch.float32,
            )
        for l_idx, layer_num in enumerate(self.config.cross_layers):
            if memory_frames:
                layer_k_list = [f.keys[l_idx].to(device=device) for f in memory_frames]
                layer_v_list = [f.values[l_idx].to(device=device) for f in memory_frames]
                mem_keys[layer_num] = torch.cat(layer_k_list, dim=2)
                mem_values[layer_num] = torch.cat(layer_v_list, dim=2)

        shallow_states = None
        shallow_target = self.config.shallow_layer

        for layer_idx_0, native_layer in enumerate(core.layers):
            current_layer_1based = layer_idx_0 + 1

            layer_outputs = native_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if current_layer_1based in self.config.cross_layers and memory_frames:
                block = self.cross_blocks[str(current_layer_1based)]
                mat = memory_matrices.get(current_layer_1based) if memory_matrices is not None else None
                cross_hidden_states = block(
                    hidden_states=hidden_states,
                    key_states=mem_keys[current_layer_1based],
                    value_states=mem_values[current_layer_1based],
                    cos_q=cos_q,
                    sin_q=sin_q,
                    cross_attention_mask=cross_mask,
                    memory_matrix=mat,
                )
                if native_visual_query_mask is not None:
                    text_mask = (
                        ~native_visual_query_mask & attention_mask_2d.bool()
                    ).unsqueeze(-1)
                    hidden_states = torch.where(text_mask, cross_hidden_states, hidden_states)
                else:
                    hidden_states = cross_hidden_states

            if current_layer_1based == shallow_target:
                shallow_states = (
                    hidden_states if native_path else hidden_states[:, -self.config.num_readout_tokens :, :]
                ).float()

        if shallow_states is None:
            raise RuntimeError(f"Failed to capture shallow states at target layer {shallow_target}")

        hidden_states = core.norm(hidden_states)
        deep_states = (
            hidden_states if native_path else hidden_states[:, -self.config.num_readout_tokens :, :]
        ).float()
        return deep_states, shallow_states

    def read_native_queries_batch(
        self,
        frame_sets: Sequence[Sequence[FrameKV]],
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        current_frame_ids: Sequence[int],
        image_token_mask: Optional[torch.Tensor] = None,
        *,
        exclude_current: Optional[bool] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run native layers in batch while each sample reads its own FrameKV set.

        ``image_token_mask`` identifies the exact native visual query tokens.
        The matching current-frame KV is excluded because that frame is
        already represented by the native query sequence.  The optional mask
        only controls which query positions receive the residual; omitting it
        treats every valid token as a text/query token for legacy callers.
        """
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [B,S,H]")
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        if len(frame_sets) != batch_size or len(current_frame_ids) != batch_size:
            raise ValueError("frame_sets/current_frame_ids must match inputs_embeds batch size")
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(f"attention_mask must have shape {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}")
        if any(type(fid) is not int or fid < 0 for fid in current_frame_ids):
            raise ValueError("current_frame_ids must contain non-negative integers")
        if image_token_mask is not None and image_mask is not None:
            raise ValueError("pass only one of image_token_mask or image_mask")
        if image_token_mask is None:
            image_token_mask = image_mask
        if image_token_mask is None:
            image_mask = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool, device=inputs_embeds.device
            )
            if exclude_current is None:
                # Synthetic legacy queries are generally all-valid; a padded
                # sequence is native-like and already contains its current
                # frame, so exclude the matching KV in that case.
                exclude_current = not bool(torch.all(attention_mask.bool()).item())
        else:
            if image_token_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"image_token_mask must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(image_token_mask.shape)}"
                )
            image_mask = image_token_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if exclude_current is None:
                exclude_current = True
        assert exclude_current is not None
        for frames in frame_sets:
            self._validate_frames(frames, enforce_max_frames=not bool(exclude_current))
        core = self.native_core
        if hidden_size != core.config.hidden_size:
            raise ValueError(f"inputs_embeds hidden size {hidden_size} != native hidden size {core.config.hidden_size}")
        device = core.embed_tokens.weight.device
        h = inputs_embeds.to(device=device, dtype=core.embed_tokens.weight.dtype)
        native_mask_2d = attention_mask.to(device=device)
        image_mask = image_mask.to(device=device)
        valid_query_mask = native_mask_2d.bool() & ~image_mask
        attn_impl = getattr(core.config, "_attn_implementation", "eager")
        if attn_impl == "flash_attention_2":
            native_mask = native_mask_2d
        else:
            q_abs = torch.arange(seq_len, device=device).view(seq_len, 1)
            k_abs = torch.arange(seq_len, device=device).view(1, seq_len)
            allowed = (k_abs <= q_abs).unsqueeze(0) & native_mask_2d.bool().unsqueeze(1)
            if image_token_mask is not None:
                # Right-padding queries are not part of the native sequence
                # semantics.  Keep them self-contained so a cross residual on
                # an earlier text token cannot leak into padded outputs on a
                # later native layer.
                padding_queries = ~native_mask_2d.bool()
                allowed = allowed & ~padding_queries.unsqueeze(-1)
                diagonal = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
                allowed = allowed | (padding_queries.unsqueeze(-1) & diagonal)
            native_mask = torch.full(
                (batch_size, 1, seq_len, seq_len), torch.finfo(h.dtype).min,
                dtype=h.dtype, device=device
            )
            native_mask.masked_fill_(allowed.unsqueeze(1), 0.0)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = core.rotary_emb(h, position_ids)

        memory_sets: List[Tuple[FrameKV, ...]] = []
        for frames, current_id in zip(frame_sets, current_frame_ids):
            selected = tuple(
                f for f in frames
                if not (exclude_current and f.frame_id == current_id)
            )
            memory_sets.append(selected)

        max_memory_tokens = max(
            1,
            max((sum(f.num_tokens for f in frames) for frames in memory_sets), default=0),
        )
        cross_masks: Dict[int, torch.Tensor] = {}
        padded_keys: Dict[int, torch.Tensor] = {}
        padded_values: Dict[int, torch.Tensor] = {}
        has_visible_memory = torch.tensor(
            [any(frame.frame_id <= current_id for frame in frames)
             for frames, current_id in zip(memory_sets, current_frame_ids)],
            dtype=torch.bool,
            device=device,
        )
        for lay_pos, layer_idx in enumerate(self.config.cross_layers):
            block = self.cross_blocks[str(layer_idx)]
            k_batch = torch.zeros(
                (batch_size, block.num_kv_heads, max_memory_tokens, block.head_dim),
                dtype=torch.float32, device=device
            )
            v_batch = torch.zeros_like(k_batch)
            mask_batch = torch.full(
                (batch_size, 1, seq_len, max_memory_tokens), float("-inf"),
                dtype=torch.float32, device=device
            )
            for b, frames in enumerate(memory_sets):
                offset = 0
                for frame in frames:
                    n = frame.num_tokens
                    k_batch[b, :, offset : offset + n] = frame.keys[lay_pos].to(device=device, dtype=torch.float32)[0]
                    v_batch[b, :, offset : offset + n] = frame.values[lay_pos].to(device=device, dtype=torch.float32)[0]
                    if frame.frame_id <= current_frame_ids[b]:
                        mask_batch[b, 0, :, offset : offset + n] = 0.0
                    offset += n
                if not has_visible_memory[b]:
                    # SDPA returns NaNs for an all-masked row.  A dummy key is
                    # harmless because its cross result is never selected for
                    # this sample, and it keeps mixed current-only batches
                    # well-defined.
                    mask_batch[b, 0, :, 0] = 0.0
                    if not exclude_current:
                        raise ValueError(
                            f"sample {b} has no FrameKV visible at current_frame_id={current_frame_ids[b]}"
                        )
            padded_keys[layer_idx] = k_batch
            padded_values[layer_idx] = v_batch
            cross_masks[layer_idx] = mask_batch

        native_mask_after_cross = native_mask
        if image_token_mask is not None and attn_impl != "flash_attention_2":
            # Once historical cross attention has changed text/query states,
            # later causal layers must not feed that delta back into image or
            # right-padding queries.  Keep their native interactions with
            # other protected positions and their diagonal self path.
            protected_queries = (
                (~native_mask_2d.bool() | image_mask) & has_visible_memory.unsqueeze(1)
            )
            allowed_after = allowed & ~(
                protected_queries.unsqueeze(-1) & valid_query_mask.unsqueeze(1)
            )
            diagonal = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
            allowed_after = allowed_after | (protected_queries.unsqueeze(-1) & diagonal)
            native_mask_after_cross = torch.full(
                (batch_size, 1, seq_len, seq_len), torch.finfo(h.dtype).min,
                dtype=h.dtype, device=device
            )
            native_mask_after_cross.masked_fill_(allowed_after.unsqueeze(1), 0.0)

        shallow_states = None
        cross_seen = False
        for layer_num, native_layer in enumerate(core.layers, start=1):
            out = native_layer(
                h,
                attention_mask=native_mask_after_cross if cross_seen else native_mask,
                position_ids=position_ids,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=position_embeddings,
            )
            h = out[0] if isinstance(out, tuple) else out
            if layer_num in self.config.cross_layers and bool(has_visible_memory.any().item()):
                block = self.cross_blocks[str(layer_num)]
                cross_h = block(
                    hidden_states=h,
                    key_states=padded_keys[layer_num],
                    value_states=padded_values[layer_num],
                    cos_q=position_embeddings[0],
                    sin_q=position_embeddings[1],
                    cross_attention_mask=cross_masks[layer_num],
                )
                apply_cross = valid_query_mask & has_visible_memory.unsqueeze(1)
                h = torch.where(apply_cross.unsqueeze(-1), cross_h, h)
                cross_seen = True
            if layer_num == self.config.shallow_layer:
                shallow_states = h.float()
        if shallow_states is None:
            raise RuntimeError("Failed to capture shallow states")
        return core.norm(h).float(), shallow_states

    def read_native_queries(
        self,
        frames: Sequence[FrameKV],
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        current_frame_id: int,
        image_token_mask: Optional[torch.Tensor] = None,
        *,
        exclude_current: Optional[bool] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-sample convenience wrapper for ``read_native_queries_batch``."""
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise ValueError("inputs_embeds must have shape [1,S,H]")
        return self.read_native_queries_batch(
            [frames],
            inputs_embeds,
            attention_mask,
            [current_frame_id],
            image_token_mask=image_token_mask,
            exclude_current=exclude_current,
            image_mask=image_mask,
        )

    def read_memory(
        self,
        frames: Sequence[FrameKV],
        prompt: str,
        frame_ids: Optional[Sequence[int]] = None,
        observation_times: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.memory_mode == "delta":
            raise RuntimeError(
                "read_memory is only available in 'consume' mode. In 'delta' mode, use read_delta to avoid dropping state."
            )
        # Consume memory is episode-scoped; its session may contain more than
        # the legacy bounded training window.  Delta retains the explicit
        # bounded-state contract below.
        self._validate_frames(frames, enforce_max_frames=False)
        return self._execute_language_layers(
            frames,
            prompt,
            frame_ids=frame_ids,
            observation_times=observation_times,
            memory_matrices=None,
        )

    def read_delta(
        self,
        frames: Sequence[FrameKV],
        prompt: str,
        previous: Optional[DeltaMemoryState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, DeltaMemoryState]:
        if self.config.memory_mode != "delta":
            raise RuntimeError("read_delta is only supported when memory_mode='delta'")

        self._validate_frames(frames)
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt must be non-empty")

        num_cross_layers = len(self.config.cross_layers)
        core_device = self.native_core.embed_tokens.weight.device

        if previous is not None:
            if not isinstance(previous, DeltaMemoryState):
                raise TypeError(f"previous must be an instance of DeltaMemoryState, got {type(previous)}")
            if previous.owner is not self:
                raise ValueError("previous state was created by foreign owner")
            if previous.revision != self._revision:
                raise ValueError(f"previous state has stale revision {previous.revision} (current {self._revision})")
            if previous.prompt != clean_prompt:
                raise ValueError(f"previous state prompt {previous.prompt!r} does not match current prompt {clean_prompt!r}")
            if len(previous.matrices) != num_cross_layers:
                raise ValueError(
                    f"previous state matrices count {len(previous.matrices)} does not match cross layers count {num_cross_layers}"
                )
            if frames[0].frame_id <= previous.last_frame_id:
                raise ValueError(
                    f"First frame_id {frames[0].frame_id} must be strictly greater than previous last_frame_id {previous.last_frame_id}"
                )

            current_matrices = list(previous.matrices)
            for lay_pos, lay_idx in enumerate(self.config.cross_layers):
                m = current_matrices[lay_pos]
                block = self.cross_blocks[str(lay_idx)]
                expected_shape = (1, block.num_kv_heads, block.head_dim, block.head_dim)
                if m.shape != expected_shape:
                    raise ValueError(f"Matrix at layer {lay_idx} has shape {tuple(m.shape)}, expected {expected_shape}")
                if m.dtype != torch.float32:
                    raise ValueError(f"Matrix at layer {lay_idx} must be float32, got {m.dtype}")
                if m.device != core_device:
                    raise ValueError(f"Matrix at layer {lay_idx} device {m.device} does not match model device {core_device}")
            prev_frame_count = previous.frame_count
        else:
            current_matrices = []
            for lay_idx in self.config.cross_layers:
                block = self.cross_blocks[str(lay_idx)]
                s_init = torch.zeros(
                    (1, block.num_kv_heads, block.head_dim, block.head_dim),
                    dtype=torch.float32,
                    device=core_device,
                )
                current_matrices.append(s_init)
            prev_frame_count = 0

        # Sequentially update S frame-by-frame (pure functional, no in-place modification of previous S)
        for f in frames:
            next_m_list = []
            for lay_pos, lay_idx in enumerate(self.config.cross_layers):
                block = self.cross_blocks[str(lay_idx)]
                s_curr = current_matrices[lay_pos]
                k = f.keys[lay_pos].to(device=core_device)
                v = f.values[lay_pos].to(device=core_device)
                s_next = delta_update(s_curr, k, v, block.write_logits)
                next_m_list.append(s_next)
            current_matrices = next_m_list

        next_matrices_tuple = tuple(current_matrices)
        next_state = DeltaMemoryState(
            matrices=next_matrices_tuple,
            last_frame_id=frames[-1].frame_id,
            frame_count=prev_frame_count + len(frames),
            prompt=clean_prompt,
            owner=self,
            revision=self._revision,
        )

        matrix_dict = {lay_idx: next_matrices_tuple[i] for i, lay_idx in enumerate(self.config.cross_layers)}
        deep, shallow = self._execute_language_layers(frames, clean_prompt, memory_matrices=matrix_dict)
        return deep, shallow, next_state

    def forward_delta(
        self,
        images_window: Sequence[List[Any]],
        frame_ids: Sequence[int],
        prompt: str,
        previous: Optional[DeltaMemoryState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, DeltaMemoryState]:
        if len(images_window) != len(frame_ids):
            raise ValueError(
                f"Length mismatch: images_window ({len(images_window)}) vs frame_ids ({len(frame_ids)})"
            )
        frames = [self.project_frame(self.encode_image(imgs), frame_id=fid)
                  for imgs, fid in zip(images_window, frame_ids)]
        return self.read_delta(frames, prompt, previous=previous)

    def forward(
        self,
        images_window: Sequence[List[Any]],
        frame_ids: Sequence[int],
        prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(images_window) != len(frame_ids):
            raise ValueError(
                f"Length mismatch: images_window ({len(images_window)}) vs frame_ids ({len(frame_ids)})"
            )
        frames = [self.project_frame(self.encode_image(imgs), frame_id=fid)
                  for imgs, fid in zip(images_window, frame_ids)]
        return self.read_memory(frames, prompt)


_DEFAULT_SESSION_MAX_FRAMES = object()


class VisionSession:
    def __init__(self, model: MossInternVL, max_frames: Any = _DEFAULT_SESSION_MAX_FRAMES):
        self.model = model
        self.max_frames = model.config.max_frames if max_frames is _DEFAULT_SESSION_MAX_FRAMES else max_frames
        if self.max_frames is not None and (
            type(self.max_frames) is bool or not isinstance(self.max_frames, int) or self.max_frames <= 0
        ):
            raise ValueError(f"max_frames must be positive int or None, got {self.max_frames}")
        self._cache: deque[FrameKV] = deque(maxlen=self.max_frames)
        self._lock = threading.RLock()
        self.episode_id: Optional[str] = None
        self.prompt: Optional[str] = None
        self._model_revision: int = model._revision

    def reset(self, episode_id: str, prompt: str) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be non-empty string")
        clean_prompt = prompt.strip()
        tokenizer = getattr(self.model.policy.embedder, "tokenizer", None)
        if tokenizer is None or not callable(tokenizer):
            raise AttributeError("VisionSession requires policy.embedder.tokenizer")
        tokens = tokenizer(clean_prompt, return_tensors="pt")
        token_ids = tokens.get("input_ids") if isinstance(tokens, dict) else getattr(tokens, "input_ids", None)
        if token_ids is None or not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            raise ValueError("tokenizer output must provide 2D input_ids")
        text_len = token_ids.shape[1]
        if text_len > self.model.config.max_text_tokens:
            raise ValueError(
                f"Prompt token length {text_len} exceeds max_text_tokens {self.model.config.max_text_tokens}"
            )
        with self._lock:
            self._cache.clear()
            self.episode_id = episode_id.strip()
            self.prompt = clean_prompt
            self._model_revision = self.model._revision

    def append(self, images: List[Any], frame_id: int, observation_time: Optional[float] = None) -> None:
        if self.model.training:
            raise RuntimeError("VisionSession.append is only allowed when model is in eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError("VisionSession.append requires torch.no_grad() to be active")
        if self.prompt is None or self.episode_id is None:
            raise RuntimeError("VisionSession prompt and episode_id must be set via reset() before appending")
        if self._model_revision != self.model._revision:
            raise RuntimeError("VisionSession invalidated due to model revision change (train/stage update)")
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"frame_id must be non-negative integer, got {frame_id} (type: {type(frame_id).__name__})")
        if len(self._cache) > 0 and frame_id <= self._cache[-1].frame_id:
            raise ValueError(
                f"frame_id {frame_id} must be strictly greater than previous {self._cache[-1].frame_id}"
            )

        features = self.model.encode_image(images)
        self.append_frame(
            self.model.project_frame(features, frame_id=frame_id, observation_time=observation_time)
        )

    def append_frame(self, frame: FrameKV) -> None:
        """Append an already projected frame exactly once.

        This is used by the asynchronous consume path so vision encoding and
        K/V projection are never repeated during planning.
        """
        if self.model.training:
            raise RuntimeError("VisionSession.append_frame is only allowed when model is in eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError("VisionSession.append_frame requires torch.no_grad() to be active")
        if self.prompt is None or self.episode_id is None:
            raise RuntimeError("VisionSession prompt and episode_id must be set via reset() before appending")
        if self._model_revision != self.model._revision:
            raise RuntimeError("VisionSession invalidated due to model revision change (train/stage update)")
        if not isinstance(frame, FrameKV):
            raise TypeError(f"frame must be FrameKV, got {type(frame).__name__}")
        self.model._validate_frames([frame], enforce_max_frames=False)
        with self._lock:
            if len(self._cache) > 0 and frame.frame_id <= self._cache[-1].frame_id:
                raise ValueError(
                    f"frame_id {frame.frame_id} must be strictly greater than previous {self._cache[-1].frame_id}"
                )
            if self._cache and frame.observation_time is not None:
                previous_time = self._cache[-1].observation_time
                if previous_time is not None and frame.observation_time < previous_time:
                    raise ValueError(
                        f"observation_time {frame.observation_time} must be non-decreasing from {previous_time}"
                    )

            cached_frame = FrameKV(
                frame_id=frame.frame_id,
                keys=tuple(k.detach() for k in frame.keys),
                values=tuple(v.detach() for v in frame.values),
                owner=self.model,
                revision=frame.revision,
                num_tokens=frame.num_tokens,
                native_features=frame.native_features.detach() if frame.native_features is not None else None,
                architecture_revision=frame.architecture_revision,
                temporal_coordinate=frame.temporal_coordinate,
                observation_time=frame.observation_time,
            )
            self._cache.append(cached_frame)

    def snapshot(self, upto_frame_id: Optional[int] = None) -> Tuple[FrameKV, ...]:
        """Return an immutable prefix of the session for one planning cutoff."""
        with self._lock:
            if self.prompt is None or self.episode_id is None:
                raise RuntimeError("VisionSession not initialized. Call reset() first.")
            if upto_frame_id is None:
                return tuple(self._cache)
            if type(upto_frame_id) is not int or upto_frame_id < 0:
                raise ValueError(f"upto_frame_id must be a non-negative integer, got {upto_frame_id}")
            return tuple(frame for frame in self._cache if frame.frame_id <= upto_frame_id)

    def clear(self) -> None:
        """Release cached frames and episode identity."""
        with self._lock:
            self._cache.clear()
            self.episode_id = None
            self.prompt = None

    def query(self, upto_frame_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model.training:
            raise RuntimeError("VisionSession.query is only allowed when model is in eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError("VisionSession.query requires torch.no_grad() to be active")
        if self.prompt is None or self.episode_id is None:
            raise RuntimeError("VisionSession not initialized. Call reset() first.")
        frames = self.snapshot(upto_frame_id)
        if len(frames) == 0:
            raise RuntimeError("Cannot query an empty VisionSession")
        if self._model_revision != self.model._revision:
            raise RuntimeError("VisionSession invalidated due to model revision change")

        frame_ids = [frame.frame_id for frame in frames]
        times = [frame.observation_time for frame in frames]
        if all(t is not None for t in times):
            return self.model.read_memory(
                frames,
                self.prompt,
                frame_ids=frame_ids,
                observation_times=[float(t) for t in times if t is not None],
            )
        return self.model.read_memory(frames, self.prompt, frame_ids=frame_ids)

    @property
    def frames(self) -> Tuple[FrameKV, ...]:
        with self._lock:
            return tuple(self._cache)


class FrameKVSession(VisionSession):
    """Unbounded episode-scoped projected-frame session for consume planning."""

    def __init__(self, model: MossInternVL):
        super().__init__(model, max_frames=None)

    def reset(self, episode_id: str, prompt: str) -> None:
        # The async vision worker may be using the model tokenizer while a new
        # episode is reset.  Prompt-length validation belongs to the native
        # fuser; avoid a second concurrent tokenizer call here.
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be non-empty string")
        with self._lock:
            self._cache.clear()
            self.episode_id = episode_id.strip()
            self.prompt = prompt.strip()
            self._model_revision = self.model._revision
