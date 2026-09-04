"""Truncated MOSS-VL backbone and the MOSS-Action policy.

The policy keeps an append-only streaming visual KV cache and decodes a short
continuous action chunk from a handful of *ephemeral* action queries. The
queries live only for the duration of one planning call: their key/value entries
are cropped out of the cache immediately afterwards, so a later frame can never
attend to a plan that was never executed.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

RETAINED_LAYERS = 24
RETAINED_CROSS_ATTENTION_LAYERS = (2, 6, 10, 14, 18, 22)
# Only used by the offline parity CLI; the policy reads the final hidden state.
TAP_LAYERS = (14, 18, 23)


@dataclass(frozen=True)
class MossActionConfig:
    moss_hidden_size: int = 4096
    state_dim: int = 9
    action_dim: int = 7
    chunk_size: int = 8
    action_hidden_size: int = 1024
    control_interval: float = 0.1
    delay_scale: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.moss_hidden_size,
            self.state_dim,
            self.action_dim,
            self.chunk_size,
            self.action_hidden_size,
        )
        if any(value < 1 for value in positive):
            raise ValueError("all dimensions and counts must be positive")
        if self.control_interval <= 0 or self.delay_scale <= 0:
            raise ValueError("control_interval and delay_scale must be positive")

    @property
    def condition_dim(self) -> int:
        """Robot state, its one-step difference, and the newest frame's age."""
        return 2 * self.state_dim + 1

    @property
    def micro_horizon(self) -> float:
        """Physical seconds covered by one chunk at the training control rate."""
        return self.chunk_size * self.control_interval


@dataclass(frozen=True)
class BackboneTaps:
    hidden: tuple[torch.Tensor, ...]
    text_mask: torch.Tensor


@dataclass
class MossStreamState:
    """Incremental text/vision cache owned by one robot episode."""

    past_key_values: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    next_text_position: int
    full_vision_token_info: list[dict[str, Any]] | None = None
    vision_tokens: int = 0
    frame_count: int = 0


@dataclass(frozen=True)
class ActionChunk:
    """One plan: `actions[j]` is meant to execute at `start_time + j * interval`."""

    actions: torch.Tensor
    start_time: float
    plan_time: float
    interval: float
    visual_age: float


@dataclass(frozen=True)
class CheckpointAudit:
    deleted_layer_indices: tuple[int, ...]
    discarded_after_load: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _moss_core(module: nn.Module) -> nn.Module:
    """Return the inner MossVLModel, whether given it or the CausalLM wrapper."""
    candidates = [getattr(module, "model", None), module]
    base_model = getattr(module, "base_model", None)
    candidates.extend(
        [
            base_model,
            getattr(base_model, "model", None),
            getattr(getattr(base_model, "model", None), "model", None),
        ]
    )
    for candidate in candidates:
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            return candidate
    raise TypeError("expected a MOSS-VL model exposing language_model.layers")


def _language_layers(module: nn.Module) -> nn.ModuleList:
    layers = _moss_core(module).language_model.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("MOSS-VL language_model.layers must be an nn.ModuleList")
    return layers


def validate_layer_identity(module: nn.Module) -> None:
    layers = _language_layers(module)
    if len(layers) != RETAINED_LAYERS:
        raise ValueError(
            f"truncated MOSS-VL must contain exactly {RETAINED_LAYERS} layers"
        )
    actual = tuple(
        index
        for index, layer in enumerate(layers)
        if hasattr(layer, "cross_attn_attn_gate")
        or type(layer).__name__ == "MossVLCrossAttentionDecoderLayer"
    )
    if actual != RETAINED_CROSS_ATTENTION_LAYERS:
        raise ValueError(
            "retained native cross-attention layers mismatch: "
            f"expected={RETAINED_CROSS_ATTENTION_LAYERS}, actual={actual}"
        )


def _layer_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError(f"cannot extract a hidden tensor from {type(output).__name__}")


def realtime_frame_segment(timestamp: float) -> str:
    """Return the exact single-frame text wrapper used by MOSS realtime inference."""
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("relative frame timestamp must be finite and nonnegative")
    return (
        "<|silence|><|vision_start|><|time_start|>"
        f"{timestamp:.1f} seconds"
        "<|time_end|><|image|><|vision_end|>"
    )


def _realtime_mrope_positions(
    input_ids: torch.Tensor,
    grid_thw: torch.Tensor,
    start: int,
    image_token_id: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build MOSS XRoPE positions for arbitrary batch sizes and frame counts."""
    batch_size, seq_len = input_ids.shape
    positions = torch.zeros(
        3, batch_size, seq_len, dtype=torch.long, device=input_ids.device
    )
    all_vision_chunks = []
    frame_offset = 0
    max_next_position = start

    for b in range(batch_size):
        current = start
        sample_vision_chunks = []
        for token_index, token in enumerate(input_ids[b]):
            if int(token) != image_token_id:
                positions[:, b, token_index] = current
                current += 1
                continue
            if frame_offset >= len(grid_thw):
                raise ValueError("stream segment has more image tokens than frames")
            _, grid_h, grid_w = map(int, grid_thw[frame_offset].tolist())
            height, width = grid_h // merge_size, grid_w // merge_size
            y = torch.arange(height, device=input_ids.device).view(-1, 1)
            x = torch.arange(width, device=input_ids.device).view(1, -1)
            base = torch.full(
                (height, width), current, dtype=torch.long, device=input_ids.device
            )
            grid_positions = torch.stack(
                (base, base + y, base + x), dim=0
            ).reshape(3, -1)
            separator = current + max(height, width)
            sample_vision_chunks.append(
                torch.cat(
                    (
                        grid_positions,
                        torch.full(
                            (3, 1), separator, dtype=torch.long, device=input_ids.device
                        ),
                    ),
                    dim=1,
                )
            )
            positions[:, b, token_index] = separator
            current = separator + 1
            frame_offset += 1
        max_next_position = max(max_next_position, current)
        if sample_vision_chunks:
            all_vision_chunks.append(torch.cat(sample_vision_chunks, dim=1))

    if frame_offset != len(grid_thw):
        raise ValueError(
            f"stream segment image tokens ({frame_offset}) mismatch frames ({len(grid_thw)})"
        )
    vision_positions = (
        torch.cat(all_vision_chunks, dim=1).unsqueeze(1)
        if all_vision_chunks
        else torch.zeros(3, 1, 0, dtype=torch.long, device=input_ids.device)
    )
    return positions, vision_positions, max_next_position


class TruncatedMossBackbone(nn.Module):
    """MOSS-VL vision encoder plus raw decoder outputs at H14/H18/H23."""

    def __init__(self, moss_model: nn.Module):
        super().__init__()
        self.moss = _moss_core(moss_model)
        validate_layer_identity(self.moss)

    @property
    def layers(self) -> nn.ModuleList:
        return _language_layers(self.moss)

    def _call_with_taps(
        self, module: nn.Module, inputs: Mapping[str, Any]
    ) -> tuple[Any, BackboneTaps]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for index in TAP_LAYERS:

            def capture(
                _module: nn.Module, _args: tuple[Any, ...], output: Any, *, layer=index
            ) -> None:
                captured[layer] = _layer_tensor(output)

            handles.append(self.layers[index].register_forward_hook(capture))
        try:
            output = module(**dict(inputs))
        finally:
            for handle in handles:
                handle.remove()
        missing = set(TAP_LAYERS) - set(captured)
        if missing:
            raise RuntimeError(
                f"MOSS-VL did not execute tap layers {sorted(missing)}; "
                "a visible image and its cross-attention mask must be present"
            )
        mask = inputs.get("attention_mask")
        if mask is None:
            batch, tokens = captured[TAP_LAYERS[-1]].shape[:2]
            mask = torch.ones(
                batch,
                tokens,
                dtype=torch.bool,
                device=captured[TAP_LAYERS[-1]].device,
            )
        else:
            mask = mask[:, -captured[TAP_LAYERS[-1]].shape[1] :].to(
                device=captured[TAP_LAYERS[-1]].device, dtype=torch.bool
            )
        return output, BackboneTaps(
            tuple(captured[index] for index in TAP_LAYERS), mask
        )

    def _forward_with_taps(
        self, inputs: Mapping[str, Any]
    ) -> tuple[Any, BackboneTaps]:
        return self._call_with_taps(self.moss, inputs)

    def forward(self, **moss_inputs: Any) -> BackboneTaps:
        inputs = dict(moss_inputs)
        inputs.pop("labels", None)
        inputs["use_cache"] = False
        _, taps = self._forward_with_taps(inputs)
        return taps

    def forward_with_queries(
        self,
        moss_inputs: Mapping[str, Any],
        query_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run language_model with action queries appended, with explicit positions and masks.

        This is the exact full-recompute analogue of `decode_action_queries`:
        1. Encodes vision features once via vision encoder.
        2. Computes 3D MRoPE position_ids for text tokens and assigns queries
           consecutive positions starting at `next_text_position`.
        3. Constructs a cross_attention_mask where action queries have full visibility
           into all historical visual frames (with padding tail masked out).
        4. Calls language_model directly, bypassing any top-level wrapper assumptions.
        """
        values = dict(moss_inputs)
        batch, queries, width = query_embeds.shape
        input_ids = values.get("input_ids")
        pixel_values = values.get("pixel_values")
        grid_thw = values.get("grid_thw")
        media_nums_per_sample = values.get("media_nums_per_sample")

        if input_ids is not None and pixel_values is not None and grid_thw is not None:
            # Full multimodal path (standard training and parity)
            config = self.moss.config
            image_token_id = int(config.image_token_id)
            merge_size = int(self.moss.visual.spatial_merge_size)

            vision_method = getattr(self.moss, "get_vision_features_chunked", None)
            if not callable(vision_method):
                vision_method = self.moss.get_vision_features
            vision_states, info = vision_method(
                pixel_values, grid_thw, media_nums_per_sample
            )
            vision_states = vision_states.to(
                device=input_ids.device,
                dtype=self.moss.get_input_embeddings().weight.dtype,
            )

            text_pos, vis_pos, next_position = _realtime_mrope_positions(
                input_ids, grid_thw, 0, image_token_id, merge_size
            )
            query_pos = (
                next_position
                + torch.arange(queries, device=input_ids.device)
            ).view(1, 1, -1).expand(3, batch, -1)
            full_text_pos = torch.cat((text_pos, query_pos), dim=-1)

            prefix_embeds = self.moss.get_input_embeddings()(input_ids)
            full_embeds = torch.cat(
                (prefix_embeds, query_embeds.to(prefix_embeds)), dim=1
            )

            raw_attention = values.get("attention_mask")
            if raw_attention is None:
                raw_attention = torch.ones_like(input_ids)
            query_attention = torch.ones(
                batch, queries, dtype=raw_attention.dtype, device=raw_attention.device
            )
            full_attention = torch.cat((raw_attention, query_attention), dim=1)

            frames_per_sample = (
                max(media_nums_per_sample)
                if media_nums_per_sample is not None
                else int((input_ids == image_token_id).sum(dim=1).max().item())
            )
            visible_prefix = (input_ids == image_token_id).cumsum(dim=1).unsqueeze(-1) > torch.arange(
                frames_per_sample, device=input_ids.device
            )
            # Action queries are causally after all frames, so they can see all frames.
            visible_queries = torch.ones(
                batch, queries, frames_per_sample, dtype=torch.bool, device=input_ids.device
            )
            full_visible = torch.cat((visible_prefix, visible_queries), dim=1)
            cross_attention_mask = (~full_visible).unsqueeze(1)
            cross_attention_mask = self.moss._expand_cross_attention_mask(
                cross_attention_mask, info, target_dtype=vision_states.dtype
            )
            minimum = torch.finfo(cross_attention_mask.dtype).min
            visible_rows = (cross_attention_mask != minimum).any(dim=-1).to(
                cross_attention_mask.dtype
            )[..., None]
            cross_attention_mask = cross_attention_mask * visible_rows

            call = {
                "input_ids": None,
                "inputs_embeds": full_embeds,
                "attention_mask": full_attention,
                "position_ids": full_text_pos,
                "cross_attention_states": vision_states,
                "vision_position_ids": vis_pos,
                "cross_attention_mask": cross_attention_mask,
                "full_text_row_masked_out_mask": visible_rows,
                "use_cache": False,
            }
            output = self.moss.language_model(**call)
            hidden = _layer_tensor(output)
            return hidden[:, -queries:]

        # Fallback for synthetic/text-only test stubs
        input_ids = values.pop("input_ids", None)
        inputs_embeds = values.pop("inputs_embeds", None)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("backbone requires input_ids or inputs_embeds")
            inputs_embeds = self.moss.get_input_embeddings()(input_ids)
        full_embeds = torch.cat(
            (inputs_embeds, query_embeds.to(inputs_embeds)), dim=1
        )
        mask = values.get("attention_mask")
        if mask is not None:
            query_mask = torch.ones(
                batch, queries, dtype=mask.dtype, device=mask.device
            )
            values["attention_mask"] = torch.cat((mask, query_mask), dim=1)
        values["inputs_embeds"] = full_embeds
        values["use_cache"] = False
        output = self.moss(**values)
        hidden = _layer_tensor(output)
        return hidden[:, -queries:]

    @torch.no_grad()
    def start_stream(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> MossStreamState:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("streaming MOSS requires input_ids with shape [1, tokens]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("streaming attention_mask must align with input_ids")
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        output = self.moss.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise RuntimeError("MOSS-VL did not return a cache during stream prefill")
        return MossStreamState(
            cache,
            input_ids,
            attention_mask,
            input_ids.shape[1],
        )

    @torch.no_grad()
    def append_stream_frame(
        self, state: MossStreamState, frame_inputs: Mapping[str, Any]
    ) -> BackboneTaps:
        values = dict(frame_inputs)
        input_ids = values.get("input_ids")
        grid_thw = values.get("grid_thw")
        pixel_values = values.get("pixel_values")
        if input_ids is None or grid_thw is None or pixel_values is None:
            raise ValueError("stream frame requires input_ids, grid_thw, and pixel_values")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("stream frame input_ids must have shape [1, tokens]")
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("stream grid_thw must have shape [frames, 3]")
        if not bool((grid_thw[:, 0] == 1).all()):
            raise ValueError("append_stream_frame accepts individually encoded frames only")

        cache_layers = getattr(state.past_key_values, "layers", ())
        for layer_index in RETAINED_CROSS_ATTENTION_LAYERS:
            if layer_index >= len(cache_layers):
                continue
            cache_layer = cache_layers[layer_index]
            if cache_layer.get_seq_length() > state.vision_tokens:
                cache_layer.crop(state.vision_tokens)

        config = self.moss.config
        image_token_id = int(config.image_token_id)
        merge_size = int(self.moss.visual.spatial_merge_size)
        text_positions, vision_positions, next_position = _realtime_mrope_positions(
            input_ids,
            grid_thw,
            state.next_text_position,
            image_token_id,
            merge_size,
        )

        vision_method = getattr(self.moss, "get_vision_features_chunked", None)
        if not callable(vision_method):
            vision_method = self.moss.get_vision_features
        vision_states, current_info = vision_method(
            pixel_values,
            grid_thw,
            values.get("media_nums_per_sample"),
        )
        vision_states = vision_states.to(
            device=input_ids.device,
            dtype=self.moss.get_input_embeddings().weight.dtype,
        )
        actual_new_tokens = int(current_info[0]["total_length"])
        padded_new_tokens = vision_states.shape[1]
        vision_cache_position = torch.arange(
            state.vision_tokens,
            state.vision_tokens + padded_new_tokens,
            device=input_ids.device,
        )
        if padded_new_tokens > actual_new_tokens:
            vision_positions = F.pad(
                vision_positions, (0, padded_new_tokens - actual_new_tokens)
            )

        medias = (
            []
            if state.full_vision_token_info is None
            else list(state.full_vision_token_info[0]["medias"])
        )
        for media in current_info[0]["medias"]:
            shifted = dict(media)
            shifted["start"] = state.vision_tokens + int(media["start"])
            shifted["end"] = state.vision_tokens + int(media["end"])
            medias.append(shifted)
        full_vision_token_info = [
            {
                "medias": medias,
                "total_length": state.vision_tokens + actual_new_tokens,
                "pad_start": state.vision_tokens + actual_new_tokens,
                "pad_end": state.vision_tokens + padded_new_tokens,
            }
        ]

        new_attention = values.get("attention_mask")
        if new_attention is None:
            new_attention = torch.ones_like(input_ids)
        full_input_ids = torch.cat((state.input_ids, input_ids), dim=1)
        full_attention = torch.cat((state.attention_mask, new_attention), dim=1)
        total_frames = state.frame_count + len(grid_thw)
        visible = (full_input_ids == image_token_id).cumsum(dim=1).unsqueeze(-1) > torch.arange(
            total_frames, device=input_ids.device
        )
        cross_attention_mask = (~visible).unsqueeze(1)[:, :, -input_ids.shape[1] :, :]
        cross_attention_mask = self.moss._expand_cross_attention_mask(
            cross_attention_mask,
            full_vision_token_info,
            target_dtype=vision_states.dtype,
        )
        minimum = torch.finfo(cross_attention_mask.dtype).min
        visible_rows = (cross_attention_mask != minimum).any(dim=-1).to(
            cross_attention_mask.dtype
        )[..., None]
        cross_attention_mask = cross_attention_mask * visible_rows
        cache_position = torch.arange(
            state.input_ids.shape[1],
            full_input_ids.shape[1],
            device=input_ids.device,
        )
        call = {
            "input_ids": None,
            "attention_mask": full_attention,
            "position_ids": text_positions,
            "past_key_values": state.past_key_values,
            "inputs_embeds": self.moss.get_input_embeddings()(input_ids),
            "cross_attention_states": vision_states,
            "vision_position_ids": vision_positions,
            "cross_attention_mask": cross_attention_mask,
            "full_text_row_masked_out_mask": visible_rows,
            "cache_position": cache_position,
            "vision_cache_position": vision_cache_position,
            "use_cache": True,
        }
        output, taps = self._call_with_taps(self.moss.language_model, call)
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise RuntimeError("MOSS-VL dropped its cache while appending a frame")
        state.past_key_values = cache
        state.input_ids = full_input_ids
        state.attention_mask = full_attention
        state.next_text_position = next_position
        state.full_vision_token_info = full_vision_token_info
        state.vision_tokens += actual_new_tokens
        state.frame_count = total_frames
        return BackboneTaps(
            tuple(hidden[:, -1:, :] for hidden in taps.hidden),
            torch.ones(1, 1, dtype=torch.bool, device=input_ids.device),
        )

    @contextmanager
    def _ephemeral_cache(self, state: MossStreamState) -> Iterator[None]:
        """Restore every self-attention cache length after a throwaway branch.

        Cross-attention layers hold vision keys rather than text keys, so their
        length is governed by `state.vision_tokens` and must be left alone; the
        action branch carries no new frame.
        """
        cache_layers = getattr(state.past_key_values, "layers", ())
        committed = {
            index: layer.get_seq_length()
            for index, layer in enumerate(cache_layers)
            if index not in RETAINED_CROSS_ATTENTION_LAYERS
        }
        try:
            yield
        finally:
            for index, length in committed.items():
                layer = cache_layers[index]
                if layer.get_seq_length() > length:
                    layer.crop(length)
                if layer.get_seq_length() != length:
                    raise RuntimeError(
                        "ephemeral action queries corrupted the streaming cache: "
                        f"layer {index} is {layer.get_seq_length()}, expected {length}"
                    )

    @torch.no_grad()
    def decode_action_queries(
        self, state: MossStreamState, query_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Run action queries against the committed prefix, then drop their KV.

        The persistent stream is left byte-identical: no new frame is appended,
        `input_ids` / `attention_mask` / vision bookkeeping are untouched, and
        the queries' key-value entries are cropped on the way out.
        """
        if query_embeds.ndim != 3 or query_embeds.shape[0] != 1:
            raise ValueError("action queries must have shape [1, chunk, hidden]")
        queries = query_embeds.shape[1]
        if queries < 1:
            raise ValueError("a planning call needs at least one action query")
        prefix_tokens = state.input_ids.shape[1]
        cache_position = torch.arange(
            prefix_tokens,
            prefix_tokens + queries,
            device=query_embeds.device,
        )
        # Consecutive positions: p_j = next_text_position + j, exactly matching
        # the training-time forward_with_queries path.
        position_ids = (
            state.next_text_position
            + torch.arange(queries, device=query_embeds.device)
        ).view(1, 1, -1).expand(3, 1, -1)
        attention_mask = torch.cat(
            (
                state.attention_mask,
                torch.ones(
                    1, queries, dtype=state.attention_mask.dtype, device=query_embeds.device
                ),
            ),
            dim=1,
        )
        # Cross-attention layers reuse the cached vision keys, but the cache also
        # holds padding slots. Every frame so far is visible to every query; the
        # expansion below is what masks the padding tail.
        cross_attention_mask = None
        full_text_row_masked_out_mask = None
        if state.full_vision_token_info is not None and state.frame_count:
            visible = torch.zeros(
                1,
                1,
                queries,
                state.frame_count,
                dtype=torch.bool,
                device=query_embeds.device,
            )
            cross_attention_mask = self.moss._expand_cross_attention_mask(
                visible,
                state.full_vision_token_info,
                target_dtype=query_embeds.dtype,
            )
            minimum = torch.finfo(cross_attention_mask.dtype).min
            full_text_row_masked_out_mask = (
                (cross_attention_mask != minimum)
                .any(dim=-1)
                .to(cross_attention_mask.dtype)[..., None]
            )
            cross_attention_mask = cross_attention_mask * full_text_row_masked_out_mask
        with self._ephemeral_cache(state):
            output = self.moss.language_model(
                input_ids=None,
                inputs_embeds=query_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=state.past_key_values,
                cross_attention_mask=cross_attention_mask,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                cache_position=cache_position,
                use_cache=True,
            )
        return _layer_tensor(output)


def _time_embedding(times: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=times.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = times.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if width % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(dtype=times.dtype)


class ActionQueryDecoder(nn.Module):
    """Builds ephemeral action-query embeddings and reads their hidden states.

    Fusion with vision and language happens inside MOSS, so the readout is a
    plain MLP. Each query carries its own predicted execution delay, because the
    j-th action of a chunk executes one control period later than the (j-1)-th
    and therefore acts on visual evidence that is correspondingly staler.
    """

    def __init__(self, config: MossActionConfig):
        super().__init__()
        width = config.moss_hidden_size
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.chunk_size = config.chunk_size
        self.delay_scale = config.delay_scale
        self.condition_in = nn.Linear(config.condition_dim, width)
        self.query_embedding = nn.Parameter(torch.empty(config.chunk_size, width))
        self.delay_in = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.head = nn.Sequential(
            nn.RMSNorm(width),
            nn.Linear(width, config.action_hidden_size),
            nn.SiLU(),
            nn.Linear(config.action_hidden_size, config.action_dim),
        )
        nn.init.normal_(self.query_embedding, std=0.02)
        # A small but nonzero output layer: zero-init would stall the first step
        # by zeroing the gradient of everything upstream of the head.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def condition(
        self,
        robot_state: torch.Tensor,
        state_velocity: torch.Tensor,
        visual_age: torch.Tensor,
    ) -> torch.Tensor:
        batch = robot_state.shape[0]
        if robot_state.shape != (batch, self.state_dim):
            raise ValueError(f"robot_state must have shape [batch, {self.state_dim}]")
        if state_velocity.shape != robot_state.shape:
            raise ValueError("state_velocity must align with robot_state")
        if visual_age.ndim == 0:
            visual_age = visual_age.expand(batch)
        if visual_age.shape != (batch,):
            raise ValueError("visual_age must be scalar or shape [batch]")
        return torch.cat(
            (
                robot_state,
                state_velocity,
                visual_age.unsqueeze(-1).to(robot_state) * self.delay_scale,
            ),
            dim=-1,
        )

    def query_embeddings(
        self,
        condition: torch.Tensor,
        query_delays: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[batch, chunk_size, moss_hidden]` ephemeral query inputs."""
        batch = condition.shape[0]
        if query_delays.shape != (batch, self.chunk_size):
            raise ValueError(
                f"query_delays must have shape [batch, {self.chunk_size}]"
            )
        hidden = self.query_embedding[None].expand(batch, -1, -1)
        hidden = hidden + self.condition_in(condition).unsqueeze(1)
        delays = (query_delays.to(condition) * self.delay_scale).reshape(-1)
        delay_embedding = self.delay_in(
            _time_embedding(delays, hidden.shape[-1])
        ).view(batch, self.chunk_size, -1)
        return hidden + delay_embedding

    def forward(self, action_hidden: torch.Tensor) -> torch.Tensor:
        if action_hidden.ndim != 3 or action_hidden.shape[1] != self.chunk_size:
            raise ValueError(
                f"action hidden states must have shape [batch, {self.chunk_size}, hidden]"
            )
        return self.head(action_hidden)


class MossActionVLA(nn.Module):
    """Streaming visual context, ephemeral action queries, continuous chunk."""

    def __init__(self, backbone: TruncatedMossBackbone, config: MossActionConfig):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.decoder = ActionQueryDecoder(config)

    def default_query_delays(
        self,
        visual_age: torch.Tensor,
        *,
        inference_latency: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """d_j = visual_age + inference_latency + j * control_interval."""
        batch = visual_age.shape[0]
        offsets = torch.arange(
            self.config.chunk_size, device=visual_age.device, dtype=visual_age.dtype
        )
        latency = (
            inference_latency
            if isinstance(inference_latency, torch.Tensor)
            else torch.full_like(visual_age, float(inference_latency))
        )
        if latency.shape != (batch,):
            raise ValueError("inference_latency must be scalar or shape [batch]")
        base = visual_age + latency
        return base[:, None] + offsets[None] * self.config.control_interval

    def predict_chunk(
        self,
        moss_inputs: Mapping[str, Any],
        robot_state: torch.Tensor,
        state_velocity: torch.Tensor,
        visual_age: torch.Tensor,
        *,
        query_delays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full-recompute path: one planning time per sample.

        Action queries are appended to the prefix and run through all 24 layers,
        matching the exact compute graph of `decode_action_queries` at deploy.
        """
        batch = robot_state.shape[0]
        if visual_age.ndim == 0:
            visual_age = visual_age.expand(batch)
        if query_delays is None:
            query_delays = self.default_query_delays(visual_age)
        condition = self.decoder.condition(
            robot_state, state_velocity, visual_age
        )
        queries = self.decoder.query_embeddings(condition, query_delays)
        needs_backbone_grad = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        with nullcontext() if needs_backbone_grad else torch.no_grad():
            query_hidden = self.backbone.forward_with_queries(
                moss_inputs, queries
            )
        return self.decoder(query_hidden)

    def forward(
        self,
        moss_inputs: Mapping[str, Any],
        robot_state: torch.Tensor,
        state_velocity: torch.Tensor,
        actions: torch.Tensor,
        visual_age: torch.Tensor,
        *,
        query_delays: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        **_unused: Any,
    ) -> dict[str, torch.Tensor]:
        predicted = self.predict_chunk(
            moss_inputs,
            robot_state,
            state_velocity,
            visual_age,
            query_delays=query_delays,
        )
        target = actions.to(predicted)
        if target.shape != predicted.shape:
            raise ValueError(
                f"actions must have shape {tuple(predicted.shape)}, got {tuple(target.shape)}"
            )
        errors = (predicted - target).abs().mean(dim=-1)
        if valid_mask is None:
            loss = errors.mean()
        else:
            weights = valid_mask.to(errors)
            if weights.shape != errors.shape:
                raise ValueError("valid_mask must align with the action chunk")
            total = weights.sum()
            if not bool(total > 0):
                raise ValueError("every planning time needs at least one valid action")
            loss = (errors * weights).sum() / total
        return {"loss": loss, "predicted_actions": predicted}
        target = actions.to(predicted)
        if target.shape != predicted.shape:
            raise ValueError(
                f"actions must have shape {tuple(predicted.shape)}, got {tuple(target.shape)}"
            )
        errors = (predicted - target).abs().mean(dim=-1)
        if valid_mask is None:
            loss = errors.mean()
        else:
            weights = valid_mask.to(errors)
            if weights.shape != errors.shape:
                raise ValueError("valid_mask must align with the action chunk")
            total = weights.sum()
            if not bool(total > 0):
                raise ValueError("every planning time needs at least one valid action")
            loss = (errors * weights).sum() / total
        return {"loss": loss, "predicted_actions": predicted}

    def create_stream(
        self,
        processor: Any,
        instruction: str,
        *,
        system_prompt: str | None = None,
    ) -> "StreamingMossActionSession":
        return StreamingMossActionSession(
            self, processor, instruction, system_prompt=system_prompt
        )


class StreamingMossActionSession:
    """One append-only MOSS cache per episode; action queries never persist."""

    def __init__(
        self,
        policy: MossActionVLA,
        processor: Any,
        instruction: str,
        *,
        system_prompt: str | None = None,
    ):
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("stream instruction must not be empty")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": instruction})
        input_ids = processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        device = next(policy.parameters()).device
        input_ids = input_ids.to(device)
        self.policy = policy
        self.processor = processor
        self.state = policy.backbone.start_stream(input_ids)
        self.origin_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self.last_frame_timestamp: float | None = None
        self._lock = threading.Lock()
        self.plan_latency_ema: float = 0.05

    def _append_frame(
        self,
        image: Any,
        *,
        timestamp: float | None = None,
    ) -> BackboneTaps:
        with self._lock:
            timestamp = time.monotonic() if timestamp is None else float(timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("stream timestamp must be finite")
            if self.last_timestamp is not None and timestamp < self.last_timestamp:
                raise ValueError("stream timestamps must be non-decreasing")
            if self.origin_timestamp is None:
                self.origin_timestamp = timestamp
            self.last_timestamp = timestamp
            segment = realtime_frame_segment(timestamp - self.origin_timestamp)
            frame_inputs = dict(
                self.processor(
                    text=segment,
                    images=[image],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            )
            device = next(self.policy.parameters()).device
            frame_inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in frame_inputs.items()
            }
            taps = self.policy.backbone.append_stream_frame(self.state, frame_inputs)
            self.last_frame_timestamp = timestamp
            return taps

    def append_frame(self, image: Any, *, timestamp: float | None = None) -> None:
        """Grow the visual KV cache without planning."""
        self._append_frame(image, timestamp=timestamp)

    # Kept for the async perception worker, which encodes frames on its own thread.
    def encode_frame(self, image: Any, *, timestamp: float | None = None) -> float:
        self._append_frame(image, timestamp=timestamp)
        return float(self.last_frame_timestamp)

    @torch.no_grad()
    def plan(
        self,
        robot_state: torch.Tensor,
        state_velocity: torch.Tensor,
        *,
        plan_timestamp: float | None = None,
        inference_latency: float | None = None,
        clamp: bool = True,
    ) -> ActionChunk:
        """Decode one micro-chunk from the committed prefix, then drop the queries."""
        with self._lock:
            if self.last_frame_timestamp is None:
                raise RuntimeError("a frame must be appended before planning")
            call_start = time.monotonic()
            plan_timestamp = (
                call_start if plan_timestamp is None else float(plan_timestamp)
            )
            if not math.isfinite(plan_timestamp):
                raise ValueError("plan timestamp must be finite")
            visual_age = max(0.0, plan_timestamp - self.last_frame_timestamp)
            # Use tracked EMA latency if no explicit latency override was provided
            latency = self.plan_latency_ema if inference_latency is None else float(inference_latency)
            device = next(self.policy.parameters()).device
            dtype = self.policy.decoder.query_embedding.dtype
            robot_state = robot_state.to(device=device, dtype=dtype)
            state_velocity = state_velocity.to(device=device, dtype=dtype)
            if robot_state.ndim == 1:
                robot_state = robot_state.unsqueeze(0)
            if state_velocity.ndim == 1:
                state_velocity = state_velocity.unsqueeze(0)
            if robot_state.shape[0] != 1:
                raise ValueError("a streaming session plans for one robot at a time")
            age = torch.full((1,), visual_age, device=device, dtype=dtype)
            delays = self.policy.default_query_delays(
                age, inference_latency=latency
            )
            condition = self.policy.decoder.condition(
                robot_state, state_velocity, age
            )
            queries = self.policy.decoder.query_embeddings(condition, delays)
            hidden = self.policy.backbone.decode_action_queries(self.state, queries)
            actions = self.policy.decoder(hidden)
            if clamp:
                actions = actions.clamp(-1.0, 1.0)
            actual_latency = time.monotonic() - call_start
            # Smooth EMA update
            self.plan_latency_ema = 0.9 * self.plan_latency_ema + 0.1 * actual_latency
            return ActionChunk(
                actions=actions[0],
                start_time=plan_timestamp + latency,
                plan_time=plan_timestamp,
                interval=self.policy.config.control_interval,
                visual_age=visual_age,
            )

    @torch.no_grad()
    def predict_chunk(
        self,
        image: Any,
        robot_state: torch.Tensor,
        state_velocity: torch.Tensor,
        *,
        timestamp: float | None = None,
        plan_timestamp: float | None = None,
        inference_latency: float | None = None,
        clamp: bool = True,
    ) -> ActionChunk:
        self._append_frame(image, timestamp=timestamp)
        return self.plan(
            robot_state,
            state_velocity,
            plan_timestamp=plan_timestamp
            if plan_timestamp is not None
            else self.last_frame_timestamp,
            inference_latency=inference_latency,
            clamp=clamp,
        )


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
    }


def load_trainable_state_dict(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
    }
    expected, received = set(parameters), set(state)
    if expected != received:
        raise ValueError(
            f"trainable state mismatch: missing={sorted(expected - received)[:8]}, "
            f"unexpected={sorted(received - expected)[:8]}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(
                    f"shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


_DELETED_LAYER_PATTERN = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.")


def audit_loading_info(loading_info: Mapping[str, Any]) -> CheckpointAudit:
    missing = tuple(sorted(loading_info.get("missing_keys") or ()))
    mismatched_raw = loading_info.get("mismatched_keys") or ()
    mismatched = tuple(sorted(str(item) for item in mismatched_raw))
    errors = tuple(loading_info.get("error_msgs") or ())
    unexpected = tuple(sorted(loading_info.get("unexpected_keys") or ()))
    deleted_indices: set[int] = set()
    invalid_unexpected = []
    for key in unexpected:
        match = _DELETED_LAYER_PATTERN.search(key)
        if match is None or int(match.group(1)) < RETAINED_LAYERS:
            invalid_unexpected.append(key)
        else:
            deleted_indices.add(int(match.group(1)))
    expected_deleted = set(range(RETAINED_LAYERS, 48))
    if (
        missing
        or mismatched
        or errors
        or invalid_unexpected
        or deleted_indices != expected_deleted
    ):
        raise RuntimeError(
            "strict MOSS-VL checkpoint audit failed: "
            f"missing={list(missing)[:8]}, mismatched={list(mismatched)[:8]}, "
            f"invalid_unexpected={invalid_unexpected[:8]}, errors={list(errors)[:4]}, "
            f"deleted_layers={sorted(deleted_indices)}"
        )
    return CheckpointAudit(
        tuple(sorted(deleted_indices)),
        ("lm_head.weight",),
        unexpected,
        missing,
        mismatched,
    )


def _truncated_config(checkpoint: Path) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True
    )
    if int(getattr(config, "vision_seq_pad_multiple", -1)) != 1:
        raise ValueError(
            "MOSS-Action streaming requires the MOSS-VL-Realtime checkpoint; "
            "MOSS-VL-Instruct is an offline checkpoint"
        )
    config = copy.deepcopy(config)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise ValueError("MOSS-VL config is missing text_config")
    original_layers = int(getattr(text_config, "num_hidden_layers", -1))
    original_cross = tuple(
        int(index) for index in getattr(text_config, "cross_attention_layers", ())
    )
    expected_cross = tuple(range(2, 48, 4))
    if original_layers != 48 or original_cross != expected_cross:
        raise ValueError(
            "checkpoint is not the expected 48-layer MOSS-VL architecture: "
            f"layers={original_layers}, cross_attention_layers={original_cross}"
        )
    text_config.num_hidden_layers = RETAINED_LAYERS
    text_config.cross_attention_layers = list(RETAINED_CROSS_ATTENTION_LAYERS)
    return config


def load_truncated_moss(
    checkpoint: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Any = None,
    attention_backend: str = "flash_attention_2",
) -> tuple[TruncatedMossBackbone, CheckpointAudit]:
    """Load only the retained local checkpoint tensors; network access is disabled."""
    from transformers import AutoModelForCausalLM

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"local MOSS-VL checkpoint directory not found: {checkpoint}"
        )
    config = _truncated_config(checkpoint)
    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
        "attn_implementation": attention_backend,
        "low_cpu_mem_usage": True,
        "output_loading_info": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    loaded, loading_info = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), **kwargs
    )
    audit = audit_loading_info(loading_info)
    backbone = TruncatedMossBackbone(loaded)
    return backbone, audit


def checkpoint_fingerprint(checkpoint: str | Path) -> dict[str, Any]:
    root = Path(checkpoint).expanduser().resolve()
    files = sorted(root.glob("*.safetensors")) or sorted(
        root.glob("pytorch_model*.bin")
    )
    if not files:
        raise FileNotFoundError(f"no checkpoint weight files found in {root}")
    combined = hashlib.sha256()
    per_file = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        value = digest.hexdigest()
        per_file[path.name] = value
        combined.update(
            path.name.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n"
        )
    return {"combined_sha256": combined.hexdigest(), "files": per_file}


def parameter_count(module: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    }
