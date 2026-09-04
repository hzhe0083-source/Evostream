"""Truncated MOSS-VL backbone and the MOSS-Action policy."""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

RETAINED_LAYERS = 24
RETAINED_CROSS_ATTENTION_LAYERS = (2, 6, 10, 14, 18, 22)
TAP_LAYERS = (14, 18, 23)


@dataclass(frozen=True)
class MossActionConfig:
    moss_hidden_size: int = 4096
    state_dim: int = 9
    action_dim: int = 7
    horizon: int = 50
    action_hidden_size: int = 1024
    flow_layers: int = 6
    flow_heads: int = 16
    dropout: float = 0.0
    initial_action_noise: float = 0.1
    stabilization: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            self.moss_hidden_size,
            self.state_dim,
            self.action_dim,
            self.horizon,
            self.action_hidden_size,
            self.flow_layers,
            self.flow_heads,
        )
        if any(value < 1 for value in positive):
            raise ValueError(
                "all dimensions and layer counts must be positive"
            )
        if self.action_hidden_size % self.flow_heads:
            raise ValueError("action_hidden_size must be divisible by flow_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.horizon < 2:
            raise ValueError("streaming action flow needs a horizon of at least two")
        if self.initial_action_noise <= 0 or self.stabilization <= 0:
            raise ValueError("streaming flow noise and stabilization must be positive")


@dataclass(frozen=True)
class BackboneTaps:
    hidden: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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


@dataclass
class StreamingActionState:
    """Persistent action-space flow state advanced once per control tick."""

    action: torch.Tensor
    step: int = 0


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
    """Build MOSS XRoPE positions for newly appended single-frame segments."""
    positions = torch.zeros(
        3, 1, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    )
    vision_chunks = []
    current = start
    frame = 0
    for token_index, token in enumerate(input_ids[0]):
        if int(token) != image_token_id:
            positions[:, 0, token_index] = current
            current += 1
            continue
        if frame >= len(grid_thw):
            raise ValueError("stream segment has more image tokens than frames")
        _, grid_h, grid_w = map(int, grid_thw[frame].tolist())
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
        vision_chunks.append(
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
        positions[:, 0, token_index] = separator
        current = separator + 1
        frame += 1
    if frame != len(grid_thw):
        raise ValueError("stream segment has fewer image tokens than frames")
    return positions, torch.cat(vision_chunks, dim=1).unsqueeze(1), current


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


class StreamingFlowActionExpert(nn.Module):
    """Action-space flow whose integration time is robot execution time."""

    def __init__(self, config: MossActionConfig):
        super().__init__()
        width = config.action_hidden_size
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.initial_action_noise = config.initial_action_noise
        self.stabilization = config.stabilization
        self.memory_norms = nn.ModuleList(
            nn.RMSNorm(config.moss_hidden_size) for _ in TAP_LAYERS
        )
        self.memory_projections = nn.ModuleList(
            nn.Linear(config.moss_hidden_size, width, bias=False) for _ in TAP_LAYERS
        )
        self.action_in = nn.Linear(config.action_dim, width)
        self.state_in = nn.Linear(config.state_dim, width)
        self.action_query = nn.Parameter(torch.empty(width))
        self.time_in = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=config.flow_heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer, config.flow_layers, nn.RMSNorm(width)
        )
        self.action_out = nn.Linear(width, config.action_dim)
        nn.init.normal_(self.action_query, std=0.02)

    def encode_memory(
        self,
        taps: Sequence[torch.Tensor],
        text_mask: torch.Tensor,
        readout_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(taps) != len(self.memory_projections):
            raise ValueError(
                f"expected {len(self.memory_projections)} hidden taps, got {len(taps)}"
            )
        if text_mask.ndim != 2 or not bool(text_mask.any(dim=1).all()):
            raise ValueError("text_mask must contain one valid token per sample")
        selected = None
        if readout_mask is not None:
            if readout_mask.shape != text_mask.shape:
                raise ValueError("readout_mask must align with MOSS text tokens")
            selected = readout_mask.to(device=text_mask.device, dtype=torch.bool)
            selected &= text_mask.bool()
            if not bool(selected.any(dim=1).all()):
                raise ValueError("readout_mask must select a token from every sample")
        else:
            positions = torch.arange(text_mask.shape[1], device=text_mask.device)
            last = (
                positions.expand_as(text_mask)
                .masked_fill(~text_mask.bool(), -1)
                .max(1)
                .values
            )
            batch = torch.arange(text_mask.shape[0], device=text_mask.device)
        memories = []
        for hidden, norm, projection in zip(
            taps, self.memory_norms, self.memory_projections
        ):
            if hidden.shape[:2] != text_mask.shape:
                raise ValueError("every MOSS tap must align with text_mask")
            readout = hidden[selected] if selected is not None else hidden[batch, last]
            memories.append(projection(norm(readout)))
        return torch.stack(memories, dim=1)

    def forward(
        self,
        action: torch.Tensor,
        times: torch.Tensor,
        memory: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"action must have shape [batch, {self.action_dim}]")
        if memory.ndim != 3 or memory.shape[0] != action.shape[0]:
            raise ValueError("memory must have shape [batch, memory_tokens, hidden]")
        if robot_state.shape != (action.shape[0], self.state_dim):
            raise ValueError(
                f"robot_state must have shape [batch, {self.state_dim}]"
            )
        if times.ndim == 0:
            times = times.expand(action.shape[0])
        if times.shape != (action.shape[0],):
            raise ValueError("times must be scalar or shape [batch]")
        hidden = self.action_in(action).unsqueeze(1) + self.action_query[None, None]
        hidden = hidden + self.state_in(robot_state.to(hidden)).unsqueeze(1)
        hidden = hidden + self.time_in(_time_embedding(times, hidden.shape[-1]))[:, None]
        return self.action_out(self.transformer(hidden, memory))[:, 0]

    def loss(
        self,
        actions: torch.Tensor,
        memory: torch.Tensor,
        robot_state: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        flow_centers: torch.Tensor,
        times: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"actions must have shape [batch, {self.horizon}, {self.action_dim}]"
            )
        batch = actions.shape[0]
        noise = (
            torch.randn(batch, self.action_dim, device=actions.device, dtype=actions.dtype)
            if noise is None
            else noise.to(actions)
        )
        if noise.shape != (batch, self.action_dim):
            raise ValueError("streaming Flow noise must have shape [batch, action_dim]")
        flow_centers = flow_centers.to(actions)
        if flow_centers.shape != (batch, self.action_dim):
            raise ValueError("flow_centers must have shape [batch, action_dim]")
        times = times.to(device=actions.device, dtype=actions.dtype)
        if times.shape != (batch,) or not bool(((0 <= times) & (times <= 1)).all()):
            raise ValueError("Flow times must have shape [batch] and lie in [0,1]")
        if valid_mask is None:
            lengths = torch.full(
                (batch,), self.horizon, device=actions.device, dtype=torch.long
            )
        else:
            if valid_mask.shape != actions.shape[:2]:
                raise ValueError("valid_mask must align with the demonstration trajectory")
            valid_mask = valid_mask.to(device=actions.device, dtype=torch.bool)
            if not bool(valid_mask[:, 0].all()):
                raise ValueError("every stream frame needs its next action")
            lengths = valid_mask.sum(dim=1)
            expected = torch.arange(self.horizon, device=actions.device)[None] < lengths[:, None]
            if not torch.equal(valid_mask, expected):
                raise ValueError("valid actions must form a contiguous trajectory prefix")

        # Online term matches the exact persistent state used at deployment.
        online_trajectory = flow_centers
        online_derivative = (actions[:, 0] - online_trajectory) * (self.horizon - 1)
        online_sigma = self.initial_action_noise * torch.exp(
            -self.stabilization * times
        )
        online_actions = online_trajectory + online_sigma[:, None] * noise
        online_target = online_derivative - self.stabilization * (
            online_actions - online_trajectory
        )

        # A second uniform phase supplies the original trajectory-level SFP objective.
        plan_times = torch.rand_like(times)
        span = (lengths - 1).clamp_min(0)
        position = plan_times * span.to(plan_times.dtype)
        left = position.floor().long()
        right = torch.minimum(left + 1, lengths - 1)
        alpha = (position - left).unsqueeze(1)
        rows = torch.arange(batch, device=actions.device)
        plan_trajectory = actions[rows, left].lerp(actions[rows, right], alpha)
        plan_derivative = (actions[rows, right] - actions[rows, left]) * span[:, None]
        plan_sigma = self.initial_action_noise * torch.exp(
            -self.stabilization * plan_times
        )
        plan_actions = plan_trajectory + plan_sigma[:, None] * noise
        plan_target = plan_derivative - self.stabilization * (
            plan_actions - plan_trajectory
        )

        predicted = self(
            torch.cat((online_actions, plan_actions)),
            torch.cat((times, plan_times)),
            torch.cat((memory, memory)),
            torch.cat((robot_state, robot_state)),
        )
        predicted_velocity, predicted_plan_velocity = predicted.chunk(2)
        loss = (
            predicted - torch.cat((online_target, plan_target))
        ).square().mean()
        return loss, {
            "predicted_velocity": predicted_velocity,
            "target_velocity": online_target,
            "sampled_actions": online_actions,
            "trajectory_actions": online_trajectory,
            "times": times,
            "planning_velocity": predicted_plan_velocity,
            "planning_target_velocity": plan_target,
        }


class MossActionVLA(nn.Module):
    def __init__(self, backbone: TruncatedMossBackbone, config: MossActionConfig):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.flow = StreamingFlowActionExpert(config)

    def encode_memory(
        self,
        moss_inputs: Mapping[str, Any],
        action_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        needs_backbone_grad = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        with nullcontext() if needs_backbone_grad else torch.no_grad():
            taps = self.backbone(**dict(moss_inputs))
        return self.flow.encode_memory(
            taps.hidden, taps.text_mask, readout_mask=action_token_mask
        )

    def forward(
        self,
        moss_inputs: Mapping[str, Any],
        robot_state: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        flow_centers: torch.Tensor,
        flow_times: torch.Tensor,
        action_token_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_memory(moss_inputs, action_token_mask)
        loss, details = self.flow.loss(
            actions,
            memory,
            robot_state,
            valid_mask,
            flow_centers=flow_centers,
            times=flow_times,
            noise=noise,
        )
        return {"loss": loss, "memory": memory, **details}

    @torch.no_grad()
    def stream_action(
        self,
        memory: torch.Tensor,
        robot_state: torch.Tensor,
        *,
        state: StreamingActionState | None = None,
        reference_action: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        clamp: bool = True,
    ) -> tuple[torch.Tensor, StreamingActionState]:
        batch = memory.shape[0]
        expected = (batch, self.config.action_dim)
        if state is None or state.step >= self.config.horizon - 1:
            if reference_action is None:
                reference_action = (
                    torch.zeros(expected, device=memory.device, dtype=memory.dtype)
                    if state is None
                    else state.action
                )
            reference_action = reference_action.to(memory)
            if reference_action.shape != expected:
                raise ValueError(f"reference_action must have shape {expected}")
            noise = torch.randn_like(reference_action) if noise is None else noise.to(memory)
            if noise.shape != expected:
                raise ValueError(f"stream action noise must have shape {expected}")
            state = StreamingActionState(
                reference_action + self.config.initial_action_noise * noise
            )
        elif state.action.shape != expected:
            raise ValueError(f"stream action state must have shape {expected}")

        time_value = state.step / (self.config.horizon - 1)
        times = torch.full(
            (batch,), time_value, device=memory.device, dtype=memory.dtype
        )
        action = state.action + self.flow(
            state.action, times, memory, robot_state
        ) / (self.config.horizon - 1)
        if clamp:
            action = action.clamp(-1.0, 1.0)
        return action, StreamingActionState(action, state.step + 1)

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
    """One incremental MOSS cache plus persistent action flow per robot episode."""

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

    def _append_frame(
        self,
        image: Any,
        *,
        timestamp: float | None = None,
    ) -> BackboneTaps:
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
        return self.policy.backbone.append_stream_frame(self.state, frame_inputs)

    def encode_frame(
        self,
        image: Any,
        *,
        timestamp: float | None = None,
    ) -> torch.Tensor:
        """Append one frame and project its latest MOSS taps into action memory."""
        taps = self._append_frame(image, timestamp=timestamp)
        return self.policy.flow.encode_memory(taps.hidden, taps.text_mask)

    @torch.no_grad()
    def predict_action(
        self,
        image: Any,
        robot_state: torch.Tensor,
        *,
        timestamp: float | None = None,
        action_state: StreamingActionState | None = None,
        reference_action: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        clamp: bool = True,
    ) -> tuple[torch.Tensor, StreamingActionState]:
        memory = self.encode_frame(image, timestamp=timestamp)
        return self.policy.stream_action(
            memory,
            robot_state,
            state=action_state,
            reference_action=reference_action,
            noise=noise,
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
