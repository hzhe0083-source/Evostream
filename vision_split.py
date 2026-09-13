"""Inference-only split adapter for the native MOSS-VL-Realtime vision tower.

Matched upstream revision: 25e81cb952d5f353a5690f2c1ea09a725815df80.
No weights are registered here; the original tower owns every parameter.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class EncodedVision:
    """Self-contained representation of encoded visual features for one frame.

    Process-local immutable-weight inference contract:
    EncodedVision holds an exact object reference identity to the originating
    MOSS core instance (`source_core`) to guarantee that the vision encoder
    and the language model backbone share the exact same model weights without
    expensive per-frame weight hashing. In-place parameter mutation without
    instance replacement is unsupported; live packets and encoded outputs are
    valid only for the lifetime and frozen eval state of their originating core.
    """

    hidden_states: torch.Tensor
    token_info: list[dict[str, Any]]
    grid_thw: torch.Tensor
    source_core: Any = field(repr=False, compare=False)
    source_fingerprint: str | None = field(default=None, repr=False, compare=False)


@dataclass
class VisionPacket:
    hidden_states: torch.Tensor
    next_block: int
    grid_thw: torch.Tensor
    cu_seqlens: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    deepstack_features: list[torch.Tensor]
    media_nums_per_sample: Any
    frame_id: int
    observation_time: float
    model_fingerprint: str
    owner: Any = field(repr=False)
    completed: bool = False


class SplitVision:
    def __init__(self, moss: Any):
        self.moss = moss
        self.visual = moss.visual
        required = ("patch_embed", "fast_pos_embed_interpolate", "rot_pos_emb",
                    "blocks", "deepstack_visual_indexes", "merger")
        if any(not hasattr(self.visual, name) for name in required):
            raise ValueError("unsupported native MOSS visual implementation")
        source = inspect.getsource(type(self.visual))
        self.model_fingerprint = hashlib.sha256(source.encode()).hexdigest()
        self._owner = object()

    @torch.no_grad()
    def encode_prefix(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                      *, stop_after: int, frame_id: int, observation_time: float,
                      media_nums_per_sample: Any = None) -> VisionPacket:
        if self.visual.training:
            raise ValueError("split vision requires an eval-mode frozen policy")
        if not 0 < stop_after <= len(self.visual.blocks):
            raise ValueError("stop_after must be within the visual block range")
        if not math.isfinite(observation_time):
            raise ValueError("observation_time must be finite")
        if grid_thw.shape != (1, 3) or int(grid_thw[0, 0]) != 1:
            raise ValueError("split monitoring accepts one individually encoded frame")
        if bool((grid_thw <= 0).any()):
            raise ValueError("grid dimensions must be positive")
        grid = grid_thw.clone()
        visual = self.visual
        hidden = visual.patch_embed(pixel_values.to(dtype=visual.dtype))
        hidden = hidden + visual.fast_pos_embed_interpolate(grid)
        rotary = visual.rot_pos_emb(grid)
        seq_len = hidden.shape[0]
        hidden = hidden.reshape(seq_len, -1)
        rotary = rotary.reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        positions = (emb.cos(), emb.sin())
        cu = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        deepstack = []
        for index in range(stop_after):
            hidden = visual.blocks[index](hidden, cu_seqlens=cu,
                                          position_embeddings=positions)
            if index in visual.deepstack_visual_indexes:
                deepstack.append(hidden)
        return VisionPacket(hidden, stop_after, grid, cu, positions, deepstack,
                            media_nums_per_sample, frame_id, observation_time,
                            self.model_fingerprint, self._owner)

    @torch.no_grad()
    def encode_suffix(self, packet: VisionPacket) -> EncodedVision:
        if packet.owner is not self._owner or packet.model_fingerprint != self.model_fingerprint:
            raise ValueError("vision packet belongs to another model adapter")
        if packet.completed:
            raise ValueError("vision packet has already been consumed")
        if self.visual.training:
            raise ValueError("split vision requires eval mode")
        # Consume before execution: a failed suffix must not be retried partially.
        packet.completed = True
        hidden = packet.hidden_states
        deepstack = list(packet.deepstack_features)
        for index in range(packet.next_block, len(self.visual.blocks)):
            hidden = self.visual.blocks[index](hidden, cu_seqlens=packet.cu_seqlens,
                                              position_embeddings=packet.position_embeddings)
            if index in self.visual.deepstack_visual_indexes:
                deepstack.append(hidden)
        packed = self.visual.merger(hidden, deepstack)
        states, info = self.moss.convert_packed_to_batch(
            packed, packet.grid_thw, packet.media_nums_per_sample)
        return EncodedVision(
            hidden_states=states,
            token_info=info,
            grid_thw=packet.grid_thw,
            source_core=self.moss,
            source_fingerprint=self.model_fingerprint,
        )

    def spatial_features(self, packet: VisionPacket, side: int = 4) -> torch.Tensor:
        """Pool native merge-block ordering into a fixed spatial grid, not a global mean."""
        if side < 1 or packet.owner is not self._owner:
            raise ValueError("invalid spatial pooling request")
        _, h, w = (int(v) for v in packet.grid_thw[0])
        merge = int(self.visual.spatial_merge_size)
        hidden = packet.hidden_states.float()
        image = hidden.reshape(h // merge, w // merge, merge, merge, -1)
        image = image.permute(4, 0, 2, 1, 3).reshape(1, -1, h, w)
        return F.adaptive_avg_pool2d(image, (side, side))[0].flatten(1).T.contiguous()
