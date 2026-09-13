import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import (
    MossActionConfig,
    MossActionVLA,
    PlanContext,
    TruncatedMossBackbone,
)
from test_contract import (
    FakeMossCore,
    FakeProcessor,
)
from vision_split import EncodedVision, SplitVision


class TinyNativeVisualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cos, sin = position_embeddings if position_embeddings is not None else (1.0, 0.0)
        transformed = hidden_states * cos + sin
        return hidden_states + 0.1 * self.proj(transformed)


class TinyNativeVisual(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        depth: int = 10,
        deepstack_indexes: tuple[int, ...] = (2, 5, 8),
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.spatial_merge_size = spatial_merge_size
        self.deepstack_visual_indexes = deepstack_indexes
        self.patch_proj = nn.Linear(3, dim, bias=False)
        self.blocks = nn.ModuleList([TinyNativeVisualBlock(dim) for _ in range(depth)])
        self.merger_linear = nn.Linear(dim * (1 + len(deepstack_indexes)), dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_proj.weight.dtype

    def patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values shape: [N, 3]
        return self.patch_proj(pixel_values)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        seq_len = int(grid_thw[0, 1] * grid_thw[0, 2])
        return torch.zeros(seq_len, self.dim, dtype=self.dtype)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        seq_len = int(grid_thw[0, 1] * grid_thw[0, 2])
        return torch.zeros(seq_len, self.dim // 2, dtype=self.dtype)

    def merger(self, last_hidden: torch.Tensor, deepstack_features: list[torch.Tensor]) -> torch.Tensor:
        all_features = [last_hidden] + list(deepstack_features)
        concatenated = torch.cat(all_features, dim=-1)
        return self.merger_linear(concatenated)

    def forward_complete(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self.patch_embed(pixel_values.to(dtype=self.dtype))
        hidden = hidden + self.fast_pos_embed_interpolate(grid_thw)
        rotary = self.rot_pos_emb(grid_thw)
        seq_len = hidden.shape[0]
        hidden = hidden.reshape(seq_len, -1)
        rotary = rotary.reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        positions = (emb.cos(), emb.sin())
        cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu = F.pad(cu, (1, 0), value=0)
        deepstack = []
        for index in range(len(self.blocks)):
            hidden = self.blocks[index](hidden, cu_seqlens=cu, position_embeddings=positions)
            if index in self.deepstack_visual_indexes:
                deepstack.append(hidden)
        packed = self.merger(hidden, deepstack)
        return packed, deepstack


class TinyNativeMoss(nn.Module):
    def __init__(self, visual: TinyNativeVisual):
        super().__init__()
        self.visual = visual

    def convert_packed_to_batch(
        self,
        packed: torch.Tensor,
        grid_thw: torch.Tensor,
        media_nums_per_sample: Any = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        seq_len = packed.shape[0]
        states = packed.unsqueeze(0)
        info = [
            {
                "medias": [
                    {
                        "start": 0,
                        "end": seq_len,
                        "length": seq_len,
                        "num_frames": 1,
                        "grid_h": int(grid_thw[0, 1]),
                        "grid_w": int(grid_thw[0, 2]),
                        "vision_tokens_per_frame": seq_len,
                        "has_separator": True,
                    }
                ],
                "total_length": seq_len,
                "pad_start": seq_len,
                "pad_end": seq_len,
            }
        ]
        return states, info


class TestVisionSplitNative(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 16
        self.depth = 10
        self.deepstack_indexes = (2, 5, 8)
        self.visual = TinyNativeVisual(
            dim=self.dim,
            depth=self.depth,
            deepstack_indexes=self.deepstack_indexes,
            spatial_merge_size=2,
        ).eval()
        self.moss = TinyNativeMoss(self.visual)
        self.adapter = SplitVision(self.moss)
        self.grid_thw = torch.tensor([[1, 4, 4]])
        self.num_tokens = 16  # 4 * 4
        self.pixel_values = torch.randn(self.num_tokens, 3)

    def test_split_vision_init_validation(self):
        bad_visual = SimpleNamespace()
        with self.assertRaises(ValueError):
            SplitVision(SimpleNamespace(visual=bad_visual))

    def test_encode_prefix_eval_mode_check(self):
        self.visual.train()
        with self.assertRaises(ValueError):
            self.adapter.encode_prefix(
                self.pixel_values,
                self.grid_thw,
                stop_after=2,
                frame_id=0,
                observation_time=0.0,
            )
        self.visual.eval()

    def test_encode_prefix_argument_validation(self):
        with self.assertRaises(ValueError):
            self.adapter.encode_prefix(
                self.pixel_values,
                self.grid_thw,
                stop_after=0,
                frame_id=0,
                observation_time=0.0,
            )
        with self.assertRaises(ValueError):
            self.adapter.encode_prefix(
                self.pixel_values,
                self.grid_thw,
                stop_after=self.depth + 1,
                frame_id=0,
                observation_time=0.0,
            )
        with self.assertRaises(ValueError):
            self.adapter.encode_prefix(
                self.pixel_values,
                self.grid_thw,
                stop_after=2,
                frame_id=0,
                observation_time=float("inf"),
            )
        with self.assertRaises(ValueError):
            self.adapter.encode_prefix(
                self.pixel_values,
                torch.tensor([[2, 4, 4]]),
                stop_after=2,
                frame_id=0,
                observation_time=0.0,
            )

    def test_split_parity_multiple_stop_indices(self):
        packed_complete, _ = self.visual.forward_complete(self.pixel_values, self.grid_thw)
        expected_states, expected_info = self.moss.convert_packed_to_batch(
            packed_complete, self.grid_thw
        )

        for stop in (2, 4, 8):
            packet = self.adapter.encode_prefix(
                self.pixel_values,
                self.grid_thw,
                stop_after=stop,
                frame_id=1,
                observation_time=0.1,
            )
            self.assertEqual(packet.next_block, stop)
            expected_deepstack_count = sum(1 for idx in self.deepstack_indexes if idx < stop)
            self.assertEqual(len(packet.deepstack_features), expected_deepstack_count)

            encoded = self.adapter.encode_suffix(packet)
            torch.testing.assert_close(encoded.hidden_states, expected_states, rtol=1e-5, atol=1e-5)
            self.assertEqual(encoded.token_info, expected_info)
            self.assertTrue(torch.equal(encoded.grid_thw, self.grid_thw))

    def test_packet_owner_mismatch_rejection(self):
        packet = self.adapter.encode_prefix(
            self.pixel_values,
            self.grid_thw,
            stop_after=2,
            frame_id=0,
            observation_time=0.0,
        )
        other_adapter = SplitVision(self.moss)
        with self.assertRaises(ValueError):
            other_adapter.encode_suffix(packet)

    def test_packet_reuse_rejection(self):
        packet = self.adapter.encode_prefix(
            self.pixel_values,
            self.grid_thw,
            stop_after=2,
            frame_id=0,
            observation_time=0.0,
        )
        self.assertFalse(packet.completed)
        _ = self.adapter.encode_suffix(packet)
        self.assertTrue(packet.completed)
        with self.assertRaises(ValueError):
            self.adapter.encode_suffix(packet)

    def test_prefix_no_merger_and_suffix_no_repeated_prefix(self):
        original_merger = self.visual.merger
        merger_mock = MagicMock(side_effect=original_merger)
        self.visual.merger = merger_mock

        call_counts = [0] * len(self.visual.blocks)

        def make_hook(idx):
            def hook(module, inp, out):
                call_counts[idx] += 1
            return hook

        hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(self.visual.blocks)]

        stop = 4
        try:
            packet = self.adapter.encode_prefix(
                self.pixel_values,
                self.grid_thw,
                stop_after=stop,
                frame_id=0,
                observation_time=0.0,
            )
            # Prefix executed blocks 0..stop-1, zero calls to merger
            self.assertEqual(call_counts[:stop], [1] * stop)
            self.assertEqual(call_counts[stop:], [0] * (len(self.visual.blocks) - stop))
            self.assertEqual(merger_mock.call_count, 0)

            _ = self.adapter.encode_suffix(packet)
            # Suffix executed blocks stop..len-1, prefix blocks not repeated, merger called exactly once
            self.assertEqual(call_counts, [1] * len(self.visual.blocks))
            self.assertEqual(merger_mock.call_count, 1)
        finally:
            self.visual.merger = original_merger
            for h in hooks:
                h.remove()

    def test_spatial_features_ordering_and_shape(self):
        # Construct known structured sequence to verify unmerged 2x2 patch layout
        # H=4, W=4, merge_size=2 -> (H/2)*(W/2) = 4 macro-blocks, each 2x2.
        # Check that spatial_features correctly restores 2D spatial arrangement.
        h, w = 4, 4
        merge = self.visual.spatial_merge_size
        # Assign pixel (r, c) value r * 10 + c
        spatial_grid = torch.zeros(h, w)
        for r in range(h):
            for c in range(w):
                spatial_grid[r, c] = r * 10 + c

        # Native Qwen/MOSS merge block ordering:
        # reshape(h // merge, merge, w // merge, merge) -> permute(0, 2, 1, 3)
        reshaped = spatial_grid.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3)
        tokens_ordered = reshaped.reshape(h * w)

        # Create packet with hidden_states = tokens_ordered.unsqueeze(-1).expand(-1, dim)
        hidden = tokens_ordered.unsqueeze(-1).expand(-1, self.dim).clone()
        packet = self.adapter.encode_prefix(
            self.pixel_values,
            self.grid_thw,
            stop_after=2,
            frame_id=0,
            observation_time=0.0,
        )
        packet.hidden_states = hidden

        # side=4 pools 1:1 on 4x4 image
        pooled = self.adapter.spatial_features(packet, side=4)
        self.assertEqual(pooled.shape, (16, self.dim))

        # Check that pooled rows follow row-major order: r*10 + c
        reconstructed = pooled[:, 0].reshape(h, w)
        torch.testing.assert_close(reconstructed, spatial_grid)

        with self.assertRaises(ValueError):
            self.adapter.spatial_features(packet, side=0)

        other_adapter = SplitVision(self.moss)
        with self.assertRaises(ValueError):
            other_adapter.spatial_features(packet, side=4)


class TestPreencodedModelInterface(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.width = 16
        self.fake_moss = FakeMossCore(self.width).eval()
        self.backbone = TruncatedMossBackbone(self.fake_moss).eval()
        config = MossActionConfig(
            moss_hidden_size=self.width,
            state_dim=2,
            action_dim=3,
            chunk_size=4,
            action_hidden_size=self.width,
            control_interval=0.1,
        )
        self.policy = MossActionVLA(self.backbone, config).eval()
        self.processor = FakeProcessor()
        self.instruction = "pick up the cube"

    def test_append_encoded_vision_matches_append_stream_frame(self):
        image = np.full((2, 2, 3), 128, dtype=np.uint8)

        session_base = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs_base = dict(
            self.processor(
                text="<|frame|>",
                images=[image],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        taps_base = self.backbone.append_stream_frame(session_base.state, frame_inputs_base)

        session_pre = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs_pre = dict(
            self.processor(
                text="<|frame|>",
                images=[image],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        encoded = self.backbone.encode_complete_vision(frame_inputs_pre)
        self.assertIsInstance(encoded, EncodedVision)

        vcalls_before = self.fake_moss.vision_calls
        taps_pre = self.backbone.append_encoded_vision(session_pre.state, frame_inputs_pre, encoded)
        self.assertEqual(self.fake_moss.vision_calls, vcalls_before)

        for t1, t2 in zip(taps_base.hidden, taps_pre.hidden):
            torch.testing.assert_close(t1, t2)
        torch.testing.assert_close(taps_base.text_mask, taps_pre.text_mask)

        self.assertEqual(session_base.state.frame_count, session_pre.state.frame_count)
        self.assertEqual(session_base.state.vision_tokens, session_pre.state.vision_tokens)
        self.assertEqual(session_base.state.next_text_position, session_pre.state.next_text_position)
        torch.testing.assert_close(session_base.state.input_ids, session_pre.state.input_ids)
        torch.testing.assert_close(session_base.state.attention_mask, session_pre.state.attention_mask)

        for idx in range(len(session_base.state.past_key_values.layers)):
            self.assertEqual(
                session_base.state.past_key_values.get_seq_length(idx),
                session_pre.state.past_key_values.get_seq_length(idx),
            )

    def test_grid_thw_mismatch_raises(self):
        session = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        encoded = self.backbone.encode_complete_vision(frame_inputs)
        bad_encoded = EncodedVision(
            encoded.hidden_states,
            encoded.token_info,
            torch.tensor([[1, 8, 8]]),
            source_core=self.fake_moss,
        )
        with self.assertRaises(ValueError):
            self.backbone.append_encoded_vision(session.state, frame_inputs, bad_encoded)

    def test_no_kv_changes_during_prefix_encoding(self):
        session = self.policy.create_stream(self.processor, self.instruction)
        lengths_before = [l.get_seq_length() for l in session.state.past_key_values.layers]

        frame_inputs = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        _ = self.backbone.encode_complete_vision(frame_inputs)
        lengths_after = [l.get_seq_length() for l in session.state.past_key_values.layers]
        self.assertEqual(lengths_before, lengths_after)

    def test_original_state_dict_keys_unchanged(self):
        state_keys = set(self.backbone.state_dict().keys())
        moss_keys = {f"moss.{k}" for k in self.fake_moss.state_dict().keys()}
        self.assertEqual(state_keys, moss_keys)

    def test_append_preencoded_frame_chronological_and_duplicate_checks(self):
        session = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        encoded = self.backbone.encode_complete_vision(frame_inputs)

        # First frame: valid
        session.append_preencoded_frame(
            frame_inputs, encoded, frame_id=1, timestamp=1.0, origin_timestamp=1.0
        )
        self.assertEqual(session._last_committed_frame_id, 1)
        self.assertEqual(session.last_timestamp, 1.0)

        # Timestamp before origin
        with self.assertRaises(ValueError):
            session.append_preencoded_frame(
                frame_inputs, encoded, frame_id=2, timestamp=0.5, origin_timestamp=1.0
            )

        # Different episode origin
        with self.assertRaises(ValueError):
            session.append_preencoded_frame(
                frame_inputs, encoded, frame_id=2, timestamp=1.5, origin_timestamp=0.8
            )

        # Duplicate or backwards timestamp (must be strictly chronological)
        with self.assertRaises(ValueError):
            session.append_preencoded_frame(
                frame_inputs, encoded, frame_id=2, timestamp=1.0, origin_timestamp=1.0
            )
        with self.assertRaises(ValueError):
            session.append_preencoded_frame(
                frame_inputs, encoded, frame_id=2, timestamp=0.9, origin_timestamp=1.0
            )

        # Duplicate or decreasing frame_id
        with self.assertRaises(ValueError):
            session.append_preencoded_frame(
                frame_inputs, encoded, frame_id=1, timestamp=1.1, origin_timestamp=1.0
            )
        with self.assertRaises(ValueError):
            session.append_preencoded_frame(
                frame_inputs, encoded, frame_id=0, timestamp=1.1, origin_timestamp=1.0
            )

        # Valid second frame
        session.append_preencoded_frame(
            frame_inputs, encoded, frame_id=2, timestamp=1.2, origin_timestamp=1.0
        )
        self.assertEqual(session._last_committed_frame_id, 2)
        self.assertEqual(session.last_timestamp, 1.2)

    def test_plan_returns_detached_plan_context(self):
        session = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        encoded = self.backbone.encode_complete_vision(frame_inputs)
        session.append_preencoded_frame(
            frame_inputs, encoded, frame_id=0, timestamp=1.0, origin_timestamp=1.0
        )

        chunk = session.plan(
            robot_state=torch.zeros(1, 2),
            state_velocity=torch.zeros(1, 2),
            plan_timestamp=1.1,
        )
        self.assertIsNotNone(chunk.context)
        self.assertIsInstance(chunk.context, PlanContext)
        self.assertFalse(chunk.context.context_tensor.requires_grad)
        self.assertEqual(chunk.context.source_state_time, 1.1)
        self.assertEqual(chunk.context.source_frame_time, 1.0)
        self.assertEqual(chunk.context.context_tensor.shape, (self.width,))

    def test_same_model_two_adapters_accepted(self):
        class FullFakeMoss(FakeMossCore):
            def __init__(self, width: int):
                super().__init__(width)
                self.visual = TinyNativeVisual(
                    dim=width, depth=4, deepstack_indexes=(1,), spatial_merge_size=2
                ).eval()

            def convert_packed_to_batch(self, packed, grid_thw, media_nums_per_sample=None):
                tokens_per_frame = 5
                pad_end = 8
                states = torch.zeros(1, pad_end, self.width)
                info = [
                    {
                        "medias": [
                            {
                                "start": 0,
                                "end": tokens_per_frame,
                                "length": tokens_per_frame,
                                "num_frames": 1,
                                "grid_h": int(grid_thw[0, 1]),
                                "grid_w": int(grid_thw[0, 2]),
                                "vision_tokens_per_frame": 4,
                                "has_separator": True,
                            }
                        ],
                        "total_length": tokens_per_frame,
                        "pad_start": tokens_per_frame,
                        "pad_end": pad_end,
                    }
                ]
                return states, info

        full_moss = FullFakeMoss(self.width).eval()
        backbone = TruncatedMossBackbone(full_moss).eval()
        config = MossActionConfig(
            moss_hidden_size=self.width,
            state_dim=2,
            action_dim=3,
            chunk_size=4,
            action_hidden_size=self.width,
            control_interval=0.1,
        )
        policy = MossActionVLA(backbone, config).eval()

        # Two distinct SplitVision adapters binding the same moss core instance
        adapter_a = SplitVision(full_moss)
        adapter_b = SplitVision(full_moss)

        frame_inputs_a = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        grid_thw_a = frame_inputs_a["grid_thw"]
        pixel_values_a = torch.randn(int(grid_thw_a[0, 1] * grid_thw_a[0, 2]), 3)
        packet_a = adapter_a.encode_prefix(
            pixel_values_a, grid_thw_a, stop_after=2, frame_id=0, observation_time=0.0
        )
        encoded_a = adapter_a.encode_suffix(packet_a)

        frame_inputs_b = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        grid_thw_b = frame_inputs_b["grid_thw"]
        pixel_values_b = torch.randn(int(grid_thw_b[0, 1] * grid_thw_b[0, 2]), 3)
        packet_b = adapter_b.encode_prefix(
            pixel_values_b, grid_thw_b, stop_after=2, frame_id=1, observation_time=0.1
        )
        encoded_b = adapter_b.encode_suffix(packet_b)

        self.assertIs(encoded_a.source_core, full_moss)
        self.assertIs(encoded_b.source_core, full_moss)

        session = policy.create_stream(self.processor, self.instruction)
        taps_a = backbone.append_encoded_vision(session.state, frame_inputs_a, encoded_a)
        self.assertIsNotNone(taps_a)
        taps_b = backbone.append_encoded_vision(session.state, frame_inputs_b, encoded_b)
        self.assertIsNotNone(taps_b)
        self.assertEqual(session.state.frame_count, 2)

    def test_foreign_model_identical_grid_rejected_before_cache_mutation(self):
        # Foreign model with identical architecture, dimensions, and weights
        foreign_moss = FakeMossCore(self.width).eval()
        foreign_moss.load_state_dict(self.fake_moss.state_dict())
        foreign_backbone = TruncatedMossBackbone(foreign_moss).eval()

        session = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        # First append a valid frame so cache has non-zero length
        valid_encoded = self.backbone.encode_complete_vision(frame_inputs)
        self.backbone.append_encoded_vision(session.state, frame_inputs, valid_encoded)
        self.assertEqual(session.state.frame_count, 1)

        cache_lengths_before = [l.get_seq_length() for l in session.state.past_key_values.layers]
        frame_count_before = session.state.frame_count
        vision_tokens_before = session.state.vision_tokens
        input_ids_before = session.state.input_ids.clone()

        # Encode with foreign backbone (identical grid!)
        foreign_encoded = foreign_backbone.encode_complete_vision(frame_inputs)
        self.assertTrue(torch.equal(foreign_encoded.grid_thw, valid_encoded.grid_thw))

        # Must reject foreign core
        with self.assertRaises(ValueError) as ctx:
            self.backbone.append_encoded_vision(session.state, frame_inputs, foreign_encoded)
        self.assertIn("foreign or missing model core instance", str(ctx.exception))

        # Verify absolutely no cache mutation or state mutation occurred
        cache_lengths_after = [l.get_seq_length() for l in session.state.past_key_values.layers]
        self.assertEqual(cache_lengths_before, cache_lengths_after)
        self.assertEqual(session.state.frame_count, frame_count_before)
        self.assertEqual(session.state.vision_tokens, vision_tokens_before)
        torch.testing.assert_close(session.state.input_ids, input_ids_before)

    def test_malformed_missing_owner_rejected(self):
        session = self.policy.create_stream(self.processor, self.instruction)
        frame_inputs = dict(
            self.processor(
                text="<|frame|>",
                images=[np.zeros((2, 2, 3), dtype=np.uint8)],
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        encoded = self.backbone.encode_complete_vision(frame_inputs)

        # None source_core
        malformed = EncodedVision(
            encoded.hidden_states,
            encoded.token_info,
            encoded.grid_thw,
            source_core=None,
        )
        with self.assertRaises(ValueError):
            self.backbone.append_encoded_vision(session.state, frame_inputs, malformed)
