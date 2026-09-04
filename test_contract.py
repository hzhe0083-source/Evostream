"""Architecture checks plus optional real-checkpoint streaming parity."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from streaming import (
    AsyncPerception,
    ChunkExecutor,
    ChunkPlanner,
    LatestObservation,
)

try:
    import torch
    from torch import nn

    from model import (
        TAP_LAYERS,
        ActionChunk,
        MossActionConfig,
        MossActionVLA,
        TruncatedMossBackbone,
        _realtime_mrope_positions,
        _layer_tensor,
        _moss_core,
        audit_loading_info,
        load_truncated_moss,
        validate_layer_identity,
    )

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


class StreamingContractTests(unittest.TestCase):
    def test_mailbox_drops_intermediate_observations(self) -> None:
        mailbox = LatestObservation()
        mailbox.publish("old", np.zeros(2), timestamp=1.0)
        latest_version = mailbox.publish("new", np.ones(2), timestamp=2.0)
        observation = mailbox.wait_for_new(0, timeout=0.01)
        self.assertEqual(observation.version, latest_version)
        self.assertEqual(observation.image, "new")

    def test_perception_and_planning_have_separate_mailboxes(self) -> None:
        observations = LatestObservation()
        features = LatestObservation()
        perception = AsyncPerception(
            observations,
            features,
            lambda observation: f"encoded:{observation.image}",
        )
        perception.start()
        try:
            observations.publish("frame", np.zeros(2), timestamp=1.0)
            encoded = features.wait_for_new(0, timeout=1.0)
        finally:
            perception.stop()
        self.assertIsNotNone(encoded)
        self.assertEqual(encoded.image, "encoded:frame")
        self.assertEqual(perception.frames_encoded, 1)

    def test_executor_interpolates_a_chunk_across_control_ticks(self) -> None:
        executor = ChunkExecutor(action_dim=2)
        chunk = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        executor.submit(chunk, start_time=10.0, interval=0.1)
        np.testing.assert_allclose(executor.action_at(10.0), [0.0, 0.0])
        np.testing.assert_allclose(executor.action_at(10.15), [1.0, 1.0])
        np.testing.assert_allclose(executor.action_at(10.25), [2.0, 2.0])
        with self.assertRaises(RuntimeError):
            executor.action_at(10.45)

    def test_newest_chunk_wins_without_ensembling(self) -> None:
        executor = ChunkExecutor(action_dim=1)
        executor.submit(np.zeros((4, 1), dtype=np.float32), 0.0, 0.1)
        executor.submit(np.ones((4, 1), dtype=np.float32), 0.1, 0.1)
        # Both chunks cover t=0.15; the unensembled executor must not average.
        np.testing.assert_allclose(executor.action_at(0.15), [1.0])

    def test_temporal_ensemble_blends_overlapping_chunks(self) -> None:
        executor = ChunkExecutor(action_dim=1, ensemble_lambda=0.0)
        executor.submit(np.zeros((4, 1), dtype=np.float32), 0.0, 0.1)
        executor.submit(np.ones((4, 1), dtype=np.float32), 0.1, 0.1)
        # Zero decay weights both plans equally, so the vote is the mean.
        np.testing.assert_allclose(executor.action_at(0.15), [0.5])

    def test_planner_submits_chunks_to_the_executor(self) -> None:
        features = LatestObservation()
        executor = ChunkExecutor(action_dim=2)
        planner = ChunkPlanner(
            features,
            executor,
            lambda observation: {
                "actions": np.asarray(observation.image, dtype=np.float32),
                "start_time": 0.0,
                "interval": 1.0,
                "visual_age": 0.25,
            },
            period=0.01,
        )
        planner.start()
        try:
            features.publish([[2.0, 3.0], [4.0, 5.0]], np.zeros(2), timestamp=1.0)
            self.assertTrue(planner.wait_for_first_chunk(1.0))
        finally:
            planner.stop()
        planner.raise_if_failed()
        self.assertGreaterEqual(planner.chunks_planned, 1)
        self.assertAlmostEqual(planner.last_visual_age, 0.25)
        np.testing.assert_allclose(executor.action_at(0.0), [2.0, 3.0])
        np.testing.assert_allclose(executor.action_at(1.0), [4.0, 5.0])


if HAS_TORCH:
    try:
        import h5py

        from data import LiberoHDF5Dataset, MossActionCollator

        HAS_H5PY = True
    except ModuleNotFoundError:
        HAS_H5PY = False

    class FakeAttention(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.q_proj = nn.Linear(width, width)
            self.k_proj = nn.Linear(width, width)
            self.v_proj = nn.Linear(width, width)
            self.o_proj = nn.Linear(width, width)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.o_proj(
                (self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)).tanh()
            )

    class FakeSelfAttentionDecoderLayer(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.self_attn = FakeAttention(width)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden + self.self_attn(hidden)

    class MossVLCrossAttentionDecoderLayer(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.cross_attn = FakeAttention(width)
            self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden + self.cross_attn(hidden)

    class FakeCache:
        def __init__(self):
            self.layers = [FakeCacheLayer() for _ in range(24)]
            # Mirrors mllama: cross-attention layers retain vision K/V, so a
            # later token batch attends to earlier frames without resending them.
            self.vision_signal: Any = None

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self.layers[layer_idx].get_seq_length()

    class FakeCacheLayer:
        def __init__(self):
            self.length = 0

        def get_seq_length(self) -> int:
            return self.length

        def crop(self, length: int) -> None:
            self.length = min(self.length, length)

    class FakeLanguageModel(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            cross = {2, 6, 10, 14, 18, 22}
            self.layers = nn.ModuleList(
                MossVLCrossAttentionDecoderLayer(width)
                if index in cross
                else FakeSelfAttentionDecoderLayer(width)
                for index in range(24)
            )
            self.embed_tokens = nn.Embedding(128, width)
            self.norm = nn.RMSNorm(width)
            self.prefill_calls = 0

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            cross_attention_states: torch.Tensor | None = None,
            past_key_values: Any = None,
            use_cache: bool = False,
            cache_position: torch.Tensor | None = None,
            vision_cache_position: torch.Tensor | None = None,
            **_kwargs: Any,
        ) -> Any:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("fake language model requires token inputs")
                inputs_embeds = self.embed_tokens(input_ids)
            self.prefill_calls += int(past_key_values is None)
            hidden = inputs_embeds
            if cross_attention_states is not None:
                vision_signal = cross_attention_states.mean().to(hidden)
                if past_key_values is not None:
                    past_key_values.vision_signal = vision_signal
            elif getattr(past_key_values, "vision_signal", None) is not None:
                # No new frame: read the cached vision K/V, exactly as mllama
                # does when `cross_attention_states` is None mid-stream.
                vision_signal = past_key_values.vision_signal
            else:
                vision_signal = 0.0
            for layer in self.layers:
                if isinstance(layer, MossVLCrossAttentionDecoderLayer):
                    hidden = hidden + vision_signal
                hidden = layer(hidden)
            if use_cache:
                if past_key_values is None:
                    past_key_values = FakeCache()
                if cache_position is not None and len(cache_position):
                    text_length = int(cache_position[-1]) + 1
                    for index, layer in enumerate(past_key_values.layers):
                        if index not in {2, 6, 10, 14, 18, 22}:
                            layer.length = max(layer.length, text_length)
                if vision_cache_position is not None and len(vision_cache_position):
                    for index in {2, 6, 10, 14, 18, 22}:
                        past_key_values.layers[index].length += len(
                            vision_cache_position
                        )
            return SimpleNamespace(
                last_hidden_state=self.norm(hidden),
                past_key_values=past_key_values,
            )

    class FakeMossCore(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width
            self.language_model = FakeLanguageModel(width)
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=width),
                image_token_id=99,
                vision_seq_pad_multiple=8,
            )
            self.visual = SimpleNamespace(spatial_merge_size=2)
            self.vision_calls = 0

        @property
        def layers(self):
            return self.language_model.layers

        def get_input_embeddings(self):
            return self.language_model.embed_tokens

        def get_vision_features_chunked(
            self,
            pixel_values: torch.Tensor,
            _grid_thw: torch.Tensor,
            _media_nums_per_sample: Any,
        ):
            self.vision_calls += 1
            counts = _media_nums_per_sample if _media_nums_per_sample is not None else [len(_grid_thw) if _grid_thw is not None else 1]
            batch = len(counts)
            signal = (pixel_values.float().mean(dim=-1, keepdim=True) / 255.0).to(
                self.get_input_embeddings().weight
            )
            tokens_per_frame = 5
            max_frames = max(counts) if counts else 1
            total_tokens = max_frames * tokens_per_frame
            pad_end = int(math.ceil(total_tokens / 8.0) * 8)
            states = torch.zeros(batch, pad_end, self.width, dtype=signal.dtype)
            frame_offset = 0
            info = []
            for b, count in enumerate(counts):
                sample_tokens = count * tokens_per_frame
                if count > 0:
                    sample_sig = signal[frame_offset : frame_offset + count].mean()
                    states[b, :sample_tokens] = sample_sig
                frame_offset += count
                medias = [
                    {
                        "start": i * tokens_per_frame,
                        "end": (i + 1) * tokens_per_frame,
                        "length": tokens_per_frame,
                        "num_frames": 1,
                        "grid_h": 4,
                        "grid_w": 4,
                        "vision_tokens_per_frame": 4,
                        "has_separator": True,
                    }
                    for i in range(count)
                ]
                info.append({
                    "medias": medias,
                    "total_length": count * tokens_per_frame,
                    "pad_start": count * tokens_per_frame,
                    "pad_end": pad_end,
                })
            return states, info

        def _expand_cross_attention_mask(
            self,
            mask: torch.Tensor,
            info: list[dict[str, Any]],
            target_dtype: torch.dtype,
        ) -> torch.Tensor:
            batch = mask.shape[0]
            expanded_samples = []
            max_pad_end = max(item["pad_end"] for item in info)
            for b in range(batch):
                item_info = info[b if b < len(info) else 0]
                repeats = [
                    media["vision_tokens_per_frame"] + 1
                    for media in item_info["medias"]
                ]
                sample_mask = mask[b:b+1, :, :, :len(repeats)]
                sample_expanded = sample_mask.to(target_dtype).masked_fill(
                    sample_mask, torch.finfo(target_dtype).min
                )
                if repeats:
                    sample_expanded = sample_expanded.repeat_interleave(
                        torch.tensor(repeats, device=mask.device), dim=-1
                    )
                result = torch.full(
                    (*sample_expanded.shape[:-1], max_pad_end),
                    torch.finfo(target_dtype).min,
                    dtype=target_dtype,
                    device=mask.device,
                )
                result[..., : sample_expanded.shape[-1]] = sample_expanded
                expanded_samples.append(result)
            return torch.cat(expanded_samples, dim=0)

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            pixel_values: torch.Tensor | None = None,
            grid_thw: torch.Tensor | None = None,
            media_nums_per_sample: Any = None,
            **kwargs: Any,
        ) -> Any:
            vision_states = None
            if pixel_values is not None:
                vision_states, _ = self.get_vision_features_chunked(
                    pixel_values, grid_thw, media_nums_per_sample
                )
            return self.language_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cross_attention_states=vision_states,
                **kwargs,
            )

    class FakeTokenizer:
        def apply_chat_template(self, *_args: Any, **kwargs: Any):
            if kwargs.get("tokenize", True):
                return torch.tensor([[1, 2, 3]])
            return "<chat>"

        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<|vision_end|>": 11}.get(token, -1)

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def __call__(self, *, text: Any, images: Any, **_kwargs: Any):
            counts = (
                [value.count("<|vision_end|>") for value in text]
                if isinstance(text, list)
                else [1]
            )
            input_ids = torch.zeros(len(counts), 1 + 2 * max(counts), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, count in enumerate(counts):
                tokens = [10] + [token for _ in range(count) for token in (99, 11)]
                input_ids[row, : len(tokens)] = torch.tensor(tokens)
                attention_mask[row, : len(tokens)] = 1
            image_values = [float(np.asarray(image).mean()) for image in images]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.tensor(image_values).view(-1, 1),
                "grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
                "media_nums_per_sample": counts,
            }

    class FakeConditionalGeneration(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.model = FakeMossCore(width)
            self.lm_head = nn.Linear(width, 100)

        @property
        def language_model(self):
            return self.model.language_model

        def forward(self, **_kwargs: Any) -> Any:
            raise AssertionError("the language-model head wrapper must not run")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ModelContractTests(unittest.TestCase):
    def _policy(self) -> MossActionVLA:
        config = MossActionConfig(
            moss_hidden_size=16,
            state_dim=2,
            action_dim=3,
            chunk_size=4,
            action_hidden_size=16,
            control_interval=0.1,
        )
        return MossActionVLA(TruncatedMossBackbone(FakeMossCore(16)), config)

    def _inputs(self) -> dict[str, torch.Tensor]:
        return {
            "inputs_embeds": torch.randn(2, 7, 16),
            "attention_mask": torch.ones(2, 7, dtype=torch.bool),
        }

    def _readouts(self) -> torch.Tensor:
        readouts = torch.zeros(2, 7, dtype=torch.bool)
        readouts[0, 5] = True
        readouts[1, 4] = True
        return readouts

    def test_layer_identity_is_exact(self) -> None:
        validate_layer_identity(FakeMossCore(16))
        broken = FakeMossCore(16)
        broken.layers[22] = FakeSelfAttentionDecoderLayer(16)
        with self.assertRaises(ValueError):
            validate_layer_identity(broken)

    def test_causal_lm_head_is_removed_from_backbone(self) -> None:
        wrapped = FakeConditionalGeneration(16)
        self.assertIs(_moss_core(wrapped), wrapped.model)
        backbone = TruncatedMossBackbone(wrapped)
        self.assertFalse(
            any(name.startswith("lm_head") for name, _ in backbone.named_parameters())
        )
        taps = backbone(**self._inputs())
        self.assertEqual(taps.hidden[-1].shape, (2, 7, 16))

    def test_chunk_loss_shapes_and_cross_attention_gradient(self) -> None:
        policy = self._policy()
        self.assertTrue(all(parameter.requires_grad for parameter in policy.parameters()))
        output = policy(
            self._inputs(),
            torch.randn(2, 2),
            torch.randn(2, 2),
            torch.randn(2, 4, 3),
            torch.zeros(2),
            valid_mask=torch.ones(2, 4, dtype=torch.bool),
        )
        self.assertEqual(output["predicted_actions"].shape, (2, 4, 3))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        gradient = policy.decoder.condition_in.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        backbone_gradient = policy.backbone.layers[14].cross_attn.q_proj.weight.grad
        self.assertIsNotNone(backbone_gradient)
        self.assertGreater(float(backbone_gradient.abs().sum()), 0.0)

    def test_training_and_streaming_query_parity(self) -> None:
        """The core P0 invariant: training and streaming decode the exact same actions."""
        policy = self._policy().eval()
        image = np.full((2, 2, 3), 128, dtype=np.uint8)
        state = torch.tensor([[0.5, -0.5]])
        delta = torch.tensor([[0.1, -0.1]])

        # 1. Streaming path: prefill -> append frame -> plan with ephemeral queries
        processor = FakeProcessor()
        session = policy.create_stream(processor, "pick up the cube")
        session.append_frame(image, timestamp=0.0)
        streamed_chunk = session.plan(state, delta, plan_timestamp=0.0)

        # 2. Training path: prepare_streaming_moss_inputs -> predict_chunk (all-at-once)
        from data import prepare_streaming_moss_inputs
        moss_inputs = prepare_streaming_moss_inputs(
            processor, [[image]], ["pick up the cube"], [[0.0]]
        )
        trained_chunk = policy.predict_chunk(
            moss_inputs, state, delta, torch.tensor([0.0])
        )

        torch.testing.assert_close(
            streamed_chunk.actions, trained_chunk[0], atol=1e-4, rtol=1e-4
        )

    def test_valid_mask_excludes_padded_tail_actions(self) -> None:
        policy = self._policy().eval()
        inputs = self._inputs()
        actions = torch.zeros(2, 4, 3)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[:, :2] = True
        torch.manual_seed(0)
        masked = policy(
            inputs,
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            actions,
            torch.zeros(2),
            valid_mask=mask,
        )
        # Padding the tail with a wildly different value must not move the loss.
        actions[:, 2:] = 99.0
        torch.manual_seed(0)
        padded = policy(
            inputs,
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            actions,
            torch.zeros(2),
            valid_mask=mask,
        )
        torch.testing.assert_close(masked["loss"], padded["loss"])

    def test_query_delays_advance_by_one_control_interval(self) -> None:
        policy = self._policy()
        delays = policy.default_query_delays(
            torch.tensor([0.2, 0.0]), inference_latency=0.05
        )
        torch.testing.assert_close(
            delays,
            torch.tensor(
                [[0.25, 0.35, 0.45, 0.55], [0.05, 0.15, 0.25, 0.35]]
            ),
        )

    def test_visual_age_changes_the_predicted_chunk(self) -> None:
        torch.manual_seed(11)
        policy = self._policy().eval()
        inputs = self._inputs()
        fresh = policy.predict_chunk(
            inputs, torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2)
        )
        stale = policy.predict_chunk(
            inputs, torch.zeros(2, 2), torch.zeros(2, 2), torch.full((2,), 0.5)
        )
        self.assertFalse(torch.allclose(fresh, stale))

    def test_action_queries_leave_the_streaming_cache_untouched(self) -> None:
        """The core ephemeral invariant: a plan must never enter the prefix."""
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        session.append_frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0.0)
        cache = session.state.past_key_values
        before = [layer.get_seq_length() for layer in cache.layers]
        tokens_before = session.state.input_ids.shape[1]
        frames_before = session.state.frame_count
        vision_before = session.state.vision_tokens
        position_before = session.state.next_text_position

        chunk = session.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.05)

        self.assertEqual(chunk.actions.shape, (4, 3))
        self.assertEqual(
            [layer.get_seq_length() for layer in cache.layers], before
        )
        self.assertEqual(session.state.input_ids.shape[1], tokens_before)
        self.assertEqual(session.state.frame_count, frames_before)
        self.assertEqual(session.state.vision_tokens, vision_before)
        self.assertEqual(session.state.next_text_position, position_before)

    def test_repeated_planning_is_stationary(self) -> None:
        """Planning twice without a new frame must give the same chunk."""
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        session.append_frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0.0)
        first = session.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.05)
        second = session.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.05)
        torch.testing.assert_close(first.actions, second.actions)
        self.assertAlmostEqual(first.visual_age, 0.05, places=6)

    def test_stream_appends_only_new_frames_and_tracks_visual_age(self) -> None:
        torch.manual_seed(7)
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        state = torch.zeros(1, 2)
        first = session.predict_chunk(
            np.zeros((2, 2, 3), dtype=np.uint8), state, state, timestamp=10.0
        )
        second = session.predict_chunk(
            np.full((2, 2, 3), 255, dtype=np.uint8), state, state, timestamp=10.1
        )
        self.assertEqual(policy.backbone.moss.language_model.prefill_calls, 1)
        self.assertEqual(policy.backbone.moss.vision_calls, 2)
        self.assertEqual(session.state.frame_count, 2)
        self.assertEqual(session.state.vision_tokens, 10)
        self.assertEqual(session.state.past_key_values.get_seq_length(2), 13)
        self.assertIsInstance(first, ActionChunk)
        self.assertFalse(torch.allclose(first.actions, second.actions))

    def test_action_queries_read_the_cached_vision(self) -> None:
        """Queries carry no new frame, so they must attend to the cached K/V."""
        policy = self._policy().eval()
        chunks = []
        for value in (0, 255):
            session = policy.create_stream(FakeProcessor(), "pick up the cube")
            session.append_frame(
                np.full((2, 2, 3), value, dtype=np.uint8), timestamp=0.0
            )
            chunks.append(
                session.plan(
                    torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.0
                ).actions
            )
        self.assertFalse(torch.allclose(chunks[0], chunks[1]))

    def test_plan_rollback_leaves_zero_trace_on_later_plans(self) -> None:
        """P0 regression test: Path A (with plan) and Path B (without plan) must produce identical chunks."""
        policy = self._policy().eval()
        processor = FakeProcessor()

        # Path A: frame0 -> plan (creates ephemeral KV) -> rollback -> frame1 -> plan
        session_a = policy.create_stream(processor, "pick up the cube")
        session_a.append_frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0.0)
        _ = session_a.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.05)
        session_a.append_frame(np.full((2, 2, 3), 255, dtype=np.uint8), timestamp=0.1)
        chunk_a = session_a.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.1)

        # Path B: frame0 -> frame1 -> plan (clean, never planned at t=0.05)
        session_b = policy.create_stream(processor, "pick up the cube")
        session_b.append_frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0.0)
        session_b.append_frame(np.full((2, 2, 3), 255, dtype=np.uint8), timestamp=0.1)
        chunk_b = session_b.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.1)

        # Assert zero drift
        torch.testing.assert_close(chunk_a.actions, chunk_b.actions, atol=1e-6, rtol=1e-6)

    def test_concurrent_append_and_plan_is_thread_safe(self) -> None:
        """P0 regression test: session mutex prevents cache corruption under concurrency."""
        import threading
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        session.append_frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0.0)
        errors: list[BaseException] = []

        def appender():
            try:
                for i in range(1, 15):
                    session.append_frame(
                        np.full((2, 2, 3), i, dtype=np.uint8), timestamp=0.05 * i
                    )
                    time.sleep(0.002)
            except BaseException as e:
                errors.append(e)

        def planner():
            try:
                for i in range(25):
                    session.plan(
                        torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.03 * i
                    )
                    time.sleep(0.001)
            except BaseException as e:
                errors.append(e)

        t1 = threading.Thread(target=appender)
        t2 = threading.Thread(target=planner)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [])
        self.assertEqual(session.state.frame_count, 15)

    def test_planning_requires_a_frame_first(self) -> None:
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        with self.assertRaises(RuntimeError):
            session.plan(torch.zeros(1, 2), torch.zeros(1, 2), plan_timestamp=0.0)

    def test_realtime_mrope_reserves_the_visual_grid(self) -> None:
        input_ids = torch.tensor([[5, 99, 6]])
        text, vision, next_position = _realtime_mrope_positions(
            input_ids,
            torch.tensor([[1, 4, 6]]),
            10,
            99,
            2,
        )
        self.assertEqual(text[:, 0, 0].tolist(), [10, 10, 10])
        self.assertEqual(text[:, 0, 1].tolist(), [14, 14, 14])
        self.assertEqual(text[:, 0, 2].tolist(), [15, 15, 15])
        self.assertEqual(vision.shape, (3, 1, 7))
        self.assertEqual(next_position, 16)

    def test_checkpoint_audit_allows_only_deleted_layers(self) -> None:
        info = {
            "unexpected_keys": [
                f"model.language_model.layers.{index}.weight" for index in range(24, 48)
            ]
        }
        audit = audit_loading_info(info)
        self.assertEqual(audit.deleted_layer_indices, tuple(range(24, 48)))
        with self.assertRaises(RuntimeError):
            audit_loading_info({**info, "missing_keys": ["model.visual.weight"]})

@unittest.skipUnless(
    HAS_TORCH and globals().get("HAS_H5PY", False), "torch/h5py unavailable"
)
class DataContractTests(unittest.TestCase):
    def _write(self, path: Path, steps: int = 6) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            data.attrs["problem_info"] = json.dumps(
                {"language_instruction": "pick up the cube"}
            )
            demo = data.create_group("demo_0")
            actions = np.linspace(-1.0, 1.0, steps * 7, dtype=np.float32).reshape(
                steps, 7
            )
            demo.create_dataset("actions", data=actions)
            observations = demo.create_group("obs")
            frames = np.arange(steps * 2 * 3 * 3, dtype=np.uint8).reshape(
                steps, 2, 3, 3
            )
            observations.create_dataset("agentview_rgb", data=frames)
            rows = np.arange(steps, dtype=np.float32)[:, None]
            observations.create_dataset(
                "joint_states", data=rows + np.arange(7, dtype=np.float32)[None] / 10
            )
            observations.create_dataset(
                "gripper_states", data=rows + np.arange(2, dtype=np.float32)[None] / 10
            )
        return actions, frames

    def test_each_sample_is_one_planning_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.hdf5"
            actions, frames = self._write(path)
            dataset = LiberoHDF5Dataset(
                path,
                chunk_size=3,
                action_offset=1,
                context_frames=2,
                max_visual_age_steps=0,
            )
            # One item per planning time rather than one item per episode.
            self.assertEqual(len(dataset), 5)
            first = dataset[0]
            self.assertEqual(first["planning_time"], 0)
            self.assertEqual(first["actions"].shape, (3, 7))
            np.testing.assert_allclose(first["actions"], actions[1:4])
            np.testing.assert_array_equal(
                first["action_valid_mask"], [True, True, True]
            )
            self.assertEqual(first["robot_state"].shape, (9,))
            self.assertEqual(first["state_velocity"].shape, (9,))
            np.testing.assert_array_equal(
                np.asarray(first["images"][-1]), frames[0, ::-1]
            )
            # A later planning time sees more context and a moving state.
            third = dataset[3]
            self.assertEqual(third["planning_time"], 3)
            self.assertEqual(len(third["images"]), 2)
            np.testing.assert_array_equal(
                np.asarray(third["images"][-1]), frames[3, ::-1]
            )
            self.assertGreater(float(np.abs(third["state_velocity"]).max()), 0.0)

    def test_action_chunk_pads_and_masks_the_episode_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.hdf5"
            actions, _ = self._write(path)
            dataset = LiberoHDF5Dataset(
                path, chunk_size=3, action_offset=1, max_visual_age_steps=0
            )
            last = dataset[len(dataset) - 1]
            self.assertEqual(last["planning_time"], 4)
            np.testing.assert_allclose(last["actions"][0], actions[5])
            np.testing.assert_allclose(last["actions"][1], actions[5])
            np.testing.assert_array_equal(
                last["action_valid_mask"], [True, False, False]
            )

    def test_delay_aware_sampling_cuts_the_prefix_before_the_planning_time(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.hdf5"
            _, frames = self._write(path)
            fresh = LiberoHDF5Dataset(
                path, chunk_size=2, context_frames=2, max_visual_age_steps=0
            )
            self.assertEqual(float(fresh[4]["visual_age"]), 0.0)
            np.testing.assert_array_equal(
                np.asarray(fresh[4]["images"][-1]), frames[4, ::-1]
            )
            stale = LiberoHDF5Dataset(
                path, chunk_size=2, context_frames=2, max_visual_age_steps=2, seed=0
            )
            ages = {float(stale[index]["visual_age"]) for index in range(len(stale))}
            self.assertTrue(ages - {0.0}, "delay-aware training produced no staleness")
            for index in range(len(stale)):
                row = stale[index]
                steps_late = round(float(row["visual_age"]) / 0.1)
                newest = max(0, row["planning_time"] - steps_late)
                np.testing.assert_array_equal(
                    np.asarray(row["images"][-1]), frames[newest, ::-1]
                )
                # Delays include a non-zero planning latency L > 0
                self.assertGreater(float(row["query_delays"][0]), float(row["visual_age"]))

    def test_collator_batches_planning_times_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.hdf5"
            self._write(path)
            dataset = LiberoHDF5Dataset(
                path, chunk_size=3, context_frames=2, max_visual_age_steps=0
            )
            batch = MossActionCollator(FakeProcessor())([dataset[2], dataset[3]])
            self.assertEqual(batch["robot_state"].shape, (2, 9))
            self.assertEqual(batch["state_velocity"].shape, (2, 9))
            self.assertEqual(batch["actions"].shape, (2, 3, 7))
            self.assertEqual(batch["query_delays"].shape, (2, 3))
            self.assertEqual(batch["visual_age"].shape, (2,))


def _capture_raw_taps(
    model: nn.Module, inputs: dict[str, Any]
) -> dict[int, torch.Tensor]:
    core = model.model if hasattr(model, "model") else model
    layers = core.language_model.layers
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index in TAP_LAYERS:

        def capture(
            _module: nn.Module, _args: tuple[Any, ...], output: Any, *, layer=index
        ) -> None:
            captured[layer] = _layer_tensor(output).detach().float().cpu()

        handles.append(layers[index].register_forward_hook(capture))
    try:
        core(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(TAP_LAYERS):
        raise RuntimeError(f"failed to capture raw taps: {sorted(captured)}")
    return captured


def run_prefix_parity(argv: list[str]) -> None:
    if not HAS_TORCH:
        raise RuntimeError("prefix parity requires torch and project dependencies")
    parser = argparse.ArgumentParser(
        description="Compare full and truncated raw MOSS-VL taps"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    args = parser.parse_args(argv)

    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from data import prepare_moss_inputs

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir() or not args.image.is_file():
        raise FileNotFoundError(
            "checkpoint directory and parity image must exist locally"
        )
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if device.type != "cuda":
        dtype = torch.float32
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True
    )
    inputs = prepare_moss_inputs(
        processor,
        [Image.open(args.image).convert("RGB")],
        [args.instruction],
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    common = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
        "device_map": {"": str(device)},
        "attn_implementation": "flash_attention_2"
        if device.type == "cuda"
        else "eager",
        "low_cpu_mem_usage": True,
    }
    full = AutoModelForCausalLM.from_pretrained(str(checkpoint), **common).eval()
    with torch.inference_mode():
        expected = _capture_raw_taps(full, inputs)
    del full
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    truncated, _audit = load_truncated_moss(
        checkpoint,
        dtype=dtype,
        device_map={"": str(device)},
        attention_backend="flash_attention_2" if device.type == "cuda" else "eager",
    )
    truncated.eval()
    with torch.inference_mode():
        actual = _capture_raw_taps(truncated.moss, inputs)
    report = {}
    for index in TAP_LAYERS:
        torch.testing.assert_close(
            actual[index], expected[index], atol=args.atol, rtol=args.rtol
        )
        difference = (actual[index] - expected[index]).abs()
        report[f"H{index}"] = {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }
    print(json.dumps({"status": "PASS", "raw_prefix_parity": report}, indent=2))


def run_streaming_parity(argv: list[str]) -> None:
    """Compare one-frame full recompute with instruction-prefill + frame append."""
    if not HAS_TORCH:
        raise RuntimeError("streaming parity requires torch and project dependencies")
    parser = argparse.ArgumentParser(
        description="Compare full and incremental MOSS-Action control memories"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    args = parser.parse_args(argv)

    from PIL import Image
    from transformers import AutoProcessor

    from data import prepare_streaming_moss_inputs
    from model import realtime_frame_segment

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir() or not args.image.is_file():
        raise FileNotFoundError(
            "checkpoint directory and parity image must exist locally"
        )
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if device.type != "cuda":
        dtype = torch.float32
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True
    )
    backbone, _audit = load_truncated_moss(
        checkpoint,
        dtype=dtype,
        device_map={"": str(device)},
        attention_backend="flash_attention_2" if device.type == "cuda" else "eager",
    )
    if args.policy is not None:
        from model import checkpoint_fingerprint, load_trainable_state_dict

        payload = torch.load(args.policy, map_location="cpu", weights_only=True)
        if payload.get("format") != "moss_action_v6":
            raise ValueError("streaming parity policy must be moss_action_v6")
        if (
            payload["moss_checkpoint"]["combined_sha256"]
            != checkpoint_fingerprint(checkpoint)["combined_sha256"]
        ):
            raise ValueError("streaming parity policy uses different base weights")
        policy = MossActionVLA(backbone, MossActionConfig(**payload["config"])).to(device)
        load_trainable_state_dict(policy, payload["trainable_state"])
        del payload["trainable_state"]
        backbone = policy.backbone
    else:
        # Default policy for architecture/numerical parity verification
        hidden_size = int(backbone.moss.config.text_config.hidden_size)
        policy = MossActionVLA(
            backbone,
            MossActionConfig(moss_hidden_size=hidden_size, state_dim=9, action_dim=7, chunk_size=8),
        ).to(device)
    backbone.eval()
    policy.eval()
    image = Image.open(args.image).convert("RGB")

    prefix = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.instruction}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    frame_inputs = processor(
        text=realtime_frame_segment(0.0),
        images=[image],
        add_special_tokens=False,
        return_tensors="pt",
    )
    frame_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in frame_inputs.items()
    }
    second_frame_inputs = processor(
        text=realtime_frame_segment(0.1),
        images=[image],
        add_special_tokens=False,
        return_tensors="pt",
    )
    second_frame_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in second_frame_inputs.items()
    }
    first_full_inputs = prepare_streaming_moss_inputs(
        processor, [[image]], [args.instruction]
    )
    first_full_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in first_full_inputs.items()
    }
    prefix_text = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )
    second_full_inputs = processor(
        text=[
            prefix_text
            + realtime_frame_segment(0.0)
            + realtime_frame_segment(0.1)
        ],
        images=[image, image],
        padding=True,
        return_tensors="pt",
    )
    second_full_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in second_full_inputs.items()
    }

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        state = backbone.start_stream(prefix)
        first_incremental = backbone.append_stream_frame(state, frame_inputs)
        first_full = backbone(**first_full_inputs)
        second_incremental = backbone.append_stream_frame(state, second_frame_inputs)
        second_full = backbone(**second_full_inputs)

    report = {}
    for name, incremental, full in (
        ("first_frame", first_incremental, first_full),
        ("second_frame", second_incremental, second_full),
    ):
        report[name] = {}
        full_last = full.text_mask.long().sum(dim=1) - 1
        for offset, layer in enumerate(TAP_LAYERS):
            expected = full.hidden[offset][0, full_last[0]].float().cpu()
            actual = incremental.hidden[offset][0, 0].float().cpu()
            torch.testing.assert_close(
                actual, expected, atol=args.atol, rtol=args.rtol
            )
            difference = (actual - expected).abs()
            report[name][f"H{layer}"] = {
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
            }

    # Action Query and Rollback Parity verification
    dummy_state = torch.zeros(1, policy.config.state_dim, device=device, dtype=dtype)
    dummy_vel = torch.zeros(1, policy.config.state_dim, device=device, dtype=dtype)
    zero_age = torch.zeros(1, device=device, dtype=dtype)

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        # 1. Action chunk parity (training full forward vs streaming cache decode)
        full_chunk = policy.predict_chunk(
            first_full_inputs, dummy_state, dummy_vel, zero_age
        )[0].float().cpu()
        stream_chunk = policy.create_stream(processor, args.instruction)
        stream_chunk.append_frame(image, timestamp=0.0)
        cached_plan = stream_chunk.plan(dummy_state, dummy_vel, plan_timestamp=0.0).actions.float().cpu()
        torch.testing.assert_close(full_chunk, cached_plan, atol=args.atol, rtol=args.rtol)
        action_diff = (full_chunk - cached_plan).abs()

        # 2. Path A vs Path B rollback parity on actual Action Outputs
        # Path A: frame0 -> plan (rollback) -> frame1 -> plan
        sess_a = policy.create_stream(processor, args.instruction)
        sess_a.append_frame(image, timestamp=0.0)
        _ = sess_a.plan(dummy_state, dummy_vel, plan_timestamp=0.05)
        sess_a.append_frame(image, timestamp=0.1)
        plan_a = sess_a.plan(dummy_state, dummy_vel, plan_timestamp=0.1).actions.float().cpu()

        # Path B: frame0 -> (no plan) -> frame1 -> plan
        sess_b = policy.create_stream(processor, args.instruction)
        sess_b.append_frame(image, timestamp=0.0)
        sess_b.append_frame(image, timestamp=0.1)
        plan_b = sess_b.plan(dummy_state, dummy_vel, plan_timestamp=0.1).actions.float().cpu()

        torch.testing.assert_close(plan_a, plan_b, atol=args.atol, rtol=args.rtol)
        rollback_diff = (plan_a - plan_b).abs()

    action_parity_report = {
        "full_vs_cached_action_max_abs": float(action_diff.max()),
        "full_vs_cached_action_mean_abs": float(action_diff.mean()),
        "path_a_vs_path_b_rollback_max_abs": float(rollback_diff.max()),
        "path_a_vs_path_b_rollback_mean_abs": float(rollback_diff.mean()),
    }

    print(
        json.dumps(
            {
                "status": "PASS",
                "streaming_parity": report,
                "action_query_parity": action_parity_report,
                "cached_text_tokens": state.input_ids.shape[1],
                "cached_vision_tokens": state.vision_tokens,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "parity":
        run_prefix_parity(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "streaming-parity":
        run_streaming_parity(sys.argv[2:])
    else:
        unittest.main()
