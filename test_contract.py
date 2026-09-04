"""Architecture checks plus optional real-checkpoint streaming parity."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from streaming import (
    AsyncPerception,
    LatestAction,
    LatestObservation,
    StreamingActionWorker,
)

try:
    import torch
    from torch import nn

    from model import (
        TAP_LAYERS,
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

    def test_perception_and_action_decoding_have_separate_mailboxes(self) -> None:
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

    def test_action_worker_publishes_each_flow_step_immediately(self) -> None:
        features = LatestObservation()
        actions = LatestAction(action_dim=2)
        worker = StreamingActionWorker(
            features,
            actions,
            lambda observation: np.asarray(observation.image, dtype=np.float32) + 1,
            period=0.01,
        )
        worker.start()
        try:
            features.publish([2, 3], np.zeros(2), timestamp=1.0)
            action = actions.wait_for_new(0, timeout=1.0)
            next_action = actions.wait_for_new(action.version, timeout=1.0)
        finally:
            worker.stop()
        self.assertIsNotNone(action)
        self.assertIsNotNone(next_action)
        np.testing.assert_array_equal(action.value, [3, 4])
        self.assertGreaterEqual(worker.actions_generated, 2)


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
            vision_signal = (
                0.0
                if cross_attention_states is None
                else cross_attention_states.mean().to(hidden)
            )
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
            batch = pixel_values.shape[0]
            signal = (pixel_values.float().mean(dim=1, keepdim=True) / 255.0).to(
                self.get_input_embeddings().weight
            )
            states = torch.zeros(batch, 8, self.width, dtype=signal.dtype)
            states[:, :5] = signal.unsqueeze(-1)
            info = [{
                "medias": [{
                    "start": 0,
                    "end": 5,
                    "length": 5,
                    "num_frames": 1,
                    "grid_h": 4,
                    "grid_w": 4,
                    "vision_tokens_per_frame": 4,
                    "has_separator": True,
                }],
                "total_length": 5,
                "pad_start": 5,
                "pad_end": 8,
            }]
            return states, info

        def _expand_cross_attention_mask(
            self,
            mask: torch.Tensor,
            info: list[dict[str, Any]],
            target_dtype: torch.dtype,
        ) -> torch.Tensor:
            repeats = [
                media["vision_tokens_per_frame"] + 1
                for media in info[0]["medias"]
            ]
            expanded = mask.to(target_dtype).masked_fill(
                mask, torch.finfo(target_dtype).min
            )
            expanded = expanded.repeat_interleave(
                torch.tensor(repeats, device=mask.device), dim=-1
            )
            result = torch.full(
                (*expanded.shape[:-1], info[0]["pad_end"]),
                torch.finfo(target_dtype).min,
                dtype=target_dtype,
                device=mask.device,
            )
            result[..., : expanded.shape[-1]] = expanded
            return result

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
            horizon=5,
            action_hidden_size=16,
            flow_layers=2,
            flow_heads=4,
            initial_action_noise=0.1,
            stabilization=2.0,
        )
        return MossActionVLA(TruncatedMossBackbone(FakeMossCore(16)), config)

    def _inputs(self) -> dict[str, torch.Tensor]:
        return {
            "inputs_embeds": torch.randn(2, 7, 16),
            "attention_mask": torch.ones(2, 7, dtype=torch.bool),
        }

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

    def test_action_path_has_shapes_loss_and_cross_attention_gradient(self) -> None:
        policy = self._policy()
        self.assertTrue(all(parameter.requires_grad for parameter in policy.parameters()))
        output = policy(
            self._inputs(),
            torch.randn(2, 2),
            torch.randn(2, 5, 3),
            torch.ones(2, 5, dtype=torch.bool),
            flow_centers=torch.zeros(2, 3),
            flow_times=torch.zeros(2),
        )
        self.assertEqual(output["memory"].shape, (2, 3, 16))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        gradient = policy.flow.memory_projections[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        backbone_gradient = policy.backbone.layers[14].cross_attn.q_proj.weight.grad
        self.assertIsNotNone(backbone_gradient)
        self.assertGreater(float(backbone_gradient.abs().sum()), 0.0)

    def test_every_frame_end_receives_an_action_target(self) -> None:
        policy = self._policy()
        readouts = torch.zeros(2, 7, dtype=torch.bool)
        readouts[0, [2, 5]] = True
        readouts[1, 4] = True
        output = policy(
            self._inputs(),
            torch.randn(3, 2),
            torch.randn(3, 5, 3),
            torch.ones(3, 5, dtype=torch.bool),
            flow_centers=torch.zeros(3, 3),
            flow_times=torch.zeros(3),
            action_token_mask=readouts,
        )
        self.assertEqual(output["memory"].shape, (3, 3, 16))
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_streaming_flow_emits_one_persistent_action_per_tick(self) -> None:
        torch.manual_seed(3)
        policy = self._policy().eval()
        state = torch.randn(2, 2)
        memory = policy.encode_memory(self._inputs())
        first, flow_state = policy.stream_action(
            memory,
            state,
            reference_action=torch.zeros(2, 3),
            noise=torch.zeros(2, 3),
        )
        second, flow_state = policy.stream_action(memory, state, state=flow_state)
        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(second.shape, (2, 3))
        self.assertEqual(flow_state.step, 2)

    def test_streaming_flow_target_stabilizes_around_demo_trajectory(self) -> None:
        policy = self._policy().eval()
        actions = torch.full((2, 5, 3), 0.25)
        _, details = policy.flow.loss(
            actions,
            policy.encode_memory(self._inputs()),
            torch.zeros(2, 2),
            torch.ones(2, 5, dtype=torch.bool),
            flow_centers=torch.full((2, 3), 0.25),
            noise=torch.ones(2, 3),
            times=torch.zeros(2),
        )
        torch.testing.assert_close(
            details["trajectory_actions"], torch.full((2, 3), 0.25)
        )
        torch.testing.assert_close(
            details["sampled_actions"], torch.full((2, 3), 0.35)
        )
        torch.testing.assert_close(
            details["target_velocity"], torch.full((2, 3), -0.2)
        )
        ramp = (
            torch.arange(1, 6, dtype=torch.float32)[None, :, None]
            .expand(2, -1, 3)
            / 10
        )
        _, ramp_details = policy.flow.loss(
            ramp,
            policy.encode_memory(self._inputs()),
            torch.zeros(2, 2),
            flow_centers=torch.zeros(2, 3),
            noise=torch.zeros(2, 3),
            times=torch.full((2,), 0.5),
        )
        torch.testing.assert_close(
            ramp_details["target_velocity"], torch.full((2, 3), 0.4)
        )

    def test_streaming_flow_cycle_restarts_from_last_action(self) -> None:
        policy = self._policy().eval()
        memory = policy.encode_memory(self._inputs())
        previous = torch.full((2, 3), 0.2)
        from model import StreamingActionState

        _, state = policy.stream_action(
            memory,
            torch.zeros(2, 2),
            state=StreamingActionState(previous, policy.config.horizon - 1),
            noise=torch.zeros(2, 3),
        )
        self.assertEqual(state.step, 1)

    def test_stream_appends_only_new_frames_and_updates_action_memory(self) -> None:
        torch.manual_seed(7)
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        state = torch.zeros(1, 2)
        noise = torch.randn(1, 5, 3)
        first, action_state = session.predict_action(
            np.zeros((2, 2, 3), dtype=np.uint8),
            state,
            timestamp=10.0,
            reference_action=torch.zeros(1, 3),
            noise=noise[:, 0],
        )
        second, _ = session.predict_action(
            np.full((2, 2, 3), 255, dtype=np.uint8),
            state,
            timestamp=10.1,
            action_state=action_state,
        )
        self.assertEqual(policy.backbone.moss.language_model.prefill_calls, 1)
        self.assertEqual(policy.backbone.moss.vision_calls, 2)
        self.assertEqual(session.state.frame_count, 2)
        self.assertEqual(session.state.vision_tokens, 10)
        self.assertEqual(session.state.past_key_values.get_seq_length(2), 13)
        self.assertFalse(torch.allclose(first, second))

    def test_incremental_stream_memory_trains_the_action_projection(self) -> None:
        policy = self._policy().eval()
        session = policy.create_stream(FakeProcessor(), "pick up the cube")
        memories = [
            session.encode_frame(
                np.full((2, 2, 3), value, dtype=np.uint8),
                timestamp=0.1 * index,
            )
            for index, value in enumerate((0, 255))
        ]
        torch.cat(memories).sum().backward()
        gradient = policy.flow.memory_projections[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(session.state.frame_count, 2)
        self.assertEqual(policy.backbone.moss.vision_calls, 2)

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
    def test_hdf5_alignment_upright_image_and_masked_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.hdf5"
            with h5py.File(path, "w") as handle:
                data = handle.create_group("data")
                data.attrs["problem_info"] = json.dumps(
                    {"language_instruction": "pick up the cube"}
                )
                demo = data.create_group("demo_0")
                actions = np.linspace(-1.0, 1.0, 28, dtype=np.float32).reshape(4, 7)
                demo.create_dataset("actions", data=actions)
                observations = demo.create_group("obs")
                frames = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
                observations.create_dataset("agentview_rgb", data=frames)
                rows = np.arange(4, dtype=np.float32)[:, None]
                observations.create_dataset(
                    "joint_states",
                    data=rows + np.arange(7, dtype=np.float32)[None] / 10,
                )
                observations.create_dataset(
                    "gripper_states",
                    data=rows + np.arange(2, dtype=np.float32)[None] / 10,
                )

            dataset = LiberoHDF5Dataset(path, horizon=3, action_offset=1)
            self.assertEqual(len(dataset), 1)
            first = dataset[0]
            self.assertEqual(len(first["images"]), 3)
            np.testing.assert_array_equal(
                np.asarray(first["images"][0]), frames[0, ::-1]
            )
            np.testing.assert_allclose(first["actions"][0], actions[1:4])
            np.testing.assert_allclose(first["flow_centers"][0], actions[0])
            self.assertEqual(float(first["flow_times"][0]), 0.0)
            self.assertEqual(float(first["flow_times"][1]), 0.5)
            np.testing.assert_array_equal(
                first["action_valid_mask"][0], [True, True, True]
            )
            np.testing.assert_allclose(
                first["actions"][2], np.repeat(actions[3:4], 3, axis=0)
            )
            np.testing.assert_array_equal(
                first["action_valid_mask"][2], [True, False, False]
            )
            self.assertEqual(first["instruction"], "pick up the cube")

            windowed = LiberoHDF5Dataset(
                path,
                horizon=2,
                action_offset=0,
                frame_stride=1,
                frame_interval=0.05,
            )
            self.assertEqual(len(windowed), 1)
            row = windowed[0]
            self.assertEqual(len(row["images"]), 4)
            for index, image in enumerate(row["images"]):
                np.testing.assert_array_equal(np.asarray(image), frames[index, ::-1])
            np.testing.assert_allclose(
                row["frame_timestamps"], [0.0, 0.05, 0.1, 0.15]
            )
            np.testing.assert_allclose(row["actions"][:, 0], actions)

            batch = MossActionCollator(FakeProcessor())([windowed[0]])
            self.assertEqual(batch["action_token_mask"].sum().item(), 4)
            self.assertEqual(batch["robot_state"].shape, (4, 9))
            self.assertEqual(batch["actions"].shape, (4, 2, 7))


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
        if payload.get("format") != "moss_action_v5":
            raise ValueError("streaming parity policy must be moss_action_v5")
        if (
            payload["moss_checkpoint"]["combined_sha256"]
            != checkpoint_fingerprint(checkpoint)["combined_sha256"]
        ):
            raise ValueError("streaming parity policy uses different base weights")
        policy = MossActionVLA(backbone, MossActionConfig(**payload["config"])).to(device)
        load_trainable_state_dict(policy, payload["trainable_state"])
        del payload["trainable_state"]
        backbone = policy.backbone
    backbone.eval()
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
    print(
        json.dumps(
            {
                "status": "PASS",
                "streaming_parity": report,
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
