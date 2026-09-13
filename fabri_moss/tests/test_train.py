import argparse
import copy
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.tests.test_core import TinyFabriVLAPolicy
from fabri_moss.train import (
    compute_flow_kd_loss,
    initialize_adapter_weights,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class DummyActionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.last_flow_input = None

    def _predict_velocity(self, fused_tokens, state, noisy_actions, t, shallow_tokens=None):
        self.last_flow_input = (noisy_actions.detach().clone(), t.detach().clone())
        feat = noisy_actions.sum(dim=-1, keepdim=True) + t.view(-1, 1, 1)
        feat = feat + fused_tokens.sum(dim=-1, keepdim=True)[:, :1, :]
        return noisy_actions * self.scale + feat * 0.1


def test_flow_kd_loss_noise_time_and_frozen_teacher():
    student_head = DummyActionHead()
    teacher_head = copy.deepcopy(student_head)
    teacher_head.eval()
    for p in teacher_head.parameters():
        p.requires_grad = False

    s_deep = torch.randn(1, 16, 1024, requires_grad=True)
    s_shallow = torch.randn(1, 16, 1024)
    t_deep = torch.randn(1, 1024, 1024)
    t_shallow = torch.randn(1, 1024, 1024)
    state = torch.randn(1, 24)
    actions = torch.randn(1, 50, 24)
    action_mask = torch.zeros(1, 50, 24)
    action_mask[:, :, :4] = 1.0

    fixed_noise = torch.randn(1, 50, 24)
    fixed_t = torch.tensor([0.5])

    total_loss, gt_loss, kd_loss, v_s, v_t = compute_flow_kd_loss(
        student_head=student_head,
        teacher_head=teacher_head,
        student_deep=s_deep,
        student_shallow=s_shallow,
        teacher_deep=t_deep,
        teacher_shallow=t_shallow,
        state=state,
        actions=actions,
        action_mask=action_mask,
        kd_weight=1.0,
        fixed_noise=fixed_noise,
        fixed_t=fixed_t,
    )

    assert torch.isfinite(total_loss)
    assert torch.isfinite(gt_loss)
    assert torch.isfinite(kd_loss)
    assert torch.all(v_s[:, :, 4:] == 0)
    assert torch.all(v_t[:, :, 4:] == 0)

    for student_input, teacher_input in zip(student_head.last_flow_input, teacher_head.last_flow_input):
        assert torch.equal(student_input, teacher_input)
    total_loss.backward()
    assert student_head.scale.grad is not None
    assert s_deep.grad is not None
    for p in teacher_head.parameters():
        assert p.grad is None


def test_adapter_checkpoint_roundtrip_and_strict_rejections():
    policy = TinyFabriVLAPolicy(num_layers=14)
    moss_config = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=2,
        shallow_layer=6,
    )
    student = MossInternVL(policy, moss_config)
    optimizer = torch.optim.AdamW(student.bridge_parameters(), lr=1e-4)

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "dummy_sha_abc123"}
    args = argparse.Namespace(steps=10, lr=1e-4, stage="bridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "adapter.pt"
        save_adapter_checkpoint(
            output_path=ckpt_path,
            student_model=student,
            optimizer=optimizer,
            step=5,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
        )
        assert ckpt_path.exists()

        original_readout = student.readout_embeddings.detach().clone()
        original_k = student.cross_blocks['3'].k_proj.weight.detach().clone()
        with torch.no_grad():
            student.readout_embeddings.add_(1.0)
            student.cross_blocks['3'].k_proj.weight.zero_()
        step = load_adapter_checkpoint(
            resume_path=ckpt_path,
            student_model=student,
            optimizer=optimizer,
            expected_stage="bridge",
            expected_norm_stats=norm_stats,
            current_base_metadata=base_meta,
        )
        assert step == 5
        assert torch.equal(student.readout_embeddings, original_readout)
        assert torch.equal(student.cross_blocks['3'].k_proj.weight, original_k)

        # Rejection: Base SHA mismatch
        with pytest.raises(ValueError, match="Base checkpoint SHA256 mismatch"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student,
                optimizer=optimizer,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata={"checkpoint_sha256": "different_sha"},
            )

        # Rejection: Stage mismatch
        with pytest.raises(ValueError, match="Stage mismatch"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student,
                optimizer=optimizer,
                expected_stage="expert",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
            )

        # Rejection: Config mismatch
        mismatched_cfg = MossConfig(cross_layers=(3, 6), num_readout_tokens=8)
        student_mismatch = MossInternVL(policy, mismatched_cfg)
        with pytest.raises(ValueError, match="Config mismatch"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student_mismatch,
                optimizer=optimizer,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
            )

        # Rejection: Norm stats mismatch
        bad_norm = copy.deepcopy(norm_stats)
        bad_norm["action"]["max"][0] = 999.0
        with pytest.raises(ValueError, match="norm_stats mismatch"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student,
                optimizer=optimizer,
                expected_stage="bridge",
                expected_norm_stats=bad_norm,
                current_base_metadata=base_meta,
            )


def test_warmstart_window_mismatch_and_no_side_effects():
    import random

    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg2 = MossConfig(cross_layers=(3, 6, 10, 14), num_readout_tokens=16, max_frames=2, shallow_layer=6)
    student2 = MossInternVL(policy, cfg2)
    student2.set_training_stage("bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "dummy_base_sha"}
    args = argparse.Namespace(steps=5, lr=1e-4, stage="bridge")
    opt2 = torch.optim.AdamW(student2.bridge_parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "source_adapter.pt"
        save_adapter_checkpoint(
            output_path=ckpt_path,
            student_model=student2,
            optimizer=opt2,
            step=10,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
        )

        # Target model with window=5
        cfg5 = MossConfig(cross_layers=(3, 6, 10, 14), num_readout_tokens=16, max_frames=5, shallow_layer=6)
        student5 = MossInternVL(policy, cfg5)
        student5.set_training_stage("bridge")
        opt5 = torch.optim.AdamW(student5.bridge_parameters(), lr=1e-4)

        # Record RNG states
        torch_rng_before = torch.get_rng_state()
        py_rng_before = random.getstate()
        opt5_state_before = copy.deepcopy(opt5.state_dict())

        # Call initialize_adapter_weights (window 2 -> 5)
        provenance = initialize_adapter_weights(
            path=ckpt_path,
            student_model=student5,
            expected_norm_stats=norm_stats,
            current_base_metadata=base_meta,
        )

        assert provenance["source_window"] == 2
        assert provenance["target_window"] == 5
        assert provenance["source_stage"] == "bridge"
        assert provenance["target_stage"] == "bridge"
        assert provenance["source_step"] == 10

        # Check RNG and optimizer states are NOT changed by initialize_adapter_weights
        assert torch.equal(torch.get_rng_state(), torch_rng_before)
        assert random.getstate() == py_rng_before
        assert opt5.state_dict() == opt5_state_before

        # Verify actual param values loaded match student2
        for k in student5.cross_blocks.keys():
            assert torch.equal(student5.cross_blocks[k].k_proj.weight, student2.cross_blocks[k].k_proj.weight)
        assert torch.equal(student5.readout_embeddings, student2.readout_embeddings)

        # Rejection: Source expert -> target bridge
        expert_ckpt_path = Path(tmpdir) / "expert_adapter.pt"
        student2.set_training_stage("expert")
        save_adapter_checkpoint(
            output_path=expert_ckpt_path,
            student_model=student2,
            optimizer=opt2,
            step=12,
            stage="expert",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
        )

        target_bridge = MossInternVL(policy, cfg5)
        target_bridge.set_training_stage("bridge")
        with pytest.raises(ValueError, match="Cannot initialize target stage 'bridge' from source stage 'expert'"):
            initialize_adapter_weights(
                path=expert_ckpt_path,
                student_model=target_bridge,
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
            )


def test_resume_window_mismatch_and_contract_mismatch():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg2 = MossConfig(cross_layers=(3, 6, 10, 14), num_readout_tokens=16, max_frames=2, shallow_layer=6)
    student2 = MossInternVL(policy, cfg2)
    student2.set_training_stage("bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "dummy_base_sha"}
    args = argparse.Namespace(steps=5, lr=1e-4, stage="bridge")
    opt2 = torch.optim.AdamW(student2.bridge_parameters(), lr=1e-4)

    data_contract2 = {
        "context_mode": "window",
        "window": 2,
        "frame_stride": 5,
        "min_context_frames": 1,
        "seed": 4042,
        "max_episodes": None,
        "split": "train",
        "active_episode_ids": [0, 1],
        "metadata_files_sha256": {"info.json": "aaa"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt2.pt"
        save_adapter_checkpoint(
            output_path=ckpt_path,
            student_model=student2,
            optimizer=opt2,
            step=5,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
            data_contract=data_contract2,
        )

        # Resume with max_frames=5 must be rejected (resume 2->5 rejection)
        cfg5 = MossConfig(cross_layers=(3, 6, 10, 14), num_readout_tokens=16, max_frames=5, shallow_layer=6)
        student5 = MossInternVL(policy, cfg5)
        student5.set_training_stage("bridge")
        with pytest.raises(ValueError, match="Config mismatch for field 'max_frames'"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student5,
                optimizer=None,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract2,
            )

        # Resume with mismatched expected_data_contract must be rejected
        mismatched_contract = copy.deepcopy(data_contract2)
        mismatched_contract["seed"] = 9999
        with pytest.raises(ValueError, match="data_contract mismatch for 'seed'"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=None,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=mismatched_contract,
            )

        # Resume with mismatched contract keys (extra key in expected_data_contract)
        extra_key_contract = copy.deepcopy(data_contract2)
        extra_key_contract["extra_field"] = "unexpected"
        with pytest.raises(ValueError, match="data_contract keys mismatch"):
            load_adapter_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=None,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=extra_key_contract,
            )


def test_main_cli_wiring(monkeypatch):
    from PIL import Image
    import sys
    from fabri_moss import train

    policy = TinyFabriVLAPolicy(num_layers=14)
    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "fake_sha_base"}

    # Mock load_native_checkpoint
    monkeypatch.setattr(
        train,
        "load_native_checkpoint",
        lambda **kwargs: (policy, {}, norm_stats, base_meta),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "data"
        out = Path(tmpdir) / "output"
        meta = root / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json_info := '{"chunks_size": 1000}')
        (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "reach"}\n')
        (meta / "episodes.jsonl").write_text('{"episode_index": 0, "length": 5, "tasks": ["reach"]}\n')
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "videos" / "chunk-000" / "observation.images.image").mkdir(parents=True)
        (root / "videos" / "chunk-000" / "observation.images.image" / "episode_000000.mp4").touch()

        import pandas as pd
        df = pd.DataFrame({
            "frame_index": [0, 1, 2, 3, 4],
            "episode_index": [0] * 5,
            "task_index": [0] * 5,
            "observation.state": [[0.5] * 4 for _ in range(5)],
            "action": [[0.1, -0.2, 0.3, 0.4] for _ in range(5)],
        })
        df.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")

        import fabri_moss.data
        monkeypatch.setattr(
            fabri_moss.data,
            "decode_exact_video_frame",
            lambda path, frame_idx: Image.new("RGB", (448, 448)),
        )

        recorded_train_step_batches = []
        real_train_single_step = train.train_single_step

        def mock_train_single_step(**kwargs):
            recorded_train_step_batches.append(kwargs["batch"])
            return {
                "loss": 0.123,
                "gt_loss": 0.05,
                "kd_loss": 0.073,
                "grad_norm": 0.5,
                "kv_grads": {},
            }

        monkeypatch.setattr(train, "train_single_step", mock_train_single_step)

        # Run main with consume mode and fixed-sample
        test_args = [
            "train.py",
            "--data-root", str(root),
            "--output-dir", str(out),
            "--steps", "3",
            "--window", "3",
            "--min-context-frames", "2",
            "--context-mode", "consume",
            "--fixed-sample",
            "--save-every", "1",
            "--threads", "1",
            "--device", "cpu",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        train.main()

        assert (out / "train_metrics.jsonl").exists()
        assert (out / "adapter_step_1.pt").exists()
        assert (out / "adapter_step_2.pt").exists()
        assert (out / "adapter_final.pt").exists()
        assert (out / "metrics.json").exists()

        # Check metrics.json contents for contract and provenance
        import json
        with open(out / "metrics.json") as f:
            summary = json.load(f)
        assert "data_contract" in summary
        contract = summary["data_contract"]
        assert contract["context_mode"] == "consume"
        assert contract["window"] == 3
        assert contract["min_context_frames"] == 2
        assert contract["active_episode_ids"] == [0]
        assert "info.json" in contract["metadata_files_sha256"]

        # Check train_metrics.jsonl contains episode_id, frame_ids, context_mode
        with open(out / "train_metrics.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3
        for entry in lines:
            assert entry["context_mode"] == "consume"
            assert entry["episode_id"] == 0
            assert isinstance(entry["frame_ids"], list)

        # Check resume and init-adapter mutual exclusivity in CLI
        test_mut_args = [
            "train.py",
            "--data-root", str(root),
            "--output-dir", str(out / "sub"),
            "--resume", str(out / "adapter_step_1.pt"),
            "--init-adapter", str(out / "adapter_step_1.pt"),
        ]
        monkeypatch.setattr(sys, "argv", test_mut_args)
        with pytest.raises(ValueError, match="Cannot specify both --resume and --init-adapter"):
            train.main()
