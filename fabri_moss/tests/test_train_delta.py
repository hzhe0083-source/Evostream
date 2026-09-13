import argparse
import copy
import dataclasses
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn

from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.tests.test_core import TinyFabriVLAPolicy
import fabri_moss.train_delta as train_delta
from fabri_moss.train_delta import (
    create_optimizer_and_scheduler,
    evaluate_delta,
    load_delta_checkpoint,
    require_training_device,
    save_delta_checkpoint,
)


class FlowTinyActionHead(nn.Module):
    """TinyActionHead with real trainable linear parameter predicting velocity for flow loss."""
    def __init__(self, hidden_size: int = 128, action_dim: int = 24):
        super().__init__()
        self.action_dim = action_dim
        self.linear = nn.Linear(hidden_size, action_dim)
        self.weight_proj = nn.Linear(action_dim, action_dim)

    def _predict_velocity(self, fused_tokens, state, noisy_actions, t, shallow_tokens=None):
        # fused_tokens: [1, 16, hidden_size]
        # state: [1, 24]
        # noisy_actions: [1, 50, 24]
        # t: [1]
        tok_feat = self.linear(fused_tokens.mean(dim=1, keepdim=True))  # [1, 1, 24]
        v = self.weight_proj(noisy_actions) + tok_feat * 0.05 + t.view(-1, 1, 1) * 0.01
        return v


def setup_tiny_policy_and_data(tmpdir, num_episodes=8):
    root = Path(tmpdir) / "data"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"chunks_size": 1000}')
    (meta / "tasks.jsonl").write_text(
        '{"task_index": 0, "task": "Task A"}\n{"task_index": 1, "task": "Task B"}\n'
    )

    episodes = []
    for i in range(num_episodes):
        t_name = "Task A" if i < (num_episodes // 2) else "Task B"
        episodes.append({"episode_index": i, "length": 6, "tasks": [t_name]})

    with open(meta / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    chunk0_dir = root / "data" / "chunk-000"
    chunk0_dir.mkdir(parents=True)
    video_dir = root / "videos" / "chunk-000" / "observation.images.image"
    video_dir.mkdir(parents=True)

    for i in range(num_episodes):
        (video_dir / f"episode_{i:06d}.mp4").touch()
        task_i = 0 if i < (num_episodes // 2) else 1
        df = pd.DataFrame({
            "frame_index": list(range(6)),
            "episode_index": [i] * 6,
            "task_index": [task_i] * 6,
            "observation.state": [[0.5] * 4 for _ in range(6)],
            "action": [[0.1, -0.2, 0.3, 0.4] for _ in range(6)],
        })
        df.to_parquet(chunk0_dir / f"episode_{i:06d}.parquet")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    return root, norm_stats


def test_create_optimizer_and_scheduler_parameter_grouping():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    student.set_training_stage("expert")

    optimizer, scheduler = create_optimizer_and_scheduler(
        student_model=student,
        stage="expert",
        lr=1e-4,
        head_lr=1e-5,
        weight_decay=1e-4,
        total_steps=100,
        warmup_steps=10,
    )

    group_initial_lrs = [g.get("initial_lr", g["lr"]) for g in optimizer.param_groups]
    group_wds = [g["weight_decay"] for g in optimizer.param_groups]

    assert 1e-4 in group_initial_lrs
    assert 1e-5 in group_initial_lrs
    assert 0.0 in group_wds
    assert 1e-4 in group_wds

    initial_lr = scheduler.get_last_lr()[0]
    assert initial_lr == 0.0

    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] > 0.0


def test_delta_checkpoint_save_and_resume_roundtrip():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    student.set_training_stage("bridge")

    optimizer, scheduler = create_optimizer_and_scheduler(
        student_model=student,
        stage="bridge",
        lr=1e-4,
        total_steps=50,
        warmup_steps=5,
    )

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}
    data_contract = {
        "dataset_class": "MetaWorldEpisodes",
        "format": "fabri_delta_v1",
        "split": "train",
        "max_chunk_size": 5,
        "seed": 4042,
        "augmentation": True,
    }
    training_contract = {"stage": "bridge", "lr": 1e-4}
    args = argparse.Namespace(epochs=5, lr=1e-4, stage="bridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "last.pt"

        save_delta_checkpoint(
            output_path=ckpt_path,
            student_model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            epoch=2,
            next_episode_cursor=4,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
            train_data_contract=data_contract,
            val_data_contract=data_contract,
            training_contract=training_contract,
        )

        assert ckpt_path.exists()

        student2 = MossInternVL(policy, cfg)
        student2.set_training_stage("bridge")
        opt2, sched2 = create_optimizer_and_scheduler(student2, stage="bridge")

        resume_info = load_delta_checkpoint(
            resume_path=ckpt_path,
            student_model=student2,
            optimizer=opt2,
            scheduler=sched2,
            expected_stage="bridge",
            expected_norm_stats=norm_stats,
            current_base_metadata=base_meta,
            expected_data_contract=data_contract,
            expected_training_contract=training_contract,
        )

        assert resume_info["step"] == 10
        assert resume_info["epoch"] == 2
        assert resume_info["next_episode_cursor"] == 4

        for p1, p2 in zip(student.cross_blocks.parameters(), student2.cross_blocks.parameters()):
            assert torch.equal(p1, p2)


def test_resume_strict_rejection_on_mismatch_and_no_weight_mutation():
    """Verify that any contract/shape/hash mismatch rejects before modifying weights."""
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    opt, sched = create_optimizer_and_scheduler(student, stage="bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}
    data_contract = {"seed": 4042, "split": "train"}
    training_contract = {"lr": 1e-4}
    args = argparse.Namespace(epochs=5, lr=1e-4, stage="bridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "last.pt"
        save_delta_checkpoint(
            output_path=ckpt_path,
            student_model=student,
            optimizer=opt,
            scheduler=sched,
            step=1,
            epoch=0,
            next_episode_cursor=1,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
            train_data_contract=data_contract,
            val_data_contract=data_contract,
            training_contract=training_contract,
        )

        # Record initial weights of student2
        student2 = MossInternVL(policy, cfg)
        init_weight = student2.readout_embeddings.clone()
        opt2, sched2 = create_optimizer_and_scheduler(student2, stage="bridge")

        # 1. Base SHA mismatch
        with pytest.raises(ValueError, match="Base checkpoint SHA256 mismatch"):
            load_delta_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata={"checkpoint_sha256": "different_sha"},
                expected_data_contract=data_contract,
                expected_training_contract=training_contract,
            )
        assert torch.equal(student2.readout_embeddings, init_weight)

        # 2. Data contract mismatch
        with pytest.raises(ValueError, match="data_contract mismatch"):
            load_delta_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract={"seed": 9999},
                expected_training_contract=training_contract,
            )
        assert torch.equal(student2.readout_embeddings, init_weight)

        # 3. Training contract mismatch
        with pytest.raises(ValueError, match="training_contract mismatch"):
            load_delta_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract,
                expected_training_contract={"lr": 5e-5},
            )
        assert torch.equal(student2.readout_embeddings, init_weight)


def test_evaluate_delta_determinism_and_rng_isolation(monkeypatch):
    """Verify evaluate_delta returns identical results and leaves training RNG & train mode unchanged."""
    policy = TinyFabriVLAPolicy(num_layers=14)
    policy.action_head = FlowTinyActionHead(hidden_size=128, action_dim=24)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    student.train()

    def mock_get_vl_embeddings(images, image_mask, prompt, return_cls_only=False, shallow_layer_index=6):
        return torch.randn(1, 16, 128), torch.randn(1, 16, 128)

    monkeypatch.setattr(policy, "get_vl_embeddings", mock_get_vl_embeddings, raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        root, norm_stats = setup_tiny_policy_and_data(tmpdir, num_episodes=4)
        monkeypatch.setattr(
            "fabri_moss.delta_data.decode_all_video_frames",
            lambda video_path, target_frame_indices: {idx: Image.new("RGB", (32, 32)) for idx in target_frame_indices},
        )

        from fabri_moss.delta_data import MetaWorldEpisodes
        val_dataset = MetaWorldEpisodes(
            root=root,
            norm_stats=norm_stats,
            horizon=10,
            split="val",
            val_fraction=0.5,
            seed=4042,
            augmentation=False,
        )

        teacher_head = copy.deepcopy(policy.action_head)
        teacher_head.eval()

        torch_rng_before = torch.get_rng_state()

        res1 = evaluate_delta(
            student_model=student,
            teacher_policy=policy,
            teacher_head=teacher_head,
            val_dataset=val_dataset,
            device="cpu",
            val_episodes=2,
            kd_weight=1.0,
            seed=4042,
        )

        torch_rng_middle = torch.get_rng_state()
        assert torch.equal(torch_rng_before, torch_rng_middle)
        assert student.training is True  # train mode restored

        res2 = evaluate_delta(
            student_model=student,
            teacher_policy=policy,
            teacher_head=teacher_head,
            val_dataset=val_dataset,
            device="cpu",
            val_episodes=2,
            kd_weight=1.0,
            seed=4042,
        )

        assert res1["val_total_loss"] == res2["val_total_loss"]
        assert res1["val_gt_loss"] == res2["val_gt_loss"]
        assert res1["val_kd_loss"] == res2["val_kd_loss"]
        assert res1["val_episode_ids"] == res2["val_episode_ids"]


def test_cli_training_actual_parameter_updates_and_tbptt(monkeypatch):
    """Test full CLI run: real forward_delta, TBPTT recurrence across >4 decisions, parameters actually change."""
    policy = TinyFabriVLAPolicy(num_layers=14)
    flow_head = FlowTinyActionHead(hidden_size=128, action_dim=24)
    policy.action_head = flow_head

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "fake_base_sha"}

    def mock_load(**kwargs):
        return policy, {}, norm_stats, base_meta

    def mock_get_vl_embeddings(images, image_mask, prompt, return_cls_only=False, shallow_layer_index=6):
        return torch.randn(1, 16, 128), torch.randn(1, 16, 128)

    monkeypatch.setattr(policy, "get_vl_embeddings", mock_get_vl_embeddings, raising=False)
    monkeypatch.setattr(train_delta, "load_native_checkpoint", mock_load)
    monkeypatch.setattr(train_delta, "assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr(train_delta, "require_training_device", lambda p, d: None)

    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = setup_tiny_policy_and_data(tmpdir, num_episodes=8)
        out = Path(tmpdir) / "output"

        monkeypatch.setattr(
            "fabri_moss.delta_data.decode_all_video_frames",
            lambda video_path, target_frame_indices: {idx: Image.new("RGB", (32, 32)) for idx in target_frame_indices},
        )

        cli_args = argparse.Namespace(
            data_root=str(root),
            output_dir=str(out),
            fabri_root="/root/FabriVLA",
            checkpoint="/root/models/FabriVLA/checkpoint_step_93000.pt",
            vlm="/root/models/InternVL3_5-1B",
            device="cpu",
            arm_key="metaworld_sawyer",
            stage="expert",
            lr=1e-3,
            head_lr=1e-3,
            weight_decay=0.0,
            kd_weight=1.0,
            grad_clip_norm=1.0,
            seed=4042,
            epochs=2,
            episodes_per_step=2,
            tbptt_decisions=2,
            warmup_steps=0,  # 0 warmup to immediately update with full lr
            save_every=1,
            val_every=1,
            val_episodes=2,
            num_workers=0,
            threads=1,
            max_episodes=None,
            max_updates=3,
            resume=None,
            init_adapter=None,
            augmentation=True,
        )

        initial_head_weight = flow_head.linear.weight.detach().clone()
        train_delta.main(cli_args)

        # Verify weights actually updated
        updated_head_weight = flow_head.linear.weight.detach().clone()
        assert not torch.equal(initial_head_weight, updated_head_weight)

        # Check files
        assert (out / "run_config.json").exists()
        assert (out / "train_metrics.jsonl").exists()
        assert (out / "last.pt").exists()
        assert (out / "final.pt").exists()

        with open(out / "train_metrics.jsonl") as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 3
        for l in lines:
            assert "lr_used" in l
            assert "lr_next" in l
            assert "grad_norm" in l
            assert l["grad_norm"] > 0


def test_continuous_run4_vs_run2_then_resume4_parity(monkeypatch):
    """Verify that running 4 updates continuously vs running 2 updates and resuming to 4 yields parity."""
    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "fake_base_sha"}

    def make_policy():
        torch.manual_seed(9999)
        pol = TinyFabriVLAPolicy(num_layers=14)
        fh = FlowTinyActionHead(hidden_size=128, action_dim=24)
        pol.action_head = fh
        fh.linear.weight.data.fill_(0.5)

        def mock_get_vl_embeddings(images, image_mask, prompt, return_cls_only=False, shallow_layer_index=6):
            g = torch.Generator().manual_seed(1234)
            return torch.randn(1, 16, 128, generator=g), torch.randn(1, 16, 128, generator=g)

        pol.get_vl_embeddings = mock_get_vl_embeddings
        return pol

    monkeypatch.setattr(train_delta, "load_native_checkpoint", lambda **kw: (make_policy(), {}, norm_stats, base_meta))
    monkeypatch.setattr(train_delta, "assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr(train_delta, "require_training_device", lambda p, d: None)

    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = setup_tiny_policy_and_data(tmpdir, num_episodes=8)
        dir_cont = Path(tmpdir) / "continuous"
        dir_res = Path(tmpdir) / "resumed"

        monkeypatch.setattr(
            "fabri_moss.delta_data.decode_all_video_frames",
            lambda video_path, target_frame_indices: {idx: Image.new("RGB", (32, 32)) for idx in target_frame_indices},
        )

        base_kwargs = dict(
            data_root=str(root),
            fabri_root="/root/FabriVLA",
            checkpoint="/root/models/FabriVLA/checkpoint_step_93000.pt",
            vlm="/root/models/InternVL3_5-1B",
            device="cpu",
            arm_key="metaworld_sawyer",
            stage="expert",
            lr=1e-3,
            head_lr=1e-3,
            weight_decay=0.0,
            kd_weight=1.0,
            grad_clip_norm=1.0,
            seed=4042,
            epochs=2,
            episodes_per_step=2,
            tbptt_decisions=2,
            warmup_steps=2,
            save_every=1,
            val_every=0,
            val_episodes=0,
            num_workers=0,
            threads=1,
            max_episodes=None,
            init_adapter=None,
            augmentation=False,
        )

        # Run continuous 4 updates
        args_cont = argparse.Namespace(output_dir=str(dir_cont), max_updates=4, resume=None, **base_kwargs)
        train_delta.main(args_cont)

        # Run 2 updates
        args_part1 = argparse.Namespace(output_dir=str(dir_res), max_updates=2, resume=None, **base_kwargs)
        train_delta.main(args_part1)

        # Resume to 4 updates
        args_part2 = argparse.Namespace(
            output_dir=str(dir_res),
            max_updates=4,
            resume=str(dir_res / "last.pt"),
            **base_kwargs,
        )
        train_delta.main(args_part2)

        ckpt_cont = torch.load(str(dir_cont / "final.pt"), weights_only=False)
        ckpt_res = torch.load(str(dir_res / "final.pt"), weights_only=False)

        assert ckpt_cont["step"] == 4
        assert ckpt_res["step"] == 4

        # Verify cross blocks and action head parity
        for k in ckpt_cont["cross_blocks"]:
            assert torch.allclose(ckpt_cont["cross_blocks"][k], ckpt_res["cross_blocks"][k], atol=1e-2)
        for k in ckpt_cont["action_head"]:
            assert torch.allclose(ckpt_cont["action_head"][k], ckpt_res["action_head"][k], atol=1e-4)

        # Verify episode sequences in train_metrics.jsonl
        with open(dir_cont / "train_metrics.jsonl") as f:
            lines_cont = [json.loads(l) for l in f]
        with open(dir_res / "train_metrics.jsonl") as f:
            lines_res = [json.loads(l) for l in f]

        assert len(lines_cont) == 4
        assert len(lines_res) == 4
        for c, r in zip(lines_cont, lines_res):
            assert c["step"] == r["step"]
            assert c["episode_ids"] == r["episode_ids"]
            assert math.isclose(c["loss"], r["loss"], rel_tol=1e-4, abs_tol=1e-4)
            assert "memory_bytes" in c and "frame_count" in c and "norm" in c
            assert "observed_frames" in c

        # Verify metrics.json summary and run_config
        assert (dir_cont / "metrics.json").exists()
        summary_cont = json.loads((dir_cont / "metrics.json").read_text())
        assert summary_cont["status"] == "paused_max_updates"
        assert summary_cont["step"] == 4
        assert summary_cont["checkpoint"] == str(dir_cont / "last.pt")

        run_config = json.loads((dir_cont / "run_config.json").read_text())
        assert "trainable_parameter_count" in run_config
        assert "frozen_parameter_count" in run_config
        assert "model_config" in run_config


def test_exact_extra_key_contract_rejection():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    opt, sched = create_optimizer_and_scheduler(student, stage="bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}
    data_contract = {"seed": 4042, "split": "train"}
    training_contract = {"lr": 1e-4}
    args = argparse.Namespace(epochs=5, lr=1e-4, stage="bridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt_extra_data.pt"
        extra_data = {"seed": 4042, "split": "train", "extra_unwanted": True}
        save_delta_checkpoint(
            output_path=ckpt_path,
            student_model=student,
            optimizer=opt,
            scheduler=sched,
            step=1,
            epoch=0,
            next_episode_cursor=1,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
            train_data_contract=extra_data,
            val_data_contract=data_contract,
            training_contract=training_contract,
        )

        student2 = MossInternVL(policy, cfg)
        opt2, sched2 = create_optimizer_and_scheduler(student2, stage="bridge")

        with pytest.raises(ValueError, match="train_data_contract mismatch"):
            load_delta_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract,
                expected_training_contract=training_contract,
            )

        ckpt_path2 = Path(tmpdir) / "ckpt_extra_tr.pt"
        extra_tr = {"lr": 1e-4, "extra_flag": "forbidden"}
        save_delta_checkpoint(
            output_path=ckpt_path2,
            student_model=student,
            optimizer=opt,
            scheduler=sched,
            step=1,
            epoch=0,
            next_episode_cursor=1,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
            train_data_contract=data_contract,
            val_data_contract=data_contract,
            training_contract=extra_tr,
        )

        with pytest.raises(ValueError, match="training_contract mismatch"):
            load_delta_checkpoint(
                resume_path=ckpt_path2,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract,
                expected_training_contract=training_contract,
            )


def test_val_data_contract_strict_validation():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    opt, sched = create_optimizer_and_scheduler(student, stage="bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}
    data_contract = {"seed": 4042, "split": "train"}
    val_contract = {"seed": 4042, "split": "val"}
    training_contract = {"lr": 1e-4}
    args = argparse.Namespace(epochs=5, lr=1e-4, stage="bridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt_val.pt"
        save_delta_checkpoint(
            output_path=ckpt_path,
            student_model=student,
            optimizer=opt,
            scheduler=sched,
            step=1,
            epoch=0,
            next_episode_cursor=1,
            stage="bridge",
            args=args,
            norm_stats=norm_stats,
            base_metadata=base_meta,
            train_data_contract=data_contract,
            val_data_contract=val_contract,
            training_contract=training_contract,
        )

        student2 = MossInternVL(policy, cfg)
        opt2, sched2 = create_optimizer_and_scheduler(student2, stage="bridge")

        with pytest.raises(ValueError, match="val_data_contract mismatch"):
            load_delta_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract,
                expected_training_contract=training_contract,
                expected_val_data_contract={"seed": 9999, "split": "val"},
            )

        raw_ckpt = torch.load(str(ckpt_path), weights_only=False)
        del raw_ckpt["val_data_contract"]
        no_val_path = Path(tmpdir) / "no_val.pt"
        torch.save(raw_ckpt, str(no_val_path))

        with pytest.raises(KeyError, match="val_data_contract"):
            load_delta_checkpoint(
                resume_path=no_val_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract,
                expected_training_contract=training_contract,
                expected_val_data_contract=val_contract,
            )


def test_cursor_and_step_validation_boundaries():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    opt, sched = create_optimizer_and_scheduler(student, stage="bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}
    train_c = {"active_episode_ids": list(range(8))}
    val_c = {"active_episode_ids": [8]}
    train_contract = {
        "episodes_per_step": 4,
        "epochs": 2,
        "scheduler_budget": 4,
    }
    args = argparse.Namespace(epochs=2, lr=1e-4, stage="bridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        def make_ckpt(step, epoch, cursor, last_epoch=None, step_val=None):
            s_val = step if step_val is None else step_val
            p = Path(tmpdir) / f"ckpt_{step}_{epoch}_{cursor}_{random.randint(0, 100000)}.pt"
            sched_dict = sched.state_dict()
            sched_dict["last_epoch"] = step if last_epoch is None else last_epoch
            ckpt = {
                "format": "fabri_delta_v1",
                "step": s_val,
                "epoch": epoch,
                "next_episode_cursor": cursor,
                "stage": "bridge",
                "config": dataclasses.asdict(student.config),
                "cross_blocks": student.cross_blocks.state_dict(),
                "readout_embeddings": student.readout_embeddings.detach().cpu(),
                "optimizer": opt.state_dict(),
                "scheduler": sched_dict,
                "norm_stats": norm_stats,
                "base_metadata": base_meta,
                "train_data_contract": train_c,
                "val_data_contract": val_c,
                "training_contract": train_contract,
                "rng_state": {"torch": torch.get_rng_state(), "random": random.getstate()},
            }
            torch.save(ckpt, str(p))
            return p

        student2 = MossInternVL(policy, cfg)
        opt2, sched2 = create_optimizer_and_scheduler(student2, stage="bridge")

        p1 = make_ckpt(step=1, epoch=0, cursor=2)
        with pytest.raises(ValueError, match="not on an optimizer boundary"):
            load_delta_checkpoint(
                resume_path=p1,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=train_c,
                expected_training_contract=train_contract,
            )

        p2 = make_ckpt(step=2, epoch=0, cursor=10)
        with pytest.raises(ValueError, match="exceeds dataset episode count"):
            load_delta_checkpoint(
                resume_path=p2,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=train_c,
                expected_training_contract=train_contract,
            )

        p3 = make_ckpt(step=4, epoch=2, cursor=4)
        with pytest.raises(ValueError, match="must have next_episode_cursor=0"):
            load_delta_checkpoint(
                resume_path=p3,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=train_c,
                expected_training_contract=train_contract,
            )

        p4 = make_ckpt(step=2, epoch=1, cursor=4, last_epoch=2)
        with pytest.raises(ValueError, match="Checkpoint step mismatch"):
            load_delta_checkpoint(
                resume_path=p4,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=train_c,
                expected_training_contract=train_contract,
            )

        p5 = make_ckpt(step=3, epoch=1, cursor=4, last_epoch=1)
        with pytest.raises(ValueError, match="Scheduler state last_epoch"):
            load_delta_checkpoint(
                resume_path=p5,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=train_c,
                expected_training_contract=train_contract,
            )

        p6 = make_ckpt(step=1, epoch=0, cursor=4, step_val=True)
        with pytest.raises(TypeError, match="must be an integer"):
            load_delta_checkpoint(
                resume_path=p6,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=train_c,
                expected_training_contract=train_contract,
            )


def test_validate_args_boundaries():
    base = dict(
        data_root="/tmp",
        output_dir="/tmp/out",
        epochs=5,
        episodes_per_step=4,
        tbptt_decisions=4,
        save_every=25,
        threads=4,
        warmup_steps=100,
        val_every=100,
        val_episodes=50,
        num_workers=2,
        max_episodes=None,
        max_updates=None,
        lr=1e-4,
        head_lr=1e-5,
        grad_clip_norm=1.0,
        weight_decay=1e-4,
        kd_weight=1.0,
    )

    train_delta.validate_args(argparse.Namespace(**base))

    with pytest.raises(ValueError, match="epochs"):
        train_delta.validate_args(argparse.Namespace(**{**base, "epochs": 0}))

    with pytest.raises(ValueError, match="episodes_per_step"):
        train_delta.validate_args(argparse.Namespace(**{**base, "episodes_per_step": -1}))

    with pytest.raises(ValueError, match="tbptt_decisions"):
        train_delta.validate_args(argparse.Namespace(**{**base, "tbptt_decisions": 0}))

    with pytest.raises(ValueError, match="save_every"):
        train_delta.validate_args(argparse.Namespace(**{**base, "save_every": -5}))

    with pytest.raises(ValueError, match="threads"):
        train_delta.validate_args(argparse.Namespace(**{**base, "threads": 0}))

    with pytest.raises(ValueError, match="warmup_steps"):
        train_delta.validate_args(argparse.Namespace(**{**base, "warmup_steps": -1}))

    with pytest.raises(ValueError, match="val_every"):
        train_delta.validate_args(argparse.Namespace(**{**base, "val_every": -1}))

    with pytest.raises(ValueError, match="val_episodes"):
        train_delta.validate_args(argparse.Namespace(**{**base, "val_episodes": -1}))

    with pytest.raises(ValueError, match="num_workers"):
        train_delta.validate_args(argparse.Namespace(**{**base, "num_workers": -1}))

    with pytest.raises(ValueError, match="max_episodes"):
        train_delta.validate_args(argparse.Namespace(**{**base, "max_episodes": 0}))

    with pytest.raises(ValueError, match="max_updates"):
        train_delta.validate_args(argparse.Namespace(**{**base, "max_updates": -2}))

    with pytest.raises(ValueError, match="lr"):
        train_delta.validate_args(argparse.Namespace(**{**base, "lr": 0.0}))

    with pytest.raises(ValueError, match="lr"):
        train_delta.validate_args(argparse.Namespace(**{**base, "lr": float("inf")}))

    with pytest.raises(ValueError, match="head_lr"):
        train_delta.validate_args(argparse.Namespace(**{**base, "head_lr": -1e-4}))

    with pytest.raises(ValueError, match="grad_clip_norm"):
        train_delta.validate_args(argparse.Namespace(**{**base, "grad_clip_norm": 0.0}))

    with pytest.raises(ValueError, match="weight_decay"):
        train_delta.validate_args(argparse.Namespace(**{**base, "weight_decay": -0.01}))

    with pytest.raises(ValueError, match="kd_weight"):
        train_delta.validate_args(argparse.Namespace(**{**base, "kd_weight": -1.0}))


def test_completed_resume_exits_early_without_overwriting(capsys, monkeypatch, tmp_path):
    policy = TinyFabriVLAPolicy(num_layers=14)
    policy.action_head = FlowTinyActionHead(hidden_size=128, action_dim=24)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    opt, sched = create_optimizer_and_scheduler(student, stage="bridge")

    root, norm_stats = setup_tiny_policy_and_data(str(tmp_path), num_episodes=4)
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}

    monkeypatch.setattr(
        train_delta,
        "load_native_checkpoint",
        lambda **kwargs: (policy, {}, norm_stats, base_meta),
    )
    monkeypatch.setattr(train_delta, "require_training_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_delta, "assert_native_fa2", lambda *args, **kwargs: "Mock FA2")

    from fabri_moss.delta_data import MetaWorldEpisodes
    train_ds = MetaWorldEpisodes(
        root=root,
        norm_stats=norm_stats,
        horizon=50,
        max_chunk_size=5,
        split="train",
        seed=4042,
        val_fraction=0.1,
        augmentation=False,
    )
    val_ds = MetaWorldEpisodes(
        root=root,
        norm_stats=norm_stats,
        horizon=50,
        max_chunk_size=5,
        split="val",
        seed=4042,
        val_fraction=0.1,
        augmentation=False,
    )
    train_c = train_ds.get_data_contract()
    val_c = val_ds.get_data_contract()

    est_steps_per_epoch = math.ceil(len(train_ds) / 2)
    scheduler_budget = 2 * est_steps_per_epoch

    training_c = {
        "format": "fabri_delta_v1",
        "stage": "bridge",
        "lr": 1e-4,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "kd_weight": 1.0,
        "grad_clip_norm": 1.0,
        "epochs": 2,
        "episodes_per_step": 2,
        "tbptt_decisions": 2,
        "warmup_steps": 0,
        "scheduler_budget": scheduler_budget,
        "validation_noise": "uniform_fixed_seed",
        "validation_t": 0.5,
        "val_every": 0,
        "val_episodes": 0,
        "teacher_backend": "flash_attention_2",
    }

    final_step = 2 * est_steps_per_epoch
    for _ in range(final_step):
        opt.step()
        sched.step()

    ckpt_path = tmp_path / "completed.pt"
    save_delta_checkpoint(
        output_path=ckpt_path,
        student_model=student,
        optimizer=opt,
        scheduler=sched,
        step=final_step,
        epoch=2,
        next_episode_cursor=0,
        stage="bridge",
        args=argparse.Namespace(epochs=2),
        norm_stats=norm_stats,
        base_metadata=base_meta,
        train_data_contract=train_c,
        val_data_contract=val_c,
        training_contract=training_c,
    )

    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    args = argparse.Namespace(
        data_root=str(root),
        output_dir=str(out_dir),
        resume=str(ckpt_path),
        init_adapter=None,
        epochs=2,
        episodes_per_step=2,
        tbptt_decisions=2,
        save_every=1,
        threads=1,
        warmup_steps=0,
        val_every=0,
        val_episodes=0,
        num_workers=0,
        max_episodes=None,
        max_updates=None,
        lr=1e-4,
        head_lr=1e-5,
        grad_clip_norm=1.0,
        weight_decay=1e-4,
        kd_weight=1.0,
        stage="bridge",
        seed=4042,
        device="cpu",
        arm_key="metaworld_sawyer",
        fabri_root="/tmp",
        checkpoint="/tmp/ckpt",
        vlm="/tmp/vlm",
        augmentation=False,
    )

    train_delta.main(args)
    captured = capsys.readouterr().out
    assert "[Finished]" in captured
    assert not (out_dir / "run_config.json").exists()
    assert not (out_dir / "train_metrics.jsonl").exists()


def test_gpu_model_resume_requires_torch_cuda():
    policy = TinyFabriVLAPolicy(num_layers=14)
    cfg = MossConfig(
        cross_layers=(3, 6, 10, 14),
        num_readout_tokens=16,
        max_frames=5,
        shallow_layer=6,
        memory_mode="delta",
    )
    student = MossInternVL(policy, cfg)
    opt, sched = create_optimizer_and_scheduler(student, stage="bridge")

    norm_stats = {
        "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }
    base_meta = {"checkpoint_sha256": "delta_base_sha_123"}
    data_contract = {"seed": 4042, "split": "train"}
    training_contract = {"lr": 1e-4}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "no_cuda_rng.pt"
        ckpt = {
            "format": "fabri_delta_v1",
            "step": 1,
            "epoch": 0,
            "next_episode_cursor": 1,
            "stage": "bridge",
            "config": dataclasses.asdict(student.config),
            "cross_blocks": student.cross_blocks.state_dict(),
            "readout_embeddings": student.readout_embeddings.detach().cpu(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "norm_stats": norm_stats,
            "base_metadata": base_meta,
            "train_data_contract": data_contract,
            "val_data_contract": data_contract,
            "training_contract": training_contract,
            "rng_state": {"torch": torch.get_rng_state(), "random": random.getstate()},
        }
        torch.save(ckpt, str(ckpt_path))

        student2 = MossInternVL(policy, cfg)
        opt2, sched2 = create_optimizer_and_scheduler(student2, stage="bridge")

        with pytest.raises(KeyError, match="torch_cuda"):
            load_delta_checkpoint(
                resume_path=ckpt_path,
                student_model=student2,
                optimizer=opt2,
                scheduler=sched2,
                expected_stage="bridge",
                expected_norm_stats=norm_stats,
                current_base_metadata=base_meta,
                expected_data_contract=data_contract,
                expected_training_contract=training_contract,
                device="cuda:0",
            )
