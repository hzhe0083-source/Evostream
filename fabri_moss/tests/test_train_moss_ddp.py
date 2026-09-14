from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import fabri_moss.train_moss_ddp as trainer


def test_stage_epoch_defaults_are_absolute_targets():
    bridge = trainer.parse_args(["--data-root", "data", "--output-dir", "out", "--stage", "bridge"])
    joint = trainer.parse_args(["--data-root", "data", "--output-dir", "out", "--stage", "joint"])
    assert bridge.epochs == 1
    assert joint.epochs == 2
    assert joint.joint_train_action_expert is False
    assert joint.native_kd_weight == 0.0


def test_chunk_targets_share_encoding_and_read_only_prefixes(monkeypatch):
    weight = torch.nn.Parameter(torch.tensor(2.0))
    seen, encoded, states = [], [], []
    policy = SimpleNamespace(action_head=object(), get_vl_embeddings=lambda **kw: (torch.zeros(1), None))
    student = SimpleNamespace(
        policy=policy,
        encode_image=lambda images: encoded.append(images) or torch.tensor(float(images[0])),
        project_frame=lambda features, fid: features * weight,
        read_memory=lambda frames, prompt: seen.append(len(frames)) or (sum(frames), None),
    )
    def loss(**kw):
        states.append(kw['state'].clone())
        value = kw['student_deep'].square()
        return value, value, value * 0, None, None
    monkeypatch.setattr(trainer, 'compute_flow_kd_loss', loss)
    sample = {'images_window': [[1], [2], [3]], 'frame_ids': [10, 11, 12], 'prompt': 'test',
              'state': torch.tensor([[1.], [2.], [3.]]), 'state_mask': torch.ones(3, 1),
              'actions': torch.zeros(3, 2, 1), 'action_mask': torch.ones(3, 2, 1),
              'visible_counts': [1, 2, 3]}
    total, _, _, count = trainer._sample_loss(student, object(), sample, 'cpu', 1.0)
    total.backward()
    assert len(encoded) == 3 and seen == [1, 2, 3] and count == 6
    assert torch.equal(torch.cat(states), sample['state'])
    assert weight.grad.item() == pytest.approx(368.)


@dataclass
class _Config:
    cross_layers: tuple = (1,)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_blocks = torch.nn.Linear(1, 1)
        self.readout_embeddings = torch.nn.Parameter(torch.ones(1))
        self.config = _Config()
        self.training_stage = 'bridge'
    def set_training_stage(self, stage):
        self.training_stage = stage


def test_v2_resume_restores_optimizer_and_cursor(tmp_path):
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    (model.cross_blocks.weight.sum() + model.readout_embeddings.sum()).backward()
    optimizer.step()
    args = SimpleNamespace(stage='bridge')
    path = tmp_path / 'last.pt'
    meta, contract = {'checkpoint_sha256': 'source93k'}, {'context_mode': 'causal'}
    trainer._save(path, model, optimizer, 7, 2, 3, 160, args, {}, meta, contract, 1,
                  trainer._rng_states(1, 'cpu'))
    clone = _Model()
    restored_optimizer = torch.optim.AdamW(clone.parameters(), lr=.5)
    cursor = trainer._load_resume(path, clone, restored_optimizer, args, meta, contract, 1)
    assert cursor == (7, 2, 3, 160)
    assert torch.equal(model.cross_blocks.weight, clone.cross_blocks.weight)
    assert restored_optimizer.param_groups[0]['lr'] == .1
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_resume_rejects_writer_checkpoint(tmp_path):
    path = tmp_path / 'writer.pt'
    torch.save({'format': 'predictive_adapter_v1'}, path)
    with pytest.raises(ValueError, match='moss_cross_adapter_v2'):
        trainer._load_resume(path, _Model(), None, SimpleNamespace(stage='bridge'), {}, {}, 1)


def test_joint_surface_is_action_head_plus_fp32_qkvo_lora():
    from fabri_moss.core import MossConfig, MossInternVL
    from fabri_moss.tests.test_core import TinyFabriVLAPolicy

    policy = TinyFabriVLAPolicy(hidden_size=128, num_layers=6)
    model = MossInternVL(policy, MossConfig(cross_layers=(2, 4, 6), max_frames=3))
    args = SimpleNamespace(lora_rank=8, lora_alpha=16.0, lora_dropout=0.1)
    modules = trainer.configure_lora(model, args)
    model.set_training_stage("joint")
    trainable = trainer.configure_trainable_parameters(model, "joint")

    assert len(modules) == 12  # 3 MOSS layers x q/k/v/o
    assert all(m.rank == 8 and m.alpha == 16.0 and m.dropout_p == 0.1 for m in modules.values())
    assert all(p.dtype == torch.float32 for p in trainable)
    names = trainer._trainable_names(model)
    assert names and all(("action_head" in n or ".lora_" in n) for n in names)
    assert not any(".base." in n for n in names)
    assert not any("cross_blocks" in n or "readout_embeddings" in n for n in names)


def test_joint_recovery_can_freeze_action_expert():
    from fabri_moss.core import MossConfig, MossInternVL
    from fabri_moss.tests.test_core import TinyFabriVLAPolicy

    model = MossInternVL(
        TinyFabriVLAPolicy(hidden_size=128, num_layers=6),
        MossConfig(cross_layers=(2, 4, 6), max_frames=3),
    )
    trainer.configure_lora(model, SimpleNamespace(lora_rank=8, lora_alpha=16.0, lora_dropout=0.1))
    model.set_training_stage("joint")
    trainable = trainer.configure_trainable_parameters(model, "joint", train_action_expert=False)
    names = trainer._trainable_names(model)
    assert names and all(".lora_" in n for n in names)
    assert not any("action_head" in n for n in names)
    assert all(p.dtype == torch.float32 for p in trainable)


def test_native_kd_adds_current_only_anchor_with_fixed_inputs(monkeypatch):
    calls = []
    policy = SimpleNamespace(action_head=object(), get_vl_embeddings=lambda **kw: (torch.zeros(1), None))
    student = SimpleNamespace(
        policy=policy,
        encode_image=lambda images: torch.tensor(float(images[0])),
        project_frame=lambda features, fid, **kw: features,
        read_memory=lambda frames, prompt, **kw: calls.append(len(frames)) or (torch.tensor([[float(len(frames))]], requires_grad=True), None),
    )
    def fake_loss(**kw):
        value = kw["student_deep"].sum()
        return value, value, value * 0 + 2.0, None, None
    monkeypatch.setattr(trainer, "compute_flow_kd_loss", fake_loss)
    sample = {
        "images_window": [[1], [2]], "frame_ids": [10, 11], "prompt": "test",
        "state": torch.zeros(1, 1), "state_mask": torch.ones(1, 1),
        "actions": torch.zeros(1, 2, 1), "action_mask": torch.ones(1, 2, 1),
        "visible_counts": [2], "fixed_noise": torch.zeros(1, 2, 1), "fixed_t": torch.tensor([0.5]),
    }
    total, _, kd, count = trainer._sample_loss(
        student, object(), sample, "cpu", 1.0, native_kd_weight=0.5
    )
    assert calls == [2, 1]
    assert count == 2
    assert total.item() == pytest.approx(6.0)
    assert kd.item() == pytest.approx(6.0)


def test_lora_context_bypasses_current_only_sample_and_keeps_history_sample_active():
    from fabri_moss.lora import FP32LoRALinear, lora_context

    base = torch.nn.Linear(4, 4, bias=False)
    module = FP32LoRALinear(base, rank=1, alpha=1.0, dropout=0.0)
    with torch.no_grad():
        module.lora_A.fill_(1.0)
        module.lora_B.fill_(1.0)
    inputs = torch.ones(2, 3, 4)
    native = base(inputs)
    with lora_context(module, enabled=False):
        disabled = module(inputs)
    assert torch.equal(disabled, native)
    with lora_context(module, sample_mask=torch.tensor([False, True])):
        output = module(inputs)
    assert torch.equal(output[0], native[0])
    assert not torch.equal(output[1], native[1])


def test_lora_context_masks_visual_and_padding_tokens():
    from fabri_moss.lora import FP32LoRALinear, lora_context

    base = torch.nn.Linear(4, 4, bias=False)
    module = FP32LoRALinear(base, rank=1, alpha=1.0, dropout=0.0)
    with torch.no_grad():
        module.lora_A.fill_(1.0)
        module.lora_B.fill_(1.0)
    inputs = torch.ones(1, 3, 4)
    native = base(inputs)
    with lora_context(module, token_mask=torch.tensor([[True, False, True]])):
        output = module(inputs)
    assert torch.equal(output[:, 1], native[:, 1])
    assert not torch.equal(output[:, 0], native[:, 0])


def test_joint_checkpoint_roundtrip_restores_lora_without_base_policy(tmp_path):
    from fabri_moss.core import MossConfig, MossInternVL
    from fabri_moss.tests.test_core import TinyFabriVLAPolicy

    def make_model():
        model = MossInternVL(
            TinyFabriVLAPolicy(hidden_size=128, num_layers=6),
            MossConfig(cross_layers=(2, 4, 6), max_frames=3),
        )
        trainer.configure_lora(model, SimpleNamespace(lora_rank=8, lora_alpha=16.0, lora_dropout=0.1))
        model.set_training_stage("joint")
        trainer.configure_trainable_parameters(model, "joint")
        return model

    args = SimpleNamespace(
        stage="joint", epochs=2, global_batch_size=2, seed=7,
        context_mode="causal", window=3, frame_stride=1,
        min_context_frames=1, decision_stride=None, execution_horizon=5,
        lr=1e-4, action_lr=1e-5, base_lr=5e-6, kd_weight=1.0,
        grad_clip_norm=1.0,
    )
    model = make_model()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    path = tmp_path / "joint.pt"
    contract = {"context_mode": "causal", "data_fingerprint": "x"}
    meta = {"checkpoint_sha256": "base"}
    trainer._save(path, model, optimizer, 3, 1, 0, 0, args, {}, meta, contract, 1, trainer._rng_states(1, "cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "lora_state" in payload and "base_policy" not in payload

    clone = make_model()
    clone_opt = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad], lr=9e-3)
    restored = trainer._load_resume(path, clone, clone_opt, args, meta, contract, 1)
    assert restored == (3, 1, 0, 0)
    for key, value in trainer.lora_state_dict(clone.policy).items():
        assert torch.equal(value, payload["lora_state"][key])
