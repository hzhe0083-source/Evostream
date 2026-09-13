from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import fabri_moss.train_moss_ddp as trainer


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
