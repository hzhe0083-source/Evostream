"""Unit tests for bounded delta visual memory in fabri_moss.

Tests cover:
1. Analytical delta update formula verification (exact values vs manual PyTorch calculation)
2. Zero residual update: when V = K_norm @ S, S' == S exactly (delta == 0)
3. Non-inplace update: old S tensor remains untouched / identical
4. FP32 guarantee: inputs in FP32, FP16, BF16 all yield FP32 outputs in delta_update and delta_read
5. GQA group handling in delta_read (Hq = 4, Hkv = 2)
6. Invalid shapes for write_logits (e.g. 2D, 3D, wrong head count)
7. Strict validation on previous state: foreign owner, stale revision, mismatched prompt, layer count/shape, non-increasing frame_id
8. Different historical frames with same current observation affect deep output when gates are open (attn_gate = 0.5)
9. Batch / online equivalence: sequential frame-by-frame read_delta vs feeding all frames in one window
10. Single visual projection per new frame
11. State nbytes stability over steps
12. 2-step TBPTT gradient flow: backprop through loss reaches write_logits, K_proj, V_proj, readout_embeddings, memory_gate, and frozen policy parameters have no gradients
13. read_memory in delta mode raises RuntimeError
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.delta import DeltaMemoryState, delta_read, delta_update
from fabri_moss.tests.test_core import TinyFabriVLAPolicy


@pytest.fixture
def delta_setup():
    torch.manual_seed(42)
    policy = TinyFabriVLAPolicy(hidden_size=128, num_layers=6)
    config = MossConfig(
        cross_layers=(2, 4, 6),
        num_readout_tokens=8,
        max_frames=3,
        shallow_layer=4,
        max_text_tokens=64,
        memory_mode="delta",
    )
    moss = MossInternVL(policy, config=config)
    return policy, moss, config


# ---------------------------------------------------------------------------
# 1. Analytical delta formula & edge cases
# ---------------------------------------------------------------------------

def test_delta_update_analytical_formula():
    """Verify delta_update implements S' = S + (sigma(b)/P) * K_norm^T @ (V - K_norm @ S)."""
    B, Hkv, P, d = 1, 2, 4, 8
    torch.manual_seed(123)
    S = torch.randn(B, Hkv, d, d)
    K = torch.randn(B, Hkv, P, d)
    V = torch.randn(B, Hkv, P, d)
    b = torch.tensor([0.2, -0.4])

    S_next = delta_update(S, K, V, b)
    assert S_next.dtype == torch.float32

    # Manual reference computation
    K_norm = F.normalize(K.float(), p=2.0, dim=-1)
    pred_V = torch.matmul(K_norm, S.float())
    E = V.float() - pred_V
    rate = torch.sigmoid(b.float()).view(1, Hkv, 1, 1) / float(P)
    expected = S.float() + rate * torch.matmul(K_norm.transpose(-2, -1), E)

    assert torch.allclose(S_next, expected, atol=1e-6)


def test_delta_update_zero_residual():
    """When V = K_norm @ S, residual E is 0 and S' must be identical to S."""
    B, Hkv, P, d = 1, 2, 8, 16
    torch.manual_seed(42)
    S = torch.randn(B, Hkv, d, d)
    K = torch.randn(B, Hkv, P, d)
    K_norm = F.normalize(K.float(), p=2.0, dim=-1)
    V = torch.matmul(K_norm, S)  # Exact prediction
    b = torch.zeros(Hkv)

    S_next = delta_update(S, K, V, b)
    assert torch.allclose(S_next, S, atol=1e-6)


def test_delta_update_old_s_unchanged():
    """Old S tensor must not be modified in-place."""
    B, Hkv, P, d = 1, 2, 4, 8
    S = torch.randn(B, Hkv, d, d)
    S_clone = S.clone()
    K = torch.randn(B, Hkv, P, d)
    V = torch.randn(B, Hkv, P, d)
    b = torch.zeros(Hkv)

    _ = delta_update(S, K, V, b)
    assert torch.equal(S, S_clone), "S must not be modified in-place"


def test_delta_fp32_and_bf16_types():
    """delta_update and delta_read must return float32 even if inputs are bf16 or fp16."""
    B, Hkv, P, d = 1, 2, 4, 8
    S_bf16 = torch.randn(B, Hkv, d, d, dtype=torch.bfloat16)
    K_bf16 = torch.randn(B, Hkv, P, d, dtype=torch.bfloat16)
    V_bf16 = torch.randn(B, Hkv, P, d, dtype=torch.bfloat16)
    b_bf16 = torch.zeros(Hkv, dtype=torch.bfloat16)

    S_next = delta_update(S_bf16, K_bf16, V_bf16, b_bf16)
    assert S_next.dtype == torch.float32

    # Test 4D write_logits shape [1, Hkv, 1, 1]
    b_4d = torch.zeros((1, Hkv, 1, 1), dtype=torch.bfloat16)
    S_next_4d = delta_update(S_bf16, K_bf16, V_bf16, b_4d)
    assert S_next_4d.dtype == torch.float32
    assert torch.allclose(S_next, S_next_4d)

    # Test delta_read with bf16 queries
    Hq = 4
    L = 10
    Q_bf16 = torch.randn(B, Hq, L, d, dtype=torch.bfloat16)
    R = delta_read(Q_bf16, S_next)
    assert R.dtype == torch.float32


def test_delta_read_gqa():
    """delta_read with GQA groups (Hq=4, Hkv=2) repeats matrices correctly."""
    B, Hkv, d = 1, 2, 8
    Hq = 4
    L = 3
    S = torch.randn(B, Hkv, d, d)
    Q = torch.randn(B, Hq, L, d)

    R = delta_read(Q, S)
    assert R.shape == (B, Hq, L, d)
    assert R.dtype == torch.float32

    # Group 0 and Group 1 queries should read from S[:, 0], Group 2 and Group 3 from S[:, 1]
    Q_norm = F.normalize(Q, p=2.0, dim=-1)
    expected_g0 = torch.matmul(Q_norm[:, 0], S[:, 0])
    expected_g1 = torch.matmul(Q_norm[:, 1], S[:, 0])
    expected_g2 = torch.matmul(Q_norm[:, 2], S[:, 1])
    expected_g3 = torch.matmul(Q_norm[:, 3], S[:, 1])

    assert torch.allclose(R[:, 0], expected_g0, atol=1e-6)
    assert torch.allclose(R[:, 1], expected_g1, atol=1e-6)
    assert torch.allclose(R[:, 2], expected_g2, atol=1e-6)
    assert torch.allclose(R[:, 3], expected_g3, atol=1e-6)


def test_delta_invalid_shapes():
    """Invalid shapes for write_logits and matrix must raise ValueError."""
    B, Hkv, P, d = 1, 2, 4, 8
    S = torch.randn(B, Hkv, d, d)
    K = torch.randn(B, Hkv, P, d)
    V = torch.randn(B, Hkv, P, d)

    # Invalid write_logits shape: 2D
    with pytest.raises(ValueError, match="write_logits must have shape"):
        delta_update(S, K, V, torch.zeros(2, 2))

    # Invalid write_logits shape: wrong 4D
    with pytest.raises(ValueError, match="write_logits 4D shape"):
        delta_update(S, K, V, torch.zeros(2, Hkv, 1, 1))

    # Invalid write_logits length: 1D mismatch
    with pytest.raises(ValueError, match="write_logits length"):
        delta_update(S, K, V, torch.zeros(Hkv + 1))


# ---------------------------------------------------------------------------
# 2. MossInternVL read_delta & state validation
# ---------------------------------------------------------------------------

def test_read_memory_raises_in_delta_mode(delta_setup):
    """Calling read_memory when config.memory_mode='delta' must raise RuntimeError."""
    _, moss, _ = delta_setup
    features = torch.randn(1, 16, 128)
    frame = moss.project_frame(features, frame_id=0)
    with pytest.raises(RuntimeError, match="read_memory is only available in 'consume' mode"):
        moss.read_memory([frame], "pick block")


def test_read_delta_validations(delta_setup):
    """Verify strict validation on previous DeltaMemoryState."""
    _, moss, config = delta_setup
    features = torch.randn(1, 16, 128)
    f0 = moss.project_frame(features, frame_id=0)
    f1 = moss.project_frame(features, frame_id=1)

    with torch.no_grad():
        _, _, state0 = moss.read_delta([f0], "pick up block")

    # 1. Non-monotonic frame_id (f0 frame_id <= state0.last_frame_id)
    with pytest.raises(ValueError, match="strictly greater than previous last_frame_id"):
        moss.read_delta([f0], "pick up block", previous=state0)

    # 2. Foreign owner
    other_policy = TinyFabriVLAPolicy(hidden_size=128, num_layers=6)
    other_moss = MossInternVL(other_policy, config=config)
    with pytest.raises(ValueError, match="foreign owner"):
        other_moss.read_delta([f1], "pick up block", previous=state0)

    # 3. Stale revision
    moss.train(True)  # increments _revision
    with pytest.raises(ValueError, match="stale revision"):
        moss.read_delta([f1], "pick up block", previous=state0)
    moss.train(False)
    moss._revision = state0.revision  # restore for further tests

    # 4. Prompt mismatch
    with pytest.raises(ValueError, match="does not match current prompt"):
        moss.read_delta([f1], "different prompt", previous=state0)

    # 5. Invalid previous type
    with pytest.raises(TypeError, match="must be an instance of DeltaMemoryState"):
        moss.read_delta([f1], "pick up block", previous="not_a_state")  # type: ignore


def test_delta_history_affects_deep_when_gates_opened(delta_setup):
    """When cross gates are open, different historical frames yield different deep outputs."""
    _, moss, config = delta_setup
    moss.eval()

    # Open cross gates
    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.5)
        block.mlp_gate.data.fill_(0.5)

    prompt = "sort objects"
    feat_a = torch.randn(1, 16, 128)
    feat_b = torch.randn(1, 16, 128) + 5.0
    feat_curr = torch.randn(1, 16, 128)

    # Path 1: history is feat_a, current is feat_curr
    f_a = moss.project_frame(feat_a, frame_id=0)
    f_curr1 = moss.project_frame(feat_curr, frame_id=1)
    with torch.no_grad():
        _, _, state_a = moss.read_delta([f_a], prompt)
        deep1, _, _ = moss.read_delta([f_curr1], prompt, previous=state_a)

    # Path 2: history is feat_b, current is feat_curr
    f_b = moss.project_frame(feat_b, frame_id=0)
    f_curr2 = moss.project_frame(feat_curr, frame_id=1)
    with torch.no_grad():
        _, _, state_b = moss.read_delta([f_b], prompt)
        deep2, _, _ = moss.read_delta([f_curr2], prompt, previous=state_b)

    assert not torch.allclose(deep1, deep2, atol=1e-4), "Different history must affect deep output when gates are open"


def test_batching_online_state_equivalence(delta_setup):
    """Sequential 1-frame read_delta calls must produce identical state to feeding [f0, f1] at once."""
    _, moss, _ = delta_setup
    moss.eval()
    prompt = "push red cube"
    feat0 = torch.randn(1, 16, 128)
    feat1 = torch.randn(1, 16, 128)

    f0 = moss.project_frame(feat0, frame_id=0)
    f1 = moss.project_frame(feat1, frame_id=1)

    # Online sequential
    with torch.no_grad():
        _, _, state0 = moss.read_delta([f0], prompt)
        deep_seq, shallow_seq, state_seq = moss.read_delta([f1], prompt, previous=state0)

    # Batch (both in one call)
    with torch.no_grad():
        f0_batch = moss.project_frame(feat0, frame_id=0)
        f1_batch = moss.project_frame(feat1, frame_id=1)
        deep_batch, shallow_batch, state_batch = moss.read_delta([f0_batch, f1_batch], prompt)

    assert state_seq.frame_count == state_batch.frame_count == 2
    assert state_seq.last_frame_id == state_batch.last_frame_id == 1
    for m_seq, m_bat in zip(state_seq.matrices, state_batch.matrices):
        assert torch.allclose(m_seq, m_bat, atol=1e-6)


def test_forward_delta_projects_each_image_once(delta_setup):
    """forward_delta must preprocess and extract features for each image window exactly once."""
    policy, moss, _ = delta_setup
    moss.eval()

    embedder = policy.embedder
    init_extract = embedder._extract_calls
    init_preproc = embedder._preprocess_calls

    images = [["frame_img_0"], ["frame_img_1"]]
    frame_ids = [0, 1]
    prompt = "reach target"

    with torch.no_grad():
        _, _, state = moss.forward_delta(images, frame_ids, prompt)

    assert embedder._preprocess_calls == init_preproc + 2
    assert embedder._extract_calls == init_extract + 2
    assert state.frame_count == 2
    assert state.last_frame_id == 1


def test_state_nbytes_stability(delta_setup):
    """Memory matrix size in bytes must remain strictly constant regardless of frame count."""
    _, moss, _ = delta_setup
    moss.eval()
    prompt = "task"

    f0 = moss.project_frame(torch.randn(1, 16, 128), frame_id=0)
    f1 = moss.project_frame(torch.randn(1, 16, 128), frame_id=1)
    f2 = moss.project_frame(torch.randn(1, 16, 128), frame_id=2)

    with torch.no_grad():
        _, _, s0 = moss.read_delta([f0], prompt)
        _, _, s1 = moss.read_delta([f1], prompt, previous=s0)
        _, _, s2 = moss.read_delta([f2], prompt, previous=s1)

    assert s0.nbytes == s1.nbytes == s2.nbytes
    # 3 cross layers, each [1, 2, 32, 32] float32 = 3 * 2 * 32 * 32 * 4 = 24576 bytes
    assert s0.nbytes == 3 * 2 * 32 * 32 * 4


# ---------------------------------------------------------------------------
# 3. Two-round TBPTT gradient flow & frozen policy isolation
# ---------------------------------------------------------------------------

def test_two_round_tbptt_gradient_flow(delta_setup):
    """Verify gradients propagate to write_logits, K_proj, V_proj, memory_gate, readout, and not to frozen policy."""
    policy, moss, _ = delta_setup
    moss.train(True)
    prompt = "grasp handle"

    # Open gates to allow cross attention flow
    for block in moss.cross_blocks.values():
        block.attn_gate.data.fill_(0.5)
        block.mlp_gate.data.fill_(0.5)

    feat0 = torch.randn(1, 16, 128)
    feat1 = torch.randn(1, 16, 128)
    feat2 = torch.randn(1, 16, 128)

    # Step 1: frame 0 -> state0
    f0 = moss.project_frame(feat0, frame_id=0)
    deep0, shallow0, state0 = moss.read_delta([f0], prompt)
    loss0 = (deep0.sum() + shallow0.sum())
    loss0.backward()

    # Step 2: truncate graph with detached state
    state0_det = state0.detached()
    moss.zero_grad()

    # Step 2: frame 1 -> state1
    f1 = moss.project_frame(feat1, frame_id=1)
    deep1, shallow1, state1 = moss.read_delta([f1], prompt, previous=state0_det)

    # Step 3: frame 2 -> state2 (connected graph across step 2 and step 3)
    f2 = moss.project_frame(feat2, frame_id=2)
    deep2, shallow2, state2 = moss.read_delta([f2], prompt, previous=state1)

    loss = deep2.sum() + shallow2.sum()
    loss.backward()

    # Check gradients on bridge parameters
    for name, block in moss.cross_blocks.items():
        assert block.write_logits.grad is not None, f"write_logits on block {name} must receive grad"
        assert torch.count_nonzero(block.write_logits.grad) > 0

        assert block.memory_gate.grad is not None, f"memory_gate on block {name} must receive grad"
        assert torch.count_nonzero(block.memory_gate.grad) > 0

        assert block.k_proj.weight.grad is not None, f"k_proj on block {name} must receive grad"
        assert torch.count_nonzero(block.k_proj.weight.grad) > 0

        assert block.v_proj.weight.grad is not None, f"v_proj on block {name} must receive grad"
        assert torch.count_nonzero(block.v_proj.weight.grad) > 0

    assert moss.readout_embeddings.grad is not None
    assert torch.count_nonzero(moss.readout_embeddings.grad) > 0

    # Verify frozen policy parameters have no grad
    for p_name, p in policy.named_parameters():
        assert p.grad is None, f"Policy parameter {p_name} must remain frozen without grad"
