"""Bounded delta visual memory functions and state for FabriVLA / MOSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn.functional as F


def delta_update(
    matrix: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    write_logits: torch.Tensor,
) -> torch.Tensor:
    """Update delta visual memory matrix S with a single new frame (parallel average over tokens).

    Args:
        matrix: Old memory tensor S of shape [B, Hkv, d, d].
        keys: Key tensor K of shape [B, Hkv, P, d] (already 3D RoPE applied).
        values: Value tensor V of shape [B, Hkv, P, d].
        write_logits: Learnable parameter b of shape [Hkv] or [1, Hkv, 1, 1].

    Returns:
        New memory tensor S' of shape [B, Hkv, d, d] in FP32 (pure functional, no in-place).
    """
    S = matrix.float()
    K = keys.float()
    V = values.float()
    b = write_logits.float()

    if S.ndim != 4:
        raise ValueError(f"matrix must have 4 dims [B, Hkv, d, d], got {tuple(S.shape)}")
    if K.ndim != 4:
        raise ValueError(f"keys must have 4 dims [B, Hkv, P, d], got {tuple(K.shape)}")
    if V.ndim != 4:
        raise ValueError(f"values must have 4 dims [B, Hkv, P, d], got {tuple(V.shape)}")

    B, Hkv, P, d = K.shape
    if S.shape != (B, Hkv, d, d):
        raise ValueError(f"matrix shape {tuple(S.shape)} does not match expected {(B, Hkv, d, d)}")
    if V.shape != (B, Hkv, P, d):
        raise ValueError(f"values shape {tuple(V.shape)} does not match expected {(B, Hkv, P, d)}")
    if P <= 0:
        raise ValueError(f"P must be > 0, got {P}")

    if b.ndim == 1:
        if b.shape[0] != Hkv:
            raise ValueError(f"write_logits length {b.shape[0]} does not match Hkv {Hkv}")
        rate = torch.sigmoid(b).view(1, Hkv, 1, 1) / float(P)
    elif b.ndim == 4:
        if b.shape != (1, Hkv, 1, 1):
            raise ValueError(f"write_logits 4D shape {tuple(b.shape)} must be (1, {Hkv}, 1, 1)")
        rate = torch.sigmoid(b) / float(P)
    else:
        raise ValueError(f"write_logits must have shape [Hkv] or [1, Hkv, 1, 1], got shape {tuple(b.shape)}")

    K_norm = F.normalize(K, p=2.0, dim=-1)
    pred_V = torch.matmul(K_norm, S)
    E = V - pred_V
    delta = rate * torch.matmul(K_norm.transpose(-2, -1), E)
    return S + delta


def delta_read(
    queries: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Read from delta visual memory S using queries Q.

    Args:
        queries: Language query tensor Q of shape [B, Hq, L, d] (already rotated).
        matrix: Memory tensor S of shape [B, Hkv, d, d].

    Returns:
        Readout tensor R = Q_hat @ S (repeated across GQA groups) of shape [B, Hq, L, d] in FP32.
    """
    Q = queries.float()
    S = matrix.float()

    if Q.ndim != 4:
        raise ValueError(f"queries must have 4 dims [B, Hq, L, d], got {tuple(Q.shape)}")
    if S.ndim != 4:
        raise ValueError(f"matrix must have 4 dims [B, Hkv, d, d], got {tuple(S.shape)}")

    B, Hq, L, d = Q.shape
    Bm, Hkv, dm1, dm2 = S.shape

    if B != Bm:
        raise ValueError(f"Batch size mismatch: queries B={B} vs matrix B={Bm}")
    if d != dm1 or d != dm2:
        raise ValueError(f"Dimension mismatch: query head_dim={d} vs matrix shape {(dm1, dm2)}")
    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({Hkv})")

    num_kv_groups = Hq // Hkv
    if num_kv_groups > 1:
        S_rep = S.repeat_interleave(num_kv_groups, dim=1)
    else:
        S_rep = S

    Q_norm = F.normalize(Q, p=2.0, dim=-1)
    return torch.matmul(Q_norm, S_rep)


@dataclass(frozen=True)
class DeltaMemoryState:
    """Frozen state holding bounded delta visual memory matrices and metadata."""
    matrices: Tuple[torch.Tensor, ...]
    last_frame_id: int
    frame_count: int
    prompt: str
    owner: Any
    revision: int

    def detached(self) -> DeltaMemoryState:
        """Return a new DeltaMemoryState with detached tensors (for TBPTT truncation)."""
        return DeltaMemoryState(
            matrices=tuple(m.detach() for m in self.matrices),
            last_frame_id=self.last_frame_id,
            frame_count=self.frame_count,
            prompt=self.prompt,
            owner=self.owner,
            revision=self.revision,
        )

    @property
    def nbytes(self) -> int:
        """Total memory size of matrices in bytes."""
        return sum(m.nelement() * m.element_size() for m in self.matrices)
