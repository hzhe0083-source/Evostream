"""Small dependency-free FP32 LoRA modules used by the MOSS trainer."""

from __future__ import annotations

import math
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Dict, Iterable, Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_LORA_CONTEXT: ContextVar[Tuple[bool, Tensor | None]] = ContextVar(
    "fabri_moss_lora_context", default=(True, None)
)


class FP32LoRALinear(nn.Module):
    """A frozen linear projection with a trainable FP32 low-rank residual.

    The wrapped projection keeps its original dtype (usually BF16).  LoRA
    weights and their optimizer state are always FP32, so an optimizer never
    updates a BF16 base tensor.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"LoRA target must expose in_features/out_features, got {type(base).__name__}")
        if isinstance(rank, bool) or int(rank) != rank or int(rank) <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank!r}")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError(f"alpha must be finite and positive, got {alpha!r}")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(self.dropout_p)

        weight = getattr(base, "weight", None)
        if weight is None:
            raise TypeError("LoRA target must expose a weight tensor")
        device = weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, int(base.in_features), device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(int(base.out_features), self.rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        # Native attention projections normally receive only ``input``.  Keep
        # arbitrary extra arguments for custom Linear-compatible modules.
        base_input = input
        base_weight = getattr(self.base, "weight", None)
        if base_weight is not None and base_input.dtype != base_weight.dtype:
            base_input = base_input.to(dtype=base_weight.dtype)
        base_out = self.base(base_input, *args, **kwargs)
        lora_enabled, sample_mask = _LORA_CONTEXT.get()
        if not lora_enabled:
            return base_out
        residual = F.linear(self.dropout(input.float()), self.lora_A)
        residual = F.linear(residual, self.lora_B).mul(self.scaling)
        if sample_mask is not None:
            sample_mask = sample_mask.to(device=residual.device, dtype=residual.dtype).flatten()
            if input.ndim == 0 or input.shape[0] != sample_mask.numel():
                raise ValueError(
                    "LoRA sample mask must match the first input dimension: "
                    f"input={tuple(input.shape)}, mask={tuple(sample_mask.shape)}"
                )
            residual = residual * sample_mask.view(-1, *([1] * (residual.ndim - 1)))
        return base_out + residual.to(dtype=base_out.dtype, device=base_out.device)

    def _apply(self, fn):
        # ``Module.to(dtype=...)`` is commonly called on the whole policy;
        # preserve the FP32 optimizer tensors when that happens.
        super()._apply(fn)
        self.lora_A.data = self.lora_A.data.float()
        self.lora_B.data = self.lora_B.data.float()
        return self

    def lora_state(self) -> Dict[str, Tensor]:
        return {"lora_A": self.lora_A.detach().cpu(), "lora_B": self.lora_B.detach().cpu()}


def _resolve_parent(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    if not parts or any(not p for p in parts):
        raise ValueError(f"invalid module path {dotted_name!r}")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise AttributeError(f"module path {dotted_name!r} missing {part!r}")
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    root: nn.Module,
    module_paths: Iterable[str],
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
) -> Dict[str, FP32LoRALinear]:
    """Replace each named Linear-like module with :class:`FP32LoRALinear`.

    Calling this twice is idempotent when the existing wrapper has the same
    specification; a changed specification is rejected to prevent mixed runs.
    """
    result: Dict[str, FP32LoRALinear] = {}
    for path in module_paths:
        parent, leaf = _resolve_parent(root, path)
        current = getattr(parent, leaf)
        if isinstance(current, FP32LoRALinear):
            if (current.rank, current.alpha, current.dropout_p) != (int(rank), float(alpha), float(dropout)):
                raise ValueError(f"LoRA specification mismatch for {path}")
            result[path] = current
            continue
        wrapped = FP32LoRALinear(current, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, leaf, wrapped)
        result[path] = wrapped
    return result


def lora_modules(root: nn.Module) -> Dict[str, FP32LoRALinear]:
    """Return all wrapped projections keyed by their module path."""
    found: Dict[str, FP32LoRALinear] = {}
    for name, module in root.named_modules():
        if isinstance(module, FP32LoRALinear):
            found[name] = module
    return found


def lora_state_dict(root: nn.Module) -> Dict[str, Tensor]:
    """Flatten only trainable LoRA tensors; frozen base weights are omitted."""
    state: Dict[str, Tensor] = {}
    for name, module in lora_modules(root).items():
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(root: nn.Module, state: Mapping[str, Tensor], *, strict: bool = True) -> None:
    modules = lora_modules(root)
    expected = {f"{name}.{suffix}" for name in modules for suffix in ("lora_A", "lora_B")}
    incoming = set(state)
    missing, unexpected = expected - incoming, incoming - expected
    if strict and (missing or unexpected):
        raise ValueError(f"LoRA state keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, module in modules.items():
        for suffix, parameter in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            key = f"{name}.{suffix}"
            if key not in state:
                continue
            value = torch.as_tensor(state[key])
            if value.shape != parameter.shape:
                raise ValueError(f"LoRA shape mismatch for {key}: {tuple(value.shape)} != {tuple(parameter.shape)}")
            if value.dtype != torch.float32:
                raise TypeError(f"LoRA checkpoint tensor {key} must be float32, got {value.dtype}")
            with torch.no_grad():
                parameter.copy_(value.to(device=parameter.device))


def lora_spec(root: nn.Module) -> Dict[str, object]:
    modules = lora_modules(root)
    specs = {(m.rank, m.alpha, m.dropout_p) for m in modules.values()}
    if len(specs) > 1:
        raise ValueError("inconsistent LoRA specifications")
    rank, alpha, dropout = next(iter(specs), (0, 0.0, 0.0))
    return {
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "modules": sorted(modules),
        "parameter_dtype": "float32",
    }


def set_lora_train_mode(root: nn.Module, training: bool = True) -> None:
    for module in lora_modules(root).values():
        # Keep the frozen native projection in eval mode; only the residual's
        # dropout needs a training flag.
        module.dropout.train(training)


@contextmanager
def lora_context(
    root: nn.Module,
    *,
    enabled: bool = True,
    sample_mask: Tensor | None = None,
):
    """Temporarily enable LoRA, optionally per batch sample.

    Native current-only queries use ``enabled=False`` to preserve FabriVLA
    exactly.  Mixed batches pass a boolean ``sample_mask`` so only samples
    with historical FrameKV receive the residual.
    """
    if sample_mask is not None:
        if sample_mask.ndim != 1 or sample_mask.dtype != torch.bool:
            raise ValueError("sample_mask must be a one-dimensional bool tensor")
        sample_mask = sample_mask.detach()
    token = _LORA_CONTEXT.set((bool(enabled), sample_mask))
    try:
        yield
    finally:
        _LORA_CONTEXT.reset(token)
