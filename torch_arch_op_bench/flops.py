"""FLOP counting via :class:`torch.utils.flop_counter.FlopCounterMode`.

Two flavors are exposed:

- :func:`count_fwd_flops` — runs the model under ``torch.no_grad()`` and
  returns total FLOPs per ATen op for the forward pass only.
- :func:`count_fwd_bwd_flops` — runs the full forward + backward step and
  returns separate forward-only and backward-only per-op FLOP dicts.

Backward FLOPs are derived as ``total - fwd``: a fwd+bwd run is performed
under FlopCounterMode, then a fwd-only ``no_grad`` run is subtracted. Any
op that the FlopCounter has no formula for is silently treated as 0 FLOPs
(typical for views and most pointwise ops, which dominate runtime but not
arithmetic).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


# A per-op FLOP map keyed on the stable string name (``aten::mm.default``-ish).
FlopCounts = dict[str, int]


def _op_key(op) -> str:
    """Stable string key for an ATen op (works for OpOverload or str)."""
    return str(op)


def _global_counts(mode: FlopCounterMode) -> Counter[str]:
    """Pull the global per-op FLOP counter out of FlopCounterMode."""
    raw = mode.flop_counts.get("Global", {})
    out: Counter[str] = Counter()
    for op, n in raw.items():
        out[_op_key(op)] += int(n)
    return out


def count_fwd_flops(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
) -> FlopCounts:
    """FLOPs per ATen op for a single forward pass under ``no_grad``."""
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(*example_inputs)
    return dict(_global_counts(fc))


def count_fwd_bwd_flops(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
) -> tuple[FlopCounts, FlopCounts]:
    """Return ``(fwd_flops, bwd_flops)`` per-op dicts.

    The model is left in ``train()`` mode. Inputs that are floating tensors
    are cloned with ``requires_grad_(True)`` so backward sees gradients.
    """
    fwd_only = count_fwd_flops(model, example_inputs)

    model.train()
    grad_inputs: list[torch.Tensor] = []
    for x in example_inputs:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            grad_inputs.append(x.detach().clone().requires_grad_(True))
        else:
            grad_inputs.append(x)

    with FlopCounterMode(display=False) as fc:
        out = model(*grad_inputs)
        _scalar_loss(out).backward()
    total = _global_counts(fc)

    bwd: FlopCounts = {}
    for op, n_total in total.items():
        n_bwd = n_total - fwd_only.get(op, 0)
        if n_bwd > 0:
            bwd[op] = n_bwd
    return fwd_only, bwd


def _scalar_loss(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out.float().sum()
    if isinstance(out, (list, tuple)):
        parts = [_scalar_loss(o) for o in out if isinstance(o, torch.Tensor)]
        if not parts:
            raise TypeError("Model output sequence has no tensors for loss")
        return sum(parts)
    if isinstance(out, dict):
        parts = [_scalar_loss(v) for v in out.values() if isinstance(v, torch.Tensor)]
        if not parts:
            raise TypeError("Model output dict has no tensors for loss")
        return sum(parts)
    raise TypeError(f"Cannot derive scalar loss from output of type {type(out)!r}")
