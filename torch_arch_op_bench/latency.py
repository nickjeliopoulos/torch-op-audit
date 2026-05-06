"""CUDA latency attribution via :mod:`torch.profiler`.

Two passes are run, mirroring :mod:`flops`:

1. ``no_grad`` forward-only profile → forward kernel time per op.
2. Full forward+backward profile → total kernel time per op; backward
   time per op is ``total - fwd`` (clamped to non-negative).

ATen-only filtering
-------------------
``key_averages()`` returns events at every level of the call stack: ATen
ops, the underlying CUDA kernels (cuBLAS GEMMs, cuDNN convs, cutlass
attention kernels, ``vectorized_elementwise_kernel<...>`` templates,
etc.), memory transfers (``Memcpy DtoD``), and profiler internals
(``Command Buffer Full``).  PyTorch attributes the same ``self_cuda_time``
to BOTH the leaf-ATen op AND the launched kernel, so summing all events
double-counts every operation.

We filter to ``aten::*`` events only.  This:
  * eliminates the double-counting (each op's CUDA time appears once)
  * keeps the post-mortem useful — only ATen ops missing from the registry
    appear there, not opaque kernel signatures
  * is safe in our context (PyTorch model forward/backward), where every
    CUDA kernel is launched by an ATen op anyway
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile


LatencyCounts = dict[str, float]  # microseconds, per op key


def _is_aten_event(key: str) -> bool:
    """True for ATen-level profiler events (e.g. ``aten::mm``).

    Filters out CUDA kernel names, memory transfers, and profiler internals.
    """
    return key.startswith("aten::") or key.startswith("aten.")


def _sum_self_cuda_us(prof: profile) -> Counter[str]:
    """Sum self-CUDA time per op, keeping only ATen-level events."""
    out: Counter[str] = Counter()
    for ev in prof.key_averages():
        if not _is_aten_event(ev.key):
            continue
        # self_cuda_time_total is in microseconds (renamed to self_device_time_total
        # in newer torch; fall back if needed).
        us = getattr(ev, "self_cuda_time_total", None)
        if us is None:
            us = getattr(ev, "self_device_time_total", 0.0)
        if us:
            out[ev.key] += float(us)
    return out


def _warmup_and_sync(model, inputs, *, iters: int, backward: bool) -> None:
    for _ in range(iters):
        if backward:
            out = model(*inputs)
            _scalar_loss(out).backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            for x in inputs:
                if isinstance(x, torch.Tensor) and x.grad is not None:
                    x.grad = None
        else:
            with torch.no_grad():
                model(*inputs)
    torch.cuda.synchronize()


def measure_fwd_latency(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    warmup: int = 10,
    iters: int = 50,
) -> LatencyCounts:
    """Forward-only kernel time per ATen op, summed over ``iters`` runs."""
    model.eval()
    _warmup_and_sync(model, example_inputs, iters=warmup, backward=False)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            with torch.no_grad():
                model(*example_inputs)
        torch.cuda.synchronize()
    return dict(_sum_self_cuda_us(prof))


def measure_fwd_bwd_latency(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    warmup: int = 10,
    iters: int = 50,
) -> tuple[LatencyCounts, LatencyCounts]:
    """Return ``(fwd_us, bwd_us)`` per-op CUDA self-time dicts.

    Forward is measured under ``no_grad``; backward time per op is
    ``profile(fwd+bwd) - profile(fwd_only)``, clamped at zero.
    """
    fwd_only = measure_fwd_latency(model, example_inputs, warmup=warmup, iters=iters)

    model.train()
    grad_inputs: list[torch.Tensor] = []
    for x in example_inputs:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            grad_inputs.append(x.detach().clone().requires_grad_(True))
        else:
            grad_inputs.append(x)

    _warmup_and_sync(model, grad_inputs, iters=warmup, backward=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            out = model(*grad_inputs)
            _scalar_loss(out).backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            for x in grad_inputs:
                if isinstance(x, torch.Tensor) and x.grad is not None:
                    x.grad = None
        torch.cuda.synchronize()
    total = _sum_self_cuda_us(prof)

    bwd: LatencyCounts = {}
    for op, us in total.items():
        delta = us - fwd_only.get(op, 0.0)
        if delta > 0:
            bwd[op] = delta
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
