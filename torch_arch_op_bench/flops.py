"""FLOP counting for PyTorch models.

:class:`FlopCounterMode` is used for ops it natively supports (matmul,
conv, SDPA). A companion :class:`_CustomFlopCounter` — our own
``TorchDispatchMode`` subclass — counts ops that ``FlopCounterMode`` has
no formulas for: normalizations, softmax, and pooling.

The two modes are nested so each op is seen by both, but only one of them
has a formula for any given op, preventing double-counting:

    with FlopCounterMode(...) as fc, _CustomFlopCounter() as cfc:
        model(*inputs)

Custom FLOP formulas (generic / paper-convention counts)
---------------------------------------------------------
Batch / layer / group norm (forward):  7 × numel(input)
    mean (N) + variance (2N) + normalise+scale+shift (4N)

Softmax (forward):  5 × numel(input)
    max (N) + subtract (N) + exp (N) + sum (N) + divide (N)
    (numerically-stable Softmax formulation)

Max pooling:  (kH × kW − 1) × numel(output)
    comparisons per output element

Adaptive average pooling:  (kH × kW) × numel(output)
    additions per output element; kernel inferred from spatial in/out ratio
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from .classes import _normalize_op_name


FlopCounts = dict[str, int]


# ---------------------------------------------------------------------------
# Custom formulas  (called as formula(*args, **kwargs) inside __torch_dispatch__)
# ---------------------------------------------------------------------------

def _bn_flops(input: torch.Tensor, *args, **kwargs) -> int:
    """7 FLOPs/element: mean (N) + variance (2N) + normalise+scale+shift (4N)."""
    return 7 * input.numel()


def _softmax_flops(input: torch.Tensor, *args, **kwargs) -> int:
    """5 FLOPs/element: max + subtract + exp + sum + divide."""
    return 5 * input.numel()


def _max_pool2d_flops(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    *args,
    **kwargs,
) -> int:
    """(kH × kW − 1) comparisons per output element."""
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size[:2]
    sh = kh if stride is None else (stride if isinstance(stride, int) else stride[0])
    sw = kw if stride is None else (stride if isinstance(stride, int) else stride[1])
    ph = padding if isinstance(padding, int) else padding[0]
    pw = padding if isinstance(padding, int) else padding[1]
    N, C, H, W = input.shape
    out_h = math.floor((H + 2 * ph - kh) / sh) + 1
    out_w = math.floor((W + 2 * pw - kw) / sw) + 1
    return max(0, kh * kw - 1) * N * C * out_h * out_w


def _matrix_exp_flops(input: torch.Tensor, *args, **kwargs) -> int:
    """Generic Padé-13 estimate: ~13 × N³ FLOPs per [N, N] matrix, times batch.

    Decomposes as roughly 7 matmuls + 1 linear-solve over an N×N matrix
    (~6N³ for the matmuls, ~ ⅔N³ for the solve, plus scaling/squaring).
    13·N³ is a generous round number that hides constant factors but is
    accurate to within an order of magnitude for the realistic N range.
    """
    *batch, n, _ = input.shape
    batch_count = math.prod(batch) if batch else 1
    return 13 * (n ** 3) * batch_count


def _adaptive_avg_pool2d_flops(
    input: torch.Tensor,
    output_size,
    **kwargs,
) -> int:
    """(kH × kW) additions per output element; kernel inferred from spatial ratio."""
    N, C, H, W = input.shape
    out_h = output_size if isinstance(output_size, int) else (output_size[0] or H)
    out_w = output_size if isinstance(output_size, int) else (output_size[1] or W)
    kh = H // out_h if out_h > 0 else 1
    kw = W // out_w if out_w > 0 else 1
    return kh * kw * N * C * out_h * out_w


def _build_custom_formulas() -> dict:
    """Map ATen OpOverload → formula callable for ops FlopCounterMode misses.

    Only terminal (compute-level) ops are registered. Dispatch wrappers
    (``batch_norm``, ``layer_norm``, ``softmax``, ``conv2d``) are
    intentionally omitted to prevent double-counting.
    """
    ops = torch.ops.aten
    formulas: dict = {}

    _try_register = [
        # batch / group norm — terminal ops only
        (ops.cudnn_batch_norm,              "default", _bn_flops),
        (ops.native_batch_norm,             "default", _bn_flops),
        (ops._native_batch_norm_legit,      "default", _bn_flops),
        (ops._native_batch_norm_legit_no_training, "default", _bn_flops),
        (ops.native_group_norm,             "default", _bn_flops),
        # layer norm — native op; "layer_norm" wrapper NOT registered
        (ops.native_layer_norm,             "default", _bn_flops),
        # softmax — internal op; "softmax" wrapper NOT registered
        (ops._softmax,                      "default", _softmax_flops),
        # matrix exponential (Padé approximant; sequence of GEMMs)
        (ops.linalg_matrix_exp,             "default", _matrix_exp_flops),
        # pooling
        (ops.max_pool2d,                    "default", _max_pool2d_flops),
        (ops.max_pool2d_with_indices,       "default", _max_pool2d_flops),
        (ops.adaptive_avg_pool2d,           "default", _adaptive_avg_pool2d_flops),
    ]

    for op_packet, overload_name, formula in _try_register:
        try:
            overload = getattr(op_packet, overload_name)
            formulas[overload] = formula
        except AttributeError:
            pass  # op / overload absent in this torch version

    return formulas


# Eagerly built at import time so the attribute lookups run once.
_CUSTOM_FORMULAS: dict = _build_custom_formulas()


def estimate_flops(func, args=(), kwargs=None, output=None) -> int | None:
    """Best-effort FLOP estimate for one ATen operator invocation.

    This is intentionally per-event and lightweight for live audit hooks. It
    covers common tensor contractions plus the custom formulas above. Unknown
    ops return ``None`` so callers can distinguish "not estimated" from zero.
    """
    kwargs = kwargs or {}

    custom = _CUSTOM_FORMULAS.get(func)
    if custom is not None:
        try:
            return int(custom(*args, **kwargs))
        except Exception:
            return None

    op_name = _normalize_op_name(str(func))
    try:
        if op_name == "mm":
            return _mm_flops(*args, output=output, **kwargs)
        if op_name == "addmm":
            return _addmm_flops(*args, output=output, **kwargs)
        if op_name == "bmm":
            return _bmm_flops(*args, output=output, **kwargs)
        if op_name == "baddbmm":
            flops = _bmm_flops(*args, output=output, **kwargs)
            return None if flops is None else flops + (_numel(output) or 0)
        if op_name == "matmul":
            return _matmul_flops(*args, output=output, **kwargs)
        if op_name == "linear":
            return _linear_flops(*args, output=output, **kwargs)
        if op_name in {"convolution", "_convolution", "conv2d", "cudnn_convolution"}:
            return _conv_flops(op_name, *args, output=output, **kwargs)
    except Exception:
        return None
    return None


def _tensor(value) -> torch.Tensor | None:
    return value if isinstance(value, torch.Tensor) else None


def _numel(value) -> int | None:
    tensor = _tensor(value)
    return int(tensor.numel()) if tensor is not None else None


def _mm_flops(mat1, mat2, *args, output=None, **kwargs) -> int | None:
    mat1 = _tensor(mat1)
    mat2 = _tensor(mat2)
    if mat1 is None or mat2 is None or mat1.dim() < 2 or mat2.dim() < 2:
        return None
    m = int(mat1.shape[-2])
    k = int(mat1.shape[-1])
    n = int(mat2.shape[-1])
    return 2 * m * n * k


def _addmm_flops(input, mat1, mat2, *args, output=None, **kwargs) -> int | None:
    mm = _mm_flops(mat1, mat2, output=output)
    if mm is None:
        return None
    out_numel = _numel(output) or _numel(input) or 0
    return mm + out_numel


def _bmm_flops(batch_or_mat1, mat1_or_mat2, mat2=None, *args, output=None, **kwargs) -> int | None:
    # ``bmm`` args are (mat1, mat2); ``baddbmm`` args are (input, batch1, batch2).
    if mat2 is None:
        mat1 = _tensor(batch_or_mat1)
        mat2_tensor = _tensor(mat1_or_mat2)
    else:
        mat1 = _tensor(mat1_or_mat2)
        mat2_tensor = _tensor(mat2)
    if mat1 is None or mat2_tensor is None or mat1.dim() < 3 or mat2_tensor.dim() < 3:
        return None
    b = int(mat1.shape[-3])
    m = int(mat1.shape[-2])
    k = int(mat1.shape[-1])
    n = int(mat2_tensor.shape[-1])
    return 2 * b * m * n * k


def _matmul_flops(mat1, mat2, *args, output=None, **kwargs) -> int | None:
    mat1 = _tensor(mat1)
    mat2 = _tensor(mat2)
    out = _tensor(output)
    if mat1 is None or mat2 is None or out is None:
        return None
    if mat1.dim() == 1 and mat2.dim() == 1:
        return 2 * int(mat1.shape[0])
    k = int(mat1.shape[-1]) if mat1.dim() > 1 else int(mat1.shape[0])
    return 2 * int(out.numel()) * k


def _linear_flops(input, weight, bias=None, *args, output=None, **kwargs) -> int | None:
    input = _tensor(input)
    weight = _tensor(weight)
    if input is None or weight is None or input.dim() < 1 or weight.dim() < 2:
        return None
    in_features = int(weight.shape[-1])
    out_features = int(weight.shape[0])
    batch = int(input.numel()) // max(in_features, 1)
    flops = 2 * batch * in_features * out_features
    if bias is not None:
        flops += batch * out_features
    return flops


def _conv_flops(op_name, input, weight, *args, output=None, **kwargs) -> int | None:
    input = _tensor(input)
    weight = _tensor(weight)
    out = _tensor(output)
    if input is None or weight is None or out is None or input.dim() < 3 or weight.dim() < 3:
        return None

    groups = kwargs.get("groups")
    if groups is None:
        if op_name == "conv2d" and len(args) >= 5:
            groups = args[4]
        elif len(args) >= 7:
            groups = args[6]
    groups = int(groups or 1)

    kernel_ops = int(weight.shape[1]) * math.prod(int(v) for v in weight.shape[2:])
    return 2 * int(out.numel()) * kernel_ops


# ---------------------------------------------------------------------------
# Custom TorchDispatchMode
# ---------------------------------------------------------------------------

class _CustomFlopCounter(TorchDispatchMode):
    """Counts FLOPs for ops not covered by FlopCounterMode.

    Designed to be nested *inside* ``FlopCounterMode`` so that each op
    passes through both modes.  Only ops present in ``_CUSTOM_FORMULAS``
    produce non-zero counts here; everything else is a transparent pass-
    through.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flop_counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Let the op execute (passes through remaining modes, including
        # FlopCounterMode, then to the actual kernel).
        out = func(*args, **kwargs)
        formula = _CUSTOM_FORMULAS.get(func)
        if formula is not None:
            try:
                n = formula(*args, **kwargs)
                if n:
                    self.flop_counts[str(func)] += n
            except Exception:
                pass
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op_key(op) -> str:
    return str(op)


def _global_counts(mode: FlopCounterMode) -> Counter[str]:
    raw = mode.flop_counts.get("Global", {})
    out: Counter[str] = Counter()
    for op, n in raw.items():
        out[_op_key(op)] += int(n)
    return out


def _merge(*counters: Counter[str]) -> FlopCounts:
    merged: Counter[str] = Counter()
    for c in counters:
        merged.update(c)
    return dict(merged)


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_fwd_flops(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
) -> FlopCounts:
    """FLOPs per ATen op for a single forward pass (no_grad)."""
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc, _CustomFlopCounter() as cfc:
            model(*example_inputs)
    return _merge(_global_counts(fc), cfc.flop_counts)


def count_fwd_bwd_flops(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
) -> tuple[FlopCounts, FlopCounts]:
    """Return ``(fwd_flops, bwd_flops)`` per-op dicts."""
    fwd_only = count_fwd_flops(model, example_inputs)

    model.train()
    grad_inputs = [
        x.detach().clone().requires_grad_(True)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x
        for x in example_inputs
    ]

    with FlopCounterMode(display=False) as fc, _CustomFlopCounter() as cfc:
        out = model(*grad_inputs)
        _scalar_loss(out).backward()
    total = _merge(_global_counts(fc), cfc.flop_counts)

    bwd: FlopCounts = {
        op: n - fwd_only.get(op, 0)
        for op, n in total.items()
        if n - fwd_only.get(op, 0) > 0
    }
    return fwd_only, bwd
