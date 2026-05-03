"""Forward / backward ATen graph capture via AOT Autograd.

Runs ``aot_module_simplified`` with stash-only fwd/bwd compilers so we
get the post-decomposition ``fx.GraphModule`` for the forward pass and,
when requested, the joint backward graph as well. The captured graphs
are the source of truth for op classification and FLOP counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import fx, nn
from torch._functorch.aot_autograd import aot_module_simplified


@dataclass
class CapturedGraphs:
    """Decomposed ATen graphs for one ``(model, input-shape)`` capture."""

    fwd: fx.GraphModule | None
    bwd: fx.GraphModule | None


def capture(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    capture_bwd: bool = True,
    device: str | torch.device = "cuda",
) -> CapturedGraphs:
    """Capture the decomposed forward (and optionally backward) ATen graphs.

    Parameters
    ----------
    model:
        The ``nn.Module`` to trace. Moved to ``device`` and switched into
        ``train()`` mode when ``capture_bwd`` is True (else ``eval()``).
    example_inputs:
        Positional tensor arguments matching the model's forward signature.
        Floating tensors are cloned with ``requires_grad_(True)`` when
        capturing backward so AOT Autograd produces a joint graph.
    capture_bwd:
        If True, run a synthetic ``loss.backward()`` to trigger backward
        graph compilation. If False, only the forward graph is captured.
    device:
        CUDA device the model and inputs are placed on. CPU is supported
        for testing but the rest of the pipeline assumes CUDA.

    Returns
    -------
    :class:`CapturedGraphs` with the forward graph and (optionally) the
    backward graph. ``bwd`` is ``None`` when ``capture_bwd`` is False or
    the model has no learnable parameters and no input requires grad.
    """
    device = torch.device(device)
    model = model.to(device)
    model = model.train() if capture_bwd else model.eval()

    prepared: list[torch.Tensor] = []
    for x in example_inputs:
        if not isinstance(x, torch.Tensor):
            prepared.append(x)
            continue
        x = x.to(device)
        if capture_bwd and x.is_floating_point():
            x = x.detach().clone().requires_grad_(True)
        prepared.append(x)

    captured: dict[str, fx.GraphModule | None] = {"fwd": None, "bwd": None}

    def fw_compiler(gm: fx.GraphModule, _inputs):
        captured["fwd"] = gm
        return gm

    def bw_compiler(gm: fx.GraphModule, _inputs):
        captured["bwd"] = gm
        return gm

    compiled = aot_module_simplified(
        model,
        prepared,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )

    if capture_bwd:
        out = compiled(*prepared)
        loss = _scalar_loss(out)
        loss.backward()
    else:
        with torch.no_grad():
            compiled(*prepared)

    return CapturedGraphs(fwd=captured["fwd"], bwd=captured["bwd"])


def iter_aten_nodes(gm: fx.GraphModule):
    """Yield ``call_function`` nodes whose target is an ATen op.

    Skips placeholders, outputs, ``get_attr``, and any non-ATen targets.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        # OpOverload exposes its qualified name via ``_schema.name`` or ``__qualname__``.
        if _is_aten_target(target):
            yield node


def _is_aten_target(target) -> bool:
    mod = getattr(target, "__module__", "") or ""
    name = getattr(target, "__qualname__", "") or ""
    if mod.startswith("torch._ops") or mod.startswith("torch.ops"):
        return True
    if hasattr(target, "_schema"):
        schema_name = getattr(target._schema, "name", "")
        return schema_name.startswith("aten::")
    return name.startswith("aten.")


def op_name(target) -> str:
    """Return a stable string name for an ATen op target.

    Prefers the schema name (e.g. ``aten::mm.default``); falls back to
    ``__qualname__``.
    """
    if hasattr(target, "_schema"):
        schema = target._schema
        overload = getattr(target, "_overloadname", None) or "default"
        return f"{schema.name}.{overload}"
    return getattr(target, "__qualname__", str(target))


def _scalar_loss(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out.float().sum()
    if isinstance(out, (list, tuple)):
        parts = [_scalar_loss(o) for o in out if isinstance(o, torch.Tensor)]
        if not parts:
            raise TypeError("Model output sequence contains no tensors to derive loss from")
        return sum(parts)
    if isinstance(out, dict):
        parts = [_scalar_loss(v) for v in out.values() if isinstance(v, torch.Tensor)]
        if not parts:
            raise TypeError("Model output dict contains no tensors to derive loss from")
        return sum(parts)
    raise TypeError(f"Cannot derive scalar loss from output of type {type(out)!r}")
