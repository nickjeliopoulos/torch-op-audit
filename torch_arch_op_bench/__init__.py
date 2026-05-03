"""TorchArchOpBench — per-operator-class FLOP and latency benchmarking."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .classes import build_op_to_class
from .flops import count_fwd_bwd_flops, count_fwd_flops
from .latency import measure_fwd_bwd_latency, measure_fwd_latency
from .report import Report, aggregate, detailed, get_gpu_name


__version__ = "0.0.1"
__all__ = ["benchmark", "Report"]


def benchmark(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    fwd: bool = True,
    bwd: bool = True,
    warmup: int = 10,
    iters: int = 50,
    classes: Mapping[str, list[str]] | None = None,
    device: str | torch.device = "cuda",
) -> Report:
    """Run the per-class FLOP and latency benchmark.

    At least one of ``fwd`` / ``bwd`` must be True. The model and inputs
    are moved to ``device`` (CUDA expected). FLOPs are counted via
    ``FlopCounterMode``; latency via ``torch.profiler``. Backward
    measurements are derived as ``total - fwd``.
    """
    if not (fwd or bwd):
        raise ValueError("benchmark() requires at least one of fwd=True or bwd=True")

    op_to_class = build_op_to_class(classes)
    device = torch.device(device)
    model = model.to(device)
    example_inputs = tuple(
        x.to(device) if isinstance(x, torch.Tensor) else x for x in example_inputs
    )

    fwd_flops: dict[str, int] = {}
    bwd_flops: dict[str, int] = {}
    fwd_lat: dict[str, float] = {}
    bwd_lat: dict[str, float] = {}

    if bwd:
        fwd_flops, bwd_flops = count_fwd_bwd_flops(model, example_inputs)
        fwd_lat, bwd_lat = measure_fwd_bwd_latency(
            model, example_inputs, warmup=warmup, iters=iters
        )
    else:
        fwd_flops = count_fwd_flops(model, example_inputs)
        fwd_lat = measure_fwd_latency(model, example_inputs, warmup=warmup, iters=iters)

    fwd_summary = aggregate(fwd_flops, fwd_lat, op_to_class) if fwd else None
    fwd_detailed_df = detailed(fwd_flops, fwd_lat, op_to_class) if fwd else None
    bwd_summary = aggregate(bwd_flops, bwd_lat, op_to_class) if bwd else None
    bwd_detailed_df = detailed(bwd_flops, bwd_lat, op_to_class) if bwd else None

    return Report(
        fwd_summary=fwd_summary,
        fwd_detailed=fwd_detailed_df,
        bwd_summary=bwd_summary,
        bwd_detailed=bwd_detailed_df,
        gpu_name=get_gpu_name(device),
    )
