"""TorchArchOpBench — per-operator-class FLOP and latency benchmarking."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .classes import build_op_to_class
from .flops import count_fwd_bwd_flops, count_fwd_flops
from .latency import measure_fwd_bwd_latency, measure_fwd_latency
from .report import Report, aggregate, detailed, missed_ops, get_gpu_name, input_shape_str
from .trace import PhaseEvent, trace_module_phases, write_trace_jsonl


__version__ = "0.1"
__all__ = ["benchmark", "Report", "PhaseEvent", "trace_module_phases", "write_trace_jsonl"]


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
    trace: bool = False,
    trace_iters: int = 1,
    trace_include_types: Sequence[str] | None = None,
    trace_max_depth: int | None = None,
) -> Report:
    """Run the per-class FLOP and latency benchmark.

    At least one of ``fwd`` / ``bwd`` must be True. The model and inputs
    are moved to ``device`` (CUDA expected). FLOPs are counted via
    ``FlopCounterMode``; latency via ``torch.profiler``. Backward
    measurements are derived as ``total - fwd``.

    When ``trace`` is True, an additional forward-pass module-phase timeline is
    captured (see :func:`trace_module_phases`) and attached to the returned
    report; ``trace_iters`` forward passes are recorded into the trace.
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
            model, example_inputs, warmup=warmup, iters=iters, device=device
        )
    else:
        fwd_flops = count_fwd_flops(model, example_inputs)
        fwd_lat = measure_fwd_latency(
            model, example_inputs, warmup=warmup, iters=iters, device=device
        )

    fwd_summary = aggregate(fwd_flops, fwd_lat, op_to_class) if fwd else None
    fwd_detailed_df = detailed(fwd_flops, fwd_lat, op_to_class) if fwd else None
    fwd_missed_df = missed_ops(fwd_flops, fwd_lat, op_to_class) if fwd else None
    bwd_summary = aggregate(bwd_flops, bwd_lat, op_to_class) if bwd else None
    bwd_detailed_df = detailed(bwd_flops, bwd_lat, op_to_class) if bwd else None
    bwd_missed_df = missed_ops(bwd_flops, bwd_lat, op_to_class) if bwd else None

    phase_events: list | None = None
    trace_meta: dict | None = None
    if trace:
        # Separate, clean forward pass so hook overhead never pollutes the
        # FLOP/latency tables above. Forward-only timeline (see trace module).
        phase_events, trace_meta = trace_module_phases(
            model,
            example_inputs,
            iters=trace_iters,
            device=device,
            include_types=trace_include_types,
            max_depth=trace_max_depth,
        )

    return Report(
        fwd_summary=fwd_summary,
        fwd_detailed=fwd_detailed_df,
        fwd_missed=fwd_missed_df,
        bwd_summary=bwd_summary,
        bwd_detailed=bwd_detailed_df,
        bwd_missed=bwd_missed_df,
        gpu_name=get_gpu_name(device),
        input_shape=input_shape_str(example_inputs),
        phase_events=phase_events,
        trace_meta=trace_meta,
    )
