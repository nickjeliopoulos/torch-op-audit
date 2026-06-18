#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from torch_arch_op_bench import (  # noqa: E402
    AuditConfig,
    aggregate_audit_events,
    capture_events,
)
from torch_arch_op_bench.report import get_gpu_name, input_shape_str  # noqa: E402


DEFAULT_MODELS = [
    "vit_small_patch16_224",
    "convnext_small",
]

DEFAULT_OP_INCLUDE_NAMES = [
    "addmm",
    "mm",
    "matmul",
    "linear",
    "convolution",
    "conv2d",
    "native_layer_norm",
    "gelu",
    "relu",
]


def check_timm_model(
    model_name: str,
    device: str,
    *,
    config: AuditConfig,
    max_events: int,
) -> None:
    import timm

    model = timm.create_model(model_name, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        events = capture_events(model, [x], config=config, device=device)

    module_events = [e for e in events if e.kind == "module" and e.phase == "stop"]
    operator_events = [e for e in events if e.kind == "operator" and e.phase == "stop"]
    if config.modules:
        assert module_events, model_name
    else:
        assert not module_events, model_name
    if config.operators:
        assert operator_events, model_name
    else:
        assert not operator_events, model_name
    if config.record_flops and config.operators:
        assert any((e.flops or 0) > 0 for e in operator_events), model_name
    has_events = bool(events)
    if config.record_shapes and has_events:
        assert any(e.input_shapes is not None for e in events), model_name
    else:
        assert not any(e.input_shapes is not None for e in events), model_name
    if config.record_dtypes and has_events:
        assert any(e.input_dtypes is not None for e in events), model_name
    else:
        assert not any(e.input_dtypes is not None for e in events), model_name

    operator_summary = aggregate_audit_events(events, kind="operator")
    module_summary = aggregate_audit_events(events, kind="module")
    if config.operators:
        assert int(operator_summary["count"].sum()) > 0
    if config.modules:
        assert int(module_summary["count"].sum()) > 0

    _print_report(
        model_name=model_name,
        device=device,
        inputs=[x],
        config=config,
        events=events,
        operator_summary=operator_summary,
        module_summary=module_summary,
        max_events=max_events,
    )


def _print_report(
    *,
    model_name: str,
    device: str,
    inputs: list[torch.Tensor],
    config: AuditConfig,
    events,
    operator_summary,
    module_summary,
    max_events: int,
) -> None:
    gpu_name = get_gpu_name(device)
    input_shape = input_shape_str(inputs)
    op_events = sum(1 for e in events if e.kind == "operator" and e.phase == "stop")
    module_events = sum(1 for e in events if e.kind == "module" and e.phase == "stop")

    print(f"\nGPU: {gpu_name}  |  input: {input_shape}  |  model: {model_name}")
    print(f"AuditConfig: {config}")
    print(f"Captured {op_events} operator events and {module_events} module events")
    print(f"\n=== operator audit ({gpu_name}, {input_shape}) ===")
    print(operator_summary.to_string())
    print(f"\n=== module audit ({gpu_name}, {input_shape}) ===")
    print(module_summary.to_string())
    print("\n=== queue contents ===")
    _print_queue_contents(events, max_events=max_events)


def _print_queue_contents(events, *, max_events: int) -> None:
    shown = events if max_events <= 0 else events[:max_events]
    for idx, event in enumerate(shown):
        duration = "" if event.duration_s is None else f"{event.duration_s * 1_000:.3f}ms"
        flops = "" if event.flops is None else str(event.flops)
        input_shapes = "" if event.input_shapes is None else event.input_shapes
        name = event.name
        if len(name) > 72:
            name = name[:69] + "..."
        print(
            f"{idx:04d} "
            f"{event.phase:<5} "
            f"{event.kind:<8} "
            f"{event.class_name:<19} "
            f"t={event.t_start:.6f} "
            f"dur={duration:<10} "
            f"flops={flops:<14} "
            f"shape={input_shapes} "
            f"name={name}"
        )
    if max_events > 0 and len(events) > max_events:
        print(f"... {len(events) - max_events} more event(s) omitted; pass --max-events 0 to show all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | xpu")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--modules", action="store_true", help="Enable nn.Module hooks")
    parser.add_argument("--operators", action="store_true", help="Enable ATen operator hooks")
    parser.add_argument("--record-flops", action="store_true", help="Estimate FLOPs on stop events")
    parser.add_argument("--record-shapes", action="store_true", help="Record input tensor shapes")
    parser.add_argument("--record-dtypes", action="store_true", help="Record input tensor dtypes")
    parser.add_argument("--include-unknown-ops", action="store_true", help="Emit unclassified ops as other")
    parser.add_argument(
        "--include-unknown-modules",
        action="store_true",
        help="Emit unclassified modules as other",
    )
    parser.add_argument("--sync", action="store_true", help="Synchronize device around events")
    parser.add_argument(
        "--module-max-depth",
        type=int,
        default=2,
        help="Maximum module depth to audit; use -1 for no limit",
    )
    parser.add_argument(
        "--module-include-types",
        nargs="*",
        default=None,
        help="Optional module type allow-list, e.g. Linear LayerNorm Attention",
    )
    parser.add_argument(
        "--op-include-names",
        nargs="*",
        default=DEFAULT_OP_INCLUDE_NAMES,
        help="Optional operator allow-list; pass the flag with no values to audit all known ops",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=120,
        help="Maximum queue events to print per model; use 0 to print all",
    )
    args = parser.parse_args(argv)
    config = AuditConfig(
        modules=args.modules,
        operators=args.operators,
        record_flops=args.record_flops,
        record_shapes=args.record_shapes,
        record_dtypes=args.record_dtypes,
        include_unknown_ops=args.include_unknown_ops,
        include_unknown_modules=args.include_unknown_modules,
        module_max_depth=None if args.module_max_depth < 0 else args.module_max_depth,
        module_include_types=args.module_include_types or None,
        op_include_names=args.op_include_names or None,
        sync=args.sync,
    )

    failed = 0
    for model_name in args.models:
        try:
            check_timm_model(
                model_name,
                args.device,
                config=config,
                max_events=args.max_events,
            )
            print(f"  PASS  {model_name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {model_name}")
            traceback.print_exc()

    print(f"\n{len(args.models) - failed}/{len(args.models)} TIMM models passed on {args.device}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
