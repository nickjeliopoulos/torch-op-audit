#!/usr/bin/env python
"""Standalone checks for the queue-based audit API.

Run directly from the repo root:

    python tests/test_api_core.py --device cpu
"""

from __future__ import annotations

import argparse
import os
from queue import Queue
import sys
import traceback

import torch
from torch import nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_arch_op_bench import (  # noqa: E402
    AuditConfig,
    aggregate_audit_events,
    attach_hooks,
    capture_events,
)
from torch_arch_op_bench.classes import (  # noqa: E402
    build_module_to_class,
    build_op_to_class,
    classify,
    classify_module,
)


def _drain(q: Queue) -> list:
    events = []
    while not q.empty():
        events.append(q.get())
    return events


def check_classifier_registry(device: str) -> None:
    op_table = build_op_to_class()
    assert classify("aten::mm.default", op_table) == "tensor_contraction"
    assert classify("aten.addmm.default", op_table) == "tensor_contraction"
    assert classify("aten::native_layer_norm", op_table) == "stat_normalization"
    assert classify("aten::relu", op_table) == "elementwise"
    assert classify("_unknown_", op_table) == "other"

    module_table = build_module_to_class()
    assert classify_module("Linear", module_table) == "tensor_contraction"
    assert classify_module("torch.nn.LayerNorm", module_table) == "stat_normalization"
    assert classify_module("_MysteryModule_", module_table) == "other"


def check_module_queue(device: str) -> None:
    q: Queue = Queue()
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    x = torch.randn(4, 16)
    cfg = AuditConfig(operators=False, modules=True, record_shapes=True, record_dtypes=True)

    with attach_hooks(model, event_queue=q, config=cfg, device=device):
        model(x.to(device))

    events = _drain(q)
    starts = [e for e in events if e.kind == "module" and e.phase == "start"]
    stops = [e for e in events if e.kind == "module" and e.phase == "stop"]
    assert starts and len(starts) == len(stops)
    assert {e.module_type for e in stops} >= {"Linear", "ReLU"}
    assert starts[0].input_shapes == [[4, 16]]
    assert starts[0].input_dtypes == ["torch.float32"]


def check_operator_queue(device: str) -> None:
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x = torch.randn(4, 16)
    cfg = AuditConfig(modules=False, operators=True, record_flops=True)

    events = capture_events(model, [x], config=cfg, device=device)
    stops = [e for e in events if e.kind == "operator" and e.phase == "stop"]
    assert stops
    assert any(e.class_name == "tensor_contraction" for e in stops)
    assert any((e.flops or 0) > 0 for e in stops)
    assert not any(e.kind == "module" for e in events)


def check_module_flops_and_event_report(device: str) -> None:
    model = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 8))
    x = torch.randn(2, 32)
    cfg = AuditConfig(modules=True, operators=True, record_flops=True)

    events = capture_events(model, [x], config=cfg, device=device)
    module_stops = [e for e in events if e.kind == "module" and e.phase == "stop"]
    assert any(e.module_type == "Linear" and (e.flops or 0) > 0 for e in module_stops)

    summary = aggregate_audit_events(events, kind="operator")
    assert int(summary["count"].sum()) > 0
    assert summary["flops"].sum() > 0


def check_unknown_ops(device: str) -> None:
    model = nn.Identity()
    x = torch.randn(4, 4, device=device)

    skipped = capture_events(
        model,
        run=lambda: torch.linalg.vector_norm(x),
        config=AuditConfig(modules=False, operators=True),
        device=device,
    )
    assert not [e for e in skipped if e.kind == "operator"]

    included = capture_events(
        model,
        run=lambda: torch.linalg.vector_norm(x),
        config=AuditConfig(modules=False, operators=True, include_unknown_ops=True),
        device=device,
    )
    stops = [e for e in included if e.kind == "operator" and e.phase == "stop"]
    assert any(e.class_name == "other" for e in stops)


def check_unknown_modules(device: str) -> None:
    class Mystery(nn.Module):
        def forward(self, x):
            return x + 1

    x = torch.randn(2, 2)
    skipped = capture_events(
        Mystery(),
        [x],
        config=AuditConfig(modules=True, operators=False),
        device=device,
    )
    assert not [e for e in skipped if e.kind == "module"]

    included = capture_events(
        Mystery(),
        [x],
        config=AuditConfig(modules=True, operators=False, include_unknown_modules=True),
        device=device,
    )
    stops = [e for e in included if e.kind == "module" and e.phase == "stop"]
    assert len(stops) == 1
    assert stops[0].class_name == "other"


def check_exception_stop_event(device: str) -> None:
    class Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    q: Queue = Queue()
    model = Boom()
    x = torch.randn(2, 2).to(device)
    cfg = AuditConfig(operators=False, modules=True, include_unknown_modules=True)

    try:
        with attach_hooks(model, event_queue=q, config=cfg, device=device):
            model(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    stops = [e for e in _drain(q) if e.kind == "module" and e.phase == "stop"]
    assert len(stops) == 1
    assert "RuntimeError: boom" in (stops[0].error or "")


CHECKS = [
    check_classifier_registry,
    check_module_queue,
    check_operator_queue,
    check_module_flops_and_event_report,
    check_unknown_ops,
    check_unknown_modules,
    check_exception_stop_event,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | xpu")
    args = parser.parse_args(argv)

    failed = 0
    for check in CHECKS:
        try:
            check(args.device)
            print(f"  PASS  {check.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {check.__name__}")
            traceback.print_exc()

    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed on {args.device}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
