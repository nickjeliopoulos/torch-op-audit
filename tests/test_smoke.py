#!/usr/bin/env python
"""Device-agnostic smoke tests for torch_arch_op_bench. No pytest.

Runs every check on the device you pass (default ``cpu``)::

    python tests/test_smoke.py                # cpu
    python tests/test_smoke.py --device xpu   # Aurora Intel GPUs
    python tests/test_smoke.py --device cuda

Each check raises ``AssertionError`` on failure; the runner reports
PASS/FAIL per check and exits non-zero if any failed. FLOP/structure checks
hold on any device; checks that need device-side timing are guarded so they
only assert when timing data was actually captured.
"""

from __future__ import annotations

import argparse
import json
import os
from queue import Queue
import sys
import traceback

import torch
from torch import nn

# Allow running straight from the repo without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_arch_op_bench import (
    AuditConfig,
    aggregate_audit_events,
    attach_hooks,
    benchmark,
    capture_events,
)
from torch_arch_op_bench.classes import (
    build_module_to_class,
    build_op_to_class,
    classify,
    classify_module,
)
from torch_arch_op_bench.trace import trace_module_phases


def _drain(q: Queue) -> list:
    events = []
    while not q.empty():
        events.append(q.get())
    return events


def test_classifier_normalization(device: str) -> None:
    """Operator-class taxonomy lookup (pure CPU, device-independent)."""
    table = build_op_to_class()
    assert classify("aten::mm.default", table) == "tensor_contraction"
    assert classify("aten.addmm.default", table) == "tensor_contraction"
    assert classify("aten::native_layer_norm", table) == "stat_normalization"
    assert classify("aten::relu", table) == "elementwise"
    # Unknown op falls into the catch-all.
    assert classify("_unknown_", table) == "other"

    modules = build_module_to_class()
    assert classify_module("Linear", modules) == "tensor_contraction"
    assert classify_module("torch.nn.LayerNorm", modules) == "stat_normalization"
    assert classify_module("_MysteryModule_", modules) == "other"


def test_audit_module_queue(device: str) -> None:
    """Module hooks emit start/stop records into a normal Python queue."""
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


def test_audit_operator_queue(device: str) -> None:
    """Operator dispatch captures class labels and per-event FLOPs."""
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x = torch.randn(4, 16)
    cfg = AuditConfig(modules=False, operators=True, record_flops=True)

    events = capture_events(model, [x], config=cfg, device=device)
    stops = [e for e in events if e.kind == "operator" and e.phase == "stop"]
    assert stops
    assert any(e.class_name == "tensor_contraction" for e in stops)
    assert any((e.flops or 0) > 0 for e in stops)
    assert not any(e.kind == "module" for e in events)


def test_audit_combined_module_flops_and_report(device: str) -> None:
    """Operator FLOPs roll into active module spans and aggregate into reports."""
    model = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 8))
    x = torch.randn(2, 32)
    cfg = AuditConfig(modules=True, operators=True, record_flops=True)

    events = capture_events(model, [x], config=cfg, device=device)
    module_stops = [e for e in events if e.kind == "module" and e.phase == "stop"]
    assert any(e.module_type == "Linear" and (e.flops or 0) > 0 for e in module_stops)

    summary = aggregate_audit_events(events, kind="operator")
    assert int(summary["count"].sum()) > 0
    assert summary["flops"].sum() > 0


def test_audit_unknown_ops(device: str) -> None:
    """Unknown operators are skipped by default and included as other on request."""
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


def test_audit_unknown_modules(device: str) -> None:
    """Unknown modules are skipped by default and optionally emitted as other."""
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


def test_audit_exception_stop_event(device: str) -> None:
    """Open module spans are stopped with error metadata when forward raises."""
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


def test_module_phase_trace(device: str) -> None:
    """Module-phase launch trace: names, shapes, ordering, iterations."""
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    x = torch.randn(4, 16)

    events, meta = trace_module_phases(model, [x], iters=2, device=device)

    assert meta["event"] == "meta" and meta["iters"] == 2
    assert "t0_abs" in meta
    assert events, "expected at least one phase event"

    names = {e["name"] for e in events}
    assert {"0", "1", "2"} <= names
    assert {e["iter"] for e in events} == {0, 1}

    for e in events:
        assert e["event"] == "module"
        assert e["t_exit"] >= e["t_enter"]
        assert e["dur"] >= 0.0

    lin0 = next(e for e in events if e["name"] == "0")
    assert lin0["type"] == "Linear"
    assert lin0["input_shapes"] == [[4, 16]]


def test_phase_trace_streams_to_pipe_fd(device: str) -> None:
    """When TORCH_OP_AUDIT_EVENT_FD is set, events stream as JSONL down the fd."""
    r_fd, w_fd = os.pipe()
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    x = torch.randn(2, 8)

    os.environ["TORCH_OP_AUDIT_EVENT_FD"] = str(w_fd)
    try:
        events, _ = trace_module_phases(model, [x], iters=1, device=device)
    finally:
        os.environ.pop("TORCH_OP_AUDIT_EVENT_FD", None)

    # Streamed to the fd -> the in-memory buffer is empty.
    assert events == []

    with os.fdopen(r_fd) as rf:
        lines = [json.loads(line) for line in rf if line.strip()]

    assert lines[0]["event"] == "meta"
    assert any(line["event"] == "module" for line in lines)


def test_benchmark_fwd_only(device: str) -> None:
    """Forward FLOP/latency benchmark on a Linear-heavy model."""
    model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
    x = torch.randn(64, 512)

    report = benchmark(model, [x], fwd=True, bwd=False, warmup=2, iters=5, device=device)
    assert report.fwd_summary is not None
    assert report.bwd_summary is None

    df = report.fwd_summary
    assert "tensor_contraction" in df.index
    # FLOPs are analytical (device-independent): matmuls dominate.
    assert df.loc["tensor_contraction", "%_flop"] > 99.0
    # Runtime split only holds when device timing was actually captured.
    if df["latency_ms"].sum() > 0:
        assert abs(df["%_runtime"].sum() - 100.0) < 1e-6


def test_benchmark_fwd_bwd(device: str) -> None:
    """Forward+backward benchmark; backward FLOPs are still matmul-dominated."""
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    x = torch.randn(32, 256)

    report = benchmark(model, [x], fwd=True, bwd=True, warmup=2, iters=5, device=device)
    assert report.fwd_summary is not None
    assert report.bwd_summary is not None
    assert report.bwd_summary.loc["tensor_contraction", "flops"] > 0


def test_trace_via_benchmark(device: str) -> None:
    """benchmark(trace=True) attaches a phase trace + meta to the report."""
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
    x = torch.randn(8, 64)

    report = benchmark(
        model, [x], fwd=True, bwd=False, warmup=1, iters=2,
        device=device, trace=True, trace_iters=2,
    )
    assert report.phase_events, "expected phase events on the report"
    assert report.trace_meta is not None and report.trace_meta["event"] == "meta"
    assert {e["name"] for e in report.phase_events} >= {"0", "1", "2"}


def test_timm_audit_events(device: str) -> None:
    """TIMM models can be audited directly without config orchestration."""
    import timm

    cfg = AuditConfig(
        modules=True,
        operators=True,
        record_flops=True,
        module_max_depth=2,
        op_include_names=[
            "addmm",
            "mm",
            "matmul",
            "linear",
            "convolution",
            "conv2d",
            "native_layer_norm",
            "gelu",
            "relu",
        ],
    )
    for model_name in ["vit_small_patch16_224", "convnext_small"]:
        model = timm.create_model(model_name, pretrained=False).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            events = capture_events(model, [x], config=cfg, device=device)
        assert any(e.kind == "module" and e.phase == "stop" for e in events), model_name
        assert any(e.kind == "operator" and e.phase == "stop" for e in events), model_name
        assert int(aggregate_audit_events(events)["count"].sum()) > 0


TESTS = [
    test_classifier_normalization,
    test_audit_module_queue,
    test_audit_operator_queue,
    test_audit_combined_module_flops_and_report,
    test_audit_unknown_ops,
    test_audit_unknown_modules,
    test_audit_exception_stop_event,
    test_module_phase_trace,
    test_phase_trace_streams_to_pipe_fd,
    test_benchmark_fwd_only,
    test_benchmark_fwd_bwd,
    test_trace_via_benchmark,
    test_timm_audit_events,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | xpu (default: cpu)")
    args = parser.parse_args(argv)
    device = args.device

    failed = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn(device)
            print(f"  PASS  {name}")
        except Exception:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()

    total = len(TESTS)
    print(f"\n{total - failed}/{total} passed on {device}")
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    # torch nightlies on the oneAPI/level_zero stack can SIGSEGV during
    # interpreter teardown; flush and hard-exit so the real result is the
    # process exit code (and no core dump is produced on shutdown).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
