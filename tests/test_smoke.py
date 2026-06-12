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
import sys
import traceback

import torch
from torch import nn

# Allow running straight from the repo without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_arch_op_bench import benchmark  # noqa: E402
from torch_arch_op_bench.classes import build_op_to_class, classify  # noqa: E402
from torch_arch_op_bench.trace import trace_module_phases  # noqa: E402


def test_classifier_normalization(device: str) -> None:
    """Operator-class taxonomy lookup (pure CPU, device-independent)."""
    table = build_op_to_class()
    assert classify("aten::mm.default", table) == "tensor_contraction"
    assert classify("aten.addmm.default", table) == "tensor_contraction"
    assert classify("aten::native_layer_norm", table) == "stat_normalization"
    assert classify("aten::relu", table) == "elementwise"
    # Unknown op falls into the catch-all.
    assert classify("aten::some_op_we_dont_know", table) == "elementwise"


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


TESTS = [
    test_classifier_normalization,
    test_module_phase_trace,
    test_phase_trace_streams_to_pipe_fd,
    test_benchmark_fwd_only,
    test_benchmark_fwd_bwd,
    test_trace_via_benchmark,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | xpu (default: cpu)")
    args = parser.parse_args(argv)

    device = args.device
    print(f"torch {torch.__version__} | running smoke tests on device: {device}\n")

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
