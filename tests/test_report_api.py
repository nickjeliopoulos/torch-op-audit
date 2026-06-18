#!/usr/bin/env python
"""Standalone checks for report-generation APIs.

Run directly from the repo root:

    python tests/test_report_api.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from torch import nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_arch_op_bench import (  # noqa: E402
    AuditConfig,
    aggregate_audit_events,
    benchmark,
    capture_events,
    detailed_audit_events,
)


def check_benchmark_report_write(device: str) -> None:
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    x = torch.randn(8, 128)

    report = benchmark(model, [x], fwd=True, bwd=False, warmup=1, iters=2, device=device)
    assert report.fwd_summary is not None
    assert report.bwd_summary is None
    assert "tensor_contraction" in report.fwd_summary.index

    with tempfile.TemporaryDirectory() as tmp:
        report.write(tmp, latex=False, csv=True)
        written = list(Path(tmp).glob("*fwd_summary.csv"))
    assert written


def check_event_report_tables(device: str) -> None:
    model = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 8))
    x = torch.randn(2, 32)
    events = capture_events(
        model,
        [x],
        config=AuditConfig(modules=True, operators=True, record_flops=True),
        device=device,
    )

    summary = aggregate_audit_events(events, kind="operator")
    detailed = detailed_audit_events(events, kind="operator")
    assert int(summary["count"].sum()) > 0
    assert summary["flops"].sum() > 0
    assert not detailed.empty
    assert {"kind", "name", "class", "flops", "latency_ms", "error"} <= set(detailed.columns)


CHECKS = [
    check_benchmark_report_write,
    check_event_report_tables,
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
