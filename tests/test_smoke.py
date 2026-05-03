"""Smoke tests. Require CUDA; skipped otherwise."""

from __future__ import annotations

import pytest
import torch
from torch import nn

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
def test_mlp_fwd_only():
    from torch_arch_op_bench import benchmark

    model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).cuda()
    x = torch.randn(64, 512, device="cuda")

    report = benchmark(model, [x], fwd=True, bwd=False, warmup=2, iters=5)
    assert report.fwd_summary is not None
    assert report.bwd_summary is None

    df = report.fwd_summary
    assert "tensor_contraction" in df.index
    # Linear-only model: matmuls should dominate FLOPs.
    assert df.loc["tensor_contraction", "%_flop"] > 99.0
    # Coverage is exactly 100% by construction (catch-all class).
    assert abs(df["%_flop"].sum() - 100.0) < 1e-6 or df["%_flop"].sum() == 0
    assert abs(df["%_runtime"].sum() - 100.0) < 1e-6


@cuda_only
def test_mlp_fwd_bwd():
    from torch_arch_op_bench import benchmark

    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)).cuda()
    x = torch.randn(32, 256, device="cuda")

    report = benchmark(model, [x], fwd=True, bwd=True, warmup=2, iters=5)
    assert report.fwd_summary is not None
    assert report.bwd_summary is not None
    # Backward of a Linear-heavy model should still be dominated by matmul.
    assert report.bwd_summary.loc["tensor_contraction", "flops"] > 0


def test_classifier_normalization():
    from torch_arch_op_bench.classes import build_op_to_class, classify

    table = build_op_to_class()
    assert classify("aten::mm.default", table) == "tensor_contraction"
    assert classify("aten.addmm.default", table) == "tensor_contraction"
    assert classify("aten::native_layer_norm", table) == "stat_normalization"
    assert classify("aten::relu", table) == "elementwise"
    # Unknown op falls into the catch-all.
    assert classify("aten::some_op_we_dont_know", table) == "elementwise"
