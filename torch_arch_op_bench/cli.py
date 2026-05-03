"""Command-line entry point.

Loads a YAML config via OmegaConf, instantiates the model from a dotted
import path, builds pinned-shape inputs, and runs :func:`benchmark`.

Example
-------
::

    python -m torch_arch_op_bench.cli --config configs/example.yaml --fwd --bwd
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from . import benchmark


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def _resolve_dotted(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected dotted import path, got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


def _build_model(cfg: DictConfig) -> torch.nn.Module:
    factory = _resolve_dotted(cfg.model["import"])
    kwargs = OmegaConf.to_container(cfg.model.get("kwargs", {}), resolve=True) or {}
    return factory(**kwargs)


def _build_inputs(cfg: DictConfig) -> list[torch.Tensor]:
    dtype = _DTYPES[cfg.input.get("dtype", "float32")]
    shapes = cfg.input.shape
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], (list, tuple)):
        return [torch.randn(*list(s), dtype=dtype) for s in shapes]
    return [torch.randn(*list(shapes), dtype=dtype)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fwd", action="store_true", help="run forward benchmark")
    parser.add_argument("--bwd", action="store_true", help="run backward benchmark")
    parser.add_argument("--out", type=Path, default=None, help="override output dir")
    args = parser.parse_args(argv)

    cfg: DictConfig = OmegaConf.load(args.config)  # type: ignore[assignment]

    fwd = args.fwd or bool(cfg.benchmark.get("fwd", False))
    bwd = args.bwd or bool(cfg.benchmark.get("bwd", False))
    if not (fwd or bwd):
        parser.error("specify --fwd and/or --bwd (or set them in the config)")

    model = _build_model(cfg)
    inputs = _build_inputs(cfg)

    classes_cfg = cfg.get("classes", None)
    classes = OmegaConf.to_container(classes_cfg, resolve=True) if classes_cfg else None

    report = benchmark(
        model,
        inputs,
        fwd=fwd,
        bwd=bwd,
        warmup=int(cfg.benchmark.get("warmup", 10)),
        iters=int(cfg.benchmark.get("iters", 50)),
        classes=classes,  # type: ignore[arg-type]
        device="cuda",
    )

    out_dir = args.out or Path(cfg.output.get("dir", "./results"))
    report.write(
        out_dir,
        latex=bool(cfg.output.get("latex", True)),
        csv=bool(cfg.output.get("csv", True)),
    )

    print(f"GPU: {report.gpu_name}")
    if report.fwd_summary is not None:
        print(f"=== forward ({report.gpu_name}) ===")
        print(report.fwd_summary.to_string())
    if report.bwd_summary is not None:
        print(f"=== backward ({report.gpu_name}) ===")
        print(report.bwd_summary.to_string())
    print(f"\nWrote tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
