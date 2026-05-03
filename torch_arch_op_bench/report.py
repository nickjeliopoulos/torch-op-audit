"""Aggregate per-op FLOP and latency dicts into per-class pandas tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from .classes import CATCH_ALL_CLASS, classify


_SUMMARY_COLUMNS = ["count", "flops", "latency_ms", "%_flop", "%_runtime"]


def aggregate(
    flops: Mapping[str, int],
    latency_us: Mapping[str, float],
    op_to_class: Mapping[str, str],
) -> pd.DataFrame:
    """Build a per-class summary DataFrame from per-op FLOP/latency dicts.

    ``flops`` and ``latency_us`` may have disjoint key sets — every op key
    appearing in either dict is classified and aggregated; missing values
    are treated as 0.
    """
    all_ops = set(flops) | set(latency_us)
    rows: dict[str, dict[str, float]] = {}
    for op in all_ops:
        cls = classify(op, op_to_class)
        row = rows.setdefault(
            cls,
            {"count": 0, "flops": 0.0, "latency_ms": 0.0},
        )
        row["count"] += 1
        row["flops"] += float(flops.get(op, 0))
        row["latency_ms"] += float(latency_us.get(op, 0.0)) / 1_000.0

    # Ensure the catch-all class always appears so coverage is provably 100%.
    rows.setdefault(
        CATCH_ALL_CLASS,
        {"count": 0, "flops": 0.0, "latency_ms": 0.0},
    )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "class"

    total_flops = df["flops"].sum()
    total_latency = df["latency_ms"].sum()
    df["%_flop"] = (df["flops"] / total_flops * 100.0) if total_flops > 0 else 0.0
    df["%_runtime"] = (df["latency_ms"] / total_latency * 100.0) if total_latency > 0 else 0.0
    df["count"] = df["count"].astype(int)
    return df[_SUMMARY_COLUMNS].sort_values("%_runtime", ascending=False)


def detailed(
    flops: Mapping[str, int],
    latency_us: Mapping[str, float],
    op_to_class: Mapping[str, str],
) -> pd.DataFrame:
    """Per-op breakdown: one row per ATen op, with its assigned class."""
    all_ops = sorted(set(flops) | set(latency_us))
    records = []
    for op in all_ops:
        records.append(
            {
                "op": op,
                "class": classify(op, op_to_class),
                "flops": int(flops.get(op, 0)),
                "latency_ms": float(latency_us.get(op, 0.0)) / 1_000.0,
            }
        )
    df = pd.DataFrame.from_records(records)
    return df.sort_values(["class", "latency_ms"], ascending=[True, False]).reset_index(drop=True)


@dataclass
class Report:
    """Container holding fwd / bwd summary + detailed DataFrames."""

    fwd_summary: pd.DataFrame | None
    fwd_detailed: pd.DataFrame | None
    bwd_summary: pd.DataFrame | None
    bwd_detailed: pd.DataFrame | None
    gpu_name: str = "unknown_gpu"

    def write(self, out_dir: str | Path, *, latex: bool = True, csv: bool = True) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = _slugify(self.gpu_name)
        for name, df in self._iter_named_frames():
            if df is None:
                continue
            stem = f"{slug}__{name}"
            if csv:
                df.to_csv(out / f"{stem}.csv")
            if latex:
                (out / f"{stem}.tex").write_text(
                    _to_latex(df, caption=name, gpu_name=self.gpu_name)
                )

    def _iter_named_frames(self):
        yield "fwd_summary", self.fwd_summary
        yield "fwd_detailed", self.fwd_detailed
        yield "bwd_summary", self.bwd_summary
        yield "bwd_detailed", self.bwd_detailed


def _slugify(name: str) -> str:
    """Filesystem-safe slug derived from a GPU name."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return safe.strip("_") or "unknown_gpu"


def _to_latex(df: pd.DataFrame, *, caption: str, gpu_name: str) -> str:
    """Render a DataFrame to a booktabs-style LaTeX table."""
    pretty = caption.replace("_", " ")
    return df.to_latex(
        float_format="%.2f",
        caption=f"{pretty} ({gpu_name})",
        label=f"tab:{_slugify(gpu_name)}_{caption}",
        bold_rows=False,
    )


def get_gpu_name(device: str | "torch.device") -> str:
    """Resolve the human-readable GPU name for a CUDA device string/object.

    Returns ``"cpu"`` for non-CUDA devices and ``"unknown_gpu"`` if CUDA
    is selected but unavailable.
    """
    import torch

    dev = torch.device(device)
    if dev.type != "cuda":
        return dev.type
    if not torch.cuda.is_available():
        return "unknown_gpu"
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_name(idx)
