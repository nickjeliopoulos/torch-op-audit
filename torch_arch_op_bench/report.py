"""Aggregate per-op FLOP and latency dicts into per-class pandas tables."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import pandas as pd
import torch

from .classes import CATCH_ALL_CLASS, classify, is_explicitly_registered


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


def aggregate_audit_events(
    events: Iterable[Any],
    *,
    kind: str | None = None,
) -> pd.DataFrame:
    """Aggregate completed audit events into the standard class summary table."""
    rows: dict[str, dict[str, float]] = {}
    for event in events:
        if _event_value(event, "phase") != "stop":
            continue
        if kind is not None and _event_value(event, "kind") != kind:
            continue
        cls = str(_event_value(event, "class_name", CATCH_ALL_CLASS) or CATCH_ALL_CLASS)
        row = rows.setdefault(
            cls,
            {"count": 0, "flops": 0.0, "latency_ms": 0.0},
        )
        row["count"] += 1
        row["flops"] += float(_event_value(event, "flops", 0) or 0)
        row["latency_ms"] += float(_event_value(event, "duration_s", 0.0) or 0.0) * 1_000.0

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


def detailed_audit_events(
    events: Iterable[Any],
    *,
    kind: str | None = None,
) -> pd.DataFrame:
    """Build a per-event audit table from completed stop events."""
    records = []
    for event in events:
        if _event_value(event, "phase") != "stop":
            continue
        if kind is not None and _event_value(event, "kind") != kind:
            continue
        records.append(
            {
                "kind": _event_value(event, "kind"),
                "name": _event_value(event, "name"),
                "class": _event_value(event, "class_name", CATCH_ALL_CLASS),
                "flops": int(_event_value(event, "flops", 0) or 0),
                "latency_ms": float(_event_value(event, "duration_s", 0.0) or 0.0) * 1_000.0,
                "error": _event_value(event, "error"),
            }
        )
    if not records:
        return pd.DataFrame(columns=["kind", "name", "class", "flops", "latency_ms", "error"])
    return pd.DataFrame.from_records(records).sort_values(
        ["kind", "class", "latency_ms"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


_MISSED_COLUMNS = ["op", "normalized_name", "flops", "latency_ms", "%_of_total_latency"]


def missed_ops(
    flops: Mapping[str, int],
    latency_us: Mapping[str, float],
    op_to_class: Mapping[str, str],
) -> pd.DataFrame:
    """Post-mortem report: ops that fell through to the catch-all.

    An op is "missed" if it is not explicitly listed in any class definition
    — i.e. it landed in the catch-all solely because we don't know what it
    is. This table is the primary diagnostic for expanding the registry.

    Rows are sorted by ``latency_ms`` descending so the most impactful
    unknown ops appear first.
    """
    from .classes import _normalize_op_name  # local import to avoid circular

    all_ops = sorted(set(flops) | set(latency_us))
    total_us = sum(latency_us.values()) or 1.0
    records = []
    for op in all_ops:
        if not is_explicitly_registered(op, op_to_class):
            us = float(latency_us.get(op, 0.0))
            records.append(
                {
                    "op": op,
                    "normalized_name": _normalize_op_name(op),
                    "flops": int(flops.get(op, 0)),
                    "latency_ms": us / 1_000.0,
                    "%_of_total_latency": us / total_us * 100.0,
                }
            )

    if not records:
        return pd.DataFrame(columns=_MISSED_COLUMNS)

    df = pd.DataFrame.from_records(records, columns=_MISSED_COLUMNS)
    return df.sort_values("latency_ms", ascending=False).reset_index(drop=True)


@dataclass
class Report:
    """Container holding fwd / bwd summary + detailed DataFrames."""
    fwd_summary: pd.DataFrame | None
    fwd_detailed: pd.DataFrame | None
    bwd_summary: pd.DataFrame | None
    bwd_detailed: pd.DataFrame | None
    fwd_missed: pd.DataFrame | None
    bwd_missed: pd.DataFrame | None
    gpu_name: str = "unknown_gpu"
    input_shape: str = ""
    # Optional module-phase launch trace (populated when benchmark(trace=True)).
    phase_events: list[dict[str, Any]] | None = None
    trace_meta: dict[str, Any] | None = None

    def write(self, out_dir: str | Path, *, latex: bool = True, csv: bool = True) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = _slugify(self.gpu_name)
        shape = _slugify(self.input_shape) if self.input_shape else ""
        prefix = f"{slug}__{shape}" if shape else slug
        for name, df in self._iter_named_frames():
            if df is None:
                continue
            stem = f"{prefix}__{name}"
            if csv:
                df.to_csv(out / f"{stem}.csv")
            if latex:
                (out / f"{stem}.tex").write_text(
                    _to_latex(df, caption=name, gpu_name=self.gpu_name, input_shape=self.input_shape)
                )
        if self.phase_events is not None or self.trace_meta is not None:
            with (out / f"{prefix}__phase_trace.jsonl").open("w") as f:
                if self.trace_meta is not None:
                    f.write(json.dumps(self.trace_meta) + "\n")
                for event in self.phase_events or []:
                    f.write(json.dumps(event) + "\n")

    def _iter_named_frames(self):
        yield "fwd_summary", self.fwd_summary
        yield "fwd_detailed", self.fwd_detailed
        yield "fwd_missed", self.fwd_missed
        yield "bwd_summary", self.bwd_summary
        yield "bwd_detailed", self.bwd_detailed
        yield "bwd_missed", self.bwd_missed


def _slugify(name: str) -> str:
    """Filesystem-safe slug derived from a GPU name."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return safe.strip("_") or "unknown_gpu"


def _to_latex(df: pd.DataFrame, *, caption: str, gpu_name: str, input_shape: str = "") -> str:
    """Render a DataFrame to a booktabs-style LaTeX table."""
    pretty = caption.replace("_", " ")
    annotation = gpu_name
    if input_shape:
        annotation = f"{gpu_name}, {input_shape}"
    slug = _slugify(gpu_name)
    shape_slug = (_slugify(input_shape) + "_") if input_shape else ""
    return df.to_latex(
        float_format="%.2f",
        caption=f"{pretty} ({annotation})",
        label=f"tab:{slug}_{shape_slug}{caption}",
        bold_rows=False,
    )


def input_shape_str(example_inputs: "Sequence[torch.Tensor]") -> str:
    """Human-readable shape annotation, e.g. ``8x3x224x224``.

    Multiple inputs are separated by ``__``:  ``8x3x224x224__64x512``.
    Non-tensor inputs are skipped.
    """

    parts = [
        "x".join(str(d) for d in x.shape)
        for x in example_inputs
        if isinstance(x, torch.Tensor)
    ]
    return "__".join(parts)


def get_gpu_name(device: str | "torch.device") -> str:
    """Resolve the human-readable device name (CUDA, Intel XPU, or CPU).

    Returns the GPU model for ``cuda``/``xpu`` devices and the device type
    (e.g. ``"cpu"``) otherwise.
    """

    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(idx)
    if dev.type == "xpu" and getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        idx = dev.index if dev.index is not None else torch.xpu.current_device()
        return torch.xpu.get_device_name(idx)
    return dev.type
