"""Module-phase launch tracing.

Records a time-ordered timeline of ``nn.Module`` forward invocations: for each
module that runs, its qualified name, module type, launch time relative to a
reference ``t0``, exit time, and the shapes of its tensor inputs. The result is
a sequence of nested "phase" spans (akin to NVTX ranges) that can be overlaid on
power / frequency telemetry to see which program phase is active over time.

This is a *forward* timeline captured with ``register_forward_pre_hook`` /
``register_forward_hook``. Timestamps are taken on the CPU dispatch thread
(``time.perf_counter``), so on an async accelerator they mark when each module
*launched* its work, not when the device finished it -- the right, lowest-
overhead signal for phase annotation. Pass ``sync=True`` to insert a device
sync around each span for device-accurate (but far costlier) timing.

Backward-phase tracing is intentionally out of scope here: forward hooks do not
fire during ``backward()``. Capturing backward phases would need
``register_full_backward_hook`` and is left for a later stage.

Events are pushed through a pluggable *sink* so the same instrumentation serves
two callers without changing the benchmarked code:

* standalone profiling -> :class:`BufferSink` collects events in memory and the
  caller writes them to a JSONL file;
* the GEOPM controller (Stage 2) -> when ``TORCH_OP_AUDIT_EVENT_FD`` names an
  inherited pipe write-fd, :class:`JsonlFdSink` streams one JSON line per event
  to the parent process.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .report import get_gpu_name, input_shape_str


### Env var naming an inherited pipe write-fd that phase events stream to. Set by
### the GEOPM controller in Stage 2; unset for standalone profiling.
EVENT_FD_ENV = "TORCH_OP_AUDIT_EVENT_FD"


@dataclass
class PhaseEvent:
    """One ``nn.Module`` forward invocation (a phase span)."""

    name: str                       # qualified module name, e.g. "blocks.0.attn"
    type: str                       # module class name, e.g. "Attention"
    depth: int                      # structural depth (dots in the qualified name)
    call_index: int                 # monotonic invocation counter across the trace
    t_enter: float                  # seconds since t0 when forward began (CPU dispatch)
    t_exit: float                   # seconds since t0 when forward returned
    dur: float                      # t_exit - t_enter
    input_shapes: list[list[int]]   # shapes of tensor inputs, in order
    input_dtypes: list[str]         # matching dtypes (str(dtype))
    iter: int                       # which trace iteration produced this event

    def to_record(self) -> dict[str, Any]:
        rec = asdict(self)
        rec["event"] = "module"
        return rec


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class BufferSink:
    """Collect the meta record and events in memory (standalone path)."""

    def __init__(self) -> None:
        self.meta: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def emit_meta(self, meta: dict[str, Any]) -> None:
        self.meta = meta

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def close(self) -> None:  # noqa: D401 - nothing to release
        pass


class JsonlFdSink:
    """Stream one JSON line per record to a file descriptor (Stage 2 pipe)."""

    def __init__(self, fd: int) -> None:
        # Line-buffered text wrapper; takes ownership of the fd and closes it on
        # close() so the parent's read end sees EOF when the run ends.
        self._f = os.fdopen(fd, "w", buffering=1, closefd=True)

    def emit_meta(self, meta: dict[str, Any]) -> None:
        self._write(meta)

    def emit(self, event: dict[str, Any]) -> None:
        self._write(event)

    def _write(self, obj: dict[str, Any]) -> None:
        try:
            self._f.write(json.dumps(obj) + "\n")
            self._f.flush()
        except (BrokenPipeError, ValueError):
            # Parent closed the read end (or the stream is gone). Drop silently
            # so instrumentation never crashes the benchmarked workload.
            pass

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def _default_sink():
    """Pick a sink from the environment: pipe fd if present, else in-memory."""
    fd = os.environ.get(EVENT_FD_ENV)
    if fd:
        try:
            return JsonlFdSink(int(fd))
        except (ValueError, OSError):
            pass
    return BufferSink()


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class ModulePhaseTracer:
    """Registers forward hooks on every named submodule to record phase spans.

    Use as a context manager around the model invocation(s). Enter/exit pairing
    is per ``id(module)`` so nested and repeated invocations attribute correctly.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        sink,
        t0: float,
        include_types: Sequence[str] | None = None,
        max_depth: int | None = None,
        sync: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.sink = sink
        self.t0 = t0
        self.include_types = set(include_types) if include_types else None
        self.max_depth = max_depth
        self.sync = sync
        self.device = device
        self.iter = 0  # advanced by the driver between iterations
        self._handles: list = []
        self._pending: dict[int, list[PhaseEvent]] = {}
        self._call_index = 0

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        if name == "":  # skip the root module
            return False
        if self.max_depth is not None and name.count(".") > self.max_depth:
            return False
        if self.include_types is not None and type(module).__name__ not in self.include_types:
            return False
        return True

    def _pre_hook(self, name: str):
        def hook(module, args, kwargs):
            if self.sync:
                _sync(self.device)
            ev = PhaseEvent(
                name=name,
                type=type(module).__name__,
                depth=name.count("."),
                call_index=self._call_index,
                t_enter=time.perf_counter() - self.t0,
                t_exit=float("nan"),
                dur=float("nan"),
                input_shapes=_tensor_shapes(args, kwargs),
                input_dtypes=_tensor_dtypes(args, kwargs),
                iter=self.iter,
            )
            self._call_index += 1
            self._pending.setdefault(id(module), []).append(ev)
        return hook

    def _post_hook(self):
        def hook(module, args, output):
            stack = self._pending.get(id(module))
            if not stack:
                return
            if self.sync:
                _sync(self.device)
            ev = stack.pop()
            ev.t_exit = time.perf_counter() - self.t0
            ev.dur = ev.t_exit - ev.t_enter
            self.sink.emit(ev.to_record())
        return hook

    def __enter__(self) -> "ModulePhaseTracer":
        post = self._post_hook()
        for name, module in self.model.named_modules():
            if not self._should_hook(name, module):
                continue
            self._handles.append(
                module.register_forward_pre_hook(self._pre_hook(name), with_kwargs=True)
            )
            self._handles.append(module.register_forward_hook(post))
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def _tensor_shapes(args, kwargs) -> list[list[int]]:
    return [list(v.shape) for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)]


def _tensor_dtypes(args, kwargs) -> list[str]:
    return [str(v.dtype) for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)]


def _sync(device) -> None:
    """Block until queued work on ``device`` finishes (no-op on cpu/None)."""
    if device is not None and torch.device(device).type != "cpu":
        getattr(torch, torch.device(device).type).synchronize()


# ---------------------------------------------------------------------------
# Driver + serialization
# ---------------------------------------------------------------------------

def trace_module_phases(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    iters: int = 1,
    t0: float | None = None,
    sink=None,
    device: str | torch.device = "cuda",
    include_types: Sequence[str] | None = None,
    max_depth: int | None = None,
    sync: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trace the forward-pass module-phase timeline.

    Runs ``model.eval()`` forward under ``no_grad`` for ``iters`` iterations with
    forward hooks on every named submodule, emitting one :class:`PhaseEvent` per
    invocation through ``sink``.

    Parameters
    ----------
    iters:
        Number of forward passes to trace, sharing one ``t0`` so the timeline
        spans all iterations (useful for seeing steady-state repetition).
    t0:
        Reference time (``perf_counter`` seconds) all ``t_enter`` values are
        relative to. Defaults to "now" at the start of the run.
    sink:
        Event sink. Defaults to a :class:`JsonlFdSink` on the fd named by
        ``TORCH_OP_AUDIT_EVENT_FD`` if set, else an in-memory :class:`BufferSink`.
    include_types / max_depth:
        Optional filters: hook only modules whose class name is in
        ``include_types``, and/or whose structural depth is ``<= max_depth``.

    Returns
    -------
    ``(events, meta)``. ``events`` is the list of event records when a
    :class:`BufferSink` is used, or ``[]`` when streaming to a file descriptor.
    """
    device = torch.device(device)
    model = model.to(device)
    example_inputs = tuple(
        x.to(device) if isinstance(x, torch.Tensor) else x for x in example_inputs
    )

    if sink is None:
        sink = _default_sink()
    if t0 is None:
        t0 = time.perf_counter()

    meta = {
        "event": "meta",
        # CLOCK_MONOTONIC anchor for t0: a cross-process-stable reference the
        # GEOPM controller can reconcile against in Stage 2.
        "t0_abs": time.clock_gettime(time.CLOCK_MONOTONIC),
        "model": type(model).__name__,
        "gpu_name": get_gpu_name(device),
        "input_shape": input_shape_str(example_inputs),
        "device": str(device),
        "iters": iters,
        "torch_version": torch.__version__,
        "pid": os.getpid(),
    }
    sink.emit_meta(meta)

    model.eval()
    tracer = ModulePhaseTracer(
        model,
        sink=sink,
        t0=t0,
        include_types=include_types,
        max_depth=max_depth,
        sync=sync,
        device=device,
    )
    with torch.no_grad(), tracer:
        for k in range(iters):
            tracer.iter = k
            model(*example_inputs)

    sink.close()
    return list(getattr(sink, "events", [])), meta


def write_trace_jsonl(
    path: str | Path,
    events: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    """Write a phase trace to a JSONL file: optional ``meta`` line then events."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        if meta is not None:
            f.write(json.dumps(meta) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
