"""Live queue-based instrumentation for PyTorch models."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from queue import Queue
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from .classes import (
    build_module_to_class,
    build_op_to_class,
    classify,
    classify_module,
    is_explicitly_registered,
    is_module_explicitly_registered,
    _normalize_module_type,
    _normalize_op_name,
)
from .flops import estimate_flops


@dataclass
class AuditEvent:
    """One live audit record emitted by module hooks or operator dispatch."""

    kind: str
    phase: str
    name: str
    class_name: str
    call_id: int
    t_start: float
    t_stop: float | None = None
    duration_s: float | None = None
    flops: int | None = None
    module_name: str | None = None
    module_type: str | None = None
    op_name: str | None = None
    depth: int | None = None
    input_shapes: list[list[int]] | None = None
    input_dtypes: list[str] | None = None
    error: str | None = None
    event: str = "audit"

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditConfig:
    """Controls what the live audit hooks record."""

    operators: bool = True
    modules: bool = True
    record_flops: bool = True
    record_shapes: bool = False
    record_dtypes: bool = False
    include_unknown_ops: bool = False
    include_unknown_modules: bool = False
    module_max_depth: int | None = None
    module_include_types: Sequence[str] | None = None
    op_include_names: Sequence[str] | None = None
    sync: bool = False
    op_classes: Mapping[str, Iterable[str]] | None = None
    module_classes: Mapping[str, Iterable[str]] | None = None


@dataclass
class _ActiveModule:
    call_id: int
    name: str
    module_type: str
    class_name: str
    depth: int
    t_start: float
    flops: int | None


class AuditHookContext:
    """Context manager returned by :func:`attach_hooks`."""

    def __init__(
        self,
        model: nn.Module,
        *,
        event_queue,
        config: AuditConfig | None = None,
        device: str | torch.device | None = None,
        t0: float | None = None,
    ) -> None:
        self.model = model
        self.event_queue = event_queue
        self.config = config or AuditConfig()
        self.device = torch.device(device) if device is not None else None
        self.t0 = t0

        self.op_to_class = build_op_to_class(self.config.op_classes)
        self.module_to_class = build_module_to_class(self.config.module_classes)
        self.module_include_types = (
            {_normalize_module_type(t) for t in self.config.module_include_types}
            if self.config.module_include_types
            else None
        )
        self.op_include_names = (
            {_normalize_op_name(name) for name in self.config.op_include_names}
            if self.config.op_include_names
            else None
        )

        self._handles: list[Any] = []
        self._pending_modules: dict[int, list[_ActiveModule]] = {}
        self._active_modules: list[_ActiveModule] = []
        self._op_mode: _AuditDispatchMode | None = None
        self._call_id = 0

    def __enter__(self) -> "AuditHookContext":
        if self.t0 is None:
            self.t0 = time.perf_counter()
        if self.device is not None:
            self.model.to(self.device)
        if self.config.modules:
            self._attach_module_hooks()
        if self.config.operators:
            self._op_mode = _AuditDispatchMode(self)
            self._op_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._op_mode is not None:
            self._op_mode.__exit__(exc_type, exc, tb)
            self._op_mode = None

        if exc is not None:
            self._drain_active_modules(_format_error(exc))

        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_modules.clear()
        self._active_modules.clear()
        return False

    def _attach_module_hooks(self) -> None:
        post = self._module_post_hook()
        for name, module in self.model.named_modules():
            if not self._should_hook_module(name, module):
                continue
            self._handles.append(
                module.register_forward_pre_hook(
                    self._module_pre_hook(name),
                    with_kwargs=True,
                )
            )
            self._handles.append(module.register_forward_hook(post))

    def _should_hook_module(self, name: str, module: nn.Module) -> bool:
        module_type = type(module).__name__
        normalized_type = _normalize_module_type(module_type)
        depth = _module_depth(name)
        if self.config.module_max_depth is not None and depth > self.config.module_max_depth:
            return False
        if self.module_include_types is not None and normalized_type not in self.module_include_types:
            return False
        registered = is_module_explicitly_registered(module_type, self.module_to_class)
        return registered or self.config.include_unknown_modules

    def _module_pre_hook(self, name: str):
        def hook(module, args, kwargs):
            if self.config.sync:
                _sync(self.device)
            module_type = type(module).__name__
            class_name = classify_module(module_type, self.module_to_class)
            display_name = name or "<root>"
            call_id = self._next_call_id()
            t_start = self._elapsed()
            active = _ActiveModule(
                call_id=call_id,
                name=display_name,
                module_type=module_type,
                class_name=class_name,
                depth=_module_depth(name),
                t_start=t_start,
                flops=0 if self.config.record_flops and self.config.operators else None,
            )
            self._pending_modules.setdefault(id(module), []).append(active)
            self._active_modules.append(active)
            self._emit(
                AuditEvent(
                    kind="module",
                    phase="start",
                    name=display_name,
                    class_name=class_name,
                    call_id=call_id,
                    t_start=t_start,
                    module_name=display_name,
                    module_type=module_type,
                    depth=active.depth,
                    input_shapes=_tensor_shapes(args, kwargs) if self.config.record_shapes else None,
                    input_dtypes=_tensor_dtypes(args, kwargs) if self.config.record_dtypes else None,
                )
            )

        return hook

    def _module_post_hook(self):
        def hook(module, args, output):
            stack = self._pending_modules.get(id(module))
            if not stack:
                return
            if self.config.sync:
                _sync(self.device)
            active = stack.pop()
            self._remove_active_module(active)
            self._emit_module_stop(active)

        return hook

    def _dispatch_op(self, func, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_key = str(func)
        op_name = _normalize_op_name(op_key)
        if self.op_include_names is not None and op_name not in self.op_include_names:
            return func(*args, **kwargs)

        registered = is_explicitly_registered(op_key, self.op_to_class)
        if not registered and not self.config.include_unknown_ops:
            return func(*args, **kwargs)

        class_name = classify(op_key, self.op_to_class)
        call_id = self._next_call_id()
        if self.config.sync:
            _sync(self.device)
        t_start = self._elapsed()
        self._emit(
            AuditEvent(
                kind="operator",
                phase="start",
                name=op_key,
                class_name=class_name,
                call_id=call_id,
                t_start=t_start,
                op_name=op_name,
                input_shapes=_tensor_shapes(args, kwargs) if self.config.record_shapes else None,
                input_dtypes=_tensor_dtypes(args, kwargs) if self.config.record_dtypes else None,
            )
        )

        try:
            output = func(*args, **kwargs)
        except BaseException as exc:
            t_stop = self._elapsed()
            self._emit(
                AuditEvent(
                    kind="operator",
                    phase="stop",
                    name=op_key,
                    class_name=class_name,
                    call_id=call_id,
                    t_start=t_start,
                    t_stop=t_stop,
                    duration_s=t_stop - t_start,
                    op_name=op_name,
                    error=_format_error(exc),
                )
            )
            raise

        if self.config.sync:
            _sync(self.device)
        t_stop = self._elapsed()
        flops = (
            estimate_flops(func, args, kwargs, output)
            if self.config.record_flops
            else None
        )
        self._add_flops_to_active_modules(flops)
        self._emit(
            AuditEvent(
                kind="operator",
                phase="stop",
                name=op_key,
                class_name=class_name,
                call_id=call_id,
                t_start=t_start,
                t_stop=t_stop,
                duration_s=t_stop - t_start,
                flops=flops,
                op_name=op_name,
            )
        )
        return output

    def _emit_module_stop(self, active: _ActiveModule, *, error: str | None = None) -> None:
        t_stop = self._elapsed()
        self._emit(
            AuditEvent(
                kind="module",
                phase="stop",
                name=active.name,
                class_name=active.class_name,
                call_id=active.call_id,
                t_start=active.t_start,
                t_stop=t_stop,
                duration_s=t_stop - active.t_start,
                flops=active.flops,
                module_name=active.name,
                module_type=active.module_type,
                depth=active.depth,
                error=error,
            )
        )

    def _drain_active_modules(self, error: str) -> None:
        while self._active_modules:
            active = self._active_modules.pop()
            self._emit_module_stop(active, error=error)

    def _remove_active_module(self, active: _ActiveModule) -> None:
        for idx in range(len(self._active_modules) - 1, -1, -1):
            if self._active_modules[idx] is active:
                del self._active_modules[idx]
                return

    def _add_flops_to_active_modules(self, flops: int | None) -> None:
        if flops is None:
            return
        for active in self._active_modules:
            if active.flops is None:
                active.flops = 0
            active.flops += int(flops)

    def _next_call_id(self) -> int:
        call_id = self._call_id
        self._call_id += 1
        return call_id

    def _elapsed(self) -> float:
        if self.t0 is None:
            self.t0 = time.perf_counter()
        return time.perf_counter() - self.t0

    def _emit(self, event: AuditEvent) -> None:
        _enqueue(self.event_queue, event)


class _AuditDispatchMode(TorchDispatchMode):
    def __init__(self, ctx: AuditHookContext) -> None:
        super().__init__()
        self.ctx = ctx

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.ctx._dispatch_op(func, args, kwargs)


def attach_hooks(
    model: nn.Module,
    *,
    event_queue,
    config: AuditConfig | None = None,
    device: str | torch.device | None = None,
    t0: float | None = None,
) -> AuditHookContext:
    """Attach module/operator audit hooks around arbitrary user code."""
    return AuditHookContext(
        model,
        event_queue=event_queue,
        config=config,
        device=device,
        t0=t0,
    )


def capture_events(
    model: nn.Module,
    inputs: Sequence[Any] | None = None,
    *,
    run=None,
    config: AuditConfig | None = None,
    device: str | torch.device | None = None,
) -> list[AuditEvent]:
    """Run one callable or ``model(*inputs)`` and return queued audit events."""
    if run is None and inputs is None:
        raise ValueError("provide either inputs or run")

    device_obj = torch.device(device) if device is not None else None
    prepared_inputs = tuple(inputs or ())
    if device_obj is not None:
        prepared_inputs = tuple(
            value.to(device_obj) if isinstance(value, torch.Tensor) else value
            for value in prepared_inputs
        )

    q: Queue[AuditEvent] = Queue()
    with attach_hooks(model, event_queue=q, config=config, device=device_obj):
        if run is not None:
            run()
        else:
            model(*prepared_inputs)
    return _drain_queue(q)


def _drain_queue(q: Queue[AuditEvent]) -> list[AuditEvent]:
    events: list[AuditEvent] = []
    while not q.empty():
        events.append(q.get())
    return events


def _enqueue(sink, event: AuditEvent) -> None:
    if hasattr(sink, "put"):
        sink.put(event)
        return
    if hasattr(sink, "append"):
        sink.append(event)
        return
    if callable(sink):
        sink(event)
        return
    raise TypeError("event_queue must provide put(), append(), or be callable")


def _module_depth(name: str) -> int:
    return 0 if not name else name.count(".")


def _tensor_shapes(args, kwargs) -> list[list[int]]:
    return [list(v.shape) for v in _iter_tensors(args, kwargs)]


def _tensor_dtypes(args, kwargs) -> list[str]:
    return [str(v.dtype) for v in _iter_tensors(args, kwargs)]


def _iter_tensors(args, kwargs):
    kwargs = kwargs or {}
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            yield value


def _sync(device) -> None:
    if device is not None and torch.device(device).type != "cpu":
        getattr(torch, torch.device(device).type).synchronize()


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
