"""Operator-class registry.

Maps ATen operator names to a small set of non-overlapping classes. The
default taxonomy mirrors Ivanov et al. ("Data Movement Is All You Need"):
``tensor_contraction``, ``stat_normalization``, ``elementwise``. The
``elementwise`` class doubles as the catch-all so coverage is always 100%.

The op membership of each class lives in ``classes/default.json`` at the repo
top level (not in code) so the taxonomy can be edited and version-controlled on
its own. Op names are matched on the bare overload-less name (e.g. ``mm`` from
``aten::mm.default`` or ``aten.mm``); use :func:`classify` to look up a class
for any string in those forms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


# Top-level ``classes/default.json`` (sibling of the package directory).
_DEFAULT_CLASSES_PATH = Path(__file__).resolve().parent.parent / "classes" / "default.json"


def load_classes_json(path: str | Path) -> dict[str, set[str]]:
    """Load a ``{class_name: [op, ...]}`` JSON taxonomy into ``{class: set}``."""
    with open(path) as f:
        raw = json.load(f)
    return {cls: set(ops) for cls, ops in raw.items()}


DEFAULT_CLASSES: dict[str, set[str]] = load_classes_json(_DEFAULT_CLASSES_PATH)

CATCH_ALL_CLASS = "other"


def _normalize_op_name(name: str) -> str:
    """Strip namespace and overload suffix from an op name.

    Accepts forms like ``aten::mm.default``, ``aten.mm.default``, ``aten::mm``,
    ``mm.default``, or ``mm``. Returns the bare op name (``mm``).
    """
    # drop namespace
    if "::" in name:
        name = name.split("::", 1)[1]
    elif name.startswith("aten."):
        name = name[len("aten.") :]
    # drop overload (".default", ".out", ".Tensor", ...)
    if "." in name:
        name = name.split(".", 1)[0]
    return name


def build_op_to_class(
    overrides: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, str]:
    """Flatten the class registry into a ``{op_name: class_name}`` map.

    ``overrides`` replaces the default op set for any classes it lists
    (per-class replacement, not merge). Raises ``ValueError`` if the
    resulting registry assigns the same op to more than one class.
    """
    table: dict[str, set[str]] = {cls: set(ops) for cls, ops in DEFAULT_CLASSES.items()}
    if overrides:
        for cls, ops in overrides.items():
            table[cls] = set(ops)

    flat: dict[str, str] = {}
    for cls, ops in table.items():
        for op in ops:
            if op in flat and flat[op] != cls:
                raise ValueError(
                    f"Op {op!r} is assigned to both {flat[op]!r} and {cls!r}; "
                    "operator classes must be non-overlapping"
                )
            flat[op] = cls
    return flat


def classify(op_name: str, op_to_class: Mapping[str, str]) -> str:
    """Return the class name for ``op_name``, falling back to the catch-all."""
    return op_to_class.get(_normalize_op_name(op_name), CATCH_ALL_CLASS)


def is_explicitly_registered(op_name: str, op_to_class: Mapping[str, str]) -> bool:
    """True if ``op_name`` has an explicit entry in the registry.

    Returns False for ops that would be routed to the catch-all, i.e. ops
    that are not named in any class definition. This is the signal used to
    build the post-mortem "missed ops" report.
    """
    return _normalize_op_name(op_name) in op_to_class


def all_classes(op_to_class: Mapping[str, str]) -> list[str]:
    """All class labels that appear in the registry, with the catch-all included."""
    seen = {*op_to_class.values(), CATCH_ALL_CLASS}
    return sorted(seen)
