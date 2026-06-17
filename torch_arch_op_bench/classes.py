"""Operator and module class registries.

Maps ATen operator names to a small set of non-overlapping classes. The
default taxonomy mirrors Ivanov et al. ("Data Movement Is All You Need"):
``tensor_contraction``, ``stat_normalization``, ``elementwise``. Unknown
operators and modules are assigned to the catch-all class ``other`` when the
caller chooses to include them.

The op and module membership of each class lives in top-level JSON files under
``classes/`` so the taxonomy can be edited and version-controlled on its own.
Op names are matched on the bare overload-less name (e.g. ``mm`` from
``aten::mm.default`` or ``aten.mm``); module names are matched on class name
(e.g. ``Linear`` from ``torch.nn.Linear``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


_CLASSES_DIR = Path(__file__).resolve().parent.parent / "classes"
_LEGACY_CLASSES_PATH = _CLASSES_DIR / "default.json"
_DEFAULT_OP_CLASSES_PATH = _CLASSES_DIR / "default_ops.json"
_DEFAULT_MODULE_CLASSES_PATH = _CLASSES_DIR / "default_modules.json"


def load_classes_json(path: str | Path) -> dict[str, set[str]]:
    """Load a ``{class_name: [name, ...]}`` JSON taxonomy into ``{class: set}``."""
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return {}
    raw = json.loads(text)
    return {cls: set(ops) for cls, ops in raw.items()}


def _load_default_op_classes() -> dict[str, set[str]]:
    table = load_classes_json(_DEFAULT_OP_CLASSES_PATH)
    if table:
        return table
    return load_classes_json(_LEGACY_CLASSES_PATH)


DEFAULT_OP_CLASSES: dict[str, set[str]] = _load_default_op_classes()
DEFAULT_MODULE_CLASSES: dict[str, set[str]] = load_classes_json(_DEFAULT_MODULE_CLASSES_PATH)

# Backwards-compatible alias for the old operator-only registry name.
DEFAULT_CLASSES: dict[str, set[str]] = DEFAULT_OP_CLASSES

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


def _normalize_module_type(name: str) -> str:
    """Return a bare module class name from a dotted path or class ``repr``."""
    name = name.strip()
    if name.startswith("<class '") and name.endswith("'>"):
        name = name[len("<class '") : -2]
    return name.rsplit(".", 1)[-1]


def _build_name_to_class(
    defaults: Mapping[str, Iterable[str]],
    overrides: Mapping[str, Iterable[str]] | None = None,
    *,
    normalize,
) -> dict[str, str]:
    table: dict[str, set[str]] = {cls: set(names) for cls, names in defaults.items()}
    if overrides:
        for cls, names in overrides.items():
            table[cls] = set(names)

    flat: dict[str, str] = {}
    for cls, names in table.items():
        for name in names:
            normalized = normalize(name)
            if normalized in flat and flat[normalized] != cls:
                raise ValueError(
                    f"Name {name!r} is assigned to both {flat[normalized]!r} and {cls!r}; "
                    "classes must be non-overlapping"
                )
            flat[normalized] = cls
    return flat


def build_op_to_class(
    overrides: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, str]:
    """Flatten the class registry into a ``{op_name: class_name}`` map.

    ``overrides`` replaces the default op set for any classes it lists
    (per-class replacement, not merge). Raises ``ValueError`` if the
    resulting registry assigns the same op to more than one class.
    """
    return _build_name_to_class(DEFAULT_OP_CLASSES, overrides, normalize=_normalize_op_name)


def build_module_to_class(
    overrides: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, str]:
    """Flatten the module registry into a ``{module_type: class_name}`` map."""
    return _build_name_to_class(
        DEFAULT_MODULE_CLASSES,
        overrides,
        normalize=_normalize_module_type,
    )


def classify(op_name: str, op_to_class: Mapping[str, str]) -> str:
    """Return the class name for ``op_name``, falling back to the catch-all."""
    return op_to_class.get(_normalize_op_name(op_name), CATCH_ALL_CLASS)


def classify_module(module_type: str, module_to_class: Mapping[str, str]) -> str:
    """Return the class name for ``module_type``, falling back to the catch-all."""
    return module_to_class.get(_normalize_module_type(module_type), CATCH_ALL_CLASS)


def is_explicitly_registered(op_name: str, op_to_class: Mapping[str, str]) -> bool:
    """True if ``op_name`` has an explicit entry in the registry.

    Returns False for ops that would be routed to the catch-all, i.e. ops
    that are not named in any class definition. This is the signal used to
    build the post-mortem "missed ops" report.
    """
    return _normalize_op_name(op_name) in op_to_class


def is_module_explicitly_registered(
    module_type: str,
    module_to_class: Mapping[str, str],
) -> bool:
    """True if ``module_type`` has an explicit entry in the module registry."""
    return _normalize_module_type(module_type) in module_to_class


def all_classes(name_to_class: Mapping[str, str]) -> list[str]:
    """All class labels that appear in the registry, with the catch-all included."""
    seen = {*name_to_class.values(), CATCH_ALL_CLASS}
    return sorted(seen)
