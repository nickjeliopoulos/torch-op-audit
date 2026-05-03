"""Operator-class registry.

Maps ATen operator names to a small set of non-overlapping classes. The
default taxonomy mirrors Ivanov et al. ("Data Movement Is All You Need"):
``tensor_contraction``, ``stat_normalization``, ``elementwise``. The
``elementwise`` class doubles as the catch-all so coverage is always 100%.

Op names are matched on the bare overload-less name (e.g. ``mm`` from
``aten::mm.default`` or ``aten.mm``). Use :func:`classify` to look up a
class for any string in those forms.
"""

from __future__ import annotations

from typing import Iterable, Mapping


DEFAULT_CLASSES: dict[str, set[str]] = {
    "tensor_contraction": {
        "mm",
        "addmm",
        "bmm",
        "baddbmm",
        "matmul",
        "linear",
        "convolution",
        "_convolution",
        "convolution_backward",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_efficient_attention_backward",
        "_scaled_dot_product_flash_attention_backward",
        "_scaled_dot_product_cudnn_attention_backward",
    },
    "stat_normalization": {
        "native_layer_norm",
        "native_layer_norm_backward",
        "native_batch_norm",
        "native_batch_norm_backward",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit_no_training",
        "_native_batch_norm_legit_functional",
        "native_group_norm",
        "native_group_norm_backward",
        "rms_norm",
        "_softmax",
        "_softmax_backward_data",
        "_log_softmax",
        "_log_softmax_backward_data",
        "var",
        "var_mean",
        "std",
        "mean",
    },
    "elementwise": {
        # arithmetic
        "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_",
        "rsub", "neg", "abs", "pow", "sqrt", "rsqrt", "exp", "log",
        "reciprocal", "addcmul", "addcdiv",
        # activations
        "relu", "relu_", "gelu", "gelu_backward",
        "silu", "silu_backward", "sigmoid", "sigmoid_backward",
        "tanh", "tanh_backward", "hardtanh", "hardtanh_backward",
        "hardswish", "hardswish_backward", "hardsigmoid",
        "leaky_relu", "leaky_relu_backward", "elu", "elu_backward",
        "threshold_backward",
        # comparisons / select
        "where", "clamp", "clamp_min", "clamp_max",
        "eq", "ne", "lt", "le", "gt", "ge",
        "minimum", "maximum",
        # dropout
        "dropout", "native_dropout", "native_dropout_backward",
        # layout / view (zero-FLOP, classified here for time accounting)
        "view", "_unsafe_view", "reshape", "permute", "transpose",
        "contiguous", "clone", "expand", "squeeze", "unsqueeze",
        "cat", "stack", "split", "split_with_sizes", "chunk",
        "slice", "select", "narrow", "unbind", "flatten",
        "to", "_to_copy", "copy_", "fill_",
        "zeros_like", "ones_like", "empty_like", "new_zeros", "new_ones",
        "detach",
        # reductions / accumulation often used for residual paths
        "sum",
    },
}

CATCH_ALL_CLASS = "elementwise"


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


def all_classes(op_to_class: Mapping[str, str]) -> list[str]:
    """All class labels that appear in the registry, with the catch-all included."""
    seen = {*op_to_class.values(), CATCH_ALL_CLASS}
    return sorted(seen)
