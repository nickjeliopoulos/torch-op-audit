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
        # --- matrix multiplications ---
        "mm",
        "addmm",
        "bmm",
        "baddbmm",
        "matmul",
        "linear",
        # --- convolutions (high-level + cuDNN kernel) ---
        # Note: conv2d and cudnn_convolution are dispatch wrappers / kernels
        # for the canonical "convolution" op. All three are registered here so
        # latency is attributed correctly; FLOPs are only counted at the
        # "convolution" level to avoid double-counting.
        "convolution",
        "_convolution",
        "convolution_backward",
        "conv2d",                   # Python-level wrapper seen in profiler
        "cudnn_convolution",        # actual cuDNN kernel seen in profiler
        "cudnn_convolution_transpose",
        # --- matrix exponential (sequence of GEMMs via Padé approximant) ---
        "linalg_matrix_exp",
        # --- attention ---
        # SDPA wrapper + the inner-leaf ATen ops it dispatches to (the leaf op is
        # what actually launches the cutlass / flash kernel; both are tagged here
        # so attribution is correct regardless of which level torch surfaces).
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_efficient_attention_backward",
        "_scaled_dot_product_flash_attention_backward",
        "_scaled_dot_product_cudnn_attention_backward",
        "_efficient_attention_forward",
        "_efficient_attention_backward",
        "_flash_attention_forward",
        "_flash_attention_backward",
    },
    "stat_normalization": {
        # --- layer norm ---
        "layer_norm",               # Python wrapper (latency only; FLOP counted at native level)
        "native_layer_norm",
        "native_layer_norm_backward",
        # --- batch norm ---
        # Dispatch chain: batch_norm → _batch_norm_impl_index → cudnn_batch_norm
        # All registered for latency; FLOPs only counted at terminal ops.
        "batch_norm",
        "_batch_norm_impl_index",
        "cudnn_batch_norm",         # cuDNN terminal kernel
        "cudnn_batch_norm_backward",
        "native_batch_norm",
        "native_batch_norm_backward",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit_no_training",
        "_native_batch_norm_legit_functional",
        # --- group norm ---
        "native_group_norm",
        "native_group_norm_backward",
        # --- rms norm ---
        "rms_norm",
        "_fused_rms_norm",
        "_fused_rms_norm_backward",
        # --- softmax / log-softmax ---
        "softmax",                  # Python wrapper
        "_softmax",
        "_softmax_backward_data",
        "log_softmax",
        "_log_softmax",
        "_log_softmax_backward_data",
        # --- misc statistical reductions ---
        "var",
        "var_mean",
        "std",
        "mean",
    },
    "elementwise": {
        # arithmetic
        "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_",
        "rsub", "neg", "abs", "pow", "sqrt", "rsqrt", "exp", "log",
        "log2", "log10", "log1p", "expm1",
        "reciprocal", "addcmul", "addcdiv",
        "ceil", "floor", "round", "trunc", "sign",
        # activations
        "relu", "relu_", "gelu", "gelu_backward",
        "silu", "silu_backward", "sigmoid", "sigmoid_backward",
        "tanh", "tanh_backward", "hardtanh", "hardtanh_backward",
        "hardswish", "hardswish_backward", "hardsigmoid",
        "leaky_relu", "leaky_relu_backward", "elu", "elu_backward",
        "threshold_backward",
        "softplus", "softplus_backward",
        # in-place activations seen in profiler (e.g. ReLU → clamp_min_)
        "clamp_min_", "clamp_max_",
        # comparisons / select
        "where", "clamp", "clamp_min", "clamp_max",
        "eq", "ne", "lt", "le", "gt", "ge",
        "minimum", "maximum",
        # dropout
        "dropout", "native_dropout", "native_dropout_backward",
        # pooling (sliding-window, no learned weights)
        "max_pool2d", "max_pool2d_with_indices",
        "max_pool2d_with_indices_backward",
        "avg_pool2d", "avg_pool2d_backward",
        "adaptive_avg_pool2d", "adaptive_avg_pool2d_backward",
        "adaptive_max_pool2d",
        # layout / view (zero-FLOP, classified here for time accounting)
        "view", "_unsafe_view", "reshape", "permute", "transpose",
        "contiguous", "clone", "expand", "squeeze", "unsqueeze",
        "cat", "stack", "split", "split_with_sizes", "chunk",
        "slice", "select", "narrow", "unbind", "flatten",
        "to", "_to_copy", "copy_", "fill_",
        "zeros_like", "ones_like", "empty_like", "new_zeros", "new_ones",
        "empty", "empty_strided",   # allocation (zero FLOP)
        "detach",
        # data movement / indexing seen in transformers (Swin roll, position bias)
        "t",                        # transpose (layout)
        "as_strided", "as_strided_",
        "roll",                     # Swin cyclic shift
        "index", "index_put", "index_put_",
        "index_select", "gather", "scatter", "scatter_add",
        # reductions / accumulation often used for residual paths
        "sum",
        "max", "min", "amax", "amin", "argmax", "argmin",
        "cumsum", "cumprod",
        # tensor creation
        "arange", "linspace", "logspace", "full", "full_like", "zeros", "ones",
        "rand", "randn", "rand_like", "randn_like",
        # sorting / unique (no clean class — generally cheap data movement)
        "sort", "argsort", "topk",
        "unique", "unique_consecutive", "unique_dim",
        # scalar extraction / sync points
        "_local_scalar_dense",
        # bitwise / logical
        "bitwise_and", "bitwise_or", "bitwise_not", "bitwise_xor",
        "logical_and", "logical_or", "logical_not", "logical_xor",
        "all", "any",
        # type / device
        "type_as",
    },
}

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
