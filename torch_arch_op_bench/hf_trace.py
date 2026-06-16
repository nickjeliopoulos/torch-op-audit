"""HuggingFace generate-trace entry point.

Loads a HuggingFace model and runs the module-phase tracer across one or more
``model.generate()`` calls on a real dataset sample. Produces a JSONL phase-trace
file that can be overlaid on GEOPM power/frequency telemetry.

All model-specific details live in the YAML config (model path, dtype, dataset id,
preprocessor, generate kwargs) so this file contains no model-name strings.

Usage::

    python -m torch_arch_op_bench.hf_trace --config configs/nemotron_8b_xpu.yaml

    # Override trace iterations or output directory:
    python -m torch_arch_op_bench.hf_trace \\
        --config configs/nemotron_8b_xpu.yaml \\
        --trace-iters 3 \\
        --out ./my_results
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from .report import get_gpu_name
from .trace import trace_module_phases, write_trace_jsonl

# modeling_ministral.py (shipped with the Nemotron snapshot) imports
# sdpa_mask_older_torch, which was removed in newer transformers builds.
import transformers.masking_utils as _masking_utils
if not hasattr(_masking_utils, "sdpa_mask_older_torch"):
    _masking_utils.sdpa_mask_older_torch = _masking_utils.sdpa_mask

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "long": torch.long,
}


def _load_sample(cfg) -> dict:
    """Load one (image, question) row from the configured dataset."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset.id, split=cfg.dataset.split, streaming=False)
    idx = int(cfg.dataset.get("sample_index", 0))
    sample = None
    for i, row in enumerate(ds):
        img = row.get("image") or row.get("img")
        q   = row.get("question") or row.get("query")
        if img is None or q is None:
            continue
        if i == idx:
            sample = {"image": img, "question": q}
            break
    if sample is None:
        raise RuntimeError(
            f"No usable (image, question) row at index {idx} in '{cfg.dataset.id}'"
        )
    return sample


def _build_preprocessor(cfg, model_path: str, tokenizer):
    """Return a callable ``process_messages(tokenizer, messages, ...)`` for the model.

    Two dispatch paths driven by ``config.processor.type``:

    ``custom_module``
        Imports ``config.processor.fn`` from ``config.processor.import``.
        The model dir must already be on ``sys.path`` (i.e. ``sys_path_prepend: true``).
        Used by models that ship their own preprocessing code (e.g. Nemotron's
        ``image_processing.py``).

    ``auto``
        Uses ``transformers.AutoProcessor.from_pretrained`` — standard for HF VLMs
        that don't ship custom code.
    """
    proc_type = str(cfg.processor.type)
    if proc_type == "custom_module":
        sys.path.insert(0, model_path)
        mod = importlib.import_module(str(cfg.processor["import"]))
        return getattr(mod, cfg.processor.fn)
    if proc_type == "auto":
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
        # Wrap to match the (tokenizer, messages, add_generation_prompt=True) signature
        # expected by the generate_fn builder below.
        def _auto_process(tok, messages, add_generation_prompt=True):
            return proc(messages, return_tensors="pt")
        return _auto_process
    raise ValueError(f"Unknown processor.type: {proc_type!r}. Expected 'custom_module' or 'auto'.")


def _build_generate_fn(model, tokenizer, sample: dict, process_messages, device, cfg):
    """Build a zero-arg callable that runs one full model.generate() pass.

    Mirrors run_sample() in vlm_nemo_single.py: preprocesses the sample once
    (outside the closure) then repeatedly calls generate() with the same
    pre-placed tensors.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text",  "text": sample["question"]},
            ],
        }
    ]
    batch = process_messages(tokenizer, messages, add_generation_prompt=True)

    prompt_ids   = batch["input_ids"].to(device)
    pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    image_sizes  = batch.get("image_sizes", None)

    # Any extra keys the preprocessor returned (e.g. attention_mask) are passed through.
    extra_kwargs = {
        k: v for k, v in batch.items()
        if k not in ("input_ids", "pixel_values", "image_sizes")
    }

    # Build generate kwargs from config; only include keys that are present.
    gen_cfg = cfg.get("generate", {}) or {}
    gen_kwargs: dict = {}
    for key in ("max_new_tokens", "steps", "block_length", "threshold", "shift_logits"):
        val = gen_cfg.get(key, None)
        if val is not None:
            gen_kwargs[key] = val
    if image_sizes is not None:
        gen_kwargs["image_sizes"] = image_sizes
    gen_kwargs.update(extra_kwargs)

    def generate_fn():
        with torch.inference_mode():
            model.generate(prompt_ids, pixel_values=pixel_values, **gen_kwargs)

    return generate_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trace-iters", type=int, default=None,
                        help="Override config.trace.iters")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override config.output.dir")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)

    model_path = str(cfg.model.hf_pretrained)
    device     = torch.device(str(cfg.model.device))
    dtype      = _DTYPES.get(str(cfg.model.get("dtype", "float32")), torch.float32)

    # Mirrors vlm_nemo_multi.py: resolve repo ID to local snapshot dir if needed.
    if not Path(model_path).is_dir():
        model_path = snapshot_download(model_path)

    print(f"hf_trace: loading tokenizer from {model_path}")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )

    t0_load = time.perf_counter()
    print(f"hf_trace: loading model from {model_path} (dtype={dtype}, device={device})")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        torch_dtype=dtype,
    ).to(device).eval()
    print(f"hf_trace: model loaded in {time.perf_counter() - t0_load:.1f}s  "
          f"gpu={get_gpu_name(device)}")

    process_messages = _build_preprocessor(cfg, model_path, tokenizer)

    print(f"hf_trace: loading dataset sample (id={cfg.dataset.id}, "
          f"index={cfg.dataset.get('sample_index', 0)})")
    sample = _load_sample(cfg)

    generate_fn = _build_generate_fn(model, tokenizer, sample, process_messages, device, cfg)

    warmup = int((cfg.get("benchmark") or {}).get("warmup", 2))
    if warmup > 0:
        print(f"hf_trace: running {warmup} warmup pass(es) ...")
        for _ in range(warmup):
            generate_fn()

    trace_iters = args.trace_iters or int((cfg.get("trace") or {}).get("iters", 1))
    max_depth   = (cfg.get("trace") or {}).get("max_depth", None)
    inc_types   = (cfg.get("trace") or {}).get("include_types", None)

    print(f"hf_trace: tracing {trace_iters} generate pass(es) ...")
    events, meta = trace_module_phases(
        model,
        generate_fn=generate_fn,
        iters=trace_iters,
        device=device,
        max_depth=max_depth,
        include_types=list(inc_types) if inc_types else None,
    )

    out_dir = Path(args.out or cfg.output.get("dir", "./results/hf_trace"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slug for the output filename: model dir basename + device.
    model_slug = Path(model_path).name.replace(" ", "_")
    dev_slug   = str(device).replace(":", "_")
    out_path   = out_dir / f"{model_slug}__{dev_slug}__phase_trace.jsonl"

    write_trace_jsonl(out_path, events, meta)
    print(f"hf_trace: wrote {len(events)} events to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
