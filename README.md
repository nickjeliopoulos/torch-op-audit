# torch-op-audit

Live operator/module auditing and per-class FLOP/latency reporting for PyTorch neural networks.

Attach hooks to arbitrary PyTorch scripts, stream start/stop events into a Python queue, and classify work as **tensor contractions** (matmuls, convolutions), **statistical normalizations** (layer norm, batch norm, softmax), **elementwise** operations, or **other**. You can still generate detailed CSV/LaTeX reports from the preserved benchmark path.

Inspired by Ivanov et al.'s ["Data Movement Is All You Need"](https://arxiv.org/abs/2007.00072).

## Quick Start

```bash
# Install
pip install -e .

# Run a config-backed benchmark report
python -m torch_arch_op_bench.cli --config configs/timm/timm_vit_small.yaml --fwd --bwd

# Or audit arbitrary Python directly
from queue import Queue
import torch
from torchvision.models import resnet18
from torch_arch_op_bench import AuditConfig, aggregate_audit_events, attach_hooks

model = resnet18(weights=None).cuda().eval()
x = torch.randn(8, 3, 224, 224, device="cuda")
events = Queue()

cfg = AuditConfig(modules=True, operators=True, record_flops=True)
with torch.no_grad(), attach_hooks(model, event_queue=events, config=cfg):
    model(x)

records = []
while not events.empty():
    records.append(events.get())

print(aggregate_audit_events(records, kind="operator"))
```

## Output

```
GPU: NVIDIA A100-SXM4-40GB
=== forward (NVIDIA A100-SXM4-40GB) ===
                      count         flops  latency_ms   %_flop  %_runtime
class
tensor_contraction    60500  6.264750e+11     1000.25     99.80      60.5
stat_normalization       64  1.270784e+09       35.15      0.17      25.5
elementwise           1001  1.968640e+07       10.20      0.03      13.0
```

Files written:
- `NVIDIA_A100-SXM4-40GB__fwd_summary.csv` / `.tex`
- `NVIDIA_A100-SXM4-40GB__bwd_summary.csv` / `.tex`
- `..._fwd_detailed.csv` / `.tex` (per-op breakdown)
- `..._bwd_detailed.csv` / `.tex`

## API Notes

`attach_hooks()` emits `AuditEvent` dataclasses with `phase="start"` as soon as an operator/module begins and `phase="stop"` when it returns. Stop events include elapsed time and best-effort FLOPs when a formula exists.

Unknown operators can be included as `other` with `AuditConfig(include_unknown_ops=True)`. Unknown modules are skipped by default and can be included as `other` with `AuditConfig(include_unknown_modules=True)`.

## Configuration Reports

Config-backed reports use YAML in this shape:

```yaml
model:
  import: torchvision.models.resnet18
  kwargs:
    weights: null

input:
  shape: [8, 3, 224, 224]
  dtype: float32

benchmark:
  fwd: true
  bwd: true
  warmup: 10
  iters: 50

output:
  dir: ./results
  latex: true
  csv: true
```

Or override the default 3-class operator taxonomy with custom classes:

```yaml
classes:
  my_class:
    - mm
    - addmm
```

## CLI

```bash
python -m torch_arch_op_bench.cli \
  --config configs/timm/timm_vit_small.yaml \
  --fwd --bwd \
  --out ./my_results
```

## Included configs

| Config | Model | Notes |
|---|---|---|
| `configs/timm/timm_vit_small.yaml` | ViT-Small/16 | attention + LayerNorm heavy |
| `configs/timm/timm_vit_base.yaml` | ViT-Base/16 | larger attention model |
| `configs/timm/timm_deit3_small.yaml` | DeiT-III Small/16 | ViT variant with class token |
| `configs/timm/timm_swin_small.yaml` | Swin-Small | windowed attention + patch merge |
| `configs/timm/timm_convnext_small.yaml` | ConvNeXt-Small | conv backbone with LayerNorm |

TIMM models require `timm` to be installed:

```bash
pip install -e ".[timm]"
```
