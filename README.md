# torch-op-audit

Per-operator-class FLOP and latency benchmarking for PyTorch neural networks.

Measure what fraction of your model's FLOPs and runtime comes from each type of operation: **tensor contractions** (matmuls, convolutions), **statistical normalizations** (layer norm, batch norm, softmax), and **elementwise** operations. 
Get a detailed breakdown with GPU-annotated LaTeX tables.

Inspired by Ivanov et al.'s ["Data Movement Is All You Need"](https://arxiv.org/abs/2007.00072).

## Quick Start

```bash
# Install
pip install -e .

# Benchmark ResNet18
python -m torch_arch_op_bench.cli --config configs/example.yaml --fwd --bwd

# Or in Python
from torch_arch_op_bench import benchmark
import torch
from torchvision.models import resnet18

model = resnet18(weights=None).cuda()
x = torch.randn(8, 3, 224, 224, device="cuda")

report = benchmark(model, [x], fwd=True, bwd=True)
print(report.fwd_summary)
report.write("./results")
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

## Configuration

Edit `configs/example.yaml`:

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
  --config configs/example.yaml \
  --fwd --bwd \
  --out ./my_results

# Sweep over multiple configs
python scripts/run_sweep.py --config-dir configs/sweep/ --fwd --bwd

# Run all included TIMM configs in one shot
python scripts/run_sweep.py --config-dir configs/ --fwd --bwd
```

## Included configs

| Config | Model | Notes |
|---|---|---|
| `configs/example.yaml` | ResNet-18 (torchvision) | conv-heavy baseline |
| `configs/timm_vit_small.yaml` | ViT-Small/16 | attention + LayerNorm heavy |
| `configs/timm_vit_base.yaml` | ViT-Base/16 | larger attention model |
| `configs/timm_deit3_small.yaml` | DeiT-III Small/16 | ViT variant with class token |
| `configs/timm_swin_small.yaml` | Swin-Small | windowed attention + patch merge |
| `configs/timm_convnext_small.yaml` | ConvNeXt-Small | conv backbone with LayerNorm |

TIMM models require `timm` to be installed:

```bash
pip install -e ".[timm]"
```
