# torch-op-audit

Live operator/module auditing and per-class FLOP/latency reporting for PyTorch neural networks.

Attach hooks to arbitrary PyTorch scripts, stream start/stop events into a Python queue, and classify work as **tensor contractions** (matmuls, convolutions), **statistical normalizations** (layer norm, batch norm, softmax), **elementwise** operations, or **other**. You can also aggregate captured events into report tables directly from Python.

Inspired by Ivanov et al.'s ["Data Movement Is All You Need"](https://arxiv.org/abs/2007.00072).

## Quick Start

```bash
# Install
pip install -e .
```

```python
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
=== operator audit ===
                      count         flops  latency_ms   %_flop  %_runtime
class
tensor_contraction    60500  6.264750e+11     1000.25     99.80      60.5
stat_normalization       64  1.270784e+09       35.15      0.17      25.5
elementwise           1001  1.968640e+07       10.20      0.03      13.0
```

## API Notes

`attach_hooks()` emits `AuditEvent` dataclasses with `phase="start"` as soon as an operator/module begins and `phase="stop"` when it returns. Stop events include elapsed time and best-effort FLOPs when a formula exists.

Unknown operators can be included as `other` with `AuditConfig(include_unknown_ops=True)`. Unknown modules are skipped by default and can be included as `other` with `AuditConfig(include_unknown_modules=True)`.

## Tests

The standalone API checks can be run directly:

```bash
python tests/test_api_core.py --device cpu
python tests/test_report_api.py --device cpu
python tests/test_timm_api.py --device cpu

# All-in-one compatibility smoke test
python tests/test_smoke.py --device cpu
```

TIMM models require `timm` to be installed:

```bash
pip install -e ".[timm]"
```
