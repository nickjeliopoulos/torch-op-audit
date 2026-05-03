"""Wrapper that runs the CLI over every YAML in a config directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch_arch_op_bench.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--fwd", action="store_true")
    parser.add_argument("--bwd", action="store_true")
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No YAML configs found in {args.config_dir}")

    for cfg in configs:
        print(f"\n### {cfg.name} ###")
        argv = ["--config", str(cfg)]
        if args.fwd:
            argv.append("--fwd")
        if args.bwd:
            argv.append("--bwd")
        cli_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
