"""Wrapper that runs the CLI over every YAML in a config directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from torch_arch_op_bench.cli import main as cli_main
from torch_arch_op_bench.hf_trace import main as hf_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--fwd", action="store_true")
    parser.add_argument("--bwd", action="store_true")
    parser.add_argument("--hf", action="store_true")
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No YAML configs found in {args.config_dir}")

    for cfg_path in configs:
        print(f"\n### {cfg_path.name} ###")
        raw = OmegaConf.load(cfg_path)
        if args.hf:
            hf_main(["--config", str(cfg_path)])
        else:
            argv = ["--config", str(cfg_path)]
            if args.fwd:
                argv.append("--fwd")
            if args.bwd:
                argv.append("--bwd")
            cli_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
