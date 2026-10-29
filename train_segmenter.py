#!/usr/bin/env python
"""Entry point for s4vy:
   torchrun --nproc_per_node=8 train_segmenter.py --config segomega/configs/s4vy.yaml
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segomega.engine.train_segmenter import train_segmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.opts:
        k, v = kv.split("=", 1)
        d = cfg
        *parts, last = k.split(".")
        for p in parts:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    train_segmenter(cfg)


if __name__ == "__main__":
    main()
