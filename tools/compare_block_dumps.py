"""Diff per-block first-row peeks dumped by miles.backends.fsdp_utils.models.block_dump
on train and rollout sides. Prints first divergent block + magnitude per block.

Usage: python tools/compare_block_dumps.py <dir_with_train_and_rollout_block_dump.pt>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    d = Path(sys.argv[1])
    train = torch.load(d / "train_block_dump.pt", weights_only=False)
    rollout = torch.load(d / "rollout_block_dump.pt", weights_only=False)
    tb = train["blocks"]
    rb = rollout["blocks"]
    n = min(len(tb), len(rb))
    print(f"train blocks: {len(tb)}  rollout blocks: {len(rb)}  comparing first {n}")
    print(f"K (first values per block): {train['K']}")
    print()
    print(f"{'block':>5}  {'max_abs_diff':>14}  {'mean_abs_diff':>14}  {'rollout_norm':>14}")
    first_div = None
    for i in range(n):
        diff = (tb[i] - rb[i]).abs()
        m, mn = diff.max().item(), diff.mean().item()
        rn = rb[i].norm().item()
        marker = ""
        if m > 1e-6 and first_div is None:
            first_div = i
            marker = "  <-- first divergent"
        print(f"{i:5d}  {m:14.3e}  {mn:14.3e}  {rn:14.3e}{marker}")
    print()
    if first_div is not None:
        print(f"first divergent block: {first_div}")
    else:
        print("all blocks bit-equal up to K_PEEK")


if __name__ == "__main__":
    main()
