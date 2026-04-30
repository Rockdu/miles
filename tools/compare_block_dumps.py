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
    ti = train.get("inputs", [])
    ri = rollout.get("inputs", [])
    print(f"{'block':>5}  {'in_max':>10}  {'in_mean':>10}  {'out_max':>10}  {'out_mean':>10}  {'rollout_norm':>12}")
    first_div_in = None
    first_div_out = None
    for i in range(n):
        diff_out = (tb[i] - rb[i]).abs()
        m, mn = diff_out.max().item(), diff_out.mean().item()
        rn = rb[i].norm().item()
        if i < len(ti) and i < len(ri):
            diff_in = (ti[i] - ri[i]).abs()
            im, imn = diff_in.max().item(), diff_in.mean().item()
        else:
            im = imn = float("nan")
        if im > 1e-6 and first_div_in is None:
            first_div_in = i
        if m > 1e-6 and first_div_out is None:
            first_div_out = i
        marker = ""
        if first_div_out == i:
            marker += "  out:↑↑"
        if first_div_in == i:
            marker += "  in:↑↑"
        print(f"{i:5d}  {im:10.3e}  {imn:10.3e}  {m:10.3e}  {mn:10.3e}  {rn:12.3e}{marker}")
    print()
    if first_div is not None:
        print(f"first divergent block: {first_div}")
    else:
        print("all blocks bit-equal up to K_PEEK")


if __name__ == "__main__":
    main()
