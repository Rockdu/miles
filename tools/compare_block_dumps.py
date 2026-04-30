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
    tei = train.get("encoder_inputs", [])
    rei = rollout.get("encoder_inputs", [])
    tti = train.get("temb_inputs", [])
    rti = rollout.get("temb_inputs", [])
    print(f"{'block':>5}  {'in_max':>9}  {'enc_max':>9}  {'temb_max':>9}  {'out_max':>9}  {'out_mean':>9}  {'rollout_norm':>11}")
    first_div_in = None
    first_div_out = None
    for i in range(n):
        diff_out = (tb[i] - rb[i]).abs()
        m, mn = diff_out.max().item(), diff_out.mean().item()
        rn = rb[i].norm().item()
        im = (ti[i] - ri[i]).abs().max().item() if i < len(ti) and i < len(ri) else float("nan")
        em = (tei[i] - rei[i]).abs().max().item() if i < len(tei) and i < len(rei) else float("nan")
        tm = (tti[i] - rti[i]).abs().max().item() if i < len(tti) and i < len(rti) else float("nan")
        marker = ""
        if m > 1e-6 and first_div_out is None:
            first_div_out = i
            marker += "  out:↑↑"
        print(f"{i:5d}  {im:9.2e}  {em:9.2e}  {tm:9.2e}  {m:9.2e}  {mn:9.2e}  {rn:11.2e}{marker}")
    print()
    if first_div is not None:
        print(f"first divergent block: {first_div}")
    else:
        print("all blocks bit-equal up to K_PEEK")


if __name__ == "__main__":
    main()
