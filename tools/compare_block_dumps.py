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

    re_t = train.get("raw_encoder")
    re_r = rollout.get("raw_encoder")
    if re_t is not None and re_r is not None:
        d_re = (re_t - re_r).abs()
        print(f"raw encoder_hidden_states (model entry, before txt_norm/txt_in): "
              f"max_abs_diff={d_re.max().item():.3e} mean={d_re.mean().item():.3e}")
    rh_t = train.get("raw_hidden")
    rh_r = rollout.get("raw_hidden")
    if rh_t is not None and rh_r is not None:
        d_rh = (rh_t - rh_r).abs()
        print(f"raw hidden_states (model entry, before img_in): "
              f"max_abs_diff={d_rh.max().item():.3e} mean={d_rh.mean().item():.3e}")

    pb_t = train.get("preblocks", {}) or {}
    pb_r = rollout.get("preblocks", {}) or {}
    if pb_t and pb_r:
        print()
        print("pre-block module hooks (input/output of txt_norm/txt_in/img_in):")
        for name in ("txt_norm", "txt_in", "img_in", "time_text_embed"):
            t_slot = pb_t.get(name)
            r_slot = pb_r.get(name)
            if not t_slot or not r_slot:
                continue
            t_in, t_out = t_slot.get("in"), t_slot.get("out")
            r_in, r_out = r_slot.get("in"), r_slot.get("out")
            if t_in is not None and r_in is not None:
                d_in = (t_in - r_in).abs()
                d_out = (t_out - r_out).abs() if (t_out is not None and r_out is not None) else None
                print(f"  {name:>9}  in: max={d_in.max().item():.3e} mean={d_in.mean().item():.3e}"
                      + (f"   out: max={d_out.max().item():.3e} mean={d_out.mean().item():.3e}"
                         if d_out is not None else ""))
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
    if first_div_out is not None:
        print(f"first divergent block: {first_div_out}")
    else:
        print("all blocks bit-equal up to K_PEEK")


if __name__ == "__main__":
    main()
