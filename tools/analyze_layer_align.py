"""Offline analysis of /tmp/layer_align_records.pt produced by
tools/layer_align_diff_vs_sgld.py.

Pairs corresponding modules between diffusers and sgld, computes
per-module input + output diffs. Also extracts the joint attention
output for sgld and reports its breakdown into text/image halves.
"""
from __future__ import annotations

import torch

PATH = "/tmp/layer_align_records.pt"


def _flatten_first_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        for v in x.values():
            r = _flatten_first_tensor(v)
            if r is not None:
                return r
        return None
    if isinstance(x, (tuple, list)):
        for v in x:
            r = _flatten_first_tensor(v)
            if r is not None:
                return r
    return None


def _stats(a, b):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return {"shape_a": tuple(a.shape), "shape_b": tuple(b.shape), "shape_mismatch": True}
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    a_norm = af.norm().item()
    b_norm = bf.norm().item()
    return {
        "shape": tuple(af.shape),
        "abs_max": d.max().item(),
        "abs_mean": d.mean().item(),
        "rel_mean": d.mean().item() / max(af.abs().mean().item(), 1e-12),
        "a_norm": a_norm,
        "b_norm": b_norm,
        "norm_diff_rel": abs(a_norm - b_norm) / max(a_norm, 1e-12),
    }


def _get_in(rec):
    """Extract first tensor input from records[name]: tries args then kwargs."""
    a = _flatten_first_tensor(rec.get("args"))
    if a is not None:
        return a
    return _flatten_first_tensor(rec.get("kwargs"))


def _get_out(rec):
    return _flatten_first_tensor(rec.get("out"))


MAPPING = [
    ("img_mod.1",          "img_mod.1",          "img modulation Linear"),
    ("txt_mod.1",          "txt_mod.1",          "txt modulation Linear"),
    ("img_norm1",          "img_norm1.norm",     "img norm1 (sgld inner FP32LayerNorm)"),
    ("img_norm2",          "img_norm2.norm",     "img norm2 (sgld inner FP32LayerNorm)"),
    ("txt_norm1",          "txt_norm1.norm",     "txt norm1"),
    ("txt_norm2",          "txt_norm2.norm",     "txt norm2"),
    ("attn.to_q",          "attn.to_q",          "img Q projection"),
    ("attn.to_k",          "attn.to_k",          "img K projection"),
    ("attn.to_v",          "attn.to_v",          "img V projection"),
    ("attn.add_q_proj",    "attn.add_q_proj",    "txt Q projection"),
    ("attn.add_k_proj",    "attn.add_k_proj",    "txt K projection"),
    ("attn.add_v_proj",    "attn.add_v_proj",    "txt V projection"),
    ("attn.norm_q",        "attn.norm_q",        "img Q RMSNorm (sgld bypassed via fused-inplace)"),
    ("attn.norm_k",        "attn.norm_k",        "img K RMSNorm (sgld bypassed via fused-inplace)"),
    ("attn.norm_added_q",  "attn.norm_added_q",  "txt Q RMSNorm"),
    ("attn.norm_added_k",  "attn.norm_added_k",  "txt K RMSNorm"),
    ("attn.to_out.0",      "attn.to_out.0",      "img attn output projection"),
    ("attn.to_add_out",    "attn.to_add_out",    "txt attn output projection"),
    ("img_mlp.net.0.proj", "img_mlp.net.0.proj", "img MLP linear 1"),
    ("img_mlp.net.2",      "img_mlp.net.2",      "img MLP linear 2"),
    ("txt_mlp.net.0.proj", "txt_mlp.net.0.proj", "txt MLP linear 1"),
    ("txt_mlp.net.2",      "txt_mlp.net.2",      "txt MLP linear 2"),
]


def _fmt(s):
    if s is None:
        return f"{'no records':>14s}{'':>26s}"
    if "shape_mismatch" in s:
        return f"shape: {s['shape_a']} vs {s['shape_b']}".ljust(40)
    return f"{s['abs_max']:>10.3e}{s['abs_mean']:>10.3e}{s['rel_mean']:>10.3e}{s['norm_diff_rel']:>10.3e}"


def main():
    data = torch.load(PATH, weights_only=False)
    diff = data["diff"]
    sgld = data["sgld"]
    bo = data["block_out"]

    print("=" * 110)
    print("BLOCK-LEVEL OUTPUT DIFF")
    print("=" * 110)
    enc_st = _stats(bo["d_enc"], bo["s_enc"])
    hid_st = _stats(bo["d_hid"], bo["s_hid"])
    print(f"  encoder_out  abs_max={enc_st['abs_max']:.3e}  abs_mean={enc_st['abs_mean']:.3e}  rel_mean={enc_st['rel_mean']:.3e}  norm_diff_rel={enc_st['norm_diff_rel']:.3e}")
    print(f"  hidden_out   abs_max={hid_st['abs_max']:.3e}  abs_mean={hid_st['abs_mean']:.3e}  rel_mean={hid_st['rel_mean']:.3e}  norm_diff_rel={hid_st['norm_diff_rel']:.3e}")

    # Top-level attn module output (post to_out projections) split by stream
    print()
    print("=" * 110)
    print("ATTN MODULE OUTPUT  (post to_out projections)")
    print("=" * 110)
    d_attn = diff["attn"]["out"]; s_attn = sgld["attn"]["out"]
    img_st = _stats(d_attn[0], s_attn[0])
    txt_st = _stats(d_attn[1], s_attn[1])
    print(f"  img stream: abs_max={img_st['abs_max']:.3e}  abs_mean={img_st['abs_mean']:.3e}  rel_mean={img_st['rel_mean']:.3e}  norm_diff_rel={img_st['norm_diff_rel']:.3e}")
    print(f"  txt stream: abs_max={txt_st['abs_max']:.3e}  abs_mean={txt_st['abs_mean']:.3e}  rel_mean={txt_st['rel_mean']:.3e}  norm_diff_rel={txt_st['norm_diff_rel']:.3e}")
    print(f"  ASYMMETRY ratio (img rel_mean / txt rel_mean) = {img_st['rel_mean']/txt_st['rel_mean']:.1f}")

    # Per-module table
    print()
    print("=" * 110)
    print(f"{'diffusers':<22s}{'sgld':<22s}  in: {'abs_max':>10s}{'abs_mean':>10s}{'rel_mean':>10s}{'norm_drel':>10s}   out: {'abs_max':>10s}{'abs_mean':>10s}{'rel_mean':>10s}{'norm_drel':>10s}    note")
    print("=" * 110)
    for d_name, s_name, note in MAPPING:
        if d_name not in diff:
            print(f"{d_name:<22s}{s_name:<22s}  diffusers module not in records ({note})")
            continue
        if s_name not in sgld:
            print(f"{d_name:<22s}{s_name:<22s}  sgld module not in records ({note})")
            continue
        in_st = _stats(_get_in(diff[d_name]), _get_in(sgld[s_name]))
        out_st = _stats(_get_out(diff[d_name]), _get_out(sgld[s_name]))
        print(f"{d_name:<22s}{s_name:<22s}  in: {_fmt(in_st)}   out: {_fmt(out_st)}    {note}")


if __name__ == "__main__":
    main()
