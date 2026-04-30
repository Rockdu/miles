"""Side-by-side per-module comparison of one Qwen-Image transformer block,
diffusers vs sglang-diffusion.

Goals:
1. Build one block from each implementation, copy weights from diffusers
   into sgld, run the same input through both.
2. Hook every submodule's forward — record (inputs, outputs).
3. Pair corresponding submodules by a name-mapping table and report
   per-pair input / output diffs (max, mean, rel-mean).
4. Identify divergences worth aligning.

To keep the comparison clean we feed RoPE = None on both sides (sgld's
apply_qk_norm_with_optional_rope short-circuits to plain qk_norm when
cos_sin_cache=None; diffusers' QwenDoubleStreamAttnProcessor2_0 skips
the RoPE block when image_rotary_emb is None). Encoder mask is all-True.
Single GPU, bf16, single batch.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Tuple

# Must run init BEFORE importing sglang block module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sgld_minimal_init import init_minimal

init_minimal()

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as DiffQI

from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformerBlock as SgldBlock,
)

DEV = torch.device("cuda:0")
DTYPE = torch.bfloat16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------- weights
def copy_weights(sgld_block, diff_block):
    """Copy diffusers block parameters into the sgld block. Handles the
    nn.LayerNorm → LayerNormScaleShift.norm rename."""
    src = dict(diff_block.named_parameters())
    dst = dict(sgld_block.named_parameters())
    matched, missing = 0, []
    for name, dst_p in dst.items():
        cand = [name]
        if ".norm." in name:
            cand.append(name.replace(".norm.", "."))
        ok = False
        for c in cand:
            if c in src and src[c].shape == dst_p.shape:
                dst_p.data.copy_(src[c].data.to(dst_p.dtype))
                matched += 1
                ok = True
                break
        if not ok:
            missing.append(name)
    log(f"copy_weights: matched {matched}/{len(dst)}; missing={missing[:8]}{'...' if len(missing)>8 else ''}")
    if missing:
        log(f"  total missing: {len(missing)}")
    return matched, missing


# ------------------------------------------------------------ hooking
def _to_cpu(x):
    """Move tensors / nested tuples to CPU as float, keeping structure."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", torch.float32)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu(v) for v in x)
    return x  # int/None/etc.


def install_hooks(block, records: Dict[str, Any]):
    handles = []
    for name, mod in block.named_modules():
        if name == "":
            continue

        def hk(m, args, kwargs, out, _name=name):
            records[_name] = {
                "args": _to_cpu(args),
                "kwargs": {k: _to_cpu(v) for k, v in kwargs.items()},
                "out": _to_cpu(out),
            }

        handles.append(mod.register_forward_hook(hk, with_kwargs=True))
    return handles


# ---------------------------------------------------------- diff stats
def _flatten_first_tensor(x):
    """If x is a tuple/list, return the first tensor inside; if it's a
    single tensor, return it; if None, return None."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            r = _flatten_first_tensor(v)
            if r is not None:
                return r
    return None


def _stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    if a is None or b is None:
        return {"shape_a": None, "shape_b": None}
    if a.shape != b.shape:
        return {
            "shape_a": tuple(a.shape),
            "shape_b": tuple(b.shape),
            "shape_mismatch": True,
        }
    d = (a - b).abs()
    return {
        "shape": tuple(a.shape),
        "abs_max": d.max().item(),
        "abs_mean": d.mean().item(),
        "rel_mean": d.mean().item() / max(a.abs().mean().item(), 1e-12),
        "a_norm": a.norm().item(),
        "b_norm": b.norm().item(),
    }


# Diffusers name → sgld name. Some sgld modules (img_norm1.norm,
# img_norm2.norm) are inside fused wrappers — map to the inner norm.
MAPPING = [
    # input/output projections
    ("img_mod.1",          "img_mod.1"),
    ("txt_mod.1",          "txt_mod.1"),
    ("img_norm1",          "img_norm1.norm"),
    ("img_norm2",          "img_norm2.norm"),
    ("txt_norm1",          "txt_norm1.norm"),
    ("txt_norm2",          "txt_norm2.norm"),
    # attention
    ("attn.to_q",          "attn.to_q"),
    ("attn.to_k",          "attn.to_k"),
    ("attn.to_v",          "attn.to_v"),
    ("attn.add_q_proj",    "attn.add_q_proj"),
    ("attn.add_k_proj",    "attn.add_k_proj"),
    ("attn.add_v_proj",    "attn.add_v_proj"),
    ("attn.norm_q",        "attn.norm_q"),
    ("attn.norm_k",        "attn.norm_k"),
    ("attn.norm_added_q",  "attn.norm_added_q"),
    ("attn.norm_added_k",  "attn.norm_added_k"),
    ("attn.to_out.0",      "attn.to_out.0"),
    ("attn.to_add_out",    "attn.to_add_out"),
    # MLP
    ("img_mlp.net.0.proj", "img_mlp.net.0.proj"),
    ("img_mlp.net.2",      "img_mlp.net.2"),
    ("txt_mlp.net.0.proj", "txt_mlp.net.0.proj"),
    ("txt_mlp.net.2",      "txt_mlp.net.2"),
]


def main():
    log("loading diffusers transformer + extracting block 0…")
    diff_full = DiffQI.from_pretrained(
        "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
    ).to(DEV).eval()
    diff_block = diff_full.transformer_blocks[0]

    log("building fresh sgld block + copying weights…")
    sgld_block = SgldBlock(
        dim=3072, num_attention_heads=24, attention_head_dim=128,
        qk_norm="rms_norm", quant_config=None, prefix="block0",
    ).to(DEV).to(DTYPE).eval()
    copy_weights(sgld_block, diff_block)

    log("crafting deterministic shared inputs (bf16, 1×1024 image, 35 text)…")
    torch.manual_seed(0)
    S_img = 1024
    S_txt = 35
    hidden_states = torch.randn(1, S_img, 3072, dtype=DTYPE, device=DEV)
    encoder_hidden_states = torch.randn(1, S_txt, 3072, dtype=DTYPE, device=DEV)
    encoder_hidden_states_mask = torch.ones(1, S_txt, dtype=torch.bool, device=DEV)
    temb = torch.randn(1, 3072, dtype=DTYPE, device=DEV)

    diff_records: Dict[str, Any] = {}
    sgld_records: Dict[str, Any] = {}
    install_hooks(diff_block, diff_records)
    install_hooks(sgld_block, sgld_records)

    log("forward diffusers block (image_rotary_emb=None to disable RoPE)…")
    with torch.no_grad():
        d_enc_out, d_hid_out = diff_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=None,
        )
    log("forward sgld block (temb pre-SiLU'd, image_rotary_emb=None)…")
    temb_silu = F.silu(temb)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        s_enc_out, s_hid_out = sgld_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb_img_silu=temb_silu,
            temb_txt_silu=temb_silu,
            image_rotary_emb=None,
        )

    # Block-level outputs
    log("=" * 80)
    log("BLOCK-LEVEL OUTPUT DIFF")
    log("=" * 80)
    enc_stats = _stats(d_enc_out.float().cpu(), s_enc_out.float().cpu())
    hid_stats = _stats(d_hid_out.float().cpu(), s_hid_out.float().cpu())
    print(f"  encoder_out  abs_max={enc_stats['abs_max']:.4e} abs_mean={enc_stats['abs_mean']:.4e} rel_mean={enc_stats['rel_mean']:.4e}")
    print(f"  hidden_out   abs_max={hid_stats['abs_max']:.4e} abs_mean={hid_stats['abs_mean']:.4e} rel_mean={hid_stats['rel_mean']:.4e}")

    # Top-level attn module: split tuple output into img / txt halves
    log("=" * 80)
    log("ATTN MODULE OUTPUTS  (img_attn vs txt_attn outputs after to_out projections)")
    log("=" * 80)
    if "attn" in diff_records and "attn" in sgld_records:
        d_attn = diff_records["attn"]["out"]
        s_attn = sgld_records["attn"]["out"]
        if isinstance(d_attn, (tuple, list)) and isinstance(s_attn, (tuple, list)):
            d_img, d_txt = d_attn[0], d_attn[1]
            s_img, s_txt = s_attn[0], s_attn[1]
            img_st = _stats(d_img, s_img)
            txt_st = _stats(d_txt, s_txt)
            print(f"  attn.out img: {img_st}")
            print(f"  attn.out txt: {txt_st}")
        else:
            print(f"  attn out type unexpected: diff={type(d_attn)}, sgld={type(s_attn)}")

    # sgld attn.attn (USPAttention) raw joint attention output (B, S, H, D)
    log("=" * 80)
    log("SGLD attn.attn RAW JOINT ATTN OUTPUT  (only sgld has this submodule)")
    log("=" * 80)
    if "attn.attn" in sgld_records:
        s_joint = _flatten_first_tensor(sgld_records["attn.attn"]["out"])
        print(f"  sgld attn.attn out shape: {tuple(s_joint.shape)}  norm={s_joint.norm().item():.3e}")

    # save records to disk for offline analysis
    out_path = "/tmp/layer_align_records.pt"
    torch.save({"diff": diff_records, "sgld": sgld_records,
                "block_out": {"d_enc": d_enc_out.float().cpu(), "d_hid": d_hid_out.float().cpu(),
                              "s_enc": s_enc_out.float().cpu(), "s_hid": s_hid_out.float().cpu()}},
               out_path)
    log(f"records saved to {out_path}")

    # Per-module pairs
    log("=" * 80)
    log("PER-MODULE COMPARISON  (input + output of each mapped module pair)")
    log("=" * 80)
    print(f"{'diffusers':<22s}{'sgld':<22s}  "
          f"{'in: abs_max':>12s}{'abs_mean':>12s}{'rel_mean':>12s}   "
          f"{'out: abs_max':>13s}{'abs_mean':>12s}{'rel_mean':>12s}    note")

    for d_name, s_name in MAPPING:
        if d_name not in diff_records:
            print(f"{d_name:<22s}{s_name:<22s}  diffusers module not in records")
            continue
        if s_name not in sgld_records:
            print(f"{d_name:<22s}{s_name:<22s}  sgld module not in records")
            continue
        d_in = _flatten_first_tensor(diff_records[d_name]["args"]) or _flatten_first_tensor(list(diff_records[d_name]["kwargs"].values()))
        s_in = _flatten_first_tensor(sgld_records[s_name]["args"]) or _flatten_first_tensor(list(sgld_records[s_name]["kwargs"].values()))
        d_out = _flatten_first_tensor(diff_records[d_name]["out"])
        s_out = _flatten_first_tensor(sgld_records[s_name]["out"])

        in_st = _stats(d_in, s_in) if (d_in is not None and s_in is not None) else None
        out_st = _stats(d_out, s_out) if (d_out is not None and s_out is not None) else None

        if in_st and "shape_mismatch" in in_st:
            in_str = f"  shape: {in_st['shape_a']} vs {in_st['shape_b']}"
        elif in_st:
            in_str = f"  {in_st['abs_max']:>12.3e}{in_st['abs_mean']:>12.3e}{in_st['rel_mean']:>12.3e}"
        else:
            in_str = " " * 38
        if out_st and "shape_mismatch" in out_st:
            out_str = f"   shape: {out_st['shape_a']} vs {out_st['shape_b']}"
        elif out_st:
            out_str = f"   {out_st['abs_max']:>13.3e}{out_st['abs_mean']:>12.3e}{out_st['rel_mean']:>12.3e}"
        else:
            out_str = " " * 39

        # Note column: shape mismatch / shapes
        note = ""
        if out_st and isinstance(out_st.get("shape"), tuple):
            note = f"shape={out_st['shape']}"
        print(f"{d_name:<22s}{s_name:<22s}{in_str}{out_str}    {note}")


if __name__ == "__main__":
    main()
