"""Aligned variant of tools/layer_align_diff_vs_sgld.py.

Same harness, plus monkey-patches that force the two implementations onto
identical numeric paths for known-divergent boundaries:

  Alignment 1: disable sgld's fused_inplace_qknorm (forces apply_qk_norm
               to go through RMSNorm.forward, which is hookable + closer
               to diffusers' RMSNorm path).

  Alignment 2: SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 — RMSNorm uses
               forward_native (fully fp32 internal compute).

Reports the same per-module diff table so we can see how much the
asymmetric image attention drift drops once these two are aligned.
"""
from __future__ import annotations

import os
import sys

os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sgld_minimal_init import init_minimal

init_minimal()

# Monkey-patch can_use_fused_inplace_qknorm to always return False, forcing
# apply_qk_norm to fall through to the q_norm / k_norm forward call path.
import sglang.multimodal_gen.runtime.layers.layernorm as sld_ln
sld_ln.can_use_fused_inplace_qknorm = lambda head_dim, dtype: False
print("[align] disabled fused_inplace_qknorm")

# Disable fused QK norm + RoPE path too (cos_sin_cache=None means we hit
# apply_qk_norm anyway, so this is belt-and-suspenders).
os.environ["SGLANG_ENABLE_FUSED_QKNORM_ROPE"] = "0"

import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as DiffQI

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformerBlock as SgldBlock,
)

DEV = torch.device("cuda:0")
DTYPE = torch.bfloat16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def copy_weights(sgld_block, diff_block):
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
    log(f"copy_weights: matched {matched}/{len(dst)}; missing={missing[:5]}")
    return matched, missing


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", torch.float32)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


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
        "norm_diff_rel": abs(a_norm - b_norm) / max(a_norm, 1e-12),
    }


def _get_in(rec):
    a = _flatten_first_tensor(rec.get("args"))
    if a is not None:
        return a
    return _flatten_first_tensor(rec.get("kwargs"))


def _get_out(rec):
    return _flatten_first_tensor(rec.get("out"))


MAPPING = [
    ("img_mod.1",          "img_mod.1",          "img modulation Linear"),
    ("txt_mod.1",          "txt_mod.1",          "txt modulation Linear"),
    ("attn.to_q",          "attn.to_q",          "img Q projection"),
    ("attn.to_k",          "attn.to_k",          "img K projection"),
    ("attn.to_v",          "attn.to_v",          "img V projection"),
    ("attn.add_q_proj",    "attn.add_q_proj",    "txt Q projection"),
    ("attn.add_k_proj",    "attn.add_k_proj",    "txt K projection"),
    ("attn.add_v_proj",    "attn.add_v_proj",    "txt V projection"),
    ("attn.norm_q",        "attn.norm_q",        "img Q RMSNorm  (now hookable on sgld)"),
    ("attn.norm_k",        "attn.norm_k",        "img K RMSNorm"),
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

    log("crafting deterministic shared inputs…")
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

    log("forward diffusers block…")
    with torch.no_grad():
        d_enc_out, d_hid_out = diff_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=None,
        )
    log("forward sgld block…")
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

    print("=" * 110)
    print("BLOCK-LEVEL OUTPUT DIFF (after alignments)")
    print("=" * 110)
    enc_st = _stats(d_enc_out.float().cpu(), s_enc_out.float().cpu())
    hid_st = _stats(d_hid_out.float().cpu(), s_hid_out.float().cpu())
    print(f"  encoder_out  abs_max={enc_st['abs_max']:.3e}  abs_mean={enc_st['abs_mean']:.3e}  rel_mean={enc_st['rel_mean']:.3e}  norm_diff_rel={enc_st['norm_diff_rel']:.3e}")
    print(f"  hidden_out   abs_max={hid_st['abs_max']:.3e}  abs_mean={hid_st['abs_mean']:.3e}  rel_mean={hid_st['rel_mean']:.3e}  norm_diff_rel={hid_st['norm_diff_rel']:.3e}")

    print()
    print("=" * 110)
    print("ATTN MODULE OUTPUT")
    print("=" * 110)
    d_attn = diff_records["attn"]["out"]
    s_attn = sgld_records["attn"]["out"]
    img_st = _stats(d_attn[0], s_attn[0])
    txt_st = _stats(d_attn[1], s_attn[1])
    print(f"  img stream rel_mean={img_st['rel_mean']:.3e}  abs_max={img_st['abs_max']:.3e}")
    print(f"  txt stream rel_mean={txt_st['rel_mean']:.3e}  abs_max={txt_st['abs_max']:.3e}")
    print(f"  ASYMMETRY ratio (img/txt) = {img_st['rel_mean']/max(txt_st['rel_mean'], 1e-30):.1f}")

    print()
    print("=" * 110)
    print(f"{'diffusers':<22s}{'sgld':<22s}  in: {'abs_max':>10s}{'abs_mean':>10s}{'rel_mean':>10s}{'norm_drel':>10s}   out: {'abs_max':>10s}{'abs_mean':>10s}{'rel_mean':>10s}{'norm_drel':>10s}    note")
    print("=" * 110)
    for d_name, s_name, note in MAPPING:
        if d_name not in diff_records:
            print(f"{d_name:<22s}{s_name:<22s}  diffusers module not in records ({note})")
            continue
        if s_name not in sgld_records:
            print(f"{d_name:<22s}{s_name:<22s}  sgld module not in records ({note})")
            continue
        in_st = _stats(_get_in(diff_records[d_name]), _get_in(sgld_records[s_name]))
        out_st = _stats(_get_out(diff_records[d_name]), _get_out(sgld_records[s_name]))
        print(f"{d_name:<22s}{s_name:<22s}  in: {_fmt(in_st)}   out: {_fmt(out_st)}    {note}")


if __name__ == "__main__":
    main()
