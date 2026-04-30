"""Isolate whether SDPA itself produces asymmetric image-vs-text drift.

Loads the q/k/v tensors that went into the joint attention call (recorded
by tools/layer_align_diff_vs_sgld.py via attn.to_q/to_k/to_v outputs),
manually replays the qk_norm + concat, then runs F.scaled_dot_product_attention
twice — once via diffusers' attention_backend(NATIVE) path, once via
sgld's SDPABackend path — with the SAME post-norm joint q/k/v. Diffs the
two outputs split into image / text halves.

If both halves of the diff are tiny (~bit-exact), the asymmetric drift
seen in the per-module table comes from the qk_norm divergence (sgld's
fused_inplace_qknorm vs diffusers' RMSNorm.forward) propagating through
attention. If image half is still much larger than text half, SDPA itself
treats the two regions differently across calls — but then both diffusers
and sgld are calling THE SAME torch SDPA, so it should be deterministic
within a single Python process.

This is a sanity probe — uses identical pre-attention q/k/v on both
sides, so any asymmetry post-attention is purely due to attention path.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sgld_minimal_init import init_minimal

init_minimal()

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

DEV = torch.device("cuda:0")
DTYPE = torch.bfloat16

PATH = "/tmp/layer_align_records.pt"
data = torch.load(PATH, weights_only=False, map_location=DEV)
sgld = data["sgld"]
diff = data["diff"]


def get_out(rec):
    o = rec["out"]
    if isinstance(o, (tuple, list)):
        for v in o:
            if isinstance(v, torch.Tensor):
                return v
    return o


# Use sgld attn.to_q output as the canonical q tensor (same on both sides
# since Linears are bit-exact; we verified this in prior runs).
img_q_pre = get_out(sgld["attn.to_q"]).to(DEV).to(DTYPE)  # (1, 1024, 3072)
img_k_pre = get_out(sgld["attn.to_k"]).to(DEV).to(DTYPE)
img_v_pre = get_out(sgld["attn.to_v"]).to(DEV).to(DTYPE)
txt_q_pre = get_out(sgld["attn.add_q_proj"]).to(DEV).to(DTYPE)
txt_k_pre = get_out(sgld["attn.add_k_proj"]).to(DEV).to(DTYPE)
txt_v_pre = get_out(sgld["attn.add_v_proj"]).to(DEV).to(DTYPE)

print(f"img_q_pre shape={tuple(img_q_pre.shape)} dtype={img_q_pre.dtype}")
print(f"txt_q_pre shape={tuple(txt_q_pre.shape)} dtype={txt_q_pre.dtype}")

H, D = 24, 128
img_q = img_q_pre.unflatten(-1, (H, D))   # (B, S_img, H, D)
img_k = img_k_pre.unflatten(-1, (H, D))
img_v = img_v_pre.unflatten(-1, (H, D))
txt_q = txt_q_pre.unflatten(-1, (H, D))
txt_k = txt_k_pre.unflatten(-1, (H, D))
txt_v = txt_v_pre.unflatten(-1, (H, D))

# Skip qk_norm — use raw q/k for SDPA on both sides so we isolate SDPA only.
joint_q = torch.cat([txt_q, img_q], dim=1)  # (1, 35+1024, H, D)
joint_k = torch.cat([txt_k, img_k], dim=1)
joint_v = torch.cat([txt_v, img_v], dim=1)
print(f"joint shapes: q={tuple(joint_q.shape)}")

# Path A: diffusers default (NATIVE) → calls F.scaled_dot_product_attention
#   with (B, H, S, D) layout (transposed inside _native_attention)
def diff_native(q, k, v):
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=None,
                                         dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


# Path B: sgld's SDPABackend → same logic
def sgld_sdpa(q, k, v):
    impl = SDPAImpl(num_heads=H, head_size=D, causal=False, softmax_scale=None)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        return impl.forward(q, k, v, attn_metadata=None)


print("\nRunning diffusers-style NATIVE SDPA…")
out_a = diff_native(joint_q, joint_k, joint_v)
print("Running sgld-style SDPABackend…")
out_b = sgld_sdpa(joint_q, joint_k, joint_v)
print(f"out_a shape={tuple(out_a.shape)}, out_b shape={tuple(out_b.shape)}")

d = (out_a.float() - out_b.float()).abs()
print(f"\nFull joint output diff:")
print(f"  abs_max={d.max().item():.3e}  abs_mean={d.mean().item():.3e}  "
      f"rel_mean={d.mean().item()/out_a.abs().float().mean().item():.3e}")

# split halves
S_txt = 35
txt_a = out_a[:, :S_txt]; txt_b = out_b[:, :S_txt]
img_a = out_a[:, S_txt:]; img_b = out_b[:, S_txt:]
dt = (txt_a.float() - txt_b.float()).abs()
di = (img_a.float() - img_b.float()).abs()
print(f"  txt half  abs_max={dt.max().item():.3e}  abs_mean={dt.mean().item():.3e}  "
      f"rel_mean={dt.mean().item()/txt_a.abs().float().mean().item():.3e}")
print(f"  img half  abs_max={di.max().item():.3e}  abs_mean={di.mean().item():.3e}  "
      f"rel_mean={di.mean().item()/img_a.abs().float().mean().item():.3e}")

# control: run path A twice and compare — should be bit-exact
out_a2 = diff_native(joint_q, joint_k, joint_v)
ctrl = (out_a.float() - out_a2.float()).abs()
print(f"\n[control] same path twice: abs_max={ctrl.max().item():.3e}  abs_mean={ctrl.mean().item():.3e}")
