"""单 block 带真实 RoPE freqs 的 diffusers↔sgld 对齐验证。

补 verify_skip_alltrue_mask_fix.py 留下的 gap：那个 harness 用 image_rotary_emb=None
短路 RoPE。这里塞真实的 freqs_cis（sgld 格式 cos_sin_cache (S, 128)），看 patch 后
单 block 是不是仍然 0/0。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sgld_minimal_init import init_minimal

init_minimal()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from miles.backends.fsdp_utils.models.qwen_image_patch import (
    apply_qwen_image_diffusers_parity_patches,
)
apply_qwen_image_diffusers_parity_patches()

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
    for n, p in dst.items():
        for c in [n, n.replace(".norm.", ".")]:
            if c in src and src[c].shape == p.shape:
                p.data.copy_(src[c].data.to(p.dtype))
                break


def diff_stats(a, b):
    af, bf = a.float().cpu(), b.float().cpu()
    d = (af - bf).abs()
    return {
        "abs_max": d.max().item(),
        "abs_mean": d.mean().item(),
        "rel_mean": d.mean().item() / max(af.abs().mean().item(), 1e-12),
    }


log("loading diffusers transformer + extracting block 0…")
diff_full = DiffQI.from_pretrained(
    "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
).to(DEV).eval()
diff_block = diff_full.transformer_blocks[0]

log("rebuilding pos_embed freqs on CUDA (matches train-side preprocess)…")
from miles.backends.fsdp_utils.configs.qwen_image import _rebuild_pos_embed_freqs_on_cuda
_rebuild_pos_embed_freqs_on_cuda(diff_full)

log("computing real RoPE freqs from img_shapes + txt_seq_len…")
img_shapes = [[(1, 16, 16)]]   # B=1, 16x16 latent → S_img = 256
txt_seq_len = 35
diff_freqs = diff_full.pos_embed(img_shapes, max_txt_seq_len=txt_seq_len, device=DEV)
img_freqs_complex, txt_freqs_complex = diff_freqs   # (S_img, D/2) complex64, (S_txt, D/2) complex64
log(f"  diffusers img_freqs shape={tuple(img_freqs_complex.shape)} dtype={img_freqs_complex.dtype}")
log(f"  diffusers txt_freqs shape={tuple(txt_freqs_complex.shape)} dtype={txt_freqs_complex.dtype}")

# Convert to sgld cos_sin_cache layout: (S, 128) fp32 = [real | imag]
img_cos_sin = torch.cat([img_freqs_complex.real, img_freqs_complex.imag], dim=-1).to(DEV).float()
txt_cos_sin = torch.cat([txt_freqs_complex.real, txt_freqs_complex.imag], dim=-1).to(DEV).float()
log(f"  sgld img_cos_sin shape={tuple(img_cos_sin.shape)} dtype={img_cos_sin.dtype}")

log("building patched sgld block + copying weights…")
sgld_block = SgldBlock(
    dim=3072, num_attention_heads=24, attention_head_dim=128,
    qk_norm="rms_norm", quant_config=None, prefix="block0",
).to(DEV).to(DTYPE).eval()
copy_weights(sgld_block, diff_block)

log("preparing inputs (B=1 homogeneous, no mask, real RoPE)…")
torch.manual_seed(0)
hs = torch.randn(1, 256, 3072, dtype=DTYPE, device=DEV)
eh = torch.randn(1, 35, 3072, dtype=DTYPE, device=DEV)
te = torch.randn(1, 3072, dtype=DTYPE, device=DEV)

log("running diffusers block (with real complex freqs)…")
with torch.no_grad():
    d_enc, d_hid = diff_block(
        hidden_states=hs, encoder_hidden_states=eh,
        encoder_hidden_states_mask=None, temb=te,
        image_rotary_emb=(img_freqs_complex, txt_freqs_complex),
        joint_attention_kwargs={},
    )

log("running sgld block (with sgld cos_sin_cache freqs)…")
te_silu = F.silu(te)
with torch.no_grad(), set_forward_context(0, None):
    s_enc, s_hid = sgld_block(
        hidden_states=hs, encoder_hidden_states=eh,
        encoder_hidden_states_mask=None,
        temb_img_silu=te_silu, temb_txt_silu=te_silu,
        image_rotary_emb=(img_cos_sin, txt_cos_sin),
        joint_attention_kwargs={},
    )

print()
print("=" * 70)
print("  Single block, B=1, mask=None, REAL RoPE (CUDA-rebuilt freqs)")
print("=" * 70)
enc_st = diff_stats(d_enc, s_enc)
hid_st = diff_stats(d_hid, s_hid)
print(f"  encoder_out  abs_max={enc_st['abs_max']:.3e}  abs_mean={enc_st['abs_mean']:.3e}  rel_mean={enc_st['rel_mean']:.3e}")
print(f"  hidden_out   abs_max={hid_st['abs_max']:.3e}  abs_mean={hid_st['abs_mean']:.3e}  rel_mean={hid_st['rel_mean']:.3e}")
if enc_st["abs_max"] == 0.0 and hid_st["abs_max"] == 0.0:
    print("  → bit-exact 0/0 ✓")
else:
    print("  → NOT bit-exact")
