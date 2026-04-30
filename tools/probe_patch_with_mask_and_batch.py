"""Probe whether the existing miles qwen_image_patch.py keeps diffusers↔sgld
bit-exact in production-like settings: multi-batch and non-trivial encoder
mask. Single-batch unmasked already verified at 0/0 in this worktree.

Variants:
  b1_nomask     — b=1, encoder mask all True (control; expect 0/0)
  b1_mask       — b=1, encoder mask with trailing False positions
  b2_mask       — b=2 (different prompt lengths, padded), mask
  b4_mask       — b=4
  b2_nomask     — b=2 with mask all True (isolates batched gemm vs single)

The existing patch is applied to sgld for ALL variants.
Reports block-level encoder_out + hidden_out diffs.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sgld_minimal_init import init_minimal

init_minimal()

# Apply existing parity patch (sgld → diffusers) BEFORE constructing sgld block.
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
DIM = 3072
HEADS = 24
HEAD_DIM = 128
S_IMG = 1024


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def copy_weights(sgld_block, diff_block):
    src = dict(diff_block.named_parameters())
    dst = dict(sgld_block.named_parameters())
    for n, p in dst.items():
        cand = [n, n.replace(".norm.", ".")]
        for c in cand:
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
        "norm_diff_rel": abs(af.norm().item() - bf.norm().item()) / max(af.norm().item(), 1e-12),
    }


def make_inputs(B: int, mask_lens: list[int], dtype, seed: int = 0):
    """B-batched inputs with per-row text-mask length (S_txt_max = max mask_lens).
    Each row's mask is True for first mask_lens[i] positions, then False.
    """
    assert len(mask_lens) == B
    S_txt = max(mask_lens)
    torch.manual_seed(seed)
    hidden_states = torch.randn(B, S_IMG, DIM, dtype=dtype, device=DEV)
    encoder_hidden_states = torch.randn(B, S_txt, DIM, dtype=dtype, device=DEV)
    mask = torch.zeros(B, S_txt, dtype=torch.bool, device=DEV)
    for i, L in enumerate(mask_lens):
        mask[i, :L] = True
    temb = torch.randn(B, DIM, dtype=dtype, device=DEV)
    return hidden_states, encoder_hidden_states, mask, temb, S_txt


def run_pair(label, B, mask_lens, *, encoder_hidden_states_mask_kw=True):
    """Run diffusers + sgld blocks on the same inputs. Return diff stats.

    encoder_hidden_states_mask_kw=False simulates production diffusers DiT
    behaviour where the block forward gets ``encoder_hidden_states_mask=None``
    and the joint mask travels through ``joint_attention_kwargs``. We set
    True here to reproduce the in-block mask path used by the existing tests.
    """
    log(f"=== {label} (B={B}, mask_lens={mask_lens}) ===")
    hidden_states, encoder_hidden_states, mask, temb, S_txt = make_inputs(B, mask_lens, DTYPE)
    log(f"    shapes: hs={tuple(hidden_states.shape)} enc={tuple(encoder_hidden_states.shape)} mask={tuple(mask.shape)} S_txt={S_txt}")

    # Build joint_attention_kwargs the way diffusers' DiT-level forward does
    # when mask is non-None (see transformer_qwenimage.py:929-937):
    image_mask = torch.ones((B, S_IMG), dtype=torch.bool, device=DEV)
    joint_mask = torch.cat([mask, image_mask], dim=1)
    joint_attention_kwargs = {"attention_mask": joint_mask}

    with torch.no_grad():
        d_enc, d_hid = diff_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,  # production: None when joint mask is used
            temb=temb,
            image_rotary_emb=None,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    temb_silu = F.silu(temb)
    with torch.no_grad(), set_forward_context(0, None):
        s_enc, s_hid = sgld_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=mask,  # sgld processor takes the raw text mask
            temb_img_silu=temb_silu,
            temb_txt_silu=temb_silu,
            image_rotary_emb=None,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    enc_st = diff_stats(d_enc, s_enc)
    hid_st = diff_stats(d_hid, s_hid)
    print(f"    encoder_out  abs_max={enc_st['abs_max']:.3e}  abs_mean={enc_st['abs_mean']:.3e}  rel_mean={enc_st['rel_mean']:.3e}")
    print(f"    hidden_out   abs_max={hid_st['abs_max']:.3e}  abs_mean={hid_st['abs_mean']:.3e}  rel_mean={hid_st['rel_mean']:.3e}")
    return enc_st, hid_st


# ---- build models once, reuse across variants ----
log("loading diffusers transformer + extracting block 0…")
diff_full = DiffQI.from_pretrained(
    "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
).to(DEV).eval()
diff_block = diff_full.transformer_blocks[0]

log("building patched sgld block + copying weights…")
sgld_block = SgldBlock(
    dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
    qk_norm="rms_norm", quant_config=None, prefix="block0",
).to(DEV).to(DTYPE).eval()
copy_weights(sgld_block, diff_block)

# ---- variants ----
results = {}

# Control: b=1 with all-True mask (effectively no mask).
results["b1_nomask"] = run_pair("b1_nomask  (control: all-True mask)", B=1, mask_lens=[35])

# b=1 with non-trivial mask (40 positions, only first 28 valid).
results["b1_mask"] = run_pair("b1_mask    (b=1, mask 28/40 valid)", B=1, mask_lens=[28])

# b=2 with mixed mask lengths (different prompts in batch → padding).
results["b2_mask"] = run_pair("b2_mask    (b=2, mask lens [28, 60])", B=2, mask_lens=[28, 60])

# b=2 with all-True mask — isolates pure batched-gemm effect from masking.
results["b2_allTrue"] = run_pair("b2_allTrue (b=2, all-True mask)", B=2, mask_lens=[40, 40])

# b=4 with mixed mask lengths (production-like microgroup-size).
results["b4_mask"] = run_pair(
    "b4_mask    (b=4, mask lens [28, 60, 35, 50])", B=4, mask_lens=[28, 60, 35, 50]
)

# Summary
print()
print("=" * 80)
print(f"  SUMMARY (existing qwen_image_patch.py applied to sgld)")
print("=" * 80)
print(f"  {'variant':<14s}  {'enc rel_mean':>14s}  {'hid rel_mean':>14s}  {'enc abs_max':>14s}  {'hid abs_max':>14s}")
for k, (enc, hid) in results.items():
    print(f"  {k:<14s}  {enc['rel_mean']:>14.3e}  {hid['rel_mean']:>14.3e}  {enc['abs_max']:>14.3e}  {hid['abs_max']:>14.3e}")
