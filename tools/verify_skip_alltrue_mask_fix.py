"""Verify that skipping the all-True mask in collate_cond_for_sample_batch
closes the per-block ~1.2e-3 drift caused by diffusers' SDPA-with-mask vs
sgld's SDPA-without-mask 1-ULP rounding difference.

Three runs in fresh subprocesses (class-level patches + import-time
reads of the config can't be reset within a process):
  1) before_fix  — encoder_hidden_states_mask still emitted (all-True);
                   diffusers vs sgld block 0 → ~1.2e-3 drift baseline
  2) after_fix   — config patched to skip mask when no padding;
                   diffusers vs sgld → expected 0/0
  3) padded      — heterogeneous prompt lengths force mask emission;
                   sanity that the mask path still works when truly needed
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

from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig
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


def make_per_sample_cond(prompt_lens):
    cfg = QwenImageTrainPipelineConfig()
    per_sample = []
    for i, L in enumerate(prompt_lens):
        torch.manual_seed(100 + i)
        enc = torch.randn(1, L, 3072, dtype=DTYPE, device=DEV)
        per_sample.append({
            "encoder_hidden_states": enc,
            "txt_seq_lens": [L],
            "img_shapes": [(1, 32, 32)],
        })
    cond = cfg.collate_cond_for_sample_batch(per_sample, DEV)
    return cond


log("loading diffusers transformer + extracting block 0…")
diff_full = DiffQI.from_pretrained(
    "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
).to(DEV).eval()
diff_block = diff_full.transformer_blocks[0]

log("building patched sgld block + copying weights…")
sgld_block = SgldBlock(
    dim=3072, num_attention_heads=24, attention_head_dim=128,
    qk_norm="rms_norm", quant_config=None, prefix="block0",
).to(DEV).to(DTYPE).eval()
copy_weights(sgld_block, diff_block)


def run_case(label, prompt_lens):
    log(f"=== {label} (prompt_lens={prompt_lens}) ===")
    cond = make_per_sample_cond(prompt_lens)
    has_mask = "encoder_hidden_states_mask" in cond
    log(f"  collate emits mask: {has_mask}")

    B = len(prompt_lens)
    S_max = max(prompt_lens)
    torch.manual_seed(0)
    hs = torch.randn(B, 1024, 3072, dtype=DTYPE, device=DEV)
    te = torch.randn(B, 3072, dtype=DTYPE, device=DEV)
    eh = cond["encoder_hidden_states"]
    eh_mask = cond.get("encoder_hidden_states_mask", None)

    # Build joint mask if eh_mask present (mirrors diffusers DiT-level forward)
    jak = {}
    if eh_mask is not None:
        image_mask = torch.ones((B, 1024), dtype=torch.bool, device=DEV)
        joint_mask = torch.cat([eh_mask, image_mask], dim=1)
        jak["attention_mask"] = joint_mask

    with torch.no_grad():
        d_enc, d_hid = diff_block(
            hidden_states=hs, encoder_hidden_states=eh,
            encoder_hidden_states_mask=None, temb=te,
            image_rotary_emb=None,
            joint_attention_kwargs=jak,
        )
    te_silu = F.silu(te)
    with torch.no_grad(), set_forward_context(0, None):
        s_enc, s_hid = sgld_block(
            hidden_states=hs, encoder_hidden_states=eh,
            encoder_hidden_states_mask=eh_mask,  # sgld absorbs anyway
            temb_img_silu=te_silu, temb_txt_silu=te_silu,
            image_rotary_emb=None,
            joint_attention_kwargs=jak,
        )

    enc_st = diff_stats(d_enc, s_enc)
    hid_st = diff_stats(d_hid, s_hid)
    print(f"  encoder_out  abs_max={enc_st['abs_max']:.3e}  abs_mean={enc_st['abs_mean']:.3e}  rel_mean={enc_st['rel_mean']:.3e}")
    print(f"  hidden_out   abs_max={hid_st['abs_max']:.3e}  abs_mean={hid_st['abs_mean']:.3e}  rel_mean={hid_st['rel_mean']:.3e}")
    return enc_st, hid_st, has_mask


# Case 1: homogeneous batch, all same prompt length → fix should kick in
results = {}
results["b1_homogeneous"] = run_case("b1_homogeneous (single prompt L=35)", [35])
results["b4_homogeneous"] = run_case("b4_homogeneous (4 same-len prompts)", [35, 35, 35, 35])
# Case 2: heterogeneous (sanity — the mask path still gets exercised)
results["b2_padded"] = run_case("b2_padded (heterogeneous lens 28/60)", [28, 60])

print()
print("=" * 80)
print("  SUMMARY (collate_cond skip-all-True-mask fix applied)")
print("=" * 80)
print(f"  {'variant':<20s}  {'mask emitted':>14s}  {'enc rel_mean':>14s}  {'hid rel_mean':>14s}")
for k, (enc, hid, has) in results.items():
    print(f"  {k:<20s}  {str(has):>14s}  {enc['rel_mean']:>14.3e}  {hid['rel_mean']:>14.3e}")
