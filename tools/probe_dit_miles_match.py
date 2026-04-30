"""More faithful miles-actor reproduction: load fp32 (matching --fsdp-master-dtype fp32),
cast to bf16 for forward (matching MixedPrecisionPolicy(param_dtype=bf16)), no LoRA
(B=0 makes it a no-op), apply RoPE rebuild. Compare step 0 vs dump.
"""
from __future__ import annotations

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from diffusers import QwenImageTransformer2DModel as DiffQI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from miles.backends.fsdp_utils.configs.qwen_image import _rebuild_pos_embed_freqs_on_cuda

DEV = torch.device("cuda:0")
DUMP_PATH = "/root/diffusion-rl/miles/.claude/worktrees/layer-alignment-diffusers-vs-sgld/logs/align_mg1_sdewin_20260430_005005/rollout_dump_0.pt"


def cfg_combine(noise_pos, noise_neg, scale=4.0):
    combined = noise_neg + scale * (noise_pos - noise_neg)
    pos_norm = torch.norm(noise_pos, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (pos_norm / combined_norm)


def stats(a, b, label):
    d = (a.float() - b.float()).abs()
    rel = d.mean().item() / max(b.float().abs().mean().item(), 1e-12)
    return f"{label}: abs_max={d.max().item():.3e}  abs_mean={d.mean().item():.3e}  rel_mean={rel:.3e}"


def main():
    print("loading dump…")
    d = torch.load(DUMP_PATH, weights_only=False, map_location="cpu")
    sample = d["samples"][0]
    traj = sample["dit_trajectory"]
    env = sample["denoising_env"]
    pos_kw = env.pos_cond_kwargs
    neg_kw = env.neg_cond_kwargs
    rollout_t0 = sample["rollout_debug_tensors"].rollout_model_outputs[0:1].to(DEV)

    for storage_dtype, label_suffix in [
        (torch.float32, "fp32 storage → bf16 cast at forward"),
        (torch.bfloat16, "bf16 storage (load with torch_dtype=bf16)"),
    ]:
        print()
        print(f"=== {label_suffix} ===")

        m = DiffQI.from_pretrained(
            "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=storage_dtype
        ).to(DEV).eval()
        if storage_dtype == torch.float32:
            m = m.to(torch.bfloat16)  # Manual cast — mirrors FSDP MP cast at compute
        _rebuild_pos_embed_freqs_on_cuda(m)

        DTYPE = torch.bfloat16
        lat = traj.latents[0].to(DEV).to(DTYPE).unsqueeze(0)
        timestep = (traj.timesteps[0:1].to(DEV) / 1000.0).to(DTYPE)
        pos_enc = pos_kw.encoder_hidden_states[0].to(DEV).to(DTYPE).unsqueeze(0)
        neg_enc = neg_kw.encoder_hidden_states[0].to(DEV).to(DTYPE).unsqueeze(0)
        pos_seq_lens = pos_kw.txt_seq_lens
        neg_seq_lens = neg_kw.txt_seq_lens
        img_shapes = pos_kw.img_shapes

        common = dict(
            hidden_states=lat, encoder_hidden_states_mask=None,
            timestep=timestep, img_shapes=img_shapes, return_dict=False,
        )
        with torch.no_grad():
            np_pos = m(encoder_hidden_states=pos_enc, txt_seq_lens=pos_seq_lens, **common)[0]
            np_neg = m(encoder_hidden_states=neg_enc, txt_seq_lens=neg_seq_lens, **common)[0]
        out = cfg_combine(np_pos, np_neg, scale=4.0)
        print(f"  step 0 cond+CFG, no LoRA: {stats(out, rollout_t0, 'noise_pred')}")

        # Verify weight bit-exactness across the two storages
        if storage_dtype == torch.float32:
            ref_w = m.transformer_blocks[0].attn.to_q.weight.detach().clone()
            print(f"  to_q.weight dtype after manual cast: {ref_w.dtype}")
            print(f"  to_q.weight first 5: {ref_w.flatten()[:5].tolist()}")

        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
