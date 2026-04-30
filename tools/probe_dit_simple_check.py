"""Minimal probe: run diffusers DiT (no LoRA, no CFG) on the rollout dump's
cond-branch only, compare against rollout's CFG-combined output.

Goal: rule out probe-level bugs (LoRA wrap, CFG combine formula, etc.) by
isolating the simplest possible path.

Three configs:
  A. cond-only, no LoRA, no RoPE rebuild  → pure diffusers default
  B. cond-only, no LoRA, with RoPE rebuild → check RoPE alignment
  C. cond+CFG, with LoRA + RoPE rebuild  → matches main probe

For each, dump first sample, first timestep, both abs_max and abs_mean of
(train_noise - rollout_noise[t=0]).
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
from peft import LoraConfig, get_peft_model

DEV = torch.device("cuda:0")
DTYPE = torch.bfloat16
DUMP_PATH = "/root/diffusion-rl/miles/.claude/worktrees/layer-alignment-diffusers-vs-sgld/logs/align_mg1_sdewin_20260430_005005/rollout_dump_0.pt"

LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "img_mlp.net.0.proj", "img_mlp.net.2",
    "txt_mlp.net.0.proj", "txt_mlp.net.2",
]


def cfg_combine(noise_pos, noise_neg, scale=4.0):
    combined = noise_neg + scale * (noise_pos - noise_neg)
    pos_norm = torch.norm(noise_pos, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (pos_norm / combined_norm)


def make_model(use_lora: bool, rebuild_rope: bool):
    m = DiffQI.from_pretrained(
        "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
    ).to(DEV).eval()
    if use_lora:
        cfg = LoraConfig(r=64, lora_alpha=128, target_modules=LORA_TARGETS,
                         init_lora_weights="gaussian")
        m = get_peft_model(m, cfg).eval()
    if rebuild_rope:
        _rebuild_pos_embed_freqs_on_cuda(m)
    return m


def stats(a, b):
    d = (a.float() - b.float()).abs()
    rel = d.mean().item() / max(b.float().abs().mean().item(), 1e-12)
    return f"abs_max={d.max().item():.3e}  abs_mean={d.mean().item():.3e}  rel_mean={rel:.3e}"


def main():
    print("loading dump…")
    d = torch.load(DUMP_PATH, weights_only=False, map_location="cpu")
    sample = d["samples"][0]
    traj = sample["dit_trajectory"]
    env = sample["denoising_env"]
    lat = traj.latents[0].to(DEV).to(DTYPE).unsqueeze(0)  # (1, 256, 64)
    timestep = traj.timesteps[0:1].to(DEV).to(DTYPE) / 1000.0
    pos_kw = env.pos_cond_kwargs
    neg_kw = env.neg_cond_kwargs
    pos_enc = pos_kw.encoder_hidden_states[0].to(DEV).to(DTYPE).unsqueeze(0)
    neg_enc = neg_kw.encoder_hidden_states[0].to(DEV).to(DTYPE).unsqueeze(0)
    pos_seq_lens = pos_kw.txt_seq_lens
    neg_seq_lens = neg_kw.txt_seq_lens
    img_shapes = pos_kw.img_shapes
    rollout_t0 = sample["rollout_debug_tensors"].rollout_model_outputs[0:1].to(DEV)

    common = dict(
        hidden_states=lat, encoder_hidden_states_mask=None,
        timestep=timestep, img_shapes=img_shapes, return_dict=False,
    )

    for label, use_lora, rebuild_rope, with_cfg in [
        ("A: cond-only, no LoRA, no RoPE rebuild", False, False, False),
        ("B: cond-only, no LoRA, RoPE rebuild",    False, True,  False),
        ("C: cond-only, LoRA, RoPE rebuild",       True,  True,  False),
        ("D: cond+CFG, no LoRA, RoPE rebuild",     False, True,  True),
        ("E: cond+CFG, LoRA, RoPE rebuild",        True,  True,  True),
    ]:
        print()
        print(f"=== {label} ===")
        torch.manual_seed(0)  # reproducible LoRA init
        m = make_model(use_lora, rebuild_rope)
        with torch.no_grad():
            np_pos = m(encoder_hidden_states=pos_enc, txt_seq_lens=pos_seq_lens, **common)[0]
            if with_cfg:
                np_neg = m(encoder_hidden_states=neg_enc, txt_seq_lens=neg_seq_lens, **common)[0]
                out = cfg_combine(np_pos, np_neg, scale=4.0)
            else:
                # Compare cond-only against rollout cond-only — but we don't have
                # raw cond noise dumped. So compare against rollout (post-CFG)
                # to see how close cond-only naturally is. Useful as a relative
                # number but the absolute interpretation differs.
                out = np_pos
        print(f"  vs rollout_t0 ({'post-CFG' if with_cfg else 'cond-vs-postCFG-rough'}): {stats(out, rollout_t0)}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
