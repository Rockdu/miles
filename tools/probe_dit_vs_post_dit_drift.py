"""Isolate DiT-internal vs post-DiT drift by recomputing noise_pred train-side
on the saved rollout dump and diffing against the dumped rollout noise_pred.

The rollout dump (saved by --diffusion-debug-mode) contains for each sample:
  - dit_trajectory.latents[t]    : DiT input latent at each timestep
  - dit_trajectory.timesteps[t]  : timestep values
  - rollout_model_outputs[t]     : CFG-combined noise_pred dumped by sgld
  - denoising_env.pos_cond_kwargs / neg_cond_kwargs : conditioning + RoPE freqs

We recompute noise_pred on the TRAIN side (diffusers DiT + miles' RoPE rebuild
+ LoRA wrap with B=0 ≡ no-op) and diff against the rollout dump.

Interpretation:
  - small noise_pred diff (≪ 1e-3) → DiT internals are aligned, the residual
    log_prob drift seen in align-check (~2.3e-5) comes from POST-DiT
    computation (scheduler step, log_prob formula, sde noise sampling).
  - large noise_pred diff (~1e-2 abs_mean, the "bf16 floor") → DiT internals
    do drift but log_prob averaging dampens it.
"""
from __future__ import annotations

import os
import sys
import time

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as DiffQI

# Reuse miles' train-side RoPE rebuild
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from miles.backends.fsdp_utils.configs.qwen_image import (
    QwenImageTrainPipelineConfig,
    _rebuild_pos_embed_freqs_on_cuda,
)
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cfg_combine(noise_pos, noise_neg, true_cfg_scale=4.0):
    """Mirror miles QwenImageTrainPipelineConfig.cfg_combine."""
    scale = true_cfg_scale
    combined = noise_neg + scale * (noise_pos - noise_neg)
    pos_norm = torch.norm(noise_pos, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    combined = combined * (pos_norm / combined_norm)
    return combined


def main():
    log(f"loading rollout dump: {DUMP_PATH}")
    d = torch.load(DUMP_PATH, weights_only=False, map_location="cpu")
    samples = d["samples"]
    log(f"  {len(samples)} samples")

    log("loading diffusers Qwen-Image transformer (bf16)…")
    model = DiffQI.from_pretrained(
        "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
    ).to(DEV).eval()

    # Apply LoRA (PEFT gaussian → A~N(0,1/r), B=0). With B=0, ΔW=0, so the
    # model is mathematically identical to the base; but the code path
    # exactly matches what miles' train side runs.
    log("wrapping with PEFT LoRA (B=0 init)…")
    cfg = LoraConfig(r=64, lora_alpha=128, target_modules=LORA_TARGETS,
                     init_lora_weights="gaussian")
    model = get_peft_model(model, cfg).eval()

    # Match miles' preprocess_model_before_fsdp: rebuild RoPE on CUDA so
    # train-side freqs match sgld rollout-side freqs bit-exactly.
    log("rebuilding RoPE freqs on CUDA…")
    _rebuild_pos_embed_freqs_on_cuda(model)

    # Iterate samples
    all_diffs = []
    for s_idx, sample in enumerate(samples):
        traj = sample["dit_trajectory"]
        env = sample["denoising_env"]
        rollout_noise_pred = sample["rollout_debug_tensors"].rollout_model_outputs.to(DEV)
        latents_traj = traj.latents.to(DEV)              # (T+1, S, D)
        timesteps = traj.timesteps.to(DEV)               # (T,)
        T = timesteps.shape[0]

        pos_kw = env.pos_cond_kwargs
        neg_kw = env.neg_cond_kwargs
        pos_enc = pos_kw.encoder_hidden_states[0].to(DEV).to(DTYPE)
        neg_enc = neg_kw.encoder_hidden_states[0].to(DEV).to(DTYPE)
        if pos_enc.dim() == 2:
            pos_enc = pos_enc.unsqueeze(0)
        if neg_enc.dim() == 2:
            neg_enc = neg_enc.unsqueeze(0)
        pos_seq_lens = pos_kw.txt_seq_lens
        neg_seq_lens = neg_kw.txt_seq_lens
        img_shapes = pos_kw.img_shapes  # [[[1, 16, 16]]] — diffusers peels [0] internally

        # Build encoder_hidden_states_mask: production rollout has B=1 with no
        # padding so mask is None (consistent with our config-side fix).
        # The diffusers DiT accepts mask=None.

        sample_diffs = []
        for t_idx in range(T):
            lat = latents_traj[t_idx].to(DTYPE)             # (S_img, D)
            if lat.dim() == 2:
                lat = lat.unsqueeze(0)                      # (1, S_img, D)
            # Match sgld's order: divide in fp32 first, then cast to bf16
            # (casting before dividing loses precision at bf16).
            timestep = (timesteps[t_idx:t_idx+1] / 1000.0).to(DTYPE)

            # Cond branch
            with torch.no_grad():
                noise_pos = model(
                    hidden_states=lat,
                    encoder_hidden_states=pos_enc,
                    encoder_hidden_states_mask=None,
                    timestep=timestep,
                    txt_seq_lens=pos_seq_lens,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                noise_neg = model(
                    hidden_states=lat,
                    encoder_hidden_states=neg_enc,
                    encoder_hidden_states_mask=None,
                    timestep=timestep,
                    txt_seq_lens=neg_seq_lens,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
            train_combined = cfg_combine(noise_pos, noise_neg, true_cfg_scale=4.0)

            # Compare against rollout's CFG-combined noise_pred for this step
            roll_combined = rollout_noise_pred[t_idx:t_idx+1]
            d_abs = (train_combined.float() - roll_combined.float()).abs()
            sample_diffs.append({
                "abs_max": d_abs.max().item(),
                "abs_mean": d_abs.mean().item(),
                "rel_mean": d_abs.mean().item() / max(roll_combined.abs().float().mean().item(), 1e-12),
                "ref_norm": roll_combined.float().norm().item(),
            })

        # Per-sample summary
        max_step = max(d["abs_max"] for d in sample_diffs)
        mean_step = sum(d["abs_mean"] for d in sample_diffs) / len(sample_diffs)
        rel_step = sum(d["rel_mean"] for d in sample_diffs) / len(sample_diffs)
        log(f"  sample {s_idx}: T={T}  abs_max={max_step:.3e}  abs_mean={mean_step:.3e}  rel_mean={rel_step:.3e}")
        if s_idx == 0:
            for ti, dd in enumerate(sample_diffs):
                log(f"    step {ti}: abs_max={dd['abs_max']:.3e}  abs_mean={dd['abs_mean']:.3e}  rel_mean={dd['rel_mean']:.3e}")
        all_diffs.append(sample_diffs)

    # Aggregate
    flat = [d for s in all_diffs for d in s]
    abs_max_all = max(d["abs_max"] for d in flat)
    abs_mean_all = sum(d["abs_mean"] for d in flat) / len(flat)
    rel_mean_all = sum(d["rel_mean"] for d in flat) / len(flat)
    print()
    print("=" * 70)
    print(f"  TRAIN-SIDE noise_pred (CFG-combined) vs ROLLOUT dumped noise_pred")
    print(f"  total samples × steps: {len(flat)}  ({len(samples)} × {len(all_diffs[0])})")
    print("=" * 70)
    print(f"  abs_max  (worst single element across all)  : {abs_max_all:.3e}")
    print(f"  abs_mean (mean across all elements)         : {abs_mean_all:.3e}")
    print(f"  rel_mean (relative to rollout |noise|.mean) : {rel_mean_all:.3e}")
    print()
    print("  Reference comparison (from align-check log_prob_mean_abs_diff): 2.28e-5")
    print("  → if noise_pred abs_mean ~ 1e-2: DiT-internal drift is dominant; log_prob averaging dampens it")
    print("  → if noise_pred abs_mean ~ 1e-5 or smaller: DiT path is bit-exact; drift comes from post-DiT")


if __name__ == "__main__":
    main()
