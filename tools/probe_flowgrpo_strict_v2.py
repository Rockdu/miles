"""直接用 flowGRPO 自己的 sde_step_with_logprob + 真实 FlowMatchEulerDiscreteScheduler，
精确复现 flowGRPO 的 rollout↔train 路径。
"""
from __future__ import annotations

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch

# 直接 import flowGRPO 原版函数
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "flow_grpo")))
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def main():
    device = torch.device("cuda:0")
    DTYPE = torch.bfloat16

    # 真实的 Qwen-Image scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.15,
        use_dynamic_shifting=True,
    )
    # 触发 sigma init
    scheduler.set_timesteps(num_inference_steps=10, mu=0.5, device=device)
    print("scheduler.sigmas:", scheduler.sigmas)
    print("scheduler.timesteps:", scheduler.timesteps)

    # 模拟 rollout 第一个 sde 步: i=0, t=timesteps[0]
    torch.manual_seed(0)
    latents_bf16 = torch.randn(1, 256, 64, device=device, dtype=DTYPE)
    noise_pred_bf16 = torch.randn(1, 256, 64, device=device, dtype=DTYPE) * 0.5

    t = scheduler.timesteps[0]  # 真实 timestep

    print()
    print("=== flowGRPO ROLLOUT 路径 ===")
    # 严格按 rollout 调用
    torch.manual_seed(42)
    latents_dtype = latents_bf16.dtype
    prev_sample_rollout, log_prob_rollout, prev_latents_mean_rollout, std_dev_t_rollout = sde_step_with_logprob(
        scheduler,
        noise_pred_bf16.float(),
        t.unsqueeze(0).repeat(latents_bf16.shape[0]),
        latents_bf16.float(),
        noise_level=1.2,
    )
    print(f"  prev_sample dtype: {prev_sample_rollout.dtype}")
    print(f"  log_prob: {log_prob_rollout.item():.10f}")

    # cast 回 bf16 存
    if prev_sample_rollout.dtype != latents_dtype:
        prev_sample_stored = prev_sample_rollout.to(latents_dtype)
    else:
        prev_sample_stored = prev_sample_rollout
    print(f"  bf16 stored next_latent shape={prev_sample_stored.shape} dtype={prev_sample_stored.dtype}")
    diff_quant = (prev_sample_rollout - prev_sample_stored.float()).abs()
    print(f"  bf16 quantization on prev_sample: abs_max={diff_quant.max().item():.3e}  abs_mean={diff_quant.mean().item():.3e}")

    print()
    print("=== flowGRPO TRAIN 路径 ===")
    # 重新 set_timesteps 因为 rollout 已经把 step_index 推进了
    scheduler.set_timesteps(num_inference_steps=10, mu=0.5, device=device)
    # 严格按 train 调用：noise_pred 同样 fp32，sample = sample["latents"][:,j].float() = 同一个 latents_bf16.float()
    # （因为 sample["latents"][:, 0] 就是 rollout 第一步的输入）
    # prev_sample = sample["next_latents"][:, 0].float() = rollout 第一步的输出（bf16）.float()
    prev_sample_train, log_prob_train, prev_sample_mean_train, std_dev_t_train = sde_step_with_logprob(
        scheduler,
        noise_pred_bf16.float(),                     # 同样的 noise_pred
        t.unsqueeze(0).repeat(latents_bf16.shape[0]),
        latents_bf16.float(),                         # 同样的 sample
        prev_sample=prev_sample_stored.float(),       # bf16 → fp32 复活
        noise_level=1.2,
    )
    print(f"  log_prob: {log_prob_train.item():.10f}")

    print()
    print("=== 对比 ===")
    log_prob_diff = (log_prob_rollout - log_prob_train).abs().item()
    prev_sample_mean_diff = (prev_latents_mean_rollout - prev_sample_mean_train).abs().max().item()
    print(f"  prev_sample_mean abs_max diff (should be 0): {prev_sample_mean_diff:.3e}")
    print(f"  |log_prob_rollout - log_prob_train| = {log_prob_diff:.3e}")
    if log_prob_diff < 1e-10:
        print("  → bit-exact aligned ✓ (用户说的「严格一致」)")
    else:
        print("  → 不是 bit-exact，差异由 bf16 prev_sample round-trip 引入")


if __name__ == "__main__":
    main()
