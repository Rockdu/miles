"""精确复现 flowGRPO 的 rollout↔train log_prob 计算路径，看是不是真的 bit-exact。

flowGRPO rollout 路径（pipeline_with_logprob.py 184-195）:
  - sde_step_with_logprob(model_output.float(), sample.float(), prev_sample=None)
    → 内部 fp32 算 prev_sample = mean + std*sqrt(-dt)*variance_noise
    → log_prob 用这个 fp32 prev_sample 算
  - latents = (returned fp32 prev_sample).to(bf16)  ← cast 回 bf16
  - all_latents.append(bf16 latents)
  - all_log_probs.append(fp32 log_prob)  ← log_prob 是 cast **之前** 算的

flowGRPO train 路径（train_qwenimage.py 308-313）:
  - sde_step_with_logprob(noise_pred.float(), sample["latents"].float(),
                          prev_sample=sample["next_latents"].float())  ← bf16 → fp32 复活
  - log_prob 用这个 bf16-cast-back 的 prev_sample 算

如果 flowGRPO 训推「严格一致」，则两边 log_prob 应该 bit-exact。
"""
from __future__ import annotations

import math
import torch

# 直接 inline flowGRPO 的 sde_step_with_logprob 公式（去 scheduler 依赖，纯数学）
def flowgrpo_log_prob(model_output, sample, prev_sample, sigma, sigma_prev, noise_level):
    """精确复现 sd3_sde_with_logprob.py 的 sde 分支。"""
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma_max_ref = sigma  # for σ != 1 case, doesn't matter; use sigma itself
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max_ref, sigma))) * noise_level

    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is None:
        # rollout 分支：自己采 variance_noise 算 prev_sample
        variance_noise = torch.randn_like(model_output, dtype=torch.float32)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    # 模拟 Qwen-Image latent shape: (B=1, S=256, D=64)
    sample_bf16 = torch.randn(1, 256, 64, device=device, dtype=torch.bfloat16)
    model_output_bf16 = torch.randn(1, 256, 64, device=device, dtype=torch.bfloat16) * 0.5
    sigma = torch.tensor(0.85, device=device, dtype=torch.float32)
    sigma_prev = torch.tensor(0.78, device=device, dtype=torch.float32)
    noise_level = 1.2

    print("=== flowGRPO ROLLOUT 路径 ===")
    # rollout：prev_sample=None，内部采 variance_noise，log_prob 用 fp32 prev_sample 算
    torch.manual_seed(42)  # 固定 variance_noise 采样
    prev_sample_fp32_rollout, log_prob_rollout = flowgrpo_log_prob(
        model_output_bf16, sample_bf16, prev_sample=None,
        sigma=sigma, sigma_prev=sigma_prev, noise_level=noise_level,
    )
    print(f"  prev_sample dtype: {prev_sample_fp32_rollout.dtype}")
    print(f"  log_prob:          {log_prob_rollout.item():.10f}")

    # rollout 之后：cast prev_sample 到 bf16 存起来
    prev_sample_bf16_stored = prev_sample_fp32_rollout.to(torch.bfloat16)
    print(f"  stored next_latent dtype: {prev_sample_bf16_stored.dtype}")
    print(f"  bf16 quantization on prev_sample:")
    diff = (prev_sample_fp32_rollout - prev_sample_bf16_stored.float()).abs()
    print(f"    abs_max={diff.max().item():.3e}  abs_mean={diff.mean().item():.3e}")

    print()
    print("=== flowGRPO TRAIN 路径 ===")
    # train: prev_sample = bf16(stored).float() ← bf16 量化损失带回来
    prev_sample_train = prev_sample_bf16_stored.float()
    _, log_prob_train = flowgrpo_log_prob(
        model_output_bf16, sample_bf16, prev_sample=prev_sample_train,
        sigma=sigma, sigma_prev=sigma_prev, noise_level=noise_level,
    )
    print(f"  log_prob:          {log_prob_train.item():.10f}")

    print()
    print("=== 对比 ===")
    log_prob_diff = (log_prob_rollout - log_prob_train).abs().item()
    print(f"  |log_prob_rollout - log_prob_train| = {log_prob_diff:.3e}")
    if log_prob_diff < 1e-10:
        print("  → bit-exact aligned ✓")
    elif log_prob_diff < 1e-6:
        print("  → 接近 fp32 ulp 噪底（用户口中的「严格一致」）")
    else:
        print("  → 显著差异，bf16 量化路径有影响")


if __name__ == "__main__":
    main()
