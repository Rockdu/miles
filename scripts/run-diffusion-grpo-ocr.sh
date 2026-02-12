# ps -ef | grep train.py | grep -v grep
# nohup bash /data/zhiheng/miles/scripts/run-diffusion-grpo-ocr.sh > /data/zhiheng/miles/logs/diffusion_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# pkill -f "/data/zhiheng/miles/train_async.py"
# rollout needs 1 gpu for now, or there's going to be precision issue.
# parameter rollout-num-gpus and --rollout-num-gpus-per-engine  only makes sense in sglang diffusion case.
#!/usr/bin/env bash
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# WandB: enable if WANDB_API_KEY is present.
RUN_NAME="diffusion_grpo_$(date +%Y%m%d_%H%M%S)"
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 8
    --diffusion-log-image-interval 10
    --disable-wandb-random-suffix
  )
fi
# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# Minimal diffusion GRPO run, aligned with flow_grpo single-node settings.
# Debug toggle: set DEBUG=1 for faster iterations.
DEBUG=${DEBUG:-0}


# hf-checkpoint can be any text generation model from HuggingFace, used to generate initial prompts for diffusion model.
ARGS=(
  --train-backend fsdp
  --diffusion-train
  --rollout-function-path miles.rollout.diffusion_rollout.generate_rollout

  # ----- Data / Prompt Source -----
  --hf-checkpoint gpt2
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl"
  --input-key input

  # ----- Rollout (Sampling) -----
  --rollout-batch-size 8
  --n-samples-per-prompt 16
  --num-rollout 100000
  --diffusion-num-batches-per-epoch 2
  --diffusion-num-steps 10
  --diffusion-timestep-fraction 0.99
  --diffusion-model stabilityai/stable-diffusion-3.5-medium
  --diffusion-cfg
  --diffusion-guidance-scale 4.5
  --diffusion-noise-level 0.7
  --diffusion-height 512
  --diffusion-width 512
  --diffusion-weight-update-from-disk
  --diffusion-weight-update-from-disk-buffer-path /.cache/miles_diffusion_weights_buffer.pt

  # ----- Reward -----
  --diffusion-reward ocr:1.0
  --reward-key avg
  --disable-rewards-normalization
  --disable-grpo-std-normalization

  # ----- Training -----
  --diffusion-train-batch-size 4
  --diffusion-grad-accum-steps 4
  --diffusion-clip-range 1e-2
  --diffusion-global-std 1
  --diffusion-beta 0.04
  --lr 3e-4
  --diffusion-dtype fp32
  --global-batch-size 32

  # ----- Infra / Placement -----
  --num-gpus-per-node 4
  --actor-num-gpus-per-node 2
  --rollout-num-gpus 2
  --offload-rollout
  --rollout-num-gpus-per-engine 2
  --sglang-disable-cuda-graph
  --sglang-mem-fraction-static 0.7
  --sglang-cuda-graph-max-bs 16
)

# Override with a fast debug config (small rollout + low steps/resolution).
if [[ "${DEBUG}" == "1" ]]; then
  ARGS+=(
    --rollout-batch-size 1
    --n-samples-per-prompt 1
    --diffusion-num-batches-per-epoch 1
    --diffusion-num-steps 2
    --diffusion-height 256
    --diffusion-width 256
    --global-batch-size 1
    --diffusion-train-batch-size 1
  )
fi

python -u "${ROOT_DIR}/train_async.py" "${ARGS[@]}" "${WANDB_ARGS[@]}"

