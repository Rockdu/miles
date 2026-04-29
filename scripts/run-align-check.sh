#!/usr/bin/env bash
# Align-check fast launch: 32 prompts × 1 sample on a single GPU.
# Purpose: collect train↔rollout noise_pred / log_prob diffs in one rollout
# under frozen weights. Knobs (apply-qwen-image-sgl-d-patch, fsdp-cfg-batching,
# sample-microbatch, etc.) can be flipped via env vars below.
#
# Outputs (under logs/$RUN_NAME):
#   - rollout_dump_0.pt   per-sample debug dump (load via --load-debug-rollout-data
#                          to skip rollout in subsequent training-only re-runs)
#   - wandb run with train/align/* metrics

# NOTE: do NOT pkill sgl/ray/python broadly here — this script may run on a
# shared box where another rockdu training is using a different GPU. Rely on
# CUDA_VISIBLE_DEVICES + a per-run RUN_NAME for isolation.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="${RUN_NAME:-align_check_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"

# Knobs (export to override before sourcing)
APPLY_PATCH_FLAG="${APPLY_PATCH_FLAG:---apply-qwen-image-sgl-d-patch}"
CFG_BATCHING_FLAG="${CFG_BATCHING_FLAG:-}"            # set to --fsdp-cfg-batching for joint
LOAD_DUMP="${LOAD_DUMP:-}"                            # set to a path template to skip rollout
EXTRA_ARGS=(${EXTRA_ARGS:-})

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --disable-wandb-random-suffix
  )
fi

python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

LOAD_FLAGS=()
SAVE_FLAGS=()
if [[ -n "${LOAD_DUMP}" ]]; then
  LOAD_FLAGS=(
    --load-debug-rollout-data "${LOAD_DUMP}"
    --debug-train-only
  )
else
  SAVE_FLAGS=(
    --save-debug-rollout-data "${SAVE_DIR}/rollout_dump_{rollout_id}.pt"
    --diffusion-debug-mode
  )
fi

python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Qwen/Qwen-Image \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 1 \
  --num-rollout 1 \
  --diffusion-microgroup-size 8 \
  --micro-batch-size-sample 1 \
  --micro-batch-size-tstep 10 \
  --diffusion-train-iter-order sample_major \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 1 \
  --colocate \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --diffusion-init-lora-weight gaussian \
  --lr 0 \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 0 \
  --debug-skip-optimizer-step \
  --use-miles-router \
  --sglang-server-concurrency 4 \
  --update-weight-buffer-size 2147483648 \
  --diffusion-model Qwen/Qwen-Image \
  --diffusion-reward ocr:1.0 \
  --ocr-num-workers 1 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type ocr \
  --fsdp-master-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 10 \
  --num-steps-per-rollout 1 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-true-cfg-scale 4.0 \
  --diffusion-noise-level 1.2 \
  --diffusion-height 256 \
  --diffusion-width 256 \
  ${APPLY_PATCH_FLAG} \
  ${CFG_BATCHING_FLAG} \
  "${SAVE_FLAGS[@]}" \
  "${LOAD_FLAGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${WANDB_ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/run.log"
