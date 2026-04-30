#!/usr/bin/env bash
# Align-check fast launch: 32 prompts × 1 sample on a single GPU, frozen
# weights, --diffusion-debug-mode. Knobs via env vars below.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="${RUN_NAME:-align_check_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"

# Knobs (export to override before sourcing)
APPLY_PATCH_FLAG="${APPLY_PATCH_FLAG---apply-qwen-image-sgl-d-patch}"
CFG_BATCHING_FLAG="${CFG_BATCHING_FLAG-}"            # set to --fsdp-cfg-batching for joint
USE_LORA="${USE_LORA-1}"                              # set to 0 to disable LoRA
USE_GRAD_CKPT="${USE_GRAD_CKPT-1}"                    # set to 0 to disable gradient checkpointing
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE-32}"
MICROGROUP_SIZE="${MICROGROUP_SIZE-1}"
# microgroup batches same-prompt samples — n_samples_per_prompt must be >= mg
# for the microgroup to actually run at batch=mg. Defaults to mg.
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT-${MICROGROUP_SIZE}}"
TSTEP_MB="${TSTEP_MB-1}"
SAMPLE_MB="${SAMPLE_MB-1}"
LOAD_DUMP="${LOAD_DUMP:-}"                            # set to a path template to skip rollout
EXTRA_ARGS=(${EXTRA_ARGS:-})

LORA_FLAGS=()
if [[ "${USE_LORA}" == "1" ]]; then
  LORA_FLAGS=(
    --use-lora
    --lora-rank 64
    --lora-alpha 128
    --diffusion-init-lora-weight gaussian
  )
fi
GC_FLAGS=()
if [[ "${USE_GRAD_CKPT}" == "1" ]]; then
  GC_FLAGS=(--gradient-checkpointing)
fi

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
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-rollout 1 \
  --diffusion-microgroup-size "${MICROGROUP_SIZE}" \
  --micro-batch-size-sample "${SAMPLE_MB}" \
  --micro-batch-size-tstep "${TSTEP_MB}" \
  --diffusion-train-iter-order timestep_major \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 1 \
  --colocate \
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
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window \
  --diffusion-sde-window-size 2 \
  --diffusion-sde-window-range 0,5 \
  --diffusion-height 256 \
  --diffusion-width 256 \
  ${APPLY_PATCH_FLAG} \
  ${CFG_BATCHING_FLAG} \
  "${LORA_FLAGS[@]}" \
  "${GC_FLAGS[@]}" \
  "${SAVE_FLAGS[@]}" \
  "${LOAD_FLAGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${WANDB_ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/run.log"
