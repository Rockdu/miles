"""Replay a saved checkpoint for train/rollout divergence forensics.

Phase 1 (generate): load a weights-only dist-ckpt, generate a single 32k
sample through the normal colocate rollout, and save the full Sample list
(tokens + images + rollout_log_probs) via --save-debug-rollout-data.

Phase 2 (megatron_dump): load the same ckpt plus the saved samples
(--load-debug-rollout-data -> debug_train_only, no sglang), and run the
fwd_only (old_log_probs) pass with the dumper enabled — one clean forward
over the fixed sequence, dumping per-layer tensors for the comparator.

Both phases run TP2/DP1 (2 GPUs) with the optimizer disabled: weights are
never updated, so the dumped tensors describe exactly the loaded checkpoint.

Usage (inside the miles container):
  python examples/mathvision_vlm_repro/replay_ckpt_dump.py --phase generate \
      --ckpt-iter-dir /scratch/ckpts/iter_0000107
  python examples/mathvision_vlm_repro/replay_ckpt_dump.py --phase megatron_dump \
      --ckpt-iter-dir /scratch/ckpts/iter_0000107
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ReplayArgs(U.ExecuteTrainConfig):
    phase: Literal["generate", "megatron_dump"] = "generate"
    # Point at a specific iter_XXXXXXX directory (recognized by the iter_\d{7}
    # name, bypassing latest_checkpointed_iteration.txt).
    ckpt_iter_dir: str = ""
    replay_root: str = "/scratch/replay"
    model_name: str = "Qwen3.5-9B"
    megatron_model_type: str = "qwen3.5-9B"
    hardware: Literal["H200", "B300", "H100"] = "H200"
    num_gpus_per_node: int = 2  # TP2/DP1
    rollout_max_response_len: int = 32768
    num_samples: int = 1
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    # Dumper knobs for phase 2 (fwd_only). Filter keeps volume sane; empty = all.
    dumper_filter: str = ""
    extra_args: str = ""


def tag_of(args: ReplayArgs) -> str:
    return os.path.basename(args.ckpt_iter_dir.rstrip("/")) or "unknown_iter"


def execute(args: ReplayArgs):
    assert args.ckpt_iter_dir, "--ckpt-iter-dir is required"
    out_dir = f"{args.replay_root}/{tag_of(args)}"
    U.exec_command(f"mkdir -p {out_dir}")

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--load {args.ckpt_iter_dir} "
        "--no-load-optim "
        "--no-load-rng "
        "--finetune "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/mathvision/train.parquet "
        "--input-key problem "
        "--label-key answer "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking": true}' """
        "--rollout-shuffle "
        "--rm-type math "
        "--custom-generate-function-path examples.geo3k_vlm.rollout.generate "
        "--num-rollout 1 "
        f"--rollout-batch-size {args.num_samples} "
        "--n-samples-per-prompt 1 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 0.95 "
        "--rollout-top-k 20 "
        f"--global-batch-size {args.num_samples} "
        """--multimodal-keys '{"image": "images"}' """
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--qkv-format bshd "
        "--micro-batch-size 1 "
        "--log-probs-chunk-size 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )

    # Optimizer disabled: forward/backward runs but weights never move, so the
    # dumps describe the checkpoint itself. Also removes Adam from memory.
    optimizer_args = "--optimizer adam --lr 0.0 --lr-decay-style constant --debug-disable-optimizer "

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "
    if args.hardware == "B300":
        sglang_args += "--sglang-attention-backend trtllm_mha "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        f"--attention-backend {'auto' if args.hardware == 'B300' else 'flash'} "
        "--megatron-to-hf-mode bridge "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    if args.phase == "generate":
        phase_args = f"--save-debug-rollout-data {out_dir}/rollout_{{rollout_id}}.pt "
    else:  # megatron_dump
        filter_kv = f"filter='{args.dumper_filter}' " if args.dumper_filter else ""
        phase_args = (
            f"--load-debug-rollout-data {out_dir}/rollout_{{rollout_id}}.pt "
            f"--dumper-dir {out_dir}/dumps "
            f"--dumper-fwd-only enable=true {filter_kv}"
            "--dumper-source-patcher-config-train "
            "examples/mathvision_vlm_repro/dump_patches/megatron_gdn.yaml "
            "--dumper-fwd-bwd enable=false "
            "--dumper-inference enable=false "
        )

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{perf_args} {sglang_args} {misc_args} {phase_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={"MILES_FREEZE_VISION_MODEL": "1"},
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ReplayArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
