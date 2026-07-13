"""Reproduce the Qwen3.5/3.6-VL logprob_abs_diff explosion (#ext-amazon-radixark).

Lightweight setting from Jiajun Li (2026-06-24): Qwen3.5-9B + MathVision,
32k response len, megatron bridge, frozen ViT, lr 2e-6, rollout_bs 8,
n_samples_per_prompt 8 -> train_rollout_logprob_abs_diff grows steadily
(0.006 -> 0.0216 within 100 steps). 16k response len stays flat; 32k is the
dominant trigger. Sampling params follow Xinpeng Wei's full setting
(temp 1.0 / top-p 0.95 / top-k 20 / thinking on).

Usage (inside the miles container):
  python examples/mathvision_vlm_repro/run_mathvision_qwen3_5_9b.py
  python examples/mathvision_vlm_repro/run_mathvision_qwen3_5_9b.py --mode debug_minimal
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3.5-9B"
    megatron_model_type: str = "qwen3.5-9B"
    num_gpus_per_node: int = 8
    hardware: Literal["H200", "B200", "B300", "H100"] = "B300"
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    # The repro trigger. 32k explodes, 16k stays flat — flip for the control run.
    rollout_max_response_len: int = 32768
    lr: float = 2e-6
    wandb_project: str = "miles-vlm"
    # Verify every weight transfer to the rollout engines (startup check plus a
    # snapshot/reset/resend/compare cycle after every Nth rollout's update).
    check_weight_update_interval: int = 1
    # Move Adam state to CPU. Required when the Adam states cannot shard across
    # DP (e.g. 2x80GB H100 -> TP2/DP1 leaves ~81GB/GPU for a 9B model); the
    # 4-GPU H100 config (TP2/DP2, ~45GB/GPU) fits without it.
    optimizer_cpu_offload: bool = False
    # Checkpointing for post-hoc dump/replay analysis. Weights-only (no optim/rng)
    # to keep each dist-ckpt ~20GB. save_on_diff_threshold arms per-step saves once
    # the train/rollout logprob gap crosses it, bracketing the explosion window.
    save_dir: str = ""
    save_interval: int = 50
    save_on_diff_threshold: float = 0.0


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    # Bridge mode loads the HF checkpoint directly; no torch_dist conversion needed.
    prepare_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_data.py")
    U.exec_command(
        f"[ -f {args.data_dir}/mathvision/train.parquet ] || "
        f"python {prepare_script} --out-dir {args.data_dir}/mathvision"
    )


def get_wandb_args(args: ScriptArgs) -> str:
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""
    resp_len_tag = f"{args.rollout_max_response_len // 1024}k"
    return (
        "--use-wandb "
        f"--wandb-project {args.wandb_project} "
        f"--wandb-group qwen3.5-9b-mathvision-{resp_len_tag}-repro "
        f"--wandb-key '{os.environ['WANDB_API_KEY']}' "
    )


def execute(args: ScriptArgs):
    debug = args.mode == "debug_minimal"

    ckpt_args = f"--hf-checkpoint {args.model_dir}/{args.model_name} "
    if args.save_dir:
        ckpt_args += (
            f"--save {args.save_dir} "
            f"--save-interval {args.save_interval} "
            "--no-save-optim "
            "--no-save-rng "
        )
        if args.save_on_diff_threshold:
            ckpt_args += (
                f"--save-on-diff-threshold {args.save_on_diff_threshold} "
                "--save-on-diff-max-saves 15 "
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
        f"--num-rollout {4 if debug else 150} "
        f"--rollout-batch-size {2 if debug else 8} "
        f"--n-samples-per-prompt {2 if debug else 8} "
        f"--rollout-max-response-len {1024 if debug else args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 0.95 "
        "--rollout-top-k 20 "
        f"--global-batch-size {4 if debug else 64} "
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
        # Qwen3.5's GDN layers reject packed (thd) sequences in megatron-core
        # ("GDN does not support packed sequence for now"), so run padded bshd,
        # which requires a static micro batch size instead of dynamic batching.
        "--qkv-format bshd "
        "--micro-batch-size 1 "
        "--log-probs-chunk-size 4096 "  # chunk logits when computing log probs to avoid OOM
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "
    if args.hardware == "B300":
        # B300 (sm103): flashinfer attention not yet supported for this path
        sglang_args += "--sglang-attention-backend trtllm_mha "

    check_args = ""
    if args.check_weight_update_interval:
        check_args = (
            "--check-weight-update-equal "
            f"--check-weight-update-interval {args.check_weight_update_interval} "
        )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # B300: auto selects FA2 via TE (patched for sm103); other GPUs: flash (FA2 direct)
        f"--attention-backend {'auto' if args.hardware == 'B300' else 'flash'} "
        "--megatron-to-hf-mode bridge "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{get_wandb_args(args)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{check_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            # Freeze the vision tower (ViT + projector); consumed in
            # miles/backends/megatron_utils/model_provider.py::_apply_bridge_runtime_config
            "MILES_FREEZE_VISION_MODEL": "1",
            # The H100 devbox holds 4 of the node's 8 GPUs; NVLS multicast fails
            # on a partial NVSwitch domain, so force it off there.
            **({"NCCL_NVLS_ENABLE": "0"} if args.hardware == "H100" else {}),
        },
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
