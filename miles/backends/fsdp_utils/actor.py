import logging
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils import train_metric_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.sde_log_prob import sde_step_with_logprob
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils import tracking_utils
from miles.utils.profile_utils import TrainProfiler

from .configs.train_pipeline_config import get_train_pipeline_config
import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config

from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from .diffusion_update_weight_utils import DiffusionUpdateWeightFromTensor, DiffusionUpdateWeightFromTensorLoRA

logger = logging.getLogger(__name__)

class FSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO.

    Loads only the DiT (transformer) from a diffusers pipeline, wraps it with
    FSDP, and trains with a PPO-clipped objective aligned with flow GRPO.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)
        
        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        self._compute_dtype = _resolve_compute_dtype(args.diffusion_dtype)

        # Load model without text_encoder / VAE / tokenizer
        with self._get_init_weight_context_manager():
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.hf_checkpoint,
                torch_dtype=self._compute_dtype,
                trust_remote_code=True,
                text_encoder=None,
                vae=None,
                tokenizer=None,
            )
            model = pipeline.transformer
            self.scheduler = pipeline.scheduler
            del pipeline

        self.train_pipeline_config = get_train_pipeline_config(args.diffusion_model)

        if getattr(args, "use_lora", False):
            model = apply_lora(model, args, self.train_pipeline_config)

        model.train()

        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Move to GPU first, then FSDP shard — FSDP2 shards at init time
        # and converts params to DTensor. Must be on GPU for NCCL collectives.
        model.to(torch.cuda.current_device())

        # Rebuild RoPE freq caches on CUDA to bit-match the sglang-d
        # rollout side (which meta-inits and always rebuilds on CUDA).
        self.train_pipeline_config.preprocess_model_before_fsdp(model)

        model = apply_fsdp2(
            model,
            mesh=self.parallel_state.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
        )
        # Force a sync to ensure sharding is complete and old memory is freed.
        torch.cuda.synchronize()
        clear_memory()
        self.model = model

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # sglang-d now supports /update_weights_from_tensor (PR #20464).
        self.weight_updater = (
            DiffusionUpdateWeightFromTensorLoRA(self.args, self.model)
            if getattr(self.args, "use_lora", False)
            else DiffusionUpdateWeightFromTensor(self.args, self.model)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        return

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return

        print_memory("before offload DiT")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after sleep DiT")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.weight_updater is None:
            dist.barrier(group=get_gloo_group())
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()
        clear_memory()
    
    def _get_init_weight_context_manager(self):
        """Return a context manager for model initialization.

        Non-rank-0 ranks use accelerate's ``init_empty_weights`` (params on
        meta device, no allocation). Rank 0 uses ``torch.device("cpu")``
        (already a context manager since PyTorch 1.X — sets default device
        for tensor construction inside the block).
        """
        from accelerate import init_empty_weights

        if dist.get_rank() != 0:
            return init_empty_weights()
        return torch.device("cpu")

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log."""
        if "lr" not in log_dict and hasattr(self, "optimizer"):
            try:
                log_dict["lr"] = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if self.parallel_state.dp_cp_rank == 0:
            dp_size = self.parallel_state.dp_cp_size
            gathered = [None] * dp_size
            dist.gather_object(
                log_dict,
                gathered,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )
            reduced = {k: sum(d[k] for d in gathered) / dp_size for k in log_dict}
            reduced["epoch"] = float(rollout_id)
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            # wandb.define_metric("train/*", step_metric="train/step") pulls the
            # x-axis value from this key; ``train/step`` subsumes what we used
            # to also log as a separate ``global_step`` metric.
            reduced["train/step"] = float(step)
            tracking_utils.log(self.args, reduced, step_key="train/step")
            # Stdout mirror so we can spot misalignment / divergence without wandb.
            # Use scientific notation with 6 significant digits so fp32-level
            # alignment metrics (log_prob_mean_abs_diff, ratio_abs_minus_1, etc.)
            # don't get truncated to 0.0000 by a fixed-decimal format.
            print(
                f"[train step {int(step)}] rollout={rollout_id} "
                + " ".join(f"{k}={v:.6e}" for k, v in sorted(reduced.items()) if k not in ("epoch", "rollout/step", "train/step")),
                flush=True,
            )
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )

    def train(self, rollout_id: int, rollout_data_ref) -> None:  # type: ignore[override]
        # Always wake_up: first call moves from CPU to GPU; subsequent calls
        # are no-ops (offload_train=False) or re-load from CPU (offload_train=True).
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            # Fetch this DP rank's data directly — already split by
            # _split_train_data_by_dp in the RolloutManager.
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
        
        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Diffusion GRPO training loop, aligned with flow GRPO.

        Flow GRPO reference: sglang/3rdparty/flow_grpo/scripts/train_sd3.py:869-944
        Per timestep j:
          1. noise_pred = DiT(latents[j], timesteps[j], encoder_hidden_states)
          2. _, log_prob_new, _, _ = sde_step_with_logprob(scheduler, noise_pred, ...)
          3. ratio = exp(log_prob_new - log_prob_old[j])
          4. loss = max(-adv[j] * ratio, -adv[j] * clamp(ratio))
          5. loss.backward()
        """
        device = torch.cuda.current_device()

        denoising_envs = rollout_data["denoising_env"]
        dit_trajectories = rollout_data["dit_trajectory"]
        rewards = torch.tensor(rollout_data["rewards"], device=device, dtype=torch.float32)
        rollout_log_probs_list = rollout_data["rollout_log_probs"]
        rollout_debug_list = rollout_data.get("rollout_debug_tensors") or [None] * len(denoising_envs)
        sde_step_indices_list = rollout_data.get("sde_step_indices") or [None] * len(denoising_envs)

        batch_size = len(denoising_envs)
        guidance_scale = float(getattr(self.args, "diffusion_guidance_scale", 0))
        true_cfg_scale_arg = getattr(self.args, "diffusion_true_cfg_scale", None)
        true_cfg_scale = float(true_cfg_scale_arg) if true_cfg_scale_arg is not None else None
        # Mirror sglang-d: use true_cfg_scale when set, else guidance_scale.
        cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        use_cfg = cfg_scale > 0
        clip_range = float(getattr(self.args, "diffusion_clip_range", 1e-4))
        adv_clip_max = float(getattr(self.args, "diffusion_adv_clip_max", 5.0))
        noise_level = float(getattr(self.args, "diffusion_noise_level", 0.7))
        num_timesteps = dit_trajectories[0].timesteps.shape[0]

        advantages = rewards.unsqueeze(1).expand(-1, num_timesteps).clone()
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Use rollout's exact timesteps AND derive matching sigmas.
        # Qwen-Image flow-match uses `use_dynamic_shifting=True` (mu depends on
        # image resolution), so the rollout's timesteps are post-shift.
        timesteps_ref = dit_trajectories[0].timesteps.to(device).float()
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        sigmas_ref = timesteps_ref / float(num_train_timesteps)
        sigmas_ref = torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

        self.scheduler.timesteps = timesteps_ref
        self.scheduler.sigmas = sigmas_ref
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

        train_num_timesteps = max(1, num_timesteps)

        num_microbatches = max(1, int(getattr(self.args, "num_microbatches", 1)))
        # `--diffusion-timestep-batch` is retained for backward CLI compat but
        # superseded by sample-dim batching: each DiT forward now processes the
        # whole microbatch (M samples) at one timestep, mirroring flow_grpo's
        # `compute_log_prob` which batches in the sample dim and loops timesteps.
        num_steps_per_rollout = (batch_size + num_microbatches - 1) // num_microbatches

        tpc = self.train_pipeline_config
        _dt = self._compute_dtype

        def _cast_cond(d: dict) -> dict:
            """Cast floating-point tensors to compute dtype; leave bool masks /
            lists unchanged. The bool encoder_hidden_states_mask must NOT be
            cast to bf16 — diffusers reads it as a bool / int mask.
            """
            out: dict = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    out[k] = v.to(_dt)
                else:
                    out[k] = v
            return out

        with timer("actor_train"):
            for step_id in range(num_steps_per_rollout):
                self.optimizer.zero_grad(set_to_none=True)
                log_stats = defaultdict(list)

                traj_start = step_id * num_microbatches
                traj_end = min(batch_size, traj_start + num_microbatches)
                M = traj_end - traj_start

                # Per-sample preparation collected into lists, then stacked.
                lat_list: list[torch.Tensor] = []
                nxt_list: list[torch.Tensor] = []
                ts_list: list[torch.Tensor] = []
                lpo_list: list[torch.Tensor] = []
                adv_list: list[torch.Tensor] = []
                pos_kw_list: list[dict] = []
                neg_kw_list: list[dict] = []
                T_sde: int | None = None

                for i in range(traj_start, traj_end):
                    latents, next_latents, timesteps_i = tpc.prepare_trajectory(dit_trajectories[i], device)
                    env = denoising_envs[i]
                    pos_kw_list.append(tpc.prepare_cond_kwargs(env.pos_cond_kwargs, device))
                    if use_cfg:
                        neg_kw_list.append(tpc.prepare_cond_kwargs(env.neg_cond_kwargs, device))
                    log_prob_old_i = rollout_log_probs_list[i].to(device, dtype=torch.float32)
                    advantage_i = advantages[i]

                    sde_idx = sde_step_indices_list[i]
                    if sde_idx is not None:
                        idx = torch.as_tensor(sde_idx, device=device, dtype=torch.long)
                        latents = latents[idx]
                        next_latents = next_latents[idx]
                        timesteps_i = timesteps_i[idx]
                        log_prob_old_i = log_prob_old_i[idx]
                        advantage_i = advantage_i[: idx.numel()]
                        cur_T = int(idx.numel())
                    else:
                        cur_T = train_num_timesteps

                    if T_sde is None:
                        T_sde = cur_T
                    else:
                        # Per-sample SDE windows can start at different timesteps
                        # but must have equal length so we can stack to (M, T_sde, ...).
                        # `sde_window` strategy guarantees equal length by design.
                        assert cur_T == T_sde, (
                            f"per-sample SDE window length must match across microbatch "
                            f"(got {T_sde} and {cur_T})"
                        )

                    lat_list.append(latents)
                    nxt_list.append(next_latents)
                    ts_list.append(timesteps_i)
                    lpo_list.append(log_prob_old_i)
                    adv_list.append(advantage_i)

                # Stacked batch tensors. (M, T_sde, ...) where samples may have
                # *different* timestep values per slot j (per-sample windows).
                latents_mb = torch.stack(lat_list, dim=0)              # (M, T_sde, C, H, W)
                next_latents_mb = torch.stack(nxt_list, dim=0)
                timesteps_mb = torch.stack(ts_list, dim=0)             # (M, T_sde)
                log_prob_old_mb = torch.stack(lpo_list, dim=0)         # (M, T_sde)
                advantage_mb = torch.stack(adv_list, dim=0)            # (M, T_sde)

                # Collate cond kwargs across the microbatch. For CFG, collate
                # pos+neg jointly so encoder_hidden_states / mask use a unified
                # max_seq_len across both halves and we can run one batch=2M
                # DiT forward (mirroring flow_grpo's `compute_log_prob` cat).
                if use_cfg:
                    cond_collated = tpc.collate_cond_for_sample_batch(
                        pos_kw_list + neg_kw_list, device
                    )
                else:
                    cond_collated = tpc.collate_cond_for_sample_batch(pos_kw_list, device)
                cond_kw = _cast_cond(cond_collated)

                for j in range(T_sde):
                    lat_j = latents_mb[:, j]                            # (M, C, H, W)
                    nxt_j = next_latents_mb[:, j]                       # (M, C, H, W)
                    ts_j = timesteps_mb[:, j]                           # (M,)
                    lpo_j = log_prob_old_mb[:, j]                       # (M,)
                    adv_j = advantage_mb[:, j]                          # (M,)

                    # sgl-d's Qwen DiT divides timestep by num_train_timesteps
                    # inside forward; diffusers' Qwen DiT does NOT — so we must
                    # pre-scale here to land at the same time-embedding input.
                    # Ref: sglang/.../models/dits/qwen_image.py (`timestep = timestep / 1000`).
                    ts_for_model = ts_j / float(num_train_timesteps)

                    if use_cfg:
                        h = torch.cat([lat_j, lat_j], dim=0)            # (2M, C, H, W)
                        ts_combined = torch.cat([ts_for_model, ts_for_model], dim=0)
                    else:
                        h = lat_j
                        ts_combined = ts_for_model

                    # Match rollout's compute dtype exactly. Rollout runs under
                    # torch.autocast("cuda", <dtype>) so all inputs enter the DiT
                    # as that dtype. Without explicit cast here, FSDP MixedPrecision
                    # only casts params but leaves fp32 inputs → first matmul runs
                    # at higher precision than rollout → systematic noise_pred drift.
                    # When diffusion_dtype=fp32, this is a no-op (inputs already fp32).
                    noise_pred_combined = self.model(
                        hidden_states=h.to(_dt),
                        timestep=ts_combined.to(_dt),
                        return_dict=False,
                        **cond_kw,
                    )[0]

                    if use_cfg:
                        noise_pred_pos, noise_pred_neg = noise_pred_combined.chunk(2, dim=0)
                        noise_pred = tpc.cfg_combine(
                            noise_pred_pos,
                            noise_pred_neg,
                            guidance_scale,
                            true_cfg_scale=true_cfg_scale,
                        )
                    else:
                        noise_pred = noise_pred_combined

                    _, log_prob_new, _, _ = sde_step_with_logprob(
                        self.scheduler,
                        noise_pred.float(),
                        ts_j,
                        lat_j.float(),
                        prev_sample=nxt_j.float(),
                        noise_level=noise_level,
                    )                                                   # log_prob_new: (M,)

                    ratio = torch.exp(log_prob_new - lpo_j)
                    unclipped = -adv_j * ratio
                    clipped = -adv_j * torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    loss = torch.mean(torch.maximum(unclipped, clipped))
                    # Sample axis already mean-reduced by torch.mean above; only
                    # the timestep loop remains, hence divide by T_sde to produce
                    # the same total grad scale as the old per-sample-per-tchunk
                    # accumulation (which divided by num_microbatches × T_sde/tb).
                    if not getattr(self.args, "debug_skip_optimizer_step", False):
                        (loss / T_sde).backward()

                    with torch.no_grad():
                        per_elem = torch.maximum(unclipped, clipped)
                        log_stats["loss"].append(loss.detach())
                        log_stats["loss_abs_mean"].append(per_elem.abs().mean().detach())
                        log_stats["adv_abs_mean"].append(adv_j.abs().mean().detach())
                        log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
                        log_stats["approx_kl"].append(
                            0.5 * torch.mean((log_prob_new - lpo_j) ** 2).detach()
                        )
                        log_stats["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
                        )
                        log_stats["log_prob_new_idx_0"].append(log_prob_new[0].detach())
                        log_stats["log_prob_old_idx_0"].append(lpo_j[0].detach())
                        log_stats["log_prob_mean_abs_diff"].append(torch.mean(torch.abs(log_prob_new - lpo_j)).detach())

                self.prof.step(rollout_id=rollout_id)
                # One optimizer step per step_id.
                if not getattr(self.args, "debug_skip_optimizer_step", False):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    log_stats["grad_norm"].append(grad_norm.detach())
                    self.optimizer.step()
                    self.lr_scheduler.step()
                else:
                    # Keep weights frozen so noise_pred / log_prob alignment checks
                    # remain interpretable across iterations.
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Prefix with "train/" so wandb groups these under the Train panel
                # and picks up define_metric("train/*", step_metric="train/step") —
                # otherwise they fall into the default "Charts" section and plot
                # against wandb's auto-incrementing internal step.
                reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
                self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def _resolve_compute_dtype(name: str) -> torch.dtype:
    """Map --diffusion-dtype string to torch.dtype. Single source of truth."""
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16  # default

def apply_lora(model: torch.nn.Module, args: Namespace, train_pipeline_config) -> None:
    """Apply PEFT LoRA to the model.

    Args:
        model: The model to apply LoRA to.
        args: Arguments containing LoRA settings.
        train_pipeline_config: The train pipeline config.
    """
    from peft import LoraConfig, get_peft_model

    targets = getattr(args, "lora_target_modules", None) or train_pipeline_config.lora_target_modules
    init_lora_weight = getattr(args, "diffusion_init_lora_weight", "gaussian")
    # "kaiming-uniform" is the default for PEFT's LoraConfig (passed as `True`).
    # Other init methods: "gaussian", "olora", "pissa", "pissa_niter_N", "loftq", ...
    if init_lora_weight == "kaiming-uniform":
        init_lora_weight = True
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        init_lora_weights=init_lora_weight,
    ))
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model

def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (--diffusion-dtype, --fp16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    diffusion_dtype = getattr(args, "diffusion_dtype", None) if args is not None else None
    param_dtype = _resolve_compute_dtype(diffusion_dtype)
    # flow_grpo's FSDP1 bf16 MixedPrecision reduces gradients in bf16. Keep
    # Miles' default fp32 reduce unless the run explicitly opts into parity.
    if getattr(args, "bf16_reduce", False):
        reduce_dtype = torch.bfloat16
        logger.info("--bf16-reduce set: using bf16 grad reduce (matches flow_grpo)")
    else:
        reduce_dtype = torch.float32

    logger.info(f"FSDP: wrapping {len(modules)} modules of type {layer_cls_to_wrap}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
