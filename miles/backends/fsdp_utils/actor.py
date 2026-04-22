import logging
import os
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.sde_log_prob import sde_step_with_logprob
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils import tracking_utils

from .configs.train_pipeline_config import get_train_pipeline_config
import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config

from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from .diffusion_update_weight_utils import DiffusionUpdateWeightFromTensor, DiffusionUpdateWeightFromTensorLoRA

logger = logging.getLogger(__name__)


def _rebuild_pos_embed_freqs_on_cuda(model) -> None:
    """Rebuild ``QwenEmbedRope.pos_freqs`` / ``neg_freqs`` on CUDA for
    bit-exact train/rollout RoPE alignment.

    diffusers' ``QwenEmbedRope.__init__`` runs ``torch.arange(4096)`` +
    ``torch.pow(theta, ...)`` on CPU, and the forward pass only moves the
    resulting tensors to the target device via ``.to(device)`` — the
    underlying fp32 bytes stay CPU-computed.  sglang-d's DiT loads under
    ``init_empty_weights`` (meta) and takes the meta-rebuild branch
    inside its own ``QwenEmbedRope.forward``, recomputing the same
    frequencies on CUDA.  CPU and CUDA implementations of ``torch.pow``
    differ by fp32 ULPs, so the two caches end up byte-different even
    though they represent the same mathematical values.  That tiny
    divergence propagates through RoPE → attention → every block and
    produces an end-to-end ``noise_pred`` drift of 1-3e-02 with frozen
    weights.

    This function walks the model, finds every ``QwenEmbedRope`` (or
    similar) module, and rebuilds its freq caches on CUDA using the same
    formulas but with explicit ``device=`` at every allocation.  Must be
    called after ``model.to(cuda)`` and before FSDP sharding.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type != "cuda":
        return

    for submod in model.modules():
        # Match by attribute shape rather than class name so we also
        # handle ``QwenEmbedLayer3DRope`` and similar variants.
        if not (
            hasattr(submod, "pos_freqs")
            and hasattr(submod, "neg_freqs")
            and hasattr(submod, "rope_params")
            and hasattr(submod, "axes_dim")
            and hasattr(submod, "theta")
        ):
            continue
        theta = submod.theta

        def _rope_params_cuda(index: torch.Tensor, dim: int) -> torch.Tensor:
            inv_freq = 1.0 / torch.pow(
                theta,
                torch.arange(0, dim, 2, device=device).to(torch.float32).div(dim),
            )
            freqs = torch.outer(index, inv_freq)
            return torch.polar(torch.ones_like(freqs), freqs)

        pos_idx = torch.arange(4096, device=device)
        neg_idx = torch.arange(4096, device=device).flip(0) * -1 - 1
        submod.pos_freqs = torch.cat(
            [_rope_params_cuda(pos_idx, d) for d in submod.axes_dim], dim=1
        )
        submod.neg_freqs = torch.cat(
            [_rope_params_cuda(neg_idx, d) for d in submod.axes_dim], dim=1
        )
        # Clear any LRU cache that may hold CPU-derived freqs.
        _cvf = getattr(submod, "_compute_video_freqs", None)
        if _cvf is not None and hasattr(_cvf, "cache_clear"):
            _cvf.cache_clear()
        logger.info(
            "[rope_cuda_rebuild] module=%s axes_dim=%s pos_freqs.device=%s",
            submod.__class__.__name__,
            submod.axes_dim,
            submod.pos_freqs.device,
        )


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

        # Load the diffusion pipeline; keep only transformer + scheduler.
        # --diffusion-dtype controls the training-side DiT compute precision.
        # Must match the rollout engine's compute dtype for clean log-prob
        # alignment; pass the same value to sglang-d via sglang_dit_dtype.
        dtype = _resolve_compute_dtype(args.diffusion_dtype)
        self._compute_dtype = dtype
        pipeline = DiffusionPipeline.from_pretrained(
            args.diffusion_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = pipeline.transformer
        self.scheduler = pipeline.scheduler
        del pipeline
        clear_memory()

        self.train_pipeline_config = get_train_pipeline_config(args.diffusion_model)

        if getattr(args, "use_lora", False):
            from peft import LoraConfig, get_peft_model

            targets = getattr(args, "lora_target_modules", None) or self.train_pipeline_config.lora_target_modules
            init = getattr(args, "diffusion_init_lora_weight", "gaussian")
            # "kaiming-uniform" is the common name for PEFT's default (passed as `True`).
            # Everything else is a string PEFT already recognises ("gaussian", "olora",
            # "pissa", "pissa_niter_N", "loftq", ...).
            if init == "kaiming-uniform":
                init = True
            model = get_peft_model(model, LoraConfig(
                r=getattr(args, "lora_rank", 64),
                lora_alpha=getattr(args, "lora_alpha", 64),
                target_modules=targets,
                init_lora_weights=init,
            ))
            if dist.get_rank() == 0:
                model.print_trainable_parameters()

        model.train()

        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Move to GPU first, then FSDP shard — FSDP2 shards at init time
        # and converts params to DTensor. Must be on GPU for NCCL collectives.
        model.to(torch.cuda.current_device())

        # BIT-EXACT RoPE FIX: diffusers' ``QwenEmbedRope`` builds
        # ``pos_freqs`` / ``neg_freqs`` on CPU in ``__init__`` (via
        # ``torch.arange(4096)`` with no device=), and their ``forward``
        # only *moves* them to the target device via ``.to(device)``.
        # The values inside are therefore the CPU-computed result of
        # ``torch.pow(theta, ...)`` and ``torch.polar(...)``.
        # sglang-d's DiT loads under ``init_empty_weights`` (meta), so
        # its forward hits the meta-rebuild branch and recomputes the
        # frequencies on CUDA.  CPU and CUDA implementations of
        # ``torch.pow`` differ by fp32 ULPs → the two sides end up with
        # byte-different RoPE caches → RoPE outputs differ → attention
        # outputs differ → every block's output drifts → end-to-end
        # noise_pred mean|Δ| ~1-3e-02 with frozen weights.
        # Rebuild freqs on CUDA here, before FSDP shards the model, to
        # match sglang-d exactly.
        _rebuild_pos_embed_freqs_on_cuda(model)
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
        # Allow bypass for alignment debugging: both training and rollout load
        # from the same HF checkpoint, so in theory no sync is needed until
        # training actually updates weights.
        disable_sync = bool(getattr(self.args, "debug_disable_weight_sync", False))
        if self.args.debug_train_only or disable_sync:
            self.weight_updater = None
            if disable_sync and dist.get_rank() == 0:
                logger.info("[debug] weight sync disabled via --debug-disable-weight-sync")
        elif getattr(self.args, "use_lora", False):
            self.weight_updater = DiffusionUpdateWeightFromTensorLoRA(self.args, self.model)
        else:
            self.weight_updater = DiffusionUpdateWeightFromTensor(self.args, self.model)

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        return

    @timer
    def sleep(self) -> None:
        if self.args.offload_train:
            self.model.cpu()
            move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after sleep DiT")

    @timer
    def wake_up(self) -> None:
        if self.args.offload_train:
            self.model.cuda()
            move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

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
            print(
                f"[train step {int(step)}] rollout={rollout_id} "
                + " ".join(f"{k}={v:.4f}" for k, v in sorted(reduced.items()) if k not in ("epoch", "rollout/step", "train/step")),
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
        self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            # Fetch this DP rank's data directly — already split by
            # _split_train_data_by_dp in the RolloutManager.
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.offload_train:
            self.sleep()

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
        # Per-sample sde-window step indices (from step_strategy_hub.sde_window).
        # When set, the trajectory / log_probs come back full-length and we
        # slice to this subset — mirroring flow_grpo, which only computes
        # log_prob / loss on in-window steps.
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

        # Broadcast scalar reward to per-timestep advantage.
        # rewards shape: (batch_size,) -> (batch_size, num_timesteps)
        advantages = rewards.unsqueeze(1).expand(-1, num_timesteps).clone()
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Use rollout's exact timesteps AND derive matching sigmas.
        # Qwen-Image flow-match uses `use_dynamic_shifting=True` (mu depends on
        # image resolution), so the rollout's timesteps are post-shift. We can't
        # just call `set_timesteps(N)` with `use_dynamic_shifting=False`, because
        # that would produce unshifted sigmas while `scheduler.sigmas[step_index]`
        # would then disagree with the rollout's std_dev_t / prev_sample_mean.
        # Flow matching invariant (no invert_sigmas): sigma_i = t_i / num_train_timesteps.
        timesteps_ref = dit_trajectories[0].timesteps.to(device).float()
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        sigmas_ref = timesteps_ref / float(num_train_timesteps)
        # Append terminal sigma=0 to match FlowMatchEulerDiscreteScheduler.set_timesteps().
        sigmas_ref = torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

        self.scheduler.timesteps = timesteps_ref
        self.scheduler.sigmas = sigmas_ref
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

        train_num_timesteps = max(1, num_timesteps)

        trajectories_per_step = max(1, int(getattr(self.args, "diffusion_gradient_accumulation_steps", 1)))
        timestep_batch = int(getattr(self.args, "diffusion_timestep_batch", 1))
        num_steps_per_rollout = (batch_size + trajectories_per_step - 1) // trajectories_per_step

        for step_id in range(num_steps_per_rollout):
            self.optimizer.zero_grad(set_to_none=True)
            log_stats = defaultdict(list)

            traj_start = step_id * trajectories_per_step
            traj_end = min(batch_size, traj_start + trajectories_per_step)

            # Inner loop: accumulate gradients over multiple trajectories.
            for i in range(traj_start, traj_end):
                tpc = self.train_pipeline_config
                latents, next_latents, timesteps_i = tpc.prepare_trajectory(dit_trajectories[i], device)
                env = denoising_envs[i]
                pos_cond = tpc.prepare_cond_kwargs(env.pos_cond_kwargs, device)
                neg_cond = tpc.prepare_cond_kwargs(env.neg_cond_kwargs, device) if use_cfg else None
                log_prob_old_i = rollout_log_probs_list[i].to(device, dtype=torch.float32)
                advantage_i = advantages[i]
                reward_i = rewards[i]

                # Restrict to the flow_grpo-style SDE window (if any). Trajectory and
                # log_probs come back full-length so scheduler.timesteps/sigmas stay
                # correct for any j via `index_for_timestep`; we just index in.
                sde_idx = sde_step_indices_list[i]
                if sde_idx is not None:
                    idx = torch.as_tensor(sde_idx, device=device, dtype=torch.long)
                    latents = latents[idx]
                    next_latents = next_latents[idx]
                    timesteps_i = timesteps_i[idx]
                    log_prob_old_i = log_prob_old_i[idx]
                    advantage_i = advantage_i[: idx.numel()]
                    sample_train_steps = int(idx.numel())
                else:
                    sample_train_steps = train_num_timesteps

                # Batch multiple timesteps for GPU utilization.
                for t_start in range(0, sample_train_steps, timestep_batch):
                    t_end = min(sample_train_steps, t_start + timestep_batch)
                    tb = t_end - t_start
                    lat_chunk = latents[t_start:t_end]
                    ts_chunk = timesteps_i[t_start:t_end]

                    # sgl-d's Qwen DiT divides timestep by num_train_timesteps
                    # inside forward; diffusers' Qwen DiT does NOT — so we must
                    # pre-scale here to land at the same time-embedding input.
                    # Ref: sglang/.../models/dits/qwen_image.py (`timestep = timestep / 1000`).
                    #
                    # sgl-d's ``timestep`` entering its DiT is fp32 (see
                    # ``denoising.py:expand_timestep_before_forward`` → no
                    # dtype cast, just ``t_device.repeat(bsz)``), so its
                    # ``(timestep / 1000).to(dtype)`` is effectively
                    # ``(fp32 / 1000).to(bf16)`` — the same order this line
                    # uses.  Do NOT pre-cast ``ts_chunk`` to bf16 here;
                    # that would use bf16 arithmetic and diverge from sgl-d
                    # by 1 bf16 ULP at most sigmas.
                    ts_chunk_for_model = ts_chunk / float(num_train_timesteps)

                    pos_batch = tpc.expand_cond_for_timestep_batch(pos_cond, tb)
                    if t_start == 0 and i == traj_start:
                        alloc = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        print(f"[DEBUG] before first forward: allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)

                    # Match rollout's compute dtype exactly. Rollout runs under
                    # torch.autocast("cuda", <dtype>) so all inputs enter the DiT
                    # as that dtype. Without explicit cast here, FSDP MixedPrecision
                    # only casts params but leaves fp32 inputs → first matmul runs
                    # at higher precision than rollout → systematic noise_pred drift.
                    # When diffusion_dtype=fp32, this is a no-op (inputs already fp32).
                    _dt = self._compute_dtype
                    _cast = lambda d: {k: v.to(_dt) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

                    # [latent fingerprint] Print SHA-256 of the exact bf16
                    # tensor going into training DiT + metadata. Runs once
                    # per rollout at (i=traj_start, step_id=0, t_start=0).
                    # Purpose: rule out an fp32→bf16 round-trip artifact on
                    # miles side, and provide a byte-level fingerprint that
                    # can be cross-checked against an optional sglang-d
                    # sha256 print at denoising.py:1149 ``ctx.latents``.
                    # If the two hashes match, DiT input is byte-equal on
                    # both sides → the residual ~2e-02 is truly DiT math.
                    if (
                        i == traj_start and step_id == 0 and t_start == 0
                    ):
                        import hashlib
                        # .numpy() rejects bf16; view as uint8 to get raw
                        # bytes regardless of dtype (sha256 cares about
                        # bytes, not numerics).
                        _h = lambda t: hashlib.sha256(
                            t.contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest()[:16]
                        # Print per-timestep hashes (one row per DiT call)
                        # so we can match against sglang-d's per-step_index
                        # sha print (in denoising.py with MILES_DUMP_LATENT_HASH).
                        # ``sde_idx`` gives the absolute trajectory step
                        # indices for this batch, so the k-th row
                        # corresponds to sglang's step_index=sde_idx[k].
                        _lc_bf16 = lat_chunk.to(_dt).detach().contiguous().cpu()
                        _sde_idx_list = (
                            sde_idx.tolist() if hasattr(sde_idx, "tolist")
                            else (list(sde_idx) if sde_idx is not None else None)
                        )
                        # Latent per-row
                        for _k in range(_lc_bf16.shape[0]):
                            _row = _lc_bf16[_k]
                            _abs_step = (
                                _sde_idx_list[t_start + _k]
                                if _sde_idx_list is not None else t_start + _k
                            )
                            print(
                                f"[latent fingerprint i={i} chunk_row={_k} "
                                f"sglang_step_index={_abs_step}] "
                                f"sha_bf16={_h(_row)} "
                                f"dtype={_row.dtype} shape={tuple(_row.shape)} "
                                f"contig={_row.is_contiguous()} stride={_row.stride()} "
                                f"norm={_row.float().norm().item():.6f}",
                                flush=True,
                            )
                        # Timestep: miles pre-scales by /num_train_timesteps
                        # in fp32, then .to(bf16). sglang's DiT does the
                        # divide internally in bf16. If the two paths don't
                        # produce bit-equal post-scale bf16 timesteps, that's
                        # a real residual source.
                        try:
                            _ts_raw = ts_chunk.detach().cpu()
                            _ts_fed = ts_chunk_for_model.to(_dt).detach().cpu()
                            # sglang-actual path: fp32_ts / 1000 then cast
                            # to bf16 (matches denoising.py line 1182 +
                            # qwen_image.py line 1229 where ``timestep``
                            # entering the DiT is fp32 from the scheduler,
                            # divided in fp32, then cast to target dtype).
                            _ts_sglang_actual = (
                                (ts_chunk / 1000.0).to(torch.bfloat16)
                            ).detach().cpu()
                            # Hypothetical bf16-first path for contrast:
                            # cast ts→bf16 first, divide in bf16.
                            _ts_bf16_path = (
                                (ts_chunk.to(torch.bfloat16) / 1000.0)
                                .to(torch.bfloat16).detach().cpu()
                            )
                            for _k in range(_ts_raw.shape[0]):
                                _abs_step = (
                                    _sde_idx_list[t_start + _k]
                                    if _sde_idx_list is not None else t_start + _k
                                )
                                print(
                                    f"[ts fingerprint i={i} chunk_row={_k} "
                                    f"sglang_step_index={_abs_step}] "
                                    f"raw_val={_ts_raw[_k].item():.6f} "
                                    f"miles_fed_sha={_h(_ts_fed[_k:_k+1])} "
                                    f"miles_fed_val={_ts_fed[_k].float().item():.8f} "
                                    f"sglang_actual_sha={_h(_ts_sglang_actual[_k:_k+1])} "
                                    f"sglang_actual_val={_ts_sglang_actual[_k].float().item():.8f} "
                                    f"bf16_path_sha={_h(_ts_bf16_path[_k:_k+1])} "
                                    f"bf16_path_val={_ts_bf16_path[_k].float().item():.8f} "
                                    f"miles==sglang_actual: {torch.equal(_ts_fed[_k:_k+1], _ts_sglang_actual[_k:_k+1])}",
                                    flush=True,
                                )
                        except Exception as _e:
                            print(f"[ts fingerprint] failed: {_e}", flush=True)
                        # Encoder hidden states + mask (from pos_batch).
                        try:
                            _p = _cast(pos_batch)
                            _ehs = _p.get("encoder_hidden_states")
                            _mask = _p.get("encoder_hidden_states_mask")
                            _img_shapes = _p.get("img_shapes")
                            _txt_seq_lens = _p.get("txt_seq_lens")
                            def _h_flat(t):
                                # Flatten+contiguous so 0-d and non-contig
                                # are handled; then view as bytes.
                                b = t.detach().contiguous().cpu().flatten()
                                if b.numel() == 0:
                                    return "empty"
                                return hashlib.sha256(
                                    b.view(torch.uint8).numpy().tobytes()
                                ).hexdigest()[:16]
                            def _fmt(x):
                                if isinstance(x, torch.Tensor):
                                    return (
                                        f"sha={_h_flat(x)} dtype={x.dtype} "
                                        f"shape={tuple(x.shape)} "
                                        f"norm={x.float().norm().item():.6f}"
                                    )
                                if isinstance(x, list):
                                    parts = []
                                    for _j, _e in enumerate(x):
                                        if isinstance(_e, torch.Tensor):
                                            parts.append(
                                                f"[{_j}] sha={_h_flat(_e)} "
                                                f"dtype={_e.dtype} shape={tuple(_e.shape)} "
                                                f"norm={_e.float().norm().item():.6f}"
                                            )
                                        else:
                                            parts.append(f"[{_j}] type={type(_e).__name__}")
                                    return "list(" + " ; ".join(parts) + ")"
                                return f"type={type(x).__name__}"
                            print(
                                f"[miles pos_cond i={i}] "
                                f"ehs={_fmt(_ehs)} | "
                                f"mask={_fmt(_mask)} | "
                                f"img_shapes={_img_shapes} "
                                f"txt_seq_lens={_txt_seq_lens}",
                                flush=True,
                            )
                        except Exception as _e:
                            print(f"[miles pos_cond] failed: {_e}", flush=True)

                    # (pos_embed freqs were rebuilt on CUDA at model
                    # load time in ``FSDPTrainRayActor.init`` via
                    # ``_rebuild_pos_embed_freqs_on_cuda`` — that's the
                    # bit-exact RoPE fix that makes train/rollout
                    # noise_pred align to 0.0 mean|Δ|.)

                    # Register forward hooks to dump hidden_states hash
                    # after pre-block ops (img_in, txt_norm, txt_in,
                    # time_text_embed) and after each transformer block,
                    # on the FIRST forward only. Hooks unregister
                    # themselves after the forward. Also install an
                    # intra-block hash hook on block 0 to find the first
                    # sub-op where miles/sglang diverge.
                    _block_hooks = []
                    _intra_installed_block0 = None
                    _orig_blk0_forward_ref = None
                    if (
                        i == traj_start and step_id == 0 and t_start == 0
                    ):
                        # Install intra-block hash via monkey-patching
                        # block 0's forward method. Matches sglang's
                        # qwen_image.py intra-print stages.
                        import os as _os_intra_m
                        _blocks_ref = getattr(self.model, "transformer_blocks", None)
                        if (
                            _blocks_ref is not None and len(_blocks_ref) > 0
                            and _os_intra_m.environ.get("MILES_DUMP_INTRA")
                            in ("1", "true", "True")
                        ):
                            import hashlib as _hashlib_mintra
                            def _mih(t):
                                if not isinstance(t, torch.Tensor):
                                    return f"type={type(t).__name__}"
                                b = t.detach().contiguous().cpu().flatten()
                                if b.numel() == 0:
                                    return "empty"
                                return _hashlib_mintra.sha256(
                                    b.view(torch.uint8).numpy().tobytes()
                                ).hexdigest()[:16]
                            _blk0 = _blocks_ref[0]
                            _orig_blk0_forward_ref = _blk0.forward
                            def _mifmt(tag, t, _ctr):
                                if isinstance(t, torch.Tensor):
                                    print(
                                        f"[miles intra blk0 call{_ctr}] {tag} "
                                        f"sha={_mih(t)} dtype={t.dtype} "
                                        f"shape={tuple(t.shape)} "
                                        f"norm={t.float().norm().item():.6f}",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"[miles intra blk0 call{_ctr}] {tag} "
                                        f"type={type(t).__name__}",
                                        flush=True,
                                    )
                            _blk0_call_ctr = [0]
                            def _patched_blk0_forward(
                                hidden_states,
                                encoder_hidden_states,
                                encoder_hidden_states_mask,
                                temb,
                                image_rotary_emb=None,
                                joint_attention_kwargs=None,
                                modulate_index=None,
                            ):
                                _c = _blk0_call_ctr[0]
                                _blk0_call_ctr[0] += 1
                                if _c >= 4:
                                    return _orig_blk0_forward_ref(
                                        hidden_states=hidden_states,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                                        temb=temb,
                                        image_rotary_emb=image_rotary_emb,
                                        joint_attention_kwargs=joint_attention_kwargs,
                                        modulate_index=modulate_index,
                                    )
                                self_blk = _blk0
                                _mifmt("enter_hs", hidden_states, _c)
                                _mifmt("enter_ehs", encoder_hidden_states, _c)
                                _mifmt("temb", temb, _c)
                                # Replicate block forward with hash points:
                                img_mod_params = self_blk.img_mod(temb)
                                temb_for_txt = temb
                                if getattr(self_blk, "zero_cond_t", False):
                                    temb_for_txt = torch.chunk(temb, 2, dim=0)[0]
                                txt_mod_params = self_blk.txt_mod(temb_for_txt)
                                _mifmt("img_mod_params", img_mod_params, _c)
                                _mifmt("txt_mod_params", txt_mod_params, _c)
                                img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
                                txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
                                img_normed = self_blk.img_norm1(hidden_states)
                                img_modulated, img_gate1 = self_blk._modulate(
                                    img_normed, img_mod1, modulate_index,
                                )
                                _mifmt("img_modulated (post_img_norm1)", img_modulated, _c)
                                _mifmt("img_gate1", img_gate1, _c)
                                txt_normed = self_blk.txt_norm1(encoder_hidden_states)
                                txt_modulated, txt_gate1 = self_blk._modulate(
                                    txt_normed, txt_mod1,
                                )
                                _mifmt("txt_modulated (post_txt_norm1)", txt_modulated, _c)
                                joint_attention_kwargs = joint_attention_kwargs or {}
                                attn_output = self_blk.attn(
                                    hidden_states=img_modulated,
                                    encoder_hidden_states=txt_modulated,
                                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                                    image_rotary_emb=image_rotary_emb,
                                    **joint_attention_kwargs,
                                )
                                img_attn_output, txt_attn_output = attn_output
                                _mifmt("img_attn_output (post_attn)", img_attn_output, _c)
                                _mifmt("txt_attn_output (post_attn)", txt_attn_output, _c)
                                hidden_states = hidden_states + img_gate1 * img_attn_output
                                encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
                                _mifmt("hs_post_residual", hidden_states, _c)
                                img_normed2 = self_blk.img_norm2(hidden_states)
                                img_modulated2, img_gate2 = self_blk._modulate(
                                    img_normed2, img_mod2, modulate_index,
                                )
                                _mifmt("img_modulated2 (post_img_norm2+residual)", img_modulated2, _c)
                                img_mlp_output = self_blk.img_mlp(img_modulated2)
                                _mifmt("img_mlp_output", img_mlp_output, _c)
                                hidden_states = hidden_states + img_gate2 * img_mlp_output
                                _mifmt("hs_post_img_mlp_add", hidden_states, _c)
                                txt_normed2 = self_blk.txt_norm2(encoder_hidden_states)
                                txt_modulated2, txt_gate2 = self_blk._modulate(
                                    txt_normed2, txt_mod2,
                                )
                                txt_mlp_output = self_blk.txt_mlp(txt_modulated2)
                                encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
                                if encoder_hidden_states.dtype == torch.float16:
                                    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
                                if hidden_states.dtype == torch.float16:
                                    hidden_states = hidden_states.clip(-65504, 65504)
                                return encoder_hidden_states, hidden_states
                            _blk0.forward = _patched_blk0_forward
                            _intra_installed_block0 = _blk0

                            # Also monkey-patch block 0 attention forward
                            # to hash QKV linear outputs, qk_norm, RoPE,
                            # SDPA inputs/output, to_out — pinpoint first
                            # divergent sub-op.
                            _attn0 = _blk0.attn
                            _orig_attn0_forward_ref = _attn0.forward
                            _attn0_call_ctr = [0]
                            from diffusers.models.transformers.transformer_qwenimage import (
                                apply_rotary_emb_qwen as _apply_rotary_qwen,
                            )
                            import torch.nn.functional as _F
                            def _mafmt(tag, t, _c):
                                if isinstance(t, torch.Tensor):
                                    print(
                                        f"[miles attn blk0 call{_c}] {tag} "
                                        f"sha={_mih(t)} dtype={t.dtype} "
                                        f"shape={tuple(t.shape)} "
                                        f"norm={t.float().norm().item():.6f}",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"[miles attn blk0 call{_c}] {tag} "
                                        f"type={type(t).__name__}",
                                        flush=True,
                                    )
                            def _patched_attn0_forward(
                                hidden_states,
                                encoder_hidden_states=None,
                                encoder_hidden_states_mask=None,
                                attention_mask=None,
                                image_rotary_emb=None,
                                **kwargs,
                            ):
                                _c = _attn0_call_ctr[0]
                                _attn0_call_ctr[0] += 1
                                if _c >= 4:
                                    return _orig_attn0_forward_ref(
                                        hidden_states=hidden_states,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                                        attention_mask=attention_mask,
                                        image_rotary_emb=image_rotary_emb,
                                        **kwargs,
                                    )
                                attn = _attn0
                                seq_txt = encoder_hidden_states.shape[1]
                                _mafmt("enter_hs", hidden_states, _c)
                                _mafmt("enter_ehs", encoder_hidden_states, _c)
                                img_query = attn.to_q(hidden_states)
                                img_key = attn.to_k(hidden_states)
                                img_value = attn.to_v(hidden_states)
                                txt_query = attn.add_q_proj(encoder_hidden_states)
                                txt_key = attn.add_k_proj(encoder_hidden_states)
                                txt_value = attn.add_v_proj(encoder_hidden_states)
                                _mafmt("img_q_post_linear", img_query, _c)
                                _mafmt("img_k_post_linear", img_key, _c)
                                _mafmt("img_v_post_linear", img_value, _c)
                                _mafmt("txt_q_post_linear", txt_query, _c)
                                _mafmt("txt_k_post_linear", txt_key, _c)
                                _mafmt("txt_v_post_linear", txt_value, _c)
                                img_query = img_query.unflatten(-1, (attn.heads, -1))
                                img_key = img_key.unflatten(-1, (attn.heads, -1))
                                img_value = img_value.unflatten(-1, (attn.heads, -1))
                                txt_query = txt_query.unflatten(-1, (attn.heads, -1))
                                txt_key = txt_key.unflatten(-1, (attn.heads, -1))
                                txt_value = txt_value.unflatten(-1, (attn.heads, -1))
                                _mafmt("img_q_post_unflatten", img_query, _c)
                                _mafmt("img_k_post_unflatten", img_key, _c)
                                _mafmt("txt_q_post_unflatten", txt_query, _c)
                                _mafmt("txt_k_post_unflatten", txt_key, _c)
                                if attn.norm_q is not None:
                                    img_query = attn.norm_q(img_query)
                                if attn.norm_k is not None:
                                    img_key = attn.norm_k(img_key)
                                if attn.norm_added_q is not None:
                                    txt_query = attn.norm_added_q(txt_query)
                                if attn.norm_added_k is not None:
                                    txt_key = attn.norm_added_k(txt_key)
                                _mafmt("img_q_post_qknorm_only", img_query, _c)
                                _mafmt("img_k_post_qknorm_only", img_key, _c)
                                _mafmt("txt_q_post_qknorm_only", txt_query, _c)
                                _mafmt("txt_k_post_qknorm_only", txt_key, _c)
                                if image_rotary_emb is not None:
                                    img_freqs, txt_freqs = image_rotary_emb
                                    _mafmt("img_freqs_complex", img_freqs, _c)
                                    _mafmt("txt_freqs_complex", txt_freqs, _c)
                                    # Also dump real [cos|sin] form so we
                                    # can compare to sglang's cos_sin_cache.
                                    if isinstance(img_freqs, torch.Tensor) and img_freqs.is_complex():
                                        _ifr = torch.cat([img_freqs.real, img_freqs.imag], dim=-1).contiguous()
                                        _mafmt("img_freqs_real(cos|sin)", _ifr, _c)
                                    if isinstance(txt_freqs, torch.Tensor) and txt_freqs.is_complex():
                                        _tfr = torch.cat([txt_freqs.real, txt_freqs.imag], dim=-1).contiguous()
                                        _mafmt("txt_freqs_real(cos|sin)", _tfr, _c)
                                    img_query = _apply_rotary_qwen(img_query, img_freqs, use_real=False)
                                    img_key = _apply_rotary_qwen(img_key, img_freqs, use_real=False)
                                    txt_query = _apply_rotary_qwen(txt_query, txt_freqs, use_real=False)
                                    txt_key = _apply_rotary_qwen(txt_key, txt_freqs, use_real=False)
                                _mafmt("img_q_post_qknorm_rope", img_query, _c)
                                _mafmt("img_k_post_qknorm_rope", img_key, _c)
                                _mafmt("txt_q_post_qknorm_rope", txt_query, _c)
                                _mafmt("txt_k_post_qknorm_rope", txt_key, _c)
                                joint_query = torch.cat([txt_query, img_query], dim=1)
                                joint_key = torch.cat([txt_key, img_key], dim=1)
                                joint_value = torch.cat([txt_value, img_value], dim=1)
                                _mafmt("joint_q", joint_query, _c)
                                _mafmt("joint_k", joint_key, _c)
                                _mafmt("joint_v", joint_value, _c)
                                # SDPA — diffusers dispatches via
                                # dispatch_attention_fn. Replicate its most
                                # common path: F.scaled_dot_product_attention
                                # with q/k/v transposed to [B, H, S, D].
                                jq = joint_query.transpose(1, 2)
                                jk = joint_key.transpose(1, 2)
                                jv = joint_value.transpose(1, 2)
                                joint_hidden_states = _F.scaled_dot_product_attention(
                                    jq, jk, jv,
                                    attn_mask=attention_mask,
                                    dropout_p=0.0,
                                    is_causal=False,
                                ).transpose(1, 2)
                                _mafmt("post_sdpa", joint_hidden_states, _c)
                                joint_hidden_states = joint_hidden_states.flatten(2, 3)
                                joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
                                _mafmt("post_flatten", joint_hidden_states, _c)
                                txt_attn_output = joint_hidden_states[:, :seq_txt, :]
                                img_attn_output = joint_hidden_states[:, seq_txt:, :]
                                _mafmt("txt_split", txt_attn_output, _c)
                                _mafmt("img_split", img_attn_output, _c)
                                img_attn_output = attn.to_out[0](img_attn_output.contiguous())
                                if len(attn.to_out) > 1:
                                    img_attn_output = attn.to_out[1](img_attn_output)
                                _mafmt("img_post_to_out", img_attn_output, _c)
                                txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
                                _mafmt("txt_post_to_add_out", txt_attn_output, _c)
                                return img_attn_output, txt_attn_output
                            _attn0.forward = _patched_attn0_forward
                        import hashlib as _hashlib_blk
                        def _blk_sha(t):
                            if not isinstance(t, torch.Tensor):
                                return f"type={type(t).__name__}"
                            b = t.detach().contiguous().cpu().flatten()
                            if b.numel() == 0:
                                return "empty"
                            return _hashlib_blk.sha256(
                                b.view(torch.uint8).numpy().tobytes()
                            ).hexdigest()[:16]
                        # Pre-block hooks: single tensor output, not tuple.
                        def _make_single_hook(_tag):
                            def _hook(_mod, _inputs, _outputs):
                                try:
                                    t = _outputs
                                    if isinstance(t, tuple):
                                        t = t[0]
                                    if isinstance(t, torch.Tensor):
                                        print(
                                            f"[miles preblk] stage={_tag} "
                                            f"sha={_blk_sha(t)} "
                                            f"dtype={t.dtype} "
                                            f"shape={tuple(t.shape)} "
                                            f"norm={t.float().norm().item():.6f}",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"[miles preblk] stage={_tag} "
                                            f"type={type(t).__name__}",
                                            flush=True,
                                        )
                                except Exception as _ee:
                                    print(f"[miles preblk {_tag}] hook failed: {_ee}", flush=True)
                            return _hook
                        def _make_blk_hook(_idx):
                            def _hook(_mod, _inputs, _outputs):
                                try:
                                    ehs, hs = _outputs
                                    print(
                                        f"[miles block {_idx:02d}] "
                                        f"hs_sha={_blk_sha(hs)} hs_norm={hs.float().norm().item():.6f} "
                                        f"ehs_sha={_blk_sha(ehs)} ehs_norm={ehs.float().norm().item():.6f}",
                                        flush=True,
                                    )
                                except Exception as _ee:
                                    print(f"[miles block {_idx}] hook failed: {_ee}", flush=True)
                            return _hook
                        # Hook pre-block submodules by name.
                        for _attr_name in ["img_in", "txt_norm", "txt_in", "time_text_embed"]:
                            _sub = getattr(self.model, _attr_name, None)
                            if _sub is not None:
                                _h = _sub.register_forward_hook(
                                    _make_single_hook(f"post_{_attr_name}")
                                )
                                _block_hooks.append(_h)
                        _blocks_attr = getattr(self.model, "transformer_blocks", None)
                        if _blocks_attr is not None:
                            for _bi, _blk in enumerate(_blocks_attr):
                                _h = _blk.register_forward_hook(_make_blk_hook(_bi))
                                _block_hooks.append(_h)

                    noise_pred_pos = self.model(
                        hidden_states=lat_chunk.to(_dt),
                        timestep=ts_chunk_for_model.to(_dt),
                        return_dict=False,
                        **_cast(pos_batch),
                    )[0]

                    for _h in _block_hooks:
                        _h.remove()
                    if _intra_installed_block0 is not None and _orig_blk0_forward_ref is not None:
                        _intra_installed_block0.forward = _orig_blk0_forward_ref
                        _attn0_ref = getattr(_intra_installed_block0, "attn", None)
                        _orig_attn_ref = locals().get("_orig_attn0_forward_ref")
                        if _attn0_ref is not None and _orig_attn_ref is not None:
                            _attn0_ref.forward = _orig_attn_ref

                    if t_start == 0 and i == traj_start:
                        alloc = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        print(f"[DEBUG] after first forward: allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)

                    if use_cfg and neg_cond is not None:
                        neg_batch = tpc.expand_cond_for_timestep_batch(neg_cond, tb)
                        noise_pred_neg = self.model(
                            hidden_states=lat_chunk.to(_dt),
                            timestep=ts_chunk_for_model.to(_dt),
                            return_dict=False,
                            **_cast(neg_batch),
                        )[0]
                        noise_pred = tpc.cfg_combine(
                            noise_pred_pos,
                            noise_pred_neg,
                            guidance_scale,
                            true_cfg_scale=true_cfg_scale,
                        )
                    else:
                        noise_pred = noise_pred_pos

                    # DEBUG: compare training's noise_pred with rollout's stored
                    # model_output for the first trajectory / first chunk. If the
                    # two DiT implementations match numerically, this diff should
                    # be ~1e-3 (or smaller). Bigger diffs pinpoint a forward-input
                    # mismatch (latent, timestep, cond_kwargs) rather than SDE math.
                    if i == traj_start and step_id == 0 and rollout_debug_list[i] is not None:
                        rdt = rollout_debug_list[i]
                        if rdt is not None and rdt.rollout_model_outputs is not None:
                            ro_mo = rdt.rollout_model_outputs.to(device).float()
                            # Match training's sde-window slicing: training slices
                            # latents/next_latents/timesteps with ``sde_idx`` above so
                            # noise_pred corresponds to trajectory steps
                            # ``sde_idx[t_start:t_end]`` — compare ro_mo against the
                            # same trajectory steps, not ``ro_mo[t_start:t_end]``.
                            if sde_idx is not None:
                                ro_mo_sliced = ro_mo[idx]
                            else:
                                ro_mo_sliced = ro_mo
                            if t_start == 0:
                                print(
                                    f"[rollout_model_outputs] full shape={tuple(ro_mo.shape)} "
                                    f"norm_overall={ro_mo.norm().item():.3f}",
                                    flush=True,
                                )
                            # Expected shape: (T, C, H, W) or (T, N, C) per trajectory.
                            # Slice to the current chunk's timesteps.
                            ro_chunk = ro_mo_sliced[t_start:t_end].to(noise_pred.dtype)
                            if ro_chunk.shape == noise_pred.shape:
                                diff = (noise_pred - ro_chunk).abs()
                                print(
                                    f"[noise_pred align traj=0 chunk_t={t_start}:{t_end}] "
                                    f"mean={diff.mean().item():.4e} max={diff.max().item():.4e} "
                                    f"train_norm={noise_pred.norm().item():.3f} "
                                    f"rollout_norm={ro_chunk.norm().item():.3f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[noise_pred align] shape mismatch: "
                                    f"train={tuple(noise_pred.shape)} vs rollout_chunk={tuple(ro_chunk.shape)}",
                                    flush=True,
                                )

                    _, log_prob_new, _, _ = sde_step_with_logprob(
                        self.scheduler,
                        noise_pred.float(),
                        timesteps_i[t_start:t_end],
                        latents[t_start:t_end].float(),
                        prev_sample=next_latents[t_start:t_end].float(),
                        noise_level=noise_level,
                    )

                    adv_chunk = advantage_i[t_start:t_end]
                    old_chunk = log_prob_old_i[t_start:t_end]

                    ratio = torch.exp(log_prob_new - old_chunk)
                    unclipped = -adv_chunk * ratio
                    clipped = -adv_chunk * torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    loss = torch.mean(torch.maximum(unclipped, clipped))
                    if not getattr(self.args, "debug_skip_optimizer_step", False):
                        loss.backward()

                    with torch.no_grad():
                        per_elem = torch.maximum(unclipped, clipped)
                        log_stats["loss"].append(loss.detach())
                        # Diagnostic: abs-mean shows raw loss magnitude before sign cancellation
                        log_stats["loss_abs_mean"].append(per_elem.abs().mean().detach())
                        log_stats["adv_abs_mean"].append(adv_chunk.abs().mean().detach())
                        log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
                        log_stats["approx_kl"].append(
                            0.5 * torch.mean((log_prob_new - old_chunk) ** 2).detach()
                        )
                        log_stats["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
                        )
                        log_stats["log_prob_new_idx_0"].append(log_prob_new[0].detach())
                        log_stats["log_prob_old_idx_0"].append(old_chunk[0].detach())
                        log_stats["log_prob_mean_abs_diff"].append(torch.mean(torch.abs(log_prob_new - old_chunk)).detach())

            # One optimizer step per step_id.
            if not getattr(self.args, "debug_skip_optimizer_step", False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
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
