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

        Per optim window of M samples × T_sde timesteps, slide a
        (sample_microbatch, tstep_microbatch) tile across the (M, T_sde) grid
        and accumulate gradients. Two presets:

          sample_microbatch=M, tstep_microbatch=1, iter_order=sample_major
            equivalent to "batch by samples" (forward batch = M, loop T_sde
            times); peak activation memory ∝ M.

          sample_microbatch=1, tstep_microbatch=T_sde, iter_order=timestep_major
            equivalent to "batch by timesteps" (forward batch = T_sde, loop M
            times); peak activation memory ∝ T_sde.

        Loss scaling is uniform across plans: per-tile mean PPO loss / n_tiles
        → net gradient = mean over (M, T_sde), matching the all-timesteps
        accumulation grad scale flow_grpo and the previous miles-d sample-major
        loop produce.
        """
        device = torch.cuda.current_device()

        denoising_envs = rollout_data["denoising_env"]
        dit_trajectories = rollout_data["dit_trajectory"]
        rewards = torch.tensor(rollout_data["rewards"], device=device, dtype=torch.float32)
        rollout_log_probs_list = rollout_data["rollout_log_probs"]
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
        num_steps_per_rollout = (batch_size + num_microbatches - 1) // num_microbatches

        tpc = self.train_pipeline_config

        # Plan: (sample_microbatch, tstep_microbatch, iter_order). Defaults
        # reproduce the previous "batch by samples" behavior.
        sample_mb_arg = getattr(self.args, "diffusion_train_sample_microbatch", None)
        tstep_mb_arg = max(1, int(getattr(self.args, "diffusion_train_tstep_microbatch", 1)))
        iter_order = getattr(self.args, "diffusion_train_iter_order", "sample_major")
        assert iter_order in ("sample_major", "timestep_major"), iter_order

        with timer("actor_train"):
            for step_id in range(num_steps_per_rollout):
                self.optimizer.zero_grad(set_to_none=True)

                traj_start = step_id * num_microbatches
                traj_end = min(batch_size, traj_start + num_microbatches)
                grids = self._build_train_grids(
                    traj_start=traj_start,
                    traj_end=traj_end,
                    dit_trajectories=dit_trajectories,
                    denoising_envs=denoising_envs,
                    rollout_log_probs_list=rollout_log_probs_list,
                    sde_step_indices_list=sde_step_indices_list,
                    advantages=advantages,
                    train_num_timesteps=train_num_timesteps,
                    use_cfg=use_cfg,
                    device=device,
                )

                # Effective tile sizes, clamped to the grid.
                M_w, T_w = grids["M"], grids["T_sde"]
                sm = min(sample_mb_arg if sample_mb_arg is not None else M_w, M_w)
                tm = min(tstep_mb_arg, T_w)
                sm = max(1, sm)
                tm = max(1, tm)

                log_stats = self._run_optim_window(
                    grids=grids,
                    sample_mb=sm,
                    tstep_mb=tm,
                    iter_order=iter_order,
                    use_cfg=use_cfg,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    clip_range=clip_range,
                    noise_level=noise_level,
                    num_train_timesteps=num_train_timesteps,
                )

                self.prof.step(rollout_id=rollout_id)
                if not getattr(self.args, "debug_skip_optimizer_step", False):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    log_stats["grad_norm"].append(grad_norm.detach())
                    self.optimizer.step()
                    self.lr_scheduler.step()
                else:
                    # Keep weights frozen so noise_pred / log_prob alignment
                    # checks remain interpretable across iterations.
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Prefix with "train/" so wandb groups these under the Train
                # panel and picks up define_metric("train/*",
                # step_metric="train/step") — otherwise they fall into the
                # default "Charts" section and plot against wandb's
                # auto-incrementing internal step.
                reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
                self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

    def _build_train_grids(
        self,
        *,
        traj_start: int,
        traj_end: int,
        dit_trajectories: list,
        denoising_envs: list,
        rollout_log_probs_list: list,
        sde_step_indices_list: list,
        advantages: torch.Tensor,
        train_num_timesteps: int,
        use_cfg: bool,
        device: torch.device,
    ) -> dict:
        """Build per-window (M, T_sde, ...) grids ready for tile slicing.

        Per-sample SDE windows can start at different timesteps but must have
        equal length T_sde so they stack cleanly. The `sde_window` strategy
        guarantees this.
        """
        tpc = self.train_pipeline_config

        lat_list, nxt_list, ts_list, lpo_list, adv_list = [], [], [], [], []
        pos_kw_list, neg_kw_list = [], []
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
                assert cur_T == T_sde, (
                    f"per-sample SDE window length must match across microbatch "
                    f"(got {T_sde} and {cur_T})"
                )
            lat_list.append(latents)
            nxt_list.append(next_latents)
            ts_list.append(timesteps_i)
            lpo_list.append(log_prob_old_i)
            adv_list.append(advantage_i)

        # Stacked grids — (M, T_sde, ...).
        latents_mb = torch.stack(lat_list, dim=0)
        next_latents_mb = torch.stack(nxt_list, dim=0)
        timesteps_mb = torch.stack(ts_list, dim=0)
        log_prob_old_mb = torch.stack(lpo_list, dim=0)
        advantage_mb = torch.stack(adv_list, dim=0)

        # Collate cond kwargs once for the whole window. For CFG, pack
        # [pos | neg] into a single (2M, ...) collate so they share a unified
        # max_seq_len; tile slicing then re-splits the halves.
        if use_cfg:
            cond_collated = self.train_pipeline_config.collate_cond_for_sample_batch(
                pos_kw_list + neg_kw_list, device
            )
        else:
            cond_collated = self.train_pipeline_config.collate_cond_for_sample_batch(
                pos_kw_list, device
            )

        return {
            "lat": latents_mb,
            "nxt": next_latents_mb,
            "ts": timesteps_mb,
            "lpo": log_prob_old_mb,
            "adv": advantage_mb,
            "cond": cond_collated,
            "M": int(traj_end - traj_start),
            "T_sde": int(T_sde or 0),
        }

    def _run_optim_window(
        self,
        *,
        grids: dict,
        sample_mb: int,
        tstep_mb: int,
        iter_order: str,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        clip_range: float,
        noise_level: float,
        num_train_timesteps: int,
    ) -> dict[str, list[torch.Tensor]]:
        """Iterate (sample_mb × tstep_mb) tiles across the (M, T_sde) grid,
        running one DiT forward + PPO loss + backward per tile.

        All tiles share the same loss scaling: ``(loss_tile / n_tiles).backward()``.
        Net gradient is therefore mean over (M × T_sde) cells regardless of
        plan — flipping plans changes wall-clock and memory, not the optimizer
        update direction (modulo bf16 reduction order).
        """
        device = grids["lat"].device
        M, T = grids["M"], grids["T_sde"]
        s_chunks = _chunked_indices(M, sample_mb, device)
        t_chunks = _chunked_indices(T, tstep_mb, device)
        n_tiles = len(s_chunks) * len(t_chunks)

        if iter_order == "sample_major":
            outer, inner = t_chunks, s_chunks
        else:
            outer, inner = s_chunks, t_chunks

        log_stats: dict[str, list[torch.Tensor]] = defaultdict(list)
        skip_step = bool(getattr(self.args, "debug_skip_optimizer_step", False))

        for o in outer:
            for i in inner:
                s_idx, t_idx = (i, o) if iter_order == "sample_major" else (o, i)
                loss = self._forward_tile(
                    s_idx=s_idx,
                    t_idx=t_idx,
                    grids=grids,
                    use_cfg=use_cfg,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    clip_range=clip_range,
                    noise_level=noise_level,
                    num_train_timesteps=num_train_timesteps,
                    log_stats=log_stats,
                )
                if not skip_step:
                    (loss / n_tiles).backward()

        return log_stats

    def _forward_tile(
        self,
        *,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        grids: dict,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        clip_range: float,
        noise_level: float,
        num_train_timesteps: int,
        log_stats: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        """One DiT forward over an (m, k) tile flattened to batch=(m*k).

        Inputs:
          s_idx: 1-D LongTensor — sample-axis indices into M (size m).
          t_idx: 1-D LongTensor — timestep-axis indices into T_sde (size k).

        Returns the mean PPO loss for the tile (scalar). Caller scales by
        1/n_tiles before backward.
        """
        device = grids["lat"].device
        _dt = self._compute_dtype
        tpc = self.train_pipeline_config
        m, k = int(s_idx.numel()), int(t_idx.numel())
        M_total = grids["M"]

        # (m, k, ...) → (m*k, ...) for one DiT forward.
        lat_tile = grids["lat"][s_idx][:, t_idx]                # (m, k, C, H, W)
        nxt_tile = grids["nxt"][s_idx][:, t_idx]
        ts_tile = grids["ts"][s_idx][:, t_idx]                  # (m, k)
        lpo_tile = grids["lpo"][s_idx][:, t_idx]                # (m, k)
        adv_tile = grids["adv"][s_idx][:, t_idx]                # (m, k)

        h_flat = lat_tile.reshape(m * k, *lat_tile.shape[2:])
        ts_flat = ts_tile.reshape(m * k)

        # sgl-d's Qwen DiT divides timestep by num_train_timesteps inside
        # forward; diffusers' Qwen DiT does NOT — pre-scale here so both
        # land at the same time-embedding input.
        ts_for_model = ts_flat / float(num_train_timesteps)

        cond_tile = _slice_collated_cond(
            grids["cond"], s_idx=s_idx, k=k, M_total=M_total, use_cfg=use_cfg
        )
        cond_tile = _cast_cond_to_dtype(cond_tile, _dt)

        if use_cfg:
            h = torch.cat([h_flat, h_flat], dim=0)              # (2*m*k, C, H, W)
            ts_combined = torch.cat([ts_for_model, ts_for_model], dim=0)
        else:
            h = h_flat
            ts_combined = ts_for_model

        # Match rollout's compute dtype exactly. Rollout runs under
        # torch.autocast("cuda", <dtype>) so all inputs enter the DiT as
        # that dtype. Without explicit cast here, FSDP MixedPrecision only
        # casts params but leaves fp32 inputs → first matmul runs at higher
        # precision than rollout → systematic noise_pred drift.
        noise_pred_combined = self.model(
            hidden_states=h.to(_dt),
            timestep=ts_combined.to(_dt),
            return_dict=False,
            **cond_tile,
        )[0]

        if use_cfg:
            noise_pred_pos, noise_pred_neg = noise_pred_combined.chunk(2, dim=0)
            noise_pred_flat = tpc.cfg_combine(
                noise_pred_pos,
                noise_pred_neg,
                guidance_scale,
                true_cfg_scale=true_cfg_scale,
            )
        else:
            noise_pred_flat = noise_pred_combined

        # SDE log-prob is per-element along batch; (m*k,) in/out.
        _, log_prob_new_flat, _, _ = sde_step_with_logprob(
            self.scheduler,
            noise_pred_flat.float(),
            ts_flat,
            h_flat.float(),
            prev_sample=nxt_tile.reshape(m * k, *nxt_tile.shape[2:]).float(),
            noise_level=noise_level,
        )                                                       # (m*k,)
        log_prob_new = log_prob_new_flat.reshape(m, k)
        ratio = torch.exp(log_prob_new - lpo_tile)
        unclipped = -adv_tile * ratio
        clipped = -adv_tile * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        per_cell = torch.maximum(unclipped, clipped)
        loss = per_cell.mean()

        with torch.no_grad():
            log_stats["loss"].append(loss.detach())
            log_stats["loss_abs_mean"].append(per_cell.abs().mean().detach())
            log_stats["adv_abs_mean"].append(adv_tile.abs().mean().detach())
            log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
            log_stats["approx_kl"].append(
                0.5 * torch.mean((log_prob_new - lpo_tile) ** 2).detach()
            )
            log_stats["clipfrac"].append(
                torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
            )
            # Pin first sample / first tstep of the tile for time-series
            # alignment debugging across runs.
            log_stats["log_prob_new_idx_0"].append(log_prob_new[0, 0].detach())
            log_stats["log_prob_old_idx_0"].append(lpo_tile[0, 0].detach())
            log_stats["log_prob_mean_abs_diff"].append(
                torch.mean(torch.abs(log_prob_new - lpo_tile)).detach()
            )

        return loss


def _chunked_indices(n: int, chunk: int, device: torch.device) -> list[torch.Tensor]:
    """Split range(n) into 1-D LongTensor chunks of size <= ``chunk``.

    Used to slice the (M, T_sde) train grid into tiles. Always returns at
    least one chunk; the last chunk may be shorter when n % chunk != 0.
    """
    if n <= 0:
        return []
    chunk = max(1, chunk)
    return [
        torch.arange(start, min(start + chunk, n), device=device, dtype=torch.long)
        for start in range(0, n, chunk)
    ]


def _slice_collated_cond(
    cond: dict,
    *,
    s_idx: torch.Tensor,
    k: int,
    M_total: int,
    use_cfg: bool,
) -> dict:
    """Slice a window-collated cond dict to a tile of shape (m*k, ...).

    The collate is shape (M, ...) (no CFG) or (2M, ...) (CFG, packed
    [pos | neg]). For each sample row picked by ``s_idx``, repeat that row
    ``k`` times consecutively so the tile aligns with the (m, k) grid
    flattened to (m*k, ...). For CFG, slice the pos and neg halves
    independently and re-pack as [pos_mk | neg_mk] — the caller does the
    matching ``torch.cat([h_pos_mk, h_neg_mk])``.

    The slicing is dtype-agnostic; tensors keep their original dtype
    (caller casts to compute dtype after slicing).
    """
    m = int(s_idx.numel())

    def _slice_value(v, rows: torch.Tensor):
        if isinstance(v, torch.Tensor):
            return v.index_select(0, rows).repeat_interleave(k, dim=0)
        if isinstance(v, list):
            picked = [v[int(r)] for r in rows.tolist()]
            return [x for x in picked for _ in range(k)]
        # scalars / None / strings: pass through
        return v

    out: dict = {}
    if use_cfg:
        # rows in the pos half (0..M-1) and the neg half (M..2M-1).
        s_idx_neg = s_idx + M_total
        for key, v in cond.items():
            pos_part = _slice_value(v, s_idx)
            neg_part = _slice_value(v, s_idx_neg)
            if isinstance(v, torch.Tensor):
                out[key] = torch.cat([pos_part, neg_part], dim=0)
            elif isinstance(v, list):
                out[key] = pos_part + neg_part
            else:
                out[key] = v
        return out

    for key, v in cond.items():
        out[key] = _slice_value(v, s_idx)
    return out


def _cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    """Cast floating-point tensors to the model's compute dtype; leave bool
    masks / int / list / scalar values untouched. The bool
    encoder_hidden_states_mask must NOT be cast — diffusers reads it as a
    bool/int mask.
    """
    out: dict = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


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
