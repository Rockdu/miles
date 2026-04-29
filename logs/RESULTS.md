# Align-check feature + ablation summary

## What's new
- `--align-check`-style flow: launch script `scripts/run-align-check.sh` runs a
  fast 32-prompt × 1-sample rollout on a single GPU with frozen weights
  (`--lr 0 --debug-skip-optimizer-step`) and `--diffusion-debug-mode`. It dumps
  `rollout_dump_{rollout_id}.pt` so a follow-up run with `--debug-train-only
  --load-debug-rollout-data <path>` can replay rollout-side noise_pred without
  re-spawning sgl-d.
- `actor.py` now threads `rollout_debug_tensors.rollout_model_outputs` through
  `_train_core` → `_build_train_grids` → `_forward_tile` and computes new
  `train/align/*` wandb metrics (also printed to stdout):
    - `noise_pred_abs_mean_diff`, `noise_pred_abs_max_diff`, `noise_pred_rel_l2_diff`
    - `log_prob_max_abs_diff` (the existing mean was kept)
- `wandb_utils.py` registers `train/align/*` under the train/step axis.
- `scripts/run-align-check.sh` exposes ablation knobs via env:
    `APPLY_PATCH_FLAG` (set to `""` to drop `--apply-qwen-image-sgl-d-patch`),
    `CFG_BATCHING_FLAG` (`--fsdp-cfg-batching` for joint),
    `USE_LORA=0`, `USE_GRAD_CKPT=0`, `LOAD_DUMP=<path>`, `EXTRA_ARGS=...`.

## Bug fixes uncovered along the way
- `placement_group.py:create_rollout_manager`: diffusion RolloutManager was
  claiming `num_gpus=1` whenever `--rollout-num-gpus<=1`, starving the engine
  actor (`num_gpus=0.2` on the same bundle) → `init_rollout_engines` hung
  forever. Fixed: RM is `num_gpus=0` for diffusion in all configs.
- Same RM was also taking `num_cpus=1`, fully consuming bundle CPU=1 →
  engine starved on CPU. Fixed: `num_cpus=0.2` in colocated diffusion case.
- `_create_placement_group` bundle resources bumped from `{GPU:1, CPU:1}` to
  `{GPU:1, CPU:16}` so the colocated bundle can host all of:
  RM (0.2) + Engine (0.2) + Train (~0.8) + 1+ OcrRewardActor(s) (1 each)
  without any actor pending on CPU starvation.
- `tools/parse_align_log.py`: small parser for the per-train-step align lines.

## Ablation results (Qwen-Image, 256x256, 10 steps, frozen weights, GPU 7)

| Run | Patch | LoRA | CFG | dtype | noise_pred mean | noise_pred max | rel_l2 | log_prob mean | ratio−1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline_patch_on | ON | gaussian r64 a128 | split | bf16 | 2.51e-02 | 5.60e-01 | 2.84e-02 | 6.73e-04 | 6.72e-04 |
| no_lora_patch_on  | ON | none              | split | bf16 | 2.51e-02 | 5.60e-01 | 2.84e-02 | 6.73e-04 | 6.72e-04 |
| baseline_patch_off | OFF | gaussian r64 a128 | split | bf16 | 3.14e-02 | 8.46e-01 | 3.53e-02 | 9.65e-04 | 9.62e-04 |
| fp32_patch_on     | ON | gaussian r64 a128 | split | fp32* | 5.16e-02 | 1.52e+00 | 6.19e-02 | 3.09e-03 | 3.04e-03 |
| no_gc_patch_on    | ON | gaussian r64 a128 | split | bf16 | OOM | - | - | - | - |
| cfg_joint_patch_on | ON | gaussian r64 a128 | joint | bf16 | crash† | - | - | - | - |

\* fp32 train-side only. sgl-d rollout still loads `param_dtype: torch.bfloat16`
  regardless of `--diffusion-forward-dtype fp32` — this is the pre-existing
  miles/sglang-d miswiring; train↔rollout dtype mismatch is the source of the
  WORSE drift here, not the bf16 path noise per se.

† joint CFG raises `Sizes of tensors must match except in dimension 0. Expected
  size 35 but got size 6` when packing variable-length pos/neg encoder_hidden_states
  along batch — `_pack_cond_for_joint_cfg` predates the v2 changes and never
  handled the variable-text case. Split CFG (default) is unaffected.

## Findings

1. **Patch (`--apply-qwen-image-sgl-d-patch`) helps ~20%** on noise_pred mean
   diff: `2.51e-02 (on)` vs `3.14e-02 (off)`. Confirms RoPE-on-CUDA + RMSNorm
   patches close one source of train↔rollout drift, but the 2.5e-02 floor with
   patch on is still well above the 2e-3 target.
2. **gaussian-init LoRA (B=0) is identity at init** — train(base+B@A=0) and
   rollout(base) match bit-exact through the upload path, so the 2.5e-02 drift
   is **not** LoRA merge-vs-additive (consistent with v2's split-CFG
   structure already aligning train and rollout's CFG path).
3. **The remaining 2.5e-02 mean diff is bf16 numerical noise compounded
   across 60 DiT blocks**. Closing it requires either:
   - All-fp32 forward on both sides (sgl-d currently ignores the dtype flag,
     so this needs an sgl-d-side fix to honor `--diffusion-forward-dtype fp32`).
   - Or, a kernel-level audit that finds another structural divergence in the
     bf16 path (less likely after the patch already covers RoPE + RMSNorm).
4. **Joint CFG path is broken** for variable-length text — would need to pad
   pos/neg encoder_hidden_states to a common max len before cat'ing in
   `_pack_cond_for_joint_cfg`. Not tackled here (out of scope; split is
   default and works).

## Files changed (vs diffusion_RL_v0.1)
- `miles/backends/fsdp_utils/actor.py`
- `miles/utils/wandb_utils.py`
- `miles/ray/placement_group.py`
- `scripts/run-align-check.sh` (new)
- `tools/parse_align_log.py` (new)

