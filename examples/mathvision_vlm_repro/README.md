# MathVision × Qwen3.5-9B — logprob_abs_diff explosion repro

Lightweight reproduction of the VLM RL instability reported in
`#ext-amazon-radixark`: `train_rollout_logprob_abs_diff` grows steadily during
GRPO training of Qwen3.5/3.6-VL models with **32k** rollout response length,
eventually exploding (>0.1) and collapsing the run. 16k stays flat; lowering LR
only delays the blow-up; text-only models don't show it.

| | Source setting (Xinpeng Wei, 27B) | This repro (Jiajun Li, 9B) |
|---|---|---|
| Model | Qwen3.6-27B | Qwen/Qwen3.5-9B (natively multimodal) |
| Data | MathVision | MathVision (`test` split as train pool) |
| Response len | 32k | 32k |
| LR | 2e-6 | 2e-6 |
| Batch | rollout 64 × 8 samples | rollout 8 × 8 samples |
| Mode | async, staleness=1 | sync colocate (also reproduced sync upstream) |
| ViT | frozen | frozen (`MILES_FREEZE_VISION_MODEL=1`) |
| Expected | diff >0.1 at step ~30–50, reward → 0 | diff 0.006 → 0.0216 within 100 steps |

## Run

```bash
export WANDB_API_KEY=...                               # optional, project miles-vlm
python examples/mathvision_vlm_repro/run_mathvision_qwen3_5_9b.py
# smoke test: --mode debug_minimal; control run: --rollout-max-response-len 16384
```

## What to watch

- `train_rollout_logprob_abs_diff` — the repro signal: steady growth from
  ~0.006 with no matching growth in response length.
- `response_len` — must actually approach the 32k cap for the trigger to be
  exercised (thinking mode is enabled for this reason).
- `grad_norm`, `reward` — in the full-size setting these explode / collapse
  after the diff passes ~0.05.

## Notes

- Vision-tower freezing has no CLI flag; this example env-gates it in
  `model_provider.py::_apply_bridge_runtime_config` (bridge providers expose
  `freeze_vision_model` / `freeze_vision_projection`).
- Requires the activation-recompute fix in `model_provider.py` (present since
  2026-06-24); without it 32k contexts OOM with ~3MB/token activations.
- Rollout uses `examples.geo3k_vlm.rollout.generate` (tensor-only multimodal
  train inputs filter) — no VLM-specific logic beyond that.
