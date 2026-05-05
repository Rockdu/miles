# Single-Prompt Multi-Generation

Single-prompt multi-generation means generating multiple trajectories for the
same prompt in one rollout group. In Miles Diffusion, this is both an algorithmic
requirement and a performance knob:

- Algorithmically, Miles Diffusion compares multiple sampled outputs within a
  prompt group, so each prompt should have multiple sampled outputs.
- System-wise, packing multiple samples from the same prompt into one SGLang-D
  request can improve rollout throughput and reduce request overhead.

This document summarizes how miles-diffusion currently supports this path, what
the main knobs mean, and what still needs validation before treating it as the
default high-throughput configuration.

## Terminology

| Term | Meaning |
| --- | --- |
| Prompt group | One prompt plus its sampled outputs. In code, samples in the same group share `group_index`. |
| `rollout_batch_size` | Number of prompts per rollout. |
| `n_samples_per_prompt` | Number of generated samples per prompt. This is the Miles Diffusion group size. |
| `diffusion_microgroup_size` | Number of same-prompt samples packed into one SGLang-D `/rollout/generate` request. |
| `num_outputs_per_prompt` | SGLang-D request field used by miles to ask the engine for multiple outputs for one prompt. |
| `samples_per_rollout` | `rollout_batch_size * n_samples_per_prompt`. |

For example, the 4-GPU OCR recipe uses:

```text
rollout_batch_size = 32 prompts
n_samples_per_prompt = 16 samples per prompt
samples_per_rollout = 32 * 16 = 512 samples
diffusion_microgroup_size = 16
```

This means each prompt produces 16 samples, and the rollout path tries to send
all 16 samples for one prompt in a single SGLang-D request.

## Current Miles Flow

The current implementation is prompt-group first.

1. The data source reads `rollout_batch_size` prompts.
2. Each prompt is copied `n_samples_per_prompt` times.
3. The copied samples share the same `group_index` and receive unique sample
   indices.
4. The SGLang-D rollout path splits each group into one or more microgroups.
5. Each microgroup is sent to SGLang-D with `num_outputs_per_prompt` equal to
   the microgroup length.
6. Rewards are computed for all samples in the group.
7. Miles Diffusion reward normalization is applied per prompt group, unless
   configured otherwise.
8. The grouped samples are flattened before training.

The key distinction is:

```text
n_samples_per_prompt controls the algorithmic group size.
diffusion_microgroup_size controls how aggressively the group is packed for rollout.
```

Changing `diffusion_microgroup_size` should not change the Miles Diffusion group
semantics, but it can change memory pressure, latency, and SGLang-D batching
behavior.

## Accuracy and Reward Semantics

For Miles Diffusion, the important invariant is that rewards from the same prompt
are normalized together.

By default, miles-diffusion computes:

```text
advantage = sample_reward - mean(rewards in the same prompt group)
```

When `--globalize-reward-std` is enabled, the mean remains per prompt group but
the standard deviation is computed over the full rollout batch. This matches the
Miles Diffusion Qwen-Image recipes.

The rollout path also assigns deterministic seed ranges per prompt group:

```text
seed_base = rollout_seed + group_index * n_samples_per_prompt
sample seeds = seed_base, seed_base + 1, ...
```

SGLang-D currently expands from the first seed inside the request, so miles keeps
the microgroup seed ranges disjoint. A future SGLang-D seed-list API would make
this less implicit.

## Current Recipe Settings

| Recipe | `rollout_batch_size` | `n_samples_per_prompt` | `diffusion_microgroup_size` | Interpretation |
| --- | ---: | ---: | ---: | --- |
| OCR 2-GPU | 16 | 16 | 16 | Full prompt group packed into one rollout request. |
| OCR 4-GPU | 32 | 16 | 16 | Same per-prompt packing, larger prompt batch. |
| PickScore 2-GPU | 16 | 16 | 16 | Full group packing in the scaled-down recipe. |
| PickScore 4-GPU | 32 | 16 | 8 | One prompt group is split into two rollout requests, reducing per-request memory pressure. |

The training-side global batch is derived from:

```text
global_batch_size = rollout_batch_size * n_samples_per_prompt / num_steps_per_rollout
```

So increasing `n_samples_per_prompt` changes both rollout volume and training
batch math. For pure rollout packing experiments, keep `n_samples_per_prompt`
fixed and tune only `diffusion_microgroup_size`.

## Performance Expectations

Single-prompt multi-generation is expected to help when rollout GPUs are
under-utilized. It improves the rollout path by batching multiple same-prompt
samples into one DiT denoising pass, instead of sending the same prompt as many
independent single-output requests.

The expected benefits are:

- fewer HTTP requests per rollout;
- better rollout-side GPU utilization;
- less duplicated prompt-conditioning overhead;
- higher samples-per-second when memory headroom is sufficient.

The tradeoff is memory. Larger `diffusion_microgroup_size` increases the
effective batch size inside SGLang-D. For Qwen-Image, SGLang PR #21988 reports
strong speedup at 512 x 512 when `num_outputs_per_prompt` is increased, but also
substantial peak-memory growth. At 1024 x 1024, very large multi-output batches
can become VAE-decode bound or OOM.

## SGLang-D Dependency

Miles already sends `num_outputs_per_prompt` to SGLang-D, but Qwen-Image needs a
SGLang-D-side condition-batch fix for this to be reliable.

The relevant upstream SGLang PR is:

```text
sgl-project/sglang#21988
```

That PR fixes the mismatch where latent samples are expanded for
`num_outputs_per_prompt > 1`, but prompt and negative-prompt conditioning remain
at the original prompt batch size.

Practical implication:

- If the installed SGLang-D revision includes the fix, Qwen-Image multi-output
  generation can be tested directly.
- If the installed SGLang-D revision does not include the fix,
  `diffusion_microgroup_size > 1` may fail or produce invalid Qwen-Image
  conditioning shapes.
- For reproducible experiments, pin the exact SGLang-D commit rather than a
  moving branch tip.

## Known Limitations

| Limitation | Impact |
| --- | --- |
| Qwen-Image multi-output depends on the SGLang-D condition-batch fix. | The miles-side request path exists, but the engine version must support it. |
| SGLang-D does not currently accept an explicit seed list for one multi-output request. | Miles relies on `seed, seed + 1, ...` expansion and keeps microgroup seed ranges disjoint. |
| SGLang-D rollout does not yet support oversampling plus abort. | `over_sampling_batch_size` must equal `rollout_batch_size` in the current diffusion rollout path. |
| Reward model throughput can dominate. | PickScore can become slower than OCR; increasing rollout batch alone may not improve end-to-end iteration time. |
| Larger output tensors increase IO pressure. | For image and especially video rollout, output encoding/compression may become necessary before adding heavier transfer systems. |
| Batch-invariant behavior still needs per-model validation. | Multi-output batching should be checked against serial generation for reward, trajectory, and log-prob behavior. |

## Validation Checklist

Before enabling large microgroups by default, validate the following for the
target model and task:

1. `diffusion_microgroup_size=1` and `diffusion_microgroup_size=N` produce
   compatible rollout tensor shapes.
2. Samples in a prompt group have distinct seeds and non-identical outputs.
3. `rollout_log_probs` are finite and have the expected timestep shape.
4. Train-side log-prob replay remains within the target tolerance.
5. Reward group statistics are healthy, especially `group_std_avg` and
   zero-std group counts.
6. Rollout GPU utilization improves without pushing VAE decode or reward
   scoring into the bottleneck.
7. End-to-end iteration time improves, not only rollout generation time.

For incremental testing, use this order:

```text
diffusion_microgroup_size = 1 -> 2 -> 4 -> 8 -> 16
```

Keep the rest of the recipe fixed while sweeping this knob.

## Practical Guidance

- Use `n_samples_per_prompt=16` for Miles Diffusion Qwen-Image OCR/PickScore
  experiments unless intentionally changing the algorithmic group size.
- Use `diffusion_microgroup_size=1` for correctness debugging.
- Use `diffusion_microgroup_size=8` or `16` for throughput testing, depending on
  available memory and reward-model cost.
- Prefer OCR for early rollout-throughput validation because the reward path is
  lighter than PickScore.
- Treat PickScore results as end-to-end system tests, not pure rollout
  benchmarks, because the reward model can dominate iteration time.
- For T2V, start with few-frame or single-frame settings first. The same
  single-prompt multi-generation idea applies, but video rollout has much higher
  memory and IO pressure.

## Example Configuration

OCR 4-GPU Miles Diffusion rollout:

```bash
--rollout-batch-size 32
--n-samples-per-prompt 16
--num-steps-per-rollout 2
--diffusion-microgroup-size 16
--micro-batch-size-sample 4
--micro-batch-size-tstep 2
--globalize-reward-std
```

PickScore 4-GPU Miles Diffusion rollout:

```bash
--rollout-batch-size 32
--n-samples-per-prompt 16
--num-steps-per-rollout 2
--diffusion-microgroup-size 8
--micro-batch-size-sample 8
--micro-batch-size-tstep 1
--globalize-reward-std
```

The OCR setting is a better first target for validating whether single-prompt
multi-generation raises rollout utilization. The PickScore setting is more
representative of full system behavior because it includes a heavier reward
worker.
