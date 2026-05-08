# Miles-diffusion Weights Update

miles-diffusion and sglang-d support two weight update modes:

- `update_weights_from_disk`: the training side writes the new weights to a model directory, and the rollout side reloads them from disk. This path is simple and useful for debugging, but less efficient.
- `update_weights_from_tensor`: the training side exposes GPU tensors directly to the rollout side, and the rollout side reads and loads them through CUDA IPC handles. This is the preferred path when RL training needs frequent weight synchronization.

This document focuses on `update_weights_from_tensor`: efficient weight update based on CUDA IPC.

## Core logic

`update_weights_from_tensor` does not send full tensors to sglang-d through a communication protocol such as HTTP. The underlying mechanism is CUDA IPC: miles exports IPC handles and metadata from CUDA storage, serializes only these small objects, and sends them to sglang-d. sglang-d then deserializes the handles and reopens the same GPU memory from another process on the same GPU. This avoids copying the full weights across processes. The main costs become small metadata communication and the actual weight loading work on the rollout side.

A simplified flow:

```python
# miles training rank
for name, param in model.state_dict().items():
    if isinstance(param, DTensor):
        param = param.redistribute(Replicate(), async_op=True).to_local()
    bucket.append((name, param.cuda()))

bucket = FlattenedTensorBucket(bucket)
payload = MultiprocessingSerializer.serialize({
    "transformer": {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }
}, output_str=True)

engine.update_weights_from_tensor.remote(
    serialized_named_tensors=rank_payloads,
    load_format="flattened_bucket",
    target_modules=["transformer"],
)
```

On the miles side, the updater creates IPC gather groups by rollout engine, iterates over the training model `state_dict()`, serializes each bucket, gathers the serialized buckets to the source rank in the group, and lets that source rank call the corresponding sglang-d engine. On the sglang-d side, the HTTP `/update_weights_from_tensor` endpoint forwards the request to the scheduler and GPU worker. The worker selects `serialized_named_tensors[tp_rank]` according to its TP rank, deserializes the CUDA IPC tensor, and then uses `WeightsUpdater` to resolve `target_modules`, reconstruct the bucket, and call the module loading logic.

## FSDP and TP

When miles uses FSDP, parameters in `state_dict()` may be `DTensor` shards. Before update, miles first gathers them into replicated full tensors:

```python
param = param.redistribute(
    placements=[Replicate()] * param.device_mesh.ndim,
    async_op=True,
).to_local()
```

Therefore, the rollout side receives full tensors rather than FSDP shards. This keeps the training side independent from sglang-d's TP sharding rules. miles creates communication groups according to `rollout_num_gpus_per_engine` so that the training-side group rank index is aligned with the sglang-d TP rank.

If sglang-d enables TP, each TP rank first receives the full tensor from its rank-specific payload. During weight loading, sglang-d's model-layer logic shards the tensor as needed. For example, linear layers and attention projections are sharded on the rollout side rather than manually by miles.

## Flattened bucket

Serializing parameters one by one would create many small tensors, IPC handles, and Python objects. `FlattenedTensorBucket` first groups tensors by dtype, flattens multiple tensors into one contiguous large tensor, and stores metadata such as `name`, `shape`, `dtype`, `start_idx`, and `end_idx` for each parameter. After receiving the payload, sglang-d uses this metadata to view slices from the large tensor back as the original parameter list.

This reduces the number of IPC handles and serialized objects, and also helps reduce GPU memory fragmentation, while still allowing the rollout side to recover parameter names and shapes. The bucket limit is controlled by `--update-weight-buffer-size`, whose default value is `512 * 1024**2` bytes. Increasing it usually reduces the number of requests, but may also increase peak memory usage and the latency of each load.

## Profiling

SGLang PR [#20464](https://github.com/sgl-project/sglang/pull/20464) profiled updating the Qwen-Image transformer on a single H200 GPU. The results show that a 512MB bucket needs 82 requests and takes about 23.6s in total; a 2GB bucket needs 20 requests and takes about 7.75s; an 8GB bucket needs 5 requests and takes about 4.53s, which is the fastest configuration in this experiment.

| Bucket size | Num bucket | time/bucket (s) | Total time (s) |
|---|---:|---:|---:|
| 0.5G (512MB) | 82 | 0.288 | 23.618 |
| 1G | 40 | 0.380 | 15.185 |
| 2G | 20 | 0.388 | 7.754 |
| 4G | 10 | 1.127 | 11.266 |
| 8G | 5 | 0.906 | 4.530 |
| 20G | 2 | 4.198 | 8.396 |

(TODO: profiling on 2GPU and 4GPU)

## References

- miles: `miles/backends/fsdp_utils/diffusion_update_weight_utils.py`
- sglang-d: `sglang/multimodal_gen/runtime/post_training/weights_updater.py`
- SGLang PR: https://github.com/sgl-project/sglang/pull/20464
