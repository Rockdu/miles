# Update weights from tensor

miles-diffusion 和 sglang-d 支持两种权重更新方式：

- `update_weights_from_disk`：训练侧把新权重写到模型目录，rollout 侧再从磁盘加载。实现简单，适合调试，但效率较低。
- `update_weights_from_tensor`：训练侧直接把 GPU tensor 暴露给 rollout 侧，rollout 侧通过 CUDA IPC handle读取并加载。RL 训练中频繁同步权重时，这条路径更常用。

这里重点介绍 `update_weights_from_tensor`————基于CUDA IPC的高效权重更新。

## 核心逻辑

`update_weights_from_tensor` 并不是把完整 tensor 通过通信协议比如 HTTP 传给 sglang-d。底层使用 CUDA IPC：miles 侧从 CUDA storage 导出 IPC handle 和 metadata，序列化后只发送这些小对象；sglang-d 侧反序列化 handle，在同一张卡的另一个进程里重新访问这块显存。这样避免了跨进程拷贝完整权重，主要开销变成少量 metadata 通信以及 rollout 侧真正 load 权重。

简化流程如下： 
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

miles 侧负责建立按 rollout engine 划分的 IPC gather group，遍历训练模型 `state_dict()`，把每个 bucket 序列化后 gather 到组内 src rank，再由 src rank 调用对应的 sglang-d engine。sglang-d 侧的 HTTP `/update_weights_from_tensor` 会把请求转给 scheduler 和 GPU worker；worker 根据自己的 TP rank 选择 `serialized_named_tensors[tp_rank]`，反序列化 CUDA IPC tensor，然后由 `WeightsUpdater` 定位 `target_modules`、还原 bucket、调用模块的 load 逻辑。

## FSDP 与 TP

如果 miles 使用 FSDP，`state_dict()` 中的参数可能是 `DTensor` shard。更新前 miles 会先 gather 成 replicated full tensor：

```python
param = param.redistribute(
    placements=[Replicate()] * param.device_mesh.ndim,
    async_op=True,
).to_local()
```

因此发送给 rollout 的不是 FSDP 分片，而是完整 tensor。这样训练侧不需要理解 sglang-d 的 TP 切分规则。Miles 侧按 `rollout_num_gpus_per_engine` 建通信组，就是为了让训练侧 group rank index 和 sglang-d TP rank 对齐。

如果 sglang-d 开了 TP，每个 TP rank 会先拿到本 rank 对应 payload 里的完整 tensor，再在 load weight 时由 sglang-d 模型层逻辑切 shard，例如 linear / attention projection 的切分都在 rollout 侧完成。

## Flattened bucket

逐参数序列化会产生大量小 tensor、IPC handle 和 Python 对象。`FlattenedTensorBucket` 会先按 dtype 分组，把多个 tensor flatten 成一个连续大 tensor，并保存每个参数的 `name`、`shape`、`dtype`、`start_idx`、`end_idx` 等 metadata。sglang-d 收到后用 metadata 从大 tensor view 回原参数列表。

这么做的好处是减少 IPC handle 数量和序列化对象数量，以及能减少显存碎片化，同时仍然能在 rollout 侧恢复参数名和 shape。bucket 上限由 `--update-weight-buffer-size` 控制，默认是 `512 * 1024**2` bytes。调大它通常能减少请求数量，但也可能增加单次 load 的峰值显存和耗时。

## Profiling

SGLang PR [#20464](https://github.com/sgl-project/sglang/pull/20464) 在 H200 单卡、Qwen-Image transformer 更新上进行了profiling。结果是：512MB bucket 需要 82 个请求，总时间约 23.6s；2GB bucket 需要 20 个请求，总时间约 7.75s；8GB bucket 需要 5 个请求，总时间约 4.53s，是该组实验中最快的配置。

| Bucket size | Num bucket | time/bucket (s) | Total time (s) |
|---|---:|---:|---:|
| 0.5G (512MB) | 82 | 0.288 | 23.618 |
| 1G | 40 | 0.380 | 15.185 |
| 2G | 20 | 0.388 | 7.754 |
| 4G | 10 | 1.127 | 11.266 |
| 8G | 5 | 0.906 | 4.530 |
| 20G | 2 | 4.198 | 8.396 |

(TODO: profiling on 2GPU and 4GPU)

## 参考

- miles: `miles/backends/fsdp_utils/diffusion_update_weight_utils.py`
- sglang-d: `sglang/multimodal_gen/runtime/post_training/weights_updater.py`
- SGLang PR: https://github.com/sgl-project/sglang/pull/20464
