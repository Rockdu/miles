# Qwen-Image Block: diffusers vs sgld 逐 module 对齐分析

> **术语注**：`QwenImageTransformerBlock` 是 dual-stream DiT。两个流都在
> 同一个 transformer block 内部：
> - `hidden_states`：图像 latent 流（1024 个 image token）
> - `encoder_hidden_states`：**文本流**，是 prompt 经 text encoder 编完作为 input
>   喂进 DiT 的，**不是 text encoder 模块本身**。block 内部 `[txt, img]`
>   cat 起来跑 joint attention，再分回两路输出。
> - `encoder_hidden_states_mask`：**DiT joint attention 内的文本侧 mask**
>   （哪几个 text token 是真，哪几个是 padding），跟外部 text encoder 没关系
>
> 本文表里 `encoder rel_mean` = DiT block 文本流输出的 diff，
> `hidden rel_mean` = DiT block 图像流输出的 diff。两个都是 DiT 出来的。



2026-04-29 调查记录。目的：把同一份权重加载到 diffusers 和 sglang-diffusion
的 `QwenImageTransformerBlock` 上，跑同一组输入，对每个 submodule 的
input/output 装 forward hook，逐 module 比较 bf16 数值差异，找出 train
(diffusers) ↔ rollout (sglang-d) 训推不一致的真实分支点。

worktree：`feat/batching-loop-revamp-v2` →
`.claude/worktrees/layer-alignment-diffusers-vs-sgld`

> ⚠️ **文档勘误（2026-04-29 二轮）**：本文档第一版把 root cause 错归给
> "diffusers `nn.LayerNorm` 是 bf16 内部"。实测验证后这是错的：PyTorch
> 的 `F.layer_norm(bf16_input)` 已经是 fp32 internal，与
> `F.layer_norm(bf16_input.float())` bit-exact。真正的分歧在 modulate
> （norm 之后的 `(1+scale)*x + shift`）和残差加法上，而且**已经存在的
> `miles/backends/fsdp_utils/models/qwen_image_patch.py` 已经把 sgld
> 降到 bf16 modulate 与 diffusers 对齐，验证下两侧 bit-exact 0/0**。
> 详见末尾「修正与现有 patch 验证」一节。

## 测量配置

- block 0 of `Qwen/Qwen-Image` transformer
- 同一份 diffusers 权重 → copy 进 sgld block (32/32 参数全部 shape-match)
- 输入：bf16, B=1, S_img=1024, S_txt=35, hidden_dim=3072
- `image_rotary_emb=None`（短路 RoPE，避免叠加 RoPE-CPU-vs-CUDA 已知 drift）
- diffusers attention backend = NATIVE (默认)；sgld = TORCH_SDPA backend
- 单 GPU，bf16
- harness：`tools/layer_align_diff_vs_sgld.py`，记录全 dump 到
  `/tmp/layer_align_records.pt`，`tools/analyze_layer_align.py` 离线分析

## 顶层 block output diff（unpatched 双方）

|  | abs_max | abs_mean | rel_mean | norm_diff_rel |
|---|---|---|---|---|
| `encoder_out` (text stream) | 2.56e2 | 3.55 | **0.443%** | 0.029% |
| `hidden_out` (image stream) | 1.33e4 | 79.7 | **8.55%** | 1.82% |

`hidden_out` 比 `encoder_out` 大 19×（rel_mean），主要被 image-side joint
attention 的 softmax 放大效应拉开。

## attn 模块输出 diff（post `to_out` projection）

|  | rel_mean | norm_diff_rel | 说明 |
|---|---|---|---|
| img stream attn output | **27.1%** | 0.077% | 同 norm 不同元素 |
| txt stream attn output | **0.327%** | 0.0087% |  |

**asymmetry ratio (img / txt) = 82.9 ×**。norm 几乎相同（总能量一致）但
element-wise 巨幅差异 → 不是简单 bf16 round-off，是 attention softmax 把局部
扰动放大到 image 段。

## 逐 module 表（n=23 配对）

| diffusers | sgld | input rel_mean | output rel_mean | 备注 |
|---|---|---|---|---|
| `img_mod.1`            | `img_mod.1`            | **0** | **0** | bit-exact ✓ |
| `txt_mod.1`            | `txt_mod.1`            | **0** | **0** | bit-exact ✓ |
| `img_norm1`            | `img_norm1.norm`       | — | — | sgld 路径不走 `.norm.forward`（融合 triton kernel） |
| `img_norm2`            | `img_norm2.norm`       | — | — | 同上 |
| `txt_norm1` / `txt_norm2` | `.norm`             | — | — | 同上 |
| `attn.to_q/k/v`        | `attn.to_q/k/v`        | 0.15% | 0.17% | Linear bit-exact，diff 全部继承自 input |
| `attn.add_q_proj/...`  | `attn.add_q_proj/...`  | 0.14% | 0.17% | 同上 |
| `attn.norm_q/k`        | `attn.norm_q/k`        | (default fused-inplace 路径短路 sgld 的 RMSNorm.forward) |
| `attn.to_out.0`        | `attn.to_out.0`        | **12.7%** | 27.1% | 输入 = joint attn output[txt:] |
| `attn.to_add_out`      | `attn.to_add_out`      | 0.24% | 0.33% | 输入 = joint attn output[:txt] |
| `img_mlp.net.0.proj`   | `img_mlp.net.0.proj`   | 17.7% | 18.3% | 继承 attn output |
| `img_mlp.net.2`        | `img_mlp.net.2`        | 19.3% | 11.4% | |
| `txt_mlp.net.0/2.proj` | `txt_mlp.net.0/2.proj` | 0.27% | 0.38% | |

关键观察：
1. **所有 Linear 层 bit-exact**（含 mod、Q/K/V、to_out、MLP net.0/net.2）。
   权重 copy 严格逐 byte 一致 → cuBLAS gemm 在两侧给完全相同结果。
2. Pre-attn 各路 input 仅相差 ~0.15% rel —— 这一点 diff 来自 norm+modulate
   path 的差异，**但 norm 本身不是源**（见下方独立验证）。
3. Post-attn image 段被放大到 27%，text 段保持 0.33%，asymmetry 83×

## 三层独立隔离验证

### A. SDPA 本身在两侧 bit-exact

`tools/probe_sdpa_only.py`：把 sgld 记录的 q/k/v 同时喂进 (1) diffusers
`_native_attention` (2) sgld `SDPAImpl.forward`，跳过 qk_norm。

```
Full joint output diff:
  abs_max=0.000e+00  abs_mean=0.000e+00  rel_mean=0.000e+00
  txt half  abs_max=0.000e+00 abs_mean=0.000e+00 rel_mean=0.000e+00
  img half  abs_max=0.000e+00 abs_mean=0.000e+00 rel_mean=0.000e+00
[control] same path twice: abs_max=0.000e+00 abs_mean=0.000e+00
```

→ **F.scaled_dot_product_attention 在 NATIVE 和 SDPABackend 两条路径下完全
等价**。SDPA 本身不是 divergence 源。

### B. RMSNorm 在两侧 bit-exact

把同一份 q (bf16, (1, 1024, 24, 128)) 同时喂给：
- `diffusers.models.normalization.RMSNorm`（variance fp32 + 后续混 dtype）
- `sgld.multimodal_gen.runtime.layers.layernorm.RMSNorm.forward_native`
  (`SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1`，全 fp32)

```
output diff:
  abs_max  = 0.0000e+00
  abs_mean = 0.0000e+00
  rel_mean = 0.0000e+00
```

→ **RMSNorm 在两侧 bit-exact**。RMSNorm 本身不是 divergence 源。

### C. nn.LayerNorm 在两侧也 bit-exact ⚠️ 第一版结论错在这里

第一版假设 diffusers 的 `nn.LayerNorm(bf16_input, elementwise_affine=False)`
内部走 bf16，sgld 的 `FP32LayerNorm` 走 fp32 → 两边不同。**实测后这个
假设是错的**：

```python
ln = torch.nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6).to(DTYPE)
out_default = ln(x_bf16)
out_explicit_fp32 = F.layer_norm(x_bf16.float(), ..., None, None, eps).to(bf16)

(out_default - out_explicit_fp32).abs().max() = 0.000e+00   ← bit-exact!
```

PyTorch 的 LayerNorm CUDA kernel **在 bf16 输入上已经走 fp32 accumulator**，
然后 cast 回 bf16。所以 diffusers `nn.LayerNorm` 和 sgld `FP32LayerNorm` 的
norm 输出 bit-exact 一致。

第一版基于「nn.LayerNorm 是 bf16 内部」推断的 root cause 链不成立。

### D. attn.norm_q output 局部巨型 diff（仍然观察到）

```
attn.norm_q output:
  abs_max =3.175e+01    ← 单元素巨型 diff
  abs_mean=5.151e-03
  rel_mean=2.885e-03
```

机制不变：input 的 ~0.17% rel diff（来自 modulate 路径），通过 RMSNorm 的
`x * rsqrt(variance + eps)` 在低 variance head 上被放大上千倍。**RMSNorm
本身两侧 bit-exact**（验证 B），但它对 input diff 高度敏感，把上游 modulate
path 引入的小 diff 放大成局部巨型 diff。

## 真正的 divergence 来源

确认下来：**post-norm 的 modulate `(1+scale)*x + shift` + 残差加法
（gate*attn_out / gate*mlp_out）**

| 步骤 | diffusers 默认 | sgld 默认 |
|---|---|---|
| LayerNorm | nn.LayerNorm，**fp32 internal** （PyTorch 自动） | FP32LayerNorm，fp32 internal |
| `(1+scale)*x + shift` | bf16 eager (`* (1+scale.unsqueeze(1)) + shift.unsqueeze(1)`) | **CUTLASS fused 内 fp32 accumulator**，整段 norm+modulate 一个 kernel |
| `residual + gate*attn` (norm2 前) | bf16 eager | **fused 进 ScaleResidualLayerNormScaleShift**，整段 fp32 |
| `hidden + gate*mlp` (mlp 后) | bf16 eager | **MulAdd**，fp32 内部 |

差距是 modulate / 残差 这几步的 fp32 vs bf16，叠加到 60 block 上才在
hidden_out 体现成 8.55%。

## 修正与现有 patch 验证

`miles/backends/fsdp_utils/models/qwen_image_patch.py` 早就存在，做的方向
是 **sgld → diffusers**：把 sgld 的 fused fp32 modulate / 残差 / qk_norm
全部降回 bf16 eager。验证：

| variant | encoder_out rel_mean | hidden_out rel_mean |
|---|---|---|
| baseline (双方都 unpatched) | 4.43e-3 | **8.55%** |
| sgld 应用现有 patch | **0** | **0**（bit-exact ✓）|

→ **现有 patch 已经解决问题**。生产配置只要打开
`--apply-qwen-image-sgl-d-patch` 就行。

我同会话还试过反方向：写一个 `qwen_image_diffusers_patch.py`，把 diffusers
升到 fp32 fused 来匹配 sgld native。结果：

| variant | encoder_out rel_mean | hidden_out rel_mean |
|---|---|---|
| diffusers 应用反向 patch（sgld unpatched） | 3.21e-3 | **8.49%**（几乎没改善）|
| 双方都 patch | 3.41e-3 | 5.0e-3 |

eager Python 的 fp32 ops（`F.layer_norm` + `*` + `+` 各步独立 cast）**无法
bit-exact 复现 sgld 的 CUTLASS-fused kernel**（register-level rounding 不一致），
所以这个方向不是 bit-exact 可行的。该 patch 文件已删除，不必保留。

## 实测产出

- harness：`tools/layer_align_diff_vs_sgld.py`（生产 records）
- 离线分析：`tools/analyze_layer_align.py`
- SDPA 隔离测试：`tools/probe_sdpa_only.py`
- 对齐 v2（fused_inplace_qknorm 关 + RMSNorm forward_native）：
  `tools/layer_align_v2_with_alignments.py`
- sgld 最小 init helper：`tools/_sgld_minimal_init.py`
- 日志：`logs/layer_align_*.log`
- records 缓存：`/tmp/layer_align_records.pt`

## 教训（个人复盘）

第一版结论错在我假设 `nn.LayerNorm` 是 bf16 internal **没做对照实验就写**。
触发了 memory 里两条 feedback：
- `feedback_suspect_own_diagnostic_first.md` —— 看到 image vs text 80×
  非对称应该先怀疑自己的假设
- `feedback_always_include_control.md` —— 任何 root cause 推断必须有
  「关掉这个旋钮」的对照组验证

正确流程应当是：写文档前先把现有 `qwen_image_patch.py` 应用一遍跑出 0/0，
立刻就知道 root cause 是什么、修复方向是哪个。

## 后续二：production 真有一个 mask-path drift（已修）

应用现有 patch 后单 block forward `(B=1, mask=all-True)` 还差 1.18e-3。
追根：

- miles 训练侧 `collate_cond_for_sample_batch` 即使在 homogeneous microbatch
  （所有 sample 同 prompt，本来不需要 mask）也会发一个 all-True 的
  `encoder_hidden_states_mask`。
- diffusers DiT-level forward 看到 `encoder_hidden_states_mask is not None`
  → 构造 joint_attention_mask（也 all-True） → 通过 attention_mask kwarg
  传到 SDPA。
- SDPA 在 `attn_mask=all_True_bool` vs `attn_mask=None` 下输出 1 bf16 ULP
  级别的差异（实测 abs_max ~ 9.77e-4，abs_mean ~ 1e-7）。本身很小，但
  通过 60 个 transformer block 的 residual 累积放大成 hidden_out 1.18e-3。
- sgld rollout 永远不传 mask（一条一条过），走 `attn_mask=None` 路径。

→ 训推走的 SDPA 走入了不同 bf16 量化路径，per-block ~1.2e-3 drift，60 个
block 累积约 ~5-7e-3 noise_pred drift（跟
`DIFFUSION_PRECISION_ABLATION.md` 综合视图 5e-3 量级吻合，是其重要分量
之一）。

修法（已实施）：在 `miles/backends/fsdp_utils/configs/qwen_image.py` 的
`collate_cond_for_sample_batch` 里只在 **真有 padding** 时才发 mask：

```python
if min(seq_lens) == max_len:
    mask = None  # all sample 同长度 → 没 padding → 不需要 mask
```

verification (`tools/verify_skip_alltrue_mask_fix.py`)：

| variant | mask emitted | enc rel_mean | hid rel_mean |
|---|---|---|---|
| B=1 homogeneous | False | **0** | **0** (bit-exact ✓) |
| B=4 homogeneous | False | 1.69e-3 | 9.0e-4 |
| B=2 heterogeneous (28/60) | True | 5.13e-2 | 2.17e-3 |

- B=1：完全消除了 1.18e-3 mask-path drift
- B=4：剩下的 1.7e-3 是纯 cuBLAS batched-gemm reduction-order，跟 mask 无关
  （Ablation 2 同源，要消除得 train↔rollout 用同一 batch dim）
- 异构 case：mask 该发还是发，路径不变；production train 不会走到这里

---

## 后续一：mask + multi-batch 探查（不是 bug）

应用现有 patch 后再做了 multi-batch / 带 mask 的探查
（`tools/probe_patch_with_mask_and_batch.py`）：

| variant | enc rel_mean | hid rel_mean |
|---|---|---|
| b1_nomask | 1.18e-3 | 6.54e-4 |
| b1_mask | 1.33e-3 | 6.57e-4 |
| **b2_mask（异构 prompt 长度 [28, 60]）** | **7.94e-2** | 3.20e-3 |
| b2_allTrue | 1.74e-3 | 1.03e-3 |
| **b4_mask（异构长度 [28, 60, 35, 50]）** | **7.13e-2** | 5.41e-3 |

`b2_mask` / `b4_mask` 的 8% encoder 漂移看起来很大，但**这是测试 setup 偏离了
sgld 的实际工作负载，不是 patch 的 bug**：

- sgld production microgroup 是「同一 prompt × N 个不同 latent」一组，
  encoder_hidden_states 各 row 完全相同 → 不需要 padding → mask=None
- 现有 patch 的 `_patched_usp_attention_forward` 注释明确说了
  「only the ``encoder_hidden_states_mask=not None`` path needs a real mask;
  rank-0 workloads without padding go through this None path」
- diffusers 在 train side 如果接到异构 prompt microbatch 会构造 joint_mask，
  但 sgld 那一侧根本不会接到这种 batch

也就是说我合成的 b2_mask/b4_mask 输入在 production 不会真实出现。如果哪天
sgld 真要支持异构 prompt batch（比如 colocate 多 prompt），那时候才需要把
mask 从 `cross_attention_kwargs` 一路传到 USPAttention 并 reshape 成
(B, 1, 1, S)。现在不需要。

---

## 下次再做训推精度类调查的标准动作

1. **第一步**：在 sgld 上应用
   `apply_qwen_image_diffusers_parity_patches()`（生产配置开关
   `--apply-qwen-image-sgl-d-patch`）。这把 baseline 拉到 bit-exact 0/0。
2. **第二步**：在这个对齐 baseline 之上引入要研究的变量（LoRA merge / batch /
   CFG / RoPE 等），测出来的 drift 才是这个变量真实贡献的。
3. 不要直接拿 unpatched 双方比较 —— 8.55% 那个数会把所有未知 + 已知 +
   已修复的差异混在一起。
