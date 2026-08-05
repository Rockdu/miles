"""Qwen3.5/3.6/Qwen3-Next (GatedDeltaNet) adaptations.

Three hooks:

* packing — a config-time packed-doc reset that feeds cu_seqlens to fla
  chunk/recurrent_gated_delta_rule and seq_idx to causal_conv1d_fn per packed document. Patches the
  DecoderLayer/GatedDeltaNet class forwards; kernel logic lives in ``models/qwen3_5_moe.py``.
* precision (fp32 gather, arch default) — the qwen3-dense recipe: fp32 gather + bf16 autocast,
  plus an fp32 weight-sync override for ``A_log``. See ``_resolve_fp32_gather_precision`` for
  exactly which train/rollout gaps this closes and which it leaves open.
* precision (embed) — pin the token embedding to the compute dtype whenever gather and compute
  disagree; composes with the hook above (registration order matters: this one reads the
  autocast/gather dtypes the previous hook decided).
"""

from dataclasses import replace

import torch

from ..packing.registry import PackingPatch, register_packing_patch
from ..precision import ModuleSel, PrecisionPolicyHook, PrecisionSpec, Rule, dtype_name, register_precision_policy


def _applies(hf_config) -> bool:
    """True for GatedDeltaNet archs (Qwen3.5/3.6, Qwen3-Next): a linear_attention layer type or qwen3_5."""
    if hf_config is None:
        return False
    model_type = str(getattr(hf_config, "model_type", "") or "")
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return (layer_types is not None and "linear_attention" in layer_types) or "qwen3_5" in model_type


def _apply():
    from ...models.qwen3_5_moe import apply_gateddeltanet_packing_patch

    return apply_gateddeltanet_packing_patch()


def _fp32_gather_applies(hf_config, args) -> bool:
    """Default-on for GDN archs. fp16 runs and runs that disabled the fp32 master keep the old
    bf16-gather behavior instead of erroring: the recipe consumes the master, so without one (or
    under fp16) there is nothing correct to gather."""
    return _applies(hf_config) and not getattr(args, "fp16", False) and getattr(args, "keep_fp32_master", True)


def _resolve_fp32_gather_precision(base_policy, hf_config, args):
    """The qwen3-dense precision recipe for GatedDeltaNet archs: fp32 gather + bf16 autocast.

    What this closes (all verified by measurement on Qwen3.5-4B, 2026-08-04):

    * **A_log end-to-end.** The checkpoint declares ``A_log`` fp32 and sglang holds it in an fp32
      container, so under bf16 gather the trained-effective value (bf16-rounded) diverged from the
      saved/deployed one (fp32 master) as soon as the master drifted off the bf16 grid — measured at
      up to 25% relative on a 128-token decay kernel for a 1e-3 drift. With fp32 gather the forward
      consumes the fp32 master directly, and ``_resolve_sync_dtype`` ships it fp32 into rollout's
      fp32 container: training, rollout, and deployment now share one value.
    * **Matmuls stay aligned even after drift.** autocast's bf16 cast of the fp32-gathered weight is
      the same rounding as the bf16 weight sync, so every Linear on the two sides keeps consuming
      identical values.

    What this deliberately leaves open: sglang's containers for the norm weights, ``dt_bias`` and
    ``conv1d.weight`` are bf16, so after drift those ops see fp32 masters on the training side but
    bf16-synced values on the rollout side (~0.1-0.4% systematic). Closing that gap needs a
    qwen3.5 rollout contract on the sglang side (fp32 containers), as qwen3 dense's contract did.

    Costs mirror qwen3 dense: fp32 all-gather doubles param communication and unsharded memory.
    This is the arch default (no flag): opting out means ``--fp16`` or ``--disable-fp32-master``,
    both of which fall back to the plain bf16/fp16-gather policy.
    """
    return replace(
        base_policy,
        param_dtype=torch.float32,
        autocast_dtype=torch.bfloat16,
        sync_dtype_resolver=_resolve_sync_dtype,
    )


def _resolve_sync_dtype(name: str, checkpoint_dtype: torch.dtype) -> torch.dtype:
    """A_log syncs fp32: rollout's container is fp32 (qwen3_5.py builds it so explicitly), and the
    training forward now consumes the fp32 master — a bf16 sync would silently re-quantize the one
    parameter this recipe exists to protect. Everything else keeps its checkpoint dtype: the other
    rollout containers are bf16, so an fp32 sync would just be rounded at copy_ anyway."""
    if name.endswith(".linear_attn.A_log"):
        return torch.float32
    return checkpoint_dtype


def _precision_applies(hf_config, args) -> bool:
    return _applies(hf_config)


def _resolve_precision(base_policy, hf_config, args):
    """Gather the token embedding at the compute dtype when it differs from the gather dtype.

    ``F.embedding`` is not an autocast-covered op, so the embedding output carries the *gathered*
    weight dtype, and it seeds the residual stream. ``Qwen3_5RMSNorm`` ends in ``output.type_as(x)``
    and every residual add promotes, so that one dtype propagates through the whole activation path:
    an fp32-gather run computes its norms and residual adds in fp32 while autocast runs the matmuls
    at the compute dtype, which is exactly the train/rollout mismatch this pins shut. Matmul weights
    keep the run's gather dtype — only the embedding is moved.

    No-op under the default policy, where compute is the gather dtype and nothing disagrees.
    """
    compute_dtype = base_policy.autocast_dtype
    if compute_dtype is None or compute_dtype == base_policy.param_dtype:
        return base_policy
    rule = Rule(ModuleSel(fqn="*embed_tokens"), gather=dtype_name(compute_dtype))
    return replace(base_policy, precision_spec=PrecisionSpec(rules=base_policy.precision_spec.rules + (rule,)))


register_packing_patch(PackingPatch("gated_deltanet_packing", _applies, "config", _apply))
# Order matters: the fp32-gather hook decides gather/autocast dtypes; the embed hook reads them.
register_precision_policy(
    PrecisionPolicyHook("gated_deltanet_fp32_gather", _fp32_gather_applies, _resolve_fp32_gather_precision)
)
register_precision_policy(PrecisionPolicyHook("gated_deltanet_embed_gather", _precision_applies, _resolve_precision))
