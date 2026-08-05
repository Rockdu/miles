"""Qwen3.5/3.6/Qwen3-Next (GatedDeltaNet) adaptations.

* packing — a config-time packed-doc reset that feeds cu_seqlens to fla
  chunk/recurrent_gated_delta_rule and seq_idx to causal_conv1d_fn per packed document. Patches the
  DecoderLayer/GatedDeltaNet class forwards; kernel logic lives in ``models/qwen3_5_moe.py``.
* precision — the qwen3-dense recipe (fp32 gather + bf16 autocast) as the arch default; see
  ``_resolve_fp32_gather_precision``.
"""

import functools
from dataclasses import replace

import torch

from ..class_patches import ModelInstancePatchHook, register_model_instance_patch
from ..packing.registry import PackingPatch, register_packing_patch
from ..precision import PrecisionPolicyHook, register_precision_policy


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


def _precision_applies(hf_config, args) -> bool:
    """Default-on for GDN archs. fp16 runs and runs that disabled the fp32 master keep the plain
    policy instead of erroring: the recipe gathers the master, so without one (or under fp16) there
    is nothing correct to gather."""
    return _applies(hf_config) and not getattr(args, "fp16", False) and getattr(args, "keep_fp32_master", True)


def _resolve_precision(base_policy, hf_config, args):
    """The qwen3-dense precision recipe for GatedDeltaNet archs: fp32 gather + bf16 autocast.

    GDN's gating parameter ``A_log`` is checkpoint-declared fp32 (the only fp32 param family besides
    the gated-norm weights) and sglang's rollout holds it in an fp32 container. Under a bf16 gather
    the trained-effective value (bf16-rounded) diverges from the saved/deployed one (fp32 master) as
    soon as the master drifts off the bf16 grid — measured at up to 25% relative on a 128-token
    decay kernel for a 1e-3 drift, because the gate is ``-exp(A_log)·softplus(...)`` and the error
    compounds along the recurrence. ``A_log`` sits directly on the GatedDeltaNet module next to the
    deliberately-bf16 ``dt_bias``, so no module-granularity mechanism can single it out; gathering
    everything fp32 sidesteps that: the forward consumes the fp32 master, ``_resolve_sync_dtype``
    ships it fp32 into rollout's fp32 container, and training, rollout, and deployment share one
    value. autocast keeps the matmuls at bf16 — the same rounding of the same master that the bf16
    weight sync applies — so Linear layers stay aligned with rollout even after drift.

    Costs and behavior mirror qwen3 dense's true-on-policy policy: fp32 all-gather doubles param
    communication and unsharded memory, and non-autocast ops (norms, residual adds) compute on the
    fp32 stream.
    """
    return replace(
        base_policy,
        param_dtype=torch.float32,
        autocast_dtype=torch.bfloat16,
        # autocast owns the compute dtype: without this, FSDP re-promotes the activation stream to
        # fp32 at every wrap-unit boundary and the norms/residual adds run on an fp32 stream the
        # rollout engine never sees (measured: 527/1074 fp32 activations vs rollout's 0).
        cast_forward_inputs=False,
        sync_dtype_resolver=_resolve_sync_dtype,
    )


def _resolve_sync_dtype(name: str, checkpoint_dtype: torch.dtype) -> torch.dtype:
    """A_log syncs fp32: rollout's container is fp32 (sglang builds it so explicitly), and the
    training forward consumes the fp32 master — a bf16 sync would silently re-quantize the one
    parameter this recipe exists to protect. Everything else keeps its checkpoint dtype: the other
    rollout containers are bf16, so an fp32 sync would just be rounded at copy_ anyway."""
    if name.endswith(".linear_attn.A_log"):
        return torch.float32
    return checkpoint_dtype


def _apply_embed_compute_cast(model) -> bool:
    """Pin the model's float input boundary to the compute dtype.

    The token ids are ints, so the real float boundary is the embedding output — and ``F.embedding``
    is not autocast-covered, so under fp32 gather it emits the fp32 master and seeds an fp32 stream
    that ``Qwen3_5RMSNorm``'s ``type_as`` and every residual add then propagate end to end
    (measured: 29 fp32 activation names with the boundary cast alone disabled). Casting the output
    to bf16 is value-identical to what the rollout engine computes: bf16(master) is exactly the
    rounding the bf16 weight sync applies before its own bf16 embedding lookup.
    """
    embed = model.get_input_embeddings()
    if getattr(embed, "_gdn_compute_cast", False):
        return False
    orig_forward = embed.forward

    @functools.wraps(orig_forward)
    def forward(*args, **kwargs):
        return orig_forward(*args, **kwargs).to(torch.bfloat16)

    embed.forward = forward
    embed._gdn_compute_cast = True
    return True


register_packing_patch(PackingPatch("gated_deltanet_packing", _applies, "config", _apply))
register_precision_policy(PrecisionPolicyHook("gated_deltanet_fp32_gather", _precision_applies, _resolve_precision))
register_model_instance_patch(
    ModelInstancePatchHook("gated_deltanet_embed_compute_cast", _precision_applies, _apply_embed_compute_cast)
)
