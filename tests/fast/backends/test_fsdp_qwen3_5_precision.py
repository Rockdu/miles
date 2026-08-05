"""Qwen3.5 (GatedDeltaNet) default precision recipe: fp32 gather + bf16 autocast."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import torch

from miles.backends.experimental.fsdp_utils.adaptations.precision import resolve_precision_policy

QWEN3_5 = SimpleNamespace(model_type="qwen3_5")


def _args(**overrides):
    base = dict(fp16=False, keep_fp32_master=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_is_fp32_gather_with_bf16_autocast():
    policy = resolve_precision_policy(QWEN3_5, _args())
    assert policy.param_dtype is torch.float32
    assert policy.autocast_dtype is torch.bfloat16
    assert policy.keep_fp32_master


def test_sync_ships_a_log_fp32():
    resolver = resolve_precision_policy(QWEN3_5, _args()).sync_dtype_resolver
    assert resolver("model.layers.0.linear_attn.A_log", torch.bfloat16) is torch.float32
    assert resolver("model.layers.0.linear_attn.dt_bias", torch.bfloat16) is torch.bfloat16
    assert resolver("model.layers.0.mlp.gate_proj.weight", torch.bfloat16) is torch.bfloat16


def test_layer_types_config_matches_too():
    hybrid = SimpleNamespace(model_type="qwen3_next", layer_types=["linear_attention", "full_attention"])
    policy = resolve_precision_policy(hybrid, _args())
    assert policy.param_dtype is torch.float32


def test_fp16_falls_back_to_plain_policy():
    policy = resolve_precision_policy(QWEN3_5, _args(fp16=True))
    assert policy.param_dtype is torch.float16
    assert policy.autocast_dtype is None
    assert policy.sync_dtype_resolver is None


def test_disabled_master_falls_back_to_plain_policy():
    policy = resolve_precision_policy(QWEN3_5, _args(keep_fp32_master=False))
    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None


def test_does_not_leak_to_other_archs():
    policy = resolve_precision_policy(SimpleNamespace(model_type="qwen3"), _args())
    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None
