"""Qwen3.5 (GatedDeltaNet) true-on-policy precision recipe: fp32 gather + bf16 autocast."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.experimental.fsdp_utils.adaptations.precision import resolve_precision_policy

QWEN3_5 = SimpleNamespace(model_type="qwen3_5")


def _args(**overrides):
    base = dict(fp16=False, keep_fp32_master=True, true_on_policy_mode=False, fsdp_precision_rules=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_policy_is_untouched():
    policy = resolve_precision_policy(QWEN3_5, _args())
    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None
    assert policy.sync_dtype_resolver is None
    assert policy.precision_spec.rules == ()


def test_true_on_policy_resolves_fp32_gather_with_bf16_autocast():
    policy = resolve_precision_policy(QWEN3_5, _args(true_on_policy_mode=True))
    assert policy.param_dtype is torch.float32
    assert policy.autocast_dtype is torch.bfloat16
    assert policy.keep_fp32_master
    # the embed hook composes on top: embedding gathers at the compute dtype
    assert [(r.select.fqn, r.gather) for r in policy.precision_spec.rules] == [("*embed_tokens", "bf16")]


def test_true_on_policy_sync_ships_a_log_fp32():
    policy = resolve_precision_policy(QWEN3_5, _args(true_on_policy_mode=True))
    resolver = policy.sync_dtype_resolver
    assert resolver("model.layers.0.linear_attn.A_log", torch.bfloat16) is torch.float32
    assert resolver("model.layers.0.linear_attn.dt_bias", torch.bfloat16) is torch.bfloat16
    assert resolver("model.layers.0.mlp.gate_proj.weight", torch.bfloat16) is torch.bfloat16


def test_true_on_policy_rejects_fp16():
    with pytest.raises(ValueError, match="requires bf16 training"):
        resolve_precision_policy(QWEN3_5, _args(true_on_policy_mode=True, fp16=True))


def test_true_on_policy_rejects_disabled_fp32_master():
    with pytest.raises(ValueError, match="requires fp32 master weights"):
        resolve_precision_policy(QWEN3_5, _args(true_on_policy_mode=True, keep_fp32_master=False))


def test_does_not_leak_to_other_archs():
    policy = resolve_precision_policy(SimpleNamespace(model_type="qwen3"), _args(true_on_policy_mode=True))
    assert policy.param_dtype is torch.bfloat16
    assert policy.precision_spec.rules == ()
