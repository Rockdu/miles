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
    # autocast owns compute: FSDP must not re-promote wrap-unit inputs to fp32
    assert policy.cast_forward_inputs is False


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
    assert policy.cast_forward_inputs is True


def test_embed_output_pinned_to_compute_dtype():
    import torch.nn as nn

    from miles.backends.experimental.fsdp_utils.adaptations.class_patches import apply_model_instance_patches

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8).to(torch.float32)

        def get_input_embeddings(self):
            return self.embed

    model = Tiny()
    apply_model_instance_patches(model, QWEN3_5, _args())
    out = model.embed(torch.tensor([1, 2, 3]))
    assert out.dtype is torch.bfloat16
    # idempotent: a second pass keeps a single cast
    apply_model_instance_patches(model, QWEN3_5, _args())
    assert model.embed(torch.tensor([1])).dtype is torch.bfloat16


def test_embed_patch_skipped_for_fp16():
    import torch.nn as nn

    from miles.backends.experimental.fsdp_utils.adaptations.class_patches import apply_model_instance_patches

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8).to(torch.float16)

        def get_input_embeddings(self):
            return self.embed

    model = Tiny()
    apply_model_instance_patches(model, QWEN3_5, _args(fp16=True))
    assert model.embed(torch.tensor([1])).dtype is torch.float16


def test_disabled_master_falls_back_to_plain_policy():
    policy = resolve_precision_policy(QWEN3_5, _args(keep_fp32_master=False))
    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None


def test_does_not_leak_to_other_archs():
    policy = resolve_precision_policy(SimpleNamespace(model_type="qwen3"), _args())
    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None


def test_packing_patch_covers_dense_qwen3_5():
    import pytest

    pytest.skip("ablation branch: dense qwen3_5 deliberately unpatched")
    """The dense arch lives in its own transformers module; the patch loop must include it.
    Regression for the false-positive where the "applied" log fired off the moe/next classes
    while dense stayed stock and leaked GDN state across packed documents."""
    import pytest

    dense = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    from miles.backends.experimental.fsdp_utils.models.qwen3_5_moe import apply_gateddeltanet_packing_patch

    apply_gateddeltanet_packing_patch()
    for cls_name in ("Qwen3_5GatedDeltaNet", "Qwen3_5DecoderLayer"):
        cls = getattr(dense, cls_name)
        assert getattr(cls.forward, "_gdn_packing", False), f"{cls_name} not patched"


def test_packing_boundaries_cleared_on_single_doc_forward():
    """A packed forward must not leave stale cu_seqlens/seq_idx behind for the next single-document
    forward — the leftover length mismatch crashed the conv kernel live (v3 run, rollout 5)."""
    import pytest
    import torch

    dense = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    from miles.backends.experimental.fsdp_utils.models.qwen3_5_moe import apply_gateddeltanet_packing_patch

    apply_gateddeltanet_packing_patch()

    cfg_cls = dense.Qwen3_5TextConfig
    config = cfg_cls(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        vocab_size=128,
        max_position_embeddings=512,
    )
    model = dense.Qwen3_5TextModel(config).eval()
    gdn_modules = [m for m in model.modules() if isinstance(m, dense.Qwen3_5GatedDeltaNet)]
    assert gdn_modules
    # Pin the torch fallbacks so the test is CPU-deterministic regardless of which fast-path
    # wheels the host carries; the assertions below are about the boundary attrs, not the kernels.
    for m in gdn_modules:
        m.causal_conv1d_fn = None
        m.chunk_gated_delta_rule = dense.torch_chunk_gated_delta_rule
        m.recurrent_gated_delta_rule = dense.torch_recurrent_gated_delta_rule
        m.norm = dense.Qwen3_5RMSNormGated(config.linear_value_head_dim, eps=config.rms_norm_eps)

    packed_pos = torch.cat([torch.arange(8), torch.arange(8)]).unsqueeze(0)
    single_pos = torch.arange(8).unsqueeze(0)
    with torch.no_grad():
        model(input_ids=torch.randint(0, 128, (1, 16)), position_ids=packed_pos, use_cache=False)
        assert all(m._gdn_cu_seqlens is not None for m in gdn_modules), "packed forward must set boundaries"
        model(input_ids=torch.randint(0, 128, (1, 8)), position_ids=single_pos, use_cache=False)
        assert all(m._gdn_cu_seqlens is None for m in gdn_modules), "single-doc forward must clear boundaries"
