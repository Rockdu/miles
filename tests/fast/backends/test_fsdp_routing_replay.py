"""Rollout routing replay (R3) unit test for the FSDP glm4_moe_lite patch (CPU-only).

One toy MoE (8 experts, top-2) driven through the three manager stages:

    stage            expert selection          gate weights
    ---------------  ------------------------  ----------------------------------
    fallthrough      own logits (== stock HF)  own logits
    record           own logits, recorded      own logits
    replay_forward   popped from the stream    own logits at the replayed experts

The replay_forward row is the R3 contract: rollout picks the experts, training
computes the weights (sigmoid -> gather -> renorm -> routed_scaling).
"""

import pytest
import torch

from miles.backends.experimental.fsdp_utils.models.glm4_moe_lite import apply_routing_replay_patch
from miles.utils.replay_base import routing_replay_manager
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

glm = pytest.importorskip("transformers.models.glm4_moe_lite.modeling_glm4_moe_lite")

N_EXPERTS, TOP_K, HIDDEN, TOKENS = 8, 2, 16, 5
# Replay.pop_* returns cuda tensors whenever cuda exists, so the module must live there too.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def moe():
    from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig

    config = Glm4MoeLiteConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=8,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.8,
        num_hidden_layers=2,
        first_k_dense_replace=1,
    )
    torch.manual_seed(0)
    stock_route_fn = glm.Glm4MoeLiteMoE.route_tokens_to_experts
    module = glm.Glm4MoeLiteMoE(config).to(DEVICE)

    class Model:  # the layer walk in apply_routing_replay_patch expects model.model.layers[i].mlp
        pass

    layer0, layer1 = Model(), Model()
    layer0.mlp, layer1.mlp = glm.Glm4MoeLiteMLP(config), module
    model = Model()
    model.model = Model()
    model.model.layers = [layer0, layer1]

    routing_replay_manager.enabled = True
    routing_replay_manager.stage = "fallthrough"
    num_registered = apply_routing_replay_patch(model, config)
    assert num_registered == 1

    yield module, stock_route_fn

    routing_replay_manager.enabled = False
    routing_replay_manager.stage = "fallthrough"
    routing_replay_manager.replays.clear()
    routing_replay_manager.current = None
    glm.Glm4MoeLiteMoE.route_tokens_to_experts = stock_route_fn


def test_fallthrough_matches_stock_hf(moe):
    module, stock_route_fn = moe
    logits = torch.randn(TOKENS, N_EXPERTS, device=DEVICE)

    patched_indices, patched_weights = module.route_tokens_to_experts(logits)
    stock_indices, stock_weights = stock_route_fn(module, logits)

    assert torch.equal(patched_indices.sort(-1).values, stock_indices.sort(-1).values)
    assert torch.allclose(patched_weights.sum(-1), stock_weights.sum(-1))


def test_replay_forward_uses_rollout_experts_with_own_weights(moe):
    module, _ = moe
    logits = torch.randn(TOKENS, N_EXPERTS, device=DEVICE)
    # deliberately NOT the argmax experts, so a pass proves substitution happened
    rollout_indices = torch.stack([torch.randperm(N_EXPERTS)[:TOP_K] for _ in range(TOKENS)]).to(DEVICE)

    replay = routing_replay_manager.replays[0]
    replay.record(rollout_indices)
    routing_replay_manager.set_current(replay)
    routing_replay_manager.stage = "replay_forward"

    indices, weights = module.route_tokens_to_experts(logits)

    assert torch.equal(indices, rollout_indices)
    expected = logits.sigmoid().gather(1, rollout_indices)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 1.8
    assert torch.allclose(weights, expected)


def test_replay_forward_padded_tokens_fall_back_to_arange(moe):
    module, _ = moe
    logits = torch.randn(TOKENS, N_EXPERTS, device=DEVICE)
    padded = torch.full((TOKENS, TOP_K), -1, dtype=torch.long)

    replay = routing_replay_manager.replays[0]
    replay.record(padded)
    routing_replay_manager.set_current(replay)
    routing_replay_manager.stage = "replay_forward"

    indices, weights = module.route_tokens_to_experts(logits)

    assert torch.equal(indices.cpu(), torch.arange(TOP_K).expand(TOKENS, -1))
    assert weights.shape == (TOKENS, TOP_K)


def test_forward_pre_hook_records_through_module_call(moe):
    module, _ = moe
    routing_replay_manager.stage = "record"

    module(torch.randn(TOKENS, HIDDEN, device=DEVICE))

    replay = routing_replay_manager.replays[0]
    assert len(replay.top_indices_list) == 1
    assert replay.top_indices_list[0].shape == (TOKENS, TOP_K)
