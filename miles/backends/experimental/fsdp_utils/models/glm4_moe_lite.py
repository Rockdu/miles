"""Rollout routing replay (R3, arXiv:2510.11370) for HF glm4_moe_lite under FSDP.

The training forward replays the rollout engine's per-token expert choices instead of
re-deriving them from its own router logits, removing routing flips as a train/rollout
mismatch source. Only the top-k selection goes through the shared routing_replay_manager
stage machine (mirroring Megatron's wrap of _compute_topk in moe_utils); the gate weights
are still computed from this module's own logits, so the gradient path through the router
is unchanged. The rollout engine must record un-fused routed experts
(--sglang-disable-shared-experts-fusion), otherwise the recorded top-k width includes the
fused shared expert and the replay width assert fires.
"""

import functools

import torch

from miles.utils.replay_base import routing_replay_manager


def _patch_route_tokens_to_experts(moe_cls):
    orig = moe_cls.route_tokens_to_experts
    if getattr(orig, "_routing_replay", False):
        return

    select_topk = routing_replay_manager.get_topk_fn(
        lambda scores, topk: torch.topk(scores, k=topk, dim=-1, sorted=False), return_probs=True
    )

    @functools.wraps(orig)
    def route_tokens_to_experts(self, router_logits):
        scores = router_logits.sigmoid()
        _, topk_indices = select_topk(scores + self.gate.e_score_correction_bias, self.top_k)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_indices, topk_weights * self.routed_scaling_factor

    route_tokens_to_experts._routing_replay = True
    moe_cls.route_tokens_to_experts = route_tokens_to_experts


def apply_routing_replay_patch(model, hf_config) -> int:
    """Patch the MoE class and register each MoE layer as a replay stream.

    Streams are keyed by absolute layer index: the rollout records one stream per model
    layer (dense layers included), same convention as Megatron's register_replay_list_moe.
    """
    assert hf_config.model_type == "glm4_moe_lite", f"routing replay not implemented for {hf_config.model_type}"
    # The patched selection is a plain top-k; group-limited routing degenerates to it only for one group.
    assert hf_config.n_group == 1, f"n_group={hf_config.n_group} checkpoints need group-limited top-k"

    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMoE

    _patch_route_tokens_to_experts(Glm4MoeLiteMoE)
    num_registered = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, Glm4MoeLiteMoE):
            routing_replay_manager.register_to_module(layer.mlp, "routing_replay", stream_idx=layer_idx)
            num_registered += 1
    assert num_registered > 0, "no Glm4MoeLiteMoE layers found"
    return num_registered
