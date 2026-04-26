"""QwenImage training pipeline config."""

from __future__ import annotations

import torch
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

def _rebuild_pos_embed_freqs_on_cuda(model) -> None:
    """Rebuild QwenEmbedRope ``pos_freqs`` / ``neg_freqs`` on the model's
    CUDA device so train-side (diffusers) matches rollout-side (sglang-d)
    bit-exactly.

    diffusers' ``QwenEmbedRope.__init__`` builds these caches on CPU
    (``torch.arange(4096)`` with no ``device=``), and its ``forward``
    only ``.to(device)``s them — values stay CPU-computed.  sglang-d
    meta-inits and its forward rebuilds on CUDA.  CPU vs CUDA
    ``torch.pow`` differ by fp32 ULPs, so the two caches byte-differ →
    RoPE output differs → every block's output drifts → frozen-weight
    ``noise_pred`` mean|Δ| ~2e-02.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type != "cuda":
        return
    for submod in model.modules():
        # Match by attribute shape rather than class name so we also
        # handle ``QwenEmbedLayer3DRope`` and similar variants
        if not (
            hasattr(submod, "pos_freqs")
            and hasattr(submod, "neg_freqs")
            and hasattr(submod, "rope_params")
            and hasattr(submod, "axes_dim")
            and hasattr(submod, "theta")
        ):
            continue
        theta = submod.theta

        def _params(index: torch.Tensor, dim: int) -> torch.Tensor:
            inv = 1.0 / torch.pow(
                theta,
                torch.arange(0, dim, 2, device=device).to(torch.float32).div(dim),
            )
            freqs = torch.outer(index, inv)
            return torch.polar(torch.ones_like(freqs), freqs)

        pos_idx = torch.arange(4096, device=device)
        neg_idx = torch.arange(4096, device=device).flip(0) * -1 - 1
        submod.pos_freqs = torch.cat(
            [_params(pos_idx, d) for d in submod.axes_dim], dim=1
        )
        submod.neg_freqs = torch.cat(
            [_params(neg_idx, d) for d in submod.axes_dim], dim=1
        )
        # clear @lru_cache function
        cvf = getattr(submod, "_compute_video_freqs", None)
        if cvf is not None and hasattr(cvf, "cache_clear"):
            cvf.cache_clear()


@register_train_pipeline_config("Qwen/Qwen-Image")
class QwenImageTrainPipelineConfig(TrainPipelineConfig):

    lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "img_mlp.net.0.proj", "img_mlp.net.2",
        "txt_mlp.net.0.proj", "txt_mlp.net.2",
    ]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs = {}
        if cond.encoder_hidden_states:
            enc = torch.cat(cond.encoder_hidden_states).to(device)
            # Ensure batch dimension: (seq_len, dim) → (1, seq_len, dim)
            if enc.ndim == 2:
                enc = enc.unsqueeze(0)
            kwargs["encoder_hidden_states"] = enc
        if cond.txt_seq_lens:
            kwargs["txt_seq_lens"] = cond.txt_seq_lens
        if cond.img_shapes:
            kwargs["img_shapes"] = cond.img_shapes
        return kwargs

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """CFG matching sglang-d's Qwen-Image rollout.

        Mirrors ``QwenImagePipelineConfig.postprocess_cfg_noise`` in sglang-d:
          - pick ``true_cfg_scale`` if set, else ``guidance_scale``
          - combine with ``uncond + scale * (cond - uncond)``
          - **rescale to cond_norm only when ``true_cfg_scale > 1.0``**
        """
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        combined = noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)
        if true_cfg_scale is not None and true_cfg_scale > 1.0:
            pos_norm = torch.norm(noise_pred_pos, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True)
            combined = combined * (pos_norm / combined_norm)
        return combined

    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        """Preprocess the model before FSDP."""
        _rebuild_pos_embed_freqs_on_cuda(model)