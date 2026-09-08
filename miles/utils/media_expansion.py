"""How many LM positions each media placeholder in a prompt expands to.

The chat template emits one placeholder token per media item. The rollout engine and
the train actor each expand that placeholder into the item's token count, so miles
computes the count in exactly one place: here. The controller ships the counts with
the sample, the train actor expands from them, and `Sample.update_from_meta_info`
checks them against the prompt length the engine reports.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.hf_config import load_hf_config

KIMI_MEDIA_PLACEHOLDER_TOKEN_ID = 163605
INKLING_IMAGE_SENTINEL_ID = -101
INKLING_AUDIO_SENTINEL_ID = -102
# Kimi's tpool_patch_merger folds every 2x2 spatial patch block into one token and averages over T.
_KIMI_MERGE_SIZE = 2

_QWEN_VL_MODEL_TYPES = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe")
_KIMI_MODEL_TYPES = ("kimi_vl", "kimi_k25")
_INKLING_MODEL_TYPES = ("inkling_mm_model",)


@dataclass(frozen=True)
class MediaExpansionSpec:
    # Token ids the chat template emits once per media item.
    placeholder_token_ids: tuple[int, ...]
    # The Qwen-VL HF processors rewrite the prompt text themselves, so the controller
    # tokenizes without them to keep one placeholder per item.
    hf_processor_expands_prompt: bool
    # Per-item token counts in prompt order, read from the tensors training receives.
    counts_from_train_inputs: Callable[[dict], list[int]]

    def token_counts(self, multimodal_train_inputs: dict | None) -> list[int]:
        if not multimodal_train_inputs:
            return []
        return self.counts_from_train_inputs(multimodal_train_inputs)


def _qwen_vl_counts(train_inputs: dict, merge_size: int) -> list[int]:
    return [t * h * w // (merge_size * merge_size) for t, h, w in train_inputs["image_grid_thw"].tolist()]


def _kimi_counts(train_inputs: dict) -> list[int]:
    return [(h // _KIMI_MERGE_SIZE) * (w // _KIMI_MERGE_SIZE) for _, h, w in train_inputs["grid_thws"].tolist()]


def _inkling_counts(train_inputs: dict) -> list[int]:
    return [
        int(count)
        for key in ("mm_vision_num_patches", "mm_audio_num_tokens")
        if key in train_inputs
        for count in train_inputs[key]
    ]


def media_expansion_spec(hf_config) -> MediaExpansionSpec | None:
    """The spec for a model family, or None for models whose prompts miles does not expand."""
    model_type = hf_config.model_type
    if model_type in _QWEN_VL_MODEL_TYPES:
        merge_size = hf_config.vision_config.spatial_merge_size
        return MediaExpansionSpec(
            placeholder_token_ids=(hf_config.image_token_id,),
            hf_processor_expands_prompt=True,
            counts_from_train_inputs=functools.partial(_qwen_vl_counts, merge_size=merge_size),
        )
    if model_type in _KIMI_MODEL_TYPES:
        return MediaExpansionSpec(
            placeholder_token_ids=(KIMI_MEDIA_PLACEHOLDER_TOKEN_ID,),
            hf_processor_expands_prompt=False,
            counts_from_train_inputs=_kimi_counts,
        )
    if model_type in _INKLING_MODEL_TYPES:
        # Inkling sentinels are expanded by its own train path; the spec only supplies counts.
        return MediaExpansionSpec(
            placeholder_token_ids=(INKLING_IMAGE_SENTINEL_ID, INKLING_AUDIO_SENTINEL_ID),
            hf_processor_expands_prompt=False,
            counts_from_train_inputs=_inkling_counts,
        )
    return None


@functools.cache
def media_expansion_spec_for_checkpoint(hf_checkpoint: str) -> MediaExpansionSpec | None:
    return media_expansion_spec(load_hf_config(hf_checkpoint))
