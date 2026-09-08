"""Per-media expansion counts must match what each model's processor would emit.

    hf_config.model_type ──► media_expansion_spec ──► placeholder ids + counts(train_inputs)

    qwen*      image_grid_thw (t,h,w)  ──► t*h*w // merge²      (T is a real axis)
    kimi*      grid_thws     (t,h,w)  ──► (h//2)*(w//2)         (T is averaged away)
    inkling    mm_vision_num_patches ++ mm_audio_num_tokens      (counts come pre-computed)
    others     None                                             (miles does not expand them)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.utils.media_expansion import KIMI_MEDIA_PLACEHOLDER_TOKEN_ID, media_expansion_spec


def _qwen_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(model_type=model_type, image_token_id=151655, vision_config=SimpleNamespace(spatial_merge_size=2))


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen3_vl", "qwen3_5"])
def test_qwen_counts_multiply_all_three_grid_axes(model_type):
    spec = media_expansion_spec(_qwen_config(model_type))
    counts = spec.token_counts({"image_grid_thw": torch.tensor([[1, 16, 16], [2, 4, 6]])})
    assert spec.placeholder_token_ids == (151655,)
    assert spec.hf_processor_expands_prompt is True
    assert counts == [64, 12]


def test_kimi_counts_ignore_the_temporal_axis():
    spec = media_expansion_spec(SimpleNamespace(model_type="kimi_k25"))
    counts = spec.token_counts({"grid_thws": torch.tensor([[3, 8, 10]])})
    assert spec.placeholder_token_ids == (KIMI_MEDIA_PLACEHOLDER_TOKEN_ID,)
    assert spec.hf_processor_expands_prompt is False
    assert counts == [20]


def test_inkling_counts_concatenate_vision_then_audio():
    spec = media_expansion_spec(SimpleNamespace(model_type="inkling_mm_model"))
    counts = spec.token_counts({"mm_vision_num_patches": torch.tensor([7, 9]), "mm_audio_num_tokens": torch.tensor([5])})
    assert counts == [7, 9, 5]
    assert spec.token_counts({"mm_vision_num_patches": torch.tensor([7])}) == [7]


def test_text_models_have_no_spec_and_no_counts():
    assert media_expansion_spec(SimpleNamespace(model_type="qwen3")) is None
    assert media_expansion_spec(_qwen_config("qwen3_vl")).token_counts(None) == []
