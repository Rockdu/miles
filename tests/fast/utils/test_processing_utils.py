"""Unit tests for processing_utils: multimodal train-input extraction and raw-id prompt encoding.

encode_multimodal_prompt, spec.hf_processor_expands_prompt = True (Qwen-VL family):

    prompt text ──tokenizer──► prompt_ids   (one <|image_pad|> per image, the HF processor never sees the text)
    images ──image_processor──► train_inputs {pixel_values, image_grid_thw}
    train_inputs ──spec──► counts [t*h*w // merge²]
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import torch

from miles.utils.media_expansion import media_expansion_spec
from miles.utils.processing_utils import encode_multimodal_prompt, extract_multimodal_train_inputs

IMAGE_PAD = 151655


class _FakeTokenizer:
    def encode(self, text, add_special_tokens):
        return [{"a": 1, "<|image_pad|>": IMAGE_PAD, "b": 2}[part] for part in text.split()]


class _FakeImageProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros(len(images), 4), "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images))}


def test_encode_multimodal_prompt_keeps_one_placeholder_per_image_for_qwen():
    spec = media_expansion_spec(
        SimpleNamespace(model_type="qwen3_vl", image_token_id=IMAGE_PAD, vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    processor = SimpleNamespace(image_processor=_FakeImageProcessor())
    prompt_ids, train_inputs, counts = encode_multimodal_prompt(
        processor, _FakeTokenizer(), spec, "a <|image_pad|> b <|image_pad|>", {"images": [object(), object()]}
    )
    assert prompt_ids == [1, IMAGE_PAD, 2, IMAGE_PAD]
    assert set(train_inputs) == {"pixel_values", "image_grid_thw"}
    assert counts == [4, 4]


def test_extract_multimodal_train_inputs_drops_qwen3_vl_token_metadata():
    pixel_values = object()
    image_grid_thw = object()
    processor_output = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "mm_token_type_ids": [[0, 1, 0]],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    assert extract_multimodal_train_inputs(processor_output) == {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    assert (
        extract_multimodal_train_inputs(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "mm_token_type_ids": [[0, 1, 0]],
            }
        )
        is None
    )
