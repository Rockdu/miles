from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from miles.utils.processing_utils import extract_multimodal_train_inputs


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


def _fake_qwen_processor(model_type):
    from types import SimpleNamespace

    image_processor = SimpleNamespace(patch_size=16, size={"shortest_edge": 65536, "longest_edge": 16777216})
    return SimpleNamespace(image_processor=image_processor, _miles_model_type=model_type)


def _captured_prompt(monkeypatch, processor, prompt):
    import qwen_vl_utils

    from miles.utils.processing_utils import process_vision_info

    captured = {}

    def fake_process_vision_info(conversations, image_patch_size):
        captured["prompt"], captured["image_patch_size"] = conversations, image_patch_size
        return [], None

    monkeypatch.setattr(qwen_vl_utils, "process_vision_info", fake_process_vision_info)
    process_vision_info(prompt, processor)
    return captured


_PROMPT = [{"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "hi"}]}]


def test_process_vision_info_resizes_qwen3_5_images_into_processor_window(monkeypatch):
    for model_type in ("qwen3_5", "qwen3_5_moe"):
        captured = _captured_prompt(monkeypatch, _fake_qwen_processor(model_type), _PROMPT)
        assert captured["image_patch_size"] == 16
        assert captured["prompt"][0]["content"][0] == {
            "type": "image",
            "image": "a.png",
            "min_pixels": 65536,
            "max_pixels": 16777216,
        }


def test_process_vision_info_leaves_other_models_untouched(monkeypatch):
    # Qwen3-VL shares Qwen3.5's processor config but is out of scope; unknown model types keep old behavior too.
    for model_type in ("qwen3_vl", "qwen2_5_vl", None):
        captured = _captured_prompt(monkeypatch, _fake_qwen_processor(model_type), _PROMPT)
        assert captured["prompt"] == _PROMPT


def test_with_image_pixel_bounds_fills_images_only_and_keeps_explicit_values():
    from miles.utils.processing_utils import _with_image_pixel_bounds

    bounds = {"min_pixels": 65536, "max_pixels": 16777216}
    prompt = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "a.png"},
                {"type": "image", "image": "b.png", "min_pixels": 1024},
                {"type": "text", "text": "hi"},
            ],
        },
    ]

    out = _with_image_pixel_bounds(prompt, bounds)

    assert out[0] == prompt[0]
    assert out[1]["content"][0] == {"type": "image", "image": "a.png", **bounds}
    assert out[1]["content"][1]["min_pixels"] == 1024
    assert out[1]["content"][1]["max_pixels"] == 16777216
    assert out[1]["content"][2] == {"type": "text", "text": "hi"}
    # The caller's prompt is not mutated.
    assert "min_pixels" not in prompt[1]["content"][0]
