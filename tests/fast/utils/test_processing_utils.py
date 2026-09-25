# What this file tests
#
# extract_multimodal_train_inputs:
#   processor output ──► drop input_ids / attention_mask / mm_token_type_ids ──► keep pixel_values, image_grid_thw
#
# process_vision_info resizes each image exactly once, into the image processor's pixel window:
#
#   238x160 image (35840 px, below Qwen3-VL's 65536 px minimum)
#       │ qwen_vl_utils, min_pixels / max_pixels taken from image_processor.size
#       ▼
#   320x224 image ──► image processor ──► 320x224, grid 1x14x20   (no second resize)
#
#   An image item that sets its own min_pixels keeps it:  238x160 ──(min_pixels=4096)──► 224x160

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from miles.utils.processing_utils import extract_multimodal_train_inputs, process_vision_info


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


def _qwen3_vl_processor():
    image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 16777216}
    )
    return SimpleNamespace(image_processor=image_processor)


def test_process_vision_info_resizes_small_image_once_into_image_processor_pixel_window():
    processor = _qwen3_vl_processor()
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "image", "image": Image.new("RGB", (238, 160))}, {"type": "text", "text": "hi"}],
        },
    ]

    image = process_vision_info(prompt, processor)["images"][0]
    image_grid_thw = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"]

    # qwen_vl_utils already upsampled to the size the image processor picks, so the grid matches the image as is.
    assert image.size == (320, 224)
    assert image_grid_thw.tolist() == [[1, 224 // 16, 320 // 16]]


def test_process_vision_info_keeps_image_items_own_pixel_window():
    image_item = {"type": "image", "image": Image.new("RGB", (238, 160)), "min_pixels": 4096}

    image = process_vision_info([{"role": "user", "content": [image_item]}], _qwen3_vl_processor())["images"][0]

    # min_pixels=4096 only rounds to multiples of 32; the caller's image item is left unmodified.
    assert image.size == (224, 160)
    assert "max_pixels" not in image_item
