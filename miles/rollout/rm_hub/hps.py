from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np
import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _sample_to_rgb_hwc_uint8(sample: Sample) -> np.ndarray:
    t = sample.generated_output
    if t is None:
        raise ValueError("generated_output is None")
    if t.ndim != 4:
        raise ValueError(f"generated_output must be 4D [C, F, H, W], got {tuple(t.shape)}")

    frame_chw = t.detach().cpu()[:, 0, :, :]
    hwc = frame_chw.float().numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        hwc = np.round(hwc * 255.0)
    return np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8))


class HPSScorer(torch.nn.Module):
    """HPS / HPSv2.1 reward scorer.

    Loads the ViT-H-14 backbone via hpsv2's own ``create_model_and_transforms``
    (so the preprocessing pipeline matches the published checkpoint exactly),
    then patches in the HPS preference-tuned weights. Scoring returns the raw
    image/text logit diagonal — the same scalar hpsv2.score() returns.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        hps_version: str = "v2.1",
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map

        self.device = torch.device(device)
        self.hps_version = hps_version

        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=str(self.device),
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download(
                "xswu/HPSv2", hps_version_map[hps_version]
            )
        checkpoint = torch.load(checkpoint_path, map_location=str(self.device))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device).eval()

        self.model = model
        self.preprocess = preprocess_val
        self.tokenizer = get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        if not prompts:
            return []

        image_batch = torch.stack([self.preprocess(img) for img in images]).to(
            self.device, non_blocking=True
        )
        text_batch = self.tokenizer(list(prompts)).to(self.device, non_blocking=True)

        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(image_batch, text_batch)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits = image_features @ text_features.T
            scores = torch.diagonal(logits)

        return [float(score) for score in scores.detach().float().cpu()]


@ray.remote
class HPSRewardActor:
    def __init__(
        self,
        *,
        hps_version: str,
        checkpoint_path: str | None = None,
    ) -> None:
        use_cuda = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        self.scorer = HPSScorer(
            device=device,
            hps_version=hps_version,
            checkpoint_path=checkpoint_path,
        )

    def score_batch(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        pil_images = [Image.fromarray(image) for image in images]
        return self.scorer(prompts, pil_images)


class AsyncHPSPool(metaclass=SingletonMeta):
    """Ray actor pool for GPU HPS reward inference."""

    def __init__(self, args) -> None:
        num_workers = args.hps_num_workers
        num_gpus_per_worker = args.hps_num_gpus_per_worker
        if num_workers <= 0:
            raise ValueError("--hps-num-workers must be positive")
        if args.hps_batch_size <= 0:
            raise ValueError("--hps-batch-size must be positive")

        self._batch_size = args.hps_batch_size
        self._actors = [
            HPSRewardActor.options(
                num_cpus=1,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy="DEFAULT",
            ).remote(
                hps_version=args.hps_version,
                checkpoint_path=args.hps_checkpoint_path,
            )
            for _ in range(num_workers)
        ]
        self._round_robin_index = 0
        logger.info(
            "Initialized HPS actor pool with %d workers, %.3f GPUs/worker, batch_size=%d, version=%s.",
            num_workers,
            num_gpus_per_worker,
            self._batch_size,
            args.hps_version,
        )

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return self._actors[i]

    async def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        if not images:
            return []

        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]


async def hps_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncHPSPool(args)
    images = [_sample_to_rgb_hwc_uint8(sample) for sample in samples]
    prompts = [sample.prompt for sample in samples]
    return await pool.score(images, prompts)
