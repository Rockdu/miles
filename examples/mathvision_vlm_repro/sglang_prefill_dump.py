"""Phase 3: sglang-side prefill scoring of saved rollout samples, with dumper.

Workflow (run each step manually; server stays up between calls):
  1. Export the ckpt to HF:
     python tools/convert_torch_dist_to_hf.py --input-dir <iter_dir> \
         --output-dir /scratch/replay/<iter>/hf --origin-hf-dir /root/models/Qwen3.5-9B -f
  2. Launch the server with dumper armed (TP1, one GPU):
     CUDA_VISIBLE_DEVICES=0 DUMPER_ENABLE=1 DUMPER_DIR=/scratch/replay/<iter>/dumps_sglang \
       DUMPER_EXP_NAME=prefill DUMPER_SOURCE_PATCHER_CONFIG=<sglang_gdn.yaml> \
       python -m sglang.launch_server --model-path /scratch/replay/<iter>/hf \
       --port 30000 --mem-fraction-static 0.85 --trust-remote-code
  3. Score (this script): single prefill forward per sample, no generation:
     python examples/mathvision_vlm_repro/sglang_prefill_dump.py \
         --rollout-pt /scratch/replay/<iter>/rollout_0.pt --port 30000
"""

import argparse

import requests
import torch

from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-pt", required=True)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--max-samples", type=int, default=1)
    args = ap.parse_args()

    data = torch.load(args.rollout_pt, weights_only=False)
    samples = [Sample.from_dict(d) for d in data["samples"]][: args.max_samples]

    for i, sample in enumerate(samples):
        prompt_len = len(sample.tokens) - sample.response_length
        payload = {
            "input_ids": sample.tokens,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": prompt_len - 1,
        }
        images = (sample.multimodal_inputs or {}).get("images")
        if images:
            payload["image_data"] = [encode_image_for_rollout_engine(im) for im in images]

        requests.post(f"http://127.0.0.1:{args.port}/flush_cache", timeout=60)
        r = requests.post(f"http://127.0.0.1:{args.port}/generate", json=payload, timeout=3600)
        r.raise_for_status()
        out = r.json()

        lp = [x[0] for x in out["meta_info"]["input_token_logprobs"][-sample.response_length :]]
        orig = sample.rollout_log_probs or []
        n = min(len(lp), len(orig))
        if n:
            mad = sum(abs(a - b) for a, b in zip(lp[:n], orig[:n])) / n
            print(f"sample {i}: prefill-vs-original-rollout logprob mean_abs_diff={mad:.5f} over {n} tokens")
        print(f"sample {i}: scored {len(lp)} response tokens (len={len(sample.tokens)}); dumps written by server")


if __name__ == "__main__":
    main()
