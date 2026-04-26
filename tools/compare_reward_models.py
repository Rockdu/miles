"""Score the same (prompt, image) pairs through PickScore and HPS, side by side.

Usage:
    python tools/compare_reward_models.py \\
        --fixtures data/reward_fixtures/sample.jsonl \\
        --hps-version v2.1 \\
        --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \\
        --pickscore-model-path yuvalkirstain/PickScore_v1

Fixture format: one JSON object per line with keys ``prompt`` (str) and
``image`` (path, relative to repo root or absolute).

Prints a CSV to stdout with columns ``prompt,image,pickscore,hps`` and a
correlation summary (Pearson + Spearman) to stderr.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_fixtures(path: Path) -> list[dict]:
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def resolve_image(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def correlations(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (Pearson, Spearman). Avoids scipy/numpy hard dep."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sx2 = sum((x - mean_x) ** 2 for x in xs)
    sy2 = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    pearson = sxy / ((sx2 * sy2) ** 0.5) if sx2 > 0 and sy2 > 0 else float("nan")

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    srx2 = sum((r - mean_rx) ** 2 for r in rx)
    sry2 = sum((r - mean_ry) ** 2 for r in ry)
    srxy = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    spearman = srxy / ((srx2 * sry2) ** 0.5) if srx2 > 0 and sry2 > 0 else float("nan")
    return pearson, spearman


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hps-version", default="v2.1", choices=["v2.0", "v2.1"])
    parser.add_argument(
        "--pickscore-processor-path", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    parser.add_argument("--pickscore-model-path", default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--skip-pickscore", action="store_true")
    parser.add_argument("--skip-hps", action="store_true")
    args = parser.parse_args()

    entries = load_fixtures(args.fixtures)
    prompts = [e["prompt"] for e in entries]
    image_paths = [resolve_image(e["image"]) for e in entries]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    pick_scores: list[float] = []
    hps_scores: list[float] = []

    if not args.skip_pickscore:
        from miles.rollout.rm_hub.pickscore import PickScoreScorer

        scorer = PickScoreScorer(
            device=args.device,
            processor_path=args.pickscore_processor_path,
            model_path=args.pickscore_model_path,
        )
        pick_scores = scorer(prompts, images)
        del scorer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_hps:
        from miles.rollout.rm_hub.hps import HPSScorer

        scorer = HPSScorer(device=args.device, hps_version=args.hps_version)
        hps_scores = scorer(prompts, images)
        del scorer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    writer = csv.writer(sys.stdout)
    writer.writerow(["prompt", "image", "pickscore", "hps"])
    for i, entry in enumerate(entries):
        writer.writerow(
            [
                entry["prompt"],
                entry["image"],
                f"{pick_scores[i]:.6f}" if pick_scores else "",
                f"{hps_scores[i]:.6f}" if hps_scores else "",
            ]
        )

    if pick_scores and hps_scores and len(entries) >= 2:
        pearson, spearman = correlations(pick_scores, hps_scores)
        print(
            f"\nN={len(entries)}  pearson(pickscore, hps)={pearson:+.4f}  "
            f"spearman={spearman:+.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
