"""Print the wandb internal-step ↔ rollout_id mapping for a wandb run's image
logs. Helps interpret the rollout_media panel slider whose x-axis is the wandb
internal commit step (NOT the rollout/step value, since wandb media ignores
step_metric).

Usage:
    python tools/wandb_media_step_map.py <run_dir> [--interval N]

<run_dir> = a `wandb/run-YYYYMMDD_HHMMSS-<id>/` directory.
--interval = --diffusion-log-image-interval used in the run (default 1).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(r"sample_images_(\d+)_[0-9a-f]+\.png$")


def collect_steps(run_dir: Path) -> list[int]:
    """Return sorted unique internal commit steps where image logs landed."""
    by_step = defaultdict(int)
    for p in run_dir.rglob("sample_images_*.png"):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        by_step[int(m.group(1))] += 1
    return sorted(by_step.keys()), by_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--interval", type=int, default=1,
                    help="--diffusion-log-image-interval used in the run")
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"no such dir: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    steps, counts = collect_steps(args.run_dir)
    if not steps:
        print(f"no sample_images*.png found under {args.run_dir}")
        sys.exit(0)

    n_per_log = sorted({c for c in counts.values()})
    deltas = [steps[i+1] - steps[i] for i in range(len(steps) - 1)]
    print(f"run_dir: {args.run_dir}")
    print(f"# image log calls: {len(steps)}")
    print(f"# images per log call: {n_per_log}")
    if deltas:
        unique_deltas = sorted(set(deltas))
        avg_delta = sum(deltas) / len(deltas)
        print(f"internal-step delta between consecutive logs: "
              f"min={min(deltas)} max={max(deltas)} avg={avg_delta:.1f} "
              f"unique={unique_deltas}")
        print(f"=> commits per rollout (K) ≈ {avg_delta / args.interval:.1f} "
              f"(interval={args.interval})")
    print()
    print(f"{'log#':>5}  {'media_step':>11}  {'#imgs':>5}  {'rollout_id':>10}")
    for i, s in enumerate(steps):
        rollout_id = i * args.interval
        print(f"{i:5d}  {s:11d}  {counts[s]:5d}  {rollout_id:10d}")


if __name__ == "__main__":
    main()
