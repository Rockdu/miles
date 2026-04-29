"""Parse one or more align-check run logs and produce a side-by-side table
of train/align/* metrics per (rollout_id, train_step).

Usage:
    python tools/parse_align_log.py logs/baseline_patch_on/run.log
    python tools/parse_align_log.py logs/*.log   # multiple runs
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"\[train step (?P<step>\d+)\] rollout=(?P<rollout>\d+)\s+(?P<rest>.+)$"
)
KV_RE = re.compile(r"(\S+?)=([\-+]?[0-9.]+(?:e[+\-]?[0-9]+)?)")


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        # logs are colour-prefixed; strip ANSI codes before matching
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        # also strip the (FSDPTrainRayActor pid=...) prefix
        clean = re.sub(r"\([A-Za-z]+RayActor\s*pid=\d+\)\s*", "", clean)
        m = LINE_RE.search(clean)
        if not m:
            continue
        kvs = dict(KV_RE.findall(m.group("rest")))
        kvs = {k: float(v) for k, v in kvs.items()}
        rows.append({
            "step": int(m.group("step")),
            "rollout": int(m.group("rollout")),
            **kvs,
        })
    return rows


KEYS = [
    "train/align/noise_pred_abs_mean_diff",
    "train/align/noise_pred_abs_max_diff",
    "train/align/noise_pred_rel_l2_diff",
    "train/log_prob_mean_abs_diff",
    "train/log_prob_max_abs_diff",
    "train/ratio_abs_minus_1",
    "train/clipfrac",
    "train/grad_norm",
]


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.3e}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    runs: dict[str, list[dict]] = {}
    for arg in sys.argv[1:]:
        for p in Path().glob(arg) if any(c in arg for c in "*?[") else [Path(arg)]:
            if p.is_dir():
                # auto-discover run.log inside
                for sub in p.rglob("run.log"):
                    runs[str(sub)] = parse_log(sub)
            else:
                runs[str(p)] = parse_log(p)

    if not runs:
        print("No matching logs.", file=sys.stderr)
        sys.exit(1)

    for name, rows in runs.items():
        print("=" * 80)
        print(f"{name}  ({len(rows)} train steps)")
        print("=" * 80)
        if not rows:
            print("  (no train/align metrics found)")
            continue
        # Header
        cols = ["step", "rollout"] + KEYS
        widths = [max(len(c), 11) for c in cols]
        widths[0] = 4
        widths[1] = 7
        header = "  ".join(c[-w:].rjust(w) for c, w in zip(cols, widths))
        print(header)
        for r in rows:
            line = "  ".join(
                str(r.get(c, "—")).rjust(w) if c in ("step", "rollout") else fmt(r.get(c)).rjust(w)
                for c, w in zip(cols, widths)
            )
            print(line)


if __name__ == "__main__":
    main()
