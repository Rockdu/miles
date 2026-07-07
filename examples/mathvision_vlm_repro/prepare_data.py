"""Convert MathLLMs/MathVision into the parquet layout the geo3k VLM example uses.

Output schema (matches chenhegu/geo3k_imgurl, which miles' multimodal loader
already handles):
  problem : str  — prompt text containing one literal "<image>" placeholder
  answer  : str  — ground-truth answer for --rm-type math (boxed extraction)
  images  : list[str] — single-element list, base64 data URI of the problem image

MathVision has no train split; per the #ext-amazon-radixark repro setting the
`test` split (3040 problems) is the RL training pool and `testmini` (304) is
held out for eval.

Usage (inside the miles container):
  python examples/mathvision_vlm_repro/prepare_data.py --out-dir /root/datasets/mathvision
"""

import argparse
import base64
import io
import re

import pandas as pd
from datasets import load_dataset

IMAGE_TAG_RE = re.compile(r"<image\d+>")

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: \\boxed{{$Answer}} where $Answer "
    "is the answer to the problem.\n\n{question}"
)

OPTIONS_SUFFIX = (
    "\n\nChoices:\n{choices}\n"
    "The answer is one of the choice letters. Put only the letter in the box, "
    "e.g. Answer: \\boxed{{A}}."
)


def to_data_uri(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def convert_row(row) -> dict:
    question = row["question"].strip()

    # Normalize image placeholders: the loader pops one image per "<image>"
    # occurrence, and every MathVision problem ships exactly one image, so the
    # first <imageN> becomes <image> and any further references are dropped.
    question, n_subbed = IMAGE_TAG_RE.subn("<image>", question, count=1)
    question = IMAGE_TAG_RE.sub("", question)
    if n_subbed == 0:
        question = "<image>\n" + question

    options = row["options"]
    if options:
        letters = "ABCDEFGH"
        choices = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        question += OPTIONS_SUFFIX.format(choices=choices)

    return {
        "problem": PROMPT_TEMPLATE.format(question=question),
        "answer": str(row["answer"]).strip(),
        "images": [to_data_uri(row["decoded_image"])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/root/datasets/mathvision")
    parser.add_argument("--dataset", default="MathLLMs/MathVision")
    args = parser.parse_args()

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    for split, out_name in [("test", "train.parquet"), ("testmini", "eval.parquet")]:
        ds = load_dataset(args.dataset, split=split)
        rows = [convert_row(r) for r in ds]
        out_path = os.path.join(args.out_dir, out_name)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        n_mcq = sum(1 for r in rows if "Choices:" in r["problem"])
        print(f"{split} -> {out_path}: {len(rows)} rows ({n_mcq} multiple-choice)")


if __name__ == "__main__":
    main()
