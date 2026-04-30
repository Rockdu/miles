"""Compare sgld-saved freqs_cis (in dump) vs diffusers' pos_embed.forward
output, on the same img_shapes / text_seq_len. After rebuilding pos_freqs
on CUDA, are they bit-exact? If not, that's the source of the 1.4e-2 drift
at step 0.
"""
from __future__ import annotations

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from diffusers import QwenImageTransformer2DModel as DiffQI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from miles.backends.fsdp_utils.configs.qwen_image import _rebuild_pos_embed_freqs_on_cuda

DEV = torch.device("cuda:0")
DTYPE = torch.bfloat16
DUMP_PATH = "/root/diffusion-rl/miles/.claude/worktrees/layer-alignment-diffusers-vs-sgld/logs/align_mg1_sdewin_20260430_005005/rollout_dump_0.pt"


def main():
    print("loading dump…")
    d = torch.load(DUMP_PATH, weights_only=False, map_location="cpu")
    sample = d["samples"][0]
    pos_kw = sample["denoising_env"].pos_cond_kwargs
    img_shapes = pos_kw.img_shapes
    txt_seq_lens = pos_kw.txt_seq_lens
    sgld_freqs = pos_kw.freqs_cis
    print(f"  img_shapes={img_shapes}  txt_seq_lens={txt_seq_lens}")
    print(f"  sgld freqs_cis type: {type(sgld_freqs)}")
    if isinstance(sgld_freqs, (tuple, list)):
        for i, f in enumerate(sgld_freqs):
            if hasattr(f, "shape"):
                print(f"    [{i}] shape={tuple(f.shape)} dtype={f.dtype} device={f.device}")
            else:
                print(f"    [{i}] type={type(f)}")
    elif hasattr(sgld_freqs, "shape"):
        print(f"    shape={tuple(sgld_freqs.shape)} dtype={sgld_freqs.dtype} device={sgld_freqs.device}")

    print()
    print("loading diffusers DiT (just for pos_embed)…")
    m = DiffQI.from_pretrained(
        "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=DTYPE
    ).to(DEV).eval()

    print()
    print("=== freqs WITHOUT rebuild ===")
    diff_no_rebuild = m.pos_embed(img_shapes, max_txt_seq_len=txt_seq_lens[0], device=DEV)
    print(f"  diffusers pos_embed return type: {type(diff_no_rebuild)}")
    if isinstance(diff_no_rebuild, tuple):
        for i, f in enumerate(diff_no_rebuild):
            print(f"    [{i}] shape={tuple(f.shape)} dtype={f.dtype}")

    print()
    print("rebuilding pos_freqs on CUDA…")
    _rebuild_pos_embed_freqs_on_cuda(m)
    diff_rebuild = m.pos_embed(img_shapes, max_txt_seq_len=txt_seq_lens[0], device=DEV)

    print()
    def to_complex(s):
        half = s.shape[-1] // 2
        return torch.complex(s[..., :half], s[..., half:])

    print("=== diff sgld vs diffusers (no rebuild) ===")
    for i, (s, dnr) in enumerate(zip(sgld_freqs, diff_no_rebuild)):
        s = s.to(DEV) if hasattr(s, "to") else s
        s_c = to_complex(s.float())
        d = (s_c - dnr).abs()
        print(f"    [{i}] abs_max={d.max().item():.3e}  abs_mean={d.mean().item():.3e}  shapes sgld={tuple(s_c.shape)} diff={tuple(dnr.shape)}")

    print()
    print("=== diff sgld vs diffusers (with rebuild) ===")
    for i, (s, dr) in enumerate(zip(sgld_freqs, diff_rebuild)):
        s = s.to(DEV) if hasattr(s, "to") else s
        s_c = to_complex(s.float())
        d = (s_c - dr).abs()
        print(f"    [{i}] abs_max={d.max().item():.3e}  abs_mean={d.mean().item():.3e}")

    print()
    print("=== diff diffusers no_rebuild vs rebuild (sanity) ===")
    for i, (a, b) in enumerate(zip(diff_no_rebuild, diff_rebuild)):
        if not (hasattr(a, "shape") and hasattr(b, "shape")):
            continue
        d = (a.float() - b.float()).abs()
        print(f"    [{i}] abs_max={d.max().item():.3e}  abs_mean={d.mean().item():.3e}")


if __name__ == "__main__":
    main()
