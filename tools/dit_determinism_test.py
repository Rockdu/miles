"""Check whether diffusers Qwen-Image DiT gives bit-equal output when
run TWICE on identical inputs with frozen weights on the same GPU.

Hypothesis: if diffusers DiT alone is already non-deterministic at the
bf16 bit level (cuBLAS/cuDNN autotuner picks different tiles between
runs), then miles ↔ sglang can never be bit-exact no matter how we
patch ops — the ~2e-02 residual is an intrinsic CUDA non-determinism
floor, not a logic bug.

If diffusers is deterministic (same output byte-for-byte across runs),
then the residual IS attributable to a real miles-vs-sglang DiT
implementation difference worth drilling further.

Run: CUDA_VISIBLE_DEVICES=0 python tools/dit_determinism_test.py
"""
import os
import hashlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)

DUMP = "/tmp/dit_dump"
CALL = 1  # use dit_inputs_call1.pt


def sha(t: torch.Tensor) -> str:
    b = t.detach().contiguous().cpu().flatten()
    return hashlib.sha256(b.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    device = torch.device("cuda:0")

    # Load real rollout inputs captured in a prior miles debug run.
    in_path = os.path.join(DUMP, f"dit_inputs_call{CALL}.pt")
    inputs = torch.load(in_path, weights_only=False)
    print(f"Loaded {in_path}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} norm={v.float().norm().item():.4f}")
        else:
            print(f"  {k}: {type(v).__name__}={v}")

    dit = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    dit.requires_grad_(False)

    # Build DiT kwargs from dumped inputs.
    hs = inputs["hidden_states"].to(device)
    ehs = inputs["encoder_hidden_states"].to(device)
    mask = inputs["encoder_hidden_states_mask"]
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device)
    ts = inputs["timestep"].to(device)
    ts_bf16 = (ts / 1000.0).to(torch.bfloat16)  # match miles actor.py path

    def forward_once():
        with torch.no_grad():
            out = dit(
                hidden_states=hs,
                encoder_hidden_states=ehs,
                encoder_hidden_states_mask=mask,
                timestep=ts_bf16,
                img_shapes=inputs["img_shapes"],
                txt_seq_lens=inputs["txt_seq_lens"],
                return_dict=False,
            )[0]
        return out.detach().cpu()

    # Warm up (cuBLAS autotuner caches first config).
    _ = forward_once()
    torch.cuda.synchronize()

    print("\n=== Determinism test: two identical forwards ===")
    out_a = forward_once()
    torch.cuda.synchronize()
    out_b = forward_once()
    torch.cuda.synchronize()

    delta = (out_a.float() - out_b.float()).abs()
    same = torch.equal(out_a, out_b)
    print(
        f"run_a  sha={sha(out_a)}  norm={out_a.float().norm().item():.4f}\n"
        f"run_b  sha={sha(out_b)}  norm={out_b.float().norm().item():.4f}\n"
        f"torch.equal: {same}   mean|Δ|={delta.mean().item():.3e}   max|Δ|={delta.max().item():.3e}"
    )

    # Extra check: different batch size path (run with 2x batch via cat).
    # If autotuner remembers batch=1 result and gives batch=2 a different
    # config, the first-sample output can differ from the single-sample
    # output.
    print("\n=== Batch-shape effect: single vs stacked forwards ===")
    with torch.no_grad():
        out_solo = forward_once()
        hs2 = torch.cat([hs, hs], dim=0)
        ehs2 = torch.cat([ehs, ehs], dim=0)
        mask2 = torch.cat([mask, mask], dim=0) if isinstance(mask, torch.Tensor) else mask
        ts2 = torch.cat([ts_bf16, ts_bf16], dim=0)
        out_stacked = dit(
            hidden_states=hs2,
            encoder_hidden_states=ehs2,
            encoder_hidden_states_mask=mask2,
            timestep=ts2,
            img_shapes=inputs["img_shapes"] * 2,
            txt_seq_lens=inputs["txt_seq_lens"] * 2,
            return_dict=False,
        )[0][0].detach().cpu()  # first sample of batch-2 run
    torch.cuda.synchronize()

    delta = (out_solo[0].float() - out_stacked.float()).abs()
    same = torch.equal(out_solo[0], out_stacked)
    print(
        f"solo[0]    sha={sha(out_solo[0])}  norm={out_solo[0].float().norm().item():.4f}\n"
        f"stacked[0] sha={sha(out_stacked)}  norm={out_stacked.float().norm().item():.4f}\n"
        f"torch.equal: {same}   mean|Δ|={delta.mean().item():.3e}   max|Δ|={delta.max().item():.3e}"
    )


if __name__ == "__main__":
    main()
