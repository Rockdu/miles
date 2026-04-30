"""Dump per-transformer-block first-row first-K activations on first forward
call, on both train (diffusers) and rollout (sgl-d) sides. For drift
localization at batch>=2.

Enable via env: MILES_BLOCK_DUMP_DIR=/path/to/dir SIDE=train|rollout

After first forward, each side writes <SIDE>_block_dump.pt with shape
(num_blocks, K) of float32 values + metadata.

Usage from train side: import + call register_diffusers_block_dump() before
forward.
Usage from rollout side: included in qwen_image_patch.apply_qwen_image_diffusers_parity_patches()
when env vars are set.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

K_PEEK = 64  # first K float32 values of block output (row 0)
EXPECTED_BLOCKS = 60  # Qwen-Image has 60 transformer blocks

_lock = threading.Lock()
_state = {
    "side": None,
    "out_dir": None,
    "current": [],
    "saved": False,
}


def _peek_first_row(out: torch.Tensor) -> torch.Tensor:
    flat = out.reshape(-1)[: K_PEEK]
    return flat.detach().to(torch.float32).cpu()


def _save_dump():
    if _state["saved"] or not _state["current"] or not _state["out_dir"]:
        return
    side = _state["side"]
    p = Path(_state["out_dir"]) / f"{side}_block_dump.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "side": side,
        "K": K_PEEK,
        "blocks": _state["current"],
        "inputs": _state.get("inputs", []),
        "encoder_inputs": _state.get("encoder_inputs", []),
        "temb_inputs": _state.get("temb_inputs", []),
        "raw_encoder": _state.get("raw_encoder"),
        "raw_hidden": _state.get("raw_hidden"),
    }, p)
    _state["saved"] = True
    print(f"[block_dump:{side}] saved {len(_state['current'])} blocks → {p}", flush=True)


def _maybe_init(side: str | None = None):
    if _state["side"] is not None:
        return
    out_dir = os.environ.get("MILES_BLOCK_DUMP_DIR")
    if not out_dir:
        return
    if side is None:
        side = os.environ.get("MILES_BLOCK_DUMP_SIDE")
    if not side:
        return
    _state["side"] = side
    _state["out_dir"] = out_dir


def install_block_hook(transformer_block_cls, side: str | None = None) -> bool:
    """Wrap transformer_block_cls.forward to peek output. Idempotent. Returns
    True if installed (MILES_BLOCK_DUMP_DIR set), False otherwise."""
    _maybe_init(side)
    if _state["side"] is None:
        return False
    if getattr(transformer_block_cls, "_miles_block_dump_installed", False):
        return True

    original_forward = transformer_block_cls.forward

    def _wrapped(self, *args, **kwargs):
        h_in = args[0] if args else kwargs.get("hidden_states")
        # Try to capture other inputs
        e_in = (args[1] if len(args) > 1 else kwargs.get("encoder_hidden_states"))
        # temb is named — look up by name
        t_in = kwargs.get("temb")
        if t_in is None and len(args) >= 4:
            # diffusers signature: hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, ...
            t_in = args[3] if isinstance(args[3], torch.Tensor) else None
        with _lock:
            if not _state["saved"]:
                _state.setdefault("inputs", [])
                _state.setdefault("encoder_inputs", [])
                _state.setdefault("temb_inputs", [])
                if len(_state["inputs"]) < EXPECTED_BLOCKS and isinstance(h_in, torch.Tensor):
                    _state["inputs"].append(_peek_first_row(h_in))
                if len(_state["encoder_inputs"]) < EXPECTED_BLOCKS and isinstance(e_in, torch.Tensor):
                    _state["encoder_inputs"].append(_peek_first_row(e_in))
                if len(_state["temb_inputs"]) < EXPECTED_BLOCKS and isinstance(t_in, torch.Tensor):
                    _state["temb_inputs"].append(_peek_first_row(t_in))
        out = original_forward(self, *args, **kwargs)
        with _lock:
            if not _state["saved"] and len(_state["current"]) < EXPECTED_BLOCKS:
                target = out
                if isinstance(out, tuple):
                    target = out[-1]
                if isinstance(target, torch.Tensor):
                    _state["current"].append(_peek_first_row(target))
                if len(_state["current"]) >= EXPECTED_BLOCKS:
                    _save_dump()
        return out

    transformer_block_cls.forward = _wrapped
    transformer_block_cls._miles_block_dump_installed = True
    print(f"[block_dump:{_state['side']}] installed hook on "
          f"{transformer_block_cls.__module__}.{transformer_block_cls.__name__}",
          flush=True)
    return True


def finalize_after_forward(expected_block_count: int | None = None) -> None:
    """Call once after the FIRST forward pass to flush the dump. Subsequent
    forwards don't re-dump."""
    with _lock:
        if _state["saved"]:
            return
        if expected_block_count is not None and len(_state["current"]) < expected_block_count:
            return
        _save_dump()


def install_top_model_hook(top_model_cls, side: str | None = None) -> bool:
    """Wrap top_model_cls.forward to capture initial encoder_hidden_states
    BEFORE any txt_norm / txt_in processing. Stores in _state["raw_encoder"]."""
    _maybe_init(side)
    if _state["side"] is None:
        return False
    if getattr(top_model_cls, "_miles_top_dump_installed", False):
        return True
    original = top_model_cls.forward

    def _wrapped(self, *args, **kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        e = kwargs.get("encoder_hidden_states")
        if e is None and len(args) > 1:
            e = args[1]
        with _lock:
            if not _state["saved"] and "raw_encoder" not in _state:
                if isinstance(e, torch.Tensor):
                    _state["raw_encoder"] = _peek_first_row(e)
                if isinstance(h, torch.Tensor):
                    _state["raw_hidden"] = _peek_first_row(h)
        return original(self, *args, **kwargs)

    top_model_cls.forward = _wrapped
    top_model_cls._miles_top_dump_installed = True
    return True


def register_diffusers_block_dump():
    """Train side: install hook on diffusers' QwenImageTransformerBlock + top model."""
    try:
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformerBlock,
            QwenImageTransformer2DModel,
        )
    except ImportError:
        return False
    install_top_model_hook(QwenImageTransformer2DModel, side="train")
    return install_block_hook(QwenImageTransformerBlock, side="train")


def register_sgld_block_dump():
    """Rollout side: install hook on sglang-diffusion's QwenImageTransformerBlock + top model."""
    try:
        from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
            QwenImageTransformerBlock,
            QwenImageTransformer2DModel,
        )
    except ImportError:
        return False
    install_top_model_hook(QwenImageTransformer2DModel, side="rollout")
    return install_block_hook(QwenImageTransformerBlock, side="rollout")
