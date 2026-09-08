"""Multimodal-specific preprocessing for rollout data.

The rollout side emits one media placeholder/sentinel per media item; training
expands it to the per-item token count so the LM sees a position per vision
patch / audio frame. Two families live here: in-vocab placeholders expanded in
place from the counts the controller shipped (`media_token_counts`), and Inkling
(out-of-vocab sentinels expanded to in-vocab placeholder runs with explicit
positions). Kept separate from data.py (generic batching / CP slicing).
"""

import logging

import torch

from miles.utils.media_expansion import INKLING_AUDIO_SENTINEL_ID, INKLING_IMAGE_SENTINEL_ID
from miles.utils.types import RolloutBatch

from .cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)


def _expand_media_placeholders(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    placeholder_token_ids: tuple[int, ...],
    counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the i-th placeholder into counts[i] copies of itself; expanded response media carry no loss."""
    # TODO: expansion shifts token indices, so Sample.weight_versions span positions are not remapped here.
    is_placeholder = torch.isin(tokens, torch.tensor(placeholder_token_ids, device=tokens.device))
    num_placeholders = int(is_placeholder.sum())
    # A rollout batch is expanded once per data-iterator build, so a second pass sees the expanded ids.
    if num_placeholders == sum(counts):
        return tokens, loss_mask
    assert num_placeholders == len(counts), (
        f"{num_placeholders} media placeholder(s) in the tokens but {len(counts)} media token count(s) were shipped"
    )

    repeats = torch.ones(len(tokens), dtype=torch.long, device=tokens.device)
    repeats[is_placeholder] = torch.tensor(counts, dtype=torch.long, device=tokens.device)
    prompt_length = len(tokens) - len(loss_mask)
    expanded_loss_mask = loss_mask.repeat_interleave(repeats[prompt_length:])
    expanded_loss_mask[is_placeholder[prompt_length:].repeat_interleave(repeats[prompt_length:])] = 0
    return tokens.repeat_interleave(repeats), expanded_loss_mask


INKLING_MM_PLACEHOLDER_TOKEN_ID = 200023
INKLING_MM_AUDIO_PLACEHOLDER_TOKEN_ID = 200025


def _expand_inkling_sample(token_tensor, loss_mask, mm, sample_idx: int):
    """Replace media sentinels with placeholder runs, recording their sample-local positions into mm; idempotent."""
    spec = (
        (INKLING_IMAGE_SENTINEL_ID, INKLING_MM_PLACEHOLDER_TOKEN_ID, "mm_vision_num_patches", "mm_vision_positions"),
        (
            INKLING_AUDIO_SENTINEL_ID,
            INKLING_MM_AUDIO_PLACEHOLDER_TOKEN_ID,
            "mm_audio_num_tokens",
            "mm_audio_positions",
        ),
    )
    splices = []
    for sentinel, placeholder, counts_key, positions_key in spec:
        counts = mm.get(counts_key) if mm else None
        positions = (token_tensor == sentinel).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        assert counts is not None and positions.numel() == len(counts), (
            f"sample {sample_idx}: {positions.numel()} sentinel(s) {sentinel} but "
            f"{counts_key}={'missing' if counts is None else len(counts)}"
        )
        splices.append((positions.tolist(), [int(c) for c in counts], placeholder, positions_key))

    if not splices:
        return token_tensor

    prompt_len = len(token_tensor) - len(loss_mask)
    flat = sorted(
        (pos, n, placeholder, positions_key)
        for positions, counts, placeholder, positions_key in splices
        for pos, n in zip(positions, counts, strict=True)
    )
    assert all(
        p < prompt_len for p, _, _, _ in flat
    ), "Inkling media sentinels must be in the prompt; found one in the response"

    pieces, prev, shift = [], 0, 0
    out_positions: dict[str, list[int]] = {}
    for pos, n, placeholder, positions_key in flat:
        pieces.append(token_tensor[prev:pos])
        pieces.append(torch.full((n,), placeholder, dtype=token_tensor.dtype, device=token_tensor.device))
        start = pos + shift
        out_positions.setdefault(positions_key, []).extend(range(start, start + n))
        shift += n - 1
        prev = pos + 1
    pieces.append(token_tensor[prev:])
    for positions_key, plist in out_positions.items():
        mm[positions_key] = torch.tensor(plist, dtype=torch.long, device=token_tensor.device)
    return torch.cat(pieces)


def _expand_inkling_rollout_data_in_place(rollout_data: RolloutBatch) -> None:
    mm_list = rollout_data["multimodal_train_inputs"]
    tokens = rollout_data["tokens"]
    loss_masks = rollout_data["loss_masks"]
    old_total_lengths = list(rollout_data["total_lengths"])

    new_tokens_list, new_total_lengths = [], []
    for i, (token_tensor, loss_mask) in enumerate(zip(tokens, loss_masks, strict=False)):
        mm = mm_list[i] if i < len(mm_list) else None
        expanded = _expand_inkling_sample(token_tensor, loss_mask, mm, i)
        new_tokens_list.append(expanded)
        new_total_lengths.append(expanded.size(0))

    if new_total_lengths != old_total_lengths:
        try:
            cp_size = get_parallel_state().cp.size
        except Exception:
            cp_size = 1
        assert cp_size == 1, "Inkling multimodal expansion does not support CP>1 yet"
        rollout_data["tokens"] = new_tokens_list
        rollout_data["total_lengths"] = new_total_lengths
        logger.info(
            "Expanded Inkling image sentinels: total_lengths %s -> %s",
            old_total_lengths,
            new_total_lengths,
        )


def expand_multimodal_rollout_data_in_place(rollout_data: RolloutBatch, qkv_format: str = "thd") -> None:
    multimodal_train_inputs = rollout_data.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None and any(
        mm is not None and ("mm_vision_num_patches" in mm or "mm_audio_num_tokens" in mm)
        for mm in multimodal_train_inputs
    ):
        _expand_inkling_rollout_data_in_place(rollout_data)
        return
    counts_per_sample = rollout_data.get("media_token_counts")
    if counts_per_sample is None or not any(counts_per_sample):
        return
    placeholder_token_ids = tuple(rollout_data["media_placeholder_token_ids"])

    tokens = rollout_data["tokens"]
    loss_masks = rollout_data["loss_masks"]
    old_total_lengths = list(rollout_data["total_lengths"])
    old_response_lengths = list(rollout_data["response_lengths"])

    expanded_tokens = []
    expanded_loss_masks = []
    for token_tensor, loss_mask_tensor, counts in zip(tokens, loss_masks, counts_per_sample, strict=True):
        if counts:
            token_tensor, loss_mask_tensor = _expand_media_placeholders(
                token_tensor, loss_mask_tensor, placeholder_token_ids, counts
            )
        expanded_tokens.append(token_tensor)
        expanded_loss_masks.append(loss_mask_tensor)
    expanded_total_lengths = [t.size(0) for t in expanded_tokens]
    expanded_response_lengths = [m.size(0) for m in expanded_loss_masks]

    rollout_data["tokens"] = expanded_tokens
    rollout_data["loss_masks"] = expanded_loss_masks
    rollout_data["total_lengths"] = expanded_total_lengths
    rollout_data["response_lengths"] = expanded_response_lengths

    if expanded_total_lengths == old_total_lengths and expanded_response_lengths == old_response_lengths:
        return
    # The per-token side channels were sliced for the unexpanded lengths; re-slice them for the new ones.
    parallel_state = get_parallel_state()
    if parallel_state.cp.size > 1 and qkv_format == "thd":
        for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl"):
            values = rollout_data.get(key)
            if not values:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(
                    all_gather_with_cp(value, old_total_length, old_response_length),
                    new_total_length,
                    new_response_length,
                    qkv_format,
                )
                for value, old_total_length, old_response_length, new_total_length, new_response_length in zip(
                    values,
                    old_total_lengths,
                    old_response_lengths,
                    expanded_total_lengths,
                    expanded_response_lengths,
                    strict=False,
                )
            ]
    logger.info(
        "Expanded media placeholders: total_lengths %s -> %s, response_lengths %s -> %s",
        old_total_lengths,
        expanded_total_lengths,
        old_response_lengths,
        expanded_response_lengths,
    )
