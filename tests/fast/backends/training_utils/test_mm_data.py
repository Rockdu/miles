"""Train-side media expansion splices each placeholder into the count the controller shipped.

    tokens      [5, P, 7 | P, 9]      P = placeholder, | = prompt/response boundary
    counts      [3,     2]
    loss_mask         [1, 1]
                        ▼
    tokens      [5, P, P, P, 7 | P, P, 9]
    loss_mask                 [0, 0, 1]     expanded response media carry no loss

    The same batch is expanded once per data-iterator build, so a second call is a no-op.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import mm_data

P = 151655


@pytest.fixture(autouse=True)
def _single_cp_rank(monkeypatch):
    # a length change re-slices per-token side channels only under CP>1; pin CP=1 here
    monkeypatch.setattr(mm_data, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))


def _rollout_data():
    return {
        "tokens": [torch.tensor([5, P, 7, P, 9]), torch.tensor([1, 2, 3])],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1])],
        "total_lengths": [5, 3],
        "response_lengths": [2, 1],
        "media_token_counts": [[3, 2], []],
        "media_placeholder_token_ids": [P],
    }


def test_placeholders_expand_in_prompt_order_and_response_media_get_no_loss():
    rollout_data = _rollout_data()
    mm_data.expand_multimodal_rollout_data_in_place(rollout_data)

    assert rollout_data["tokens"][0].tolist() == [5, P, P, P, 7, P, P, 9]
    assert rollout_data["loss_masks"][0].tolist() == [0, 0, 1]
    assert rollout_data["total_lengths"] == [8, 3]
    assert rollout_data["response_lengths"] == [3, 1]
    assert rollout_data["tokens"][1].tolist() == [1, 2, 3], "text-only samples are untouched"


def test_second_expansion_is_a_no_op():
    rollout_data = _rollout_data()
    mm_data.expand_multimodal_rollout_data_in_place(rollout_data)
    mm_data.expand_multimodal_rollout_data_in_place(rollout_data)
    assert rollout_data["tokens"][0].tolist() == [5, P, P, P, 7, P, P, 9]
    assert rollout_data["total_lengths"] == [8, 3]


def test_placeholder_and_count_mismatch_is_rejected():
    rollout_data = _rollout_data()
    rollout_data["media_token_counts"] = [[3], []]
    with pytest.raises(AssertionError, match="2 media placeholder"):
        mm_data.expand_multimodal_rollout_data_in_place(rollout_data)
