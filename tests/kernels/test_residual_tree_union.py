# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.residual_tree_union import (
    fused_hybrid_union_top10,
    supports_fused_hybrid_union,
)
from vllm.triton_utils import HAS_TRITON


def _reference_union(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    target_ids: torch.Tensor,
    h2_weight: float,
    clipped_mass_calibration: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_values, first_ids = torch.topk(first_logits, k=10, dim=-1)
    second_values, second_ids = torch.topk(second_logits, k=10, dim=-1)
    first_scores = torch.exp(
        first_values.float()
        - torch.logsumexp(first_logits.float(), dim=-1, keepdim=True)
    ).clamp_min(1e-12)
    second_scores = torch.exp(
        second_values.float()
        - torch.logsumexp(second_logits.float(), dim=-1, keepdim=True)
    ).clamp_min(1e-12)
    overlaps = second_ids.unsqueeze(2) == first_ids.unsqueeze(1)
    if clipped_mass_calibration is None:
        weighted_second_scores = second_scores * h2_weight
        matching_second = torch.where(
            overlaps,
            weighted_second_scores.unsqueeze(2),
            torch.zeros_like(weighted_second_scores).unsqueeze(2),
        ).amax(dim=1)
        merged_first = torch.maximum(first_scores, matching_second)
        unique_second = torch.where(
            overlaps.any(dim=2),
            torch.zeros_like(weighted_second_scores),
            weighted_second_scores,
        )
    else:
        ranks = torch.arange(10, device=first_logits.device).unsqueeze(0)
        first_categories = overlaps.any(dim=1).to(torch.long)
        second_categories = 2 + overlaps.any(dim=2).to(torch.long)
        calibrated_first = first_scores * clipped_mass_calibration[
            first_categories, ranks
        ]
        calibrated_second = second_scores * clipped_mass_calibration[
            second_categories, ranks
        ]
        matching_second = torch.where(
            overlaps,
            calibrated_second.unsqueeze(2),
            torch.zeros_like(calibrated_second).unsqueeze(2),
        ).amax(dim=1)
        merged_first = (calibrated_first + matching_second).clamp_max(1.0)
        unique_second = torch.where(
            overlaps.any(dim=2),
            torch.zeros_like(calibrated_second),
            calibrated_second,
        )
    union_scores = torch.cat(
        (merged_first, unique_second),
        dim=1,
    )
    union_ids = torch.cat((first_ids, second_ids), dim=1)
    selected_scores, selected_indices = torch.topk(union_scores, k=10, dim=-1)
    selected_ids = union_ids.gather(1, selected_indices)
    selected_tokens = target_ids.index_select(0, selected_ids.reshape(-1)).reshape_as(
        selected_ids
    )
    return selected_tokens, selected_scores


def test_fused_hybrid_union_rejects_cpu_tensors() -> None:
    logits = torch.zeros((1, 32), dtype=torch.bfloat16)
    target_ids = torch.arange(32)

    assert not supports_fused_hybrid_union(logits, logits, target_ids)


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires CUDA and Triton",
)
@pytest.mark.parametrize("batch_size", [1, 10])
def test_fused_hybrid_union_matches_reference(batch_size: int) -> None:
    vocab_size = 32_000
    first = torch.full(
        (batch_size, vocab_size),
        -12.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    second = torch.full_like(first, -13.0)
    # Unique top logits avoid relying on torch.topk's unspecified tie order.
    first_ids = torch.tensor(
        [17, 513, 1025, 4099, 8195, 12001, 16007, 20011, 25013, 30017, 31001],
        device="cuda",
    )
    second_ids = torch.tensor(
        [17, 514, 1026, 4100, 8196, 12002, 16008, 20012, 25014, 30018, 31002],
        device="cuda",
    )
    descending = torch.arange(11, 0, -1, device="cuda", dtype=torch.float32)
    descending = descending.to(torch.bfloat16)
    for row in range(batch_size):
        first[row, first_ids] = descending - row * 0.125
        second[row, second_ids] = descending + 0.25 - row * 0.125
    target_ids = torch.arange(vocab_size, device="cuda", dtype=torch.long).roll(7)

    (
        actual_tokens,
        actual_scores,
        head_ids,
        head_scores,
    ) = fused_hybrid_union_top10(
        first,
        second,
        target_ids,
        h2_weight=0.4306,
        return_head_candidates=True,
    )
    expected_tokens, expected_scores = _reference_union(
        first,
        second,
        target_ids,
        h2_weight=0.4306,
    )

    assert torch.equal(actual_tokens, expected_tokens)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=2e-6, atol=1e-8)
    assert all(len(set(row.tolist())) == 10 for row in actual_tokens)
    assert torch.equal(head_ids[:, 0].long(), torch.topk(first, 10).indices)
    assert torch.equal(head_ids[:, 1].long(), torch.topk(second, 10).indices)
    expected_head_scores = torch.stack(
        (
            torch.topk(torch.softmax(first.float(), dim=-1), 10).values,
            torch.topk(torch.softmax(second.float(), dim=-1), 10).values,
        ),
        dim=1,
    )
    torch.testing.assert_close(head_scores, expected_head_scores, rtol=2e-6, atol=1e-8)


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires CUDA and Triton",
)
def test_fused_hybrid_union_matches_clipped_mass_calibration() -> None:
    batch_size = 3
    vocab_size = 32_000
    first = torch.full(
        (batch_size, vocab_size),
        -12.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    second = torch.full_like(first, -13.0)
    first_ids = torch.tensor(
        [17, 513, 1025, 4099, 8195, 12001, 16007, 20011, 25013, 30017, 31001],
        device="cuda",
    )
    second_ids = torch.tensor(
        [17, 514, 1026, 4100, 8196, 12002, 16008, 20012, 25014, 30018, 31002],
        device="cuda",
    )
    descending = torch.arange(11, 0, -1, device="cuda", dtype=torch.float32)
    descending = descending.to(torch.bfloat16)
    for row in range(batch_size):
        first[row, first_ids] = descending - row * 0.125
        second[row, second_ids] = descending + 0.25 - row * 0.125
    target_ids = torch.arange(vocab_size, device="cuda", dtype=torch.long).roll(11)
    calibration = torch.linspace(
        0.05,
        0.8,
        40,
        device="cuda",
        dtype=torch.float32,
    ).reshape(4, 10)

    actual_tokens, actual_scores = fused_hybrid_union_top10(
        first,
        second,
        target_ids,
        h2_weight=1.0,
        clipped_mass_calibration=calibration,
    )
    expected_tokens, expected_scores = _reference_union(
        first,
        second,
        target_ids,
        h2_weight=1.0,
        clipped_mass_calibration=calibration,
    )

    assert torch.equal(actual_tokens, expected_tokens)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=2e-6, atol=1e-8)
