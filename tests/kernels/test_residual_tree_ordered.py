# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.residual_tree_ordered import (
    fused_ordered_distinct_top1,
    fused_ordered_head_top1,
    supports_fused_ordered_selection,
)


def _reference(
    first_logits: torch.Tensor,
    later_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    conditioned = torch.cat((first_logits.unsqueeze(1), later_logits), dim=1).float()
    selected: list[torch.Tensor] = []
    for head_index in range(conditioned.shape[1]):
        token = conditioned[:, head_index].argmax(dim=1)
        selected.append(token)
        if head_index + 1 < conditioned.shape[1]:
            conditioned[:, head_index + 1 :].scatter_(
                2,
                token[:, None, None].expand(
                    -1, conditioned.shape[1] - head_index - 1, 1
                ),
                float("-inf"),
            )
    tokens = torch.stack(selected, dim=1)
    probabilities = torch.softmax(conditioned, dim=2).gather(
        2, tokens.unsqueeze(2)
    ).squeeze(2)
    return tokens, probabilities


def test_fused_ordered_selection_rejects_cpu() -> None:
    first = torch.zeros((1, 32), dtype=torch.bfloat16)
    later = torch.zeros((1, 9, 32), dtype=torch.bfloat16)
    assert not supports_fused_ordered_selection(first, later)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("vocab_size", [31744, 32000])
def test_fused_ordered_selection_matches_reference(
    batch_size: int,
    vocab_size: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260814 + batch_size + vocab_size
    )
    first = torch.randn(
        (batch_size, vocab_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    # This is the non-contiguous batch/head layout produced by the packed
    # low-rank batched matrix multiplication in the serving path.
    later = torch.randn(
        (9, batch_size, vocab_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).transpose(0, 1)
    assert later.stride(2) == 1
    if batch_size > 1:
        assert not later.is_contiguous()

    expected_tokens, expected_probabilities = _reference(first, later)
    actual_tokens, actual_probabilities = fused_ordered_distinct_top1(first, later)

    assert torch.equal(actual_tokens, expected_tokens)
    torch.testing.assert_close(
        actual_probabilities,
        expected_probabilities,
        rtol=2e-6,
        atol=2e-9,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_ordered_selection_uses_leftmost_tie_and_prior_masks() -> None:
    vocab_size = 32000
    first = torch.zeros((1, vocab_size), dtype=torch.bfloat16, device="cuda")
    later = torch.zeros((1, 9, vocab_size), dtype=torch.bfloat16, device="cuda")
    first[0, 7] = 4
    first[0, 11] = 4
    later[:, :, 7] = 5
    later[:, :, 11] = 5
    later[:, :, 13] = 5
    later[:, :, 17] = 5
    later[:, :, 19] = 5
    later[:, :, 23] = 5
    later[:, :, 29] = 5
    later[:, :, 31] = 5
    later[:, :, 37] = 5
    later[:, :, 41] = 5

    expected_tokens, expected_probabilities = _reference(first, later)
    actual_tokens, actual_probabilities = fused_ordered_distinct_top1(first, later)

    assert torch.equal(actual_tokens, expected_tokens)
    torch.testing.assert_close(
        actual_probabilities,
        expected_probabilities,
        rtol=2e-6,
        atol=2e-9,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_independent_head_top1_preserves_repeated_winners() -> None:
    vocab_size = 32000
    first = torch.zeros((1, vocab_size), dtype=torch.bfloat16, device="cuda")
    later = torch.zeros((1, 9, vocab_size), dtype=torch.bfloat16, device="cuda")
    first[0, 17] = 8
    later[:, :, 17] = 8

    tokens, probabilities = fused_ordered_head_top1(first, later)
    all_logits = torch.cat((first.unsqueeze(1), later), dim=1).float()
    expected_tokens = all_logits.argmax(dim=2)
    expected_probabilities = torch.softmax(all_logits, dim=2).gather(
        2, expected_tokens.unsqueeze(2)
    ).squeeze(2)

    assert tokens.tolist() == [[17] * 10]
    assert torch.equal(tokens, expected_tokens)
    torch.testing.assert_close(
        probabilities,
        expected_probabilities,
        rtol=2e-6,
        atol=2e-9,
    )
