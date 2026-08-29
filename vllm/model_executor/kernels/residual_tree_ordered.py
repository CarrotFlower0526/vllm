# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused top-one selection for ordered residual-tree heads."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton


if HAS_TRITON:

    @triton.jit
    def _ordered_argmax_probability_kernel(
        first_logits_ptr,
        later_logits_ptr,
        selected_tokens_ptr,
        selected_probabilities_ptr,
        later_stride_batch,
        later_stride_head,
        later_stride_vocab,
        vocab_size: tl.constexpr,
        head_count: tl.constexpr,
        head_index: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        if head_index == 0:
            logits = tl.load(
                first_logits_ptr + row * vocab_size + offsets,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
        else:
            logits = tl.load(
                later_logits_ptr
                + row * later_stride_batch
                + (head_index - 1) * later_stride_head
                + offsets * later_stride_vocab,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
        for prior_head in tl.static_range(0, head_index):
            blocked_token = tl.load(
                selected_tokens_ptr + row * head_count + prior_head
            )
            logits = tl.where(offsets == blocked_token, -float("inf"), logits)

        maximum = tl.max(logits, axis=0)
        token = tl.argmax(logits, axis=0, tie_break_left=True)
        denominator = tl.sum(tl.exp(logits - maximum), axis=0)
        tl.store(selected_tokens_ptr + row * head_count + head_index, token)
        tl.store(
            selected_probabilities_ptr + row * head_count + head_index,
            1.0 / denominator,
        )

    @triton.jit
    def _independent_argmax_probability_kernel(
        first_logits_ptr,
        later_logits_ptr,
        selected_tokens_ptr,
        selected_probabilities_ptr,
        later_stride_batch,
        later_stride_head,
        later_stride_vocab,
        vocab_size: tl.constexpr,
        head_count: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_head = tl.program_id(0)
        row = row_head // head_count
        head_index = row_head % head_count
        offsets = tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        if head_index == 0:
            logits = tl.load(
                first_logits_ptr + row * vocab_size + offsets,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
        else:
            logits = tl.load(
                later_logits_ptr
                + row * later_stride_batch
                + (head_index - 1) * later_stride_head
                + offsets * later_stride_vocab,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)

        maximum = tl.max(logits, axis=0)
        token = tl.argmax(logits, axis=0, tie_break_left=True)
        denominator = tl.sum(tl.exp(logits - maximum), axis=0)
        tl.store(selected_tokens_ptr + row_head, token)
        tl.store(selected_probabilities_ptr + row_head, 1.0 / denominator)


def supports_fused_ordered_selection(
    first_logits: torch.Tensor,
    later_logits: torch.Tensor,
) -> bool:
    """Return whether the exact ordered-selector tensor contract is supported."""

    if not HAS_TRITON or not first_logits.is_cuda or not later_logits.is_cuda:
        return False
    if first_logits.ndim != 2 or later_logits.ndim != 3:
        return False
    batch_size, vocab_size = first_logits.shape
    if (
        batch_size <= 0
        or vocab_size <= 0
        or vocab_size > 65536
        or later_logits.shape[0] != batch_size
        or later_logits.shape[2] != vocab_size
        or not 1 <= later_logits.shape[1] + 1 <= 10
    ):
        return False
    if (
        first_logits.device != later_logits.device
        or first_logits.dtype != later_logits.dtype
        or first_logits.dtype not in {torch.float16, torch.bfloat16}
        or first_logits.stride(1) != 1
        or later_logits.stride(2) != 1
    ):
        return False
    return True


def fused_ordered_distinct_top1(
    first_logits: torch.Tensor,
    later_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one distinct token per ordered head and its conditioned mass.

    The result matches selecting each head's maximum after masking tokens chosen
    by earlier heads.  Probabilities normalize each head after the same masks.
    """

    if not supports_fused_ordered_selection(first_logits, later_logits):
        raise ValueError("unsupported tensor contract for fused ordered selection")
    batch_size, vocab_size = first_logits.shape
    head_count = later_logits.shape[1] + 1
    selected_tokens = torch.empty(
        (batch_size, head_count),
        dtype=torch.long,
        device=first_logits.device,
    )
    selected_probabilities = torch.empty(
        (batch_size, head_count),
        dtype=torch.float32,
        device=first_logits.device,
    )
    block_size = triton.next_power_of_2(vocab_size)
    for head_index in range(head_count):
        _ordered_argmax_probability_kernel[(batch_size,)](
            first_logits,
            later_logits,
            selected_tokens,
            selected_probabilities,
            later_logits.stride(0),
            later_logits.stride(1),
            later_logits.stride(2),
            vocab_size=vocab_size,
            head_count=head_count,
            head_index=head_index,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return selected_tokens, selected_probabilities


def fused_ordered_head_top1(
    first_logits: torch.Tensor,
    later_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select every head's independent top-one token and native probability."""

    if not supports_fused_ordered_selection(first_logits, later_logits):
        raise ValueError("unsupported tensor contract for fused ordered selection")
    batch_size, vocab_size = first_logits.shape
    head_count = later_logits.shape[1] + 1
    selected_tokens = torch.empty(
        (batch_size, head_count),
        dtype=torch.long,
        device=first_logits.device,
    )
    selected_probabilities = torch.empty(
        (batch_size, head_count),
        dtype=torch.float32,
        device=first_logits.device,
    )
    block_size = triton.next_power_of_2(vocab_size)
    _independent_argmax_probability_kernel[(batch_size * head_count,)](
        first_logits,
        later_logits,
        selected_tokens,
        selected_probabilities,
        later_logits.stride(0),
        later_logits.stride(1),
        later_logits.stride(2),
        vocab_size=vocab_size,
        head_count=head_count,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return selected_tokens, selected_probabilities
