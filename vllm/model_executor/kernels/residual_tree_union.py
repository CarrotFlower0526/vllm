# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused candidate selection for the residual-tree H1/H2 union path."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_TOP_K = 10
_PARTIAL_BLOCK_SIZE = 512
_GREEDY_EPS = 1e-12


if HAS_TRITON:

    @triton.jit
    def _partial_top10_lse_kernel(
        first_logits_ptr,
        second_logits_ptr,
        partial_values_ptr,
        partial_ids_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        vocab_size: tl.constexpr,
        num_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        lane_offsets = tl.arange(0, BLOCK_SIZE)
        offsets = block * BLOCK_SIZE + lane_offsets
        mask = offsets < vocab_size
        input_offsets = row * vocab_size + offsets
        first = tl.load(
            first_logits_ptr + input_offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        second = tl.load(
            second_logits_ptr + input_offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        first_max = tl.max(first, axis=0)
        second_max = tl.max(second, axis=0)
        first_sum = tl.sum(tl.exp(first - first_max), axis=0)
        second_sum = tl.sum(tl.exp(second - second_max), axis=0)
        first_lse_offset = (row * 2) * num_blocks + block
        second_lse_offset = first_lse_offset + num_blocks
        tl.store(partial_max_ptr + first_lse_offset, first_max)
        tl.store(partial_max_ptr + second_lse_offset, second_max)
        tl.store(partial_sum_ptr + first_lse_offset, first_sum)
        tl.store(partial_sum_ptr + second_lse_offset, second_sum)

        first_output = first_lse_offset * 10
        second_output = second_lse_offset * 10
        for rank in tl.static_range(0, 10):
            first_id = tl.argmax(first, axis=0, tie_break_left=True)
            second_id = tl.argmax(second, axis=0, tie_break_left=True)
            first_value = tl.max(first, axis=0)
            second_value = tl.max(second, axis=0)
            first_token = block * BLOCK_SIZE + first_id
            second_token = block * BLOCK_SIZE + second_id
            tl.store(partial_values_ptr + first_output + rank, first_value)
            tl.store(partial_values_ptr + second_output + rank, second_value)
            tl.store(partial_ids_ptr + first_output + rank, first_token)
            tl.store(partial_ids_ptr + second_output + rank, second_token)
            first = tl.where(lane_offsets == first_id, -float("inf"), first)
            second = tl.where(lane_offsets == second_id, -float("inf"), second)

    @triton.jit
    def _finalize_head_top10_kernel(
        partial_values_ptr,
        partial_ids_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        head_values_ptr,
        head_ids_ptr,
        num_blocks: tl.constexpr,
        CANDIDATE_BLOCK_SIZE: tl.constexpr,
        LSE_BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        head_index = row * 2 + head

        lse_offsets = tl.arange(0, LSE_BLOCK_SIZE)
        lse_mask = lse_offsets < num_blocks
        lse_base = head_index * num_blocks
        block_max = tl.load(
            partial_max_ptr + lse_base + lse_offsets,
            mask=lse_mask,
            other=-float("inf"),
        )
        block_sum = tl.load(
            partial_sum_ptr + lse_base + lse_offsets,
            mask=lse_mask,
            other=0.0,
        )
        global_max = tl.max(block_max, axis=0)
        global_sum = tl.sum(block_sum * tl.exp(block_max - global_max), axis=0)
        normalizer = global_max + tl.log(global_sum)

        candidate_count: tl.constexpr = num_blocks * 10
        candidate_offsets = tl.arange(0, CANDIDATE_BLOCK_SIZE)
        candidate_mask = candidate_offsets < candidate_count
        candidate_base = head_index * candidate_count
        candidates = tl.load(
            partial_values_ptr + candidate_base + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        )
        candidate_ids = tl.load(
            partial_ids_ptr + candidate_base + candidate_offsets,
            mask=candidate_mask,
            other=0,
        )
        output_base = head_index * 10
        for rank in tl.static_range(0, 10):
            candidate_index = tl.argmax(candidates, axis=0, tie_break_left=True)
            candidate_value = tl.max(candidates, axis=0)
            candidate_id = tl.sum(
                tl.where(candidate_offsets == candidate_index, candidate_ids, 0),
                axis=0,
            )
            probability = tl.maximum(tl.exp(candidate_value - normalizer), 1e-12)
            tl.store(head_values_ptr + output_base + rank, probability)
            tl.store(head_ids_ptr + output_base + rank, candidate_id)
            candidates = tl.where(
                candidate_offsets == candidate_index,
                -float("inf"),
                candidates,
            )

    @triton.jit
    def _merge_head_top10_kernel(
        head_values_ptr,
        head_ids_ptr,
        target_ids_ptr,
        clipped_mass_calibration_ptr,
        output_tokens_ptr,
        output_scores_ptr,
        h2_weight,
        USE_CLIPPED_MASS_CALIBRATION: tl.constexpr,
        UNION_BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        lanes = tl.arange(0, UNION_BLOCK_SIZE)
        top_mask = lanes < 10
        first_base = (row * 2) * 10
        second_base = first_base + 10
        first_ids = tl.load(
            head_ids_ptr + first_base + lanes,
            mask=top_mask,
            other=0,
        )
        second_ids = tl.load(
            head_ids_ptr + second_base + lanes,
            mask=top_mask,
            other=0,
        )
        raw_first_scores = tl.load(
            head_values_ptr + first_base + lanes,
            mask=top_mask,
            other=0.0,
        )
        raw_second_scores = tl.load(
            head_values_ptr + second_base + lanes,
            mask=top_mask,
            other=0.0,
        )

        if USE_CLIPPED_MASS_CALIBRATION:
            first_shared = tl.zeros((UNION_BLOCK_SIZE,), dtype=tl.int32)
            second_shared = tl.zeros((UNION_BLOCK_SIZE,), dtype=tl.int32)
            for rank in tl.static_range(0, 10):
                rank_mask = lanes == rank
                second_id = tl.sum(tl.where(rank_mask, second_ids, 0), axis=0)
                overlap = top_mask & (first_ids == second_id)
                first_shared = first_shared | overlap.to(tl.int32)
                second_shared = tl.where(
                    rank_mask,
                    (tl.sum(overlap.to(tl.int32), axis=0) > 0).to(tl.int32),
                    second_shared,
                )
            first_categories = tl.where(first_shared != 0, 1, 0)
            second_categories = tl.where(second_shared != 0, 3, 2)
            first_factors = tl.load(
                clipped_mass_calibration_ptr + first_categories * 10 + lanes,
                mask=top_mask,
                other=0.0,
            )
            second_factors = tl.load(
                clipped_mass_calibration_ptr + second_categories * 10 + lanes,
                mask=top_mask,
                other=0.0,
            )
            first_scores = raw_first_scores * first_factors
            second_scores = raw_second_scores * second_factors
        else:
            first_scores = raw_first_scores
            second_scores = raw_second_scores * h2_weight

        matching_second_scores = tl.zeros((UNION_BLOCK_SIZE,), dtype=tl.float32)
        unique_second_scores = second_scores
        for rank in tl.static_range(0, 10):
            rank_mask = lanes == rank
            second_id = tl.sum(tl.where(rank_mask, second_ids, 0), axis=0)
            second_score = tl.sum(tl.where(rank_mask, second_scores, 0.0), axis=0)
            overlap = top_mask & (first_ids == second_id)
            matching_second_scores = tl.where(
                overlap,
                tl.maximum(matching_second_scores, second_score),
                matching_second_scores,
            )
            is_duplicate = tl.sum(overlap.to(tl.int32), axis=0) > 0
            unique_second_scores = tl.where(
                rank_mask & is_duplicate,
                0.0,
                unique_second_scores,
            )

        if USE_CLIPPED_MASS_CALIBRATION:
            merged_first_scores = tl.minimum(
                first_scores + matching_second_scores, 1.0
            )
        else:
            merged_first_scores = tl.maximum(first_scores, matching_second_scores)
        union_mask = lanes < 20
        union_ids = tl.where(
            lanes < 10,
            first_ids,
            tl.load(
                head_ids_ptr + second_base + lanes - 10,
                mask=(lanes >= 10) & union_mask,
                other=0,
            ),
        )
        union_scores = tl.where(lanes < 10, merged_first_scores, 0.0)
        for rank in tl.static_range(0, 10):
            rank_mask = lanes == rank + 10
            second_score = tl.sum(
                tl.where(lanes == rank, second_scores, 0.0), axis=0
            )
            union_scores = tl.where(rank_mask, second_score, union_scores)
        # Zero the H2 entries that duplicate H1.  Rebuild this mask in the
        # union lanes to avoid materializing a 10x10 comparison matrix.
        for rank in tl.static_range(0, 10):
            rank_mask = lanes == rank + 10
            second_id = tl.sum(tl.where(rank_mask, union_ids, 0), axis=0)
            is_duplicate = (
                tl.sum((top_mask & (first_ids == second_id)).to(tl.int32), axis=0) > 0
            )
            union_scores = tl.where(rank_mask & is_duplicate, 0.0, union_scores)
        union_scores = tl.where(union_mask, union_scores, -float("inf"))

        output_base = row * 10
        for rank in tl.static_range(0, 10):
            selected_index = tl.argmax(union_scores, axis=0, tie_break_left=True)
            selected_score = tl.max(union_scores, axis=0)
            selected_id = tl.sum(
                tl.where(lanes == selected_index, union_ids, 0), axis=0
            )
            target_id = tl.load(target_ids_ptr + selected_id)
            tl.store(output_tokens_ptr + output_base + rank, target_id)
            tl.store(output_scores_ptr + output_base + rank, selected_score)
            union_scores = tl.where(
                lanes == selected_index,
                -float("inf"),
                union_scores,
            )


def supports_fused_hybrid_union(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the CUDA kernel's exact contract."""

    return bool(
        HAS_TRITON
        and first_logits.is_cuda
        and second_logits.is_cuda
        and target_ids.is_cuda
        and first_logits.ndim == 2
        and second_logits.shape == first_logits.shape
        and first_logits.dtype in {torch.float16, torch.bfloat16}
        and second_logits.dtype == first_logits.dtype
        and target_ids.dtype == torch.long
        and first_logits.is_contiguous()
        and second_logits.is_contiguous()
        and target_ids.is_contiguous()
        and first_logits.shape[0] > 0
        and first_logits.shape[1] >= _TOP_K
    )


def fused_hybrid_union_top10(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    h2_weight: float,
    clipped_mass_calibration: torch.Tensor | None = None,
    return_head_candidates: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Return a fused H1/H2 unique union.

    The optional ``[4, 10]`` calibration contains multiplicative clipped-mass
    factors for H1-only, H1-shared, H2-only, and H2-shared native ranks.  A
    shared token adds the two calibrated non-overlapping contributions.  The
    default path remains the historical weighted maximum exactly.
    """

    if not supports_fused_hybrid_union(first_logits, second_logits, target_ids):
        raise ValueError("unsupported tensor contract for fused hybrid union")
    if not 0.0 < h2_weight <= 1.0:
        raise ValueError("fused hybrid union requires 0 < H2 weight <= 1")
    if clipped_mass_calibration is not None and (
        clipped_mass_calibration.shape != (4, _TOP_K)
        or clipped_mass_calibration.device != first_logits.device
        or clipped_mass_calibration.dtype != torch.float32
        or not clipped_mass_calibration.is_contiguous()
    ):
        raise ValueError(
            "clipped-mass calibration must be a contiguous CUDA float32 [4, 10] "
            "tensor on the logits device"
        )

    batch_size, vocab_size = first_logits.shape
    num_blocks = triton.cdiv(vocab_size, _PARTIAL_BLOCK_SIZE)
    candidate_count = num_blocks * _TOP_K
    candidate_block_size = triton.next_power_of_2(candidate_count)
    lse_block_size = triton.next_power_of_2(num_blocks)

    partial_shape = (batch_size, 2, num_blocks)
    partial_values = torch.empty(
        (*partial_shape, _TOP_K),
        device=first_logits.device,
        dtype=torch.float32,
    )
    partial_ids = torch.empty(
        (*partial_shape, _TOP_K),
        device=first_logits.device,
        dtype=torch.int32,
    )
    partial_max = torch.empty(
        partial_shape, device=first_logits.device, dtype=torch.float32
    )
    partial_sum = torch.empty_like(partial_max)
    head_values = torch.empty(
        (batch_size, 2, _TOP_K),
        device=first_logits.device,
        dtype=torch.float32,
    )
    head_ids = torch.empty(
        (batch_size, 2, _TOP_K),
        device=first_logits.device,
        dtype=torch.int32,
    )
    output_tokens = torch.empty(
        (batch_size, _TOP_K),
        device=first_logits.device,
        dtype=torch.long,
    )
    output_scores = torch.empty(
        (batch_size, _TOP_K),
        device=first_logits.device,
        dtype=torch.float32,
    )

    _partial_top10_lse_kernel[(batch_size, num_blocks)](
        first_logits,
        second_logits,
        partial_values,
        partial_ids,
        partial_max,
        partial_sum,
        vocab_size=vocab_size,
        num_blocks=num_blocks,
        BLOCK_SIZE=_PARTIAL_BLOCK_SIZE,
        num_warps=8,
    )
    _finalize_head_top10_kernel[(batch_size, 2)](
        partial_values,
        partial_ids,
        partial_max,
        partial_sum,
        head_values,
        head_ids,
        num_blocks=num_blocks,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        LSE_BLOCK_SIZE=lse_block_size,
        num_warps=8,
    )
    _merge_head_top10_kernel[(batch_size,)](
        head_values,
        head_ids,
        target_ids,
        (
            clipped_mass_calibration
            if clipped_mass_calibration is not None
            else head_values
        ),
        output_tokens,
        output_scores,
        h2_weight,
        USE_CLIPPED_MASS_CALIBRATION=clipped_mass_calibration is not None,
        UNION_BLOCK_SIZE=32,
        num_warps=1,
    )
    if return_head_candidates:
        return output_tokens, output_scores, head_ids, head_values
    return output_tokens, output_scores
