# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for packed/ragged Triton tree-attention query shapes."""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _binary_tree_mask(query_len: int, width: int) -> torch.Tensor:
    mask = torch.zeros(query_len, width, dtype=torch.bool, device="cuda")
    parents = [-1] + [(node - 1) // 2 for node in range(1, query_len)]
    for node in range(query_len):
        ancestor = node
        while ancestor >= 0:
            mask[node, ancestor] = True
            ancestor = parents[ancestor]
    return mask


def _tree_mask_from_parents(
    parents: list[int], width: int | None = None
) -> torch.Tensor:
    query_len = len(parents)
    width = query_len if width is None else width
    mask = torch.zeros(query_len, width, dtype=torch.bool, device="cuda")
    for node in range(query_len):
        ancestor = node
        while ancestor >= 0:
            mask[node, ancestor] = True
            ancestor = parents[ancestor]
    return mask


def _make_case(
    *,
    query_lens: list[int],
    context_lens: list[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int = 16,
    seed: int = 0,
) -> dict[str, torch.Tensor | list[int] | float]:
    assert len(query_lens) == len(context_lens)
    assert num_query_heads % num_kv_heads == 0
    torch.manual_seed(seed)

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv_lens = [context + query for context, query in zip(context_lens, query_lens)]
    blocks_per_seq = [math.ceil(length / block_size) for length in kv_lens]
    num_blocks = sum(blocks_per_seq)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(
        len(query_lens),
        max(blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    block_cursor = 0
    for row, count in enumerate(blocks_per_seq):
        block_table[row, :count] = torch.arange(
            block_cursor,
            block_cursor + count,
            dtype=torch.int32,
            device="cuda",
        )
        block_cursor += count

    max_query_len = max(query_lens)
    tree_mask = torch.cat(
        [_binary_tree_mask(length, max_query_len) for length in query_lens], dim=0
    )
    query_start_loc = torch.tensor(
        [0, *query_lens], dtype=torch.int32, device="cuda"
    ).cumsum(0)
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "tree_mask": tree_mask,
        "query_start_loc": query_start_loc,
        "seq_lens": torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        "query_lens": query_lens,
        "context_lens": context_lens,
        "kv_lens": kv_lens,
        "scale": head_size**-0.5,
    }


def _reference(case: dict) -> torch.Tensor:
    query = case["query"]
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    block_table = case["block_table"]
    tree_mask = case["tree_mask"]
    query_lens = case["query_lens"]
    context_lens = case["context_lens"]
    kv_lens = case["kv_lens"]
    scale = case["scale"]
    block_size = key_cache.shape[1]
    outputs = []
    query_cursor = 0
    for seq_idx, (query_len, context_len, kv_len) in enumerate(
        zip(query_lens, context_lens, kv_lens)
    ):
        num_blocks = math.ceil(kv_len / block_size)
        blocks = block_table[seq_idx, :num_blocks].long()
        keys = key_cache[blocks].flatten(0, 1)[:kv_len]
        values = value_cache[blocks].flatten(0, 1)[:kv_len]
        queries = query[query_cursor : query_cursor + query_len]
        local_tree_mask = tree_mask[
            query_cursor : query_cursor + query_len, :query_len
        ]
        if queries.shape[1] != keys.shape[1]:
            repeats = queries.shape[1] // keys.shape[1]
            keys = torch.repeat_interleave(keys, repeats, dim=1)
            values = torch.repeat_interleave(values, repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", queries * scale, keys).float()
        allowed = torch.cat(
            (
                torch.ones(
                    query_len, context_len, dtype=torch.bool, device="cuda"
                ),
                local_tree_mask,
            ),
            dim=1,
        )
        scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(values.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", weights, values))
        query_cursor += query_len
    return torch.cat(outputs, dim=0)


def _run(case: dict, *, provide_ordinary_3d_buffers: bool = False) -> torch.Tensor:
    query = case["query"]
    output = torch.empty_like(query)
    ordinary_3d_kwargs = {}
    if provide_ordinary_3d_buffers:
        # These buffers make ordinary qlen=1 dispatch eligible for its 3D
        # decode kernel. Tree attention must ignore that ordinary heuristic
        # and dispatch its dedicated one-node kernel directly.
        ordinary_3d_kwargs = {
            "seq_threshold_3D": 128,
            "num_par_softmax_segments": 2,
            "softmax_segm_output": torch.empty(1, device="cuda"),
            "softmax_segm_max": torch.empty(1, device="cuda"),
            "softmax_segm_expsum": torch.empty(1, device="cuda"),
        }
    unified_attention(
        q=query,
        k=case["key_cache"],
        v=case["value_cache"],
        out=output,
        cu_seqlens_q=case["query_start_loc"],
        max_seqlen_q=max(case["query_lens"]),
        seqused_k=case["seq_lens"],
        max_seqlen_k=max(case["kv_lens"]),
        softmax_scale=case["scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=case["block_table"],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        tree_attn_mask=case["tree_mask"],
        **ordinary_3d_kwargs,
    )
    return output


def _run_ordinary_single_query(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    num_blocks = math.ceil(keys.shape[0] / block_size)
    key_cache = torch.zeros(
        num_blocks,
        block_size,
        keys.shape[1],
        keys.shape[2],
        dtype=keys.dtype,
        device="cuda",
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache.flatten(0, 1)[: keys.shape[0]].copy_(keys)
    value_cache.flatten(0, 1)[: values.shape[0]].copy_(values)
    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        max_seqlen_q=1,
        seqused_k=torch.tensor(
            [keys.shape[0]], dtype=torch.int32, device="cuda"
        ),
        max_seqlen_k=keys.shape[0],
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.arange(
            num_blocks, dtype=torch.int32, device="cuda"
        )[None, :],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        tree_attn_mask=None,
    )
    return output[0]


def _ordinary_path_reference(case: dict) -> torch.Tensor:
    query = case["query"]
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    block_table = case["block_table"]
    tree_mask = case["tree_mask"]
    query_lens = case["query_lens"]
    context_lens = case["context_lens"]
    kv_lens = case["kv_lens"]
    block_size = key_cache.shape[1]
    query_cursor = 0
    outputs = []
    for seq_idx, (query_len, context_len, kv_len) in enumerate(
        zip(query_lens, context_lens, kv_lens)
    ):
        num_blocks = math.ceil(kv_len / block_size)
        blocks = block_table[seq_idx, :num_blocks].long()
        keys = key_cache[blocks].flatten(0, 1)[:kv_len]
        values = value_cache[blocks].flatten(0, 1)[:kv_len]
        prefix = torch.arange(context_len, device="cuda")
        for local_row in range(query_len):
            allowed = torch.nonzero(
                tree_mask[query_cursor + local_row, :query_len],
                as_tuple=False,
            ).flatten()
            logical_positions = torch.cat((prefix, context_len + allowed))
            outputs.append(
                _run_ordinary_single_query(
                    query=query[
                        query_cursor + local_row : query_cursor + local_row + 1
                    ],
                    keys=keys[logical_positions],
                    values=values[logical_positions],
                    scale=case["scale"],
                    block_size=block_size,
                )
            )
        query_cursor += query_len
    return torch.stack(outputs)


@pytest.mark.parametrize(
    ("query_lens", "context_lens", "num_query_heads", "num_kv_heads", "head_size"),
    [
        ([3], [37], 24, 4, 256),  # Qwen3.5 B2D1: GQA=6, partial block.
        ([7], [29], 24, 4, 128),  # Qwen3.5 B6D2 query count.
        ([3, 7, 13], [5, 29, 17], 24, 4, 64),  # Ragged packed requests.
        ([1, 2, 3, 7, 40], [3, 5, 9, 17, 23], 24, 4, 64),
        ([3, 5, 7], [11, 19, 7], 10, 2, 64),  # GQA=5 partial blocks.
    ],
)
@torch.inference_mode()
def test_tree_attention_packed_shapes_match_reference(
    query_lens: list[int],
    context_lens: list[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> None:
    case = _make_case(
        query_lens=query_lens,
        context_lens=context_lens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
    actual = _run(case)
    expected = _reference(case)
    torch.testing.assert_close(actual, expected, atol=1.5e-2, rtol=1e-2)


@torch.inference_mode()
def test_tree_attention_b2d1_is_deterministic_per_row() -> None:
    case = _make_case(
        query_lens=[3],
        context_lens=[37],
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
    )
    first = _run(case)
    for _ in range(50):
        assert torch.equal(_run(case), first)


@torch.inference_mode()
def test_tree_attention_b2d1_b6d2_tile_boundaries_are_bitwise_ordinary() -> None:
    contexts = (29, 30, 31, 32, 61, 62, 63, 64, 93, 94, 95, 96, 392)
    for query_len in (3, 7):
        for context_len in contexts:
            for seed in range(4):
                case = _make_case(
                    query_lens=[query_len],
                    context_lens=[context_len],
                    num_query_heads=24,
                    num_kv_heads=4,
                    head_size=256,
                    block_size=128,
                    seed=seed,
                )
                assert torch.equal(_run(case), _ordinary_path_reference(case)), (
                    query_len,
                    context_len,
                    seed,
                )


@torch.inference_mode()
def test_tree_attention_ragged_requests_are_bitwise_ordinary() -> None:
    case = _make_case(
        query_lens=[3, 7, 13],
        context_lens=[5, 94, 31],
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
        block_size=64,
        seed=17,
    )
    assert torch.equal(_run(case), _ordinary_path_reference(case))


@torch.inference_mode()
def test_tree_attention_nonheap_d3_paths_are_bitwise_ordinary() -> None:
    parents = [-1, 0, 0, 2, 1, 4, 2, 5, 3, 8]
    case = _make_case(
        query_lens=[len(parents)],
        context_lens=[62],
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
        block_size=64,
        seed=23,
    )
    case["tree_mask"] = _tree_mask_from_parents(parents)
    assert torch.equal(_run(case), _ordinary_path_reference(case))


@torch.inference_mode()
def test_tree_attention_b6d2_is_repeat_deterministic_and_bitwise_ordinary() -> None:
    case = _make_case(
        query_lens=[7],
        context_lens=[94],
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
        block_size=128,
        seed=29,
    )
    expected = _ordinary_path_reference(case)
    first = _run(case)
    assert torch.equal(first, expected)
    for _ in range(50):
        assert torch.equal(_run(case), first)


@torch.inference_mode()
def test_tree_attention_root_only_bypasses_ordinary_3d_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(unified_attention.__globals__, "is_batch_invariant", False)
    case = _make_case(
        query_lens=[1, 1, 1],
        context_lens=[31, 62, 95],
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
        block_size=128,
        seed=31,
    )
    actual = _run(case, provide_ordinary_3d_buffers=True)
    assert torch.equal(actual, _ordinary_path_reference(case))


@torch.inference_mode()
def test_tree_attention_noncontiguous_mask_uses_logical_width() -> None:
    case = _make_case(
        query_lens=[7],
        context_lens=[95],
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
        block_size=128,
        seed=37,
    )
    logical_mask = case["tree_mask"]
    padded = torch.ones(
        logical_mask.shape[0],
        logical_mask.shape[1] + 5,
        dtype=torch.bool,
        device="cuda",
    )
    padded[:, : logical_mask.shape[1]].copy_(logical_mask)
    case["tree_mask"] = padded[:, : logical_mask.shape[1]]
    assert not case["tree_mask"].is_contiguous()
    assert case["tree_mask"].stride(0) > case["tree_mask"].shape[1]
    assert torch.equal(_run(case), _ordinary_path_reference(case))
