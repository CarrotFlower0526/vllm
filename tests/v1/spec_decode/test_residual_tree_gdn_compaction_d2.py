# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.spec_decode.metadata import TreeSpecDecodeMetadata
from vllm.v1.spec_decode.tree_schema import DraftTokenTree
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _b6d2_tree(*, accepted_d2_parent: int) -> DraftTokenTree:
    if accepted_d2_parent not in (1, 2):
        raise ValueError("the accepted D2 parent must be local node 1 or 2")

    other_parent = 3 - accepted_d2_parent
    parent_node_ids = [
        -1,
        0,
        0,
        other_parent,
        accepted_d2_parent,
        other_parent,
        accepted_d2_parent,
    ]
    children_by_parent = {
        0: [1, 2],
        1: [node_id for node_id in range(3, 7) if parent_node_ids[node_id] == 1],
        2: [node_id for node_id in range(3, 7) if parent_node_ids[node_id] == 2],
    }
    child_node_ids: list[int] = []
    child_start_indices: list[int] = []
    child_end_indices: list[int] = []
    for node_id in range(7):
        child_start_indices.append(len(child_node_ids))
        child_node_ids.extend(children_by_parent.get(node_id, []))
        child_end_indices.append(len(child_node_ids))

    return DraftTokenTree(
        node_token_ids=[-1, 101, 102, 103, 104, 105, 106],
        parent_node_ids=parent_node_ids,
        node_depths=[0, 1, 1, 2, 2, 2, 2],
        node_priorities=[1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
        child_start_indices=child_start_indices,
        child_end_indices=child_end_indices,
        child_node_ids=child_node_ids,
        contributor_child_node_ids=[],
        contributor_proposal_rows=[],
        contributor_head_ids=[],
        contributor_scores=[],
    )


def _tree_metadata(tree: DraftTokenTree) -> TreeSpecDecodeMetadata:
    num_nodes = tree.num_tree_nodes
    return TreeSpecDecodeMetadata(
        node_token_ids=torch.tensor(tree.node_token_ids, dtype=torch.int32),
        parent_node_ids=torch.tensor(tree.parent_node_ids, dtype=torch.int32),
        node_request_indices=torch.zeros(num_nodes, dtype=torch.int32),
        node_depths=torch.tensor(tree.node_depths, dtype=torch.int32),
        node_priorities=torch.tensor(tree.node_priorities, dtype=torch.float32),
        root_node_ids=torch.tensor([0], dtype=torch.int32),
        num_draft_tokens=[tree.num_draft_tokens],
        num_proposal_rows=[0],
        cu_num_tree_nodes=torch.tensor([num_nodes], dtype=torch.int32),
        child_start_indices=torch.tensor(
            tree.child_start_indices, dtype=torch.int32
        ),
        child_end_indices=torch.tensor(tree.child_end_indices, dtype=torch.int32),
        child_node_ids=torch.tensor(tree.child_node_ids, dtype=torch.int32),
        contributor_child_node_ids=torch.empty(0, dtype=torch.int32),
        contributor_proposal_rows=torch.empty(0, dtype=torch.int32),
        contributor_head_ids=torch.empty(0, dtype=torch.int32),
        contributor_scores=torch.empty(0, dtype=torch.float32),
        target_logits_indices=torch.arange(num_nodes, dtype=torch.int32),
        logits_indices=torch.arange(num_nodes, dtype=torch.int32),
    )


@pytest.mark.parametrize("accepted_d2_parent", [1, 2])
def test_b6d2_runner_compaction_commits_local4_for_every_mamba_group(
    accepted_d2_parent: int,
):
    """Exercise the real runner compaction for [1,4] and [2,4] paths.

    The serving B6D2 allocation pads Qwen GDN's three-row convolution history
    to nine rows.  Only the first three rows are recurrent history; the six
    reserved rows are deliberately excluded from cross-arm equality checks.
    """

    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(req_ids=["req"], num_reqs=1)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
    runner.seq_lens = torch.tensor([75], dtype=torch.int32)

    conv_width = 4
    active_conv_rows = conv_width - 1
    padded_conv_rows = active_conv_rows + 6
    conv_dim = 8
    num_blocks = 22
    group_block_tables = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32),
        torch.tensor([[8, 9, 10, 11, 12, 13, 14]], dtype=torch.int32),
        torch.tensor([[15, 16, 17, 18, 19, 20, 21]], dtype=torch.int32),
    ]
    mamba_spec = MambaSpec(
        block_size=816,
        shapes=((padded_conv_rows, conv_dim), (2, 2, 2)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=6,
    )

    groups = []
    forward_context = {}
    per_group_states: list[tuple[torch.Tensor, torch.Tensor]] = []
    expected_active_rows: list[list[torch.Tensor]] = []
    for group_idx, block_table in enumerate(group_block_tables):
        layer_names = [f"g{group_idx}.layer0", f"g{group_idx}.layer1"]
        groups.append(
            SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=layer_names)
        )
        group_states = []
        group_expected = []
        src_block = int(block_table[0, 4])
        dst_block = int(block_table[0, 0])
        for layer_idx, layer_name in enumerate(layer_names):
            conv_state = torch.full(
                (num_blocks, padded_conv_rows, conv_dim),
                -1000 - 100 * group_idx - layer_idx,
                dtype=torch.bfloat16,
            )
            ssm_state = torch.full(
                (num_blocks, 2, 2, 2),
                -2000 - 100 * group_idx - layer_idx,
                dtype=torch.bfloat16,
            )
            active_value = 10 + 10 * group_idx + layer_idx
            conv_state[src_block, :active_conv_rows].fill_(active_value)
            # Reserved rows intentionally differ from the no-spec-sized
            # reference and are not part of the numerical contract.
            conv_state[src_block, active_conv_rows:].fill_(active_value + 100)
            ssm_state[src_block].fill_(active_value + 200)
            expected_active = conv_state[src_block, :active_conv_rows].clone()
            forward_context[layer_name] = SimpleNamespace(
                kv_cache=[conv_state, ssm_state]
            )
            group_states.append((conv_state, ssm_state))
            group_expected.append(expected_active)
            assert not torch.equal(
                conv_state[dst_block, :active_conv_rows], expected_active
            )
        per_group_states.append(group_states)
        expected_active_rows.append(group_expected)

    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    runner.input_batch.block_table = [
        SimpleNamespace(
            get_device_tensor=lambda num_reqs, table=table: table[:num_reqs]
        )
        for table in group_block_tables
    ]
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )

    tree = _b6d2_tree(accepted_d2_parent=accepted_d2_parent)
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor(
            [[accepted_d2_parent, 4, -1, -1, -1, -1]], dtype=torch.int32
        )
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": tree.num_tree_nodes},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        _tree_metadata(tree),
        {},
        cache_slots_are_prevalidated=True,
    )

    for group_idx, (block_table, group_states) in enumerate(
        zip(group_block_tables, per_group_states, strict=True)
    ):
        state_block_table = mamba_get_block_table_tensor(
            block_table,
            runner.seq_lens,
            mamba_spec,
            "align",
        )
        canonical_block = int(state_block_table[0, 0])
        assert canonical_block == int(block_table[0, 0])
        for layer_idx, (conv_state, ssm_state) in enumerate(group_states):
            assert torch.equal(
                conv_state[canonical_block, :active_conv_rows],
                expected_active_rows[group_idx][layer_idx],
            )
            assert torch.equal(
                ssm_state[canonical_block],
                torch.full_like(
                    ssm_state[canonical_block],
                    210 + 10 * group_idx + layer_idx,
                ),
            )

    # The next tree verifier's root has parent -1, which the GDN builder maps
    # to local column zero.  Therefore it must read and write the committed
    # canonical block in every group.
    next_common = SimpleNamespace(
        tree_attn_mask=torch.ones(1, 1, dtype=torch.bool),
        tree_parent_local_indices_cpu=torch.tensor([-1], dtype=torch.int32),
        tree_node_depths_cpu=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_actual_tokens=1,
    )
    for block_table in group_block_tables:
        state_block_table = mamba_get_block_table_tensor(
            block_table,
            runner.seq_lens,
            mamba_spec,
            "align",
        )
        next_metadata = GDNAttentionMetadataBuilder._build_tree_state_metadata(
            object.__new__(GDNAttentionMetadataBuilder),
            next_common,
            state_block_table,
        )
        assert next_metadata is not None
        _, parent_state_indices, child_state_indices, _ = next_metadata
        assert parent_state_indices[0].item() == state_block_table[0, 0].item()
        assert child_state_indices[0].item() == state_block_table[0, 0].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_b6d2_padded_conv_active_rows_match_three_sequential_updates():
    """The routed B6D2 path preserves the three meaningful history rows."""

    torch.manual_seed(29)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    conv_dim = 10240
    conv_width = 4
    active_rows = conv_width - 1
    num_blocks = 8

    raw_tree_state = torch.randn(
        num_blocks,
        active_rows + 6,
        conv_dim,
        dtype=dtype,
        device=device,
    ).mul_(0.1)
    raw_sequential_state = raw_tree_state[:, :active_rows].clone()
    tree_state = raw_tree_state.transpose(-1, -2)
    sequential_state = raw_sequential_state.transpose(-1, -2)
    weight = torch.randn(conv_dim, conv_width, dtype=dtype, device=device).mul_(
        0.02
    )
    bias = torch.randn(conv_dim, dtype=dtype, device=device).mul_(0.02)
    tokens = torch.randn(7, conv_dim, dtype=dtype, device=device).mul_(0.1)

    root_state = torch.tensor([1], dtype=torch.int32, device=device)
    causal_conv1d_update(
        tokens[0:1].clone(),
        tree_state,
        weight,
        bias,
        "silu",
        conv_state_indices=root_state,
        validate_data=False,
    )
    causal_conv1d_update(
        tokens[0:1].clone(),
        sequential_state,
        weight,
        bias,
        "silu",
        conv_state_indices=root_state,
        validate_data=False,
    )

    d1_pairs = torch.tensor([[1, 2], [1, 3]], dtype=torch.int32, device=device)
    source_columns = torch.zeros(2, dtype=torch.int32, device=device)
    destination_columns = torch.ones(2, dtype=torch.int32, device=device)
    d1_out = causal_conv1d_update(
        tokens[1:3].clone(),
        tree_state,
        weight,
        bias,
        "silu",
        conv_state_indices=d1_pairs,
        block_idx_last_scheduled_token=destination_columns,
        initial_state_idx=source_columns,
        validate_data=False,
    )
    sequential_d1_out = causal_conv1d_update(
        tokens[2:3].clone(),
        sequential_state,
        weight,
        bias,
        "silu",
        conv_state_indices=root_state,
        validate_data=False,
    )

    # Local node 4 is a child of local node 2, so physical state block 5
    # advances from physical parent block 3.
    d2_pairs = torch.tensor(
        [[2, 4], [3, 5], [2, 6], [3, 7]], dtype=torch.int32, device=device
    )
    d2_out = causal_conv1d_update(
        tokens[3:7].clone(),
        tree_state,
        weight,
        bias,
        "silu",
        conv_state_indices=d2_pairs,
        block_idx_last_scheduled_token=torch.ones(
            4, dtype=torch.int32, device=device
        ),
        initial_state_idx=torch.zeros(4, dtype=torch.int32, device=device),
        validate_data=False,
    )
    sequential_d2_out = causal_conv1d_update(
        tokens[4:5].clone(),
        sequential_state,
        weight,
        bias,
        "silu",
        conv_state_indices=root_state,
        validate_data=False,
    )

    assert torch.equal(d1_out[1], sequential_d1_out[0])
    assert torch.equal(d2_out[1], sequential_d2_out[0])
    assert torch.equal(
        raw_tree_state[5, :active_rows], raw_sequential_state[1]
    )

    # Commit local node 4 to the canonical slot, then exercise the following
    # verifier root with the tree's padded-nine stride against the ordinary
    # three-row state layout.  Equal active rows must remain sufficient even
    # though the physical block strides differ.
    raw_tree_state[1].copy_(raw_tree_state[5])
    next_token = torch.randn(1, conv_dim, dtype=dtype, device=device).mul_(0.1)
    next_tree_out = causal_conv1d_update(
        next_token.clone(),
        tree_state,
        weight,
        bias,
        "silu",
        conv_state_indices=root_state,
        validate_data=False,
    )
    next_sequential_out = causal_conv1d_update(
        next_token.clone(),
        sequential_state,
        weight,
        bias,
        "silu",
        conv_state_indices=root_state,
        validate_data=False,
    )
    assert torch.equal(next_tree_out, next_sequential_out)
    assert torch.equal(
        raw_tree_state[1, :active_rows], raw_sequential_state[1]
    )
