# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen_gdn
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)


def test_tree_metadata_precomputes_state_routing_tensors():
    common = SimpleNamespace(
        tree_attn_mask=torch.empty(0),
        tree_parent_local_indices_cpu=torch.tensor(
            [-1, 0, 0, -1, 0, 0], dtype=torch.int32
        ),
        tree_node_depths_cpu=torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        num_actual_tokens=6,
    )
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)

    result = GDNAttentionMetadataBuilder._build_tree_state_metadata(
        object.__new__(GDNAttentionMetadataBuilder),
        common,
        block_table,
    )

    assert result is not None
    (
        token_indices,
        parent_indices,
        child_indices,
        _,
    ) = result
    (
        state_pairs,
        parent_indices_long,
        child_indices_long,
        source_columns,
        destination_columns,
    ) = GDNAttentionMetadataBuilder._build_tree_state_routing_tensors(
        parent_indices,
        child_indices,
    )
    assert torch.equal(token_indices[0], torch.tensor([0, 3]))
    assert torch.equal(token_indices[1], torch.tensor([1, 2, 4, 5]))
    assert torch.equal(parent_indices[0], torch.tensor([10, 20]))
    assert torch.equal(child_indices[0], parent_indices[0])
    assert torch.equal(
        state_pairs[1],
        torch.tensor([[10, 11], [10, 12], [20, 21], [20, 22]]),
    )
    assert parent_indices_long[1].dtype == torch.long
    assert child_indices_long[1].dtype == torch.long
    assert torch.equal(source_columns[1], torch.zeros(4, dtype=torch.int32))
    assert torch.equal(destination_columns[1], torch.ones(4, dtype=torch.int32))


def test_output_projection_dispatches_ordinary_tree_ordinary_once(monkeypatch):
    calls: list[str] = []

    def ordinary_projection(core_attn_out, z, output, num_tokens):
        del core_attn_out, z
        calls.append("ordinary_projection")
        output[:num_tokens].fill_(1)

    def tree_norm(x, weight, z, eps, activation):
        del weight, z, eps, activation
        calls.append("tree_norm")
        return torch.full_like(x, 2)

    def tree_core(**kwargs):
        del kwargs
        calls.append("tree_core")

    def tree_out_proj(x):
        calls.append("tree_out_proj")
        return x, None

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        tree_depth_token_indices=(torch.tensor([0]),),
    )
    layer = SimpleNamespace(
        prefix="test.linear_attn",
        enable_packed_recurrent_decode=False,
        kv_cache=(torch.empty(1, 2, 3), torch.empty(1, 1, 2, 2)),
        conv1d=SimpleNamespace(weight=torch.empty(4, 1, 3)),
        _forward_core_tree=tree_core,
        _tree_output_projection_pending=False,
        _ordinary_output_projection=ordinary_projection,
        norm=SimpleNamespace(
            weight=torch.ones(2),
            eps=1e-6,
            activation="silu",
        ),
        out_proj=tree_out_proj,
    )
    monkeypatch.setattr(qwen_gdn, "rmsnorm_gated_tree_fn", tree_norm)
    monkeypatch.setattr(
        qwen_gdn,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"test.linear_attn": metadata}
        ),
    )

    core_attn_out = torch.zeros(1, 1, 2)
    z = torch.zeros_like(core_attn_out)
    output = torch.zeros(1, 2)

    qwen_gdn.QwenGatedDeltaNetAttention._output_projection_tree_dispatch(
        layer, core_attn_out, z, output, num_tokens=1
    )
    assert not layer._tree_output_projection_pending
    assert torch.equal(output, torch.ones_like(output))

    qwen_gdn.QwenGatedDeltaNetAttention._forward_core(
        layer,
        mixed_qkv=torch.zeros(1, 4),
        b=torch.zeros(1, 1),
        a=torch.zeros(1, 1),
        core_attn_out=core_attn_out,
    )
    assert layer._tree_output_projection_pending

    qwen_gdn.QwenGatedDeltaNetAttention._output_projection_tree_dispatch(
        layer, core_attn_out, z, output, num_tokens=1
    )
    assert not layer._tree_output_projection_pending
    assert torch.equal(output, torch.full_like(output, 2))

    qwen_gdn.QwenGatedDeltaNetAttention._output_projection_tree_dispatch(
        layer, core_attn_out, z, output, num_tokens=1
    )
    assert not layer._tree_output_projection_pending
    assert torch.equal(output, torch.ones_like(output))
    assert calls == [
        "ordinary_projection",
        "tree_core",
        "tree_norm",
        "tree_out_proj",
        "ordinary_projection",
    ]


def test_tree_core_routes_root_and_siblings_without_state_copies(monkeypatch):
    conv_calls: list[dict] = []
    recurrent_calls: list[dict] = []
    sigmoid_calls: list[dict] = []

    def fake_conv(mixed_qkv, *_args, **kwargs):
        conv_calls.append(kwargs)
        return mixed_qkv

    def fake_packed(**kwargs):
        recurrent_calls.append(kwargs)
        kwargs["out"].zero_()

    def fake_sigmoid(**kwargs):
        sigmoid_calls.append(kwargs)
        num_rows = kwargs["a"].shape[0]
        return torch.zeros(1, num_rows, 1, 2), kwargs["initial_state"]

    monkeypatch.setattr(qwen_gdn, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(
        qwen_gdn,
        "fused_recurrent_gated_delta_rule_packed_decode",
        fake_packed,
    )
    monkeypatch.setattr(
        qwen_gdn,
        "fused_sigmoid_gating_delta_rule_update",
        fake_sigmoid,
    )

    token_indices = (torch.tensor([0, 3]), torch.tensor([1, 2, 4, 5]))
    parent_indices = (
        torch.tensor([1, 4], dtype=torch.int32),
        torch.tensor([1, 1, 4, 4], dtype=torch.int32),
    )
    child_indices = (
        torch.tensor([1, 4], dtype=torch.int32),
        torch.tensor([2, 3, 5, 6], dtype=torch.int32),
    )
    state_pairs = tuple(
        torch.stack((parent, child), dim=1)
        for parent, child in zip(parent_indices, child_indices, strict=True)
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=6,
        tree_depth_token_indices=token_indices,
        tree_depth_parent_state_indices=parent_indices,
        tree_depth_child_state_indices=child_indices,
        tree_depth_state_index_pairs=state_pairs,
        tree_depth_parent_state_indices_long=tuple(
            tensor.long() for tensor in parent_indices
        ),
        tree_depth_child_state_indices_long=tuple(
            tensor.long() for tensor in child_indices
        ),
        tree_depth_source_columns=(
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        ),
        tree_depth_destination_columns=(
            torch.ones(2, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
        ),
        tree_depth_query_start_locs=(
            torch.arange(3, dtype=torch.int32),
            torch.arange(5, dtype=torch.int32),
        ),
    )
    dummy = SimpleNamespace(
        conv1d=SimpleNamespace(bias=None),
        activation=None,
        A_log=torch.ones(1),
        dt_bias=torch.ones(1),
        head_k_dim=2,
        rearrange_mixed_qkv=lambda mixed: (
            mixed.unsqueeze(0).unsqueeze(2),
            mixed.unsqueeze(0).unsqueeze(2),
            mixed.unsqueeze(0).unsqueeze(2),
        ),
    )
    conv_state = torch.randn(7, 4, 2)
    ssm_state = torch.randn(7, 1, 2, 2)
    conv_state_before = conv_state.clone()
    ssm_state_before = ssm_state.clone()

    qwen_gdn.QwenGatedDeltaNetAttention._forward_core_tree(
        dummy,
        mixed_qkv=torch.randn(6, 4),
        b=torch.randn(6, 1),
        a=torch.randn(6, 1),
        core_attn_out=torch.empty(6, 1, 2),
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=torch.randn(4, 3),
        attn_metadata=metadata,
    )

    assert conv_calls[0]["conv_state_indices"].ndim == 1
    assert "initial_state_idx" not in conv_calls[0]
    assert torch.equal(conv_calls[1]["conv_state_indices"], state_pairs[1])
    assert torch.equal(
        conv_calls[1]["initial_state_idx"], torch.zeros(4, dtype=torch.int32)
    )
    assert torch.equal(
        conv_calls[1]["block_idx_last_scheduled_token"],
        torch.ones(4, dtype=torch.int32),
    )
    assert sigmoid_calls == []
    assert len(recurrent_calls) == 2
    assert torch.equal(recurrent_calls[0]["ssm_state_indices"], parent_indices[0])
    assert torch.equal(recurrent_calls[0]["final_state_indices"], child_indices[0])
    assert recurrent_calls[0]["mixed_qkv"].shape == (2, 4)
    assert recurrent_calls[0]["out"].shape == (2, 1, 1, 2)
    assert torch.equal(recurrent_calls[1]["ssm_state_indices"], parent_indices[1])
    assert torch.equal(recurrent_calls[1]["final_state_indices"], child_indices[1])
    assert recurrent_calls[1]["mixed_qkv"].shape == (4, 4)
    assert recurrent_calls[1]["out"].shape == (4, 1, 1, 2)
    assert torch.equal(conv_state, conv_state_before)
    assert torch.equal(ssm_state, ssm_state_before)


@pytest.mark.parametrize("branch_width", [1, 2, 6])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_tree_core_packed_direct_matches_ordinary_decode_bitwise(branch_width):
    """Every tree depth must reproduce copied-state ordinary one-token decode."""

    torch.manual_seed(101 + branch_width)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.5-27B GDN dimensions on one tensor-parallel rank.
    num_query_heads = 16
    num_value_heads = 48
    key_dim = 128
    value_dim = 128
    packed_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    conv_width = 4
    num_rows = branch_width + 1
    num_states = branch_width + 3

    root = torch.tensor([1], dtype=torch.int32, device=device)
    children = torch.arange(
        2,
        branch_width + 2,
        dtype=torch.int32,
        device=device,
    )
    child_parents = torch.ones(branch_width, dtype=torch.int32, device=device)
    token_indices = (
        torch.tensor([0], dtype=torch.long, device=device),
        torch.arange(1, num_rows, dtype=torch.long, device=device),
    )
    parent_indices = (root, child_parents)
    child_indices = (root.clone(), children)
    state_pairs = tuple(
        torch.stack((parent, child), dim=1).contiguous()
        for parent, child in zip(parent_indices, child_indices, strict=True)
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_rows,
        tree_depth_token_indices=token_indices,
        tree_depth_parent_state_indices=parent_indices,
        tree_depth_child_state_indices=child_indices,
        tree_depth_state_index_pairs=state_pairs,
        tree_depth_parent_state_indices_long=tuple(
            tensor.long() for tensor in parent_indices
        ),
        tree_depth_child_state_indices_long=tuple(
            tensor.long() for tensor in child_indices
        ),
        tree_depth_source_columns=(
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(branch_width, dtype=torch.int32, device=device),
        ),
        tree_depth_destination_columns=(
            torch.ones(1, dtype=torch.int32, device=device),
            torch.ones(branch_width, dtype=torch.int32, device=device),
        ),
    )

    mixed_qkv = torch.randn(
        num_rows, packed_dim, dtype=dtype, device=device
    ).mul_(0.1)
    a = torch.randn(num_rows, num_value_heads, dtype=dtype, device=device).mul_(
        0.1
    )
    b = torch.randn_like(a).mul_(0.1)
    conv_weights = torch.randn(
        packed_dim, conv_width, dtype=dtype, device=device
    ).mul_(0.1)
    conv_bias = torch.randn(packed_dim, dtype=dtype, device=device).mul_(0.1)
    A_log = torch.randn(
        num_value_heads, dtype=torch.float32, device=device
    ).mul_(0.1)
    dt_bias = torch.randn(num_value_heads, dtype=dtype, device=device).mul_(0.1)
    conv_seed = torch.randn(
        num_states,
        packed_dim,
        conv_width - 1,
        dtype=dtype,
        device=device,
    ).mul_(0.1)
    ssm_seed = torch.randn(
        num_states,
        num_value_heads,
        value_dim,
        key_dim,
        dtype=dtype,
        device=device,
    ).mul_(0.1)

    dummy = SimpleNamespace(
        prefix="test.linear_attn",
        conv1d=SimpleNamespace(bias=conv_bias),
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        head_k_dim=key_dim,
    )
    tree_conv_state = conv_seed.clone()
    tree_ssm_state = ssm_seed.clone()
    tree_output = torch.empty(
        num_rows,
        num_value_heads,
        value_dim,
        dtype=dtype,
        device=device,
    )
    qwen_gdn.QwenGatedDeltaNetAttention._forward_core_tree(
        dummy,
        mixed_qkv=mixed_qkv.clone(),
        b=b,
        a=a,
        core_attn_out=tree_output,
        conv_state=tree_conv_state,
        ssm_state=tree_ssm_state,
        conv_weights=conv_weights,
        attn_metadata=metadata,
    )

    # Reference: materialize the parent state into each child slot, then run
    # the unmodified ordinary one-token kernels in-place at those slots.
    reference_conv_state = conv_seed.clone()
    reference_ssm_state = ssm_seed.clone()
    reference_output = torch.empty_like(tree_output)

    root_mixed = causal_conv1d_update(
        mixed_qkv[:1].clone(),
        reference_conv_state,
        conv_weights,
        conv_bias,
        "silu",
        conv_state_indices=root,
        validate_data=False,
    )
    root_output = torch.empty(
        1, 1, num_value_heads, value_dim, dtype=dtype, device=device
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=root_mixed,
        a=a[:1],
        b=b[:1],
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=reference_ssm_state,
        out=root_output,
        ssm_state_indices=root,
        use_qk_l2norm_in_kernel=True,
    )
    reference_output[:1].copy_(root_output.squeeze(1))

    reference_conv_state.index_copy_(
        0,
        children.long(),
        reference_conv_state.index_select(0, child_parents.long()).clone(),
    )
    child_mixed = causal_conv1d_update(
        mixed_qkv[1:].clone(),
        reference_conv_state,
        conv_weights,
        conv_bias,
        "silu",
        conv_state_indices=children,
        validate_data=False,
    )
    reference_ssm_state.index_copy_(
        0,
        children.long(),
        reference_ssm_state.index_select(0, child_parents.long()).clone(),
    )
    child_output = torch.empty(
        branch_width,
        1,
        num_value_heads,
        value_dim,
        dtype=dtype,
        device=device,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=child_mixed,
        a=a[1:],
        b=b[1:],
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=reference_ssm_state,
        out=child_output,
        ssm_state_indices=children,
        use_qk_l2norm_in_kernel=True,
    )
    reference_output[1:].copy_(child_output.squeeze(1))

    assert torch.equal(tree_output, reference_output)
    assert torch.equal(tree_conv_state, reference_conv_state)
    assert torch.equal(tree_ssm_state, reference_ssm_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_tree_core_true_depth2_chain_and_commit_match_ordinary_bitwise():
    """D2 children must consume freshly written D1 parent states exactly."""

    torch.manual_seed(211)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 16
    num_value_heads = 48
    key_dim = 128
    value_dim = 128
    packed_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    conv_width = 4

    # State slot zero is the backend's invalid sentinel. Local tree nodes
    # 0..6 use slots 1..7; local node 4 is the accepted D2 leaf below node 2.
    token_depths = (
        torch.tensor([0], dtype=torch.long, device=device),
        torch.tensor([1, 2], dtype=torch.long, device=device),
        torch.tensor([3, 4, 5, 6], dtype=torch.long, device=device),
    )
    parent_depths = (
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([1, 1], dtype=torch.int32, device=device),
        torch.tensor([2, 3, 2, 3], dtype=torch.int32, device=device),
    )
    child_depths = (
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([2, 3], dtype=torch.int32, device=device),
        torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=device),
    )
    state_pairs = tuple(
        torch.stack((parent, child), dim=1).contiguous()
        for parent, child in zip(parent_depths, child_depths, strict=True)
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=7,
        tree_depth_token_indices=token_depths,
        tree_depth_parent_state_indices=parent_depths,
        tree_depth_child_state_indices=child_depths,
        tree_depth_state_index_pairs=state_pairs,
        tree_depth_source_columns=tuple(
            torch.zeros_like(parent) for parent in parent_depths
        ),
        tree_depth_destination_columns=tuple(
            torch.ones_like(child) for child in child_depths
        ),
    )

    mixed_qkv = torch.randn(7, packed_dim, dtype=dtype, device=device).mul_(0.1)
    a = torch.randn(7, num_value_heads, dtype=dtype, device=device).mul_(0.1)
    b = torch.randn_like(a).mul_(0.1)
    conv_weights = torch.randn(
        packed_dim, conv_width, dtype=dtype, device=device
    ).mul_(0.1)
    conv_bias = torch.randn(packed_dim, dtype=dtype, device=device).mul_(0.1)
    A_log = torch.randn(
        num_value_heads, dtype=torch.float32, device=device
    ).mul_(0.1)
    dt_bias = torch.randn(num_value_heads, dtype=dtype, device=device).mul_(0.1)
    conv_seed = torch.randn(
        9,
        packed_dim,
        conv_width - 1,
        dtype=dtype,
        device=device,
    ).mul_(0.1)
    ssm_seed = torch.randn(
        9,
        num_value_heads,
        value_dim,
        key_dim,
        dtype=dtype,
        device=device,
    ).mul_(0.1)
    dummy = SimpleNamespace(
        prefix="test.linear_attn",
        conv1d=SimpleNamespace(bias=conv_bias),
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        head_k_dim=key_dim,
    )

    tree_conv = conv_seed.clone()
    tree_ssm = ssm_seed.clone()
    tree_output = torch.empty(
        7, num_value_heads, value_dim, dtype=dtype, device=device
    )
    qwen_gdn.QwenGatedDeltaNetAttention._forward_core_tree(
        dummy,
        mixed_qkv=mixed_qkv.clone(),
        b=b,
        a=a,
        core_attn_out=tree_output,
        conv_state=tree_conv,
        ssm_state=tree_ssm,
        conv_weights=conv_weights,
        attn_metadata=metadata,
    )

    ordinary_conv = conv_seed.clone()
    ordinary_ssm = ssm_seed.clone()
    ordinary_output = torch.empty_like(tree_output)
    for token_indices, parents, children in zip(
        token_depths, parent_depths, child_depths, strict=True
    ):
        parents_long = parents.long()
        children_long = children.long()
        different = parents_long != children_long
        if bool(different.any()):
            selected_children = children_long[different]
            ordinary_conv.index_copy_(
                0,
                selected_children,
                ordinary_conv.index_select(0, parents_long[different]).clone(),
            )
            ordinary_ssm.index_copy_(
                0,
                selected_children,
                ordinary_ssm.index_select(0, parents_long[different]).clone(),
            )
        depth_mixed = causal_conv1d_update(
            mixed_qkv.index_select(0, token_indices).clone(),
            ordinary_conv,
            conv_weights,
            conv_bias,
            "silu",
            conv_state_indices=children,
            validate_data=False,
        )
        depth_output = torch.empty(
            token_indices.numel(),
            1,
            num_value_heads,
            value_dim,
            dtype=dtype,
            device=device,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=depth_mixed,
            a=a.index_select(0, token_indices),
            b=b.index_select(0, token_indices),
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=ordinary_ssm,
            out=depth_output,
            ssm_state_indices=children,
            use_qk_l2norm_in_kernel=True,
        )
        ordinary_output.index_copy_(
            0, token_indices, depth_output.squeeze(1)
        )

    assert torch.equal(tree_output, ordinary_output)
    assert torch.equal(tree_conv, ordinary_conv)
    assert torch.equal(tree_ssm, ordinary_ssm)

    # Commit the accepted local path [2, 4] by moving local-4 slot 5 into
    # canonical local-0 slot 1, then verify the next ordinary root state.
    tree_conv[1].copy_(tree_conv[5])
    tree_ssm[1].copy_(tree_ssm[5])
    ordinary_conv[1].copy_(ordinary_conv[5])
    ordinary_ssm[1].copy_(ordinary_ssm[5])
    next_mixed = torch.randn(1, packed_dim, dtype=dtype, device=device).mul_(0.1)
    next_a = torch.randn(1, num_value_heads, dtype=dtype, device=device).mul_(0.1)
    next_b = torch.randn_like(next_a).mul_(0.1)

    def next_root(conv_state, ssm_state):
        transformed = causal_conv1d_update(
            next_mixed.clone(),
            conv_state,
            conv_weights,
            conv_bias,
            "silu",
            conv_state_indices=child_depths[0],
            validate_data=False,
        )
        output = torch.empty(
            1, 1, num_value_heads, value_dim, dtype=dtype, device=device
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=transformed,
            a=next_a,
            b=next_b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=ssm_state,
            out=output,
            ssm_state_indices=child_depths[0],
            use_qk_l2norm_in_kernel=True,
        )
        return output

    assert torch.equal(
        next_root(tree_conv, tree_ssm),
        next_root(ordinary_conv, ordinary_ssm),
    )
    assert torch.equal(tree_conv, ordinary_conv)
    assert torch.equal(tree_ssm, ordinary_ssm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_tree_source_destination_updates_are_bitwise_and_order_invariant():
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 4
    num_states = 10

    parents = torch.tensor([1, 1, 2, 2], dtype=torch.long, device=device)
    children = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    state_pairs = torch.stack((parents, children), dim=1).to(torch.int32)
    source_columns = torch.zeros(batch, dtype=torch.int32, device=device)
    destination_columns = torch.ones(batch, dtype=torch.int32, device=device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)

    conv_dim = 256
    conv_width = 4
    conv_input = torch.randn(batch, conv_dim, device=device, dtype=dtype)
    conv_weight = torch.randn(conv_dim, conv_width, device=device, dtype=dtype)
    conv_bias = torch.randn(conv_dim, device=device, dtype=dtype)
    conv_state_seed = torch.randn(
        num_states,
        conv_dim,
        conv_width - 1,
        device=device,
        dtype=dtype,
    )

    conv_state_reference = conv_state_seed.clone()
    conv_state_reference.index_copy_(
        0,
        children,
        conv_state_reference.index_select(0, parents).clone(),
    )
    conv_output_reference = causal_conv1d_update(
        conv_input.clone(),
        conv_state_reference,
        conv_weight,
        conv_bias,
        "silu",
        conv_state_indices=children,
        validate_data=False,
    )

    def direct_conv(order: torch.Tensor):
        state = conv_state_seed.clone()
        output = causal_conv1d_update(
            conv_input.index_select(0, order).clone(),
            state,
            conv_weight,
            conv_bias,
            "silu",
            conv_state_indices=state_pairs.index_select(0, order),
            block_idx_last_scheduled_token=(destination_columns.index_select(0, order)),
            initial_state_idx=source_columns.index_select(0, order),
            validate_data=False,
        )
        return output, state

    identity = torch.arange(batch, device=device)
    conv_output, conv_state = direct_conv(identity)
    conv_output_permuted, conv_state_permuted = direct_conv(permutation)
    assert torch.equal(conv_output, conv_output_reference)
    assert torch.equal(conv_state, conv_state_reference)
    assert torch.equal(
        conv_output_permuted,
        conv_output_reference.index_select(0, permutation),
    )
    assert torch.equal(conv_state_permuted, conv_state_reference)

    num_query_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16
    packed_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = torch.randn(batch, packed_dim, device=device, dtype=dtype) * 0.1
    a = torch.randn(batch, num_value_heads, device=device, dtype=dtype) * 0.1
    b = torch.randn(batch, num_value_heads, device=device, dtype=dtype) * 0.1
    A_log = torch.zeros(num_value_heads, device=device, dtype=dtype)
    dt_bias = torch.zeros(num_value_heads, device=device, dtype=dtype)
    ssm_state_seed = (
        torch.randn(
            num_states,
            num_value_heads,
            value_dim,
            key_dim,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )

    ssm_state_reference = ssm_state_seed.clone()
    ssm_state_reference.index_copy_(
        0,
        children,
        ssm_state_reference.index_select(0, parents).clone(),
    )
    recurrent_output_reference = torch.empty(
        batch, 1, num_value_heads, value_dim, device=device, dtype=dtype
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=ssm_state_reference,
        out=recurrent_output_reference,
        ssm_state_indices=children,
        use_qk_l2norm_in_kernel=True,
    )

    def direct_recurrent(order: torch.Tensor):
        state = ssm_state_seed.clone()
        output = torch.empty(
            batch, 1, num_value_heads, value_dim, device=device, dtype=dtype
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv.index_select(0, order).contiguous(),
            a=a.index_select(0, order).contiguous(),
            b=b.index_select(0, order).contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=state,
            out=output,
            ssm_state_indices=parents.index_select(0, order),
            final_state_indices=children.index_select(0, order),
            use_qk_l2norm_in_kernel=True,
        )
        return output, state

    recurrent_output, ssm_state = direct_recurrent(identity)
    recurrent_output_permuted, ssm_state_permuted = direct_recurrent(permutation)
    assert torch.equal(recurrent_output, recurrent_output_reference)
    assert torch.equal(ssm_state, ssm_state_reference)
    assert torch.equal(
        recurrent_output_permuted,
        recurrent_output_reference.index_select(0, permutation),
    )
    assert torch.equal(ssm_state_permuted, ssm_state_reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_stock_k1_first_conv_token_matches_tree_root_bitwise():
    """Stock K1's first convolution token must match a tree root update."""

    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.5-27B GDN dimensions on one tensor-parallel rank.
    conv_dim = 2 * 16 * 128 + 48 * 128
    conv_width = 4
    num_states = 3
    state_index = torch.tensor([1], device=device, dtype=torch.int32)

    mixed_qkv = torch.randn(2, conv_dim, device=device, dtype=dtype).mul_(0.1)
    conv_weight = torch.randn(conv_dim, conv_width, device=device, dtype=dtype).mul_(
        0.1
    )

    # Stock K1 reserves one extra cache column for its draft token. B2D1
    # reserves two, but both paths begin with the same three-token history.
    history = torch.randn(
        num_states,
        conv_dim,
        conv_width - 1,
        device=device,
        dtype=dtype,
    ).mul_(0.1)
    stock_state = torch.cat(
        (
            history.clone(),
            torch.randn(num_states, conv_dim, 1, device=device, dtype=dtype).mul_(0.1),
        ),
        dim=-1,
    )
    tree_state = torch.cat(
        (
            history.clone(),
            torch.randn(num_states, conv_dim, 2, device=device, dtype=dtype).mul_(0.1),
        ),
        dim=-1,
    )
    tree_padding_before = tree_state[..., conv_width - 1 :].clone()

    stock_output = causal_conv1d_update(
        mixed_qkv.clone(),
        stock_state,
        conv_weight,
        None,
        "silu",
        conv_state_indices=state_index,
        num_accepted_tokens=torch.ones(1, device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], device=device, dtype=torch.int32),
        max_query_len=2,
        validate_data=False,
    )
    tree_output = causal_conv1d_update(
        mixed_qkv[:1].clone(),
        tree_state,
        conv_weight,
        None,
        "silu",
        conv_state_indices=state_index,
        validate_data=False,
    )

    assert torch.equal(stock_output[:1], tree_output)
    assert torch.equal(
        stock_state[1, :, : conv_width - 1],
        tree_state[1, :, : conv_width - 1],
    )
    assert torch.equal(tree_state[..., conv_width - 1 :], tree_padding_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_packed_equivalent_sigmoid_t1_matches_no_spec_packed_bitwise():
    """One fused verifier row must reproduce ordinary no-spec decode."""

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.5-27B GDN dimensions on one tensor-parallel rank.
    num_query_heads = 16
    num_value_heads = 48
    key_dim = 128
    value_dim = 128
    packed_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = torch.randn(2, packed_dim, device=device, dtype=dtype).mul_(0.1)
    a = torch.randn(2, num_value_heads, device=device, dtype=dtype).mul_(0.1)
    b = torch.randn(2, num_value_heads, device=device, dtype=dtype).mul_(0.1)
    A_log = torch.randn(num_value_heads, device=device, dtype=torch.float32).mul_(0.1)
    dt_bias = torch.randn(num_value_heads, device=device, dtype=dtype).mul_(0.1)
    state_seed = torch.randn(
        4,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    ).mul_(0.1)

    query = mixed_qkv[:, : num_query_heads * key_dim].view(
        1, 2, num_query_heads, key_dim
    )
    key = mixed_qkv[:, num_query_heads * key_dim : 2 * num_query_heads * key_dim].view(
        1, 2, num_query_heads, key_dim
    )
    value = mixed_qkv[:, 2 * num_query_heads * key_dim :].view(
        1, 2, num_value_heads, value_dim
    )

    exact_state = state_seed.clone()
    exact_output, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a[:1],
        b=b[:1],
        dt_bias=dt_bias,
        q=query[:, :1],
        k=key[:, :1],
        v=value[:, :1],
        initial_state=exact_state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1], device=device, dtype=torch.int32),
        ssm_state_indices=torch.tensor([1], device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        packed_decode_equivalent=True,
    )
    packed_state = state_seed.clone()
    packed_output = torch.empty(
        1,
        1,
        num_value_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv[:1],
        a=a[:1],
        b=b[:1],
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=packed_state,
        out=packed_output,
        ssm_state_indices=torch.tensor([1], device=device, dtype=torch.long),
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.equal(exact_output, packed_output)
    assert torch.equal(exact_state, packed_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_packed_equivalent_sigmoid_t2_matches_two_no_spec_rounds_bitwise():
    """Fused Stock K1 verification keeps the no-spec BF16 state boundary."""

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 16
    num_value_heads = 48
    key_dim = 128
    value_dim = 128
    packed_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = torch.randn(2, packed_dim, device=device, dtype=dtype).mul_(0.1)
    a = torch.randn(2, num_value_heads, device=device, dtype=dtype).mul_(0.1)
    b = torch.randn(2, num_value_heads, device=device, dtype=dtype).mul_(0.1)
    A_log = torch.randn(num_value_heads, device=device, dtype=torch.float32).mul_(0.1)
    dt_bias = torch.randn(num_value_heads, device=device, dtype=dtype).mul_(0.1)
    state_seed = torch.randn(
        4,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    ).mul_(0.1)

    query, key, value = torch.split(
        mixed_qkv,
        [
            num_query_heads * key_dim,
            num_query_heads * key_dim,
            num_value_heads * value_dim,
        ],
        dim=-1,
    )
    query = query.view(1, 2, num_query_heads, key_dim).contiguous()
    key = key.view(1, 2, num_query_heads, key_dim).contiguous()
    value = value.view(1, 2, num_value_heads, value_dim).contiguous()

    fused_state = state_seed.clone()
    fused_output, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=fused_state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 2], device=device, dtype=torch.int32),
        ssm_state_indices=torch.tensor([[1, 2]], device=device, dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        packed_decode_equivalent=True,
    )

    sequential_state = state_seed.clone()
    sequential_outputs = []
    for token_index, (source, destination) in enumerate(((1, 1), (1, 2))):
        output = torch.empty(
            1,
            1,
            num_value_heads,
            value_dim,
            device=device,
            dtype=dtype,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[token_index : token_index + 1],
            a=a[token_index : token_index + 1],
            b=b[token_index : token_index + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=sequential_state,
            out=output,
            ssm_state_indices=torch.tensor(
                [source], device=device, dtype=torch.long
            ),
            final_state_indices=torch.tensor(
                [destination], device=device, dtype=torch.long
            ),
            use_qk_l2norm_in_kernel=True,
        )
        sequential_outputs.append(output)
    sequential_output = torch.cat(sequential_outputs, dim=1)

    assert torch.equal(fused_output, sequential_output)
    assert torch.equal(fused_state, sequential_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_sigmoid_recurrent_source_destination_is_bitwise_across_rounds():
    """Tree state routing must not change Stock's one-token arithmetic."""

    torch.manual_seed(19)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.5-27B GDN dimensions on one tensor-parallel rank.
    num_query_heads = 16
    num_value_heads = 48
    key_dim = 128
    value_dim = 128
    qkv_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    A_log = torch.randn(num_value_heads, device=device, dtype=dtype).mul_(0.1)
    dt_bias = torch.randn(num_value_heads, device=device, dtype=dtype).mul_(0.1)
    state_seed = torch.randn(
        5,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    ).mul_(0.1)

    def inputs(num_rows: int):
        mixed_qkv = torch.randn(num_rows, qkv_dim, device=device, dtype=dtype).mul_(0.1)
        a = torch.randn(num_rows, num_value_heads, device=device, dtype=dtype).mul_(0.1)
        b = torch.randn(num_rows, num_value_heads, device=device, dtype=dtype).mul_(0.1)
        query, key, value = torch.split(
            mixed_qkv,
            [
                num_query_heads * key_dim,
                num_query_heads * key_dim,
                num_value_heads * value_dim,
            ],
            dim=-1,
        )
        return (
            query.view(1, num_rows, num_query_heads, key_dim).contiguous(),
            key.view(1, num_rows, num_query_heads, key_dim).contiguous(),
            value.view(1, num_rows, num_value_heads, value_dim).contiguous(),
            a,
            b,
        )

    def update(
        state: torch.Tensor,
        source: torch.Tensor,
        final: torch.Tensor | None,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        num_rows = int(source.numel())
        output, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=torch.arange(num_rows + 1, device=device, dtype=torch.int32),
            ssm_state_indices=source,
            final_state_indices=final,
            use_qk_l2norm_in_kernel=True,
            packed_decode_equivalent=True,
        )
        return output

    root_inputs = inputs(1)
    root_index = torch.tensor([1], device=device, dtype=torch.int32)
    default_state = state_seed.clone()
    default_output = update(default_state, root_index, None, *root_inputs)
    explicit_state = state_seed.clone()
    explicit_output = update(
        explicit_state, root_index, root_index.clone(), *root_inputs
    )
    assert torch.equal(explicit_output, default_output)
    assert torch.equal(explicit_state, default_state)

    # Two siblings read the committed root state but write distinct slots.
    child_inputs = inputs(2)
    parents = torch.tensor([1, 1], device=device, dtype=torch.int32)
    children = torch.tensor([2, 3], device=device, dtype=torch.int32)
    parent_after_root = default_state[1].clone()

    routed_state = default_state.clone()
    routed_output = update(routed_state, parents, children, *child_inputs)

    child_query, child_key, child_value, child_a, child_b = child_inputs
    child_mixed_qkv = torch.cat(
        (
            child_query.squeeze(0).reshape(2, -1),
            child_key.squeeze(0).reshape(2, -1),
            child_value.squeeze(0).reshape(2, -1),
        ),
        dim=-1,
    ).contiguous()
    packed_state = default_state.clone()
    packed_output = torch.empty(
        2,
        1,
        num_value_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=child_mixed_qkv,
        a=child_a,
        b=child_b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=packed_state,
        out=packed_output,
        ssm_state_indices=parents.long(),
        final_state_indices=children.long(),
        use_qk_l2norm_in_kernel=True,
    )
    assert torch.equal(routed_output, packed_output.transpose(0, 1))
    assert torch.equal(routed_state, packed_state)

    copied_state = default_state.clone()
    copied_state.index_copy_(
        0,
        children.long(),
        copied_state.index_select(0, parents.long()).clone(),
    )
    copied_output = update(copied_state, children, None, *child_inputs)

    assert torch.equal(routed_output, copied_output)
    assert torch.equal(routed_state, copied_state)
    assert torch.equal(routed_state[1], parent_after_root)

    # The next verifier root is identical whether it commits the root state
    # (rejection) or either child state (acceptance).
    next_inputs = inputs(1)
    for committed_index in (1, 2, 3):
        source = torch.tensor([committed_index], device=device, dtype=torch.int32)
        routed_next = routed_state.clone()
        copied_next = copied_state.clone()
        routed_next_output = update(routed_next, source, None, *next_inputs)
        copied_next_output = update(copied_next, source, source.clone(), *next_inputs)
        assert torch.equal(routed_next_output, copied_next_output)
        assert torch.equal(routed_next, copied_next)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_packed_recurrent_explicit_same_destination_matches_default_bitwise():
    """The tree root's explicit destination must match stock decode exactly."""

    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.5-27B GDN dimensions on one tensor-parallel rank.
    num_query_heads = 16
    num_value_heads = 48
    key_dim = 128
    value_dim = 128
    packed_dim = 2 * num_query_heads * key_dim + num_value_heads * value_dim
    mixed_qkv = torch.randn(1, packed_dim, device=device, dtype=dtype).mul_(0.1)
    a = torch.randn(1, num_value_heads, device=device, dtype=dtype).mul_(0.1)
    b = torch.randn(1, num_value_heads, device=device, dtype=dtype).mul_(0.1)
    A_log = torch.zeros(num_value_heads, device=device, dtype=dtype)
    dt_bias = torch.zeros(num_value_heads, device=device, dtype=dtype)
    state_seed = torch.randn(
        3,
        num_value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    ).mul_(0.1)
    state_indices = torch.tensor([1], device=device, dtype=torch.long)

    default_state = state_seed.clone()
    default_output = torch.empty(
        1,
        1,
        num_value_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=default_state,
        out=default_output,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    explicit_state = state_seed.clone()
    explicit_output = torch.empty_like(default_output)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=explicit_state,
        out=explicit_output,
        ssm_state_indices=state_indices,
        final_state_indices=state_indices.clone(),
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.equal(explicit_output, default_output)
    assert torch.equal(explicit_state, default_state)
