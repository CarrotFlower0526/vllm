# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qwen_gdn
from vllm.model_executor.models.eagle_residual import ResidualTreeHeadMixin
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    _ResidualTreeDraftState,
    _ResidualTreeKVPayload,
)
from vllm.v1.spec_decode.residual_tree import (
    ResidualTree,
    ResidualTreeNode,
    TreeCandidateContributor,
    TreeListwiseFeatures,
    _resolve_active_heads,
    build_tree_attention_mask,
    estimate_residual_head_lambdas,
    select_batched_b2d1_residual_trees,
    select_batched_greedy_residual_trees,
    select_residual_tree,
    tree_position_offsets,
    tree_to_draft_token_tree,
    trees_to_metadata,
    verify_greedy_tree,
    verify_greedy_tree_batch_with_nodes,
    verify_stochastic_tree,
    verify_stochastic_tree_batch,
)
from vllm.v1.spec_decode.tree_schema import DraftTokenTree
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_estimate_residual_head_lambdas_uses_overlap_and_clipped_update():
    target = torch.tensor([[0.6, 0.4, 0.0]], dtype=torch.float32)
    proposals = torch.tensor(
        [[[0.5, 0.25, 0.25], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )

    lambdas = estimate_residual_head_lambdas(target, proposals)

    assert torch.allclose(lambdas, torch.tensor([0.75, 0.15]), atol=1e-6)


def test_residual_tree_adapters_use_model_dtype_not_first_parameter(tmp_path):
    config_path = tmp_path / "shared_stack_config.json"
    checkpoint_path = tmp_path / "residual_adapters.pt"
    config_path.write_text(
        """{
  "method": "vllm_eagle3_shared_trunk_residual_heads",
  "hidden_size": 4,
  "adapter_bottleneck": 2,
  "num_residual_adapters": 1,
  "num_layers": 1,
  "freeze_base_head": false
}""",
        encoding="utf-8",
    )
    torch.save(
        {
            "0.0.weight": torch.ones(2, 4),
            "0.2.weight": torch.ones(4, 2),
        },
        checkpoint_path,
    )

    class DummyResidualModel(ResidualTreeHeadMixin, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(hidden_size=4)
            self.long_param = nn.Parameter(
                torch.zeros(1, dtype=torch.long), requires_grad=False
            )

    spec_config = SimpleNamespace(
        residual_tree_adapter_config=str(config_path),
        residual_tree_adapter_checkpoint=str(checkpoint_path),
        draft_model_config=SimpleNamespace(dtype=torch.bfloat16),
        residual_tree_head_lambdas=[1.0],
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    model = DummyResidualModel()

    model._init_residual_tree_heads(vllm_config)

    assert model.residual_tree_adapters[0][0].weight.dtype == torch.bfloat16


def test_residual_tree_loads_and_maps_independent_lm_head(tmp_path):
    config_path = tmp_path / "shared_stack_config.json"
    checkpoint_path = tmp_path / "residual_adapters.pt"
    config_path.write_text(
        """{
  "method": "vllm_eagle3_shared_trunk_residual_heads",
  "hidden_size": 2,
  "adapter_bottleneck": 2,
  "adapter_output_mode": "independent_lm_head",
  "draft_vocab_size": 3,
  "num_residual_adapters": 1,
  "num_layers": 1,
  "freeze_base_head": false,
  "serving_proposal_mode": "condition_on_prior_selected_tokens"
}""",
        encoding="utf-8",
    )
    weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=torch.float32,
    )
    torch.save({"0.weight": weight}, checkpoint_path)

    class DummyResidualModel(ResidualTreeHeadMixin, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                hidden_size=2,
                draft_vocab_size=3,
                vocab_size=5,
            )
            self.draft_id_to_target_id = nn.Parameter(
                torch.tensor([1, 1, 2], dtype=torch.long),
                requires_grad=False,
            )

    spec_config = SimpleNamespace(
        residual_tree_adapter_config=str(config_path),
        residual_tree_adapter_checkpoint=str(checkpoint_path),
        draft_model_config=SimpleNamespace(dtype=torch.float32),
        residual_tree_head_lambdas=[1.0],
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    model = DummyResidualModel()

    model._init_residual_tree_heads(vllm_config)
    logits = model.compute_residual_head_logits(torch.tensor([[2.0, 1.0]]))

    assert isinstance(model.residual_tree_adapters[0], nn.Linear)
    assert logits.shape == (1, 1, 5)
    assert torch.allclose(logits[0, 0, [1, 2, 4]], torch.tensor([2.0, 1.0, 1.0]))
    assert torch.isneginf(logits[0, 0, [0, 3]]).all()
    assert model.residual_tree_required_candidate_selection == "distinct_head_top1"


def test_residual_tree_loads_identity_low_rank_logit_residual(tmp_path):
    config_path = tmp_path / "shared_stack_config.json"
    checkpoint_path = tmp_path / "residual_adapters.pt"
    config_path.write_text(
        """{
  "method": "vllm_eagle3_shared_trunk_residual_heads",
  "hidden_size": 2,
  "adapter_bottleneck": 1,
  "adapter_depth": 2,
  "adapter_output_mode": "logit_residual",
  "adapter_activation": "identity",
  "draft_vocab_size": 3,
  "num_residual_adapters": 1,
  "num_layers": 2,
  "freeze_base_head": true,
  "serving_proposal_mode": "condition_on_prior_selected_tokens"
}""",
        encoding="utf-8",
    )
    low_rank_in = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    low_rank_out = torch.tensor([[1.0], [3.0], [-2.0]], dtype=torch.float32)
    torch.save(
        {
            "0.0.weight": low_rank_in,
            "0.2.weight": low_rank_out,
        },
        checkpoint_path,
    )

    class DummyResidualModel(ResidualTreeHeadMixin, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                hidden_size=2,
                draft_vocab_size=3,
                vocab_size=3,
            )
            self.lm_head = nn.Linear(2, 3, bias=False)
            self.logits_processor = lambda head, hidden: head(hidden)
            with torch.no_grad():
                self.lm_head.weight.copy_(
                    torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
                )

        def compute_logits(self, hidden):
            return self.lm_head(hidden)

    spec_config = SimpleNamespace(
        residual_tree_adapter_config=str(config_path),
        residual_tree_adapter_checkpoint=str(checkpoint_path),
        residual_tree_candidate_selection="hybrid_union_top10_dynamic",
        draft_model_config=SimpleNamespace(dtype=torch.float32),
        residual_tree_head_lambdas=[0.4, 0.4],
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    model = DummyResidualModel()

    model._init_residual_tree_heads(vllm_config)
    hidden = torch.tensor([[2.0, 1.0]])
    logits = model.compute_residual_head_logits(hidden)
    base = model.lm_head(hidden)
    expected_delta = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden, low_rank_in),
        low_rank_out,
    )

    assert isinstance(model.residual_tree_adapters[0][1], nn.Identity)
    assert logits.shape == (1, 2, 3)
    assert torch.equal(logits[:, 0], base)
    assert torch.equal(logits[:, 1], base + expected_delta)


def test_residual_tree_loads_one_shared_vocabulary_projection(tmp_path):
    config_path = tmp_path / "shared_stack_config.json"
    checkpoint_path = tmp_path / "residual_adapters.pt"
    config_path.write_text(
        """{
  "method": "vllm_eagle3_shared_trunk_residual_heads",
  "hidden_size": 2,
  "adapter_bottleneck": 1,
  "adapter_depth": 2,
  "adapter_output_mode": "logit_residual",
  "adapter_activation": "identity",
  "adapter_vocab_projection": "shared",
  "draft_vocab_size": 4,
  "num_residual_adapters": 3,
  "num_layers": 4,
  "freeze_base_head": true,
  "serving_proposal_mode": "condition_on_prior_selected_tokens"
}""",
        encoding="utf-8",
    )
    inputs = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
    ]
    shared_out = torch.tensor([[1.0], [2.0], [-1.0], [0.5]])
    torch.save(
        {
            **{f"{index}.0.weight": value for index, value in enumerate(inputs)},
            "shared_out.weight": shared_out,
        },
        checkpoint_path,
    )

    class DummyResidualModel(ResidualTreeHeadMixin, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                hidden_size=2,
                draft_vocab_size=4,
                vocab_size=4,
            )
            self.lm_head = nn.Linear(2, 4, bias=False)
            self.logits_processor = lambda head, hidden: head(hidden)
            with torch.no_grad():
                self.lm_head.weight.zero_()

        def compute_logits(self, hidden):
            return self.lm_head(hidden)

    spec_config = SimpleNamespace(
        residual_tree_adapter_config=str(config_path),
        residual_tree_adapter_checkpoint=str(checkpoint_path),
        residual_tree_candidate_selection="distinct_head_top1",
        draft_model_config=SimpleNamespace(dtype=torch.float32),
        residual_tree_head_lambdas=[1.0] * 4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    model = DummyResidualModel()

    model._init_residual_tree_heads(vllm_config)
    hidden = torch.tensor([[2.0, 3.0]])
    logits = model.compute_residual_head_logits(hidden)
    expected_scales = torch.tensor([2.0, 3.0, 5.0])

    assert logits.shape == (1, 4, 4)
    assert torch.equal(logits[0, 1:], expected_scales[:, None] * shared_out.T)
    assert (
        model.residual_tree_adapters[0][2].weight.data_ptr()
        == model.residual_tree_adapters[2][2].weight.data_ptr()
    )
    assert model._residual_tree_packed_logit_out_weight is None
    assert torch.equal(model._residual_tree_shared_logit_out_weight, shared_out)


@pytest.mark.parametrize(
    "candidate_selection",
    ["hybrid_top9_h2_dynamic", "hybrid_union_top10_dynamic"],
)
def test_residual_tree_loads_hybrid_tree_depth_bias(
    tmp_path,
    candidate_selection: str,
):
    config_path = tmp_path / "shared_stack_config.json"
    checkpoint_path = tmp_path / "residual_adapters.pt"
    config_path.write_text(
        """{
  "method": "vllm_eagle3_shared_trunk_residual_heads",
  "hidden_size": 2,
  "adapter_bottleneck": 2,
  "adapter_output_mode": "independent_lm_head",
  "draft_vocab_size": 12,
  "num_residual_adapters": 1,
  "num_layers": 2,
  "freeze_base_head": true,
  "serving_proposal_mode": "condition_on_prior_selected_tokens",
  "h2_tree_depth_bias_mode": "per_dynamic_tree_depth"
}""",
        encoding="utf-8",
    )
    weight = torch.zeros((12, 2), dtype=torch.bfloat16)
    depth_bias = torch.arange(8 * 12, dtype=torch.bfloat16).reshape(8, 12)
    torch.save(
        {"0.weight": weight, "tree_depth_bias": depth_bias},
        checkpoint_path,
    )

    class DummyResidualModel(ResidualTreeHeadMixin, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                hidden_size=2,
                draft_vocab_size=12,
                vocab_size=12,
            )

    spec_config = SimpleNamespace(
        residual_tree_adapter_config=str(config_path),
        residual_tree_adapter_checkpoint=str(checkpoint_path),
        residual_tree_candidate_selection=candidate_selection,
        draft_model_config=SimpleNamespace(dtype=torch.bfloat16),
        residual_tree_head_lambdas=[0.4, 0.2],
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    model = DummyResidualModel()

    model._init_residual_tree_heads(vllm_config)

    assert torch.equal(model.residual_tree_h2_tree_depth_bias.cpu(), depth_bias)
    assert "tree_depth_bias" not in model.residual_tree_adapters.state_dict()
    assert model.residual_tree_required_candidate_selection == candidate_selection


def test_select_residual_tree_merges_duplicates_and_respects_budget():
    proposal_calls = []
    transition_calls = []

    def proposal_fn(state):
        proposal_calls.append(state)
        if state == ():
            return [
                torch.tensor([0.0, 0.6, 0.3, 0.1, 0.0]),
                torch.tensor([0.0, 0.5, 0.1, 0.4, 0.0]),
            ]
        if state == (1,):
            return [
                torch.tensor([0.0, 0.0, 0.0, 0.1, 0.9]),
                torch.tensor([0.0, 0.0, 0.2, 0.0, 0.8]),
            ]
        return [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        ]

    def transition_fn(state, token_id):
        transition_calls.append((state, token_id))
        return (*state, token_id)

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=transition_fn,
        head_lambdas=torch.tensor([1.0, 1.0]),
        node_budget=2,
    )

    assert tree.num_draft_nodes == 2
    root_child = tree.children[0][0]
    assert tree.nodes[root_child].token_id == 1
    assert [item.head_id for item in tree.nodes[root_child].contributors] == [0, 1]
    assert tree.nodes[2].parent_id == root_child
    assert tree.nodes[2].token_id == 4
    assert proposal_calls == [(), (1,)]
    assert transition_calls == [((), 1)]
    assert tree.states == [(), (1,), None]


def test_select_residual_tree_can_mask_tokens_from_earlier_ordered_heads():
    def proposal_fn(state):
        assert state == ()
        return [
            torch.tensor([0.0, 0.6, 0.3, 0.1]),
            torch.tensor([0.0, 0.5, 0.1, 0.4]),
        ]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.tensor([1.0, 1.0]),
        node_budget=2,
        max_depth=1,
        candidate_selection="distinct_head_top1",
    )

    assert tree.num_draft_nodes == 2
    assert {tree.nodes[node_id].token_id for node_id in tree.children[0]} == {1, 3}
    assert all(
        len(tree.nodes[node_id].contributors) == 1 for node_id in tree.children[0]
    )
    assert tree.proposal_probs is not None
    assert torch.allclose(
        tree.proposal_probs,
        torch.tensor([[0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 0.2, 0.8]]),
    )
    head2_child = next(
        tree.nodes[node_id]
        for node_id in tree.children[0]
        if tree.nodes[node_id].token_id == 3
    )
    assert head2_child.contributors[0].proposal_prob == pytest.approx(0.8)
    assert head2_child.contributors[0].score == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("active_heads", "error_type", "message"),
    [
        ([], ValueError, "must not be empty"),
        ([0, 0], ValueError, "strictly increasing and unique"),
        ([-1], ValueError, "non-negative"),
        ([1, 0], ValueError, "strictly increasing and unique"),
        ([0, 3], ValueError, "outside the available range"),
        ([0, 1.0], TypeError, "only integers"),
        ([False], TypeError, "only integers"),
    ],
)
def test_select_residual_tree_rejects_invalid_active_heads(
    active_heads,
    error_type,
    message,
):
    proposals = [
        torch.tensor([0.7, 0.2, 0.1]),
        torch.tensor([0.1, 0.7, 0.2]),
        torch.tensor([0.2, 0.1, 0.7]),
    ]

    with pytest.raises(error_type, match=message):
        select_residual_tree(
            root_state=(),
            proposal_fn=lambda _state: proposals,
            transition_fn=lambda state, token_id: (*state, token_id),
            head_lambdas=torch.ones(3),
            active_heads=active_heads,
            node_budget=2,
            max_depth=1,
            candidate_selection="distinct_head_top1",
        )


def test_select_residual_tree_accepts_ordered_active_head_subset():
    proposals = [
        torch.tensor([0.7, 0.2, 0.1]),
        torch.tensor([0.1, 0.7, 0.2]),
        torch.tensor([0.2, 0.1, 0.7]),
    ]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=lambda _state: proposals,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        active_heads=[0, 2],
        node_budget=2,
        max_depth=1,
        candidate_selection="distinct_head_top1",
    )

    assert tree.num_draft_nodes == 2
    assert sorted(
        tree.nodes[node_id].contributors[0].head_id for node_id in tree.children[0]
    ) == [0, 2]


def test_active_head_boolean_tensor_is_a_flat_mask():
    assert _resolve_active_heads(
        torch.tensor([[False, True, False]]),
        3,
    ) == [1]
    with pytest.raises(ValueError, match="one value per head"):
        _resolve_active_heads(torch.tensor([True, False]), 3)
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_active_heads(torch.zeros(3, dtype=torch.bool), 3)


def test_select_residual_tree_can_gate_novel_tokens_by_probability_ratio():
    def proposal_fn(state):
        assert state == ()
        return [
            torch.tensor([0.7, 0.2, 0.1]),
            torch.tensor([0.8, 0.15, 0.05]),
            torch.tensor([0.6, 0.3, 0.1]),
        ]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        node_budget=3,
        max_depth=1,
        candidate_selection="gated_distinct_head_top1",
        min_novel_probability_ratio=0.2,
    )

    assert tree.num_draft_nodes == 2
    assert {tree.nodes[node_id].token_id for node_id in tree.children[0]} == {0, 1}
    repeated = next(
        tree.nodes[node_id]
        for node_id in tree.children[0]
        if tree.nodes[node_id].token_id == 0
    )
    assert {item.head_id for item in repeated.contributors} == {0, 1}
    assert tree.proposal_probs is not None
    assert torch.allclose(
        tree.proposal_probs,
        torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.8, 0.15, 0.05],
                [0.0, 0.75, 0.25],
            ]
        ),
    )


def test_select_residual_tree_stops_expansion_at_max_depth():
    proposal_calls = []
    transition_calls = []

    def proposal_fn(state):
        proposal_calls.append(state)
        if state == ():
            return [
                torch.tensor([0.6, 0.3, 0.1]),
                torch.tensor([0.1, 0.6, 0.3]),
                torch.tensor([0.1, 0.3, 0.6]),
            ]
        return [torch.tensor([1.0, 0.0, 0.0])] * 3

    def transition_fn(state, token_id):
        transition_calls.append((state, token_id))
        return (*state, token_id)

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=transition_fn,
        head_lambdas=torch.tensor([1.0, 1.0, 1.0]),
        node_budget=4,
        max_depth=1,
    )

    assert tree.num_draft_nodes == 3
    assert tree.max_depth == 1
    assert proposal_calls == [()]
    assert transition_calls == []
    assert tree.states == [(), None, None, None]


@pytest.mark.parametrize("tree_policy", ["best_first", "breadth_first"])
def test_select_residual_tree_consumes_precomputed_root_heads(tree_policy):
    root_head_probs = [
        torch.tensor([0.7, 0.2, 0.1]),
        torch.tensor([0.6, 0.3, 0.1]),
    ]
    common = {
        "root_state": (),
        "transition_fn": lambda state, token_id: (*state, token_id),
        "head_lambdas": torch.ones(2),
        "node_budget": 2,
        "max_depth": 1,
        "candidate_selection": "distinct_head_top1",
        "tree_policy": tree_policy,
    }
    reference = select_residual_tree(
        proposal_fn=lambda _state: root_head_probs,
        **common,
    )

    def unexpected_proposal(_state):
        raise AssertionError("precomputed root must bypass proposal callbacks")

    actual = select_residual_tree(
        root_head_probs=root_head_probs,
        proposal_fn=unexpected_proposal,
        **common,
    )

    assert actual.nodes == reference.nodes
    assert actual.children == reference.children
    assert torch.equal(actual.proposal_probs, reference.proposal_probs)


@pytest.mark.parametrize("tree_policy", ["best_first", "breadth_first"])
def test_batched_b2d1_selector_matches_scalar_requests(tree_policy):
    root_states = [("request", index) for index in range(3)]
    root_head_probs = torch.tensor(
        [
            [
                [0.8, 0.1, 0.1, 0.0],
                [0.1, 0.2, 0.3, 0.4],
                [0.7, 0.2, 0.1, 0.0],
            ],
            [
                [0.1, 0.7, 0.1, 0.1],
                [0.4, 0.3, 0.2, 0.1],
                [0.2, 0.6, 0.1, 0.1],
            ],
            [
                [0.0, 0.0, 0.9, 0.1],
                [0.1, 0.2, 0.3, 0.4],
                [0.0, 0.0, 1.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    lambdas = torch.tensor([0.8, 0.1, 0.7])

    actual, flat_proposals = select_batched_b2d1_residual_trees(
        root_states=root_states,
        root_head_probs=root_head_probs,
        head_lambdas=lambdas,
        active_heads=[0, 2],
        tree_policy=tree_policy,
    )
    reference = [
        select_residual_tree(
            root_state=root_state,
            root_head_probs=root_head_probs[request_index],
            proposal_fn=lambda _state: pytest.fail(
                "precomputed roots must bypass proposal_fn"
            ),
            transition_fn=lambda _state, _token: pytest.fail(
                "D1 selection must not transition"
            ),
            head_lambdas=lambdas,
            node_budget=2,
            max_depth=1,
            active_heads=[0, 2],
            candidate_selection="distinct_head_top1",
            tree_policy=tree_policy,
        )
        for request_index, root_state in enumerate(root_states)
    ]

    assert flat_proposals.shape == (6, 4)
    for actual_tree, reference_tree in zip(actual, reference):
        assert actual_tree.nodes == reference_tree.nodes
        assert actual_tree.children == reference_tree.children
        assert actual_tree.states == reference_tree.states
        assert torch.allclose(
            actual_tree.proposal_probs,
            reference_tree.proposal_probs,
        )
    assert torch.allclose(
        flat_proposals,
        torch.cat([tree.proposal_probs for tree in reference], dim=0),
    )

    metadata, packed_proposals = trees_to_metadata(actual)
    assert torch.equal(packed_proposals, flat_proposals)
    contributor_rows = metadata.contributor_proposal_rows.tolist()
    for request_index in range(3):
        start = request_index * 2
        assert sorted(contributor_rows[start : start + 2]) == [start, start + 1]


def test_batched_b2d1_selector_preserves_policy_tie_order():
    # Equal lambda*q scores enqueue head 0 first. Best-first keeps that order;
    # breadth-first uses token id before enqueue order for an in-level tie.
    root_head_probs = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    common = {
        "root_states": [()],
        "root_head_probs": root_head_probs,
        "head_lambdas": torch.ones(2),
    }

    best_first, _ = select_batched_b2d1_residual_trees(
        **common,
        tree_policy="best_first",
    )
    breadth_first, _ = select_batched_b2d1_residual_trees(
        **common,
        tree_policy="breadth_first",
    )

    assert [node.token_id for node in best_first[0].nodes[1:]] == [1, 0]
    assert [node.token_id for node in breadth_first[0].nodes[1:]] == [0, 1]
    for tree in (best_first[0], breadth_first[0]):
        proposal_rows = [node.contributors[0].proposal_row for node in tree.nodes[1:]]
        assert sorted(proposal_rows) == [0, 1]


@pytest.mark.parametrize(
    ("root_shape", "num_states", "active_heads", "message"),
    [
        ((2, 2), 2, [0, 1], "shape"),
        ((2, 2, 3), 1, [0, 1], "match root states"),
        ((1, 3, 3), 1, [0, 1, 2], "exactly two active heads"),
    ],
)
def test_batched_b2d1_selector_validates_narrow_contract(
    root_shape,
    num_states,
    active_heads,
    message,
):
    with pytest.raises(ValueError, match=message):
        select_batched_b2d1_residual_trees(
            root_states=[()] * num_states,
            root_head_probs=torch.ones(root_shape),
            head_lambdas=torch.ones(3),
            active_heads=active_heads,
        )


@pytest.mark.parametrize("tree_policy", ["best_first", "breadth_first"])
@pytest.mark.parametrize(("node_budget", "max_depth"), [(1, 1), (3, 2), (6, 3)])
def test_batched_greedy_selector_matches_scalar_for_arbitrary_topologies(
    tree_policy,
    node_budget,
    max_depth,
):
    root_states = [(request_index, ()) for request_index in range(4)]
    transition_batch_sizes: list[int] = []
    candidate_batch_sizes: list[int] = []

    def proposal_rows(state):
        request_index, path = state
        seed = request_index * 17 + sum(
            (index + 1) * token_id for index, token_id in enumerate(path)
        )
        first_token = seed % 47 + 1
        second_token = (first_token + 7) % 47 + 1
        first_backup = (first_token + 19) % 47 + 1
        second_backup = (second_token + 23) % 47 + 1
        while first_backup in (first_token, second_token):
            first_backup = first_backup % 47 + 1
        while second_backup in (first_token, second_token):
            second_backup = second_backup % 47 + 1
        first = torch.zeros(64, dtype=torch.float32)
        second = torch.zeros(64, dtype=torch.float32)
        first[first_token] = 0.7
        first[first_backup] = 0.3
        second[second_token] = 0.6
        second[second_backup] = 0.4
        return [first, second]

    def compact_candidates(states):
        candidate_batch_sizes.append(len(states))
        tokens = []
        probabilities = []
        for state in states:
            rows = proposal_rows(state)
            first_probability, first_token = rows[0].max(dim=-1)
            conditioned_second = rows[1].clone()
            conditioned_second[first_token] = 0.0
            conditioned_second /= conditioned_second.sum()
            second_probability, second_token = conditioned_second.max(dim=-1)
            tokens.append(torch.stack((first_token, second_token)))
            probabilities.append(torch.stack((first_probability, second_probability)))
        return torch.stack(tokens), torch.stack(probabilities)

    def transition_batch(states, token_ids):
        transition_batch_sizes.append(len(states))
        return [
            (state[0], (*state[1], int(token_id)))
            for state, token_id in zip(states, token_ids, strict=True)
        ]

    root_tokens, root_probabilities = compact_candidates(root_states)
    actual = select_batched_greedy_residual_trees(
        root_states=root_states,
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=compact_candidates,
        transition_batch_fn=transition_batch,
        head_lambdas=[0.8, 0.4],
        node_budget=node_budget,
        max_depth=max_depth,
        tree_policy=tree_policy,
    )
    expected = [
        select_residual_tree(
            root_state=root_state,
            proposal_fn=proposal_rows,
            transition_fn=lambda state, token_id: (
                state[0],
                (*state[1], int(token_id)),
            ),
            head_lambdas=[0.8, 0.4],
            node_budget=node_budget,
            max_depth=max_depth,
            candidate_selection="distinct_head_top1",
            tree_policy=tree_policy,
            store_proposal_probs=False,
        )
        for root_state in root_states
    ]

    for actual_tree, expected_tree in zip(actual, expected, strict=True):
        assert [
            (node.parent_id, node.token_id, node.depth) for node in actual_tree.nodes
        ] == [
            (node.parent_id, node.token_id, node.depth) for node in expected_tree.nodes
        ]
        assert [node.priority for node in actual_tree.nodes] == pytest.approx(
            [node.priority for node in expected_tree.nodes]
        )
        for actual_node, expected_node in zip(
            actual_tree.nodes[1:], expected_tree.nodes[1:], strict=True
        ):
            assert [item.head_id for item in actual_node.contributors] == [
                item.head_id for item in expected_node.contributors
            ]
            assert all(
                item.proposal_row == -1 for item in actual_node.contributors
            )
            assert all(
                item.token_id == actual_node.token_id
                for item in actual_node.contributors
            )
            assert [
                item.proposal_prob for item in actual_node.contributors
            ] == pytest.approx(
                [item.proposal_prob for item in expected_node.contributors]
            )
            assert [item.score for item in actual_node.contributors] == pytest.approx(
                [item.score for item in expected_node.contributors]
            )
        assert actual_tree.proposal_probs is None

    if node_budget > 1 and max_depth > 1:
        assert max(transition_batch_sizes) >= len(root_states)
        assert max(candidate_batch_sizes) >= len(root_states)


@pytest.mark.parametrize(
    (
        "node_budget",
        "expected_depth_counts",
        "expected_proposal_calls",
        "expected_transition_calls",
    ),
    [
        (3, {1: 3}, 1, 0),
        (12, {1: 3, 2: 9}, 4, 3),
        (39, {1: 3, 2: 9, 3: 27}, 13, 12),
    ],
)
def test_breadth_first_residual_tree_materializes_complete_levels(
    node_budget,
    expected_depth_counts,
    expected_proposal_calls,
    expected_transition_calls,
):
    proposal_calls = []
    transition_calls = []

    def proposal_fn(state):
        proposal_calls.append(state)
        return [
            torch.tensor([0.7, 0.2, 0.1]),
            torch.tensor([0.6, 0.3, 0.1]),
            torch.tensor([0.5, 0.3, 0.2]),
        ]

    def transition_fn(state, token_id):
        transition_calls.append((state, token_id))
        return (*state, token_id)

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=transition_fn,
        head_lambdas=torch.ones(3),
        node_budget=node_budget,
        candidate_selection="distinct_head_top1",
        tree_policy="breadth_first",
    )

    assert tree.num_draft_nodes == node_budget
    assert {
        depth: sum(node.depth == depth for node in tree.nodes[1:])
        for depth in expected_depth_counts
    } == expected_depth_counts
    assert len(proposal_calls) == expected_proposal_calls
    assert len(transition_calls) == expected_transition_calls
    assert tree.proposal_probs is not None
    assert tree.proposal_probs.shape == (node_budget, 3)
    assert torch.allclose(
        tree.proposal_probs[:3],
        torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.0, 0.75, 0.25],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    proposal_rows = [
        contributor.proposal_row
        for node in tree.nodes[1:]
        for contributor in node.contributors
    ]
    assert sorted(proposal_rows) == list(range(node_budget))
    assert all(len(node.contributors) == 1 for node in tree.nodes[1:])


@pytest.mark.parametrize(
    (
        "node_budget",
        "expected_proposal_batch_sizes",
        "expected_transition_batch_sizes",
        "expected_realized_states",
    ),
    [
        (12, [1, 3], [3], 4),
        (39, [1, 3, 9], [3, 9], 13),
    ],
)
def test_breadth_first_residual_tree_batches_each_expanded_level(
    node_budget,
    expected_proposal_batch_sizes,
    expected_transition_batch_sizes,
    expected_realized_states,
):
    proposal_batch_sizes = []
    transition_batch_sizes = []

    def proposals_for_state(_state):
        return [
            torch.tensor([0.7, 0.2, 0.1]),
            torch.tensor([0.6, 0.3, 0.1]),
            torch.tensor([0.5, 0.3, 0.2]),
        ]

    reference = select_residual_tree(
        root_state=(),
        proposal_fn=proposals_for_state,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        node_budget=node_budget,
        candidate_selection="distinct_head_top1",
        tree_policy="breadth_first",
    )

    def proposal_batch_fn(states):
        proposal_batch_sizes.append(len(states))
        return [proposals_for_state(state) for state in states]

    def transition_batch_fn(states, token_ids):
        transition_batch_sizes.append(len(states))
        return [(*state, token_id) for state, token_id in zip(states, token_ids)]

    def unexpected_proposal(_state):
        raise AssertionError("scalar proposal callback must not run")

    def unexpected_transition(_state, _token_id):
        raise AssertionError("scalar transition callback must not run")

    batched = select_residual_tree(
        root_state=(),
        proposal_fn=unexpected_proposal,
        transition_fn=unexpected_transition,
        proposal_batch_fn=proposal_batch_fn,
        transition_batch_fn=transition_batch_fn,
        head_lambdas=torch.ones(3),
        node_budget=node_budget,
        candidate_selection="distinct_head_top1",
        tree_policy="breadth_first",
    )

    assert proposal_batch_sizes == expected_proposal_batch_sizes
    assert transition_batch_sizes == expected_transition_batch_sizes
    assert batched.nodes == reference.nodes
    assert batched.children == reference.children
    assert torch.equal(batched.proposal_probs, reference.proposal_probs)
    assert (
        batched.states[:expected_realized_states]
        == reference.states[:expected_realized_states]
    )
    assert all(state is None for state in batched.states[expected_realized_states:])


def test_best_first_batch_prefetch_preserves_exact_tree_order():
    transition_batch_sizes = []
    proposal_batch_sizes = []

    def proposals_for_state(state):
        offset = sum(state) % 3 if state else 0
        rows = []
        for head_id in range(3):
            row = torch.zeros(5)
            row[(offset + head_id) % 5] = 0.7
            row[(offset + head_id + 1) % 5] = 0.2
            row[(offset + head_id + 2) % 5] = 0.1
            rows.append(row)
        return rows

    common_kwargs = {
        "root_state": (),
        "head_lambdas": torch.tensor([1.0, 0.6, 0.3]),
        "node_budget": 10,
        "max_depth": 3,
        "candidate_selection": "distinct_head_top1",
        "tree_policy": "best_first",
    }
    reference = select_residual_tree(
        proposal_fn=proposals_for_state,
        transition_fn=lambda state, token_id: (*state, token_id),
        **common_kwargs,
    )

    def proposal_batch_fn(states):
        proposal_batch_sizes.append(len(states))
        return [proposals_for_state(state) for state in states]

    def transition_batch_fn(states, token_ids):
        transition_batch_sizes.append(len(states))
        return [(*state, token_id) for state, token_id in zip(states, token_ids)]

    batched = select_residual_tree(
        proposal_fn=proposals_for_state,
        transition_fn=lambda _state, _token_id: pytest.fail(
            "best-first scalar transition unexpectedly ran"
        ),
        proposal_batch_fn=proposal_batch_fn,
        transition_batch_fn=transition_batch_fn,
        **common_kwargs,
    )

    assert transition_batch_sizes
    assert transition_batch_sizes[0] == 3
    assert proposal_batch_sizes == transition_batch_sizes
    assert batched.nodes == reference.nodes
    assert batched.children == reference.children
    assert torch.equal(batched.proposal_probs, reference.proposal_probs)


def test_breadth_first_residual_tree_prevents_deeper_score_preemption():
    def proposal_fn(state):
        return [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]

    common_kwargs = {
        "root_state": (),
        "proposal_fn": proposal_fn,
        "transition_fn": lambda state, token_id: (*state, token_id),
        "head_lambdas": torch.tensor([1.0, 0.2, 0.1]),
        "node_budget": 3,
        "max_depth": 2,
        "candidate_selection": "distinct_head_top1",
    }

    best_first = select_residual_tree(**common_kwargs, tree_policy="best_first")
    breadth_first = select_residual_tree(
        **common_kwargs,
        tree_policy="breadth_first",
    )

    assert best_first.max_depth == 2
    assert len(best_first.children[0]) == 2
    assert breadth_first.max_depth == 1
    assert len(breadth_first.children[0]) == 3


def test_breadth_first_residual_tree_partial_level_order_is_deterministic():
    proposal_calls = []

    def proposal_fn(state):
        proposal_calls.append(state)
        return [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        node_budget=5,
        candidate_selection="distinct_head_top1",
        tree_policy="breadth_first",
    )

    assert [node.parent_id for node in tree.nodes[1:]] == [0, 0, 0, 1, 1]
    assert [node.token_id for node in tree.nodes[1:]] == [0, 1, 2, 0, 1]
    assert [node.depth for node in tree.nodes[1:]] == [1, 1, 1, 2, 2]
    assert proposal_calls == [(), (0,), (1,), (2,)]
    assert tree.proposal_probs is not None
    assert tree.proposal_probs.shape[0] == 12


def test_breadth_first_residual_tree_respects_max_depth():
    proposal_calls = []

    def proposal_fn(state):
        proposal_calls.append(state)
        return [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        node_budget=39,
        max_depth=2,
        candidate_selection="distinct_head_top1",
        tree_policy="breadth_first",
    )

    assert tree.num_draft_nodes == 12
    assert tree.max_depth == 2
    assert [
        sum(node.depth == depth for node in tree.nodes[1:]) for depth in (1, 2)
    ] == [
        3,
        9,
    ]
    assert proposal_calls == [(), (0,), (1,), (2,)]
    assert tree.proposal_probs is not None
    assert tree.proposal_probs.shape[0] == 12


def test_head_prior_tree_scorer_uses_path_probability_without_q_multiplier():
    def proposal_fn(_state):
        return [
            torch.tensor([0.99, 0.009, 0.001]),
            torch.tensor([0.98, 0.019, 0.001]),
            torch.tensor([0.97, 0.02, 0.01]),
        ]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.tensor([100.0, 0.001, 100.0]),
        node_budget=1,
        max_depth=1,
        candidate_selection="distinct_head_top1",
        scorer_mode="head_prior",
        head_prior_probabilities=[0.05, 0.8, 0.1, 0.05],
        collect_scoring_records=True,
    )

    assert tree.nodes[1].token_id == 1
    assert tree.nodes[1].priority == pytest.approx(0.8)
    assert tree.nodes[1].contributors[0].proposal_prob == pytest.approx(0.95)
    assert tree.scoring_records[0].class_probabilities == pytest.approx(
        (0.05, 0.8, 0.1, 0.05)
    )
    assert tree.scoring_records[0].entered_node_ids == [None, 1, None]


def test_greedy_listwise_tree_scorer_receives_serving_features():
    observed = []

    def proposal_fn(_state):
        return [
            torch.tensor([0.7, 0.2, 0.1]),
            torch.tensor([0.6, 0.3, 0.1]),
            torch.tensor([0.5, 0.3, 0.2]),
        ]

    def scorer(features):
        observed.append(features)
        return [0.05, 0.1, 0.8, 0.05]

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.tensor([100.0, 0.001, 0.001]),
        node_budget=1,
        max_depth=1,
        candidate_selection="distinct_head_top1",
        scorer_mode="greedy_listwise",
        listwise_scorer=scorer,
        scorer_top_k=2,
        collect_scoring_records=True,
    )

    assert tree.nodes[1].token_id == 2
    assert tree.nodes[1].priority == pytest.approx(0.8)
    assert len(observed) == 1
    features = observed[0]
    assert features.candidate_token_ids == (0, 1, 2)
    assert features.head_ids == (0, 1, 2)
    assert features.depth == 1
    assert features.top_k == 2
    assert features.conditioned_proposal_probabilities == pytest.approx(
        (0.7, 0.75, 1.0)
    )
    assert features.base_probabilities == pytest.approx((0.7, 0.2, 0.1))
    assert features.topk_masses == pytest.approx((0.9, 1.0, 1.0))
    assert all(value >= 0.0 for value in features.logit_margins)
    assert all(value >= 0.0 for value in features.entropies)
    assert tree.scoring_records[0].entered_node_ids == [None, None, 1]


def test_uniform_tree_scorer_reserves_none_probability():
    tree = select_residual_tree(
        root_state=(),
        proposal_fn=lambda _state: [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ],
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        node_budget=3,
        max_depth=1,
        candidate_selection="distinct_head_top1",
        scorer_mode="uniform",
        collect_scoring_records=True,
    )

    assert [node.priority for node in tree.nodes[1:]] == pytest.approx(
        [0.25, 0.25, 0.25]
    )
    assert tree.scoring_records[0].class_probabilities == pytest.approx(
        (0.25, 0.25, 0.25, 0.25)
    )


def test_proposer_loads_serialized_greedy_scorer_and_maps_features(tmp_path):
    from residual_stack.greedy_tree_scorer import (
        ListwiseLinearScorer,
        ServingFeatureBatch,
    )

    training = ServingFeatureBatch(
        candidate_token_ids=np.asarray(
            [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]
        ),
        conditioned_proposal_probabilities=np.asarray(
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7], [0.4, 0.3, 0.2]]
        ),
        base_probabilities=np.asarray(
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.4, 0.3, 0.2]]
        ),
        logit_margin=np.ones((4, 3)),
        entropy=np.ones((4, 3)),
        topk_mass=np.ones((4, 3)),
        depth=np.ones(4, dtype=np.int64),
        head_ids=(1, 2, 3),
        top_k=10,
    )
    scorer = ListwiseLinearScorer.fit(
        training,
        np.asarray([10, 21, 32, 99]),
        max_iter=10,
    )
    scorer_path = tmp_path / "scorer.json"
    scorer.save(scorer_path)

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        residual_tree_greedy_scorer_path=str(scorer_path)
    )
    proposer._residual_tree_greedy_scorer = None
    proposer._residual_tree_greedy_scorer_path = None
    features = TreeListwiseFeatures(
        candidate_token_ids=(10, 11, 12),
        head_ids=(0, 1, 2),
        conditioned_proposal_probabilities=(0.7, 0.2, 0.1),
        base_probabilities=(0.7, 0.2, 0.1),
        logit_margins=(1.0, 1.0, 1.0),
        entropies=(1.0, 1.0, 1.0),
        topk_masses=(1.0, 1.0, 1.0),
        depth=1,
        top_k=10,
    )

    probabilities = proposer._score_residual_tree_candidates(features)

    expected = scorer.predict_proba(training.subset(slice(0, 1)))[0]
    assert probabilities == pytest.approx(expected)
    assert sum(probabilities) == pytest.approx(1.0)


def test_proposer_validates_runtime_tree_controls_without_model_reload():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        residual_tree=True,
        num_speculative_tokens=39,
        residual_tree_max_depth=3,
        residual_tree_tree_policy="best_first",
        residual_tree_scorer_mode="lambda_q",
        residual_tree_scorer_top_k=10,
    )
    proposer._residual_tree_runtime_config = {}

    normalized = proposer.set_residual_tree_runtime_config(
        {
            "node_budget": 12,
            "max_depth": 2,
            "tree_policy": "breadth_first",
            "batch_drafting": False,
            "scorer_mode": "greedy_listwise",
            "greedy_scorer_path": "/tmp/scorer.json",
            "trace_path": "/tmp/b12_d2.jsonl",
        }
    )

    assert normalized == {
        "node_budget": 12,
        "max_depth": 2,
        "tree_policy": "breadth_first",
        "batch_drafting": False,
        "scorer_mode": "greedy_listwise",
        "greedy_scorer_path": "/tmp/scorer.json",
        "trace_path": "/tmp/b12_d2.jsonl",
        "scorer_top_k": 10,
    }
    assert proposer._residual_tree_runtime_config == normalized

    with pytest.raises(ValueError, match="\\[1, 39\\]"):
        proposer.set_residual_tree_runtime_config({"node_budget": 40})
    with pytest.raises(ValueError, match="requires probabilities"):
        proposer.set_residual_tree_runtime_config(
            {"node_budget": 3, "scorer_mode": "head_prior"}
        )
    with pytest.raises(ValueError, match="unknown residual-tree runtime"):
        proposer.set_residual_tree_runtime_config(
            {"node_budget": 3, "unsupported": True}
        )
    with pytest.raises(TypeError, match="batch_drafting"):
        proposer.set_residual_tree_runtime_config(
            {"node_budget": 3, "batch_drafting": 1}
        )
    with pytest.raises(ValueError, match="engine maximum of 3"):
        proposer.set_residual_tree_runtime_config({"node_budget": 12, "max_depth": 4})
    with pytest.raises(ValueError, match="engine maximum of 3"):
        proposer.set_residual_tree_runtime_config(
            {"node_budget": 12, "max_depth": None}
        )
    with pytest.raises(ValueError, match="unknown runtime final-boundary"):
        proposer.set_residual_tree_runtime_config(
            {"node_budget": 12, "final_boundary_exchange_mode": "unknown"}
        )

    b60_proposer = object.__new__(SpecDecodeBaseProposer)
    b60_proposer.speculative_config = SimpleNamespace(
        residual_tree=True,
        num_speculative_tokens=60,
        residual_tree_max_depth=8,
        residual_tree_tree_policy="eagle3_dynamic",
        residual_tree_candidate_selection="distinct_head_top1",
        residual_tree_scorer_mode="lambda_q",
        residual_tree_scorer_top_k=10,
    )
    b60_proposer._residual_tree_runtime_config = {}
    boundary_config = b60_proposer.set_residual_tree_runtime_config(
        {
            "node_budget": 60,
            "max_depth": 8,
            "tree_policy": "eagle3_dynamic",
            "scorer_mode": "lambda_q",
            "final_boundary_exchange_mode": (
                "parent_supported_half_cutoff_one_swap_v1"
            ),
        }
    )
    assert boundary_config["final_boundary_exchange_mode"] == (
        "parent_supported_half_cutoff_one_swap_v1"
    )

    compact_config = b60_proposer.set_residual_tree_runtime_config(
        {
            "node_budget": 60,
            "max_depth": 8,
            "tree_policy": "eagle3_dynamic",
            "scorer_mode": "lambda_q",
            "trace_path": "/tmp/b60_d8.jsonl",
            "compact_candidate_pool_trace": True,
        }
    )
    assert compact_config["compact_candidate_pool_trace"] is True
    with pytest.raises(TypeError, match="compact_candidate_pool_trace"):
        b60_proposer.set_residual_tree_runtime_config(
            {"compact_candidate_pool_trace": 1}
        )
    with pytest.raises(ValueError, match="compact candidate-pool tracing"):
        b60_proposer.set_residual_tree_runtime_config(
            {"compact_candidate_pool_trace": True}
        )

    assert proposer.set_residual_tree_runtime_config(None) == {}


def test_selection_and_verification_traces_join_by_stable_trace_id(tmp_path):
    tree = select_residual_tree(
        root_state=(),
        proposal_fn=lambda _state: [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ],
        transition_fn=lambda state, token_id: (*state, token_id),
        head_lambdas=torch.ones(3),
        node_budget=3,
        max_depth=1,
        candidate_selection="distinct_head_top1",
        scorer_mode="uniform",
        collect_scoring_records=True,
    )
    draft_tree = tree_to_draft_token_tree(tree)
    draft_tree.trace_id = "trace-1"
    assert draft_tree.truncate(2).trace_id == "trace-1"
    metadata, _ = trees_to_metadata([draft_tree])
    assert metadata.trace_ids == ["trace-1"]
    metadata.input_token_ids = torch.tensor([99, 0, 1, 2], dtype=torch.int32)
    metadata.input_positions = torch.tensor([[7], [8], [8], [8]], dtype=torch.int64)
    metadata.input_attention_slots = torch.tensor(
        [[70], [81], [82], [83]], dtype=torch.int64
    )
    metadata.input_mamba_state_blocks = torch.tensor(
        [[10, 20], [11, 21], [12, 22], [13, 23]], dtype=torch.int32
    )
    metadata.input_attention_group_indices = [0]
    metadata.input_mamba_group_indices = [1, 2]

    path = tmp_path / "steps.jsonl"
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        residual_tree_tree_policy="best_first",
        residual_tree_candidate_selection="distinct_head_top1",
        residual_tree_max_depth=1,
    )
    proposer.num_speculative_tokens = 3
    proposer._write_residual_tree_selection_trace(
        str(path),
        trace_id="trace-1",
        request_index=0,
        request_id="request-1",
        tree=tree,
        scorer_mode="uniform",
    )

    sampler = object.__new__(RejectionSampler)
    sampler.residual_tree_trace_path = str(path)
    sampler._write_residual_tree_verification_trace(
        metadata,
        torch.tensor([0, 2, 2, 2]),
        torch.tensor([[0, 2, -1, -1]], dtype=torch.int32),
        torch.tensor([[1, -1, -1]], dtype=torch.int32),
        greedy=True,
        target_top2_token_ids=torch.tensor(
            [[0, 1], [2, 1], [2, 0], [2, 1]],
            dtype=torch.int64,
        ),
        target_top2_logits=torch.tensor(
            [[3.0, 2.5], [4.0, 1.0], [5.0, 2.0], [6.0, 3.0]],
            dtype=torch.float32,
        ),
    )

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "selection",
        "verification",
    ]
    assert {event["trace_id"] for event in events} == {"trace-1"}
    assert events[0]["request_id"] == "request-1"
    assert all(
        candidate["entered_tree"]
        for candidate in events[0]["candidate_states"][0]["candidates"]
    )
    assert events[1]["accepted_node_ids"] == [1]
    assert events[1]["nodes"][0]["verifier_top1_token_id"] == 0
    assert events[1]["nodes"][0]["verifier_top2_token_ids"] == [0, 1]
    assert events[1]["nodes"][0]["verifier_top2_logits"] == [3.0, 2.5]
    assert events[1]["nodes"][0]["verifier_top1_margin"] == 0.5
    assert events[1]["nodes"][0]["input_token_id"] == 99
    assert events[1]["nodes"][0]["input_position"] == [7]
    assert events[1]["nodes"][0]["input_attention_slots"] == [70]
    assert events[1]["nodes"][3]["input_mamba_state_blocks"] == [13, 23]
    assert events[1]["input_attention_group_indices"] == [0]
    assert events[1]["input_mamba_group_indices"] == [1, 2]


def test_tree_gdn_root_uses_no_spec_packed_equivalent_arithmetic(
    monkeypatch,
):
    calls = []

    def fake_conv(mixed_qkv, *_args, **_kwargs):
        return mixed_qkv

    def fake_packed(**kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(7.0)

    def fail_sigmoid(**_kwargs):
        raise AssertionError("tree root lost the packed-direct decode path")

    monkeypatch.setattr(qwen_gdn, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(
        qwen_gdn,
        "fused_recurrent_gated_delta_rule_packed_decode",
        fake_packed,
    )
    monkeypatch.setattr(
        qwen_gdn,
        "fused_sigmoid_gating_delta_rule_update",
        fail_sigmoid,
    )

    dummy = SimpleNamespace(
        enable_packed_recurrent_decode=True,
        conv1d=SimpleNamespace(bias=None),
        activation=None,
        A_log=torch.ones(1),
        dt_bias=torch.ones(1),
        head_k_dim=4,
        rearrange_mixed_qkv=lambda mixed: (mixed, mixed, mixed),
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        tree_depth_token_indices=(torch.tensor([0]),),
        tree_depth_parent_state_indices=(torch.tensor([1]),),
        tree_depth_child_state_indices=(torch.tensor([1]),),
        tree_depth_query_start_locs=(torch.tensor([0, 1]),),
    )
    core_attn_out = torch.empty(1, 1, 3)

    qwen_gdn.QwenGatedDeltaNetAttention._forward_core_tree(
        dummy,
        mixed_qkv=torch.ones(1, 4),
        b=torch.ones(1, 1),
        a=torch.ones(1, 1),
        core_attn_out=core_attn_out,
        conv_state=torch.zeros(1, 4, 2),
        ssm_state=torch.zeros(2, 1, 1),
        conv_weights=torch.ones(4, 2),
        attn_metadata=metadata,
    )

    assert len(calls) == 1
    assert calls[0]["ssm_state_indices"].tolist() == [1]
    assert calls[0]["final_state_indices"].tolist() == [1]
    assert torch.equal(core_attn_out, torch.full_like(core_attn_out, 7.0))


def test_draft_tree_preserves_unreferenced_generated_proposal_rows():
    def proposal_fn(state):
        if state == ():
            return [
                torch.tensor([0.0, 0.6, 0.4, 0.0]),
                torch.tensor([0.0, 0.2, 0.5, 0.3]),
            ]
        return [torch.ones(4), torch.ones(4)]

    def transition_fn(state, token_id):
        return (*state, token_id)

    tree = select_residual_tree(
        root_state=(),
        proposal_fn=proposal_fn,
        transition_fn=transition_fn,
        head_lambdas=torch.tensor([1.0, 1.0]),
        node_budget=2,
    )

    assert tree.proposal_probs is not None
    assert tree.proposal_probs.shape[0] == 4
    assert [node.token_id for node in tree.nodes] == [-1, 1, 2]

    draft_tree = tree_to_draft_token_tree(tree)

    assert draft_tree.contributor_proposal_rows == [0, 1]
    assert draft_tree.num_proposal_rows == 4
    assert draft_tree.truncate(1).num_proposal_rows == 4


def test_tree_attention_mask_and_position_offsets():
    parent_ids = torch.tensor([-1, 0, 0, 1])

    mask = build_tree_attention_mask(parent_ids)
    offsets = tree_position_offsets(parent_ids)

    expected_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, False, True, False],
            [True, True, False, True],
        ]
    )
    assert torch.equal(mask, expected_mask)
    assert torch.equal(offsets, torch.tensor([0, 1, 1, 2]))


def test_min_tokens_processor_masks_tree_rows_by_request_and_depth():
    processor = MinTokensLogitsProcessor(
        SimpleNamespace(),
        torch.device("cpu"),
        is_pin_memory=False,
    )
    processor.min_toks = {
        0: (2, [101], {2, 3}),
        1: (1, [], {4}),
    }
    logits = torch.zeros((5, 8), dtype=torch.float32)

    processor.apply_with_tree_spec_decode(
        logits,
        node_request_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        node_depths=torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32),
    )

    assert torch.isneginf(logits[0, 2])
    assert torch.isneginf(logits[0, 3])
    assert torch.isneginf(logits[3, 4])
    assert logits[1, 2] == 0
    assert logits[1, 3] == 0
    assert logits[2, 2] == 0
    assert logits[4, 4] == 0


def test_gpu_runner_tree_attention_inputs_mix_tree_and_linear_rows():
    runner = _gpu_runner_stub(req_ids=["tree-req", "linear-req"])
    tree = _branching_draft_tree()

    mask, offsets, parent_indices, depths = runner._calc_tree_attention_inputs(
        {"tree-req": tree},
        np.array([4, 2], dtype=np.int32),
        total_num_scheduled_tokens=6,
    )

    expected_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, False, True, False],
            [True, True, False, True],
            [True, False, False, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask.cpu(), expected_mask)
    assert np.array_equal(offsets, np.array([0, 1, 1, 2, 0, 1]))
    assert torch.equal(
        parent_indices,
        torch.tensor([-1, 0, 0, 1, -1, 0], dtype=torch.int32),
    )
    assert torch.equal(
        depths,
        torch.tensor([0, 1, 1, 2, 0, 1], dtype=torch.int32),
    )


def test_gdn_tree_state_metadata_maps_parent_and_child_slots_by_depth():
    builder = object.__new__(GDNAttentionMetadataBuilder)
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=4,
        block_table_tensor=torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        tree_attn_mask=torch.ones(4, 4, dtype=torch.bool),
        tree_parent_local_indices_cpu=torch.tensor(
            [-1, 0, 0, 1],
            dtype=torch.int32,
        ),
        tree_node_depths_cpu=torch.tensor([0, 1, 1, 2], dtype=torch.int32),
    )

    result = builder._build_tree_state_metadata(
        metadata,
        metadata.block_table_tensor,
    )

    assert result is not None
    token_indices, parent_states, child_states, query_start_locs = result
    assert [t.tolist() for t in token_indices] == [[0], [1, 2], [3]]
    assert [t.tolist() for t in parent_states] == [[10], [10, 10], [11]]
    assert [t.tolist() for t in child_states] == [[10], [11, 12], [13]]
    assert [t.tolist() for t in query_start_locs] == [[0, 1], [0, 1, 2], [0, 1]]


def test_gpu_runner_tree_metadata_offsets_rows_and_logits_indices():
    runner = _gpu_runner_stub(req_ids=["tree-req", "plain-req"], scratch_size=16)
    tree = _branching_draft_tree()

    metadata = runner._calc_tree_spec_decode_metadata(
        {"tree-req": tree},
        np.array([3, 0], dtype=np.int32),
        np.array([4, 6], dtype=np.int32),
    )

    assert metadata.num_draft_tokens == [3, 0]
    assert metadata.num_proposal_rows == [3, 0]
    assert metadata.node_token_ids.cpu().tolist() == [-1, 10, 11, 12, -1]
    assert metadata.parent_node_ids.cpu().tolist() == [-1, 0, 0, 1, -1]
    assert metadata.child_node_ids.cpu().tolist() == [1, 2, 3]
    assert metadata.contributor_proposal_rows.cpu().tolist() == [0, 1, 2]
    assert metadata.logits_indices.cpu().tolist() == [0, 1, 2, 3, 5]
    assert metadata.target_logits_indices.cpu().tolist() == [0, 1, 2, 3, 4]


def test_verify_greedy_tree_follows_matching_child_and_appends_fallback():
    tree = ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(1, 0, 1, 1, 0.7),
            ResidualTreeNode(2, 0, 2, 1, 0.3),
            ResidualTreeNode(3, 1, 3, 2, 0.5),
        ],
        children=[[1, 2], [3], [], []],
    )

    result = verify_greedy_tree(tree, torch.tensor([1, 3, 0, 4]))

    assert result.token_ids == [1, 3, 4]
    assert result.accepted_node_ids == [1, 3]
    assert result.num_accepted == len(result.token_ids) - 1 == 2
    assert result.stopped_at_node_id == 3
    assert result.fallback_token_id == 4


def test_verify_stochastic_tree_uses_clipped_residual_updates():
    tree = _two_child_tree()
    proposal_probs = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    target_probs = torch.tensor(
        [
            [0.3, 0.0, 0.7],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    generator = torch.Generator().manual_seed(1)
    result = verify_stochastic_tree(
        tree,
        target_probs,
        proposal_probs,
        generator=generator,
    )

    assert result.token_ids == [2, 0]
    assert result.accepted_node_ids == [2]
    assert result.num_rejected == 0


def test_stochastic_tree_small_positive_residual_remains_fallback_law():
    tree = ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(
                1,
                0,
                0,
                1,
                0.9,
                (
                    TreeCandidateContributor(
                        head_id=0,
                        proposal_row=0,
                        token_id=0,
                        proposal_prob=1.0,
                        lambda_weight=1.0,
                        score=0.9,
                    ),
                ),
            ),
        ],
        children=[[1], []],
    )
    result = verify_stochastic_tree(
        tree,
        torch.tensor([[0.9, 0.1], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        generator=torch.Generator().manual_seed(0),
        eps=0.15,
    )

    assert result.token_ids == [1]
    assert result.fallback_token_id == 1
    assert result.num_rejected == 1


def test_verify_stochastic_tree_requires_one_child_per_proposal_row():
    tree = ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(
                1,
                0,
                0,
                1,
                0.3,
                (
                    TreeCandidateContributor(
                        head_id=0,
                        proposal_row=0,
                        token_id=0,
                        proposal_prob=0.2,
                        lambda_weight=1.0,
                        score=0.2,
                    ),
                ),
            ),
            ResidualTreeNode(
                2,
                0,
                1,
                1,
                0.2,
                (
                    TreeCandidateContributor(
                        head_id=0,
                        proposal_row=0,
                        token_id=1,
                        proposal_prob=0.3,
                        lambda_weight=1.0,
                        score=0.3,
                    ),
                ),
            ),
        ],
        children=[[1, 2], [], []],
    )
    proposal_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    target_probs = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    try:
        verify_stochastic_tree(
            tree,
            target_probs,
            proposal_probs,
        )
    except ValueError as exc:
        assert "exactly one child token" in str(exc)
    else:
        raise AssertionError("one proposal row must not contribute multiple children")


def test_verify_stochastic_tree_batch_reconstructs_metadata_tree():
    tree = _two_child_tree()
    proposal_probs = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    target_probs = torch.tensor(
        [
            [0.3, 0.0, 0.7],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    tree.proposal_probs = proposal_probs
    metadata, packed_proposal_probs = trees_to_metadata([tree])

    assert packed_proposal_probs is not None
    assert torch.equal(packed_proposal_probs, proposal_probs)

    output = verify_stochastic_tree_batch(
        metadata,
        target_probs,
        packed_proposal_probs,
        generators={0: torch.Generator().manual_seed(1)},
    )

    assert torch.equal(output, torch.tensor([[2, 0, -1]], dtype=torch.int32))


def test_tree_batch_verifier_returns_accepted_node_ids():
    tree = _two_child_tree()
    metadata, _ = trees_to_metadata([tree])

    output, accepted_nodes = verify_greedy_tree_batch_with_nodes(
        metadata,
        torch.tensor([2, 0, 0]),
    )

    assert torch.equal(output, torch.tensor([[2, 0, -1]], dtype=torch.int32))
    assert torch.equal(accepted_nodes, torch.tensor([[2, -1]], dtype=torch.int32))


def test_draft_token_tree_truncate_keeps_prefix_closed_topology():
    tree = DraftTokenTree(
        node_token_ids=[-1, 10, 11, 12],
        parent_node_ids=[-1, 0, 0, 1],
        node_depths=[0, 1, 1, 2],
        node_priorities=[1.0, 0.7, 0.4, 0.3],
        child_start_indices=[0, 2, 3, 3],
        child_end_indices=[2, 3, 3, 3],
        child_node_ids=[1, 2, 3],
        contributor_child_node_ids=[1, 2, 3],
        contributor_proposal_rows=[0, 1, 2],
        contributor_head_ids=[0, 1, 0],
        contributor_scores=[0.7, 0.4, 0.3],
    )

    truncated = tree.truncate(2)

    assert truncated.node_token_ids == [-1, 10, 11]
    assert truncated.parent_node_ids == [-1, 0, 0]
    assert truncated.child_node_ids == [1, 2]
    assert truncated.child_start_indices == [0, 2, 2]
    assert truncated.child_end_indices == [2, 2, 2]
    assert truncated.contributor_child_node_ids == [1, 2]


def test_trees_to_metadata_accepts_draft_token_tree_schema():
    tree = DraftTokenTree.from_draft_tokens([4, 5])

    metadata, proposal_probs = trees_to_metadata([tree])

    assert proposal_probs is None
    assert metadata.num_draft_tokens == [2]
    assert metadata.node_token_ids.tolist() == [-1, 4, 5]
    assert metadata.parent_node_ids.tolist() == [-1, 0, 1]


def test_trees_to_metadata_offsets_draft_tree_proposal_rows():
    first = _draft_tree_with_contributors(10, proposal_rows=[0, 2])
    second = _draft_tree_with_contributors(20, proposal_rows=[0])

    metadata, _ = trees_to_metadata([first, second])

    assert first.num_proposal_rows == 3
    assert second.num_proposal_rows == 1
    assert metadata.num_proposal_rows == [3, 1]
    assert metadata.contributor_proposal_rows.tolist() == [0, 2, 3]


def test_trees_to_metadata_offsets_unreferenced_draft_tree_rows():
    first = _draft_tree_with_contributors(
        10,
        proposal_rows=[0],
        proposal_num_rows=3,
    )
    second = _draft_tree_with_contributors(20, proposal_rows=[0])

    metadata, _ = trees_to_metadata([first, second])

    assert metadata.num_proposal_rows == [3, 1]
    assert metadata.contributor_proposal_rows.tolist() == [0, 3]


def test_internal_eagle_tree_transition_satisfies_model_api_check():
    class ModelWithHeadsOnly:
        def has_residual_tree_heads(self):
            return True

        def compute_residual_head_probs(self, hidden_states):
            return hidden_states

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.pass_hidden_states_to_model = True
    proposer.parallel_drafting = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_attn_groups = [object()]
    proposer._draft_attn_layer_names = {"model.layers.0.self_attn"}
    proposer.vllm_config = _config_with_dcp_world_size(1)

    model = ModelWithHeadsOnly()
    proposer._validate_residual_tree_model_api(model)

    model.residual_tree_required_candidate_selection = "distinct_head_top1"
    proposer.speculative_config = SimpleNamespace(
        residual_tree_candidate_selection="head_top1"
    )
    with pytest.raises(ValueError, match="requires candidate selection"):
        proposer._validate_residual_tree_model_api(model)
    proposer.speculative_config.residual_tree_candidate_selection = "distinct_head_top1"
    proposer._validate_residual_tree_model_api(model)

    proposer.vllm_config = _config_with_dcp_world_size(2)
    try:
        proposer._validate_residual_tree_model_api(ModelWithHeadsOnly())
    except NotImplementedError:
        pass
    else:
        raise AssertionError("DCP must require an explicit tree transition")


def test_residual_tree_transition_uses_internal_eagle_without_model_hook():
    proposer = object.__new__(SpecDecodeBaseProposer)
    fallback = _ResidualTreeDraftState(
        proposal_hidden=torch.ones(4),
        transition_hidden=torch.ones(4),
    )
    expected = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(4),
        transition_hidden=torch.zeros(4),
    )
    calls = []

    def internal_transition(state, token_id):
        calls.append((state, token_id))
        return expected

    proposer._internal_eagle_tree_transition = internal_transition

    result = proposer._residual_tree_transition(SimpleNamespace(), fallback, 7)

    assert result is expected
    assert calls == [(fallback, 7)]


def test_residual_tree_head_probs_batch_projects_one_breadth_level():
    class BatchedHeadModel:
        @staticmethod
        def compute_residual_head_logits(hidden_states):
            first = torch.stack(
                [
                    hidden_states[:, 0],
                    hidden_states[:, 1],
                    torch.zeros_like(hidden_states[:, 0]),
                ],
                dim=-1,
            )
            second = first.flip(-1)
            return torch.stack([first, second], dim=1)

    states = [
        _ResidualTreeDraftState(
            proposal_hidden=torch.tensor([1.0, 2.0]),
            transition_hidden=torch.zeros(2),
        ),
        _ResidualTreeDraftState(
            proposal_hidden=torch.tensor([3.0, 4.0]),
            transition_hidden=torch.zeros(2),
        ),
    ]
    proposer = object.__new__(SpecDecodeBaseProposer)

    rows = proposer._residual_tree_head_probs_batch(BatchedHeadModel(), states)

    assert isinstance(rows, torch.Tensor)
    assert rows.shape == (2, 2, 3)
    assert len(rows) == 2
    assert all(len(state_rows) == 2 for state_rows in rows)
    expected_logits = BatchedHeadModel.compute_residual_head_logits(
        torch.stack([state.proposal_hidden for state in states])
    )
    expected = torch.softmax(expected_logits, dim=-1)
    assert torch.allclose(rows[0][0], expected[0, 0])
    assert torch.allclose(rows[1][1], expected[1, 1])


def test_residual_tree_root_head_probs_batch_calls_hidden_only_hook_once():
    class CountingHeadModel:
        def __init__(self):
            self.batch_sizes = []

        def compute_residual_head_logits(self, hidden_states):
            self.batch_sizes.append(hidden_states.shape[0])
            first = torch.stack(
                [hidden_states[:, 0], hidden_states[:, 1]],
                dim=-1,
            )
            return torch.stack([first, first.flip(-1)], dim=1)

    states = [
        _ResidualTreeDraftState(
            proposal_hidden=torch.tensor([float(index + 1), float(index + 4)]),
            transition_hidden=torch.zeros(2),
        )
        for index in range(3)
    ]
    proposer = object.__new__(SpecDecodeBaseProposer)
    model = CountingHeadModel()

    rows = proposer._residual_tree_root_head_probs_batch(model, states)

    assert isinstance(rows, torch.Tensor)
    assert model.batch_sizes == [3]
    assert rows.shape == (3, 2, 2)
    assert len(rows) == 3
    assert all(len(state_rows) == 2 for state_rows in rows)


@pytest.mark.parametrize("state_parameter", ["state", "states", "kwargs"])
def test_residual_tree_root_head_probs_batch_rejects_stateful_hooks(
    state_parameter,
):
    class ScalarStateModel:
        @staticmethod
        def compute_residual_head_logits(state, hidden_states):
            raise AssertionError("stateful hook must retain the scalar path")

    class BatchedStatesModel:
        @staticmethod
        def compute_residual_head_logits(states, hidden_states):
            raise AssertionError("implicit cross-request state batching is unsafe")

    class KwargsOnlyModel:
        @staticmethod
        def compute_residual_head_logits(**kwargs):
            raise AssertionError("kwargs-only hook has no explicit batch contract")

    models = {
        "state": ScalarStateModel(),
        "states": BatchedStatesModel(),
        "kwargs": KwargsOnlyModel(),
    }
    proposer = object.__new__(SpecDecodeBaseProposer)
    states = [
        _ResidualTreeDraftState(
            proposal_hidden=torch.ones(2),
            transition_hidden=torch.zeros(2),
        )
    ]

    assert (
        proposer._residual_tree_root_head_probs_batch(models[state_parameter], states)
        is None
    )


def test_residual_tree_transition_plan_deduplicates_shared_prefixes():
    root_hidden = torch.tensor([1.0, 0.0])
    left_hidden = torch.tensor([2.0, 0.0])
    right_hidden = torch.tensor([3.0, 0.0])
    left = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(2),
        transition_hidden=left_hidden,
        payload=_ResidualTreeKVPayload(
            req_index=2,
            root_position=7,
            path_token_ids=(10,),
            path_hidden_inputs=(root_hidden,),
        ),
    )
    right = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(2),
        transition_hidden=right_hidden,
        payload=_ResidualTreeKVPayload(
            req_index=2,
            root_position=7,
            path_token_ids=(11,),
            path_hidden_inputs=(root_hidden,),
        ),
    )

    plan = SpecDecodeBaseProposer._make_residual_tree_transition_plan(
        [left, left, right],
        [20, 21, 30],
    )

    assert plan.token_ids == (10, 20, 21, 11, 30)
    assert plan.parent_local_indices == (-1, 0, 0, -1, 3)
    assert plan.node_depths == (1, 2, 2, 1, 2)
    assert plan.result_local_indices == (1, 2, 4)
    assert torch.equal(plan.hidden_inputs[0], root_hidden)
    assert torch.equal(plan.hidden_inputs[1], left_hidden)
    assert torch.equal(plan.hidden_inputs[4], right_hidden)
    assert plan.child_payloads[1].path_token_ids == (10, 21)
    assert len(plan.child_payloads[1].path_hidden_inputs) == 2
    assert plan.child_payloads[1].path_kv == ()


def test_residual_tree_transition_batch_metadata_uses_union_trie_mask():
    root_hidden = torch.tensor([1.0, 0.0])
    root = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(2),
        transition_hidden=root_hidden,
        payload=_ResidualTreeKVPayload(req_index=0, root_position=7),
    )
    plan = SpecDecodeBaseProposer._make_residual_tree_batch_transition_plan(
        [root, root, root],
        [10, 11, 12],
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.device = torch.device("cpu")
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.positions = torch.tensor([8, 8, 8], dtype=torch.int64)
    common = _common_attn_metadata_for_block_table([[0, 1, 2]])
    common.dcp_local_seq_lens = None

    metadata = proposer._make_residual_tree_transition_batch_metadata(
        common,
        plan,
        torch.tensor([8, 9, 10], dtype=torch.int64),
    )

    assert metadata.query_start_loc.tolist() == [0, 3]
    assert metadata.seq_lens.tolist() == [11]
    assert metadata.max_query_len == 3
    assert metadata.max_seq_len == 11
    assert torch.equal(metadata.tree_attn_mask, torch.eye(3, dtype=torch.bool))
    assert metadata.tree_parent_local_indices_cpu.tolist() == [-1, -1, -1]
    assert metadata.tree_node_depths_cpu.tolist() == [1, 1, 1]


def test_residual_tree_transition_batch_metadata_packs_multiple_requests():
    first = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(2),
        transition_hidden=torch.tensor([1.0, 0.0]),
        payload=_ResidualTreeKVPayload(req_index=0, root_position=7),
    )
    second = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(2),
        transition_hidden=torch.tensor([0.0, 1.0]),
        payload=_ResidualTreeKVPayload(req_index=2, root_position=3),
    )
    plan = SpecDecodeBaseProposer._make_residual_tree_batch_transition_plan(
        [first, first, second],
        [10, 11, 20],
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.device = torch.device("cpu")
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.positions = torch.tensor([8, 8, 4], dtype=torch.int64)
    common = _common_attn_metadata_for_block_table([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    common.dcp_local_seq_lens = None

    metadata = proposer._make_residual_tree_transition_batch_metadata(
        common,
        plan,
        torch.tensor([8, 9, 24], dtype=torch.int64),
    )

    assert plan.request_indices == (0, 2)
    assert plan.query_start_locs == (0, 2, 3)
    assert metadata.query_start_loc.tolist() == [0, 2, 3]
    assert metadata.seq_lens.tolist() == [10, 5]
    assert metadata.num_reqs == 2
    assert metadata.max_query_len == 2
    assert metadata.tree_attn_mask.tolist() == [
        [True, False],
        [False, True],
        [True, False],
    ]
    assert metadata.block_table_tensor.tolist() == [
        [0, 1, 2],
        [6, 7, 8],
    ]


def test_internal_eagle_tree_transition_batch_runs_one_union_forward(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            *,
            input_ids,
            positions,
            inputs_embeds,
            hidden_states,
        ):
            assert inputs_embeds is None
            self.calls.append(
                (
                    input_ids.clone(),
                    positions.clone(),
                    hidden_states.clone(),
                )
            )
            token_values = input_ids.to(hidden_states.dtype).unsqueeze(-1)
            return hidden_states + token_values, hidden_states + 100

    @contextmanager
    def fake_forward_context(*_args, **_kwargs):
        yield

    monkeypatch.setattr(
        "vllm.v1.spec_decode.llm_base_proposer.set_forward_context",
        fake_forward_context,
    )
    root_hidden = torch.tensor([1.0, 0.0])
    root = _ResidualTreeDraftState(
        proposal_hidden=torch.zeros(2),
        transition_hidden=root_hidden,
        payload=_ResidualTreeKVPayload(req_index=0, root_position=3),
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.device = torch.device("cpu")
    proposer.block_size = 4
    proposer.num_speculative_tokens = 8
    proposer.input_ids = torch.zeros(16, dtype=torch.int32)
    proposer.hidden_states = torch.zeros(16, 2)
    proposer.positions = torch.zeros(16, dtype=torch.int64)
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.supports_mm_inputs = False
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(uses_mrope=False)
    )
    proposer.model = FakeModel()
    proposer._residual_tree_kv_caches = (torch.zeros(1),)
    proposer._residual_tree_common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=4,
        block_table_tensor=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([3], dtype=torch.int64),
    )
    captured_common_metadata = []
    triton_metadata = object.__new__(TritonAttentionMetadata)
    triton_metadata.tree_attn_mask = torch.ones(3, 3, dtype=torch.bool)

    def build_metadata(common_attn_metadata, draft_index):
        captured_common_metadata.append((common_attn_metadata, draft_index))
        return [triton_metadata], {"layer": triton_metadata}

    proposer.build_per_group_and_layer_attn_metadata = build_metadata
    proposer._determine_batch_execution_and_padding = (
        lambda num_tokens, use_cudagraphs: (None, num_tokens, None)
    )
    proposer._get_slot_mapping = lambda _num_tokens, slot_mapping: {
        "layer": slot_mapping
    }
    proposer.model_returns_tuple = lambda: True

    children = proposer._internal_eagle_tree_transition_batch(
        [root, root, root],
        [10, 11, 12],
    )

    assert len(proposer.model.calls) == 1
    input_ids, positions, hidden_inputs = proposer.model.calls[0]
    assert input_ids.tolist() == [10, 11, 12]
    assert positions.tolist() == [4, 4, 4]
    assert torch.equal(hidden_inputs, root_hidden.repeat(3, 1))
    assert captured_common_metadata[0][1] == 1
    assert captured_common_metadata[0][0].slot_mapping.tolist() == [4, 5, 6]
    assert [state.payload.path_token_ids for state in children] == [
        (10,),
        (11,),
        (12,),
    ]
    assert torch.equal(children[0].proposal_hidden, torch.tensor([11.0, 10.0]))
    assert torch.equal(children[0].transition_hidden, torch.tensor([101.0, 100.0]))


def test_residual_tree_state_coercion_preserves_fallback_payload():
    payload = _ResidualTreeKVPayload(req_index=0, root_position=7)
    fallback = _ResidualTreeDraftState(
        proposal_hidden=torch.ones(4),
        transition_hidden=torch.ones(4),
        payload=payload,
    )

    state = SpecDecodeBaseProposer._coerce_residual_tree_state(
        (torch.zeros(1, 4), torch.ones(1, 4)),
        fallback=fallback,
    )

    assert state.payload is payload


def test_internal_eagle_tree_transition_rejects_per_token_head_kv_quant():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer._draft_attn_layer_names = {"model.layers.0.self_attn"}
    proposer.compilation_config = _compilation_config(
        {
            "model.layers.0.self_attn": _layer_with_kv_cache(
                torch.zeros(1, 2, 1, 1),
                per_token_head_quant=True,
            )
        }
    )

    try:
        proposer._collect_draft_kv_caches()
    except NotImplementedError as exc:
        assert "per-token-head KV quantization" in str(exc)
    else:
        raise AssertionError("per-token-head KV quantization must fail closed")


def test_internal_eagle_tree_kv_slot_snapshot_and_restore():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer._draft_attn_layer_names = {"model.layers.0.self_attn"}
    proposer.block_size = 4
    kv_cache = torch.zeros(2, 2, 4, 3)
    kv_cache[1, :, 2] = 7
    proposer.compilation_config = _compilation_config(
        {"model.layers.0.self_attn": _layer_with_kv_cache(kv_cache)}
    )

    proposer._residual_tree_kv_caches = proposer._collect_draft_kv_caches()
    snapshot = proposer._snapshot_residual_tree_slot(6)
    kv_cache[1, :, 2] = 3

    payload = _ResidualTreeKVPayload(req_index=0, root_position=5, path_kv=(snapshot,))
    proposer._restore_residual_tree_path(
        payload,
        _common_attn_metadata_for_block_table([[0, 1]]),
    )

    assert torch.equal(kv_cache[1, :, 2], torch.full((2, 3), 7.0))


def test_gpu_runner_tree_kv_compaction_moves_accepted_path_slots():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    kv_cache[0, :, 3] = 9
    _install_compaction_state(runner, kv_cache)
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[3, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
    )

    assert torch.equal(kv_cache[0, :, 1], torch.full((2, 2), 9.0))


def test_gpu_runner_tree_kv_compaction_moves_uniform_type_group_slots():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    kv_cache[0, :, 3] = 9
    _install_compaction_state(runner, kv_cache)
    layer_spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = (
        UniformTypeKVCacheSpecs(
            block_size=layer_spec.block_size,
            kv_cache_specs={"layer": layer_spec},
        )
    )
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[3, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
    )

    assert torch.equal(kv_cache[0, :, 1], torch.full((2, 2), 9.0))


def test_gpu_runner_tree_kv_compaction_with_no_child_keeps_root_state():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2)
    original = kv_cache.clone()
    _install_compaction_state(runner, kv_cache)
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.full((1, 3), -1, dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    result = runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
    )

    assert result is None
    assert torch.equal(kv_cache, original)


def test_gpu_runner_tree_kv_compaction_preserves_sibling_source_slot():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    kv_cache[0, :, 1] = 11
    kv_cache[0, :, 2] = 22
    _install_compaction_state(runner, kv_cache)
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[2, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
    )

    assert torch.equal(kv_cache[0, :, 1], torch.full((2, 2), 22.0))
    assert torch.equal(kv_cache[0, :, 2], torch.full((2, 2), 22.0))


def test_gpu_runner_tree_kv_compaction_rejects_negative_slot():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2)
    original = kv_cache.clone()
    _install_compaction_state(runner, kv_cache)
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[2, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    with pytest.raises(ValueError, match="invalid slot"):
        runner._compact_tree_spec_kv_cache(
            sampler_output,
            scheduler_output,
            metadata,
            {"layer": torch.tensor([0, 1, -1, 3])},
        )

    assert torch.equal(kv_cache, original)


def test_gpu_runner_tree_kv_prevalidated_path_skips_device_reductions(monkeypatch):
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    kv_cache[0, :, 2] = 9
    conv_state = torch.zeros(5, 2, 3)
    ssm_state = torch.zeros(5, 2, 2, 2)
    conv_state[3] = 13
    ssm_state[3] = 17
    _install_compaction_state(runner, kv_cache)
    _add_mamba_compaction_state(
        runner,
        conv_state,
        ssm_state,
        block_table=torch.tensor([[0, 0, 0, 1, 2, 3, 4]], dtype=torch.int32),
    )
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[2, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    def fail_device_reduction(*args, **kwargs):
        raise AssertionError("prevalidated cache slots must not be reduced")

    monkeypatch.setattr(torch, "any", fail_device_reduction)
    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
        cache_slots_are_prevalidated=True,
    )

    assert torch.equal(kv_cache[0, :, 1], torch.full((2, 2), 9.0))
    assert torch.equal(conv_state[1], torch.full((2, 3), 13.0))
    assert torch.equal(ssm_state[1], torch.full((2, 2, 2), 17.0))


def test_gpu_runner_tree_compaction_moves_mamba_state_slots():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    conv_state = torch.zeros(5, 2, 3)
    ssm_state = torch.zeros(5, 2, 2, 2)
    conv_state[3] = 13
    ssm_state[3] = 17
    _install_compaction_state(runner, kv_cache)
    _add_mamba_compaction_state(
        runner,
        conv_state,
        ssm_state,
        block_table=torch.tensor([[0, 0, 0, 1, 2, 3, 4]], dtype=torch.int32),
    )
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[2, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
    )

    assert torch.equal(conv_state[1], torch.full((2, 3), 13.0))
    assert torch.equal(ssm_state[1], torch.full((2, 2, 2), 17.0))


def test_gpu_runner_tree_compaction_keeps_shared_attention_groups_distinct():
    runner = _gpu_runner_stub(req_ids=["req"])
    shared_kv_cache = torch.zeros(2, 2, 4, 2)
    shared_kv_cache[0, :, 2] = 13
    shared_kv_cache[1, :, 2] = 23
    _install_compaction_state(runner, shared_kv_cache)
    attention_spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=attention_spec, layer_names=["layer_a"]),
        SimpleNamespace(kv_cache_spec=attention_spec, layer_names=["layer_b"]),
    ]
    runner.compilation_config = _compilation_config(
        {
            "layer_a": _layer_with_kv_cache(shared_kv_cache),
            "layer_b": _layer_with_kv_cache(shared_kv_cache),
        }
    )
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[2, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {
            "layer_a": torch.tensor([0, 1, 2, 3]),
            "layer_b": torch.tensor([4, 5, 6, 7]),
        },
    )

    assert torch.equal(shared_kv_cache[0, :, 1], torch.full((2, 2), 13.0))
    assert torch.equal(shared_kv_cache[1, :, 1], torch.full((2, 2), 23.0))


def test_gpu_runner_tree_compaction_keeps_shared_mamba_groups_distinct():
    runner = _gpu_runner_stub(req_ids=["req"])
    # Block id zero is the reserved NULL_BLOCK_ID. Give the two logical
    # groups disjoint, valid block-table namespaces even though they share
    # the same physical state tensors.
    conv_state = torch.zeros(9, 2, 3)
    ssm_state = torch.zeros(9, 2, 2, 2)
    conv_state[3] = 13
    ssm_state[3] = 17
    conv_state[7] = 23
    ssm_state[7] = 27
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=(tuple(conv_state.shape[1:]), tuple(ssm_state.shape[1:])),
        dtypes=(conv_state.dtype, ssm_state.dtype),
        mamba_cache_mode="none",
        num_speculative_blocks=3,
    )
    runner.cache_config = SimpleNamespace(mamba_cache_mode="none")
    runner.seq_lens = torch.tensor([4], dtype=torch.int32)
    runner.input_batch.block_table = [
        SimpleNamespace(
            get_device_tensor=lambda num_reqs: torch.tensor(
                [[1, 2, 3, 4]], dtype=torch.int32
            )
        ),
        SimpleNamespace(
            get_device_tensor=lambda num_reqs: torch.tensor(
                [[5, 6, 7, 8]], dtype=torch.int32
            )
        ),
    ]
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=["mamba_a"]),
            SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=["mamba_b"]),
        ]
    )
    shared_states = [conv_state, ssm_state]
    runner.compilation_config = _compilation_config(
        {
            "mamba_a": SimpleNamespace(kv_cache=shared_states),
            "mamba_b": SimpleNamespace(kv_cache=shared_states),
        }
    )
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[2, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {},
    )

    assert torch.equal(conv_state[1], torch.full((2, 3), 13.0))
    assert torch.equal(ssm_state[1], torch.full((2, 2, 2), 17.0))
    assert torch.equal(conv_state[5], torch.full((2, 3), 23.0))
    assert torch.equal(ssm_state[5], torch.full((2, 2, 2), 27.0))


def test_gpu_runner_tree_compaction_canonicalizes_mamba_without_kv_move():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    conv_state = torch.zeros(5, 2, 3)
    ssm_state = torch.zeros(5, 2, 2, 2)
    conv_state[2] = 13
    ssm_state[2] = 17
    _install_compaction_state(runner, kv_cache)
    _add_mamba_compaction_state(
        runner,
        conv_state,
        ssm_state,
        block_table=torch.tensor([[0, 0, 0, 1, 2, 3, 4]], dtype=torch.int32),
    )
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[1, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    result = runner._compact_tree_spec_kv_cache(
        sampler_output,
        scheduler_output,
        metadata,
        {"layer": torch.tensor([0, 1, 2, 3])},
    )

    assert result is None
    assert torch.equal(conv_state[1], torch.full((2, 3), 13.0))
    assert torch.equal(ssm_state[1], torch.full((2, 2, 2), 17.0))


def test_gpu_runner_tree_state_update_skips_linear_mamba_postprocess():
    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = SimpleNamespace()
    runner.model_config = SimpleNamespace(is_hybrid=True)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
    runner.num_accepted_tokens = SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32))
    runner.input_batch = SimpleNamespace(
        num_accepted_tokens_cpu_tensor=torch.zeros(2, dtype=torch.int32)
    )
    event = SimpleNamespace(recorded=False)
    event.record = lambda: setattr(event, "recorded", True)
    runner.num_accepted_tokens_event = event

    runner._update_states_after_model_execute(
        torch.tensor(
            [
                [10, 11, -1],
                [20, -1, -1],
            ],
            dtype=torch.int32,
        ),
        SimpleNamespace(),
        tree_spec_decode=True,
    )

    assert runner.num_accepted_tokens.gpu.tolist() == [2, 1]
    assert runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [1, 1]
    assert event.recorded


def test_gpu_runner_tree_state_update_keeps_actual_count_in_none_mode():
    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = SimpleNamespace()
    runner.model_config = SimpleNamespace(is_hybrid=True)
    runner.cache_config = SimpleNamespace(mamba_cache_mode="none")
    runner.num_accepted_tokens = SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32))
    runner.input_batch = SimpleNamespace(
        num_accepted_tokens_cpu_tensor=torch.zeros(2, dtype=torch.int32)
    )
    event = SimpleNamespace(recorded=False)
    event.record = lambda: setattr(event, "recorded", True)
    runner.num_accepted_tokens_event = event

    runner._update_states_after_model_execute(
        torch.tensor(
            [
                [10, 11, -1],
                [20, -1, -1],
            ],
            dtype=torch.int32,
        ),
        SimpleNamespace(),
        tree_spec_decode=True,
    )

    assert runner.num_accepted_tokens.gpu.tolist() == [2, 1]
    assert runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [2, 1]
    assert event.recorded


def test_gpu_runner_tree_compaction_moves_output_rows_and_position_cols():
    runner = object.__new__(GPUModelRunner)
    runner.input_ids = SimpleNamespace(
        gpu=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    )
    runner.enable_prompt_embeds = False
    runner.uses_mrope = True
    runner.uses_xdrope_dim = 0
    runner.mrope_positions = SimpleNamespace(
        gpu=torch.arange(15, dtype=torch.int64).reshape(3, 5).clone()
    )
    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    sample_hidden_states = hidden_states + 100
    aux_hidden_states = [hidden_states + 200]
    original_positions = runner.mrope_positions.gpu.clone()
    original_hidden = hidden_states.clone()

    runner._compact_tree_spec_output_tensors(
        (
            torch.tensor([3], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ),
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
    )

    assert runner.input_ids.gpu[2] == 13
    assert torch.equal(
        runner.mrope_positions.gpu[:, 2],
        original_positions[:, 3],
    )
    assert torch.equal(hidden_states[2], original_hidden[3])
    assert torch.equal(sample_hidden_states[2], original_hidden[3] + 100)
    assert torch.equal(aux_hidden_states[0][2], original_hidden[3] + 200)


def test_gpu_runner_tree_kv_compaction_rejects_per_token_head_quant():
    runner = _gpu_runner_stub(req_ids=["req"])
    kv_cache = torch.zeros(1, 2, 4, 2)
    _install_compaction_state(runner, kv_cache, per_token_head_quant=True)
    tree = _branching_draft_tree()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    sampler_output = SimpleNamespace(
        accepted_tree_node_ids=torch.tensor([[3, -1, -1]], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_token_trees={"req": tree},
    )

    try:
        runner._compact_tree_spec_kv_cache(
            sampler_output,
            scheduler_output,
            metadata,
            {"layer": torch.tensor([0, 1, 2, 3])},
        )
    except NotImplementedError as exc:
        assert "per-token-head KV quantization" in str(exc)
    else:
        raise AssertionError("per-token-head KV quantization must fail closed")


def _two_child_tree() -> ResidualTree:
    return ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(
                1,
                0,
                1,
                1,
                0.4,
                (
                    TreeCandidateContributor(
                        head_id=0,
                        proposal_row=0,
                        token_id=1,
                        proposal_prob=1.0,
                        lambda_weight=1.0,
                        score=0.4,
                    ),
                ),
            ),
            ResidualTreeNode(
                2,
                0,
                2,
                1,
                0.3,
                (
                    TreeCandidateContributor(
                        head_id=1,
                        proposal_row=1,
                        token_id=2,
                        proposal_prob=0.5,
                        lambda_weight=1.0,
                        score=0.3,
                    ),
                ),
            ),
        ],
        children=[[1, 2], [], []],
    )


def _branching_draft_tree() -> DraftTokenTree:
    return DraftTokenTree(
        node_token_ids=[-1, 10, 11, 12],
        parent_node_ids=[-1, 0, 0, 1],
        node_depths=[0, 1, 1, 2],
        node_priorities=[1.0, 0.7, 0.4, 0.3],
        child_start_indices=[0, 2, 3, 3],
        child_end_indices=[2, 3, 3, 3],
        child_node_ids=[1, 2, 3],
        contributor_child_node_ids=[1, 2, 3],
        contributor_proposal_rows=[0, 1, 2],
        contributor_head_ids=[0, 1, 0],
        contributor_scores=[0.7, 0.4, 0.3],
    )


def _gpu_runner_stub(
    *,
    req_ids: list[str],
    scratch_size: int = 8,
) -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids))
    runner.arange_np = np.arange(scratch_size, dtype=np.int32)
    runner._arange_scratch = np.empty(scratch_size, dtype=np.int32)
    return runner


def _install_compaction_state(
    runner: GPUModelRunner,
    kv_cache: torch.Tensor,
    *,
    per_token_head_quant: bool = False,
) -> None:
    spec = FullAttentionSpec(
        block_size=kv_cache.shape[2],
        num_kv_heads=1,
        head_size=kv_cache.shape[-1],
        dtype=kv_cache.dtype,
    )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=spec, layer_names=["layer"]),
        ],
    )
    runner.compilation_config = _compilation_config(
        {
            "layer": _layer_with_kv_cache(
                kv_cache,
                per_token_head_quant=per_token_head_quant,
            ),
        }
    )


def _add_mamba_compaction_state(
    runner: GPUModelRunner,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    *,
    block_table: torch.Tensor,
) -> None:
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=(tuple(conv_state.shape[1:]), tuple(ssm_state.shape[1:])),
        dtypes=(conv_state.dtype, ssm_state.dtype),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
    runner.seq_lens = torch.tensor([4], dtype=torch.int32)
    runner.input_batch.block_table = [
        SimpleNamespace(get_device_tensor=lambda num_reqs: torch.empty(1, 4)),
        SimpleNamespace(get_device_tensor=lambda num_reqs: block_table[:num_reqs]),
    ]
    runner.kv_cache_config.kv_cache_groups.append(
        SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=["mamba"])
    )
    runner.compilation_config.static_forward_context["mamba"] = SimpleNamespace(
        kv_cache=[conv_state, ssm_state],
    )


def _draft_tree_with_contributors(
    token_id: int,
    *,
    proposal_rows: list[int],
    proposal_num_rows: int | None = None,
) -> DraftTokenTree:
    child_count = len(proposal_rows)
    node_token_ids = [-1] + [token_id + i for i in range(child_count)]
    parent_node_ids = [-1] + [0 for _ in range(child_count)]
    return DraftTokenTree(
        node_token_ids=node_token_ids,
        parent_node_ids=parent_node_ids,
        node_depths=[0] + [1 for _ in range(child_count)],
        node_priorities=[1.0] + [0.5 for _ in range(child_count)],
        child_start_indices=[0] + [child_count for _ in range(child_count)],
        child_end_indices=[child_count] + [child_count for _ in range(child_count)],
        child_node_ids=list(range(1, child_count + 1)),
        contributor_child_node_ids=list(range(1, child_count + 1)),
        contributor_proposal_rows=proposal_rows,
        contributor_head_ids=list(range(child_count)),
        contributor_scores=[0.5 for _ in range(child_count)],
        proposal_num_rows=proposal_num_rows,
    )


def _config_with_dcp_world_size(world_size: int):
    class ParallelConfig:
        decode_context_parallel_size = world_size

    class Config:
        parallel_config = ParallelConfig()

    return Config()


def _compilation_config(static_forward_context):
    class Config:
        pass

    config = Config()
    config.static_forward_context = static_forward_context
    return config


def _layer_with_kv_cache(kv_cache, *, per_token_head_quant=False):
    class Impl:
        _is_per_token_head_quant = per_token_head_quant

    class Layer:
        pass

    layer = Layer()
    layer.impl = Impl()
    layer.kv_cache = kv_cache
    return layer


def _common_attn_metadata_for_block_table(block_table):
    class Metadata:
        pass

    metadata = Metadata()
    metadata.block_table_tensor = torch.tensor(block_table, dtype=torch.int32)
    return metadata


def test_gdn_backend_declares_batch_invariant_tree_updates():
    assert GDNAttentionBackend.supports_batch_invariance()
