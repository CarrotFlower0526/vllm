# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    _ResidualTreeDraftState,
)


class _StockTop2Model:
    """Minimal Stock H1 hook with state-dependent deterministic candidates."""

    def __init__(self) -> None:
        self.hidden_batches: list[list[int]] = []
        self.scored_batches: list[list[int]] = []
        self.union_weights: list[float] = []

    def compute_stock_top2_greedy_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        state_ids = [int(value) for value in hidden_states[:, 0].tolist()]
        self.hidden_batches.append(state_ids)
        rows = []
        for state_id in state_ids:
            if state_id == 0:
                rows.append([10, 20])
            elif state_id == 10:
                rows.append([11, 12])
            elif state_id == 20:
                rows.append([21, 22])
            else:
                rows.append([state_id + 1, state_id + 2])
        return torch.tensor(rows, dtype=torch.long, device=hidden_states.device)

    def compute_stock_top2_greedy_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_ids = [int(value) for value in hidden_states[:, 0].tolist()]
        self.scored_batches.append(state_ids)
        tokens = self.compute_stock_top2_greedy_tokens(hidden_states)
        probabilities = torch.tensor(
            [[0.8, 0.2]] * len(state_ids),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return tokens, probabilities

    def compute_stock_top10_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        *,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_ids = [int(value) for value in hidden_states[:, 0].tolist()]
        tokens = torch.tensor(
            [
                [state_id * 100 + offset for offset in range(1, top_k + 1)]
                for state_id in state_ids
            ],
            dtype=torch.long,
            device=hidden_states.device,
        )
        probabilities = torch.tensor(
            [[0.4 / offset for offset in range(1, top_k + 1)]] * len(state_ids),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return tokens, probabilities

    def compute_hybrid_top9_h2_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, probabilities = self.compute_stock_top10_dynamic_candidates(
            hidden_states
        )
        tokens[:, -1] += 50
        probabilities[:, -1] = 0.15
        return tokens, probabilities

    def compute_hybrid_top5_h2_top5_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, probabilities = self.compute_stock_top10_dynamic_candidates(
            hidden_states
        )
        tokens[:, 5:] += 50
        probabilities[:, 5:] = 0.15
        return tokens, probabilities

    def compute_hybrid_union_top10_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        *,
        h2_to_h1_weight: float,
        return_union_provenance: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        self.union_weights.append(float(h2_to_h1_weight))
        tokens, probabilities = self.compute_stock_top10_dynamic_candidates(
            hidden_states
        )
        tokens[:, 8:] += 50
        probabilities[:, 8:] = 0.15 * h2_to_h1_weight
        if return_union_provenance:
            batch_size = int(hidden_states.shape[0])
            source_masks = torch.tensor(
                [[1] * 8 + [2] * 2] * batch_size,
                dtype=torch.long,
                device=hidden_states.device,
            )
            h1_ranks = torch.tensor(
                [list(range(1, 9)) + [0, 0]] * batch_size,
                dtype=torch.long,
                device=hidden_states.device,
            )
            h2_ranks = torch.tensor(
                [[0] * 8 + [1, 2]] * batch_size,
                dtype=torch.long,
                device=hidden_states.device,
            )
            score_heads = torch.tensor(
                [[0] * 8 + [1, 1]] * batch_size,
                dtype=torch.long,
                device=hidden_states.device,
            )
            return (
                tokens,
                probabilities,
                source_masks,
                h1_ranks,
                h2_ranks,
                score_heads,
            )
        return tokens, probabilities


def _state(state_id: int) -> _ResidualTreeDraftState:
    hidden = torch.tensor([float(state_id)], dtype=torch.bfloat16)
    return _ResidualTreeDraftState(
        proposal_hidden=hidden,
        transition_hidden=hidden.clone(),
    )


def _stock_proposer(
    *,
    node_budget: int,
    max_depth: int,
    trace_path: str | None = None,
) -> tuple[SpecDecodeBaseProposer, list[tuple[list[int], list[int]]]]:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.num_speculative_tokens = node_budget
    proposer.speculative_config = SimpleNamespace(
        residual_tree_max_depth=max_depth,
        residual_tree_batch_drafting=True,
        residual_tree_trace_path=trace_path,
        residual_tree_tree_policy="best_first",
    )
    proposer._residual_tree_training_feature_dir = None
    proposer._residual_tree_probability_trace_dir = None
    transition_batches: list[tuple[list[int], list[int]]] = []

    def transition_batch(model, states, token_ids):
        del model
        transition_batches.append(
            (
                [int(state.proposal_hidden[0].item()) for state in states],
                [int(token_id) for token_id in token_ids],
            )
        )
        return [_state(int(token_id)) for token_id in token_ids]

    proposer._residual_tree_transition_batch = transition_batch
    return proposer, transition_batches


def _children_by_parent(tree) -> dict[int, set[int]]:
    children: dict[int, set[int]] = {}
    for node_id in range(1, len(tree.node_token_ids)):
        parent_id = tree.parent_node_ids[node_id]
        children.setdefault(parent_id, set()).add(tree.node_token_ids[node_id])
    return children


def test_stock_top2_b2d1_builds_two_root_siblings_without_transition() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=2, max_depth=1)
    model = _StockTop2Model()

    trees = proposer._propose_stock_top2_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    tree = trees[0]
    assert tree.draft_token_ids == [10, 20]
    assert tree.parent_node_ids == [-1, 0, 0]
    assert tree.node_depths == [0, 1, 1]
    assert tree.num_proposal_rows == 0
    assert model.hidden_batches == [[0]]
    assert transition_batches == []


def test_stock_top2_model_api_does_not_require_residual_heads() -> None:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        residual_tree_candidate_selection="stock_top2"
    )
    proposer._can_use_internal_eagle_tree_transition = lambda: True

    proposer._validate_residual_tree_model_api(_StockTop2Model())

    with pytest.raises(NotImplementedError, match="Stock EAGLE H1 top-2 hook"):
        proposer._validate_residual_tree_model_api(object())


def test_stock_top2_b6d2_batches_both_depth_one_transitions() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=6, max_depth=2)
    model = _StockTop2Model()

    trees = proposer._propose_stock_top2_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    tree = trees[0]
    assert len(tree.draft_token_ids) == 6
    assert tree.node_depths.count(1) == 2
    assert tree.node_depths.count(2) == 4
    assert tree.num_proposal_rows == 0
    root_children = {
        tree.node_token_ids[node_id]
        for node_id in range(1, len(tree.node_token_ids))
        if tree.parent_node_ids[node_id] == 0
    }
    assert root_children == {10, 20}
    children = _children_by_parent(tree)
    node_by_token = {
        token_id: node_id for node_id, token_id in enumerate(tree.node_token_ids)
    }
    assert children[node_by_token[10]] == {11, 12}
    assert children[node_by_token[20]] == {21, 22}
    assert transition_batches == [([0, 0], [10, 20])]
    assert model.hidden_batches == [[0], [10, 20]]


@pytest.mark.parametrize(
    ("node_budget", "max_depth"),
    [(1, 1), (3, 1), (7, 2), (15, 3)],
)
def test_stock_top2_runtime_rejects_infeasible_shapes(
    node_budget: int,
    max_depth: int,
) -> None:
    proposer, _ = _stock_proposer(
        node_budget=node_budget,
        max_depth=max_depth,
    )

    with pytest.raises(ValueError, match="fit the configured binary-tree depth"):
        proposer._propose_stock_top2_trees(
            _StockTop2Model(),
            [_state(0)],
            runtime_config={},
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )


def test_stock_top2_pruned_tree_uses_real_h1_probabilities() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=10, max_depth=3)
    model = _StockTop2Model()

    trees = proposer._propose_stock_top2_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    assert len(trees[0].draft_token_ids) == 10
    assert max(trees[0].node_depths) <= 3
    assert model.scored_batches
    assert transition_batches


def test_stock_top2_runtime_rejects_trace() -> None:
    proposer, _ = _stock_proposer(
        node_budget=2,
        max_depth=1,
        trace_path="/tmp/stock-top2-trace.jsonl",
    )

    with pytest.raises(ValueError, match="does not support diagnostic traces"):
        proposer._propose_stock_top2_trees(
            _StockTop2Model(),
            [_state(0)],
            runtime_config={},
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )


def test_stock_top2_runtime_rejects_non_greedy_requests() -> None:
    proposer, _ = _stock_proposer(node_budget=2, max_depth=1)

    with pytest.raises(NotImplementedError, match="greedy verification only"):
        proposer._propose_stock_top2_trees(
            _StockTop2Model(),
            [_state(0)],
            runtime_config={},
            sampling_metadata=SimpleNamespace(all_greedy=False),
        )


def test_stock_top2_runtime_rejects_breadth_first_policy() -> None:
    proposer, _ = _stock_proposer(node_budget=6, max_depth=2)

    with pytest.raises(ValueError, match="requires best_first"):
        proposer._propose_stock_top2_trees(
            _StockTop2Model(),
            [_state(0)],
            runtime_config={"tree_policy": "breadth_first"},
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )


def test_stock_dynamic_builds_official_width10_b60d8_shape() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=60, max_depth=8)
    proposer.speculative_config.residual_tree_tree_policy = "eagle3_dynamic"
    model = _StockTop2Model()

    trees = proposer._propose_stock_dynamic_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    assert len(trees[0].draft_token_ids) == 60
    assert len(trees[0].node_token_ids) == 61
    assert max(trees[0].node_depths) <= 8
    assert [len(tokens) for _, tokens in transition_batches] == [10] * 7


def test_stock_dynamic_builds_width9_b60d8_shape() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=60, max_depth=8)
    proposer.speculative_config.residual_tree_tree_policy = "eagle3_dynamic"
    proposer.speculative_config.residual_tree_candidate_selection = (
        "stock_top9_dynamic"
    )
    model = _StockTop2Model()

    trees = proposer._propose_stock_dynamic_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    assert len(trees[0].draft_token_ids) == 60
    assert len(trees[0].node_token_ids) == 61
    assert max(trees[0].node_depths) <= 8
    assert [len(tokens) for _, tokens in transition_batches] == [9] * 7


def test_hybrid_dynamic_keeps_width10_b60d8_shape() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=60, max_depth=8)
    proposer.speculative_config.residual_tree_tree_policy = "eagle3_dynamic"
    proposer.speculative_config.residual_tree_scorer_mode = "lambda_q"
    proposer.speculative_config.residual_tree_head_lambdas = [0.4, 0.2]
    model = _StockTop2Model()

    trees = proposer._propose_hybrid_dynamic_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    assert len(trees[0].draft_token_ids) == 60
    assert len(trees[0].node_token_ids) == 61
    assert max(trees[0].node_depths) <= 8
    assert [len(tokens) for _, tokens in transition_batches] == [10] * 7


def test_hybrid_union_keeps_width10_b60d8_and_applies_weight_once() -> None:
    proposer, transition_batches = _stock_proposer(node_budget=60, max_depth=8)
    proposer.speculative_config.residual_tree_tree_policy = "eagle3_dynamic"
    proposer.speculative_config.residual_tree_scorer_mode = "lambda_q"
    proposer.speculative_config.residual_tree_head_lambdas = [0.4, 0.2]
    proposer.speculative_config.residual_tree_candidate_selection = (
        "hybrid_union_top10_dynamic"
    )
    model = _StockTop2Model()

    trees = proposer._propose_hybrid_dynamic_trees(
        model,
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert len(trees) == 1
    assert len(trees[0].draft_token_ids) == 60
    assert len(trees[0].node_token_ids) == 61
    assert max(trees[0].node_depths) <= 8
    assert [len(tokens) for _, tokens in transition_batches] == [10] * 7
    assert model.union_weights == [0.5] * 8


def test_hybrid_union_trace_records_final_b60_sources(tmp_path) -> None:
    trace_path = tmp_path / "union.jsonl"
    proposer, _ = _stock_proposer(
        node_budget=60,
        max_depth=8,
        trace_path=str(trace_path),
    )
    proposer.speculative_config.residual_tree_tree_policy = "eagle3_dynamic"
    proposer.speculative_config.residual_tree_scorer_mode = "lambda_q"
    proposer.speculative_config.residual_tree_head_lambdas = [0.4, 0.2]
    proposer.speculative_config.residual_tree_candidate_selection = (
        "hybrid_union_top10_dynamic"
    )
    proposer._residual_tree_trace_step = 0

    proposer._propose_hybrid_dynamic_trees(
        _StockTop2Model(),
        [_state(0)],
        runtime_config={},
        sampling_metadata=SimpleNamespace(all_greedy=True),
        request_ids=["request-1"],
    )

    event = json.loads(trace_path.read_text().splitlines()[0])
    provenance = event["dynamic_provenance"]
    assert event["request_id"] == "request-1"
    assert provenance["union_source_schema"] == "h1_h2_top10_union_v1"
    assert len(provenance["union_final_nodes"]) == 60
    assert {row["source_mask"] for row in provenance["union_final_nodes"]} == {
        1,
        2,
    }
