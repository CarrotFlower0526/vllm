# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.residual_tree import (
    select_batched_eagle3_dynamic_trees,
)


def _candidate_batch(states, *, width: int):
    tokens = []
    probabilities = []
    for state in states:
        base = int(state) * 100 + 1
        tokens.append(list(range(base, base + width)))
        probabilities.append([0.4 / (index + 1) for index in range(width)])
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(probabilities, dtype=torch.float32),
    )


def _assert_ancestor_closed(tree) -> None:
    for node in tree.nodes[1:]:
        assert 0 <= node.parent_id < node.node_id
        assert tree.nodes[node.parent_id].depth + 1 == node.depth


def test_stock_dynamic_expands_ten_frontier_states_per_depth() -> None:
    transition_widths = []

    def transition(states, token_ids):
        assert len(states) == len(token_ids)
        transition_widths.append(len(states))
        return list(token_ids)

    root_tokens, root_probabilities = _candidate_batch([0], width=10)
    trees = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=lambda states: _candidate_batch(states, width=10),
        transition_batch_fn=transition,
        head_lambdas=(1.0,) * 10,
        node_budget=60,
        max_depth=8,
        frontier_width=10,
    )

    assert transition_widths == [10] * 7
    assert len(trees) == 1
    assert len(trees[0].nodes) == 61
    assert max(node.depth for node in trees[0].nodes) <= 8
    _assert_ancestor_closed(trees[0])


def test_residual_dynamic_uses_same_beam_pruner_with_two_heads() -> None:
    transition_widths = []

    def transition(states, token_ids):
        assert len(states) == len(token_ids)
        transition_widths.append(len(states))
        return list(token_ids)

    root_tokens, root_probabilities = _candidate_batch([0], width=2)
    trees = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=lambda states: _candidate_batch(states, width=2),
        transition_batch_fn=transition,
        head_lambdas=(0.6, 0.3),
        node_budget=10,
        max_depth=3,
        frontier_width=10,
    )

    assert transition_widths == [2, 4]
    assert len(trees[0].nodes) == 11
    assert max(node.depth for node in trees[0].nodes) == 3
    assert {node.contributors[0].head_id for node in trees[0].nodes[1:]} == {0, 1}
    _assert_ancestor_closed(trees[0])


def test_dynamic_head_top1_merges_duplicate_tokens_before_pruning() -> None:
    tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=torch.tensor([[7, 7, 8]]),
        root_candidate_probabilities=torch.tensor([[0.4, 0.3, 0.2]]),
        candidate_batch_fn=lambda states: _candidate_batch(states, width=3),
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(0.5, 0.25, 1.0),
        node_budget=2,
        max_depth=1,
        candidate_selection="head_top1",
    )[0]

    assert [node.token_id for node in tree.nodes[1:]] == [7, 8]
    merged = tree.nodes[1]
    assert [item.head_id for item in merged.contributors] == [0, 1]
    assert merged.priority == pytest.approx(0.4 * 0.5 + 0.3 * 0.25)
    assert tree.children[0] == [1, 2]


def test_dynamic_global_pruning_admits_a_high_score_child_with_its_ancestor() -> None:
    def duplicate_children(states):
        return (
            torch.tensor(
                [[9, 9] if int(state) == 1 else [10, 10] for state in states]
            ),
            torch.ones((len(states), 2)),
        )

    tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=torch.tensor([[1, 2]]),
        root_candidate_probabilities=torch.tensor([[0.2, 0.19]]),
        candidate_batch_fn=duplicate_children,
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=2,
        max_depth=2,
        frontier_width=2,
        collect_dynamic_provenance=True,
        candidate_selection="head_top1",
    )[0]

    assert [node.token_id for node in tree.nodes[1:]] == [1, 9]
    assert tree.nodes[2].priority == pytest.approx(0.4)
    assert [item.head_id for item in tree.nodes[2].contributors] == [0, 1]
    assert tree.dynamic_provenance is not None
    assert tree.dynamic_provenance["generated_by_depth"][1] == {
        "depth": 2,
        "total": 2,
        "contributor_total": 4,
        "merged_node_count": 2,
        "head_counts": [
            {"head_id": 0, "count": 2},
            {"head_id": 1, "count": 2},
        ],
    }
    assert tree.dynamic_provenance["final_tree_by_depth"][1] == {
        "depth": 2,
        "total": 1,
        "contributor_total": 2,
        "merged_node_count": 1,
        "head_counts": [
            {"head_id": 0, "count": 1},
            {"head_id": 1, "count": 1},
        ],
    }
    _assert_ancestor_closed(tree)


def test_dynamic_depth_head_lambdas_change_only_existing_priority_weight() -> None:
    root_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    root_probabilities = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
    common = {
        "root_states": [0],
        "root_candidate_tokens": root_tokens,
        "root_candidate_probabilities": root_probabilities,
        "candidate_batch_fn": lambda states: _candidate_batch(states, width=2),
        "transition_batch_fn": lambda states, token_ids: list(token_ids),
        "head_lambdas": (1.0, 1.0),
        "node_budget": 1,
        "max_depth": 1,
        "frontier_width": 1,
    }

    raw_tree = select_batched_eagle3_dynamic_trees(**common)[0]
    calibrated_tree = select_batched_eagle3_dynamic_trees(
        **common,
        depth_head_lambdas=((0.1, 1.0),),
    )[0]

    assert raw_tree.nodes[1].token_id == 1
    assert calibrated_tree.nodes[1].token_id == 2
    assert calibrated_tree.nodes[1].contributors[0].lambda_weight == 1.0


def test_dynamic_failure_probability_scores_ordered_distinct_heads() -> None:
    common = {
        "root_states": [0],
        "root_candidate_tokens": torch.tensor([[11, 12, 13]]),
        "root_candidate_probabilities": torch.tensor([[0.4, 0.39, 0.9]]),
        "candidate_batch_fn": lambda states: _candidate_batch(states, width=3),
        "transition_batch_fn": lambda states, token_ids: list(token_ids),
        "head_lambdas": (1.0, 1.0, 1.0),
        "node_budget": 1,
        "max_depth": 1,
        "candidate_selection": "distinct_head_top1",
    }

    raw = select_batched_eagle3_dynamic_trees(**common)[0]
    failure = select_batched_eagle3_dynamic_trees(
        **common, scorer_mode="failure_probability"
    )[0]

    assert raw.nodes[1].token_id == 13
    assert failure.nodes[1].token_id == 11
    assert failure.nodes[1].priority == pytest.approx(0.4)


def test_dynamic_failure_probability_requires_conditioned_distinct_candidates() -> None:
    with pytest.raises(
        ValueError,
        match="requires distinct_head_top1",
    ):
        select_batched_eagle3_dynamic_trees(
            root_states=[0],
            root_candidate_tokens=torch.tensor([[11, 12]]),
            root_candidate_probabilities=torch.tensor([[0.4, 0.3]]),
            candidate_batch_fn=lambda states: _candidate_batch(states, width=2),
            transition_batch_fn=lambda states, token_ids: list(token_ids),
            head_lambdas=(1.0, 1.0),
            node_budget=1,
            max_depth=1,
            candidate_selection="head_top1",
            scorer_mode="failure_probability",
        )


def test_dynamic_scorers_share_candidates_and_raw_proposal_probabilities() -> None:
    original_tokens = torch.tensor([[11, 12, 13]])
    original_probabilities = torch.tensor([[0.60, 0.59, 0.58]])
    ranking_values = {
        "calibrated_chain": torch.tensor([[0.10, 0.90, 0.00]]),
        "same_candidate_oracle": torch.tensor([[0.00, 0.00, 1.00]]),
    }
    selected = {}

    for scorer_mode in (
        "lambda_q",
        "failure_probability",
        "calibrated_chain",
        "same_candidate_oracle",
    ):
        candidate_tokens = original_tokens.clone()
        candidate_probabilities = original_probabilities.clone()
        seen = {}

        def ranking(states, tokens, probabilities):
            seen["tokens"] = tokens.clone()
            seen["probabilities"] = probabilities.clone()
            return ranking_values[scorer_mode].expand_as(probabilities)

        tree = select_batched_eagle3_dynamic_trees(
            root_states=[0],
            root_candidate_tokens=candidate_tokens,
            root_candidate_probabilities=candidate_probabilities,
            candidate_batch_fn=lambda states: _candidate_batch(states, width=3),
            candidate_ranking_batch_fn=(
                ranking if scorer_mode in ranking_values else None
            ),
            transition_batch_fn=lambda states, token_ids: list(token_ids),
            head_lambdas=(1.0, 1.0, 1.0),
            node_budget=1,
            max_depth=1,
            candidate_selection="distinct_head_top1",
            scorer_mode=scorer_mode,
        )[0]

        torch.testing.assert_close(candidate_tokens, original_tokens)
        torch.testing.assert_close(candidate_probabilities, original_probabilities)
        if scorer_mode in ranking_values:
            torch.testing.assert_close(seen["tokens"], original_tokens)
            torch.testing.assert_close(
                seen["probabilities"], original_probabilities
            )
        node = tree.nodes[1]
        raw_index = original_tokens[0].tolist().index(node.token_id)
        assert node.contributors[0].proposal_prob == pytest.approx(
            original_probabilities[0, raw_index].item()
        )
        selected[scorer_mode] = node.token_id

    assert selected == {
        "lambda_q": 11,
        "failure_probability": 11,
        "calibrated_chain": 12,
        "same_candidate_oracle": 13,
    }


def test_dynamic_provenance_records_generated_frontier_and_final_counts() -> None:
    root_tokens, root_probabilities = _candidate_batch([0], width=2)
    trees = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=lambda states: _candidate_batch(states, width=2),
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(0.6, 0.3),
        node_budget=10,
        max_depth=3,
        frontier_width=10,
        collect_dynamic_provenance=True,
    )
    tree = trees[0]

    assert tree.dynamic_provenance == {
        "candidate_width": 2,
        "scorer_mode": "lambda_q",
        "frontier_width": 10,
        "frontier_h2_only_quota": 0,
        "preserve_spine_count": 0,
        "generated_by_depth": [
            {
                "depth": 1,
                "total": 2,
                "contributor_total": 2,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 1},
                    {"head_id": 1, "count": 1},
                ],
            },
            {
                "depth": 2,
                "total": 4,
                "contributor_total": 4,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 2},
                    {"head_id": 1, "count": 2},
                ],
            },
            {
                "depth": 3,
                "total": 8,
                "contributor_total": 8,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 4},
                    {"head_id": 1, "count": 4},
                ],
            },
        ],
        "continued_frontier_by_depth": [
            {
                "depth": 1,
                "total": 2,
                "contributor_total": 2,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 1},
                    {"head_id": 1, "count": 1},
                ],
            },
            {
                "depth": 2,
                "total": 4,
                "contributor_total": 4,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 2},
                    {"head_id": 1, "count": 2},
                ],
            },
        ],
        "final_tree_by_depth": [
            {
                "depth": 1,
                "total": 2,
                "contributor_total": 2,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 1},
                    {"head_id": 1, "count": 1},
                ],
            },
            {
                "depth": 2,
                "total": 4,
                "contributor_total": 4,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 2},
                    {"head_id": 1, "count": 2},
                ],
            },
            {
                "depth": 3,
                "total": 4,
                "contributor_total": 4,
                "merged_node_count": 0,
                "head_counts": [
                    {"head_id": 0, "count": 3},
                    {"head_id": 1, "count": 1},
                ],
            },
        ],
    }


def test_dynamic_union_provenance_tracks_final_node_sources() -> None:
    def candidates(states):
        tokens, probabilities = _candidate_batch(states, width=2)
        batch_size = len(states)
        return (
            tokens,
            probabilities,
            torch.tensor([[1, 2]] * batch_size),
            torch.tensor([[1, 0]] * batch_size),
            torch.tensor([[0, 1]] * batch_size),
            torch.tensor([[0, 1]] * batch_size),
        )

    root = candidates([0])
    tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root[0],
        root_candidate_probabilities=root[1],
        root_candidate_provenance=root[2:],
        candidate_batch_fn=candidates,
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=10,
        max_depth=3,
        frontier_width=10,
        collect_dynamic_provenance=True,
    )[0]

    provenance = tree.dynamic_provenance
    assert provenance is not None
    assert provenance["union_source_schema"] == "h1_h2_top10_union_v1"
    assert len(provenance["union_final_nodes"]) == 10
    assert (
        sum(
            row["count"]
            for depth in provenance["union_final_tree_by_depth"]
            for row in depth["source_counts"]
        )
        == 10
    )
    assert {row["source_mask"] for row in provenance["union_final_nodes"]} == {
        1,
        2,
    }


def test_canonical_target_path_diagnostic_separates_pruning_stages() -> None:
    root_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    root_probabilities = torch.tensor([[0.6, 0.4]], dtype=torch.float32)

    local_tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=lambda states: _candidate_batch(states, width=2),
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=1,
        max_depth=1,
        frontier_width=1,
        collect_dynamic_provenance=True,
        diagnostic_target_paths=[[3]],
    )[0]
    assert local_tree.dynamic_provenance is not None
    assert (
        local_tree.dynamic_provenance["canonical_target_path"]["stop_stage"]
        == "absent_after_local_width10"
    )
    local_probe = local_tree.dynamic_provenance[
        "canonical_spine_candidate_states"
    ]
    assert (
        local_tree.dynamic_provenance["canonical_spine_candidate_schema"]
        == "ordered_distinct_heads_same_process_v1"
    )
    assert len(local_probe) == 1
    assert local_probe[0]["source"] == "raw_tree_expansion"
    assert local_probe[0]["correct_token_available"] is False
    assert [row["token_id"] for row in local_probe[0]["candidates"]] == [1, 2]
    assert (
        local_tree.dynamic_provenance["canonical_target_path"]
        ["fixed_candidate_oracle_path_count"]
        == 0
    )

    budget_tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=lambda states: _candidate_batch(states, width=2),
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=1,
        max_depth=1,
        frontier_width=1,
        collect_dynamic_provenance=True,
        diagnostic_target_paths=[[2]],
    )[0]
    assert budget_tree.dynamic_provenance is not None
    budget_diagnostic = budget_tree.dynamic_provenance["canonical_target_path"]
    assert budget_diagnostic["stop_stage"] == "final_node_budget_pruned"
    assert budget_diagnostic["stop_details"]["candidate_global_priority_rank"] == 2
    assert budget_diagnostic["fixed_candidate_oracle_path_count"] == 1
    assert budget_diagnostic["fixed_candidate_oracle_token_ids"] == [2]

    def second_depth_candidates(states):
        return (
            torch.tensor([[3, 4]] * len(states), dtype=torch.long),
            torch.tensor([[0.1, 0.05]] * len(states), dtype=torch.float32),
        )

    frontier_tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=second_depth_candidates,
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=2,
        max_depth=2,
        frontier_width=1,
        collect_dynamic_provenance=True,
        diagnostic_target_paths=[[2, 3]],
    )[0]
    assert frontier_tree.dynamic_provenance is not None
    frontier_diagnostic = frontier_tree.dynamic_provenance["canonical_target_path"]
    assert frontier_diagnostic["retained_path_token_ids"] == [2]
    assert frontier_diagnostic["stop_stage"] == "continued_frontier_pruned"
    assert frontier_diagnostic["stop_details"]["parent_frontier_rank"] == 2
    assert frontier_diagnostic["fixed_candidate_oracle_path_count"] == 2
    assert frontier_diagnostic["fixed_candidate_oracle_token_ids"] == [2, 3]
    frontier_probe = frontier_tree.dynamic_provenance[
        "canonical_spine_candidate_states"
    ]
    assert [row["source"] for row in frontier_probe] == [
        "raw_tree_expansion",
        "canonical_spine_probe",
    ]
    assert [row["parent_path_token_ids"] for row in frontier_probe] == [
        [],
        [2],
    ]
    assert [row["correct_head_id"] for row in frontier_probe] == [1, 0]


def test_dynamic_h2_only_quota_keeps_a_residual_frontier_path() -> None:
    def candidates(states):
        batch_size = len(states)
        return (
            torch.tensor([[3, 4, 5]] * batch_size, dtype=torch.long),
            torch.tensor([[0.6, 0.3, 0.1]] * batch_size),
            torch.tensor([[1, 1, 2]] * batch_size),
            torch.tensor([[1, 2, 0]] * batch_size),
            torch.tensor([[0, 0, 1]] * batch_size),
            torch.tensor([[0, 0, 1]] * batch_size),
        )

    root = candidates([0])
    tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root[0],
        root_candidate_probabilities=root[1],
        root_candidate_provenance=root[2:],
        candidate_batch_fn=candidates,
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0, 1.0),
        node_budget=6,
        max_depth=2,
        frontier_width=2,
        collect_dynamic_provenance=True,
        diagnostic_target_paths=[[5, 5]],
        frontier_h2_only_quota=1,
    )[0]

    assert tree.dynamic_provenance is not None
    assert tree.dynamic_provenance["frontier_h2_only_quota"] == 1
    diagnostic = tree.dynamic_provenance["canonical_target_path"]
    assert diagnostic["retained_path_token_ids"] == [5]
    assert diagnostic["stop_stage"] == "final_node_budget_pruned"


def test_dynamic_preserved_spine_survives_global_node_pruning() -> None:
    def candidates(states):
        token_rows = []
        for state in states:
            token = int(state)
            token_rows.append([token + 2, token + 3])
        return (
            torch.tensor(token_rows, dtype=torch.long),
            torch.tensor([[0.1, 0.05]] * len(states)),
        )

    tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=torch.tensor([[1, 2]], dtype=torch.long),
        root_candidate_probabilities=torch.tensor([[0.9, 0.8]]),
        candidate_batch_fn=candidates,
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=3,
        max_depth=3,
        frontier_width=1,
        collect_dynamic_provenance=True,
        diagnostic_target_paths=[[1, 3, 5]],
        preserve_spine_count=1,
    )[0]

    assert tree.dynamic_provenance is not None
    assert tree.dynamic_provenance["preserve_spine_count"] == 1
    assert len(tree.dynamic_provenance["preserved_spines"][0]) == 3
    diagnostic = tree.dynamic_provenance["canonical_target_path"]
    assert diagnostic["retained_path_token_ids"] == [1, 3, 5]
    assert max(node.depth for node in tree.nodes) == 3
    _assert_ancestor_closed(tree)
