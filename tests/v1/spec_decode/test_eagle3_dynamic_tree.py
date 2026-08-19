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
        "frontier_width": 10,
        "frontier_h2_only_quota": 0,
        "preserve_spine_count": 0,
        "final_boundary_exchange": {"mode": "none", "exchanged": False},
        "generated_by_depth": [
            {
                "depth": 1,
                "total": 2,
                "head_counts": [
                    {"head_id": 0, "count": 1},
                    {"head_id": 1, "count": 1},
                ],
            },
            {
                "depth": 2,
                "total": 4,
                "head_counts": [
                    {"head_id": 0, "count": 2},
                    {"head_id": 1, "count": 2},
                ],
            },
            {
                "depth": 3,
                "total": 8,
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
                "head_counts": [
                    {"head_id": 0, "count": 1},
                    {"head_id": 1, "count": 1},
                ],
            },
            {
                "depth": 2,
                "total": 4,
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
                "head_counts": [
                    {"head_id": 0, "count": 1},
                    {"head_id": 1, "count": 1},
                ],
            },
            {
                "depth": 2,
                "total": 4,
                "head_counts": [
                    {"head_id": 0, "count": 2},
                    {"head_id": 1, "count": 2},
                ],
            },
            {
                "depth": 3,
                "total": 4,
                "head_counts": [
                    {"head_id": 0, "count": 3},
                    {"head_id": 1, "count": 1},
                ],
            },
        ],
    }


def test_parent_supported_boundary_exchange_swaps_only_one_leaf() -> None:
    def candidates(states):
        token_rows = []
        probability_rows = []
        for state in states:
            if int(state) == 1:
                token_rows.append([3, 4])
                probability_rows.append([0.5, 0.32])
            else:
                token_rows.append([5, 6])
                probability_rows.append([0.5, 0.1])
        return (
            torch.tensor(token_rows, dtype=torch.long),
            torch.tensor(probability_rows, dtype=torch.float32),
        )

    common = {
        "root_states": [0],
        "root_candidate_tokens": torch.tensor([[1, 2]], dtype=torch.long),
        "root_candidate_probabilities": torch.tensor([[0.9, 0.6]]),
        "candidate_batch_fn": candidates,
        "transition_batch_fn": lambda states, token_ids: list(token_ids),
        "head_lambdas": (1.0, 1.0),
        "node_budget": 4,
        "max_depth": 2,
        "frontier_width": 2,
        "collect_dynamic_provenance": True,
        "diagnostic_target_paths": [[1, 4]],
    }

    raw_tree = select_batched_eagle3_dynamic_trees(**common)[0]
    exchanged_tree = select_batched_eagle3_dynamic_trees(
        **common,
        final_boundary_exchange_mode=("parent_supported_half_cutoff_one_swap_v1"),
    )[0]

    assert [node.token_id for node in raw_tree.nodes[1:]] == [1, 2, 3, 5]
    assert [node.token_id for node in exchanged_tree.nodes[1:]] == [1, 2, 3, 4]
    _assert_ancestor_closed(exchanged_tree)
    provenance = exchanged_tree.dynamic_provenance
    assert provenance is not None
    exchange = provenance["final_boundary_exchange"]
    assert exchange["exchanged"] is True
    assert exchange["raw_score_floor_ratio"] == 0.5
    assert exchange["eligible_candidate_count"] == 1
    assert exchange["victim"]["parent_path_priority"] == pytest.approx(0.6)
    assert exchange["promoted"]["parent_path_priority"] == pytest.approx(0.9)
    canonical = provenance["canonical_target_path"]
    assert canonical["raw_final_tree_retained_path_token_ids"] == [1]
    assert canonical["raw_final_tree_retained_path_count"] == 1
    assert canonical["retained_path_token_ids"] == [1, 4]
    assert canonical["retained_path_count"] == 2


def test_parent_supported_boundary_exchange_requires_stronger_parent() -> None:
    root_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    root_probabilities = torch.tensor([[0.9, 0.6]])

    tree = select_batched_eagle3_dynamic_trees(
        root_states=[0],
        root_candidate_tokens=root_tokens,
        root_candidate_probabilities=root_probabilities,
        candidate_batch_fn=lambda states: (
            torch.tensor([[3, 4]] * len(states), dtype=torch.long),
            torch.tensor([[0.5, 0.49]] * len(states)),
        ),
        transition_batch_fn=lambda states, token_ids: list(token_ids),
        head_lambdas=(1.0, 1.0),
        node_budget=3,
        max_depth=2,
        frontier_width=2,
        collect_dynamic_provenance=True,
        final_boundary_exchange_mode=("parent_supported_half_cutoff_one_swap_v1"),
    )[0]

    assert tree.dynamic_provenance is not None
    assert tree.dynamic_provenance["final_boundary_exchange"]["exchanged"] is False


def test_selectable_mass_is_traced_without_changing_raw_tree() -> None:
    def candidates(states):
        tokens, probabilities = _candidate_batch(states, width=2)
        masses = torch.tensor([[1.0, 0.25]] * len(states), dtype=torch.float32)
        return tokens, probabilities, masses

    root = candidates([0])
    common = {
        "root_states": [0],
        "root_candidate_tokens": root[0],
        "root_candidate_probabilities": root[1],
        "candidate_batch_fn": candidates,
        "transition_batch_fn": lambda states, token_ids: list(token_ids),
        "head_lambdas": (1.0, 1.0),
        "node_budget": 10,
        "max_depth": 3,
        "frontier_width": 10,
        "collect_dynamic_provenance": True,
        "diagnostic_target_paths": [[1, 101, 10101]],
    }
    raw_tree = select_batched_eagle3_dynamic_trees(
        **(common | {"candidate_batch_fn": lambda states: candidates(states)[:2]})
    )[0]
    mass_tree = select_batched_eagle3_dynamic_trees(
        **common,
        root_candidate_selectable_masses=root[2],
    )[0]

    assert [
        (node.parent_id, node.token_id, node.priority) for node in mass_tree.nodes
    ] == [(node.parent_id, node.token_id, node.priority) for node in raw_tree.nodes]
    provenance = mass_tree.dynamic_provenance
    assert provenance is not None
    pool = provenance["selectable_mass_candidate_pool"]
    assert pool["tree_scoring_changed"] is False
    assert pool["frontier_selection_changed"] is False
    columns = pool["columns"]
    assert len(columns["full_node_id"]) == pool["row_count"]
    assert columns["selectable_mass_estimate"][:2] == pytest.approx([1.0, 0.25])
    assert columns["absolute_edge_probability_estimate"][:2] == pytest.approx(
        [0.4, 0.05]
    )
    selected_pool_priorities = sorted(
        priority
        for priority, flags in zip(
            columns["raw_path_priority"], columns["flags"], strict=True
        )
        if flags & (1 << 5)
    )
    assert selected_pool_priorities == pytest.approx(
        sorted(node.priority for node in raw_tree.nodes[1:])
    )


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
    local_probe = local_tree.dynamic_provenance["canonical_spine_candidate_states"]
    assert (
        local_tree.dynamic_provenance["canonical_spine_candidate_schema"]
        == "ordered_distinct_heads_same_process_v1"
    )
    assert len(local_probe) == 1
    assert local_probe[0]["source"] == "raw_tree_expansion"
    assert local_probe[0]["correct_token_available"] is False
    assert [row["token_id"] for row in local_probe[0]["candidates"]] == [1, 2]
    assert (
        local_tree.dynamic_provenance["canonical_target_path"][
            "fixed_candidate_oracle_path_count"
        ]
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
