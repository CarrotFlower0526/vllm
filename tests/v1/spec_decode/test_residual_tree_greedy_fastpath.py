# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode import residual_tree as residual_tree_module
from vllm.v1.spec_decode.residual_tree import (
    ResidualTree,
    ResidualTreeNode,
    trees_to_metadata,
    verify_greedy_tree,
    verify_greedy_tree_batch_with_nodes,
)


def _root_only_tree() -> ResidualTree:
    return ResidualTree(
        nodes=[ResidualTreeNode(0, -1, -1, 0, 1.0)],
        children=[[]],
    )


def _one_child_tree(token_id: int) -> ResidualTree:
    return ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(1, 0, token_id, 1, 1.0),
        ],
        children=[[1], []],
    )


def _two_node_tree(
    first_token_id: int,
    second_token_id: int,
    *,
    chain: bool = False,
) -> ResidualTree:
    second_parent = 1 if chain else 0
    return ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(1, 0, first_token_id, 1, 0.7),
            ResidualTreeNode(
                2,
                second_parent,
                second_token_id,
                2 if chain else 1,
                0.3,
            ),
        ],
        children=[[1], [2], []] if chain else [[1, 2], [], []],
    )


def _three_node_chain() -> ResidualTree:
    return ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(1, 0, 10, 1, 0.8),
            ResidualTreeNode(2, 1, 20, 2, 0.6),
            ResidualTreeNode(3, 2, 30, 3, 0.4),
        ],
        children=[[1], [2], [3], []],
    )


def _b6d2_tree() -> ResidualTree:
    return ResidualTree(
        nodes=[
            ResidualTreeNode(0, -1, -1, 0, 1.0),
            ResidualTreeNode(1, 0, 10, 1, 0.8),
            ResidualTreeNode(2, 0, 20, 1, 0.7),
            ResidualTreeNode(3, 1, 30, 2, 0.6),
            ResidualTreeNode(4, 1, 40, 2, 0.5),
            ResidualTreeNode(5, 2, 50, 2, 0.4),
            ResidualTreeNode(6, 2, 60, 2, 0.3),
        ],
        children=[[1, 2], [3, 4], [5, 6], [], [], [], []],
    )


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


@pytest.mark.parametrize("device", _DEVICES)
def test_b2d1_fast_path_stays_on_device_and_preserves_child_order(
    device,
    monkeypatch,
):
    trees = [
        _two_node_tree(10, 20),
        _two_node_tree(10, 20),
        _two_node_tree(10, 20),
        _two_node_tree(7, 7),
    ]
    metadata, _ = trees_to_metadata(trees, device=device)
    targets = torch.tensor(
        [
            10,
            101,
            102,
            20,
            201,
            202,
            30,
            301,
            302,
            7,
            701,
            702,
        ],
        dtype=torch.int64,
        device=device,
    )

    def fail_host_reconstruction(*_args, **_kwargs):
        raise AssertionError("B2D1 fast path reconstructed metadata on the host")

    monkeypatch.setattr(
        residual_tree_module,
        "residual_tree_from_metadata",
        fail_host_reconstruction,
    )
    monkeypatch.setattr(
        residual_tree_module,
        "_tensor_to_int_list",
        fail_host_reconstruction,
    )

    output, accepted_nodes = verify_greedy_tree_batch_with_nodes(
        metadata,
        targets,
        placeholder_token_id=-9,
    )

    assert output.cpu().tolist() == [
        [10, 101, -9],
        [20, 202, -9],
        [30, -9, -9],
        [7, 701, -9],
    ]
    assert accepted_nodes.cpu().tolist() == [
        [1, -9],
        [2, -9],
        [-9, -9],
        [1, -9],
    ]


@pytest.mark.parametrize("device", _DEVICES)
def test_b2d1_fast_path_handles_chain_and_scheduler_truncations(
    device,
    monkeypatch,
):
    trees = [
        _two_node_tree(30, 40, chain=True),
        _one_child_tree(50),
        _root_only_tree(),
        _two_node_tree(60, 70),
    ]
    metadata, _ = trees_to_metadata(trees, device=device)
    targets = torch.tensor(
        [
            30,
            40,
            303,
            50,
            505,
            77,
            70,
            606,
            707,
        ],
        dtype=torch.int32,
        device=device,
    )

    def fail_host_reconstruction(*_args, **_kwargs):
        raise AssertionError("truncated B2D1 batch left the device fast path")

    monkeypatch.setattr(
        residual_tree_module,
        "residual_tree_from_metadata",
        fail_host_reconstruction,
    )

    output, accepted_nodes = verify_greedy_tree_batch_with_nodes(
        metadata,
        targets,
        placeholder_token_id=-5,
    )

    assert output.cpu().tolist() == [
        [30, 40, 303],
        [50, 505, -5],
        [77, -5, -5],
        [70, 707, -5],
    ]
    assert accepted_nodes.cpu().tolist() == [
        [1, 2],
        [1, -5],
        [-5, -5],
        [2, -5],
    ]


@pytest.mark.parametrize("device", _DEVICES)
def test_b2d1_fast_path_handles_fully_truncated_batches(device, monkeypatch):
    def fail_host_reconstruction(*_args, **_kwargs):
        raise AssertionError("fully truncated batch left the device fast path")

    monkeypatch.setattr(
        residual_tree_module,
        "residual_tree_from_metadata",
        fail_host_reconstruction,
    )

    root_metadata, _ = trees_to_metadata(
        [_root_only_tree(), _root_only_tree()],
        device=device,
    )
    root_output, root_accepted = verify_greedy_tree_batch_with_nodes(
        root_metadata,
        torch.tensor([81, 82], dtype=torch.int64, device=device),
    )
    assert root_output.cpu().tolist() == [[81], [82]]
    assert root_accepted.shape == (2, 0)

    one_metadata, _ = trees_to_metadata(
        [_one_child_tree(50), _root_only_tree()],
        device=device,
    )
    one_output, one_accepted = verify_greedy_tree_batch_with_nodes(
        one_metadata,
        torch.tensor([50, 501, 83], dtype=torch.int64, device=device),
    )
    assert one_output.cpu().tolist() == [[50, 501], [83, -1]]
    assert one_accepted.cpu().tolist() == [[1], [-1]]


def test_b2d1_fast_path_matches_scalar_verifier_for_random_batch():
    generator = torch.Generator().manual_seed(1234)
    trees: list[ResidualTree] = []
    target_batches: list[torch.Tensor] = []
    for request_index in range(128):
        tokens = torch.randint(0, 32, (2,), generator=generator)
        tree = _two_node_tree(int(tokens[0]), int(tokens[1]))
        trees.append(tree)

        target_ids = torch.randint(0, 32, (3,), generator=generator)
        outcome = request_index % 4
        if outcome == 0:
            target_ids[0] = tokens[0]
        elif outcome == 1:
            target_ids[0] = tokens[1]
        elif outcome == 2:
            target_ids[0] = 100
        else:
            # Duplicate candidates exercise the scalar verifier's first-child
            # tie rule, independent of the original random token ids.
            duplicate = int(tokens[0])
            tree = _two_node_tree(duplicate, duplicate)
            trees[-1] = tree
            target_ids[0] = duplicate
        target_batches.append(target_ids)

    metadata, _ = trees_to_metadata(trees, device="cpu")
    flat_targets = torch.cat(target_batches)
    output, accepted_nodes = verify_greedy_tree_batch_with_nodes(
        metadata,
        flat_targets,
    )

    expected_output = torch.full_like(output, -1)
    expected_accepted = torch.full_like(accepted_nodes, -1)
    for request_index, (tree, target_ids) in enumerate(zip(trees, target_batches)):
        result = verify_greedy_tree(tree, target_ids)
        expected_output[request_index, : len(result.token_ids)] = torch.tensor(
            result.token_ids,
            dtype=torch.int32,
        )
        expected_accepted[request_index, : len(result.accepted_node_ids)] = (
            torch.tensor(result.accepted_node_ids, dtype=torch.int32)
        )

    assert torch.equal(output, expected_output)
    assert torch.equal(accepted_nodes, expected_accepted)


def test_greedy_batch_larger_tree_uses_generic_fallback(monkeypatch):
    tree = _three_node_chain()
    metadata, _ = trees_to_metadata([tree], device="cpu")
    targets = torch.tensor([10, 20, 30, 99], dtype=torch.int64)
    original_reconstruct = residual_tree_module.residual_tree_from_metadata
    reconstructed_requests: list[int] = []

    def record_reconstruction(metadata_arg, request_index):
        reconstructed_requests.append(request_index)
        return original_reconstruct(metadata_arg, request_index)

    monkeypatch.setattr(
        residual_tree_module,
        "residual_tree_from_metadata",
        record_reconstruction,
    )

    output, accepted_nodes = verify_greedy_tree_batch_with_nodes(metadata, targets)

    assert reconstructed_requests == [0]
    assert output.tolist() == [[10, 20, 30, 99]]
    assert accepted_nodes.tolist() == [[1, 2, 3]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_greedy_batch_arbitrary_tree_stays_on_device(monkeypatch):
    trees = [
        _b6d2_tree(),
        _three_node_chain(),
        _two_node_tree(40, 50),
        _root_only_tree(),
    ]
    metadata, _ = trees_to_metadata(trees, device="cuda")
    targets_by_request = [
        torch.tensor([20, 101, 60, 301, 401, 501, 999]),
        torch.tensor([10, 20, 30, 99]),
        torch.tensor([50, 401, 501]),
        torch.tensor([77]),
    ]
    targets = torch.cat(targets_by_request).cuda()

    def fail_host_reconstruction(*_args, **_kwargs):
        raise AssertionError("arbitrary greedy tree left the CUDA verifier")

    monkeypatch.setattr(
        residual_tree_module,
        "residual_tree_from_metadata",
        fail_host_reconstruction,
    )
    monkeypatch.setattr(
        residual_tree_module,
        "_tensor_to_int_list",
        fail_host_reconstruction,
    )

    output, accepted_nodes = verify_greedy_tree_batch_with_nodes(
        metadata,
        targets,
    )
    assert output.cpu().tolist() == [
        [20, 60, 999, -1, -1, -1, -1],
        [10, 20, 30, 99, -1, -1, -1],
        [50, 501, -1, -1, -1, -1, -1],
        [77, -1, -1, -1, -1, -1, -1],
    ]
    assert accepted_nodes.cpu().tolist() == [
        [2, 6, -1, -1, -1, -1],
        [1, 2, 3, -1, -1, -1],
        [2, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]
