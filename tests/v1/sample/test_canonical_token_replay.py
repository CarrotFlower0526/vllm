# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.canonical_token_replay import (
    CanonicalTokenReplayConfig,
    CanonicalTokenReplayLogitsProcessor,
    parse_canonical_token_replay,
)
from vllm.v1.spec_decode.metadata import (
    SpecDecodeMetadata,
    TreeSpecDecodeMetadata,
)
from vllm.v1.spec_decode.tree_schema import DraftTokenTree


def _config(
    token_ids: list[int],
    *,
    replay_id: str = "request-0",
    single_token_per_step: bool = True,
    proposal_only: bool = False,
    tree_construction_oracle: bool = False,
    trace_path: str | None = None,
) -> CanonicalTokenReplayConfig:
    return CanonicalTokenReplayConfig(
        token_ids=tuple(token_ids),
        replay_id=replay_id,
        single_token_per_step=single_token_per_step,
        proposal_only=proposal_only,
        tree_construction_oracle=tree_construction_oracle,
        trace_path=trace_path,
    )


def _assert_only_token_is_finite(
    logits: torch.Tensor,
    row: int,
    token_id: int,
    expected_value: float,
) -> None:
    finite = torch.isfinite(logits[row]).nonzero().flatten().tolist()
    assert finite == [token_id]
    assert logits[row, token_id].item() == expected_value


def test_parse_canonical_token_replay_schema():
    params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        extra_args={
            "canonical_token_replay": {
                "token_ids": [4, 5, 6],
                "replay_id": "prompt-7",
                "single_token_per_step": True,
                "proposal_only": True,
                "trace_path": "/tmp/replay.jsonl",
            }
        },
    )

    config = parse_canonical_token_replay(params)

    assert config == _config(
        [4, 5, 6],
        replay_id="prompt-7",
        proposal_only=True,
        trace_path="/tmp/replay.jsonl",
    )


def test_proposal_only_requires_single_token_steps():
    params = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        extra_args={
            "canonical_token_replay": {
                "token_ids": [4, 5],
                "replay_id": "prompt-7",
                "single_token_per_step": False,
                "proposal_only": True,
            }
        },
    )

    with pytest.raises(ValueError, match="single_token_per_step"):
        parse_canonical_token_replay(params)


def test_tree_construction_oracle_requires_proposal_only():
    params = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        extra_args={
            "canonical_token_replay": {
                "token_ids": [4, 5],
                "replay_id": "prompt-7",
                "single_token_per_step": True,
                "tree_construction_oracle": True,
            }
        },
    )

    with pytest.raises(ValueError, match="proposal_only"):
        parse_canonical_token_replay(params)


def _linear_and_tree_verifier_metadata():
    linear = SpecDecodeMetadata.make_dummy(
        [[4]],
        device=torch.device("cpu"),
    )
    tree = TreeSpecDecodeMetadata(
        node_token_ids=torch.tensor([-1, 4], dtype=torch.int32),
        parent_node_ids=torch.tensor([-1, 0], dtype=torch.int32),
        node_request_indices=torch.tensor([0, 0], dtype=torch.int32),
        node_depths=torch.tensor([0, 1], dtype=torch.int32),
        node_priorities=torch.tensor([1.0, 0.5], dtype=torch.float32),
        root_node_ids=torch.tensor([0], dtype=torch.int32),
        num_draft_tokens=[1],
        num_proposal_rows=[1],
        cu_num_tree_nodes=torch.tensor([2], dtype=torch.int32),
        child_start_indices=torch.tensor([0, 1], dtype=torch.int32),
        child_end_indices=torch.tensor([1, 1], dtype=torch.int32),
        child_node_ids=torch.tensor([1], dtype=torch.int32),
        contributor_child_node_ids=torch.tensor([1], dtype=torch.int32),
        contributor_proposal_rows=torch.tensor([0], dtype=torch.int32),
        contributor_head_ids=torch.tensor([0], dtype=torch.int32),
        contributor_scores=torch.tensor([0.5], dtype=torch.float32),
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1], dtype=torch.int32),
    )
    return linear, tree


def test_proposal_only_rejects_linear_and_tree_verifier_metadata():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([4, 5], proposal_only=True)],
        [[]],
    )

    processor.validate_plain_target_step(None)
    for metadata in _linear_and_tree_verifier_metadata():
        with pytest.raises(
            RuntimeError,
            match=(
                "canonical proposal-only replay forbids verifier execution"
            ),
        ):
            processor.validate_plain_target_step(metadata)


def test_plain_and_linear_bonus_masks_use_output_offsets():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([4, 2, 3, 1]), None],
        [[99], []],
    )
    original = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    plain = processor.apply_plain(original.clone())
    _assert_only_token_is_finite(plain, 0, 2, 2.0)
    assert torch.equal(plain[1], original[1])

    bonus = processor.apply_plain(
        original.clone(),
        predict_bonus_token=True,
        spec_token_ids=[[80, 81], []],
    )
    _assert_only_token_is_finite(bonus, 0, 1, 1.0)
    assert torch.equal(bonus[1], original[1])


def test_forced_token_remains_selectable_when_input_score_is_nonfinite():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([2])],
        [[]],
    )
    logits = torch.full((1, 4), float("-inf"))

    masked = processor.apply_plain(logits)

    _assert_only_token_is_finite(masked, 0, 2, 0.0)
    assert masked.argmax(dim=-1).tolist() == [2]


def test_final_reference_root_leaves_out_of_range_bonus_unmasked():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([3, 2])],
        [[99]],
    )
    original = torch.arange(5, dtype=torch.float32).reshape(1, 5)

    bonus = processor.apply_plain(
        original.clone(),
        predict_bonus_token=True,
        spec_token_ids=[[80]],
    )

    assert torch.equal(bonus, original)


def test_linear_target_mask_uses_local_draft_position():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([0, 3, 4, 5]), _config([1, 2, 0, 4], replay_id="r1")],
        [[90], [91, 92]],
    )
    original = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    masked = processor.apply_linear(original.clone(), [2, 1])

    _assert_only_token_is_finite(masked, 0, 3, 3.0)
    _assert_only_token_is_finite(masked, 1, 4, 9.0)
    _assert_only_token_is_finite(masked, 2, 0, 10.0)


def test_final_reference_root_masks_linear_root_but_not_deeper_row():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([3, 2])],
        [[99]],
    )
    original = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    masked = processor.apply_linear(original.clone(), [2])

    _assert_only_token_is_finite(masked, 0, 2, 2.0)
    assert torch.equal(masked[1], original[1])


def test_tree_mask_uses_node_depth_and_leaves_unconfigured_rows_unchanged():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([0, 2, 3, 4]), None],
        [[90], []],
    )
    original = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    masked = processor.apply_tree(
        original.clone(),
        node_request_indices=torch.tensor([0, 0, 0, 1]),
        node_depths=torch.tensor([0, 1, 2, 0]),
    )

    _assert_only_token_is_finite(masked, 0, 2, 2.0)
    _assert_only_token_is_finite(masked, 1, 3, 8.0)
    _assert_only_token_is_finite(masked, 2, 4, 14.0)
    assert torch.equal(masked[3], original[3])


def test_final_reference_root_masks_tree_root_but_not_deeper_node():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([3, 2])],
        [[99]],
    )
    original = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    masked = processor.apply_tree(
        original.clone(),
        node_request_indices=torch.tensor([0, 0]),
        node_depths=torch.tensor([0, 1]),
    )

    _assert_only_token_is_finite(masked, 0, 2, 2.0)
    assert torch.equal(masked[1], original[1])


def test_single_token_per_step_truncates_linear_and_tree_would_acceptance():
    processor = CanonicalTokenReplayLogitsProcessor(
        [
            _config([1, 2, 3]),
            _config([4, 3, 2], replay_id="r1", single_token_per_step=False),
        ],
        [[], []],
    )
    linear = torch.tensor([[1, 2, 3], [4, 3, 2]], dtype=torch.int32)

    processor.truncate_linear_output(linear, placeholder_token_id=-1)

    assert linear.tolist() == [[1, -1, -1], [4, 3, 2]]

    tree = torch.tensor([[1, 2, 3], [4, 3, 2]], dtype=torch.int32)
    accepted = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    processor.truncate_tree_output(tree, accepted, placeholder_token_id=-1)

    assert tree.tolist() == [[1, -1, -1], [4, 3, 2]]
    assert accepted.tolist() == [[-1, -1], [2, 1]]


def test_tree_trace_records_root_coverage_before_single_token_truncation(tmp_path):
    trace_path = tmp_path / "canonical_replay.jsonl"
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([7, 8], trace_path=str(trace_path))],
        [[]],
    )
    metadata = SimpleNamespace(
        root_node_ids=torch.tensor([0]),
        node_token_ids=torch.tensor([-1, 5, 7]),
        child_start_indices=torch.tensor([0, 2, 2]),
        child_end_indices=torch.tensor([2, 2, 2]),
        child_node_ids=torch.tensor([1, 2]),
        contributor_child_node_ids=torch.tensor([1, 2]),
        contributor_proposal_rows=torch.tensor([0, 1]),
        contributor_head_ids=torch.tensor([0, 1]),
        contributor_scores=torch.tensor([0.6, 0.4]),
    )
    sampled = torch.tensor([[7, 8, -1]], dtype=torch.int32)
    accepted = torch.tensor([[2, -1]], dtype=torch.int32)

    processor.trace_tree(metadata, sampled, accepted)
    processor.truncate_tree_output(sampled, accepted, placeholder_token_id=-1)

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["replay_id"] == "request-0"
    assert record["output_offset"] == 0
    assert record["canonical_token_id"] == 7
    assert record["mode"] == "tree_speculative"
    assert record["would_accept"] is True
    assert record["would_accept_draft_count"] == 1
    assert record["would_sampled_token_ids"] == [7, 8]
    assert record["returned_token_ids"] == [7]
    assert record["h1_hit"] is False
    assert record["h2_extra_hit"] is True
    assert record["root_children"][0]["contributor_head_ids"] == [0]
    assert record["root_children"][1]["contributor_head_ids"] == [1]
    assert sampled.tolist() == [[7, -1, -1]]
    assert accepted.tolist() == [[-1, -1]]


def test_linear_trace_emits_one_record_for_each_verifier_root(tmp_path):
    trace_path = tmp_path / "linear_replay.jsonl"
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([3, 4], trace_path=str(trace_path))],
        [[]],
    )
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([3, 9]),
        num_draft_tokens=[2],
    )
    sampled = torch.tensor([[3, 4, -1]], dtype=torch.int32)

    processor.trace_linear(metadata, sampled)

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["output_offset"] == 0
    assert records[0]["draft_token_ids"] == [3, 9]
    assert records[0]["would_accept"] is True
    assert records[0]["would_accept_draft_count"] == 1


def test_proposal_only_plain_trace_emits_only_position_zero(tmp_path):
    trace_path = tmp_path / "proposal_only.jsonl"
    outputs: list[list[int]] = [[]]
    processor = CanonicalTokenReplayLogitsProcessor(
        [
            _config(
                [7, 8],
                proposal_only=True,
                trace_path=str(trace_path),
            )
        ],
        outputs,
    )

    processor.trace_plain(torch.tensor([7], dtype=torch.int32))
    outputs[0].append(7)
    processor.trace_plain(torch.tensor([8], dtype=torch.int32))

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["mode"] == "plain_greedy"
    assert records[0]["output_offset"] == 0
    assert records[0]["canonical_token_id"] == 7


def test_proposal_only_linear_trace_targets_next_position(tmp_path):
    trace_path = tmp_path / "proposal_only_linear.jsonl"
    processor = CanonicalTokenReplayLogitsProcessor(
        [
            _config(
                [7, 8],
                proposal_only=True,
                trace_path=str(trace_path),
            )
        ],
        [[]],
    )

    assert processor.proposal_only_batch() is True
    assert processor.proposal_only_final_step() is False
    processor.trace_proposal_only_linear([[8]], ["req-0"])

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["mode"] == "proposal_only_linear"
    assert record["target_execution"] == "plain_no_spec"
    assert record["output_offset"] == 1
    assert record["canonical_token_id"] == 8
    assert record["h1_hit"] is True
    assert record["h2_hit"] is False
    assert record["root_children"][0]["contributor_head_ids"] == [0]


def test_proposal_only_tree_trace_records_h2_extra_for_next_position(tmp_path):
    trace_path = tmp_path / "proposal_only_tree.jsonl"
    processor = CanonicalTokenReplayLogitsProcessor(
        [
            _config(
                [7, 9],
                proposal_only=True,
                trace_path=str(trace_path),
            )
        ],
        [[]],
    )
    tree = DraftTokenTree(
        node_token_ids=[-1, 8, 9],
        parent_node_ids=[-1, 0, 0],
        node_depths=[0, 1, 1],
        node_priorities=[1.0, 0.6, 0.4],
        child_start_indices=[0, 2, 2],
        child_end_indices=[2, 2, 2],
        child_node_ids=[1, 2],
        contributor_child_node_ids=[1, 2],
        contributor_proposal_rows=[0, 1],
        contributor_head_ids=[0, 1],
        contributor_scores=[0.6, 0.4],
    )

    processor.trace_proposal_only_trees([tree], ["req-0"])

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["mode"] == "proposal_only_tree"
    assert record["output_offset"] == 1
    assert record["h1_hit"] is False
    assert record["h2_hit"] is True
    assert record["h2_extra_hit"] is True
    assert record["root_children"][1]["contributor_head_ids"] == [1]


def test_tree_oracle_trace_walks_only_the_canonical_path(tmp_path):
    trace_path = tmp_path / "proposal_only_tree_oracle.jsonl"
    processor = CanonicalTokenReplayLogitsProcessor(
        [
            _config(
                [7, 9, 10, 12],
                proposal_only=True,
                tree_construction_oracle=True,
                trace_path=str(trace_path),
            )
        ],
        [[]],
    )
    tree = DraftTokenTree(
        node_token_ids=[-1, 8, 9, 10, 11, 13],
        parent_node_ids=[-1, 0, 0, 2, 2, 3],
        node_depths=[0, 1, 1, 2, 2, 3],
        node_priorities=[1.0, 0.7, 1.0, 1.0, 0.5, 0.4],
        child_start_indices=[0, 2, 2, 4, 5, 5],
        child_end_indices=[2, 2, 4, 5, 5, 5],
        child_node_ids=[1, 2, 3, 4, 5],
        contributor_child_node_ids=[1, 2, 3, 4, 5],
        contributor_proposal_rows=[0, 1, 2, 3, 4],
        contributor_head_ids=[0, 1, 0, 1, 0],
        contributor_scores=[0.7, 1.0, 1.0, 0.5, 0.4],
    )

    assert processor.tree_construction_oracle_batch() is True
    processor.trace_proposal_only_trees([tree], ["req-0"])

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["tree_construction_oracle"] is True
    assert record["would_accept_tree_path_count"] == 2
    assert record["would_accept_tree_path_token_ids"] == [9, 10]


def test_proposal_only_final_step_skips_next_proposal():
    processor = CanonicalTokenReplayLogitsProcessor(
        [_config([7, 8], proposal_only=True)],
        [[7]],
    )

    assert processor.proposal_only_final_step() is True
