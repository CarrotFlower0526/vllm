# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.v1.spec_decode.residual_tree_numeric_trace import (
    NUMERIC_TRACE_SCHEMA,
    append_target_numeric_trace,
    build_target_numeric_trace_event,
)
from vllm.v1.spec_decode.residual_tree_trace import (
    ROOT_VERIFIER_PROB_TRACE_SCHEMA,
    TRAINING_FEATURE_TRACE_SCHEMA,
    append_residual_tree_root_verifier_probabilities,
    append_residual_tree_training_features,
)


def _trace_args():
    hidden = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.bfloat16,
    )
    logits = torch.tensor(
        [
            [0.0, 3.0, 2.0, 1.0],
            [4.0, 2.0, 3.0, 1.0],
            [0.0, 1.0, 5.0, 4.0],
        ],
        dtype=torch.bfloat16,
    )
    return {
        "step": 4,
        "mode": "tree",
        "request_ids": ["request-a", "request-b"],
        "root_row_indices": torch.tensor([0, 2]),
        "sample_hidden_states": hidden,
        "logits": logits,
        "sample_input_token_ids": torch.tensor([10, 11, 12]),
        "sample_positions": torch.tensor([[20], [21], [22]]),
        "top_k": 3,
    }


def test_build_target_numeric_trace_hashes_exact_bf16_rows():
    event = build_target_numeric_trace_event(**_trace_args())

    assert event["schema"] == NUMERIC_TRACE_SCHEMA
    assert event["step"] == 4
    assert event["mode"] == "tree"
    assert [record["sample_row_index"] for record in event["records"]] == [0, 2]
    assert [record["input_token_id"] for record in event["records"]] == [10, 12]
    assert [record["position"] for record in event["records"]] == [[20], [22]]
    assert event["records"][0]["top_token_ids"] == [1, 2, 3]
    assert event["records"][1]["top_token_ids"] == [2, 3, 1]
    assert len(event["records"][0]["hidden_sha256"]) == 64
    assert len(event["records"][0]["logits_sha256"]) == 64

    changed = _trace_args()
    changed["sample_hidden_states"] = changed["sample_hidden_states"].clone()
    changed["sample_hidden_states"][0, 0] = 2.0
    changed_event = build_target_numeric_trace_event(**changed)
    assert (
        event["records"][0]["hidden_sha256"]
        != changed_event["records"][0]["hidden_sha256"]
    )


def test_append_target_numeric_trace_is_jsonl(tmp_path):
    destination = tmp_path / "numeric.jsonl"
    append_target_numeric_trace(destination, **_trace_args())
    append_target_numeric_trace(destination, **_trace_args())

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["schema"] == NUMERIC_TRACE_SCHEMA for row in rows)


def test_append_training_features_writes_atomic_bf16_shard(tmp_path):
    destination = append_residual_tree_training_features(
        tmp_path,
        step=3,
        request_ids=["a", "b"],
        proposal_hidden=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        positions=torch.tensor([10, 11]),
    )

    payload = torch.load(destination, map_location="cpu", weights_only=False)
    assert payload["schema"] == TRAINING_FEATURE_TRACE_SCHEMA
    assert payload["proposal_hidden"].dtype == torch.bfloat16
    assert payload["proposal_hidden"].shape == (2, 4)
    assert payload["positions"].tolist() == [10, 11]
    assert payload["trace_ids"] == [
        f"{payload['pid']}:3:0",
        f"{payload['pid']}:3:1",
    ]


def test_append_root_verifier_probabilities_writes_draft_support(tmp_path):
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.5, 3.5, 2.5, 0.5, 4.5],
        ],
        dtype=torch.bfloat16,
    )
    root_rows = torch.tensor([0, 2])
    support_ids = torch.tensor([4, 1, 3])
    destination = append_residual_tree_root_verifier_probabilities(
        tmp_path,
        step=7,
        request_ids=["request-a", "request-b"],
        trace_ids=["worker:12:0", "worker:12:1"],
        root_row_indices=root_rows,
        logits=logits,
        draft_target_ids=support_ids,
    )

    payload = torch.load(destination, map_location="cpu", weights_only=False)
    expected_root_logits = logits.index_select(0, root_rows)
    expected_logsumexp = torch.logsumexp(expected_root_logits.float(), dim=-1)
    expected_support_logits = expected_root_logits.index_select(1, support_ids)

    assert payload["schema"] == ROOT_VERIFIER_PROB_TRACE_SCHEMA
    assert payload["step"] == 7
    assert payload["request_ids"] == ["request-a", "request-b"]
    assert payload["trace_ids"] == ["worker:12:0", "worker:12:1"]
    assert payload["root_row_indices"].tolist() == [0, 2]
    assert payload["target_vocab_size"] == 5
    assert payload["draft_vocab_size"] == 3
    assert payload["draft_target_ids"].tolist() == [4, 1, 3]
    assert payload["verifier_support_logits"].dtype == torch.bfloat16
    assert torch.equal(
        payload["verifier_support_logits"],
        expected_support_logits,
    )
    assert torch.equal(
        payload["verifier_full_vocab_logsumexp"],
        expected_logsumexp,
    )
    assert torch.allclose(
        payload["verifier_support_probs"],
        torch.exp(expected_support_logits.float() - expected_logsumexp[:, None]),
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_root_verifier_probabilities_reject_duplicate_support(tmp_path):
    with pytest.raises(ValueError, match="must be unique"):
        append_residual_tree_root_verifier_probabilities(
            tmp_path,
            step=0,
            request_ids=["request-a"],
            trace_ids=["worker:0:0"],
            root_row_indices=torch.tensor([0]),
            logits=torch.zeros((1, 4)),
            draft_target_ids=torch.tensor([1, 1]),
        )


def test_target_numeric_trace_rejects_bad_root_count():
    args = _trace_args()
    args["root_row_indices"] = torch.tensor([0])
    with pytest.raises(ValueError, match="root rows must match"):
        build_target_numeric_trace_event(**args)
