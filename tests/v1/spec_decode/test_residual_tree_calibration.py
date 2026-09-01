# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import torch

from vllm.v1.spec_decode.residual_tree_calibration import (
    OrderedHeadIsotonicCalibration,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_ordered_head_isotonic_calibration_applies_frozen_tables(tmp_path) -> None:
    score_02, score_04, score_08 = (
        float(torch.tensor(value, dtype=torch.float32))
        for value in (0.2, 0.4, 0.8)
    )
    tables = []
    for head in range(1, 11):
        table = {
            "head": head,
            "sample_count": 3,
            "positive_count": 1,
            "zero_weight_sample_count": 0,
            "total_weight": 3.0,
            "positive_weight": 1.0,
            "observed_minimum_score": score_02,
            "observed_maximum_score": score_08,
            "endpoint_rule": "nearest",
            "tie_rule": "aggregate",
            "outside_probability_domain": "reject",
            "blocks": [
                {
                    "minimum_score": score_02,
                    "maximum_score": score_04,
                    "sample_count": 2,
                    "positive_count": 0,
                    "total_weight": 2.0,
                    "positive_weight": 0.0,
                    "estimate": 0.1,
                },
                {
                    "minimum_score": score_08,
                    "maximum_score": score_08,
                    "sample_count": 1,
                    "positive_count": 1,
                    "total_weight": 1.0,
                    "positive_weight": 1.0,
                    "estimate": 0.9,
                },
            ],
        }
        table["mapping_sha256"] = _sha(table)
        tables.append(table)
    path = tmp_path / "calibrator.json"
    path.write_text(
        json.dumps(
            {
                "schema": "qwen3_32b_greedy_correction_isotonic_calibrator_v1",
                "tables": tables,
            }
        )
    )

    calibration = OrderedHeadIsotonicCalibration(path)
    values = torch.tensor([[0.0, 0.4, 0.4001, 1.0] * 2 + [0.2, 0.8]])
    output = calibration(values)

    torch.testing.assert_close(
        output,
        torch.tensor([[0.1, 0.1, 0.9, 0.9] * 2 + [0.1, 0.9]]),
    )
