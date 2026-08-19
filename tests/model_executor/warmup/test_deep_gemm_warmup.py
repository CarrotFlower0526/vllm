# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import torch

from vllm.model_executor.warmup.deep_gemm_warmup import (
    _fp8_linear_may_use_deep_gemm,
)


def test_non_fp8_module_does_not_query_deep_gemm_backend() -> None:
    module = torch.nn.Linear(4, 4)
    with patch(
        "vllm.model_executor.warmup.deep_gemm_warmup."
        "get_mk_alignment_for_contiguous_layout",
        side_effect=RuntimeError("DeepGEMM is unavailable"),
    ):
        assert not _fp8_linear_may_use_deep_gemm(module)
