# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in target-model numeric tracing for residual-tree correctness work.

This module is deliberately separate from the normal residual-tree event
trace.  Copying full-vocabulary logits to the CPU introduces a synchronization
point and is therefore enabled only when the caller sets
``VLLM_RESIDUAL_TREE_NUMERIC_TRACE_PATH``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

NUMERIC_TRACE_ENV = "VLLM_RESIDUAL_TREE_NUMERIC_TRACE_PATH"
NUMERIC_TRACE_SCHEMA = "vllm_residual_tree_numeric_trace_v1"


def _tensor_row_sha256(row: torch.Tensor) -> str:
    """Hash the exact tensor bytes without requiring NumPy BF16 support."""

    raw = row.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _row_values(
    values: torch.Tensor | None,
    row_index: int,
) -> int | list[int] | None:
    if values is None:
        return None
    row = values[row_index].detach().cpu()
    if row.ndim == 0:
        return int(row.item())
    return [int(value) for value in row.reshape(-1).tolist()]


def build_target_numeric_trace_event(
    *,
    step: int,
    mode: str,
    request_ids: Sequence[str],
    root_row_indices: torch.Tensor,
    sample_hidden_states: torch.Tensor,
    logits: torch.Tensor,
    sample_input_token_ids: torch.Tensor | None = None,
    sample_positions: torch.Tensor | None = None,
    top_k: int = 8,
    event: str = "target_root_numeric",
    record_metadata: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one exact-hash diagnostic event for selected sampled rows."""

    if root_row_indices.ndim != 1:
        raise ValueError("root_row_indices must be 1-D")
    if root_row_indices.numel() != len(request_ids):
        raise ValueError("root rows must match request_ids")
    if record_metadata is not None and len(record_metadata) != len(request_ids):
        raise ValueError("record_metadata must match request_ids")
    if sample_hidden_states.shape[0] != logits.shape[0]:
        raise ValueError("hidden-state and logits row counts must match")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    root_rows = root_row_indices.detach().to(device="cpu", dtype=torch.long).tolist()
    records: list[dict[str, Any]] = []
    actual_top_k = min(top_k, int(logits.shape[-1]))
    for record_index, (request_id, row_index) in enumerate(
        zip(request_ids, root_rows, strict=True)
    ):
        if row_index < 0 or row_index >= logits.shape[0]:
            raise ValueError("root row is outside sampled logits")
        row_logits = logits[row_index]
        top_values, top_ids = torch.topk(
            row_logits,
            k=actual_top_k,
            largest=True,
            sorted=True,
        )
        record = {
            "request_id": str(request_id),
            "sample_row_index": int(row_index),
            "input_token_id": _row_values(sample_input_token_ids, row_index),
            "position": _row_values(sample_positions, row_index),
            "hidden_dtype": str(sample_hidden_states.dtype),
            "hidden_shape": list(sample_hidden_states[row_index].shape),
            "hidden_sha256": _tensor_row_sha256(sample_hidden_states[row_index]),
            "logits_dtype": str(row_logits.dtype),
            "logits_shape": list(row_logits.shape),
            "logits_sha256": _tensor_row_sha256(row_logits),
            "top_token_ids": [
                int(token_id) for token_id in top_ids.detach().cpu().tolist()
            ],
            "top_logits": [
                float(value) for value in top_values.detach().float().cpu().tolist()
            ],
        }
        if record_metadata is not None:
            overlap = record.keys() & record_metadata[record_index].keys()
            if overlap:
                raise ValueError(
                    "record_metadata overwrites reserved fields: "
                    + ", ".join(sorted(overlap))
                )
            record.update(record_metadata[record_index])
        records.append(record)

    return {
        "schema": NUMERIC_TRACE_SCHEMA,
        "event": str(event),
        "step": int(step),
        "mode": str(mode),
        "records": records,
    }


def append_target_numeric_trace(
    path: str | os.PathLike[str],
    **event_args: Any,
) -> None:
    """Build and append one complete numeric-trace JSON event."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_target_numeric_trace_event(**event_args)
    line = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError(
                f"short residual-tree numeric trace write: {written} != {len(line)}"
            )
    finally:
        os.close(descriptor)


__all__ = [
    "NUMERIC_TRACE_ENV",
    "NUMERIC_TRACE_SCHEMA",
    "append_target_numeric_trace",
    "build_target_numeric_trace_event",
]
