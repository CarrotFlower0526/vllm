# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Residual-tree event tracing and atomic diagnostic sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

TRACE_SCHEMA = "vllm_residual_tree_step_trace_v1"
TRAINING_FEATURE_TRACE_ENV = "RESIDUAL_STACK_VLLM_ROOT_FEATURE_DIR"
TRAINING_FEATURE_TRACE_SCHEMA = "vllm_residual_tree_training_features_v1"
ROOT_VERIFIER_PROB_TRACE_ENV = "RESIDUAL_STACK_VLLM_ROOT_VERIFIER_PROB_DIR"
ROOT_VERIFIER_PROB_TRACE_SCHEMA = "vllm_residual_tree_root_verifier_probabilities_v1"


def append_residual_tree_trace(
    path: str | os.PathLike[str],
    event: dict[str, Any],
) -> None:
    """Append one complete JSON event with process-safe O_APPEND semantics."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": TRACE_SCHEMA, **event}
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
            raise OSError(f"short residual-tree trace write: {written} != {len(line)}")
    finally:
        os.close(descriptor)


def append_residual_tree_training_features(
    directory: str | os.PathLike[str],
    *,
    step: int,
    request_ids: list[str],
    proposal_hidden: torch.Tensor,
    positions: torch.Tensor,
) -> Path:
    """Atomically persist root proposal features for on-policy H2 training.

    This opt-in trace is intentionally binary and separate from the JSON event
    trace.  Serializing 5K-wide BF16 hidden rows as JSON would dominate both
    serving time and disk usage.  ``trace_ids`` use the exact identity scheme
    of the paired selection/verification trace so an offline builder can join
    verifier labels without putting labels or model inputs in this file.
    """

    if proposal_hidden.ndim != 2:
        raise ValueError("proposal_hidden must have shape [batch, hidden_size]")
    if positions.ndim != 1:
        raise ValueError("positions must have shape [batch]")
    batch_size = int(proposal_hidden.shape[0])
    if len(request_ids) != batch_size or int(positions.shape[0]) != batch_size:
        raise ValueError("training feature rows must match request_ids and positions")

    destination_dir = Path(directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    destination = destination_dir / f"features-{pid}-{int(step):08d}.pt"
    temporary = destination_dir / f".{destination.name}.tmp"
    payload = {
        "schema": TRAINING_FEATURE_TRACE_SCHEMA,
        "step": int(step),
        "pid": int(pid),
        "request_ids": [str(request_id) for request_id in request_ids],
        "trace_ids": [
            f"{pid}:{int(step)}:{request_index}" for request_index in range(batch_size)
        ],
        "positions": positions.detach().to(device="cpu", dtype=torch.int64),
        "proposal_hidden": proposal_hidden.detach().to(
            device="cpu", dtype=torch.bfloat16
        ),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def append_residual_tree_root_verifier_probabilities(
    directory: str | os.PathLike[str],
    *,
    step: int,
    request_ids: list[str],
    trace_ids: list[str],
    root_row_indices: torch.Tensor,
    logits: torch.Tensor,
    draft_target_ids: torch.Tensor,
) -> Path:
    """Persist root target probabilities normalized over the full vocabulary.

    The support probabilities intentionally sum to the target mass inside the
    draft vocabulary, not one. The saved full-vocabulary log-normalizer makes
    the support logits independently replayable without retaining every target
    logit.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [sample_rows, target_vocab]")
    if root_row_indices.ndim != 1:
        raise ValueError("root_row_indices must be 1-D")
    if draft_target_ids.ndim != 1:
        raise ValueError("draft_target_ids must be 1-D")
    batch_size = int(root_row_indices.shape[0])
    if len(request_ids) != batch_size or len(trace_ids) != batch_size:
        raise ValueError("root probability rows must match request and trace ids")
    if any(not trace_id for trace_id in trace_ids):
        raise ValueError("root probability rows require nonempty trace ids")
    if batch_size == 0:
        raise ValueError("root probability capture requires at least one row")

    root_indices = root_row_indices.detach().to(
        device=logits.device,
        dtype=torch.long,
    )
    if bool(((root_indices < 0) | (root_indices >= logits.shape[0])).any().item()):
        raise ValueError("root row is outside sampled logits")
    support_ids = draft_target_ids.detach().to(
        device=logits.device,
        dtype=torch.long,
    )
    target_vocab_size = int(logits.shape[1])
    if support_ids.numel() == 0:
        raise ValueError("draft target support must be nonempty")
    if bool(((support_ids < 0) | (support_ids >= target_vocab_size)).any().item()):
        raise ValueError("draft target id is outside the target vocabulary")
    if int(torch.unique(support_ids).numel()) != int(support_ids.numel()):
        raise ValueError("draft target ids must be unique")

    root_logits = logits.index_select(0, root_indices)
    full_vocab_logsumexp = torch.logsumexp(root_logits.float(), dim=-1)
    support_logits = root_logits.index_select(1, support_ids)
    support_probs = torch.exp(
        support_logits.float() - full_vocab_logsumexp.unsqueeze(1)
    )

    destination_dir = Path(directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    destination = destination_dir / (f"verifier-probs-{pid}-{int(step):08d}.pt")
    temporary = destination_dir / f".{destination.name}.tmp"
    payload = {
        "schema": ROOT_VERIFIER_PROB_TRACE_SCHEMA,
        "step": int(step),
        "pid": int(pid),
        "request_ids": [str(request_id) for request_id in request_ids],
        "trace_ids": [str(trace_id) for trace_id in trace_ids],
        "root_row_indices": root_indices.to(device="cpu", dtype=torch.int64),
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": int(support_ids.numel()),
        "draft_target_ids": support_ids.to(device="cpu", dtype=torch.int64),
        "verifier_full_vocab_logsumexp": full_vocab_logsumexp.to(
            device="cpu", dtype=torch.float32
        ),
        "verifier_support_logits": support_logits.detach().to(device="cpu"),
        "verifier_support_probs": support_probs.to(device="cpu", dtype=torch.float32),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


__all__ = [
    "ROOT_VERIFIER_PROB_TRACE_ENV",
    "ROOT_VERIFIER_PROB_TRACE_SCHEMA",
    "TRACE_SCHEMA",
    "TRAINING_FEATURE_TRACE_ENV",
    "TRAINING_FEATURE_TRACE_SCHEMA",
    "append_residual_tree_root_verifier_probabilities",
    "append_residual_tree_trace",
    "append_residual_tree_training_features",
]
