# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


@dataclass
class TreeSpecDecodeMetadata:
    """Metadata for SpecInfer-style tree verification.

    Tree nodes are flattened across the batch. Each request has one root node
    with parent -1 and token id -1. All other nodes are draft-token nodes.
    `target_logits_indices` must point to target logits for every tree node,
    including leaves, because a rejected path samples its fallback token from
    the current node's residual target distribution.
    """

    # [num_tree_nodes], roots use -1.
    node_token_ids: torch.Tensor
    # [num_tree_nodes], roots use -1.
    parent_node_ids: torch.Tensor
    # [num_tree_nodes]
    node_request_indices: torch.Tensor
    # [num_tree_nodes]
    node_depths: torch.Tensor
    # [num_tree_nodes]
    node_priorities: torch.Tensor
    # [num_reqs]
    root_node_ids: torch.Tensor
    # [num_reqs]
    num_draft_tokens: list[int]
    # [num_reqs]
    num_proposal_rows: list[int]
    # [num_reqs]
    cu_num_tree_nodes: torch.Tensor
    # [num_tree_nodes]
    child_start_indices: torch.Tensor
    # [num_tree_nodes]
    child_end_indices: torch.Tensor
    # [num_child_edges]
    child_node_ids: torch.Tensor
    # [num_contributors]
    contributor_child_node_ids: torch.Tensor
    # [num_contributors]
    contributor_proposal_rows: torch.Tensor
    # [num_contributors]
    contributor_head_ids: torch.Tensor
    # [num_contributors]
    contributor_scores: torch.Tensor
    # [num_tree_nodes]
    target_logits_indices: torch.Tensor
    # [num_logits]
    logits_indices: torch.Tensor
    # [num_actual_tokens, max_query_len], optional full-batch tree mask.
    tree_attn_mask: torch.Tensor | None = None
    # CPU [num_actual_tokens], parent local query index for each flattened row.
    tree_parent_local_indices_cpu: torch.Tensor | None = None
    # CPU [num_actual_tokens], tree depth/local linear depth for each row.
    tree_node_depths_cpu: torch.Tensor | None = None
    # Actual target-model inputs corresponding to the flattened tree nodes.
    # The root token is not part of DraftTokenTree, so these optional fields
    # are populated by GPUModelRunner after it has prepared the real input
    # rows. They are retained for correctness traces only.
    input_token_ids: torch.Tensor | None = None
    # [num_tree_nodes, num_position_dims]. Text-only positions use one column;
    # M-RoPE/XD-RoPE inputs may use more than one.
    input_positions: torch.Tensor | None = None
    # Physical cache locations are deliberately separate from logical
    # positions: siblings share a logical position but must write different
    # attention/state slots. Columns correspond to cache groups of that type.
    input_attention_slots: torch.Tensor | None = None
    input_mamba_state_blocks: torch.Tensor | None = None
    input_attention_group_indices: list[int] | None = None
    input_mamba_group_indices: list[int] | None = None
    # Stable proposer-generated ids used only to join optional JSONL traces.
    trace_ids: list[str | None] | None = None
    # CPU scalar used to specialize greedy verification by actual tree depth,
    # rather than by the usually much larger draft-node budget.
    max_tree_depth: int | None = None

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens, default=0)
        if self.max_tree_depth is not None:
            self.max_tree_depth = int(self.max_tree_depth)
            if not 0 <= self.max_tree_depth <= self.max_spec_len:
                raise ValueError("max_tree_depth must be in [0, max_spec_len]")
        if len(self.num_proposal_rows) != len(self.num_draft_tokens):
            raise ValueError("num_proposal_rows must match num_draft_tokens")
        if self.node_token_ids.ndim != 1:
            raise ValueError("node_token_ids must be 1-D")
        if self.parent_node_ids.shape != self.node_token_ids.shape:
            raise ValueError("parent_node_ids must match node_token_ids")
        if self.node_request_indices.shape != self.node_token_ids.shape:
            raise ValueError("node_request_indices must match node_token_ids")
        if self.node_depths.shape != self.node_token_ids.shape:
            raise ValueError("node_depths must match node_token_ids")
        if self.node_priorities.shape != self.node_token_ids.shape:
            raise ValueError("node_priorities must match node_token_ids")
        if self.target_logits_indices.shape != self.node_token_ids.shape:
            raise ValueError("target_logits_indices must match node_token_ids")
        if self.child_start_indices.shape != self.node_token_ids.shape:
            raise ValueError("child_start_indices must match node_token_ids")
        if self.child_end_indices.shape != self.node_token_ids.shape:
            raise ValueError("child_end_indices must match node_token_ids")
        if (
            self.input_token_ids is not None
            and self.input_token_ids.shape != self.node_token_ids.shape
        ):
            raise ValueError("input_token_ids must match node_token_ids")
        if self.input_positions is not None and (
            self.input_positions.ndim != 2
            or self.input_positions.shape[0] != self.node_token_ids.shape[0]
        ):
            raise ValueError(
                "input_positions must have one row per tree node"
            )
        for name, values in (
            ("input_attention_slots", self.input_attention_slots),
            ("input_mamba_state_blocks", self.input_mamba_state_blocks),
        ):
            if values is not None and (
                values.ndim != 2
                or values.shape[0] != self.node_token_ids.shape[0]
            ):
                raise ValueError(f"{name} must have one row per tree node")
        for name, indices, values in (
            (
                "input_attention_group_indices",
                self.input_attention_group_indices,
                self.input_attention_slots,
            ),
            (
                "input_mamba_group_indices",
                self.input_mamba_group_indices,
                self.input_mamba_state_blocks,
            ),
        ):
            if indices is not None and (
                values is None or len(indices) != values.shape[1]
            ):
                raise ValueError(f"{name} must match its cache-slot columns")
        if self.trace_ids is None:
            self.trace_ids = [None] * len(self.num_draft_tokens)
        elif len(self.trace_ids) != len(self.num_draft_tokens):
            raise ValueError("trace_ids must match num_draft_tokens")

    @property
    def num_tree_nodes(self) -> int:
        return int(self.node_token_ids.shape[0])

    @property
    def num_reqs(self) -> int:
        return len(self.num_draft_tokens)
