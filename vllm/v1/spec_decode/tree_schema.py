# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DraftTokenTree:
    """CPU-side topology for one request's speculative token tree."""

    node_token_ids: list[int]
    parent_node_ids: list[int]
    node_depths: list[int]
    node_priorities: list[float]
    child_start_indices: list[int]
    child_end_indices: list[int]
    child_node_ids: list[int]
    contributor_child_node_ids: list[int]
    contributor_proposal_rows: list[int]
    contributor_head_ids: list[int]
    contributor_scores: list[float]
    proposal_num_rows: int | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        num_nodes = len(self.node_token_ids)
        if num_nodes == 0:
            raise ValueError("a draft token tree must contain a root node")
        if self.parent_node_ids[0] != -1:
            raise ValueError("node 0 must be the root and use parent -1")
        if self.node_token_ids[0] != -1:
            raise ValueError("node 0 must be the root and use token id -1")
        for values_name, values in (
            ("parent_node_ids", self.parent_node_ids),
            ("node_depths", self.node_depths),
            ("node_priorities", self.node_priorities),
            ("child_start_indices", self.child_start_indices),
            ("child_end_indices", self.child_end_indices),
        ):
            if len(values) != num_nodes:
                raise ValueError(f"{values_name} must have one entry per node")
        for node_id, parent_id in enumerate(self.parent_node_ids):
            if parent_id >= node_id:
                raise ValueError("parents must point to earlier nodes")
        for start, end in zip(self.child_start_indices, self.child_end_indices):
            if start > end:
                raise ValueError("child start must be <= child end")
            if start < 0 or end > len(self.child_node_ids):
                raise ValueError("child range is out of bounds")
        contributor_lens = (
            len(self.contributor_child_node_ids),
            len(self.contributor_proposal_rows),
            len(self.contributor_head_ids),
            len(self.contributor_scores),
        )
        if len(set(contributor_lens)) != 1:
            raise ValueError("contributor fields must have the same length")
        inferred_rows = self._inferred_num_proposal_rows()
        if self.proposal_num_rows is not None:
            if self.proposal_num_rows < inferred_rows:
                raise ValueError(
                    "proposal_num_rows cannot be smaller than referenced rows"
                )
            if self.proposal_num_rows < 0:
                raise ValueError("proposal_num_rows must be non-negative")

    @property
    def num_tree_nodes(self) -> int:
        return len(self.node_token_ids)

    @property
    def num_draft_tokens(self) -> int:
        return self.num_tree_nodes - 1

    @property
    def draft_token_ids(self) -> list[int]:
        return self.node_token_ids[1:]

    @property
    def num_proposal_rows(self) -> int:
        if self.proposal_num_rows is not None:
            return self.proposal_num_rows
        return self._inferred_num_proposal_rows()

    def _inferred_num_proposal_rows(self) -> int:
        rows = [row for row in self.contributor_proposal_rows if row >= 0]
        if not rows:
            return 0
        return max(rows) + 1

    @classmethod
    def from_draft_tokens(cls, token_ids: list[int]) -> "DraftTokenTree":
        """Build a degenerate chain tree from legacy linear draft tokens."""

        node_token_ids = [-1, *token_ids]
        parent_node_ids = [-1, *range(len(token_ids))]
        node_depths = list(range(len(token_ids) + 1))
        node_priorities = [1.0 for _ in node_token_ids]
        child_node_ids = list(range(1, len(token_ids) + 1))
        child_start_indices: list[int] = []
        child_end_indices: list[int] = []
        for node_id in range(len(node_token_ids)):
            if node_id < len(token_ids):
                child_start_indices.append(node_id)
                child_end_indices.append(node_id + 1)
            else:
                child_start_indices.append(len(child_node_ids))
                child_end_indices.append(len(child_node_ids))
        return cls(
            node_token_ids=node_token_ids,
            parent_node_ids=parent_node_ids,
            node_depths=node_depths,
            node_priorities=node_priorities,
            child_start_indices=child_start_indices,
            child_end_indices=child_end_indices,
            child_node_ids=child_node_ids,
            contributor_child_node_ids=[],
            contributor_proposal_rows=[],
            contributor_head_ids=[],
            contributor_scores=[],
        )

    @classmethod
    def root_only(cls) -> "DraftTokenTree":
        return cls.from_draft_tokens([])

    def truncate(self, max_draft_tokens: int) -> "DraftTokenTree":
        """Return a prefix-closed tree with at most max_draft_tokens nodes."""

        max_nodes = max(1, int(max_draft_tokens) + 1)
        if max_nodes >= self.num_tree_nodes:
            return self

        keep = set(range(max_nodes))
        child_node_ids: list[int] = []
        child_start_indices: list[int] = []
        child_end_indices: list[int] = []
        for node_id in range(max_nodes):
            child_start_indices.append(len(child_node_ids))
            start = self.child_start_indices[node_id]
            end = self.child_end_indices[node_id]
            child_node_ids.extend(
                child_id for child_id in self.child_node_ids[start:end]
                if child_id in keep
            )
            child_end_indices.append(len(child_node_ids))

        contributor_child_node_ids: list[int] = []
        contributor_proposal_rows: list[int] = []
        contributor_head_ids: list[int] = []
        contributor_scores: list[float] = []
        for child_id, row, head_id, score in zip(
            self.contributor_child_node_ids,
            self.contributor_proposal_rows,
            self.contributor_head_ids,
            self.contributor_scores,
        ):
            if child_id not in keep:
                continue
            contributor_child_node_ids.append(child_id)
            contributor_proposal_rows.append(row)
            contributor_head_ids.append(head_id)
            contributor_scores.append(score)

        return DraftTokenTree(
            node_token_ids=self.node_token_ids[:max_nodes],
            parent_node_ids=self.parent_node_ids[:max_nodes],
            node_depths=self.node_depths[:max_nodes],
            node_priorities=self.node_priorities[:max_nodes],
            child_start_indices=child_start_indices,
            child_end_indices=child_end_indices,
            child_node_ids=child_node_ids,
            contributor_child_node_ids=contributor_child_node_ids,
            contributor_proposal_rows=contributor_proposal_rows,
            contributor_head_ids=contributor_head_ids,
            contributor_scores=contributor_scores,
            proposal_num_rows=self.proposal_num_rows,
            trace_id=self.trace_id,
        )
