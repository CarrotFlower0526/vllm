# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only forced replay of a canonical generated token sequence."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.spec_decode.metadata import (
        SpecDecodeMetadata,
        TreeSpecDecodeMetadata,
    )


CANONICAL_TOKEN_REPLAY_EXTRA_ARG = "canonical_token_replay"
_REPLAY_FIELDS = {
    "token_ids",
    "replay_id",
    "single_token_per_step",
    "proposal_only",
    "tree_construction_oracle",
    "trace_path",
}
_TRACE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CanonicalTokenReplayConfig:
    """Validated per-request canonical replay configuration."""

    token_ids: tuple[int, ...]
    replay_id: str
    single_token_per_step: bool
    proposal_only: bool = False
    tree_construction_oracle: bool = False
    trace_path: str | None = None

    def token_at(self, offset: int) -> int:
        if offset < 0 or offset >= len(self.token_ids):
            raise ValueError(
                f"canonical replay {self.replay_id!r} has no token at "
                f"generated offset {offset}; reference length is "
                f"{len(self.token_ids)}"
            )
        return self.token_ids[offset]

    def validate_vocab_size(self, vocab_size: int) -> None:
        invalid = [token_id for token_id in self.token_ids if token_id >= vocab_size]
        if invalid:
            raise ValueError(
                f"canonical replay {self.replay_id!r} contains token id "
                f"{invalid[0]} outside target vocabulary size {vocab_size}"
            )


def parse_canonical_token_replay(
    sampling_params: SamplingParams,
) -> CanonicalTokenReplayConfig | None:
    """Parse the canonical replay object stored in ``extra_args``."""

    extra_args = sampling_params.extra_args
    if not extra_args or CANONICAL_TOKEN_REPLAY_EXTRA_ARG not in extra_args:
        return None

    raw = extra_args[CANONICAL_TOKEN_REPLAY_EXTRA_ARG]
    if not isinstance(raw, dict):
        raise TypeError("canonical_token_replay must be an object")
    unexpected = set(raw) - _REPLAY_FIELDS
    if unexpected:
        names = ", ".join(sorted(str(name) for name in unexpected))
        raise ValueError(f"unexpected canonical_token_replay fields: {names}")
    required = _REPLAY_FIELDS - {
        "trace_path",
        "proposal_only",
        "tree_construction_oracle",
    }
    missing = required - set(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing canonical_token_replay fields: {names}")

    raw_token_ids = raw["token_ids"]
    if not isinstance(raw_token_ids, list) or not raw_token_ids:
        raise TypeError("canonical_token_replay.token_ids must be a non-empty list")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in raw_token_ids
    ):
        raise TypeError("canonical_token_replay.token_ids must contain integers")
    if any(token_id < 0 for token_id in raw_token_ids):
        raise ValueError("canonical_token_replay.token_ids must be non-negative")

    replay_id = raw["replay_id"]
    if not isinstance(replay_id, str) or not replay_id.strip():
        raise TypeError("canonical_token_replay.replay_id must be a non-empty string")
    single_token_per_step = raw["single_token_per_step"]
    if not isinstance(single_token_per_step, bool):
        raise TypeError(
            "canonical_token_replay.single_token_per_step must be a boolean"
        )
    proposal_only = raw.get("proposal_only", False)
    if not isinstance(proposal_only, bool):
        raise TypeError("canonical_token_replay.proposal_only must be a boolean")
    if proposal_only and not single_token_per_step:
        raise ValueError(
            "canonical proposal-only replay requires single_token_per_step=true"
        )
    tree_construction_oracle = raw.get("tree_construction_oracle", False)
    if not isinstance(tree_construction_oracle, bool):
        raise TypeError(
            "canonical_token_replay.tree_construction_oracle must be a boolean"
        )
    if tree_construction_oracle and not proposal_only:
        raise ValueError("tree-construction oracle requires proposal_only=true")
    trace_path = raw.get("trace_path")
    if trace_path is not None and (
        not isinstance(trace_path, str) or not trace_path.strip()
    ):
        raise TypeError(
            "canonical_token_replay.trace_path must be a non-empty string or null"
        )

    if sampling_params.temperature != 0.0:
        raise ValueError("canonical token replay requires greedy sampling")
    max_tokens = sampling_params.max_tokens
    if max_tokens is None:
        raise ValueError("canonical token replay requires a finite max_tokens")

    return CanonicalTokenReplayConfig(
        token_ids=tuple(raw_token_ids),
        replay_id=replay_id,
        single_token_per_step=single_token_per_step,
        proposal_only=proposal_only,
        tree_construction_oracle=tree_construction_oracle,
        trace_path=trace_path,
    )


class CanonicalTokenReplayLogitsProcessor:
    """Force target sampling to follow a per-request reference sequence.

    This processor deliberately synchronizes tree metadata to the CPU when
    tracing or constructing tree offsets. It is a correctness and proposal
    coverage diagnostic and must never be used for latency measurements.
    """

    def __init__(
        self,
        configs: list[CanonicalTokenReplayConfig | None],
        output_token_ids: list[list[int]],
    ) -> None:
        if len(configs) != len(output_token_ids):
            raise ValueError("canonical replay configs must match request outputs")
        if not any(config is not None for config in configs):
            raise ValueError("canonical replay processor requires an active request")
        self.configs = tuple(configs)
        self.output_token_ids = output_token_ids

    def proposal_only_batch(self) -> bool:
        """Return whether every active request uses proposal-only replay.

        Proposal-only replay deliberately does not support mixing replay and
        ordinary requests in one target forward.  The diagnostic serving
        protocol runs exactly one active request at a time, and failing closed
        here prevents an ordinary request's drafts from being discarded.
        """

        enabled = [
            config is not None and config.proposal_only for config in self.configs
        ]
        if any(enabled) and not all(enabled):
            raise NotImplementedError(
                "canonical proposal-only replay cannot share a batch with "
                "ordinary or verifier-replay requests"
            )
        return bool(enabled) and all(enabled)

    def proposal_only_final_step(self) -> bool:
        """Return true when the just-sampled token ends every reference row."""

        if not self.proposal_only_batch():
            return False
        return all(
            len(outputs) + 1 >= len(config.token_ids)
            for config, outputs in zip(
                self.configs, self.output_token_ids, strict=True
            )
            if config is not None
        )

    def tree_construction_oracle_batch(self) -> bool:
        """Return whether every active request enables the tree oracle."""

        enabled = [
            config is not None and config.tree_construction_oracle
            for config in self.configs
        ]
        if any(enabled) and not all(enabled):
            raise NotImplementedError(
                "tree-construction oracle cannot share a batch with ordinary "
                "proposal replay"
            )
        return bool(enabled) and all(enabled)

    def validate_plain_target_step(
        self,
        spec_decode_metadata: SpecDecodeMetadata | TreeSpecDecodeMetadata | None,
    ) -> None:
        """Fail before a verifier forward in proposal-only replay.

        Erasing a captured proposal is the normal mechanism that keeps the
        following target step on the ordinary no-spec path. This check is a
        fail-closed backstop: stale linear or tree metadata must never silently
        turn a proposal-quality diagnostic into verifier execution.
        """

        if self.proposal_only_batch() and spec_decode_metadata is not None:
            metadata_type = type(spec_decode_metadata).__name__
            raise RuntimeError(
                "canonical proposal-only replay forbids verifier execution; "
                f"received {metadata_type}. Captured proposals must be "
                "discarded before scheduling so every target step has "
                "spec_decode_metadata=None"
            )

    def apply_plain(
        self,
        logits: torch.Tensor,
        *,
        predict_bonus_token: bool = False,
        spec_token_ids: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Mask ordinary next-token or linear bonus-token logits."""

        self._validate_batch_logits(logits)
        if predict_bonus_token and (
            spec_token_ids is None or len(spec_token_ids) < len(self.configs)
        ):
            raise ValueError("canonical replay bonus sampling needs draft token ids")

        rows: list[int] = []
        token_ids: list[int] = []
        for req_idx, config in enumerate(self.configs):
            if config is None:
                continue
            offset = len(self.output_token_ids[req_idx])
            if predict_bonus_token:
                assert spec_token_ids is not None
                offset += len(spec_token_ids[req_idx])
                if offset >= len(config.token_ids):
                    # The bonus is discarded by a final single-root replay
                    # step, so it does not need a token beyond the reference.
                    continue
            rows.append(req_idx)
            token_ids.append(config.token_at(offset))
        return self._mask_rows(logits, rows, token_ids)

    def apply_linear(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """Mask flattened standard speculative target rows."""

        if logits.ndim != 2:
            raise ValueError("canonical replay logits must be two-dimensional")
        if len(num_draft_tokens) != len(self.configs):
            raise ValueError("linear draft counts must match replay requests")

        rows: list[int] = []
        token_ids: list[int] = []
        cursor = 0
        for req_idx, (config, num_drafts) in enumerate(
            zip(self.configs, num_draft_tokens, strict=True)
        ):
            if num_drafts < 0:
                raise ValueError("linear draft counts must be non-negative")
            if config is not None:
                root_offset = len(self.output_token_ids[req_idx])
                for local_position in range(num_drafts):
                    offset = root_offset + local_position
                    if local_position > 0 and offset >= len(config.token_ids):
                        # Deeper rows are discarded after would-accept is
                        # measured at the final reference root.
                        continue
                    rows.append(cursor + local_position)
                    token_ids.append(config.token_at(offset))
            cursor += num_drafts
        if cursor != logits.shape[0]:
            raise ValueError("linear draft counts do not match target logit rows")
        return self._mask_rows(logits, rows, token_ids)

    def apply_tree(
        self,
        logits: torch.Tensor,
        node_request_indices: torch.Tensor,
        node_depths: torch.Tensor,
    ) -> torch.Tensor:
        """Mask every flattened tree row at root offset plus node depth."""

        if logits.ndim != 2:
            raise ValueError("canonical replay logits must be two-dimensional")
        request_indices = node_request_indices.detach().cpu().tolist()
        depths = node_depths.detach().cpu().tolist()
        if len(request_indices) != logits.shape[0] or len(depths) != logits.shape[0]:
            raise ValueError("tree replay metadata must match target logit rows")

        rows: list[int] = []
        token_ids: list[int] = []
        for row, (req_idx, depth) in enumerate(
            zip(request_indices, depths, strict=True)
        ):
            req_idx = int(req_idx)
            depth = int(depth)
            if req_idx < 0 or req_idx >= len(self.configs):
                raise ValueError("tree replay request index is out of range")
            if depth < 0:
                raise ValueError("tree replay node depth must be non-negative")
            config = self.configs[req_idx]
            if config is None:
                continue
            offset = len(self.output_token_ids[req_idx]) + depth
            if depth > 0 and offset >= len(config.token_ids):
                # A non-root row beyond the final reference token is never
                # returned by single-root replay.
                continue
            rows.append(row)
            token_ids.append(config.token_at(offset))
        return self._mask_rows(logits, rows, token_ids)

    def truncate_linear_output(
        self,
        output_token_ids: torch.Tensor,
        *,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Return only the canonical root token for configured requests."""

        return self._truncate_rows(output_token_ids, placeholder_token_id)

    def truncate_tree_output(
        self,
        output_token_ids: torch.Tensor,
        accepted_tree_node_ids: torch.Tensor,
        *,
        placeholder_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate outputs and clear accepted nodes for one-root-step replay."""

        self._truncate_rows(output_token_ids, placeholder_token_id)
        if accepted_tree_node_ids.ndim != 2:
            raise ValueError("accepted tree node ids must be two-dimensional")
        if accepted_tree_node_ids.shape[0] != len(self.configs):
            raise ValueError("accepted tree rows must match replay requests")
        for req_idx, config in enumerate(self.configs):
            if config is not None and config.single_token_per_step:
                accepted_tree_node_ids[req_idx].fill_(placeholder_token_id)
        return output_token_ids, accepted_tree_node_ids

    def trace_plain(self, sampled_token_ids: torch.Tensor) -> None:
        """Write optional ordinary-greedy replay records."""

        rows = sampled_token_ids.detach().cpu().reshape(len(self.configs), -1).tolist()
        for req_idx, config in enumerate(self.configs):
            if config is None or config.trace_path is None:
                continue
            would_sample = self._valid_tokens(rows[req_idx])
            offset = len(self.output_token_ids[req_idx])
            # In proposal-only replay, position zero is the sole plain-target
            # event.  Every later position is represented by the proposal
            # captured after the preceding ordinary target step.
            if config.proposal_only and offset > 0:
                continue
            canonical = config.token_at(offset)
            self._append_trace(
                config,
                {
                    "mode": "plain_greedy",
                    "output_offset": offset,
                    "canonical_token_id": canonical,
                    "would_accept": None,
                    "draft_token_ids": [],
                    "draft_candidate_token_ids": [],
                    "would_sampled_token_ids": would_sample,
                    "returned_token_ids": [canonical],
                },
            )

    def trace_proposal_only_linear(
        self,
        draft_token_ids: list[list[int]],
        request_ids: list[str],
    ) -> None:
        """Trace stock/linear proposals for the next canonical position."""

        self._validate_proposal_rows(draft_token_ids, request_ids)
        for req_idx, (config, proposals) in enumerate(
            zip(self.configs, draft_token_ids, strict=True)
        ):
            if config is None or not config.proposal_only:
                continue
            offset = len(self.output_token_ids[req_idx]) + 1
            if offset >= len(config.token_ids):
                continue
            canonical = config.token_at(offset)
            candidates = [int(token_id) for token_id in proposals if token_id >= 0]
            h1_token = candidates[0] if candidates else None
            h1_hit = h1_token == canonical
            root_children = (
                [
                    {
                        "node_id": 1,
                        "token_id": h1_token,
                        "matches_canonical": h1_hit,
                        "contributor_head_ids": [0],
                        "head_contributors": [],
                    }
                ]
                if h1_token is not None
                else []
            )
            if config.trace_path is not None:
                self._append_trace(
                    config,
                    {
                        "mode": "proposal_only_linear",
                        "proposal_only": True,
                        "target_execution": "plain_no_spec",
                        "output_offset": offset,
                        "canonical_token_id": canonical,
                        "would_accept": h1_hit,
                        "draft_token_ids": candidates,
                        "draft_candidate_token_ids": candidates,
                        "root_children": root_children,
                        "h1_hit": h1_hit,
                        "h2_hit": False,
                        "h2_extra_hit": False,
                        "would_accept_draft_count": int(h1_hit),
                        "would_sampled_token_ids": None,
                        "returned_token_ids": [canonical],
                    },
                )

    def trace_proposal_only_trees(
        self,
        draft_token_trees: list[Any],
        request_ids: list[str],
    ) -> None:
        """Trace residual-tree root proposals for the next canonical position."""

        if len(draft_token_trees) != len(self.configs):
            raise ValueError("proposal trees must match replay requests")
        if len(request_ids) != len(self.configs):
            raise ValueError("proposal request ids must match replay requests")

        for req_idx, (config, tree) in enumerate(
            zip(self.configs, draft_token_trees, strict=True)
        ):
            if config is None or not config.proposal_only:
                continue
            offset = len(self.output_token_ids[req_idx]) + 1
            if offset >= len(config.token_ids):
                continue
            canonical = config.token_at(offset)

            start = int(tree.child_start_indices[0])
            end = int(tree.child_end_indices[0])
            contributors_by_child: dict[int, list[dict[str, int | float]]] = {}
            for child_id, proposal_row, head_id, score in zip(
                tree.contributor_child_node_ids,
                tree.contributor_proposal_rows,
                tree.contributor_head_ids,
                tree.contributor_scores,
                strict=True,
            ):
                contributors_by_child.setdefault(int(child_id), []).append(
                    {
                        "head_id": int(head_id),
                        "proposal_row": int(proposal_row),
                        "score": float(score),
                    }
                )

            root_children: list[dict[str, Any]] = []
            h1_hit = False
            h2_hit = False
            for child_id in tree.child_node_ids[start:end]:
                child_id = int(child_id)
                token_id = int(tree.node_token_ids[child_id])
                contributors = contributors_by_child.get(child_id, [])
                head_ids = sorted(
                    {int(item["head_id"]) for item in contributors}
                )
                matches = token_id == canonical
                h1_hit = h1_hit or (matches and 0 in head_ids)
                h2_hit = h2_hit or (matches and 1 in head_ids)
                root_children.append(
                    {
                        "node_id": child_id,
                        "token_id": token_id,
                        "matches_canonical": matches,
                        "contributor_head_ids": head_ids,
                        "head_contributors": contributors,
                    }
                )

            oracle_path_token_ids: list[int] = []
            oracle_parent_id = 0
            while offset + len(oracle_path_token_ids) < len(config.token_ids):
                oracle_start = int(tree.child_start_indices[oracle_parent_id])
                oracle_end = int(tree.child_end_indices[oracle_parent_id])
                expected_token = config.token_at(
                    offset + len(oracle_path_token_ids)
                )
                matching_children = [
                    int(child_id)
                    for child_id in tree.child_node_ids[oracle_start:oracle_end]
                    if int(tree.node_token_ids[int(child_id)]) == expected_token
                ]
                if not matching_children:
                    break
                oracle_parent_id = matching_children[0]
                oracle_path_token_ids.append(expected_token)

            if config.trace_path is not None:
                hit_count = int(h1_hit or h2_hit)
                self._append_trace(
                    config,
                    {
                        "mode": "proposal_only_tree",
                        "proposal_only": True,
                        "target_execution": "plain_no_spec",
                        "output_offset": offset,
                        "canonical_token_id": canonical,
                        "would_accept": bool(hit_count),
                        "draft_token_ids": [
                            int(child["token_id"]) for child in root_children
                        ],
                        "draft_candidate_token_ids": [
                            int(child["token_id"]) for child in root_children
                        ],
                        "root_children": root_children,
                        "h1_hit": h1_hit,
                        "h2_hit": h2_hit,
                        "h2_extra_hit": bool(h2_hit and not h1_hit),
                        "would_accept_draft_count": hit_count,
                        "tree_construction_oracle": (
                            config.tree_construction_oracle
                        ),
                        "would_accept_tree_path_count": len(
                            oracle_path_token_ids
                        ),
                        "would_accept_tree_path_token_ids": (
                            oracle_path_token_ids
                        ),
                        "would_sampled_token_ids": None,
                        "returned_token_ids": [canonical],
                    },
                )

    def _validate_proposal_rows(
        self,
        draft_token_ids: list[list[int]],
        request_ids: list[str],
    ) -> None:
        if len(draft_token_ids) != len(self.configs):
            raise ValueError("proposal rows must match replay requests")
        if len(request_ids) != len(self.configs):
            raise ValueError("proposal request ids must match replay requests")

    def trace_linear(
        self,
        metadata: SpecDecodeMetadata,
        sampled_token_ids: torch.Tensor,
    ) -> None:
        """Write optional standard linear speculative replay records."""

        drafts = metadata.draft_token_ids.detach().cpu().tolist()
        sampled = sampled_token_ids.detach().cpu().tolist()
        cursor = 0
        for req_idx, (config, num_drafts) in enumerate(
            zip(self.configs, metadata.num_draft_tokens, strict=True)
        ):
            request_drafts = [
                int(token_id) for token_id in drafts[cursor : cursor + num_drafts]
            ]
            cursor += num_drafts
            if config is None or config.trace_path is None:
                continue
            would_sample = self._valid_tokens(sampled[req_idx])
            accepted_count = 0
            for draft_token, sampled_token in zip(request_drafts, would_sample):
                if draft_token != sampled_token:
                    break
                accepted_count += 1
            offset = len(self.output_token_ids[req_idx])
            canonical = config.token_at(offset)
            returned = [canonical] if config.single_token_per_step else would_sample
            self._append_trace(
                config,
                {
                    "mode": "linear_speculative",
                    "output_offset": offset,
                    "canonical_token_id": canonical,
                    "would_accept": accepted_count > 0,
                    "draft_token_ids": request_drafts,
                    "draft_candidate_token_ids": request_drafts,
                    "would_accept_draft_count": accepted_count,
                    "would_sampled_token_ids": would_sample,
                    "returned_token_ids": returned,
                },
            )

    def trace_tree(
        self,
        metadata: TreeSpecDecodeMetadata,
        sampled_token_ids: torch.Tensor,
        accepted_tree_node_ids: torch.Tensor,
    ) -> None:
        """Write optional tree-root coverage and would-accept records."""

        roots = metadata.root_node_ids.detach().cpu().tolist()
        node_tokens = metadata.node_token_ids.detach().cpu().tolist()
        child_starts = metadata.child_start_indices.detach().cpu().tolist()
        child_ends = metadata.child_end_indices.detach().cpu().tolist()
        child_ids = metadata.child_node_ids.detach().cpu().tolist()
        contributor_children = (
            metadata.contributor_child_node_ids.detach().cpu().tolist()
        )
        contributor_rows = metadata.contributor_proposal_rows.detach().cpu().tolist()
        contributor_heads = metadata.contributor_head_ids.detach().cpu().tolist()
        contributor_scores = metadata.contributor_scores.detach().cpu().tolist()
        sampled = sampled_token_ids.detach().cpu().tolist()
        accepted = accepted_tree_node_ids.detach().cpu().tolist()

        contributors_by_child: dict[int, list[dict[str, int | float]]] = {}
        for child_id, proposal_row, head_id, score in zip(
            contributor_children,
            contributor_rows,
            contributor_heads,
            contributor_scores,
            strict=True,
        ):
            contributors_by_child.setdefault(int(child_id), []).append(
                {
                    "head_id": int(head_id),
                    "proposal_row": int(proposal_row),
                    "score": float(score),
                }
            )

        for req_idx, config in enumerate(self.configs):
            if config is None or config.trace_path is None:
                continue
            root = int(roots[req_idx])
            start = int(child_starts[root])
            end = int(child_ends[root])
            root_children = []
            h1_hit = False
            h2_hit = False
            offset = len(self.output_token_ids[req_idx])
            canonical = config.token_at(offset)
            for child in child_ids[start:end]:
                child = int(child)
                token_id = int(node_tokens[child])
                contributors = contributors_by_child.get(child, [])
                head_ids = {int(item["head_id"]) for item in contributors}
                matches = token_id == canonical
                h1_hit = h1_hit or (matches and 0 in head_ids)
                h2_hit = h2_hit or (matches and 1 in head_ids)
                root_children.append(
                    {
                        "node_id": child - root,
                        "token_id": token_id,
                        "matches_canonical": matches,
                        "contributor_head_ids": sorted(head_ids),
                        "head_contributors": contributors,
                    }
                )

            would_sample = self._valid_tokens(sampled[req_idx])
            would_accept = self._valid_tokens(accepted[req_idx])
            returned = [canonical] if config.single_token_per_step else would_sample
            self._append_trace(
                config,
                {
                    "mode": "tree_speculative",
                    "output_offset": offset,
                    "canonical_token_id": canonical,
                    "would_accept": len(would_accept) > 0,
                    "draft_token_ids": [child["token_id"] for child in root_children],
                    "draft_candidate_token_ids": [
                        child["token_id"] for child in root_children
                    ],
                    "root_children": root_children,
                    "h1_hit": h1_hit,
                    "h2_hit": h2_hit,
                    "h2_extra_hit": bool(h2_hit and not h1_hit),
                    "would_accepted_tree_node_ids": would_accept,
                    "would_accept_draft_count": len(would_accept),
                    "would_sampled_token_ids": would_sample,
                    "returned_token_ids": returned,
                },
            )

    def _validate_batch_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError("canonical replay logits must be two-dimensional")
        if logits.shape[0] != len(self.configs):
            raise ValueError("canonical replay logits must match request count")

    @staticmethod
    def _valid_tokens(row: list[int]) -> list[int]:
        return [int(token_id) for token_id in row if int(token_id) >= 0]

    def _mask_rows(
        self,
        logits: torch.Tensor,
        rows: list[int],
        token_ids: list[int],
    ) -> torch.Tensor:
        if not rows:
            return logits
        vocab_size = logits.shape[1]
        invalid = [token_id for token_id in token_ids if token_id >= vocab_size]
        if invalid:
            raise ValueError(
                f"canonical replay token id {invalid[0]} is outside logit "
                f"vocabulary size {vocab_size}"
            )

        row_tensor = torch.tensor(rows, dtype=torch.long, device=logits.device)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
        selected = logits.index_select(0, row_tensor)
        canonical_values = selected.gather(1, token_tensor.unsqueeze(1)).clone()
        canonical_values.masked_fill_(~torch.isfinite(canonical_values), 0.0)
        selected.fill_(float("-inf"))
        selected.scatter_(1, token_tensor.unsqueeze(1), canonical_values)
        logits.index_copy_(0, row_tensor, selected)
        return logits

    def _truncate_rows(
        self,
        output_token_ids: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        if output_token_ids.ndim != 2:
            raise ValueError("replay sampled token ids must be two-dimensional")
        if output_token_ids.shape[0] != len(self.configs):
            raise ValueError("replay sampled rows must match request count")
        for req_idx, config in enumerate(self.configs):
            if config is not None and config.single_token_per_step:
                output_token_ids[req_idx, 1:].fill_(placeholder_token_id)
        return output_token_ids

    @staticmethod
    def _append_trace(
        config: CanonicalTokenReplayConfig,
        record: dict[str, Any],
    ) -> None:
        assert config.trace_path is not None
        payload = {
            "schema_version": 1,
            "event": "canonical_token_replay",
            "diagnostic_only": True,
            "latency_valid": False,
            "replay_id": config.replay_id,
            "single_token_per_step": config.single_token_per_step,
            **record,
        }
        destination = Path(config.trace_path)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with _TRACE_LOCK:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(line)
