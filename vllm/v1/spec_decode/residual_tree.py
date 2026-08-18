# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal, cast

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.tree_schema import DraftTokenTree

_EPS = 1e-12
_BEST_FIRST_PREFETCH_LIMIT = 16


@triton.jit(do_not_specialize=["max_spec_len"])
def _verify_greedy_b2d1_kernel(
    output_token_ids_ptr,
    accepted_node_ids_ptr,
    cu_num_tree_nodes_ptr,
    root_node_ids_ptr,
    node_token_ids_ptr,
    parent_node_ids_ptr,
    target_token_ids_ptr,
    placeholder_token_id,
    max_spec_len,
):
    """One-program-per-request greedy verifier for at-most-B2D1 trees."""

    req_idx = tl.program_id(0)
    root_node = tl.load(root_node_ids_ptr + req_idx)
    request_end = tl.load(cu_num_tree_nodes_ptr + req_idx)
    num_draft_nodes = request_end - root_node - 1

    output_offset = req_idx * (max_spec_len + 1)
    accepted_offset = req_idx * max_spec_len
    root_target = tl.load(target_token_ids_ptr + root_node).to(tl.int32)
    tl.store(output_token_ids_ptr + output_offset, root_target)

    node_one = root_node + 1
    node_two = root_node + 2
    has_one = num_draft_nodes >= 1
    has_two = num_draft_nodes >= 2
    token_one = tl.load(node_token_ids_ptr + node_one, mask=has_one, other=-1)
    token_two = tl.load(node_token_ids_ptr + node_two, mask=has_two, other=-1)
    parent_one = tl.load(parent_node_ids_ptr + node_one, mask=has_one, other=-1)
    parent_two = tl.load(parent_node_ids_ptr + node_two, mask=has_two, other=-1)

    take_one = has_one & (parent_one == root_node) & (token_one == root_target)
    take_two = (
        (~take_one) & has_two & (parent_two == root_node) & (token_two == root_target)
    )
    accepted_root = take_one | take_two
    chosen_node = tl.where(take_one, node_one, node_two)
    chosen_local_node = tl.where(take_one, 1, 2)
    chosen_target = tl.load(
        target_token_ids_ptr + chosen_node,
        mask=accepted_root,
        other=placeholder_token_id,
    ).to(tl.int32)
    tl.store(
        output_token_ids_ptr + output_offset + 1,
        chosen_target,
        mask=accepted_root,
    )
    tl.store(
        accepted_node_ids_ptr + accepted_offset,
        chosen_local_node,
        mask=accepted_root,
    )

    node_one_target = tl.load(
        target_token_ids_ptr + node_one,
        mask=take_one & has_two,
        other=placeholder_token_id,
    )
    take_chain_child = (
        take_one & has_two & (parent_two == node_one) & (node_one_target == token_two)
    )
    node_two_target = tl.load(
        target_token_ids_ptr + node_two,
        mask=take_chain_child,
        other=placeholder_token_id,
    ).to(tl.int32)
    tl.store(
        output_token_ids_ptr + output_offset + 2,
        node_two_target,
        mask=take_chain_child,
    )
    tl.store(
        accepted_node_ids_ptr + accepted_offset + 1,
        2,
        mask=take_chain_child,
    )


@triton.jit
def _verify_greedy_tree_kernel(
    output_token_ids_ptr,
    accepted_node_ids_ptr,
    cu_num_tree_nodes_ptr,
    root_node_ids_ptr,
    node_token_ids_ptr,
    parent_node_ids_ptr,
    target_token_ids_ptr,
    MAX_SPEC_LEN: tl.constexpr,
    MAX_TREE_DEPTH: tl.constexpr,
):
    """One-program-per-request greedy walk for an arbitrary small tree."""

    req_idx = tl.program_id(0)
    root_node = tl.load(root_node_ids_ptr + req_idx)
    request_end = tl.load(cu_num_tree_nodes_ptr + req_idx)
    output_offset = req_idx * (MAX_SPEC_LEN + 1)
    accepted_offset = req_idx * MAX_SPEC_LEN

    current_node = root_node
    active = True
    root_target = tl.load(target_token_ids_ptr + root_node).to(tl.int32)
    tl.store(output_token_ids_ptr + output_offset, root_target)

    for step in tl.static_range(0, MAX_TREE_DEPTH):
        wanted_token = tl.load(
            target_token_ids_ptr + current_node,
            mask=active,
            other=-1,
        )
        found = False
        chosen_node = root_node
        chosen_local_node = 0
        for local_node in tl.static_range(1, MAX_SPEC_LEN + 1):
            candidate_node = root_node + local_node
            in_request = candidate_node < request_end
            candidate_token = tl.load(
                node_token_ids_ptr + candidate_node,
                mask=in_request,
                other=-2,
            )
            candidate_parent = tl.load(
                parent_node_ids_ptr + candidate_node,
                mask=in_request,
                other=-2,
            )
            take = (
                active
                & (~found)
                & in_request
                & (candidate_parent == current_node)
                & (candidate_token == wanted_token)
            )
            chosen_node = tl.where(take, candidate_node, chosen_node)
            chosen_local_node = tl.where(take, local_node, chosen_local_node)
            found = found | take

        accepted = active & found
        tl.store(
            accepted_node_ids_ptr + accepted_offset + step,
            chosen_local_node,
            mask=accepted,
        )
        next_target = tl.load(
            target_token_ids_ptr + chosen_node,
            mask=accepted,
            other=-1,
        ).to(tl.int32)
        tl.store(
            output_token_ids_ptr + output_offset + step + 1,
            next_target,
            mask=accepted,
        )
        current_node = tl.where(accepted, chosen_node, current_node)
        active = accepted


@dataclass(frozen=True)
class TreeCandidateContributor:
    """One residual head's contribution to a merged child candidate."""

    head_id: int
    proposal_row: int
    token_id: int
    proposal_prob: float
    lambda_weight: float
    score: float


@dataclass(frozen=True)
class ResidualTreeNode:
    """A node in a speculative residual-head token tree."""

    node_id: int
    parent_id: int
    token_id: int
    depth: int
    priority: float
    contributors: tuple[TreeCandidateContributor, ...] = ()


@dataclass(frozen=True)
class TreeListwiseFeatures:
    """Serving-visible features for one expanded tree state.

    Candidate columns follow ``head_ids`` exactly. Head ids are vLLM's
    zero-based residual-head ids, and ``depth`` is the candidate child depth.
    """

    candidate_token_ids: tuple[int, ...]
    head_ids: tuple[int, ...]
    conditioned_proposal_probabilities: tuple[float, ...]
    base_probabilities: tuple[float, ...]
    logit_margins: tuple[float, ...]
    entropies: tuple[float, ...]
    topk_masses: tuple[float, ...]
    depth: int
    top_k: int


@dataclass
class TreeStateScoringRecord:
    """Trace record for all ordered-head candidates at one parent state."""

    parent_id: int
    parent_priority: float
    features: TreeListwiseFeatures
    candidate_priorities: tuple[float, ...]
    class_probabilities: tuple[float, ...] | None
    entered_node_ids: list[int | None]


@dataclass(frozen=True)
class _SelectedHeadCandidate:
    head_id: int
    token_id: int
    raw_q: torch.Tensor
    conditioned_q: torch.Tensor
    proposal_row: int
    lambda_weight: float
    proposal_probability: float


@dataclass
class ResidualTree:
    """Flattened token tree with children stored as local node ids.

    Node 0 is always the root and is not a draft token. The fixed
    `node_budget` used by `select_residual_tree` counts nodes 1..N only.
    """

    nodes: list[ResidualTreeNode]
    children: list[list[int]]
    proposal_probs: torch.Tensor | None = None
    states: list[Any] = field(default_factory=list, repr=False)
    scoring_records: list[TreeStateScoringRecord] = field(
        default_factory=list,
        repr=False,
    )
    dynamic_provenance: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )

    @property
    def num_draft_nodes(self) -> int:
        return max(0, len(self.nodes) - 1)

    @property
    def max_depth(self) -> int:
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes)

    def token_ids(self) -> torch.Tensor:
        return torch.tensor([node.token_id for node in self.nodes], dtype=torch.int64)

    def parent_ids(self) -> torch.Tensor:
        return torch.tensor([node.parent_id for node in self.nodes], dtype=torch.int64)


@dataclass(frozen=True)
class TreeVerificationResult:
    token_ids: list[int]
    accepted_node_ids: list[int]
    stopped_at_node_id: int
    fallback_token_id: int | None
    num_rejected: int
    residual_masses: list[float]

    @property
    def num_accepted(self) -> int:
        return len(self.accepted_node_ids)


ProposalFn = Callable[[Any], Sequence[torch.Tensor]]
TransitionFn = Callable[[Any, int], Any]
ProposalBatchFn = Callable[[Sequence[Any]], Sequence[Sequence[torch.Tensor]]]
TransitionBatchFn = Callable[[Sequence[Any], Sequence[int]], Sequence[Any]]
CompactCandidateBatchOutput = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
CompactCandidateBatchFn = Callable[[Sequence[Any]], CompactCandidateBatchOutput]
ListwiseScorerFn = Callable[[TreeListwiseFeatures], Sequence[float]]
ChildOrder = Literal["head_id", "priority", "node_id", "random"]
CandidateSelection = Literal[
    "head_top1",
    "distinct_head_top1",
    "gated_distinct_head_top1",
]
TreePolicy = Literal["best_first", "breadth_first"]
TreeScorerMode = Literal["lambda_q", "uniform", "head_prior", "greedy_listwise"]
_QueuedTreeCandidate = tuple[
    float,
    int,
    int,
    int,
    tuple[TreeCandidateContributor, ...],
]


def estimate_residual_head_lambdas(
    target_probs: torch.Tensor,
    proposal_probs: torch.Tensor,
    *,
    eps: float = _EPS,
) -> torch.Tensor:
    """Estimate per-head residual coverable mass.

    For each position and residual head i this computes

        m_i(u) * sum_x min(q_i(x | u), r_i(x | u))

    where m_i is the current residual mass and r_i is the normalized residual.
    The residual update uses clipped accepted mass:

        a_i = min(R_{i-1}, Z_{i-1} q_i)

    Args:
        target_probs: Tensor shaped [num_positions, vocab_size].
        proposal_probs: Tensor shaped [num_positions, num_heads, vocab_size].
    """

    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [num_positions, vocab_size]")
    if proposal_probs.ndim != 3:
        raise ValueError(
            "proposal_probs must have shape [num_positions, num_heads, vocab_size]"
        )
    if proposal_probs.shape[0] != target_probs.shape[0]:
        raise ValueError("target and proposal positions must match")
    if proposal_probs.shape[2] != target_probs.shape[1]:
        raise ValueError("target and proposal vocab sizes must match")
    if target_probs.shape[0] == 0:
        raise ValueError("at least one position is required")

    residual = _normalize_rows(target_probs.to(torch.float32), eps=eps)
    proposals = _normalize_rows(proposal_probs.to(torch.float32), eps=eps)
    num_heads = proposals.shape[1]
    lambdas = torch.empty(num_heads, dtype=torch.float32, device=proposal_probs.device)

    for head_id in range(num_heads):
        q = proposals[:, head_id, :]
        residual_mass = residual.sum(dim=-1)
        normalized_residual = torch.where(
            residual_mass[:, None] > eps,
            residual / residual_mass.clamp_min(eps)[:, None],
            torch.zeros_like(residual),
        )
        overlap = torch.minimum(q, normalized_residual).sum(dim=-1)
        lambdas[head_id] = (residual_mass * overlap).mean()

        accepted = torch.minimum(residual, residual_mass[:, None] * q)
        residual = (residual - accepted).clamp_min(0.0)

    return lambdas


def select_residual_tree(
    *,
    root_state: Any,
    root_head_probs: Sequence[torch.Tensor] | None = None,
    proposal_fn: ProposalFn,
    transition_fn: TransitionFn,
    head_lambdas: torch.Tensor | Sequence[float],
    node_budget: int,
    max_depth: int | None = None,
    active_heads: Sequence[int] | torch.Tensor | None = None,
    store_proposal_probs: bool = True,
    candidate_selection: CandidateSelection = "head_top1",
    min_novel_probability_ratio: float = 0.0,
    tree_policy: TreePolicy = "best_first",
    scorer_mode: TreeScorerMode = "lambda_q",
    head_prior_probabilities: Sequence[float] | torch.Tensor | None = None,
    listwise_scorer: ListwiseScorerFn | None = None,
    scorer_top_k: int = 10,
    collect_scoring_records: bool = False,
    proposal_batch_fn: ProposalBatchFn | None = None,
    transition_batch_fn: TransitionBatchFn | None = None,
) -> ResidualTree:
    """Build a fixed-budget residual-head speculative token tree.

    For every expanded prefix node u, each active residual head contributes one
    token. With ``head_top1`` this is its independently highest-probability token
    and duplicates under the same parent are merged. With
    ``distinct_head_top1``, ordered heads condition their proposal on tokens
    already selected at this parent being unavailable and choose the
    highest-probability remaining token. The conditioned proposal is used by
    both scoring and verification. With
    ``gated_distinct_head_top1``, that replacement is made only when the best
    novel token has at least ``min_novel_probability_ratio`` times the repeated
    winner's probability; a taken novel branch also uses the conditioned
    proposal. The legacy ``lambda_q`` priority uses the selected token's
    probability under the proposal stored for verification:

        rho_hat(u + y) = rho_hat(u) * sum_i lambda_i q_i(y | u)

    ``uniform``, ``head_prior``, and ``greedy_listwise`` instead treat the
    ordered candidates plus a none-of-the-above outcome as one categorical
    distribution. Their child priority is exactly the parent path probability
    times the candidate class probability. Proposal q is an input feature for
    the learned scorer and is not multiplied a second time.

    ``best_first`` uses a global priority queue to repeatedly materialize the
    highest-priority child. ``breadth_first`` materializes complete levels
    before considering the next depth; when a budget ends within a level,
    candidates are ordered deterministically by score, parent, and token.
    With optional batch callbacks, ``breadth_first`` processes all states from
    one level at once. ``best_first`` precomputes a bounded batch of the
    highest-priority queued candidates, then consumes the same priority queue
    in the same order; prefetching does not change which nodes enter the tree.

    ``root_head_probs`` can supply an already-batched root projection. Deeper
    states still use the scalar or batched proposal callbacks. Child states are
    computed lazily. A selected node is transitioned only if its children will
    actually be proposed, so terminal leaves never trigger a wasted drafter
    forward.
    """

    if node_budget < 0:
        raise ValueError("node_budget must be non-negative")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive when provided")
    if candidate_selection not in (
        "head_top1",
        "distinct_head_top1",
        "gated_distinct_head_top1",
    ):
        raise ValueError(f"unsupported candidate_selection: {candidate_selection}")
    if tree_policy not in ("best_first", "breadth_first"):
        raise ValueError(f"unsupported tree_policy: {tree_policy}")
    if scorer_mode not in ("lambda_q", "uniform", "head_prior", "greedy_listwise"):
        raise ValueError(f"unsupported scorer_mode: {scorer_mode}")
    if scorer_mode == "greedy_listwise" and listwise_scorer is None:
        raise ValueError("greedy_listwise scorer mode requires listwise_scorer")
    if scorer_mode != "greedy_listwise" and listwise_scorer is not None:
        raise ValueError("listwise_scorer is only valid with greedy_listwise mode")
    if scorer_mode == "head_prior" and head_prior_probabilities is None:
        raise ValueError("head_prior scorer mode requires head_prior_probabilities")
    if scorer_mode != "head_prior" and head_prior_probabilities is not None:
        raise ValueError(
            "head_prior_probabilities is only valid with head_prior scorer mode"
        )
    if isinstance(scorer_top_k, bool) or int(scorer_top_k) != scorer_top_k:
        raise ValueError("scorer_top_k must be a positive integer")
    scorer_top_k = int(scorer_top_k)
    if scorer_top_k <= 0:
        raise ValueError("scorer_top_k must be a positive integer")
    if scorer_mode != "lambda_q" and candidate_selection != "distinct_head_top1":
        raise ValueError(f"{scorer_mode} scorer requires distinct_head_top1 candidates")
    if not math.isfinite(min_novel_probability_ratio) or not (
        0.0 <= float(min_novel_probability_ratio) <= 1.0
    ):
        raise ValueError("min_novel_probability_ratio must be in [0, 1]")

    lambdas = torch.as_tensor(head_lambdas, dtype=torch.float32)
    if lambdas.ndim != 1:
        raise ValueError("head_lambdas must be a 1-D tensor or sequence")
    if active_heads is not None:
        active_heads = _resolve_active_heads(active_heads, int(lambdas.numel()))

    nodes = [
        ResidualTreeNode(
            node_id=0,
            parent_id=-1,
            token_id=-1,
            depth=0,
            priority=1.0,
        )
    ]
    children: list[list[int]] = [[]]
    unrealized_state = object()
    states: list[Any] = [root_state]
    proposal_rows: list[torch.Tensor] = []
    scoring_records: list[TreeStateScoringRecord] = []
    heap: list[_QueuedTreeCandidate] = []
    heap_counter = 0
    prefetched_states: dict[int, Any] = {}
    prefetched_head_probs: dict[int, Sequence[torch.Tensor]] = {}

    def realize_states(node_ids: Sequence[int]) -> None:
        missing_node_ids = [
            node_id for node_id in node_ids if states[node_id] is unrealized_state
        ]
        if not missing_node_ids:
            return

        parent_states: list[Any] = []
        token_ids: list[int] = []
        for node_id in missing_node_ids:
            node = nodes[node_id]
            parent_state = states[node.parent_id]
            if parent_state is unrealized_state:
                # This is not expected for the current best-first or level-wise
                # breadth-first traversal, but keeps the helper correct if the
                # traversal is extended later.
                realize_states([node.parent_id])
                parent_state = states[node.parent_id]
            parent_states.append(parent_state)
            token_ids.append(node.token_id)

        if transition_batch_fn is None:
            child_states = [
                transition_fn(parent_state, token_id)
                for parent_state, token_id in zip(parent_states, token_ids)
            ]
        else:
            child_states = list(transition_batch_fn(parent_states, token_ids))
            if len(child_states) != len(missing_node_ids):
                raise ValueError(
                    "transition_batch_fn must return one state per input transition"
                )

        for node_id, child_state in zip(missing_node_ids, child_states):
            states[node_id] = child_state

    def enqueue_candidates(
        parent_id: int,
        destination: list[_QueuedTreeCandidate] | None = None,
        head_probs_override: Sequence[torch.Tensor] | None = None,
    ) -> None:
        nonlocal heap_counter
        if destination is None:
            destination = heap
        parent = nodes[parent_id]
        if max_depth is not None and parent.depth >= max_depth:
            return
        if head_probs_override is None:
            realize_states([parent_id])
            head_probs = proposal_fn(states[parent_id])
        else:
            head_probs = head_probs_override
        head_ids = _resolve_active_heads(active_heads, len(head_probs))

        selected: list[_SelectedHeadCandidate] = []
        selected_token_ids: set[int] = set()
        for head_id in head_ids:
            if head_id >= len(head_probs):
                raise ValueError(f"active head {head_id} is missing from proposals")
            if head_id >= lambdas.shape[0]:
                raise ValueError(f"lambda for active head {head_id} is missing")

            raw_q = _normalize_vector(head_probs[head_id].detach().to(torch.float32))
            q = raw_q
            value, token_id_tensor = torch.max(raw_q, dim=-1)
            top_token_id = int(token_id_tensor.item())
            if selected_token_ids and candidate_selection == "distinct_head_top1":
                q = _condition_probability_vector(
                    raw_q,
                    blocked_token_ids=selected_token_ids,
                )
                value, token_id_tensor = torch.max(q, dim=-1)
            elif (
                selected_token_ids
                and candidate_selection == "gated_distinct_head_top1"
                and top_token_id in selected_token_ids
            ):
                conditioned_q = _condition_probability_vector(
                    raw_q,
                    blocked_token_ids=selected_token_ids,
                )
                _, novel_token_id_tensor = torch.max(conditioned_q, dim=-1)
                novel_token_id = int(novel_token_id_tensor.item())
                novel_raw_probability = raw_q[novel_token_id]
                if float(novel_raw_probability.item()) >= float(
                    min_novel_probability_ratio
                ) * float(value.item()):
                    q = conditioned_q
                    value = q[novel_token_id]
                    token_id_tensor = novel_token_id_tensor

            proposal_row = -1
            if store_proposal_probs:
                proposal_row = len(proposal_rows)
                proposal_rows.append(q)

            lambda_weight = float(lambdas[head_id].item())
            proposal_prob = float(value.item())
            if proposal_prob <= 0.0 or (
                scorer_mode == "lambda_q" and lambda_weight <= 0.0
            ):
                continue
            token_id = int(token_id_tensor.item())
            selected_token_ids.add(token_id)
            selected.append(
                _SelectedHeadCandidate(
                    head_id=head_id,
                    token_id=token_id,
                    raw_q=raw_q,
                    conditioned_q=q,
                    proposal_row=proposal_row,
                    lambda_weight=lambda_weight,
                    proposal_probability=proposal_prob,
                )
            )

        if len(selected) != len(head_ids):
            raise ValueError(
                "every active residual head must contribute one positive-probability "
                "candidate"
            )

        features = (
            _tree_listwise_features(
                selected,
                depth=parent.depth + 1,
                top_k=scorer_top_k,
            )
            if scorer_mode != "lambda_q" or collect_scoring_records
            else None
        )
        class_probabilities: tuple[float, ...] | None = None
        if scorer_mode == "lambda_q":
            candidate_priorities = tuple(
                parent.priority
                * candidate.lambda_weight
                * candidate.proposal_probability
                for candidate in selected
            )
        else:
            assert features is not None
            class_probabilities = _resolve_listwise_class_probabilities(
                scorer_mode=scorer_mode,
                features=features,
                head_prior_probabilities=head_prior_probabilities,
                listwise_scorer=listwise_scorer,
            )
            candidate_priorities = tuple(
                parent.priority * probability
                for probability in class_probabilities[:-1]
            )

        merged: dict[int, list[TreeCandidateContributor]] = {}
        for candidate, score in zip(selected, candidate_priorities):
            contributor = TreeCandidateContributor(
                head_id=candidate.head_id,
                proposal_row=candidate.proposal_row,
                token_id=candidate.token_id,
                proposal_prob=candidate.proposal_probability,
                lambda_weight=candidate.lambda_weight,
                score=score,
            )
            merged.setdefault(candidate.token_id, []).append(contributor)

        if collect_scoring_records:
            assert features is not None
            scoring_records.append(
                TreeStateScoringRecord(
                    parent_id=parent_id,
                    parent_priority=parent.priority,
                    features=features,
                    candidate_priorities=candidate_priorities,
                    class_probabilities=class_probabilities,
                    entered_node_ids=[None] * len(selected),
                )
            )

        for token_id, contributors in merged.items():
            contributors_tuple = tuple(
                sorted(contributors, key=lambda item: item.score, reverse=True)
            )
            merged_score = sum(item.score for item in contributors_tuple)
            heapq.heappush(
                destination,
                (-merged_score, heap_counter, parent_id, token_id, contributors_tuple),
            )
            heap_counter += 1

    def materialize_candidate(
        candidate: _QueuedTreeCandidate,
        child_state: Any = unrealized_state,
    ) -> int | None:
        neg_score, _, parent_id, token_id, contributors = candidate
        score = -neg_score

        # A parent can receive the same token only once because candidates are
        # merged before enqueueing. Keep this guard for stale heap entries from
        # future extensions that may enqueue incrementally.
        if any(
            nodes[child_id].token_id == token_id for child_id in children[parent_id]
        ):
            return None

        node_id = len(nodes)
        node = ResidualTreeNode(
            node_id=node_id,
            parent_id=parent_id,
            token_id=token_id,
            depth=nodes[parent_id].depth + 1,
            priority=score,
            contributors=contributors,
        )
        nodes.append(node)
        children.append([])
        children[parent_id].append(node_id)
        states.append(child_state)
        if collect_scoring_records:
            for record in reversed(scoring_records):
                if record.parent_id != parent_id:
                    continue
                for index, candidate_token_id in enumerate(
                    record.features.candidate_token_ids
                ):
                    if candidate_token_id == token_id:
                        record.entered_node_ids[index] = node_id
                break
        return node_id

    def prefetch_best_first_candidates() -> None:
        if proposal_batch_fn is None or transition_batch_fn is None:
            return

        # The final selected node cannot be expanded, so at most
        # remaining_budget - 1 prefetched states can be consumed.
        remaining_budget = node_budget - (len(nodes) - 1)
        prefetch_limit = min(
            _BEST_FIRST_PREFETCH_LIMIT,
            max(0, remaining_budget - 1),
        )
        if prefetch_limit == 0:
            return

        candidates: list[_QueuedTreeCandidate] = []
        for candidate in sorted(heap):
            _, candidate_id, parent_id, token_id, _ = candidate
            if candidate_id in prefetched_states:
                continue
            child_depth = nodes[parent_id].depth + 1
            if max_depth is not None and child_depth >= max_depth:
                continue
            if any(
                nodes[child_id].token_id == token_id for child_id in children[parent_id]
            ):
                continue
            candidates.append(candidate)
            if len(candidates) >= prefetch_limit:
                break
        if not candidates:
            return

        parent_ids = [candidate[2] for candidate in candidates]
        realize_states(parent_ids)
        parent_states = [states[parent_id] for parent_id in parent_ids]
        token_ids = [candidate[3] for candidate in candidates]
        child_states = list(transition_batch_fn(parent_states, token_ids))
        if len(child_states) != len(candidates):
            raise ValueError(
                "transition_batch_fn must return one state per input transition"
            )
        child_head_probs = list(proposal_batch_fn(child_states))
        if len(child_head_probs) != len(candidates):
            raise ValueError("proposal_batch_fn must return one proposal set per state")

        for candidate, child_state, head_probs in zip(
            candidates,
            child_states,
            child_head_probs,
        ):
            candidate_id = candidate[1]
            prefetched_states[candidate_id] = child_state
            prefetched_head_probs[candidate_id] = head_probs

    if node_budget > 0 and tree_policy == "best_first":
        enqueue_candidates(0, head_probs_override=root_head_probs)
        while heap and len(nodes) - 1 < node_budget:
            prefetch_best_first_candidates()
            candidate = heapq.heappop(heap)
            candidate_id = candidate[1]
            child_state = prefetched_states.pop(
                candidate_id,
                unrealized_state,
            )
            head_probs = prefetched_head_probs.pop(candidate_id, None)
            node_id = materialize_candidate(candidate, child_state)
            if node_id is not None and len(nodes) - 1 < node_budget:
                enqueue_candidates(
                    node_id,
                    head_probs_override=head_probs,
                )
    elif node_budget > 0:
        frontier = [0]
        while frontier and len(nodes) - 1 < node_budget:
            expandable_frontier = [
                parent_id
                for parent_id in frontier
                if max_depth is None or nodes[parent_id].depth < max_depth
            ]
            if not expandable_frontier:
                break

            realize_states(expandable_frontier)
            if frontier == [0] and root_head_probs is not None:
                level_head_probs = [root_head_probs]
            elif proposal_batch_fn is None:
                level_head_probs = [
                    proposal_fn(states[parent_id]) for parent_id in expandable_frontier
                ]
            else:
                level_states = [states[parent_id] for parent_id in expandable_frontier]
                level_head_probs = list(proposal_batch_fn(level_states))
                if len(level_head_probs) != len(expandable_frontier):
                    raise ValueError(
                        "proposal_batch_fn must return one proposal set per state"
                    )

            level_candidates: list[_QueuedTreeCandidate] = []
            for parent_id, head_probs in zip(
                expandable_frontier,
                level_head_probs,
            ):
                enqueue_candidates(
                    parent_id,
                    level_candidates,
                    head_probs_override=head_probs,
                )
            if not level_candidates:
                break

            level_candidates.sort(key=lambda item: (item[0], item[2], item[3], item[1]))
            next_frontier: list[int] = []
            for candidate in level_candidates:
                if len(nodes) - 1 >= node_budget:
                    break
                node_id = materialize_candidate(candidate)
                if node_id is not None:
                    next_frontier.append(node_id)
            frontier = next_frontier

    proposal_probs = None
    if proposal_rows:
        proposal_probs = torch.stack(proposal_rows, dim=0).contiguous()

    return ResidualTree(
        nodes=nodes,
        children=children,
        proposal_probs=proposal_probs,
        states=[None if state is unrealized_state else state for state in states],
        scoring_records=scoring_records,
    )


def select_batched_b2d1_residual_trees(
    *,
    root_states: Sequence[Any],
    root_head_probs: torch.Tensor,
    head_lambdas: torch.Tensor | Sequence[float],
    active_heads: Sequence[int] | torch.Tensor | None = None,
    tree_policy: TreePolicy = "best_first",
    eps: float = _EPS,
) -> tuple[list[ResidualTree], torch.Tensor]:
    """Select the exact B2D1 distinct-head ``lambda_q`` trees in one batch.

    This is a deliberately narrow serving fast path for two sibling draft
    nodes at depth one. It reproduces the normalization and ordered
    conditioning performed by :func:`select_residual_tree`, but launches the
    full-vocabulary work once per batch and transfers one compact packet of
    selected tokens and probabilities to the CPU for tree construction.

    The returned proposal tensor is request-major and proposal-head-major with
    shape ``[2 * batch_size, vocab_size]``. Each request-local tree holds the
    corresponding two-row view, so contributor proposal-row numbering remains
    local until metadata packing applies request offsets.
    """

    if root_head_probs.ndim != 3:
        raise ValueError(
            "batched root head probabilities must have shape [batch, num_heads, vocab]"
        )
    batch_size, num_heads, vocab_size = root_head_probs.shape
    if batch_size != len(root_states):
        raise ValueError("batched root probabilities must match root states")
    if batch_size == 0:
        raise ValueError("at least one root state is required")
    if vocab_size < 2:
        raise ValueError("B2D1 distinct selection requires at least two tokens")
    if tree_policy not in ("best_first", "breadth_first"):
        raise ValueError(f"unsupported tree_policy: {tree_policy}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")

    head_ids = _resolve_active_heads(active_heads, num_heads)
    if len(head_ids) != 2:
        raise ValueError("batched B2D1 selection requires exactly two active heads")

    lambdas = torch.as_tensor(head_lambdas, dtype=torch.float32)
    if lambdas.ndim != 1:
        raise ValueError("head_lambdas must be a 1-D tensor or sequence")
    if head_ids[-1] >= lambdas.shape[0]:
        raise ValueError(f"lambda for active head {head_ids[-1]} is missing")

    if head_ids == list(range(num_heads)):
        selected_probs = root_head_probs.detach()
    else:
        head_index = torch.tensor(
            head_ids,
            dtype=torch.long,
            device=root_head_probs.device,
        )
        selected_probs = root_head_probs.detach().index_select(1, head_index)
    normalized = _normalize_rows(selected_probs, eps=eps)
    first_q = normalized[:, 0, :]
    second_raw_q = normalized[:, 1, :]

    first_values, first_tokens = torch.max(first_q, dim=-1)

    # _condition_probability_vector normalizes the already-normalized raw row
    # once more before blocking the earlier head's token. Keep that operation
    # here so the fast path follows the same floating-point computation.
    second_q = _normalize_rows(second_raw_q, eps=eps)
    blocked_indices = first_tokens.unsqueeze(-1)
    second_q = second_q.clone()
    second_q.scatter_(1, blocked_indices, 0.0)
    remaining_mass = second_q.sum(dim=-1, keepdim=True)
    use_uniform_fallback = remaining_mass <= eps
    safe_remaining_mass = torch.where(
        use_uniform_fallback,
        torch.ones_like(remaining_mass),
        remaining_mass,
    )
    second_q = second_q / safe_remaining_mass
    second_q = torch.where(
        use_uniform_fallback,
        torch.full_like(second_q, 1.0 / (vocab_size - 1)),
        second_q,
    )
    second_q.scatter_(1, blocked_indices, 0.0)
    second_values, second_tokens = torch.max(second_q, dim=-1)

    flat_proposal_probs = torch.stack((first_q, second_q), dim=1).reshape(
        batch_size * 2,
        vocab_size,
    )
    active_lambdas = lambdas[head_ids].to(device=root_head_probs.device)
    compact_packet = torch.stack(
        (
            first_tokens.to(torch.float64),
            second_tokens.to(torch.float64),
            first_values.to(torch.float64),
            second_values.to(torch.float64),
            active_lambdas[0].to(torch.float64).expand(batch_size),
            active_lambdas[1].to(torch.float64).expand(batch_size),
        ),
        dim=1,
    ).cpu()

    trees: list[ResidualTree] = []
    for request_index, packet_row in enumerate(compact_packet.tolist()):
        first_token = int(packet_row[0])
        second_token = int(packet_row[1])
        first_probability = float(packet_row[2])
        second_probability = float(packet_row[3])
        first_lambda = float(packet_row[4])
        second_lambda = float(packet_row[5])
        if first_token == second_token:
            raise AssertionError(
                "distinct-head conditioning selected a duplicate token"
            )
        if (
            first_probability <= 0.0
            or second_probability <= 0.0
            or first_lambda <= 0.0
            or second_lambda <= 0.0
        ):
            raise ValueError(
                "every active residual head must contribute one positive-probability "
                "candidate"
            )

        candidates = [
            (
                first_lambda * first_probability,
                0,
                first_token,
                head_ids[0],
                first_probability,
                first_lambda,
            ),
            (
                second_lambda * second_probability,
                1,
                second_token,
                head_ids[1],
                second_probability,
                second_lambda,
            ),
        ]
        if tree_policy == "best_first":
            candidates.sort(key=lambda item: (-item[0], item[1]))
        else:
            candidates.sort(key=lambda item: (-item[0], item[2], item[1]))

        nodes = [
            ResidualTreeNode(
                node_id=0,
                parent_id=-1,
                token_id=-1,
                depth=0,
                priority=1.0,
            )
        ]
        for node_id, candidate in enumerate(candidates, start=1):
            score, proposal_row, token_id, head_id, probability, lambda_weight = (
                candidate
            )
            contributor = TreeCandidateContributor(
                head_id=head_id,
                proposal_row=proposal_row,
                token_id=token_id,
                proposal_prob=probability,
                lambda_weight=lambda_weight,
                score=score,
            )
            nodes.append(
                ResidualTreeNode(
                    node_id=node_id,
                    parent_id=0,
                    token_id=token_id,
                    depth=1,
                    priority=score,
                    contributors=(contributor,),
                )
            )

        proposal_start = request_index * 2
        trees.append(
            ResidualTree(
                nodes=nodes,
                children=[[1, 2], [], []],
                proposal_probs=flat_proposal_probs[proposal_start : proposal_start + 2],
                states=[root_states[request_index], None, None],
            )
        )

    return trees, flat_proposal_probs


def select_batched_greedy_residual_trees(
    *,
    root_states: Sequence[Any],
    root_candidate_tokens: torch.Tensor,
    root_candidate_probabilities: torch.Tensor,
    candidate_batch_fn: CompactCandidateBatchFn,
    transition_batch_fn: TransitionBatchFn,
    head_lambdas: torch.Tensor | Sequence[float],
    node_budget: int,
    max_depth: int | None = None,
    active_heads: Sequence[int] | torch.Tensor | None = None,
    tree_policy: TreePolicy = "best_first",
) -> list[ResidualTree]:
    """Build arbitrary greedy residual trees with cross-request batching.

    Candidate projection stays on device and returns only one token and its
    conditioned probability per ordered head.  Tree policy remains a CPU
    priority-queue operation, but each scheduling round transfers one compact
    packet and batches every requested transition across serving requests.
    No full-vocabulary proposal rows are retained because greedy verification
    consumes only the selected tree tokens.

    This routine is topology-generic: ``node_budget`` and ``max_depth`` do not
    select an implementation.  Best-first preserves the scalar selector's
    bounded prefetch policy independently for each request; breadth-first
    batches complete levels across the whole request batch.
    """

    if node_budget < 0:
        raise ValueError("node_budget must be non-negative")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive when provided")
    if tree_policy not in ("best_first", "breadth_first"):
        raise ValueError(f"unsupported tree_policy: {tree_policy}")
    if not root_states:
        return []
    if (
        root_candidate_tokens.ndim != 2
        or root_candidate_probabilities.shape != root_candidate_tokens.shape
        or root_candidate_tokens.shape[0] != len(root_states)
    ):
        raise ValueError(
            "root candidate tokens/probabilities must have matching "
            "[batch, num_heads] shapes"
        )

    num_heads = int(root_candidate_tokens.shape[1])
    head_ids = _resolve_active_heads(active_heads, num_heads)
    lambdas_tensor = torch.as_tensor(head_lambdas, dtype=torch.float32)
    if lambdas_tensor.ndim != 1:
        raise ValueError("head_lambdas must be a 1-D tensor or sequence")
    if head_ids and head_ids[-1] >= lambdas_tensor.numel():
        raise ValueError(f"lambda for active head {head_ids[-1]} is missing")
    lambda_values = [float(value) for value in lambdas_tensor.detach().cpu().tolist()]
    if any(
        not math.isfinite(lambda_values[head_id]) or lambda_values[head_id] <= 0.0
        for head_id in head_ids
    ):
        raise ValueError("active greedy residual-head lambdas must be positive")

    unrealized_state = object()
    builders: list[dict[str, Any]] = []
    for root_state in root_states:
        builders.append(
            {
                "nodes": [ResidualTreeNode(0, -1, -1, 0, 1.0)],
                "children": [[]],
                "states": [root_state],
                "heap": [],
                "counter": 0,
                "prefetched": {},
                "frontier": [0],
            }
        )

    def compact_rows(
        token_tensor: torch.Tensor,
        probability_tensor: torch.Tensor,
        expected_batch_size: int,
    ) -> list[tuple[list[int], list[float]]]:
        if (
            token_tensor.ndim != 2
            or probability_tensor.shape != token_tensor.shape
            or token_tensor.shape[0] != expected_batch_size
            or token_tensor.shape[1] != num_heads
        ):
            raise ValueError(
                "compact residual candidates must have shape [batch, num_heads]"
            )
        if head_ids == list(range(num_heads)):
            selected_tokens = token_tensor
            selected_probabilities = probability_tensor
        else:
            head_index = torch.tensor(
                head_ids,
                dtype=torch.long,
                device=token_tensor.device,
            )
            selected_tokens = token_tensor.index_select(1, head_index)
            selected_probabilities = probability_tensor.index_select(1, head_index)
        packet = (
            torch.cat(
                (
                    selected_tokens.to(dtype=torch.float64),
                    selected_probabilities.to(dtype=torch.float64),
                ),
                dim=1,
            )
            .cpu()
            .tolist()
        )
        width = len(head_ids)
        rows: list[tuple[list[int], list[float]]] = []
        for packet_row in packet:
            token_row = [int(value) for value in packet_row[:width]]
            probability_row = [float(value) for value in packet_row[width:]]
            if any(
                not math.isfinite(value) or value <= 0.0 for value in probability_row
            ):
                raise ValueError(
                    "every active residual head must contribute one "
                    "positive-probability candidate"
                )
            rows.append((token_row, probability_row))
        return rows

    def enqueue_candidates(
        builder: dict[str, Any],
        parent_id: int,
        row: tuple[list[int], list[float]],
        destination: list[_QueuedTreeCandidate] | None = None,
    ) -> None:
        nodes: list[ResidualTreeNode] = builder["nodes"]
        parent = nodes[parent_id]
        if max_depth is not None and parent.depth >= max_depth:
            return
        if destination is None:
            destination = builder["heap"]
        token_row, probability_row = row
        contributors_by_token: dict[int, list[TreeCandidateContributor]] = {}
        for column, head_id in enumerate(head_ids):
            token_id = token_row[column]
            probability = probability_row[column]
            lambda_weight = lambda_values[head_id]
            score = parent.priority * lambda_weight * probability
            contributors_by_token.setdefault(token_id, []).append(
                TreeCandidateContributor(
                    head_id=head_id,
                    proposal_row=-1,
                    token_id=token_id,
                    proposal_prob=probability,
                    lambda_weight=lambda_weight,
                    score=score,
                )
            )
        for token_id, contributors in contributors_by_token.items():
            score = sum(contributor.score for contributor in contributors)
            counter = int(builder["counter"])
            heapq.heappush(
                destination,
                (
                    -score,
                    counter,
                    parent_id,
                    token_id,
                    tuple(contributors),
                ),
            )
            builder["counter"] = counter + 1

    def materialize_candidate(
        builder: dict[str, Any],
        candidate: _QueuedTreeCandidate,
        child_state: Any = unrealized_state,
    ) -> int | None:
        neg_score, _, parent_id, token_id, contributors = candidate
        nodes: list[ResidualTreeNode] = builder["nodes"]
        children: list[list[int]] = builder["children"]
        if any(
            nodes[child_id].token_id == token_id for child_id in children[parent_id]
        ):
            return None
        node_id = len(nodes)
        nodes.append(
            ResidualTreeNode(
                node_id=node_id,
                parent_id=parent_id,
                token_id=token_id,
                depth=nodes[parent_id].depth + 1,
                priority=-neg_score,
                contributors=contributors,
            )
        )
        children.append([])
        children[parent_id].append(node_id)
        builder["states"].append(child_state)
        return node_id

    def project_states(
        states: list[Any],
    ) -> tuple[list[Any], list[tuple[list[int], list[float]]]]:
        if not states:
            return [], []
        output = candidate_batch_fn(states)
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            raise TypeError(
                "compact residual candidate callback must return token/prob tensors"
            )
        rows = compact_rows(output[0], output[1], len(states))
        return states, rows

    root_rows = compact_rows(
        root_candidate_tokens,
        root_candidate_probabilities,
        len(root_states),
    )

    if node_budget > 0 and tree_policy == "best_first":
        for builder, root_row in zip(builders, root_rows, strict=True):
            enqueue_candidates(builder, 0, root_row)

        while True:
            prefetch_requests: list[tuple[int, _QueuedTreeCandidate]] = []
            for builder_index, builder in enumerate(builders):
                remaining_budget = node_budget - (len(builder["nodes"]) - 1)
                prefetch_limit = min(
                    _BEST_FIRST_PREFETCH_LIMIT,
                    max(0, remaining_budget - 1),
                )
                if prefetch_limit == 0:
                    continue
                prefetched: dict[int, Any] = builder["prefetched"]
                selected = 0
                for candidate in sorted(builder["heap"]):
                    _, candidate_id, parent_id, token_id, _ = candidate
                    if candidate_id in prefetched:
                        continue
                    child_depth = builder["nodes"][parent_id].depth + 1
                    if max_depth is not None and child_depth >= max_depth:
                        continue
                    if any(
                        builder["nodes"][child_id].token_id == token_id
                        for child_id in builder["children"][parent_id]
                    ):
                        continue
                    prefetch_requests.append((builder_index, candidate))
                    selected += 1
                    if selected >= prefetch_limit:
                        break

            if prefetch_requests:
                parent_states = [
                    builders[builder_index]["states"][candidate[2]]
                    for builder_index, candidate in prefetch_requests
                ]
                if any(state is unrealized_state for state in parent_states):
                    raise AssertionError("prefetched parent state is not realized")
                token_ids = [candidate[3] for _, candidate in prefetch_requests]
                child_states = list(transition_batch_fn(parent_states, token_ids))
                if len(child_states) != len(prefetch_requests):
                    raise ValueError(
                        "transition_batch_fn must return one state per transition"
                    )
                _, child_rows = project_states(child_states)
                for (builder_index, candidate), child_state, child_row in zip(
                    prefetch_requests,
                    child_states,
                    child_rows,
                    strict=True,
                ):
                    builders[builder_index]["prefetched"][candidate[1]] = (
                        child_state,
                        child_row,
                    )

            made_progress = False
            for builder in builders:
                if len(builder["nodes"]) - 1 >= node_budget or not builder["heap"]:
                    continue
                candidate = heapq.heappop(builder["heap"])
                child_state, child_row = builder["prefetched"].pop(
                    candidate[1],
                    (unrealized_state, None),
                )
                node_id = materialize_candidate(builder, candidate, child_state)
                if node_id is None:
                    continue
                made_progress = True
                node = builder["nodes"][node_id]
                can_expand = len(builder["nodes"]) - 1 < node_budget and (
                    max_depth is None or node.depth < max_depth
                )
                if can_expand:
                    if child_row is None:
                        raise AssertionError(
                            "expandable best-first candidate was not prefetched"
                        )
                    enqueue_candidates(builder, node_id, child_row)
            if not made_progress:
                break

    elif node_budget > 0:
        level_rows: list[dict[int, tuple[list[int], list[float]]]] = [
            {0: root_row} for root_row in root_rows
        ]
        while True:
            next_frontiers: list[list[int]] = [[] for _ in builders]
            made_progress = False
            for builder_index, builder in enumerate(builders):
                if len(builder["nodes"]) - 1 >= node_budget:
                    continue
                expandable = [
                    parent_id
                    for parent_id in builder["frontier"]
                    if max_depth is None
                    or builder["nodes"][parent_id].depth < max_depth
                ]
                level_candidates: list[_QueuedTreeCandidate] = []
                for parent_id in expandable:
                    enqueue_candidates(
                        builder,
                        parent_id,
                        level_rows[builder_index][parent_id],
                        level_candidates,
                    )
                level_candidates.sort(
                    key=lambda item: (item[0], item[2], item[3], item[1])
                )
                for candidate in level_candidates:
                    if len(builder["nodes"]) - 1 >= node_budget:
                        break
                    node_id = materialize_candidate(builder, candidate)
                    if node_id is not None:
                        next_frontiers[builder_index].append(node_id)
                        made_progress = True

            if not made_progress:
                break

            transition_items: list[tuple[int, int]] = []
            for builder_index, (builder, frontier) in enumerate(
                zip(builders, next_frontiers, strict=True)
            ):
                builder["frontier"] = frontier
                if len(builder["nodes"]) - 1 >= node_budget:
                    continue
                for node_id in frontier:
                    node = builder["nodes"][node_id]
                    if max_depth is None or node.depth < max_depth:
                        transition_items.append((builder_index, node_id))
            if not transition_items:
                break

            parent_states = [
                builders[builder_index]["states"][
                    builders[builder_index]["nodes"][node_id].parent_id
                ]
                for builder_index, node_id in transition_items
            ]
            token_ids = [
                builders[builder_index]["nodes"][node_id].token_id
                for builder_index, node_id in transition_items
            ]
            child_states = list(transition_batch_fn(parent_states, token_ids))
            if len(child_states) != len(transition_items):
                raise ValueError(
                    "transition_batch_fn must return one state per transition"
                )
            _, child_rows = project_states(child_states)
            level_rows = [{} for _ in builders]
            for (builder_index, node_id), child_state, child_row in zip(
                transition_items,
                child_states,
                child_rows,
                strict=True,
            ):
                builders[builder_index]["states"][node_id] = child_state
                level_rows[builder_index][node_id] = child_row

    trees: list[ResidualTree] = []
    for builder in builders:
        trees.append(
            ResidualTree(
                nodes=builder["nodes"],
                children=builder["children"],
                states=[
                    None if state is unrealized_state else state
                    for state in builder["states"]
                ],
            )
        )
    return trees


def select_batched_eagle3_dynamic_trees(
    *,
    root_states: Sequence[Any],
    root_candidate_tokens: torch.Tensor,
    root_candidate_probabilities: torch.Tensor,
    candidate_batch_fn: CompactCandidateBatchFn,
    transition_batch_fn: TransitionBatchFn,
    head_lambdas: torch.Tensor | Sequence[float],
    depth_head_lambdas: torch.Tensor | Sequence[Sequence[float]] | None = None,
    node_budget: int,
    max_depth: int,
    frontier_width: int = 10,
    collect_dynamic_provenance: bool = False,
    root_candidate_provenance: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    diagnostic_target_paths: Sequence[Sequence[int]] | None = None,
    frontier_h2_only_quota: int = 0,
    preserve_spine_count: int = 0,
) -> list[ResidualTree]:
    """Build the released EAGLE-3 width-10, globally pruned tree.

    The root proposal is depth one.  At each following depth, at most
    ``frontier_width`` candidates from the preceding depth are transitioned
    and expanded together.  After reaching ``max_depth``, the globally best
    ``node_budget`` generated candidates are retained.  Candidate scores are
    cumulative path probabilities, so the final selection is ancestor-closed.

    Stock supplies H1 top-10 candidates.  Residual J supplies its ordered H1
    and H2 candidates.  Both therefore share exactly the same expansion and
    pruning implementation while retaining their intended proposal sets.
    """

    if not root_states:
        return []
    if node_budget <= 0:
        raise ValueError("dynamic EAGLE-3 node_budget must be positive")
    if max_depth <= 0:
        raise ValueError("dynamic EAGLE-3 max_depth must be positive")
    if frontier_width <= 0:
        raise ValueError("dynamic EAGLE-3 frontier_width must be positive")
    if not 0 <= frontier_h2_only_quota <= frontier_width:
        raise ValueError("dynamic H2-only frontier quota must fit the frontier")
    if not 0 <= preserve_spine_count <= frontier_width:
        raise ValueError("dynamic preserved-spine count must fit the frontier")
    if frontier_h2_only_quota + preserve_spine_count > frontier_width:
        raise ValueError("dynamic frontier reservations exceed the frontier width")
    if frontier_h2_only_quota and root_candidate_provenance is None:
        raise ValueError("dynamic H2-only frontier quota requires union provenance")
    if preserve_spine_count * max_depth > node_budget:
        raise ValueError("preserved dynamic spines do not fit the node budget")
    if diagnostic_target_paths is not None:
        if not collect_dynamic_provenance:
            raise ValueError(
                "dynamic target-path diagnostics require provenance collection"
            )
        if len(diagnostic_target_paths) != len(root_states):
            raise ValueError("dynamic target paths must match the root-state batch")
        if any(
            len(path) > max_depth
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, Integral)
                or token_id < 0
                for token_id in path
            )
            for path in diagnostic_target_paths
        ):
            raise ValueError(
                "dynamic target paths must contain at most max_depth "
                "non-negative integer token ids"
            )
    if (
        root_candidate_tokens.ndim != 2
        or root_candidate_probabilities.shape != root_candidate_tokens.shape
        or root_candidate_tokens.shape[0] != len(root_states)
    ):
        raise ValueError(
            "dynamic root candidates must have matching [batch, width] shapes"
        )

    candidate_width = int(root_candidate_tokens.shape[1])
    has_candidate_provenance = root_candidate_provenance is not None
    lambdas = torch.as_tensor(head_lambdas, dtype=torch.float64, device="cpu")
    if lambdas.ndim != 1 or int(lambdas.numel()) != candidate_width:
        raise ValueError("dynamic EAGLE-3 requires one lambda per candidate column")
    lambda_values = [float(value) for value in lambdas.tolist()]
    if any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in lambda_values
    ):
        raise ValueError("dynamic EAGLE-3 lambdas must be finite and in (0, 1]")
    if depth_head_lambdas is None:
        depth_lambda_values = [lambda_values] * max_depth
    else:
        depth_lambdas = torch.as_tensor(
            depth_head_lambdas,
            dtype=torch.float64,
            device="cpu",
        )
        if depth_lambdas.shape != (max_depth, candidate_width):
            raise ValueError(
                "dynamic depth-head lambdas must have shape "
                "[max_depth, candidate_width]"
            )
        depth_lambda_values = [
            [float(value) for value in row]
            for row in depth_lambdas.tolist()
        ]
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for row in depth_lambda_values
            for value in row
        ):
            raise ValueError(
                "dynamic depth-head lambdas must be finite and in (0, 1]"
            )

    def compact_rows(
        tokens: torch.Tensor,
        probabilities: torch.Tensor,
        expected_batch: int,
        provenance: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None = None,
    ) -> list[
        tuple[
            list[int],
            list[float],
            list[int] | None,
            list[int] | None,
            list[int] | None,
            list[int] | None,
        ]
    ]:
        if (
            tokens.ndim != 2
            or probabilities.shape != tokens.shape
            or tokens.shape != (expected_batch, candidate_width)
        ):
            raise ValueError(
                "dynamic candidates must preserve the root candidate width"
            )
        tensors = [
            tokens.to(dtype=torch.float64),
            probabilities.to(dtype=torch.float64),
        ]
        if provenance is not None:
            if not collect_dynamic_provenance:
                raise ValueError(
                    "dynamic candidate provenance requires trace collection"
                )
            if len(provenance) != 4 or any(
                value.shape != tokens.shape for value in provenance
            ):
                raise ValueError(
                    "dynamic candidate provenance must contain four [batch, width] "
                    "tensors"
                )
            tensors.extend(value.to(dtype=torch.float64) for value in provenance)
        packet = torch.cat(tensors, dim=1).cpu().tolist()
        rows = []
        for packet_row in packet:
            token_row = [int(value) for value in packet_row[:candidate_width]]
            probability_row = [
                float(value)
                for value in packet_row[candidate_width : 2 * candidate_width]
            ]
            if len(set(token_row)) != candidate_width:
                raise ValueError(
                    "dynamic EAGLE-3 candidates must be distinct within a state"
                )
            if any(
                not math.isfinite(value) or value <= 0.0 or value > 1.0
                for value in probability_row
            ):
                raise ValueError(
                    "dynamic EAGLE-3 candidate probabilities must be in (0, 1]"
                )
            source_masks = h1_ranks = h2_ranks = score_heads = None
            if provenance is not None:
                metadata_rows = [
                    [
                        int(value)
                        for value in packet_row[
                            column * candidate_width : (column + 1) * candidate_width
                        ]
                    ]
                    for column in range(2, 6)
                ]
                source_masks, h1_ranks, h2_ranks, score_heads = metadata_rows
                for index, (source_mask, h1_rank, h2_rank, score_head) in enumerate(
                    zip(
                        source_masks,
                        h1_ranks,
                        h2_ranks,
                        score_heads,
                        strict=True,
                    )
                ):
                    if source_mask not in (1, 2, 3):
                        raise ValueError(
                            f"candidate {index} source mask must be 1, 2, or 3"
                        )
                    if not 0 <= h1_rank <= 10 or not 0 <= h2_rank <= 10:
                        raise ValueError(
                            f"candidate {index} source ranks must be in [0, 10]"
                        )
                    if (h1_rank > 0) != bool(source_mask & 1) or (h2_rank > 0) != bool(
                        source_mask & 2
                    ):
                        raise ValueError(
                            f"candidate {index} ranks disagree with its source mask"
                        )
                    if score_head not in (0, 1) or not (
                        source_mask & (1 << score_head)
                    ):
                        raise ValueError(
                            f"candidate {index} score head is not a supporting head"
                        )
            rows.append(
                (
                    token_row,
                    probability_row,
                    source_masks,
                    h1_ranks,
                    h2_ranks,
                    score_heads,
                )
            )
        return rows

    def project_candidate_rows(
        states: Sequence[Any],
    ) -> list[
        tuple[
            list[int],
            list[float],
            list[int] | None,
            list[int] | None,
            list[int] | None,
            list[int] | None,
        ]
    ]:
        projected = candidate_batch_fn(states)
        if (
            not isinstance(projected, tuple)
            or len(projected) not in (2, 6)
            or not all(isinstance(value, torch.Tensor) for value in projected)
        ):
            raise TypeError(
                "dynamic candidate callback must return token/probability tensors "
                "and optional four-tensor provenance"
            )
        if (len(projected) == 6) != has_candidate_provenance:
            raise ValueError(
                "dynamic candidate provenance must be present at every depth"
            )
        projected_provenance = (
            cast(
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                projected[2:],
            )
            if len(projected) == 6
            else None
        )
        return compact_rows(
            projected[0],
            projected[1],
            len(states),
            projected_provenance,
        )

    builders: list[dict[str, Any]] = []
    for root_state in root_states:
        builder = {
            "nodes": [ResidualTreeNode(0, -1, -1, 0, 1.0)],
            "children": [[]],
            "states": [root_state],
            "frontier": [],
        }
        if collect_dynamic_provenance:
            builder["continued_frontiers"] = []
            builder["union_sources"] = {}
            builder["candidate_rows"] = {}
        if preserve_spine_count:
            builder["preserved_spines"] = []
        builders.append(builder)

    def choose_frontier(
        builder: dict[str, Any],
        candidates: Sequence[int],
        preserved_nodes: Sequence[int] = (),
    ) -> list[int]:
        nodes: list[ResidualTreeNode] = builder["nodes"]
        ranked = sorted(
            candidates,
            key=lambda node_id: (-nodes[node_id].priority, node_id),
        )
        selected: list[int] = []
        selected_set: set[int] = set()

        def reserve(node_id: int) -> None:
            if node_id not in selected_set and len(selected) < frontier_width:
                selected.append(node_id)
                selected_set.add(node_id)

        for node_id in preserved_nodes:
            reserve(node_id)
        if frontier_h2_only_quota:
            union_sources: dict[int, dict[str, int]] = builder["union_sources"]
            h2_only = [
                node_id
                for node_id in ranked
                if union_sources[node_id]["source_mask"] == 2
            ]
            for node_id in h2_only[:frontier_h2_only_quota]:
                reserve(node_id)
        for node_id in ranked:
            reserve(node_id)
        return sorted(
            selected,
            key=lambda node_id: (-nodes[node_id].priority, node_id),
        )

    def add_children(
        builder: dict[str, Any],
        parent_id: int,
        row: tuple[
            list[int],
            list[float],
            list[int] | None,
            list[int] | None,
            list[int] | None,
            list[int] | None,
        ],
    ) -> list[int]:
        nodes: list[ResidualTreeNode] = builder["nodes"]
        children: list[list[int]] = builder["children"]
        states: list[Any | None] = builder["states"]
        parent = nodes[parent_id]
        child_depth_lambdas = depth_lambda_values[parent.depth]
        child_ids = []
        token_row, probability_row, source_masks, h1_ranks, h2_ranks, score_heads = row
        if collect_dynamic_provenance:
            candidate_rows = builder["candidate_rows"]
            if parent_id in candidate_rows:
                raise AssertionError(
                    "dynamic EAGLE-3 parent candidates were recorded twice"
                )
            candidate_rows[parent_id] = row
        for head_id, (token_id, probability) in enumerate(
            zip(token_row, probability_row, strict=True)
        ):
            lambda_weight = child_depth_lambdas[head_id]
            priority = parent.priority * lambda_weight * probability
            node_id = len(nodes)
            contributor = TreeCandidateContributor(
                head_id=head_id,
                proposal_row=-1,
                token_id=token_id,
                proposal_prob=probability,
                lambda_weight=lambda_weight,
                score=priority,
            )
            nodes.append(
                ResidualTreeNode(
                    node_id=node_id,
                    parent_id=parent_id,
                    token_id=token_id,
                    depth=parent.depth + 1,
                    priority=priority,
                    contributors=(contributor,),
                )
            )
            children.append([])
            children[parent_id].append(node_id)
            states.append(None)
            child_ids.append(node_id)
            if source_masks is not None:
                assert h1_ranks is not None
                assert h2_ranks is not None
                assert score_heads is not None
                builder["union_sources"][node_id] = {
                    "source_mask": source_masks[head_id],
                    "h1_rank": h1_ranks[head_id],
                    "h2_rank": h2_ranks[head_id],
                    "score_head_id": score_heads[head_id],
                }
        return child_ids

    root_rows = compact_rows(
        root_candidate_tokens,
        root_candidate_probabilities,
        len(root_states),
        root_candidate_provenance,
    )
    for builder, row in zip(builders, root_rows, strict=True):
        root_children = add_children(builder, 0, row)
        ranked_root_children = sorted(
            root_children,
            key=lambda node_id: (
                -builder["nodes"][node_id].priority,
                node_id,
            ),
        )
        preserved_root_nodes = ranked_root_children[:preserve_spine_count]
        if preserve_spine_count:
            builder["preserved_spines"] = [
                [node_id] for node_id in preserved_root_nodes
            ]
        builder["frontier"] = choose_frontier(
            builder,
            root_children,
            preserved_root_nodes,
        )
        if collect_dynamic_provenance and max_depth > 1:
            builder["continued_frontiers"].append(list(builder["frontier"]))

    for child_depth in range(2, max_depth + 1):
        expansion_items: list[tuple[int, int]] = []
        parent_states: list[Any] = []
        token_ids: list[int] = []
        for builder_index, builder in enumerate(builders):
            for node_id in builder["frontier"]:
                node = builder["nodes"][node_id]
                state = builder["states"][node.parent_id]
                if state is None:
                    raise AssertionError("dynamic frontier parent state is unavailable")
                expansion_items.append((builder_index, node_id))
                parent_states.append(state)
                token_ids.append(node.token_id)
        if not expansion_items:
            break

        transitioned = list(transition_batch_fn(parent_states, token_ids))
        if len(transitioned) != len(expansion_items):
            raise ValueError(
                "dynamic transition callback must return one state per frontier node"
            )
        rows = project_candidate_rows(transitioned)

        generated_by_builder: list[list[int]] = [[] for _ in builders]
        for (builder_index, node_id), state, row in zip(
            expansion_items,
            transitioned,
            rows,
            strict=True,
        ):
            builder = builders[builder_index]
            builder["states"][node_id] = state
            generated_by_builder[builder_index].extend(
                add_children(builder, node_id, row)
            )
        for builder, generated in zip(builders, generated_by_builder, strict=True):
            preserved_children = []
            if preserve_spine_count:
                for spine in builder["preserved_spines"]:
                    tail_children = builder["children"][spine[-1]]
                    if not tail_children:
                        raise RuntimeError("preserved dynamic spine was not expanded")
                    best_child = min(
                        tail_children,
                        key=lambda node_id: (
                            -builder["nodes"][node_id].priority,
                            node_id,
                        ),
                    )
                    spine.append(best_child)
                    preserved_children.append(best_child)
            builder["frontier"] = choose_frontier(
                builder,
                generated,
                preserved_children,
            )
            if collect_dynamic_provenance and child_depth < max_depth:
                builder["continued_frontiers"].append(list(builder["frontier"]))

    if diagnostic_target_paths is not None:
        for builder_index, builder in enumerate(builders):
            target_path = [
                int(token_id) for token_id in diagnostic_target_paths[builder_index]
            ]
            current_state = builder["states"][0]
            parent_full_node_id: int | None = 0
            parent_path_priority = 1.0
            parent_path_token_ids: list[int] = []
            candidate_states: list[dict[str, Any]] = []
            for child_depth, target_token_id in enumerate(target_path, start=1):
                recorded_row = (
                    builder["candidate_rows"].get(parent_full_node_id)
                    if parent_full_node_id is not None
                    else None
                )
                if recorded_row is None:
                    row = project_candidate_rows([current_state])[0]
                    source = "canonical_spine_probe"
                else:
                    row = recorded_row
                    source = "raw_tree_expansion"
                (
                    token_row,
                    probability_row,
                    source_masks,
                    h1_ranks,
                    h2_ranks,
                    score_heads,
                ) = row
                child_depth_lambdas = depth_lambda_values[child_depth - 1]
                entered_by_token: dict[int, int] = {}
                if parent_full_node_id is not None:
                    entered_by_token = {
                        int(builder["nodes"][child_id].token_id): int(child_id)
                        for child_id in builder["children"][parent_full_node_id]
                    }
                candidates = []
                for head_id, (token_id, probability) in enumerate(
                    zip(token_row, probability_row, strict=True)
                ):
                    candidate = {
                        "head_id": head_id,
                        "token_id": token_id,
                        "proposal_probability": probability,
                        "candidate_path_priority": (
                            parent_path_priority
                            * child_depth_lambdas[head_id]
                            * probability
                        ),
                        "entered_full_node_id": entered_by_token.get(token_id),
                    }
                    if source_masks is not None:
                        assert h1_ranks is not None
                        assert h2_ranks is not None
                        assert score_heads is not None
                        candidate.update(
                            {
                                "source_mask": source_masks[head_id],
                                "h1_rank": h1_ranks[head_id],
                                "h2_rank": h2_ranks[head_id],
                                "score_head_id": score_heads[head_id],
                            }
                        )
                    candidates.append(candidate)
                correct_head_id = next(
                    (
                        head_id
                        for head_id, token_id in enumerate(token_row)
                        if token_id == target_token_id
                    ),
                    None,
                )
                candidate_states.append(
                    {
                        "child_depth": child_depth,
                        "parent_full_node_id": parent_full_node_id,
                        "parent_path_token_ids": list(parent_path_token_ids),
                        "parent_path_priority": parent_path_priority,
                        "source": source,
                        "correct_token_id": target_token_id,
                        "correct_token_available": correct_head_id is not None,
                        "correct_head_id": correct_head_id,
                        "candidates": candidates,
                    }
                )
                if correct_head_id is None:
                    break

                correct_probability = probability_row[correct_head_id]
                parent_path_priority *= (
                    child_depth_lambdas[correct_head_id] * correct_probability
                )
                parent_path_token_ids.append(target_token_id)
                matching_child_id = entered_by_token.get(target_token_id)
                matching_child_state = (
                    builder["states"][matching_child_id]
                    if matching_child_id is not None
                    else None
                )
                if matching_child_state is None and child_depth < len(target_path):
                    transitioned = list(
                        transition_batch_fn([current_state], [target_token_id])
                    )
                    if len(transitioned) != 1:
                        raise ValueError(
                            "canonical-spine transition must return exactly one state"
                        )
                    current_state = transitioned[0]
                elif matching_child_state is not None:
                    current_state = matching_child_state
                parent_full_node_id = matching_child_id
            builder["canonical_spine_candidate_states"] = candidate_states

    trees: list[ResidualTree] = []
    for builder_index, builder in enumerate(builders):
        generated_nodes: list[ResidualTreeNode] = builder["nodes"]
        if len(generated_nodes) - 1 < node_budget:
            raise RuntimeError(
                "dynamic EAGLE-3 candidate pool cannot fill the node budget"
            )
        global_priority_order = sorted(
            range(1, len(generated_nodes)),
            key=lambda node_id: (-generated_nodes[node_id].priority, node_id),
        )
        if preserve_spine_count:
            forced_ids = {
                node_id for spine in builder["preserved_spines"] for node_id in spine
            }
            selected_set = set(forced_ids)
            for full_id in global_priority_order:
                if len(selected_set) >= node_budget:
                    break
                if full_id in selected_set:
                    continue
                parent_id = generated_nodes[full_id].parent_id
                if parent_id == 0 or parent_id in selected_set:
                    selected_set.add(full_id)
            if len(selected_set) != node_budget:
                raise RuntimeError("preserved-spine pruning could not fill the budget")
            selected_full_ids = sorted(selected_set)
        else:
            selected_full_ids = sorted(global_priority_order[:node_budget])
        selected_set = set(selected_full_ids)
        for full_id in selected_full_ids:
            parent_id = generated_nodes[full_id].parent_id
            if parent_id != 0 and parent_id not in selected_set:
                raise RuntimeError(
                    "dynamic EAGLE-3 global pruning is not ancestor-closed"
                )

        target_path_diagnostic = None
        if diagnostic_target_paths is not None:
            target_path = [
                int(token_id) for token_id in diagnostic_target_paths[builder_index]
            ]
            continued_full_ids = [
                full_node_id
                for frontier in builder["continued_frontiers"]
                for full_node_id in frontier
            ]
            continued_set = set(continued_full_ids)
            global_ranks = {
                node_id: rank
                for rank, node_id in enumerate(global_priority_order, start=1)
            }
            depth_ranks: dict[int, int] = {}
            depth_cutoffs: dict[int, float] = {}
            for depth in range(1, max_depth + 1):
                ordered = sorted(
                    (
                        node.node_id
                        for node in generated_nodes[1:]
                        if node.depth == depth
                    ),
                    key=lambda node_id: (
                        -generated_nodes[node_id].priority,
                        node_id,
                    ),
                )
                if not ordered:
                    continue
                depth_ranks.update(
                    {
                        node_id: rank
                        for rank, node_id in enumerate(ordered, start=1)
                    }
                )
                depth_cutoffs[depth] = generated_nodes[
                    ordered[min(frontier_width, len(ordered)) - 1]
                ].priority
            global_cutoff_priority = generated_nodes[
                global_priority_order[node_budget - 1]
            ].priority
            retained_token_ids: list[int] = []
            retained_nodes: list[dict[str, Any]] = []
            parent_id = 0
            stop_stage = None
            stop_depth = None
            stop_token_id = None
            stop_details: dict[str, Any] = {}
            for child_depth, target_token_id in enumerate(target_path, start=1):
                if child_depth > 1 and parent_id not in continued_set:
                    parent_depth = generated_nodes[parent_id].depth
                    depth_nodes = sorted(
                        (
                            node.node_id
                            for node in generated_nodes[1:]
                            if node.depth == parent_depth
                        ),
                        key=lambda node_id: (
                            -generated_nodes[node_id].priority,
                            node_id,
                        ),
                    )
                    parent_rank = depth_nodes.index(parent_id) + 1
                    cutoff_index = min(frontier_width, len(depth_nodes)) - 1
                    stop_stage = "continued_frontier_pruned"
                    stop_depth = child_depth
                    stop_token_id = target_token_id
                    stop_details = {
                        "parent_full_node_id": parent_id,
                        "parent_path_priority": generated_nodes[parent_id].priority,
                        "parent_frontier_rank": parent_rank,
                        "frontier_cutoff_priority": generated_nodes[
                            depth_nodes[cutoff_index]
                        ].priority,
                    }
                    break
                matching_children = [
                    child_id
                    for child_id in builder["children"][parent_id]
                    if generated_nodes[child_id].token_id == target_token_id
                ]
                if not matching_children:
                    stop_stage = "absent_after_local_width10"
                    stop_depth = child_depth
                    stop_token_id = target_token_id
                    stop_details = {"parent_full_node_id": parent_id}
                    break
                child_id = matching_children[0]
                child = generated_nodes[child_id]
                if len(child.contributors) != 1:
                    raise AssertionError(
                        "dynamic candidates must have exactly one contributor"
                    )
                contributor = child.contributors[0]
                child_record = {
                    "full_node_id": child_id,
                    "parent_full_node_id": parent_id,
                    "token_id": target_token_id,
                    "depth": child_depth,
                    "head_id": int(contributor.head_id),
                    "proposal_probability": float(contributor.proposal_prob),
                    "path_priority": child.priority,
                    "global_priority_rank": global_ranks[child_id],
                    "same_depth_generated_rank": depth_ranks[child_id],
                    "same_depth_top10_cutoff_priority": depth_cutoffs[child_depth],
                    "continued_frontier": child_id in continued_set,
                    "final_tree": child_id in selected_set,
                }
                retained_nodes.append(child_record)
                if child_id not in selected_set:
                    stop_stage = "final_node_budget_pruned"
                    stop_depth = child_depth
                    stop_token_id = target_token_id
                    stop_details = {
                        "candidate_full_node_id": child_id,
                        "candidate_path_priority": child.priority,
                        "candidate_global_priority_rank": global_ranks[child_id],
                        "candidate_same_depth_generated_rank": depth_ranks[child_id],
                        "same_depth_top10_cutoff_priority": depth_cutoffs[
                            child_depth
                        ],
                        "global_cutoff_priority": global_cutoff_priority,
                    }
                    break
                retained_token_ids.append(target_token_id)
                parent_id = child_id
            if stop_stage is None:
                stop_stage = (
                    "maximum_depth"
                    if len(target_path) == max_depth
                    else "reference_end"
                )
            target_path_diagnostic = {
                "target_token_ids": target_path,
                "retained_path_token_ids": retained_token_ids,
                "retained_path_count": len(retained_token_ids),
                "stop_stage": stop_stage,
                "stop_depth": stop_depth,
                "stop_token_id": stop_token_id,
                "stop_details": stop_details,
                "path_nodes": retained_nodes,
                "fixed_candidate_oracle_path_count": sum(
                    1
                    for state in builder["canonical_spine_candidate_states"]
                    if state["correct_token_available"]
                ),
                "fixed_candidate_oracle_token_ids": [
                    int(state["correct_token_id"])
                    for state in builder["canonical_spine_candidate_states"]
                    if state["correct_token_available"]
                ],
            }

        full_to_local = {0: 0}
        full_to_local.update(
            {full_id: local_id for local_id, full_id in enumerate(selected_full_ids, 1)}
        )
        nodes = [ResidualTreeNode(0, -1, -1, 0, 1.0)]
        children: list[list[int]] = [[] for _ in range(node_budget + 1)]
        for full_id in selected_full_ids:
            source = generated_nodes[full_id]
            local_id = full_to_local[full_id]
            parent_local_id = full_to_local[source.parent_id]
            nodes.append(
                ResidualTreeNode(
                    node_id=local_id,
                    parent_id=parent_local_id,
                    token_id=source.token_id,
                    depth=source.depth,
                    priority=source.priority,
                    contributors=source.contributors,
                )
            )
            children[parent_local_id].append(local_id)
        dynamic_provenance = None
        if collect_dynamic_provenance:

            def summarize(
                full_node_ids: Sequence[int],
                *,
                generated_nodes: Sequence[ResidualTreeNode] = generated_nodes,
            ) -> list[dict[str, Any]]:
                counts: dict[int, dict[int, int]] = {}
                for full_node_id in full_node_ids:
                    source = generated_nodes[full_node_id]
                    if len(source.contributors) != 1:
                        raise AssertionError(
                            "dynamic candidates must have exactly one contributor"
                        )
                    head_id = int(source.contributors[0].head_id)
                    depth_counts = counts.setdefault(int(source.depth), {})
                    depth_counts[head_id] = depth_counts.get(head_id, 0) + 1
                return [
                    {
                        "depth": depth,
                        "total": sum(head_counts.values()),
                        "head_counts": [
                            {"head_id": head_id, "count": count}
                            for head_id, count in sorted(head_counts.items())
                        ],
                    }
                    for depth, head_counts in sorted(counts.items())
                ]

            continued_full_ids = [
                full_node_id
                for frontier in builder["continued_frontiers"]
                for full_node_id in frontier
            ]
            dynamic_provenance = {
                "candidate_width": candidate_width,
                "frontier_width": frontier_width,
                "frontier_h2_only_quota": frontier_h2_only_quota,
                "preserve_spine_count": preserve_spine_count,
                "generated_by_depth": summarize(range(1, len(generated_nodes))),
                "continued_frontier_by_depth": summarize(continued_full_ids),
                "final_tree_by_depth": summarize(selected_full_ids),
            }
            if diagnostic_target_paths is not None:
                dynamic_provenance["canonical_spine_candidate_schema"] = (
                    "ordered_distinct_heads_same_process_v1"
                )
                dynamic_provenance["canonical_spine_candidate_states"] = builder[
                    "canonical_spine_candidate_states"
                ]
            if preserve_spine_count:
                dynamic_provenance["preserved_spines"] = [
                    [int(node_id) for node_id in spine]
                    for spine in builder["preserved_spines"]
                ]
            if target_path_diagnostic is not None:
                dynamic_provenance["canonical_target_path"] = target_path_diagnostic
            union_sources: dict[int, dict[str, int]] = builder["union_sources"]
            if union_sources:

                def summarize_union(
                    full_node_ids: Sequence[int],
                    *,
                    generated_nodes: Sequence[ResidualTreeNode] = generated_nodes,
                    union_sources: dict[int, dict[str, int]] = union_sources,
                ) -> list[dict[str, Any]]:
                    counts: dict[int, dict[str, Any]] = {}
                    source_names = {1: "h1_only", 2: "h2_only", 3: "both"}
                    for full_node_id in full_node_ids:
                        source = generated_nodes[full_node_id]
                        metadata = union_sources[full_node_id]
                        depth_counts = counts.setdefault(
                            int(source.depth),
                            {
                                "total": 0,
                                "source_counts": {},
                                "score_head_counts": {},
                                "h1_rank_counts": {},
                                "h2_rank_counts": {},
                            },
                        )
                        depth_counts["total"] += 1
                        source_name = source_names[metadata["source_mask"]]
                        source_counts = depth_counts["source_counts"]
                        source_counts[source_name] = (
                            source_counts.get(source_name, 0) + 1
                        )
                        score_counts = depth_counts["score_head_counts"]
                        score_head = metadata["score_head_id"]
                        score_counts[score_head] = score_counts.get(score_head, 0) + 1
                        for field in ("h1_rank", "h2_rank"):
                            rank = metadata[field]
                            if rank <= 0:
                                continue
                            rank_counts = depth_counts[f"{field}_counts"]
                            rank_counts[rank] = rank_counts.get(rank, 0) + 1
                    return [
                        {
                            "depth": depth,
                            "total": values["total"],
                            "source_counts": [
                                {"source": source, "count": count}
                                for source, count in sorted(
                                    values["source_counts"].items()
                                )
                            ],
                            "score_head_counts": [
                                {"head_id": head_id, "count": count}
                                for head_id, count in sorted(
                                    values["score_head_counts"].items()
                                )
                            ],
                            "h1_rank_counts": [
                                {"rank": rank, "count": count}
                                for rank, count in sorted(
                                    values["h1_rank_counts"].items()
                                )
                            ],
                            "h2_rank_counts": [
                                {"rank": rank, "count": count}
                                for rank, count in sorted(
                                    values["h2_rank_counts"].items()
                                )
                            ],
                        }
                        for depth, values in sorted(counts.items())
                    ]

                dynamic_provenance.update(
                    {
                        "union_source_schema": "h1_h2_top10_union_v1",
                        "union_generated_by_depth": summarize_union(
                            range(1, len(generated_nodes))
                        ),
                        "union_continued_frontier_by_depth": summarize_union(
                            continued_full_ids
                        ),
                        "union_final_tree_by_depth": summarize_union(selected_full_ids),
                        "union_final_nodes": [
                            {
                                "node_id": full_to_local[full_id],
                                **union_sources[full_id],
                            }
                            for full_id in selected_full_ids
                        ],
                    }
                )
        trees.append(
            ResidualTree(
                nodes=nodes,
                children=children,
                dynamic_provenance=dynamic_provenance,
            )
        )
    return trees


def build_tree_attention_mask(parent_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
    """Return an intra-tree causal mask where each node sees its ancestors.

    The returned bool tensor has shape [num_nodes, num_nodes]. Row i is the
    query node and column j is a visible key node. Prompt KV visibility is
    handled outside this helper by vLLM attention metadata.
    """

    parents = torch.as_tensor(parent_ids, dtype=torch.int64, device="cpu")
    if parents.ndim != 1:
        raise ValueError("parent_ids must be a 1-D tensor or sequence")
    num_nodes = int(parents.numel())
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)

    for node_id in range(num_nodes):
        cursor = node_id
        while cursor >= 0:
            mask[node_id, cursor] = True
            parent = int(parents[cursor].item())
            if parent >= cursor:
                raise ValueError("parent ids must point to earlier nodes")
            cursor = parent

    return mask


def tree_position_offsets(parent_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
    """Return the depth/position offset of each tree node from the root."""

    parents = torch.as_tensor(parent_ids, dtype=torch.int64, device="cpu")
    depths = torch.zeros_like(parents)
    for node_id in range(parents.numel()):
        parent = int(parents[node_id].item())
        if parent >= node_id:
            raise ValueError("parent ids must point to earlier nodes")
        if parent >= 0:
            depths[node_id] = depths[parent] + 1
    return depths


def verify_greedy_tree(
    tree: ResidualTree,
    target_token_ids_by_node: torch.Tensor | Sequence[int],
) -> TreeVerificationResult:
    """Verify a tree with SpecInfer's greedy path rule."""

    target_ids = torch.as_tensor(target_token_ids_by_node, dtype=torch.int64)
    if target_ids.numel() < len(tree.nodes):
        raise ValueError("target_token_ids_by_node must include every tree node")

    output: list[int] = []
    accepted_node_ids: list[int] = []
    current = 0

    while tree.children[current]:
        wanted = int(target_ids[current].item())
        matching_child = None
        for child_id in tree.children[current]:
            if tree.nodes[child_id].token_id == wanted:
                matching_child = child_id
                break
        if matching_child is None:
            output.append(wanted)
            return TreeVerificationResult(
                token_ids=output,
                accepted_node_ids=accepted_node_ids,
                stopped_at_node_id=current,
                fallback_token_id=wanted,
                num_rejected=0,
                residual_masses=[],
            )
        output.append(wanted)
        accepted_node_ids.append(matching_child)
        current = matching_child

    fallback = int(target_ids[current].item())
    output.append(fallback)
    return TreeVerificationResult(
        token_ids=output,
        accepted_node_ids=accepted_node_ids,
        stopped_at_node_id=current,
        fallback_token_id=fallback,
        num_rejected=0,
        residual_masses=[],
    )


def verify_stochastic_tree(
    tree: ResidualTree,
    target_probs_by_node: torch.Tensor,
    proposal_probs: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
    child_order: ChildOrder = "head_id",
    eps: float = _EPS,
) -> TreeVerificationResult:
    """Verify a token tree with residual MSS-style rejection sampling.

    At a prefix node u, each proposal row/head contributes exactly one token and
    is verified once with clipped accepted mass
    a_i = min(R_{i-1}, Z_{i-1} q_i). Only the contributed token's accepted mass
    is subtracted from the dense residual.
    """

    if target_probs_by_node.ndim != 2:
        raise ValueError("target_probs_by_node must have shape [num_nodes, vocab]")
    if target_probs_by_node.shape[0] < len(tree.nodes):
        raise ValueError("target_probs_by_node must include every tree node")
    if proposal_probs is None:
        proposal_probs = tree.proposal_probs
    if proposal_probs is None:
        raise ValueError("proposal probabilities are required for stochastic verify")
    if proposal_probs.ndim != 2:
        raise ValueError("proposal_probs must have shape [num_proposals, vocab]")
    if proposal_probs.shape[1] != target_probs_by_node.shape[1]:
        raise ValueError("proposal and target vocab sizes must match")

    targets = _normalize_rows(target_probs_by_node.to(torch.float32), eps=eps)
    proposals = _normalize_rows(proposal_probs.to(torch.float32), eps=eps)
    output: list[int] = []
    accepted_node_ids: list[int] = []
    residual_masses: list[float] = []
    num_rejected = 0
    current = 0

    while tree.children[current]:
        residual = targets[current].clone()
        accepted_child = None

        child_contributors = _iter_child_contributors(
            tree,
            current,
            child_order=child_order,
            generator=generator,
        )
        proposal_rows = [item.proposal_row for _, item in child_contributors]
        if len(set(proposal_rows)) != len(proposal_rows):
            raise ValueError(
                "each proposal row must contribute exactly one child token"
            )

        for child_id, contributor in child_contributors:
            proposal_row = contributor.proposal_row
            if proposal_row < 0:
                raise ValueError(
                    "stochastic verification requires contributor proposal rows"
                )
            q = proposals[proposal_row]
            mass = residual.sum()
            residual_masses.append(float(mass.item()))
            if float(mass.item()) <= eps:
                break

            accepted_mass = torch.minimum(residual, mass * q)
            token_id = tree.nodes[child_id].token_id
            candidate_mass = accepted_mass[token_id]
            if float(candidate_mass.item()) <= eps:
                continue

            accept_prob = float((candidate_mass / mass).item())
            accept_prob = min(1.0, max(0.0, accept_prob))

            draw = _uniform_scalar(
                device=targets.device,
                generator=generator,
                dtype=torch.float64,
            )
            if draw <= accept_prob:
                output.append(token_id)
                accepted_node_ids.append(child_id)
                accepted_child = child_id
                break

            residual[token_id] = (residual[token_id] - candidate_mass).clamp_min(0.0)
            num_rejected += 1

        if accepted_child is None:
            fallback_source = residual
            # Any positive residual, however small relative to ``eps``, is the
            # exact fallback law. Reverting to the original target here would
            # reintroduce mass already offered by deterministic candidates.
            if float(fallback_source.sum().item()) <= 0.0:
                fallback_source = targets[current]
            fallback_probs = _normalize_vector(fallback_source, eps=eps)
            fallback = _sample_from_probs(fallback_probs, generator=generator)
            output.append(fallback)
            return TreeVerificationResult(
                token_ids=output,
                accepted_node_ids=accepted_node_ids,
                stopped_at_node_id=current,
                fallback_token_id=fallback,
                num_rejected=num_rejected,
                residual_masses=residual_masses,
            )

        current = accepted_child

    fallback = _sample_from_probs(targets[current], generator=generator)
    output.append(fallback)
    return TreeVerificationResult(
        token_ids=output,
        accepted_node_ids=accepted_node_ids,
        stopped_at_node_id=current,
        fallback_token_id=fallback,
        num_rejected=num_rejected,
        residual_masses=residual_masses,
    )


def residual_tree_from_metadata(metadata: Any, req_idx: int) -> ResidualTree:
    """Reconstruct one request-local ResidualTree from TreeSpecDecodeMetadata."""

    if req_idx < 0 or req_idx >= len(metadata.num_draft_tokens):
        raise IndexError("request index is out of range")

    cu_nodes = _tensor_to_int_list(metadata.cu_num_tree_nodes)
    start = 0 if req_idx == 0 else cu_nodes[req_idx - 1]
    end = cu_nodes[req_idx]
    if end <= start:
        raise ValueError("each request must contain a root tree node")

    token_ids = _tensor_to_int_list(metadata.node_token_ids[start:end])
    parent_ids_global = _tensor_to_int_list(metadata.parent_node_ids[start:end])
    depths = _tensor_to_int_list(metadata.node_depths[start:end])
    priorities = _tensor_to_float_list(metadata.node_priorities[start:end])

    contributor_rows = _tensor_to_int_list(metadata.contributor_proposal_rows)
    contributor_heads = _tensor_to_int_list(metadata.contributor_head_ids)
    contributor_scores = _tensor_to_float_list(metadata.contributor_scores)
    contributor_child_ids = _tensor_to_int_list(metadata.contributor_child_node_ids)

    contributors_by_local_node: dict[int, list[TreeCandidateContributor]] = {}
    for child_global, proposal_row, head_id, score in zip(
        contributor_child_ids, contributor_rows, contributor_heads, contributor_scores
    ):
        if child_global < start or child_global >= end:
            continue
        local_child = child_global - start
        token_id = token_ids[local_child]
        contributors_by_local_node.setdefault(local_child, []).append(
            TreeCandidateContributor(
                head_id=head_id,
                proposal_row=proposal_row,
                token_id=token_id,
                proposal_prob=0.0,
                lambda_weight=0.0,
                score=score,
            )
        )

    nodes: list[ResidualTreeNode] = []
    children: list[list[int]] = [[] for _ in range(end - start)]
    for local_id, (token_id, parent_global, depth, priority) in enumerate(
        zip(token_ids, parent_ids_global, depths, priorities)
    ):
        parent_id = -1 if parent_global < 0 else parent_global - start
        if parent_id >= local_id:
            raise ValueError("parent ids must point to earlier nodes")
        if parent_id >= 0:
            children[parent_id].append(local_id)

        node_contributors = tuple(
            sorted(
                contributors_by_local_node.get(local_id, ()),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        nodes.append(
            ResidualTreeNode(
                node_id=local_id,
                parent_id=parent_id,
                token_id=token_id,
                depth=depth,
                priority=priority,
                contributors=node_contributors,
            )
        )

    return ResidualTree(nodes=nodes, children=children)


def tree_to_draft_token_tree(tree: ResidualTree) -> DraftTokenTree:
    child_node_ids = [child for children in tree.children for child in children]
    child_start_indices: list[int] = []
    child_end_indices: list[int] = []
    cursor = 0
    for children in tree.children:
        child_start_indices.append(cursor)
        cursor += len(children)
        child_end_indices.append(cursor)

    contributor_child_node_ids: list[int] = []
    contributor_proposal_rows: list[int] = []
    contributor_head_ids: list[int] = []
    contributor_scores: list[float] = []
    for node in tree.nodes:
        for contributor in node.contributors:
            contributor_child_node_ids.append(node.node_id)
            contributor_proposal_rows.append(contributor.proposal_row)
            contributor_head_ids.append(contributor.head_id)
            contributor_scores.append(contributor.score)

    proposal_num_rows = None
    if tree.proposal_probs is not None:
        proposal_num_rows = int(tree.proposal_probs.shape[0])

    return DraftTokenTree(
        node_token_ids=[node.token_id for node in tree.nodes],
        parent_node_ids=[node.parent_id for node in tree.nodes],
        node_depths=[node.depth for node in tree.nodes],
        node_priorities=[node.priority for node in tree.nodes],
        child_start_indices=child_start_indices,
        child_end_indices=child_end_indices,
        child_node_ids=child_node_ids,
        contributor_child_node_ids=contributor_child_node_ids,
        contributor_proposal_rows=contributor_proposal_rows,
        contributor_head_ids=contributor_head_ids,
        contributor_scores=contributor_scores,
        proposal_num_rows=proposal_num_rows,
    )


def trees_to_metadata(
    trees: Sequence[ResidualTree | DraftTokenTree],
    *,
    device: torch.device | str | None = None,
    target_logits_indices: torch.Tensor | None = None,
    logits_indices: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor | None]:
    """Pack request-local ResidualTrees into TreeSpecDecodeMetadata."""

    from vllm.v1.spec_decode.metadata import TreeSpecDecodeMetadata

    node_token_ids: list[int] = []
    parent_node_ids: list[int] = []
    node_request_indices: list[int] = []
    node_depths: list[int] = []
    node_priorities: list[float] = []
    root_node_ids: list[int] = []
    num_draft_tokens: list[int] = []
    num_proposal_rows: list[int] = []
    trace_ids: list[str | None] = []
    cu_num_tree_nodes: list[int] = []
    child_start_indices: list[int] = []
    child_end_indices: list[int] = []
    child_node_ids: list[int] = []
    contributor_child_node_ids: list[int] = []
    contributor_proposal_rows: list[int] = []
    contributor_head_ids: list[int] = []
    contributor_scores: list[float] = []
    proposal_batches: list[torch.Tensor] = []

    node_offset = 0
    proposal_offset = 0
    for req_idx, input_tree in enumerate(trees):
        tree = (
            tree_to_draft_token_tree(input_tree)
            if isinstance(input_tree, ResidualTree)
            else input_tree
        )
        root_node_ids.append(node_offset)
        num_draft_tokens.append(tree.num_draft_tokens)
        trace_ids.append(tree.trace_id)
        if (
            isinstance(input_tree, ResidualTree)
            and input_tree.proposal_probs is not None
        ):
            tree_num_proposal_rows = int(input_tree.proposal_probs.shape[0])
        else:
            tree_num_proposal_rows = tree.num_proposal_rows
        num_proposal_rows.append(tree_num_proposal_rows)
        for local_node_id in range(tree.num_tree_nodes):
            node_token_ids.append(tree.node_token_ids[local_node_id])
            parent_id = tree.parent_node_ids[local_node_id]
            parent_node_ids.append(-1 if parent_id < 0 else node_offset + parent_id)
            node_request_indices.append(req_idx)
            node_depths.append(tree.node_depths[local_node_id])
            node_priorities.append(tree.node_priorities[local_node_id])
            start = len(child_node_ids)
            child_start = tree.child_start_indices[local_node_id]
            child_end = tree.child_end_indices[local_node_id]
            child_node_ids.extend(
                node_offset + child
                for child in tree.child_node_ids[child_start:child_end]
            )
            child_start_indices.append(start)
            child_end_indices.append(len(child_node_ids))

        for child_id, row, head_id, score in zip(
            tree.contributor_child_node_ids,
            tree.contributor_proposal_rows,
            tree.contributor_head_ids,
            tree.contributor_scores,
        ):
            contributor_child_node_ids.append(node_offset + child_id)
            contributor_proposal_rows.append(-1 if row < 0 else proposal_offset + row)
            contributor_head_ids.append(head_id)
            contributor_scores.append(score)

        if (
            isinstance(input_tree, ResidualTree)
            and input_tree.proposal_probs is not None
        ):
            proposal_batches.append(input_tree.proposal_probs)
        proposal_offset += tree_num_proposal_rows
        node_offset += tree.num_tree_nodes
        cu_num_tree_nodes.append(node_offset)

    num_nodes = len(node_token_ids)
    if target_logits_indices is None:
        target_logits_indices = torch.arange(num_nodes, dtype=torch.int32)
    if logits_indices is None:
        logits_indices = torch.arange(num_nodes, dtype=torch.int32)

    metadata = TreeSpecDecodeMetadata(
        node_token_ids=torch.tensor(node_token_ids, dtype=torch.int32, device=device),
        parent_node_ids=torch.tensor(parent_node_ids, dtype=torch.int32, device=device),
        node_request_indices=torch.tensor(
            node_request_indices, dtype=torch.int32, device=device
        ),
        node_depths=torch.tensor(node_depths, dtype=torch.int32, device=device),
        node_priorities=torch.tensor(
            node_priorities, dtype=torch.float32, device=device
        ),
        root_node_ids=torch.tensor(root_node_ids, dtype=torch.int32, device=device),
        num_draft_tokens=num_draft_tokens,
        num_proposal_rows=num_proposal_rows,
        cu_num_tree_nodes=torch.tensor(
            cu_num_tree_nodes, dtype=torch.int32, device=device
        ),
        child_start_indices=torch.tensor(
            child_start_indices, dtype=torch.int32, device=device
        ),
        child_end_indices=torch.tensor(
            child_end_indices, dtype=torch.int32, device=device
        ),
        child_node_ids=torch.tensor(child_node_ids, dtype=torch.int32, device=device),
        contributor_child_node_ids=torch.tensor(
            contributor_child_node_ids, dtype=torch.int32, device=device
        ),
        contributor_proposal_rows=torch.tensor(
            contributor_proposal_rows, dtype=torch.int32, device=device
        ),
        contributor_head_ids=torch.tensor(
            contributor_head_ids, dtype=torch.int32, device=device
        ),
        contributor_scores=torch.tensor(
            contributor_scores, dtype=torch.float32, device=device
        ),
        target_logits_indices=target_logits_indices.to(device=device),
        logits_indices=logits_indices.to(device=device),
        trace_ids=trace_ids,
        max_tree_depth=max(node_depths, default=0),
    )
    proposal_probs = None
    if proposal_batches:
        proposal_probs = torch.cat(proposal_batches, dim=0).to(device=device)
    return metadata, proposal_probs


def verify_greedy_tree_batch(
    metadata: Any,
    target_token_ids_by_node: torch.Tensor,
    *,
    placeholder_token_id: int = -1,
) -> torch.Tensor:
    """Run greedy tree verification for each request in a batch."""

    output, _ = verify_greedy_tree_batch_with_nodes(
        metadata,
        target_token_ids_by_node,
        placeholder_token_id=placeholder_token_id,
    )
    return output


def _try_verify_greedy_b2d1_batch_with_nodes(
    metadata: Any,
    target_token_ids_by_node: torch.Tensor,
    *,
    placeholder_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Verify B2D1 trees without copying device metadata to the host.

    The scheduler can prefix-truncate a B2D1 tree or mix it with requests that
    do not have draft tokens.  Consequently this fast path handles every valid
    request-local tree with at most two draft nodes.  Besides the usual B2D1
    sibling layout, the only other valid two-node topology is a length-two
    chain.  Handling both topologies on device avoids a synchronizing layout
    check while preserving the generic verifier's semantics.

    ``None`` means that the batch is outside the narrow fast-path contract and
    must be handled by the generic tree reconstruction below.  Shape and device
    checks only inspect host-visible tensor metadata; no tensor value is read
    on the host.
    """

    draft_counts = metadata.num_draft_tokens
    if not draft_counts or any(count < 0 or count > 2 for count in draft_counts):
        return None
    if target_token_ids_by_node.ndim != 1:
        return None

    batch_size = len(draft_counts)
    expected_num_nodes = sum(count + 1 for count in draft_counts)
    device = target_token_ids_by_node.device
    required_metadata_tensors = (
        metadata.node_token_ids,
        metadata.parent_node_ids,
        metadata.root_node_ids,
        metadata.cu_num_tree_nodes,
    )
    if any(tensor.device != device for tensor in required_metadata_tensors):
        return None
    if (
        metadata.node_token_ids.ndim != 1
        or metadata.node_token_ids.numel() != expected_num_nodes
        or metadata.parent_node_ids.shape != metadata.node_token_ids.shape
        or metadata.root_node_ids.ndim != 1
        or metadata.root_node_ids.numel() != batch_size
        or metadata.cu_num_tree_nodes.ndim != 1
        or metadata.cu_num_tree_nodes.numel() != batch_size
        or target_token_ids_by_node.numel() < expected_num_nodes
    ):
        return None

    max_spec_len = max(draft_counts)
    output = torch.full(
        (batch_size, max_spec_len + 1),
        placeholder_token_id,
        dtype=torch.int32,
        device=device,
    )
    accepted_nodes = torch.full(
        (batch_size, max_spec_len),
        placeholder_token_id,
        dtype=torch.int32,
        device=device,
    )

    target_ids = target_token_ids_by_node.to(dtype=torch.int64)
    node_tokens = metadata.node_token_ids
    parent_ids = metadata.parent_node_ids
    if (
        device.type == "cuda"
        and max_spec_len > 0
        and target_ids.is_contiguous()
        and all(tensor.is_contiguous() for tensor in required_metadata_tensors)
    ):
        # Match the standard speculative decoder's
        # ``rejection_greedy_sample_kernel`` execution shape: one Triton
        # program owns one request and writes directly into placeholder-filled
        # dense output buffers.  The fixed two-node budget makes the tree walk
        # explicit, so no host reconstruction or device-to-host layout check is
        # needed.
        _verify_greedy_b2d1_kernel[(batch_size,)](
            output,
            accepted_nodes,
            metadata.cu_num_tree_nodes,
            metadata.root_node_ids,
            node_tokens,
            parent_ids,
            target_ids,
            placeholder_token_id,
            max_spec_len,
        )
        return output, accepted_nodes

    root_nodes = metadata.root_node_ids.to(dtype=torch.int64)
    request_ends = metadata.cu_num_tree_nodes.to(dtype=torch.int64)

    root_targets = target_ids[root_nodes]
    output[:, 0] = root_targets
    if max_spec_len == 0:
        return output, accepted_nodes

    # Clamp absent child positions to the request root.  The corresponding
    # ``has_*`` mask makes those gathered values semantically inert and keeps
    # the final root-only request from indexing one element past the tensor.
    node_one = torch.minimum(root_nodes + 1, request_ends - 1)
    node_two = torch.minimum(root_nodes + 2, request_ends - 1)
    has_one = request_ends > root_nodes + 1
    has_two = request_ends > root_nodes + 2

    token_one = node_tokens[node_one]
    token_two = node_tokens[node_two]
    parent_one = parent_ids[node_one]
    parent_two = parent_ids[node_two]

    take_one = has_one & (parent_one == root_nodes) & (token_one == root_targets)
    # Children are considered in local node-id order, so a duplicate token in
    # both B2D1 heads must select local node 1, exactly as the scalar verifier.
    take_two = (
        ~take_one & has_two & (parent_two == root_nodes) & (token_two == root_targets)
    )
    accepted_root = take_one | take_two
    chosen_node = torch.where(take_one, node_one, node_two)
    chosen_local_node = torch.where(
        take_one,
        torch.ones_like(root_nodes, dtype=torch.int32),
        torch.full_like(root_nodes, 2, dtype=torch.int32),
    )

    placeholder = torch.full(
        (batch_size,),
        placeholder_token_id,
        dtype=torch.int32,
        device=device,
    )
    output[:, 1] = torch.where(
        accepted_root,
        target_ids[chosen_node].to(dtype=torch.int32),
        placeholder,
    )
    accepted_nodes[:, 0] = torch.where(
        accepted_root,
        chosen_local_node,
        placeholder,
    )

    if max_spec_len == 2:
        # In a two-node chain, accepting local node 1 can expose local node 2.
        # B2D1 siblings do not satisfy the parent check and therefore stop at
        # their selected leaf, where output[:, 1] already contains the fallback.
        take_chain_child = (
            take_one
            & has_two
            & (parent_two == node_one)
            & (target_ids[node_one] == token_two)
        )
        output[:, 2] = torch.where(
            take_chain_child,
            target_ids[node_two].to(dtype=torch.int32),
            placeholder,
        )
        accepted_nodes[:, 1] = torch.where(
            take_chain_child,
            torch.full_like(root_nodes, 2, dtype=torch.int32),
            placeholder,
        )

    return output, accepted_nodes


def _try_verify_greedy_tree_device_batch_with_nodes(
    metadata: Any,
    target_token_ids_by_node: torch.Tensor,
    *,
    placeholder_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Verify any packed tree batch on CUDA without host reconstruction."""

    draft_counts = metadata.num_draft_tokens
    if not draft_counts or any(count < 0 for count in draft_counts):
        return None
    if target_token_ids_by_node.ndim != 1:
        return None

    batch_size = len(draft_counts)
    expected_num_nodes = sum(count + 1 for count in draft_counts)
    max_spec_len = max(draft_counts)
    device = target_token_ids_by_node.device
    required_metadata_tensors = (
        metadata.node_token_ids,
        metadata.parent_node_ids,
        metadata.root_node_ids,
        metadata.cu_num_tree_nodes,
    )
    if (
        device.type != "cuda"
        or max_spec_len <= 0
        or any(tensor.device != device for tensor in required_metadata_tensors)
        or not target_token_ids_by_node.is_contiguous()
        or not all(tensor.is_contiguous() for tensor in required_metadata_tensors)
    ):
        return None
    if (
        metadata.node_token_ids.ndim != 1
        or metadata.node_token_ids.numel() != expected_num_nodes
        or metadata.parent_node_ids.shape != metadata.node_token_ids.shape
        or metadata.root_node_ids.ndim != 1
        or metadata.root_node_ids.numel() != batch_size
        or metadata.cu_num_tree_nodes.ndim != 1
        or metadata.cu_num_tree_nodes.numel() != batch_size
        or target_token_ids_by_node.numel() < expected_num_nodes
    ):
        return None

    output = torch.full(
        (batch_size, max_spec_len + 1),
        placeholder_token_id,
        dtype=torch.int32,
        device=device,
    )
    accepted_nodes = torch.full(
        (batch_size, max_spec_len),
        placeholder_token_id,
        dtype=torch.int32,
        device=device,
    )
    _verify_greedy_tree_kernel[(batch_size,)](
        output,
        accepted_nodes,
        metadata.cu_num_tree_nodes,
        metadata.root_node_ids,
        metadata.node_token_ids,
        metadata.parent_node_ids,
        target_token_ids_by_node,
        MAX_SPEC_LEN=max_spec_len,
        MAX_TREE_DEPTH=int(getattr(metadata, "max_tree_depth", None) or max_spec_len),
    )
    return output, accepted_nodes


def verify_greedy_tree_batch_with_nodes(
    metadata: Any,
    target_token_ids_by_node: torch.Tensor,
    *,
    placeholder_token_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run greedy tree verification and return accepted local node ids."""

    b2d1_result = _try_verify_greedy_b2d1_batch_with_nodes(
        metadata,
        target_token_ids_by_node,
        placeholder_token_id=placeholder_token_id,
    )
    if b2d1_result is not None:
        return b2d1_result

    device_result = _try_verify_greedy_tree_device_batch_with_nodes(
        metadata,
        target_token_ids_by_node,
        placeholder_token_id=placeholder_token_id,
    )
    if device_result is not None:
        return device_result

    max_output_len = metadata.max_spec_len + 1
    output = torch.full(
        (len(metadata.num_draft_tokens), max_output_len),
        placeholder_token_id,
        dtype=torch.int32,
        device=target_token_ids_by_node.device,
    )
    accepted_nodes = torch.full(
        (len(metadata.num_draft_tokens), metadata.max_spec_len),
        placeholder_token_id,
        dtype=torch.int32,
        device=target_token_ids_by_node.device,
    )
    cu_nodes = _tensor_to_int_list(metadata.cu_num_tree_nodes)

    for req_idx in range(len(metadata.num_draft_tokens)):
        start = 0 if req_idx == 0 else cu_nodes[req_idx - 1]
        end = cu_nodes[req_idx]
        tree = residual_tree_from_metadata(metadata, req_idx)
        result = verify_greedy_tree(tree, target_token_ids_by_node[start:end])
        token_ids = result.token_ids[:max_output_len]
        if token_ids:
            output[req_idx, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.int32, device=output.device
            )
        if result.accepted_node_ids:
            accepted_nodes[req_idx, : len(result.accepted_node_ids)] = torch.tensor(
                result.accepted_node_ids,
                dtype=torch.int32,
                device=accepted_nodes.device,
            )

    return output, accepted_nodes


def verify_stochastic_tree_batch(
    metadata: Any,
    target_probs_by_node: torch.Tensor,
    proposal_probs: torch.Tensor,
    *,
    generators: dict[int, torch.Generator] | None = None,
    child_order: ChildOrder = "head_id",
    placeholder_token_id: int = -1,
) -> torch.Tensor:
    """Run stochastic tree verification for each request in a batch."""

    output, _ = verify_stochastic_tree_batch_with_nodes(
        metadata,
        target_probs_by_node,
        proposal_probs,
        generators=generators,
        child_order=child_order,
        placeholder_token_id=placeholder_token_id,
    )
    return output


def verify_stochastic_tree_batch_with_nodes(
    metadata: Any,
    target_probs_by_node: torch.Tensor,
    proposal_probs: torch.Tensor,
    *,
    generators: dict[int, torch.Generator] | None = None,
    child_order: ChildOrder = "head_id",
    placeholder_token_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run stochastic tree verification and return accepted local node ids."""

    max_output_len = metadata.max_spec_len + 1
    output = torch.full(
        (len(metadata.num_draft_tokens), max_output_len),
        placeholder_token_id,
        dtype=torch.int32,
        device=target_probs_by_node.device,
    )
    accepted_nodes = torch.full(
        (len(metadata.num_draft_tokens), metadata.max_spec_len),
        placeholder_token_id,
        dtype=torch.int32,
        device=target_probs_by_node.device,
    )
    cu_nodes = _tensor_to_int_list(metadata.cu_num_tree_nodes)
    generators = generators or {}

    for req_idx in range(len(metadata.num_draft_tokens)):
        start = 0 if req_idx == 0 else cu_nodes[req_idx - 1]
        end = cu_nodes[req_idx]
        tree = residual_tree_from_metadata(metadata, req_idx)
        result = verify_stochastic_tree(
            tree,
            target_probs_by_node[start:end],
            proposal_probs,
            generator=generators.get(req_idx),
            child_order=child_order,
        )
        token_ids = result.token_ids[:max_output_len]
        if token_ids:
            output[req_idx, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.int32, device=output.device
            )
        if result.accepted_node_ids:
            accepted_nodes[req_idx, : len(result.accepted_node_ids)] = torch.tensor(
                result.accepted_node_ids,
                dtype=torch.int32,
                device=accepted_nodes.device,
            )

    return output, accepted_nodes


def _iter_child_contributors(
    tree: ResidualTree,
    parent_id: int,
    *,
    child_order: ChildOrder,
    generator: torch.Generator | None,
) -> list[tuple[int, TreeCandidateContributor]]:
    entries: list[tuple[int, TreeCandidateContributor]] = []
    for child_id in tree.children[parent_id]:
        for contributor in tree.nodes[child_id].contributors:
            entries.append((child_id, contributor))

    if child_order == "head_id":
        entries.sort(key=lambda item: (item[1].head_id, item[0], -item[1].score))
    elif child_order == "priority":
        entries.sort(key=lambda item: item[1].score, reverse=True)
    elif child_order == "node_id":
        entries.sort(key=lambda item: (item[0], -item[1].score))
    elif child_order == "random":
        if entries:
            order = torch.randperm(len(entries), generator=generator).tolist()
            entries = [entries[i] for i in order]
    else:
        raise ValueError(f"unsupported child_order: {child_order}")
    return entries


def _tree_listwise_features(
    selected: Sequence[_SelectedHeadCandidate],
    *,
    depth: int,
    top_k: int,
) -> TreeListwiseFeatures:
    if not selected:
        raise ValueError("tree scoring requires at least one candidate")

    base_q = selected[0].raw_q
    base_probabilities: list[float] = []
    margins: list[float] = []
    entropies: list[float] = []
    topk_masses: list[float] = []
    for candidate in selected:
        q = candidate.conditioned_q
        if q.ndim != 1 or q.numel() == 0:
            raise ValueError("candidate proposal must be a nonempty vector")
        k = min(int(top_k), int(q.numel()))
        top_values = torch.topk(q, k=k, dim=-1, sorted=True).values
        if q.numel() >= 2:
            top_two = (
                top_values[:2]
                if k >= 2
                else torch.topk(q, k=2, dim=-1, sorted=True).values
            )
            margin = torch.log(top_two[0].clamp_min(_EPS)) - torch.log(
                top_two[1].clamp_min(_EPS)
            )
            margins.append(float(margin.item()))
        else:
            margins.append(0.0)
        positive = q > 0
        entropy = -(q[positive] * torch.log(q[positive])).sum()
        entropies.append(float(entropy.item()))
        topk_masses.append(float(top_values.sum().item()))
        base_probabilities.append(float(base_q[candidate.token_id].item()))

    return TreeListwiseFeatures(
        candidate_token_ids=tuple(candidate.token_id for candidate in selected),
        head_ids=tuple(candidate.head_id for candidate in selected),
        conditioned_proposal_probabilities=tuple(
            candidate.proposal_probability for candidate in selected
        ),
        base_probabilities=tuple(base_probabilities),
        logit_margins=tuple(margins),
        entropies=tuple(entropies),
        topk_masses=tuple(topk_masses),
        depth=int(depth),
        top_k=int(top_k),
    )


def _resolve_listwise_class_probabilities(
    *,
    scorer_mode: TreeScorerMode,
    features: TreeListwiseFeatures,
    head_prior_probabilities: Sequence[float] | torch.Tensor | None,
    listwise_scorer: ListwiseScorerFn | None,
) -> tuple[float, ...]:
    class_count = len(features.candidate_token_ids) + 1
    if len(set(features.candidate_token_ids)) != class_count - 1:
        raise ValueError("listwise scorer candidates must be distinct")
    if scorer_mode == "uniform":
        values = torch.full((class_count,), 1.0 / class_count, dtype=torch.float64)
    elif scorer_mode == "head_prior":
        if head_prior_probabilities is None:
            raise AssertionError("head-prior probabilities were not supplied")
        values = (
            torch.as_tensor(
                head_prior_probabilities,
                dtype=torch.float64,
            )
            .detach()
            .cpu()
        )
    elif scorer_mode == "greedy_listwise":
        if listwise_scorer is None:
            raise AssertionError("listwise scorer callback was not supplied")
        values = (
            torch.as_tensor(
                listwise_scorer(features),
                dtype=torch.float64,
            )
            .detach()
            .cpu()
        )
    else:
        raise AssertionError(f"unexpected listwise scorer mode: {scorer_mode}")

    if values.ndim != 1 or values.numel() != class_count:
        raise ValueError(
            "tree scorer must return one probability per candidate plus none"
        )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("tree scorer returned non-finite probabilities")
    if bool((values < 0.0).any().item()) or bool((values > 1.0).any().item()):
        raise ValueError("tree scorer probabilities must lie in [0, 1]")
    if not math.isclose(float(values.sum().item()), 1.0, abs_tol=1e-6):
        raise ValueError("tree scorer probabilities must sum to one")
    return tuple(float(value) for value in values.tolist())


def _resolve_active_heads(
    active_heads: Sequence[int] | torch.Tensor | None,
    num_heads: int,
) -> list[int]:
    if active_heads is None:
        return list(range(num_heads))
    if isinstance(active_heads, torch.Tensor):
        if active_heads.dtype == torch.bool:
            if active_heads.numel() != num_heads:
                raise ValueError(
                    "boolean active_heads mask must contain one value per head"
                )
            values = (
                torch.nonzero(active_heads.flatten(), as_tuple=False).flatten().tolist()
            )
        else:
            values = active_heads.flatten().tolist()
    else:
        values = list(active_heads)

    if not values:
        raise ValueError("active_heads must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("active_heads must contain only integers")

    head_ids = [int(value) for value in values]
    if head_ids[0] < 0:
        raise ValueError("active head ids must be non-negative")
    if any(current <= previous for previous, current in zip(head_ids, head_ids[1:])):
        raise ValueError("active head ids must be strictly increasing and unique")
    if head_ids[-1] >= num_heads:
        raise ValueError(
            f"active head {head_ids[-1]} is outside the available range "
            f"[0, {num_heads})"
        )
    return head_ids


def _normalize_rows(probs: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    probs = probs.to(torch.float32).clamp_min(0.0)
    if probs.ndim == 3:
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        return probs / denom
    if probs.ndim != 2:
        raise ValueError("expected a 2-D or 3-D probability tensor")
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / denom


def _normalize_vector(probs: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    if probs.ndim != 1:
        raise ValueError("expected a 1-D probability vector")
    probs = probs.to(torch.float32).clamp_min(0.0)
    denom = probs.sum().clamp_min(eps)
    return probs / denom


def _condition_probability_vector(
    probs: torch.Tensor,
    *,
    blocked_token_ids: Sequence[int] | set[int],
    eps: float = _EPS,
) -> torch.Tensor:
    """Condition a proposal on ordered earlier-head tokens being unavailable."""

    normalized = _normalize_vector(probs, eps=eps)
    blocked = sorted({int(token_id) for token_id in blocked_token_ids})
    if not blocked:
        return normalized
    if blocked[0] < 0 or blocked[-1] >= normalized.numel():
        raise ValueError("blocked proposal token is outside the vocabulary")
    if len(blocked) >= normalized.numel():
        raise ValueError("at least one proposal token must remain selectable")
    blocked_tensor = torch.tensor(
        blocked,
        dtype=torch.long,
        device=normalized.device,
    )
    conditioned = normalized.clone()
    conditioned[blocked_tensor] = 0.0
    remaining_mass = conditioned.sum()
    if float(remaining_mass.item()) <= eps:
        remaining = torch.ones_like(conditioned, dtype=torch.bool)
        remaining[blocked_tensor] = False
        conditioned[remaining] = 1.0 / int(remaining.sum().item())
        return conditioned
    return conditioned / remaining_mass


def _uniform_scalar(
    *,
    device: torch.device,
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> float:
    value = torch.rand((), dtype=dtype, device=device, generator=generator)
    return float(value.item())


def _sample_from_probs(
    probs: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> int:
    sample = torch.multinomial(
        probs, num_samples=1, replacement=True, generator=generator
    )
    return int(sample.item())


def _tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(item) for item in tensor.detach().cpu().flatten().tolist()]


def _tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().flatten().tolist()]
