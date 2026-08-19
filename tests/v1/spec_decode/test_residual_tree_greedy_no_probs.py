# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.eagle_residual import ResidualTreeHeadMixin
from vllm.triton_utils import HAS_TRITON
from vllm.v1.spec_decode.llm_base_proposer import (
    _build_greedy_b2d1_draft_trees,
)
from vllm.v1.spec_decode.residual_tree import (
    select_batched_b2d1_residual_trees,
    tree_to_draft_token_tree,
)


class _FixedLogits(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits[: hidden_states.shape[0]]


class _UnexpectedAdapter(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise AssertionError("stock top-2 must not execute a residual adapter")


class _TwoHeadResidualModel(ResidualTreeHeadMixin, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace(vocab_size=5)
        self.residual_tree_total_heads = 2
        self.residual_tree_freeze_base_head = True
        self.residual_tree_adapter_output_mode = "independent_lm_head"
        self.residual_tree_draft_vocab_size = 3
        self.residual_tree_adapters = nn.ModuleList(
            [
                _FixedLogits(
                    torch.tensor(
                        [
                            [1.0, 9.0, 5.0],
                            [1.0, 9.0, 5.0],
                        ]
                    )
                )
            ]
        )
        # draft ids [0, 1, 2] map to target ids [0, 2, 4].
        self.draft_id_to_target_id = nn.Parameter(
            torch.tensor([0, 1, 2], dtype=torch.long),
            requires_grad=False,
        )
        self._residual_tree_cached_draft_target_ids = None
        self._residual_tree_cached_draft_mapping_version = None
        self.lm_head = nn.Identity()
        self._base_draft_logits = torch.tensor(
            [
                [0.0, 10.0, 3.0],
                [0.0, 2.0, 10.0],
            ]
        )

        def logits_processor(_lm_head, hidden_states):
            return self._base_draft_logits[: hidden_states.shape[0]]

        self.logits_processor = logits_processor

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        draft_logits = self.logits_processor(self.lm_head, hidden_states)
        target_ids = torch.arange(3) + self.draft_id_to_target_id
        target_logits = draft_logits.new_full(
            (hidden_states.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        target_logits[:, target_ids] = draft_logits
        return target_logits


class _OrderedFourHeadResidualModel(_TwoHeadResidualModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=5)
        self.residual_tree_total_heads = 4
        self.residual_tree_draft_vocab_size = 5
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(5, dtype=torch.long),
            requires_grad=False,
        )
        self._residual_tree_cached_draft_target_ids = None
        self._residual_tree_cached_draft_mapping_version = None
        self._base_draft_logits = torch.tensor([[0.0, 9.0, 1.0, 3.0, 2.0]])
        self.residual_tree_adapters = nn.ModuleList(
            [
                _FixedLogits(torch.tensor([[0.0, 10.0, 8.0, 1.0, 2.0]])),
                _FixedLogits(torch.tensor([[0.0, 11.0, 9.0, 8.0, 1.0]])),
                _FixedLogits(torch.tensor([[0.0, 12.0, 10.0, 9.0, 8.0]])),
            ]
        )


class _OrderedFourHeadLowRankResidualModel(_OrderedFourHeadResidualModel):
    def __init__(self) -> None:
        super().__init__()
        self.residual_tree_adapter_output_mode = "logit_residual"
        deltas = [
            adapter.logits - self._base_draft_logits
            for adapter in self.residual_tree_adapters
        ]
        adapters = nn.ModuleList()
        packed_in_rows = []
        packed_out_rows = []
        for delta in deltas:
            adapter = nn.Sequential(
                nn.Linear(4, 1, bias=False),
                nn.Identity(),
                nn.Linear(1, 5, bias=False),
            )
            with torch.no_grad():
                adapter[0].weight.zero_()
                adapter[0].weight[0, 0] = 1.0
                adapter[2].weight.copy_(delta.T)
            adapters.append(adapter)
            packed_in_rows.append(adapter[0].weight)
            packed_out_rows.append(adapter[2].weight)
        self.residual_tree_adapters = adapters
        self._residual_tree_packed_independent_weight = None
        self._residual_tree_packed_logit_in_weight = torch.cat(packed_in_rows, dim=0)
        self._residual_tree_packed_logit_out_weight = torch.stack(
            packed_out_rows, dim=0
        )


def _legacy_b2d1_candidates(
    model: _TwoHeadResidualModel,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_probs = model.compute_residual_head_probs(hidden_states)
    trees, _ = select_batched_b2d1_residual_trees(
        root_states=[()] * hidden_states.shape[0],
        root_head_probs=dense_probs,
        head_lambdas=[1.0, 1.0],
    )
    tokens = torch.empty((len(trees), 2), dtype=torch.long)
    probabilities = torch.empty((len(trees), 2), dtype=torch.float32)
    for request_index, tree in enumerate(trees):
        for node in tree.nodes[1:]:
            contributor = node.contributors[0]
            tokens[request_index, contributor.head_id] = node.token_id
            probabilities[request_index, contributor.head_id] = (
                contributor.proposal_prob
            )
    return tokens, probabilities


def test_greedy_b2d1_tokens_match_dense_probability_selection() -> None:
    model = _TwoHeadResidualModel()
    hidden_states = torch.zeros((2, 4))

    actual, actual_probabilities = model.compute_residual_b2d1_greedy_candidates(
        hidden_states
    )
    legacy_tokens, legacy_probabilities = _legacy_b2d1_candidates(model, hidden_states)

    dense_probs = model.compute_residual_head_probs(hidden_states)
    first_tokens = dense_probs[:, 0].argmax(dim=-1)
    conditioned_second = dense_probs[:, 1].clone()
    conditioned_second.scatter_(1, first_tokens.unsqueeze(1), 0.0)
    conditioned_second /= conditioned_second.sum(dim=-1, keepdim=True)
    expected = torch.stack((first_tokens, conditioned_second.argmax(dim=-1)), dim=1)
    expected_probabilities = torch.stack(
        (
            dense_probs[:, 0].amax(dim=-1),
            conditioned_second.amax(dim=-1),
        ),
        dim=1,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(actual, legacy_tokens)
    # Bitwise-equal compact probabilities keep lambda*q boundary ordering
    # aligned with the dense selector for ordinary rows.
    assert torch.equal(actual_probabilities, legacy_probabilities)
    assert torch.allclose(actual_probabilities, expected_probabilities, atol=1e-6)
    assert actual.tolist() == [[2, 4], [4, 2]]
    assert model._residual_tree_cached_draft_target_ids is not None


def test_ordered_independent_heads_each_select_one_new_conditioned_token() -> None:
    model = _OrderedFourHeadResidualModel()
    hidden_states = torch.zeros((1, 4))

    tokens, probabilities = model.compute_residual_greedy_candidates(hidden_states)

    assert tokens.tolist() == [[1, 2, 3, 4]]
    expected = []
    selected: list[int] = []
    logits_by_head = [
        model._base_draft_logits,
        *(adapter.logits for adapter in model.residual_tree_adapters),
    ]
    for logits in logits_by_head:
        conditioned = logits.float().clone()
        if selected:
            conditioned[:, selected] = -torch.inf
        log_probs = torch.log_softmax(conditioned, dim=1)
        token = int(log_probs.argmax(dim=1).item())
        selected.append(token)
        expected.append(float(log_probs[0, token].exp().item()))
    assert probabilities.tolist()[0] == pytest.approx(expected)
    assert len(set(tokens.tolist()[0])) == 4


def test_state_candidate_mass_calibration_scales_only_later_heads() -> None:
    model = _OrderedFourHeadResidualModel()
    model.residual_tree_state_candidate_calibration_weight = torch.zeros((3, 4))
    model.residual_tree_state_candidate_calibration_bias = torch.tensor(
        [0.0, torch.logit(torch.tensor(0.25)), torch.logit(torch.tensor(0.75))]
    )
    hidden_states = torch.zeros((1, 4))

    model.residual_tree_state_candidate_calibration_weight = None
    model.residual_tree_state_candidate_calibration_bias = None
    reference_tokens, reference_probabilities = (
        model.compute_residual_greedy_candidates(hidden_states)
    )
    model.residual_tree_state_candidate_calibration_weight = torch.zeros((3, 4))
    model.residual_tree_state_candidate_calibration_bias = torch.tensor(
        [0.0, torch.logit(torch.tensor(0.25)), torch.logit(torch.tensor(0.75))]
    )

    tokens, probabilities = model.compute_residual_greedy_candidates(hidden_states)

    assert torch.equal(tokens, reference_tokens)
    assert probabilities[:, 0].equal(reference_probabilities[:, 0])
    assert probabilities[:, 1:].tolist()[0] == pytest.approx(
        (reference_probabilities[:, 1:] * torch.tensor([[0.5, 0.25, 0.75]])).tolist()[0]
    )


def test_fused_ordered_heads_match_reference_with_packed_linear_weights() -> None:
    model = _OrderedFourHeadResidualModel()
    hidden_states = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    later_logits = [adapter.logits[0] for adapter in model.residual_tree_adapters]
    linear_adapters = nn.ModuleList([nn.Linear(4, 5, bias=False) for _ in later_logits])
    with torch.no_grad():
        for adapter, logits in zip(linear_adapters, later_logits, strict=True):
            adapter.weight.zero_()
            adapter.weight[:, 0].copy_(logits)
    model.residual_tree_adapters = linear_adapters
    model._residual_tree_fused_ordered_heads = False
    reference_tokens, reference_probabilities = (
        model.compute_residual_greedy_candidates(hidden_states)
    )

    model._residual_tree_packed_independent_weight = torch.cat(
        [adapter.weight for adapter in linear_adapters], dim=0
    )
    model._residual_tree_fused_ordered_heads = True
    fused_tokens, fused_probabilities = model.compute_residual_greedy_candidates(
        hidden_states
    )

    assert torch.equal(fused_tokens, reference_tokens)
    assert torch.equal(fused_probabilities, reference_probabilities)


def test_fused_ordered_low_rank_heads_match_reference() -> None:
    model = _OrderedFourHeadLowRankResidualModel()
    hidden_states = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    packed_in = model._residual_tree_packed_logit_in_weight
    packed_out = model._residual_tree_packed_logit_out_weight

    model._residual_tree_fused_ordered_heads = False
    reference_tokens, reference_probabilities = (
        model.compute_residual_greedy_candidates(hidden_states)
    )

    model._residual_tree_packed_logit_in_weight = packed_in
    model._residual_tree_packed_logit_out_weight = packed_out
    model._residual_tree_fused_ordered_heads = True
    fused_tokens, fused_probabilities = model.compute_residual_greedy_candidates(
        hidden_states
    )

    assert torch.equal(fused_tokens, reference_tokens)
    assert torch.equal(fused_probabilities, reference_probabilities)


def test_selectable_mass_projection_preserves_packed_head_outputs() -> None:
    model = _OrderedFourHeadLowRankResidualModel()
    hidden_states = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    raw_tokens, raw_probabilities = model.compute_residual_greedy_candidates(
        hidden_states
    )
    packed_in = model._residual_tree_packed_logit_in_weight
    assert packed_in is not None
    mass_weight = torch.zeros((3, 4), dtype=packed_in.dtype)
    mass_bias = torch.tensor([0.0, -1.0, 1.0], dtype=packed_in.dtype)
    model.residual_tree_selectable_mass_weight = mass_weight
    model.residual_tree_selectable_mass_bias = mass_bias
    tokens, probabilities, selectable_mass = model.compute_residual_greedy_candidates(
        hidden_states,
        return_selectable_mass=True,
    )

    assert torch.equal(tokens, raw_tokens)
    assert torch.equal(probabilities, raw_probabilities)
    assert selectable_mass.shape == (1, 4)
    assert selectable_mass[0].tolist() == pytest.approx(
        [
            1.0,
            0.5,
            torch.sigmoid(torch.tensor(-1.0)).item(),
            torch.sigmoid(torch.tensor(1.0)).item(),
        ]
    )


def test_stock_top2_projects_h1_once_without_adapter_or_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TwoHeadResidualModel()
    model._base_draft_logits = model._base_draft_logits.to(torch.bfloat16)
    model.residual_tree_adapters = nn.ModuleList([_UnexpectedAdapter()])
    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)

    projection_calls: list[torch.Size] = []
    original_logits_processor = model.logits_processor

    def counted_logits_processor(lm_head, hidden):
        projection_calls.append(hidden.shape)
        return original_logits_processor(lm_head, hidden)

    topk_input_dtypes: list[torch.dtype] = []
    original_topk = torch.topk

    def checked_topk(input_tensor, *args, **kwargs):
        topk_input_dtypes.append(input_tensor.dtype)
        return original_topk(input_tensor, *args, **kwargs)

    model.logits_processor = counted_logits_processor
    monkeypatch.setattr(torch, "topk", checked_topk)

    actual = model.compute_stock_top2_greedy_tokens(hidden_states)

    assert actual.tolist() == [[2, 4], [4, 2]]
    assert projection_calls == [torch.Size((2, 4))]
    assert topk_input_dtypes == [torch.bfloat16]
    assert model._residual_tree_cached_draft_target_ids is not None


def test_stock_top2_rejects_noninjective_target_mapping() -> None:
    model = _TwoHeadResidualModel()
    with torch.no_grad():
        model.draft_id_to_target_id.copy_(torch.tensor([0, -1, -2]))

    with pytest.raises(NotImplementedError, match="injective token mapping"):
        model.compute_stock_top2_greedy_tokens(torch.zeros((1, 4)))


def test_pruned_stock_top2_returns_true_probabilities() -> None:
    model = _TwoHeadResidualModel()
    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)

    tokens, probabilities = model.compute_stock_top2_greedy_candidates(hidden_states)

    expected = torch.softmax(model._base_draft_logits.float(), dim=-1)
    expected_top2 = torch.topk(expected, k=2, dim=-1).values
    assert tokens.tolist() == [[2, 4], [4, 2]]
    assert torch.allclose(probabilities, expected_top2, atol=1e-6)


def test_hybrid_dynamic_returns_h1_top9_plus_distinct_h2() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    model.residual_tree_adapters = nn.ModuleList(
        [_FixedLogits(torch.tensor([[20.0] + [0.0] * 9 + [15.0, 1.0]]))]
    )

    tokens, probabilities = model.compute_hybrid_top9_h2_dynamic_candidates(
        torch.zeros((1, 4))
    )

    assert tokens.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
    assert len(set(tokens[0].tolist())) == 10
    expected_h1 = torch.softmax(model._base_draft_logits.float(), dim=-1)[0, :9]
    assert torch.allclose(probabilities[0, :9], expected_h1, atol=1e-6)
    assert probabilities[0, 9] > 0.99


def test_hybrid_top5_top5_returns_five_distinct_tokens_per_head() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    model.residual_tree_adapters = nn.ModuleList(
        [
            _FixedLogits(
                torch.tensor(
                    [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0]]
                )
            )
        ]
    )

    tokens, probabilities = model.compute_hybrid_top5_h2_top5_dynamic_candidates(
        torch.zeros((1, 4))
    )

    assert tokens.tolist() == [list(range(10))]
    assert len(set(tokens[0].tolist())) == 10
    expected_h1 = torch.softmax(model._base_draft_logits.float(), dim=-1)[0, :5]
    assert torch.allclose(probabilities[0, :5], expected_h1, atol=1e-6)
    assert torch.all(probabilities[0, 5:-1] > probabilities[0, 6:])


def test_hybrid_union_merges_duplicate_tokens_before_top10() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    h2_logits = torch.full((1, 12), -20.0)
    h2_logits[0, 0] = 20.0
    h2_logits[0, 10] = 19.0
    h2_logits[0, 11] = 18.0
    model.residual_tree_adapters = nn.ModuleList([_FixedLogits(h2_logits)])

    (
        tokens,
        scores,
        source_masks,
        h1_ranks,
        h2_ranks,
        score_heads,
    ) = model.compute_hybrid_union_top10_dynamic_candidates(
        torch.zeros((1, 4)),
        h2_to_h1_weight=0.5,
        return_union_provenance=True,
    )

    assert len(set(tokens[0].tolist())) == 10
    assert set(tokens[0].tolist()) == {*range(8), 10, 11}
    assert tokens[0].tolist().count(0) == 1
    expected_h1_token0 = torch.softmax(model._base_draft_logits.float(), dim=-1)[0, 0]
    token0_index = tokens[0].tolist().index(0)
    assert scores[0, token0_index].item() == pytest.approx(expected_h1_token0.item())
    assert source_masks[0, token0_index].item() == 3
    assert h1_ranks[0, token0_index].item() == 1
    assert h2_ranks[0, token0_index].item() == 1
    assert score_heads[0, token0_index].item() == 0
    for token_id, h2_rank in ((10, 2), (11, 3)):
        token_index = tokens[0].tolist().index(token_id)
        assert source_masks[0, token_index].item() == 2
        assert h1_ranks[0, token_index].item() == 0
        assert h2_ranks[0, token_index].item() == h2_rank
        assert score_heads[0, token_index].item() == 1
    assert torch.all(scores[0, :-1] >= scores[0, 1:])

    h1_tokens, _ = model.compute_hybrid_union_top10_dynamic_candidates(
        torch.zeros((1, 4)),
        h2_to_h1_weight=1e-6,
    )
    assert h1_tokens.tolist() == [list(range(10))]


def test_hybrid_union_uses_per_depth_clipped_mass_calibration() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    h2_logits = torch.full((1, 12), -20.0)
    h2_logits[0, 0] = 20.0
    h2_logits[0, 10] = 19.0
    h2_logits[0, 11] = 18.0
    model.residual_tree_adapters = nn.ModuleList([_FixedLogits(h2_logits)])
    calibration = torch.full((8, 4, 10), 0.1, dtype=torch.float32)
    calibration[:, 2, :] = 0.8
    calibration[:, 3, :] = 0.05
    model.residual_tree_hybrid_union_clipped_mass_calibration = calibration

    tokens, scores, source_masks, *_ = (
        model.compute_hybrid_union_top10_dynamic_candidates(
            torch.zeros((1, 4)),
            h2_to_h1_weight=1.0,
            tree_depth=3,
            return_union_provenance=True,
        )
    )

    assert len(set(tokens[0].tolist())) == 10
    assert 10 in tokens[0].tolist() and 11 in tokens[0].tolist()
    token0_index = tokens[0].tolist().index(0)
    h1_probability = torch.softmax(model._base_draft_logits.float(), dim=-1)[0, 0]
    h2_probability = torch.softmax(h2_logits.float(), dim=-1)[0, 0]
    expected_shared_score = 0.1 * h1_probability + 0.05 * h2_probability
    assert scores[0, token0_index].item() == pytest.approx(expected_shared_score.item())
    assert source_masks[0, token0_index].item() == 3

    with pytest.raises(ValueError, match="requires tree_depth"):
        model.compute_hybrid_union_top10_dynamic_candidates(
            torch.zeros((1, 4)),
            h2_to_h1_weight=1.0,
        )
    with pytest.raises(ValueError, match="equal H1/H2 head lambdas"):
        model.compute_hybrid_union_top10_dynamic_candidates(
            torch.zeros((1, 4)),
            h2_to_h1_weight=0.5,
            tree_depth=3,
        )


def test_hybrid_union_tree_oracle_only_rescues_raw_head_candidates() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    h2_logits = torch.full((1, 12), -20.0)
    h2_logits[0, 0] = 20.0
    h2_logits[0, 10] = 19.0
    h2_logits[0, 11] = 18.0
    model.residual_tree_adapters = nn.ModuleList([_FixedLogits(h2_logits)])
    hidden = torch.zeros((1, 4))

    ordinary_tokens, ordinary_scores = (
        model.compute_hybrid_union_top10_dynamic_candidates(
            hidden,
            h2_to_h1_weight=0.5,
        )
    )
    rescued_tokens, rescued_scores = (
        model.compute_hybrid_union_top10_dynamic_candidates(
            hidden,
            h2_to_h1_weight=0.5,
            oracle_target_tokens=torch.tensor([9]),
        )
    )
    unavailable_tokens, unavailable_scores = (
        model.compute_hybrid_union_top10_dynamic_candidates(
            hidden,
            h2_to_h1_weight=0.5,
            oracle_target_tokens=torch.tensor([99]),
        )
    )

    assert 9 not in ordinary_tokens[0].tolist()
    assert rescued_tokens[0, -1].item() == 9
    assert rescued_scores[0, -1].item() == 1.0
    assert torch.equal(unavailable_tokens, ordinary_tokens)
    assert torch.equal(unavailable_scores, ordinary_scores)


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires CUDA and Triton",
)
def test_fused_hybrid_union_provenance_describes_fused_candidates() -> None:
    model = _TwoHeadResidualModel().to("cuda")
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long, device="cuda"),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._residual_tree_fused_hybrid_union = True
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    h2_logits = torch.full((1, 12), -20.0, device="cuda", dtype=torch.bfloat16)
    h2_logits[0, 0] = 20.0
    h2_logits[0, 10] = 19.0
    h2_logits[0, 11] = 18.0
    model.residual_tree_adapters = nn.ModuleList([_FixedLogits(h2_logits)])

    tokens, scores, source_masks, h1_ranks, h2_ranks, score_heads = (
        model.compute_hybrid_union_top10_dynamic_candidates(
            torch.zeros((1, 4), device="cuda", dtype=torch.bfloat16),
            h2_to_h1_weight=0.5,
            return_union_provenance=True,
        )
    )

    assert len(set(tokens[0].tolist())) == 10
    assert set(tokens[0].tolist()) == {*range(8), 10, 11}
    assert torch.all(scores[0, :-1] >= scores[0, 1:])
    for index, token in enumerate(tokens[0].tolist()):
        mask = int(source_masks[0, index])
        assert bool(mask & 1) == bool(h1_ranks[0, index] > 0)
        assert bool(mask & 2) == bool(h2_ranks[0, index] > 0)
        assert mask & (1 << int(score_heads[0, index]))
        if token in (10, 11):
            assert mask == 2


@pytest.mark.parametrize(
    ("method_name", "h1_count"),
    [
        ("compute_hybrid_top9_h2_dynamic_candidates", 9),
        ("compute_hybrid_top8_h2_top2_dynamic_candidates", 8),
        ("compute_hybrid_top7_h2_top3_dynamic_candidates", 7),
        ("compute_hybrid_top6_h2_top4_dynamic_candidates", 6),
        ("compute_hybrid_top5_h2_top5_dynamic_candidates", 5),
    ],
)
def test_hybrid_allocations_report_h2_overlap_with_unselected_h1_top10(
    method_name: str,
    h1_count: int,
) -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    descending_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    model._base_draft_logits = descending_logits
    model.residual_tree_adapters = nn.ModuleList(
        [_FixedLogits(descending_logits.clone())]
    )

    tokens, probabilities, h1_rank_overlap = getattr(model, method_name)(
        torch.zeros((1, 4)),
        return_h1_rank_overlap=True,
    )

    assert tokens.tolist() == [list(range(10))]
    assert probabilities.shape == (1, 10)
    assert h1_rank_overlap.tolist() == [list(range(h1_count + 1, 11))]


def test_hybrid_dynamic_applies_one_bias_in_existing_linear_projection() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    model.residual_tree_adapters = nn.ModuleList([nn.Linear(4, 12, bias=False)])
    with torch.no_grad():
        model.residual_tree_adapters[0].weight.zero_()
    depth_bias = torch.zeros((8, 12))
    depth_bias[0, 10] = 20.0
    depth_bias[1, 11] = 20.0
    model.register_buffer(
        "residual_tree_h2_tree_depth_bias",
        depth_bias,
        persistent=False,
    )

    depth0_tokens, _ = model.compute_hybrid_top9_h2_dynamic_candidates(
        torch.zeros((1, 4)), tree_depth=0
    )
    depth1_tokens, _ = model.compute_hybrid_top9_h2_dynamic_candidates(
        torch.zeros((1, 4)), tree_depth=1
    )

    assert depth0_tokens[0, -1].item() == 10
    assert depth1_tokens[0, -1].item() == 11
    with pytest.raises(ValueError, match="uniform tree_depth"):
        model.compute_hybrid_top9_h2_dynamic_candidates(torch.zeros((1, 4)))


def test_hybrid_dynamic_uniform_fallback_excludes_all_h1_tokens() -> None:
    model = _TwoHeadResidualModel()
    model.config = SimpleNamespace(vocab_size=12)
    model.residual_tree_draft_vocab_size = 12
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(12, dtype=torch.long),
        requires_grad=False,
    )
    model._residual_tree_cached_draft_target_ids = None
    model._residual_tree_cached_draft_mapping_version = None
    model._base_draft_logits = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
    )
    model.residual_tree_adapters = nn.ModuleList(
        [_FixedLogits(torch.tensor([[100.0] + [0.0] * 11]))]
    )

    tokens, probabilities = model.compute_hybrid_top9_h2_dynamic_candidates(
        torch.zeros((1, 4))
    )

    assert tokens[0, :9].tolist() == list(range(9))
    assert tokens[0, 9].item() == 9
    assert probabilities[0, 9].item() == pytest.approx(1.0 / 3.0)


def test_stock_top2_initialization_ignores_embedded_residual_adapter() -> None:
    class _StockOnlyModel(ResidualTreeHeadMixin, nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                residual_tree_adapter_config="/must/not/be/read.json",
            )

    model = _StockOnlyModel()
    model._init_residual_tree_heads(
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                residual_tree_candidate_selection="stock_top2",
            )
        )
    )

    assert len(model.residual_tree_adapters) == 0
    assert model.residual_tree_total_heads == 0
    assert model.residual_tree_head_lambdas is None


def test_logit_residual_compact_candidates_match_dense_selection() -> None:
    model = _TwoHeadResidualModel()
    model.residual_tree_adapter_output_mode = "logit_residual"
    model.residual_tree_adapters = nn.ModuleList(
        [
            _FixedLogits(
                torch.tensor(
                    [
                        [0.0, -3.0, 8.0],
                        [7.0, 0.0, -2.0],
                    ]
                )
            )
        ]
    )
    hidden_states = torch.zeros((2, 4))

    actual_tokens, actual_probabilities = model.compute_residual_greedy_candidates(
        hidden_states
    )
    expected_tokens, expected_probabilities = _legacy_b2d1_candidates(
        model, hidden_states
    )

    assert model.supports_residual_greedy_candidates()
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(actual_probabilities, expected_probabilities)


def test_greedy_b2d1_uses_dense_uniform_fallback_after_softmax() -> None:
    model = _TwoHeadResidualModel()
    model._base_draft_logits = torch.tensor([[100.0, 0.0, 1.0]])
    with torch.no_grad():
        model.residual_tree_adapters[0].logits[0].copy_(torch.tensor([100.0, 0.0, 1.0]))
    hidden_states = torch.zeros((1, 4))

    actual_tokens, actual_probabilities = model.compute_residual_b2d1_greedy_candidates(
        hidden_states
    )
    expected_tokens, expected_probabilities = _legacy_b2d1_candidates(
        model, hidden_states
    )

    # H2's residual mass outside H1's target token is below 1e-12.  The
    # legacy selector therefore falls back across all remaining target ids,
    # whose first member (target id 1) is not in the draft vocabulary.
    assert actual_tokens.tolist() == expected_tokens.tolist() == [[0, 1]]
    assert torch.equal(actual_probabilities, expected_probabilities)
    assert actual_probabilities.tolist() == [[1.0, 0.25]]


def test_greedy_b2d1_tree_has_fixed_sibling_topology_without_proposals() -> None:
    trees = _build_greedy_b2d1_draft_trees(
        [[7, 9], [11, 13]],
        [[0.8, 0.2], [0.7, 0.3]],
        head_lambdas=[1.0, 1.0],
        tree_policy="best_first",
    )

    assert [tree.draft_token_ids for tree in trees] == [[7, 9], [11, 13]]
    for tree in trees:
        assert tree.parent_node_ids == [-1, 0, 0]
        assert tree.node_depths == [0, 1, 1]
        assert tree.child_node_ids == [1, 2]
        assert tree.num_proposal_rows == 0
        assert tree.contributor_proposal_rows == []


def test_greedy_b2d1_tree_rejects_duplicate_siblings() -> None:
    with pytest.raises(ValueError, match="duplicate token"):
        _build_greedy_b2d1_draft_trees(
            [[5, 5]],
            [[0.6, 0.4]],
            head_lambdas=[1.0, 1.0],
            tree_policy="best_first",
        )


@pytest.mark.parametrize("tree_policy", ["best_first", "breadth_first"])
def test_greedy_b2d1_tree_preserves_legacy_order_when_h2_wins(
    tree_policy: str,
) -> None:
    root_probs = torch.tensor(
        [[[0.1, 0.7, 0.1, 0.1], [0.1, 0.2, 0.1, 0.6]]],
        dtype=torch.float32,
    )
    lambdas = torch.tensor([0.1, 1.0])
    legacy, _ = select_batched_b2d1_residual_trees(
        root_states=[()],
        root_head_probs=root_probs,
        head_lambdas=lambdas,
        tree_policy=tree_policy,
    )
    conditioned_h2_probability = 0.6 / (1.0 - 0.2)

    optimized = _build_greedy_b2d1_draft_trees(
        [[1, 3]],
        [[0.7, conditioned_h2_probability]],
        head_lambdas=lambdas.tolist(),
        tree_policy=tree_policy,
    )[0]
    legacy_draft = tree_to_draft_token_tree(legacy[0])

    assert optimized.draft_token_ids == legacy_draft.draft_token_ids == [3, 1]
    assert optimized.truncate(1).draft_token_ids == [3]
    assert legacy_draft.truncate(1).draft_token_ids == [3]


def test_greedy_b2d1_hook_rejects_unsupported_head_layout() -> None:
    model = _TwoHeadResidualModel()
    model.residual_tree_adapter_output_mode = "hidden_residual"

    assert not model.supports_residual_b2d1_greedy_candidates()
    with pytest.raises(NotImplementedError, match="frozen base head"):
        model.compute_residual_b2d1_greedy_candidates(torch.zeros((1, 4)))


def test_greedy_b2d1_hook_rejects_noninjective_mapping() -> None:
    model = _TwoHeadResidualModel()
    with torch.no_grad():
        model.draft_id_to_target_id.copy_(torch.tensor([0, -1, -2]))

    with pytest.raises(NotImplementedError, match="injective"):
        model.compute_residual_b2d1_greedy_candidates(torch.zeros((1, 4)))


def test_greedy_b2d1_mapping_cache_tracks_weight_reload() -> None:
    model = _TwoHeadResidualModel()
    hidden_states = torch.zeros((1, 4))
    before, _ = model.compute_residual_b2d1_greedy_candidates(hidden_states)
    cached_version = model._residual_tree_cached_draft_mapping_version

    with torch.no_grad():
        # New injective targets are [1, 2, 3].
        model.draft_id_to_target_id.copy_(torch.tensor([1, 1, 1]))
    after, _ = model.compute_residual_b2d1_greedy_candidates(hidden_states)

    assert before.tolist() == [[2, 4]]
    assert after.tolist() == [[2, 3]]
    assert model._residual_tree_cached_draft_mapping_version != cached_version
