# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.kernels.residual_tree_union import (
    fused_hybrid_union_top10,
    supports_fused_hybrid_union,
)
from vllm.model_executor.kernels.residual_tree_ordered import (
    fused_ordered_distinct_top1,
    fused_ordered_head_top1,
    supports_fused_ordered_selection,
)

_RESIDUAL_TREE_GREEDY_EPS = 1e-12


class ResidualTreeAdapter(nn.Sequential):
    """Bottleneck adapter used by residual-head EAGLE tree drafting."""

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            nn.Linear(hidden_size, bottleneck, bias=False, dtype=dtype),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_size, bias=False, dtype=dtype),
        )


class ResidualTreeLogitAdapter(nn.Sequential):
    """Low-rank output-space delta used by compact residual H2 heads."""

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck: int,
        draft_vocab_size: int,
        dtype: torch.dtype,
        activation: str = "silu",
        output_projection: nn.Linear | None = None,
    ) -> None:
        if activation == "silu":
            activation_layer: nn.Module = nn.SiLU()
        elif activation == "identity":
            activation_layer = nn.Identity()
        else:
            raise ValueError(
                "residual-tree logit adapter activation must be silu or identity"
            )
        if output_projection is None:
            output_projection = nn.Linear(
                bottleneck,
                draft_vocab_size,
                bias=False,
                dtype=dtype,
            )
        super().__init__(
            nn.Linear(hidden_size, bottleneck, bias=False, dtype=dtype),
            activation_layer,
            output_projection,
        )


class ResidualTreeHeadMixin:
    """Shared residual-head API for EAGLE-3 draft models.

    The training code in this repository stores residual heads as a ModuleList
    state dict. Hidden-residual adapters reuse the draft model's normal lm_head,
    while independent LM heads own their vocabulary projection.
    """

    residual_tree_adapters: nn.ModuleList
    residual_tree_freeze_base_head: bool
    residual_tree_total_heads: int
    residual_tree_head_lambdas: list[float] | None
    residual_tree_adapter_output_mode: str
    residual_tree_draft_vocab_size: int
    residual_tree_required_candidate_selection: str | None
    residual_tree_h2_tree_depth_bias: torch.Tensor
    residual_tree_hybrid_union_clipped_mass_calibration: torch.Tensor | None
    residual_tree_state_candidate_calibration_weight: torch.Tensor | None
    residual_tree_state_candidate_calibration_bias: torch.Tensor | None
    _residual_tree_packed_independent_weight: torch.Tensor | None
    _residual_tree_packed_logit_in_weight: torch.Tensor | None
    _residual_tree_packed_logit_out_weight: torch.Tensor | None
    _residual_tree_shared_logit_out_weight: torch.Tensor | None
    _residual_tree_cached_draft_target_ids: torch.Tensor | None
    _residual_tree_cached_draft_mapping_version: int | None
    _residual_tree_fused_hybrid_union: bool
    _residual_tree_fused_ordered_heads: bool
    _residual_tree_fused_ordered_selection: bool

    def _init_residual_tree_heads(self, vllm_config: VllmConfig) -> None:
        spec_config = vllm_config.speculative_config
        self.residual_tree_adapters = nn.ModuleList()
        self.residual_tree_freeze_base_head = False
        self.residual_tree_total_heads = 0
        self.residual_tree_head_lambdas = None
        self.residual_tree_adapter_output_mode = "hidden_residual"
        self.residual_tree_draft_vocab_size = 0
        self.residual_tree_required_candidate_selection = None
        self.register_buffer(
            "residual_tree_hybrid_union_clipped_mass_calibration",
            None,
            persistent=False,
        )
        self.register_buffer(
            "residual_tree_state_candidate_calibration_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "residual_tree_state_candidate_calibration_bias",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_residual_tree_packed_independent_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_residual_tree_packed_logit_in_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_residual_tree_packed_logit_out_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_residual_tree_shared_logit_out_weight",
            None,
            persistent=False,
        )
        self._residual_tree_cached_draft_target_ids = None
        self._residual_tree_cached_draft_mapping_version = None
        self._residual_tree_fused_hybrid_union = os.environ.get(
            "VLLM_RESIDUAL_TREE_FUSED_HYBRID_UNION", "1"
        ).lower() not in {"0", "false", "no"}
        self._residual_tree_fused_ordered_heads = os.environ.get(
            "VLLM_RESIDUAL_TREE_FUSED_ORDERED_HEADS", "1"
        ).lower() not in {"0", "false", "no"}
        self._residual_tree_fused_ordered_selection = os.environ.get(
            "VLLM_RESIDUAL_TREE_FUSED_ORDERED_SELECTION", "1"
        ).lower() not in {"0", "false", "no"}
        if spec_config is None:
            return
        if getattr(spec_config, "residual_tree_candidate_selection", None) in {
            "stock_top2",
            "stock_top9_dynamic",
            "stock_top10_dynamic",
        }:
            # The stock-tree control must remain one unmodified EAGLE H1 even
            # when the draft model's HF config contains a residual adapter path.
            return

        config_path = getattr(spec_config, "residual_tree_adapter_config", None)
        hf_config_path = getattr(self.config, "residual_tree_adapter_config", None)
        config_path = config_path or hf_config_path
        if config_path is None:
            return

        config_path_obj = Path(str(config_path))
        config = _load_residual_tree_adapter_config(config_path_obj)
        method = config.get("method")
        expected = "vllm_eagle3_shared_trunk_residual_heads"
        if method != expected:
            raise ValueError(
                "residual_tree_adapter_config has unsupported method "
                f"{method!r}; expected {expected!r}"
            )
        serving_proposal_mode = config.get("serving_proposal_mode")
        if serving_proposal_mode not in {
            None,
            "raw_head_distribution",
            "condition_on_prior_selected_tokens",
        }:
            raise ValueError(
                "residual tree adapter config has unsupported "
                f"serving_proposal_mode {serving_proposal_mode!r}"
            )
        if serving_proposal_mode == "condition_on_prior_selected_tokens":
            configured_selection = getattr(
                spec_config,
                "residual_tree_candidate_selection",
                None,
            )
            self.residual_tree_required_candidate_selection = (
                configured_selection
                if configured_selection
                in {
                    "hybrid_top9_h2_dynamic",
                    "hybrid_top8_h2_top2_dynamic",
                    "hybrid_top7_h2_top3_dynamic",
                    "hybrid_top6_h2_top4_dynamic",
                    "hybrid_top5_h2_top5_dynamic",
                    "hybrid_union_top10_dynamic",
                }
                else "distinct_head_top1"
            )
        elif serving_proposal_mode == "raw_head_distribution":
            self.residual_tree_required_candidate_selection = "head_top1"

        hidden_size = int(config.get("hidden_size", self.config.hidden_size))
        if hidden_size != int(self.config.hidden_size):
            raise ValueError(
                "residual tree adapter hidden_size does not match draft model"
            )
        bottleneck = int(config["adapter_bottleneck"])
        output_mode = str(config.get("adapter_output_mode", "hidden_residual"))
        if output_mode not in {
            "hidden_residual",
            "logit_residual",
            "independent_lm_head",
        }:
            raise ValueError(
                "residual tree serving does not support adapter_output_mode "
                f"{output_mode!r}"
            )
        conditioning_mode = str(config.get("adapter_conditioning_mode", "none"))
        if conditioning_mode != "none":
            raise ValueError(
                "residual tree serving does not support adapter_conditioning_mode "
                f"{conditioning_mode!r}"
            )
        adapter_depth = int(config.get("adapter_depth", 2))
        adapter_activation = str(config.get("adapter_activation", "silu"))
        if adapter_activation not in {"silu", "identity"}:
            raise ValueError(
                "residual tree adapter_activation must be silu or identity"
            )
        if output_mode != "logit_residual" and adapter_activation != "silu":
            raise ValueError(
                "identity adapter activation is supported only for logit_residual"
            )
        if output_mode in {"hidden_residual", "logit_residual"} and adapter_depth != 2:
            raise ValueError(
                f"{output_mode} tree serving currently requires adapter_depth=2"
            )
        draft_vocab_size = int(
            config.get(
                "draft_vocab_size",
                getattr(self.config, "draft_vocab_size", 0),
            )
        )
        if output_mode in {"independent_lm_head", "logit_residual"} and (
            draft_vocab_size <= 0
        ):
            raise ValueError(f"{output_mode} heads require draft_vocab_size")
        vocab_projection = str(
            config.get("adapter_vocab_projection", "per_head")
        )
        if vocab_projection not in {"per_head", "shared"}:
            raise ValueError(
                "residual tree adapter_vocab_projection must be per_head or shared"
            )
        if vocab_projection == "shared" and (
            output_mode != "logit_residual" or adapter_activation != "identity"
        ):
            raise ValueError(
                "shared vocabulary projection requires identity logit_residual heads"
            )
        adapter_count = int(
            config.get("num_residual_adapters", config.get("num_layers", 0))
        )
        if adapter_count <= 0:
            raise ValueError("residual tree adapter config has no adapters")

        self.residual_tree_freeze_base_head = bool(
            config.get("freeze_base_head", False)
        )
        self.residual_tree_total_heads = int(config.get("num_layers", adapter_count))
        if self.residual_tree_freeze_base_head:
            expected_adapters = self.residual_tree_total_heads - 1
        else:
            expected_adapters = self.residual_tree_total_heads
        if adapter_count != expected_adapters:
            raise ValueError(
                "residual tree adapter count does not match num_layers/freeze_base_head"
            )

        dtype = _residual_tree_adapter_dtype(vllm_config)
        if output_mode == "independent_lm_head":
            adapters = nn.ModuleList(
                nn.Linear(
                    hidden_size,
                    draft_vocab_size,
                    bias=False,
                    dtype=dtype,
                )
                for _ in range(adapter_count)
            )
        elif output_mode == "logit_residual":
            shared_output = (
                nn.Linear(
                    bottleneck,
                    draft_vocab_size,
                    bias=False,
                    dtype=dtype,
                )
                if vocab_projection == "shared"
                else None
            )
            adapters = nn.ModuleList(
                ResidualTreeLogitAdapter(
                    hidden_size=hidden_size,
                    bottleneck=bottleneck,
                    draft_vocab_size=draft_vocab_size,
                    dtype=dtype,
                    activation=adapter_activation,
                    output_projection=shared_output,
                )
                for _ in range(adapter_count)
            )
        else:
            adapters = nn.ModuleList(
                ResidualTreeAdapter(
                    hidden_size=hidden_size,
                    bottleneck=bottleneck,
                    dtype=dtype,
                )
                for _ in range(adapter_count)
            )
        checkpoint = getattr(spec_config, "residual_tree_adapter_checkpoint", None)
        checkpoint = checkpoint or config.get("adapter_checkpoint")
        if checkpoint is None:
            raise ValueError("residual tree adapter checkpoint is missing")
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.exists() and not checkpoint_path.is_absolute():
            checkpoint_path = config_path_obj.parent / checkpoint_path
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if vocab_projection == "shared":
            if not isinstance(state, dict):
                raise TypeError("residual tree adapter checkpoint must be a dict")
            shared_out = state.pop("shared_out.weight", None)
            if not isinstance(shared_out, torch.Tensor):
                raise ValueError(
                    "shared vocabulary checkpoint lacks shared_out.weight"
                )
            expected_shape = (draft_vocab_size, bottleneck)
            if tuple(shared_out.shape) != expected_shape:
                raise ValueError(
                    "shared_out.weight must have shape "
                    f"{expected_shape}, got {tuple(shared_out.shape)}"
                )
            for adapter_index in range(adapter_count):
                state[f"{adapter_index}.2.weight"] = shared_out
        depth_bias_mode = config.get("h2_tree_depth_bias_mode")
        depth_bias = None
        if depth_bias_mode is not None:
            if depth_bias_mode != "per_dynamic_tree_depth":
                raise ValueError(
                    "residual tree adapter config has unsupported "
                    f"h2_tree_depth_bias_mode {depth_bias_mode!r}"
                )
            configured_selection = getattr(
                spec_config,
                "residual_tree_candidate_selection",
                None,
            )
            if (
                output_mode != "independent_lm_head"
                or not self.residual_tree_freeze_base_head
                or self.residual_tree_total_heads != 2
                or configured_selection
                not in {
                    "hybrid_top9_h2_dynamic",
                    "hybrid_top8_h2_top2_dynamic",
                    "hybrid_top7_h2_top3_dynamic",
                    "hybrid_top6_h2_top4_dynamic",
                    "hybrid_top5_h2_top5_dynamic",
                    "hybrid_union_top10_dynamic",
                }
            ):
                raise ValueError(
                    "per-depth H2 bias requires the frozen-H1, independent-H2 "
                    "hybrid dynamic path"
                )
            if not isinstance(state, dict):
                raise TypeError("residual tree adapter checkpoint must be a dict")
            depth_bias = state.pop("tree_depth_bias", None)
            if not isinstance(depth_bias, torch.Tensor):
                raise ValueError(
                    "per-depth H2 bias checkpoint lacks tensor tree_depth_bias"
                )
            expected_shape = (8, draft_vocab_size)
            if tuple(depth_bias.shape) != expected_shape:
                raise ValueError(
                    "tree_depth_bias must have shape "
                    f"{expected_shape}, got {tuple(depth_bias.shape)}"
                )
            if depth_bias.dtype != dtype:
                raise ValueError(
                    "tree_depth_bias dtype must match the draft execution dtype "
                    f"{dtype}, got {depth_bias.dtype}"
                )
        adapters.load_state_dict(state)
        if (
            output_mode == "independent_lm_head"
            and self.residual_tree_freeze_base_head
            and self.residual_tree_total_heads > 2
            and self._residual_tree_fused_ordered_heads
        ):
            # Ordered H1-H10 serving reads every later vocabulary head for the
            # same hidden rows.  Keep their weights in one contiguous backing
            # allocation so one large linear replaces nine sequential GEMMs.
            # The per-head Parameter views preserve the existing checkpoint
            # and diagnostic APIs without retaining a second weight copy.
            packed_weight = torch.cat(
                [adapter.weight.detach() for adapter in adapters], dim=0
            ).contiguous()
            rows_per_head = draft_vocab_size
            for adapter_index, adapter in enumerate(adapters):
                begin = adapter_index * rows_per_head
                adapter.weight = nn.Parameter(
                    packed_weight.narrow(0, begin, rows_per_head),
                    requires_grad=adapter.weight.requires_grad,
                )
            self._residual_tree_packed_independent_weight = packed_weight
        elif (
            output_mode == "logit_residual"
            and adapter_activation == "identity"
            and self.residual_tree_freeze_base_head
            and self.residual_tree_total_heads > 2
            and self._residual_tree_fused_ordered_heads
        ):
            # All low-rank residual heads consume the same hidden rows. Pack
            # their input factors into one GEMM and their output factors into
            # one strided-batched GEMM. Parameter views retain the existing
            # checkpoint contract without keeping duplicate weights alive.
            packed_in = torch.cat(
                [adapter[0].weight.detach() for adapter in adapters], dim=0
            ).contiguous()
            rank = bottleneck
            for adapter_index, adapter in enumerate(adapters):
                adapter[0].weight = nn.Parameter(
                    packed_in.narrow(0, adapter_index * rank, rank),
                    requires_grad=adapter[0].weight.requires_grad,
                )
            self._residual_tree_packed_logit_in_weight = packed_in
            if vocab_projection == "shared":
                # Every adapter already references this one Parameter.  The
                # non-persistent tensor view avoids another vocabulary-sized
                # allocation while making the fused serving path explicit.
                self._residual_tree_shared_logit_out_weight = (
                    adapters[0][2].weight.detach()
                )
            else:
                packed_out = torch.stack(
                    [adapter[2].weight.detach() for adapter in adapters], dim=0
                ).contiguous()
                for adapter_index, adapter in enumerate(adapters):
                    adapter[2].weight = nn.Parameter(
                        packed_out[adapter_index],
                        requires_grad=adapter[2].weight.requires_grad,
                    )
                self._residual_tree_packed_logit_out_weight = packed_out
        self.residual_tree_adapters = adapters
        self.residual_tree_adapter_output_mode = output_mode
        self.residual_tree_draft_vocab_size = draft_vocab_size
        if depth_bias is not None:
            adapter_device = next(adapters.parameters()).device
            self.register_buffer(
                "residual_tree_h2_tree_depth_bias",
                depth_bias.to(device=adapter_device).contiguous(),
                persistent=False,
            )

        union_calibration = config.get("hybrid_union_clipped_mass_calibration")
        if union_calibration is not None:
            configured_selection = getattr(
                spec_config,
                "residual_tree_candidate_selection",
                None,
            )
            if configured_selection != "hybrid_union_top10_dynamic":
                raise ValueError(
                    "hybrid-union clipped-mass calibration requires "
                    "hybrid_union_top10_dynamic candidates"
                )
            if not isinstance(union_calibration, dict) or union_calibration.get(
                "schema"
            ) != "qwen3_hybrid_union_clipped_mass_calibration_v1":
                raise ValueError("unsupported hybrid-union clipped-mass calibration")
            expected_metadata = {
                "category_order": [
                    "h1_only",
                    "h1_shared",
                    "h2_only",
                    "h2_shared",
                ],
                "rank_order": list(range(1, 11)),
                "tree_depth_order": list(range(8)),
            }
            for key, expected_value in expected_metadata.items():
                if union_calibration.get(key) != expected_value:
                    raise ValueError(
                        f"hybrid-union clipped-mass calibration has invalid {key}"
                    )
            factors = union_calibration.get("factors")
            if (
                not isinstance(factors, list)
                or len(factors) != 8
                or any(
                    not isinstance(depth_rows, list) or len(depth_rows) != 4
                    for depth_rows in factors
                )
                or any(
                    not isinstance(rank_rows, list) or len(rank_rows) != 10
                    for depth_rows in factors
                    for rank_rows in depth_rows
                )
            ):
                raise ValueError(
                    "hybrid-union clipped-mass factors must have shape [8, 4, 10]"
                )
            flat_factors = [
                float(value)
                for depth_rows in factors
                for rank_rows in depth_rows
                for value in rank_rows
            ]
            if any(
                not math.isfinite(value) or not 0.0 < value <= 1.0
                for value in flat_factors
            ):
                raise ValueError(
                    "hybrid-union clipped-mass factors must be finite and in (0, 1]"
                )
            adapter_device = next(adapters.parameters()).device
            self.residual_tree_hybrid_union_clipped_mass_calibration = torch.tensor(
                factors,
                dtype=torch.float32,
                device=adapter_device,
            ).contiguous()

        state_calibration = config.get("state_candidate_mass_calibration")
        if state_calibration is not None:
            if not isinstance(state_calibration, dict) or state_calibration.get(
                "schema"
            ) != "qwen3_32b_state_candidate_mass_calibrator_v1":
                raise ValueError("unsupported state candidate-mass calibration")
            if not (
                self.residual_tree_freeze_base_head
                and self.residual_tree_total_heads == 10
                and output_mode == "logit_residual"
                and state_calibration.get("calibrated_heads")
                == list(range(2, 11))
                and float(state_calibration.get("h1_factor", -1.0)) == 1.0
            ):
                raise ValueError(
                    "state candidate-mass calibration requires frozen H1 plus "
                    "ordered logit-residual H2-H10"
                )
            calibration_path = Path(str(state_calibration.get("checkpoint", "")))
            if not calibration_path.is_absolute():
                calibration_path = config_path_obj.parent / calibration_path
            calibration = torch.load(
                calibration_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(calibration, dict) or calibration.get(
                "schema"
            ) != state_calibration["schema"]:
                raise ValueError("state candidate-mass checkpoint has wrong schema")
            gate_weight = calibration.get("weight")
            gate_bias = calibration.get("bias")
            expected_weight_shape = (self.residual_tree_total_heads - 1, hidden_size)
            expected_bias_shape = (self.residual_tree_total_heads - 1,)
            if not isinstance(gate_weight, torch.Tensor) or tuple(
                gate_weight.shape
            ) != expected_weight_shape:
                raise ValueError(
                    "state candidate-mass weight must have shape "
                    f"{expected_weight_shape}"
                )
            if not isinstance(gate_bias, torch.Tensor) or tuple(
                gate_bias.shape
            ) != expected_bias_shape:
                raise ValueError(
                    "state candidate-mass bias must have shape "
                    f"{expected_bias_shape}"
                )
            if gate_weight.dtype != dtype or gate_bias.dtype != dtype:
                raise ValueError(
                    "state candidate-mass checkpoint dtype must match draft "
                    f"execution dtype {dtype}"
                )
            if not bool(torch.isfinite(gate_weight.float()).all()) or not bool(
                torch.isfinite(gate_bias.float()).all()
            ):
                raise ValueError("state candidate-mass checkpoint is non-finite")
            adapter_device = next(adapters.parameters()).device
            self.residual_tree_state_candidate_calibration_weight = (
                gate_weight.to(device=adapter_device).contiguous()
            )
            self.residual_tree_state_candidate_calibration_bias = (
                gate_bias.to(device=adapter_device).contiguous()
            )

        lambdas = getattr(spec_config, "residual_tree_head_lambdas", None)
        lambdas = lambdas or config.get("residual_tree_head_lambdas")
        if lambdas is not None:
            self.residual_tree_head_lambdas = [float(value) for value in lambdas]
            if len(self.residual_tree_head_lambdas) != self.residual_tree_total_heads:
                raise ValueError(
                    "residual_tree_head_lambdas must have one value per head"
                )

    def has_residual_tree_heads(self) -> bool:
        return self.residual_tree_total_heads > 0

    def compute_stock_top2_greedy_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return Stock H1's two best distinct target-token ids per state.

        This is a stock-tree control, not a residual-head proposal.  It runs
        the ordinary EAGLE H1 vocabulary projection exactly once for each
        input state, selects two draft-vocabulary logits, and maps them to
        target ids without scattering into the larger target vocabulary.
        """

        if not hasattr(self, "logits_processor") or not hasattr(self, "lm_head"):
            raise NotImplementedError(
                "stock top-2 tree drafting requires the EAGLE lm_head"
            )
        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or first_logits.ndim != 2:
            raise NotImplementedError(
                "stock top-2 tree drafting requires batched draft logits"
            )
        draft_vocab_size = int(first_logits.shape[-1])
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if draft_vocab_size < 2 or target_vocab_size < 2:
            raise NotImplementedError(
                "stock top-2 tree drafting requires at least two tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            draft_ids = torch.arange(
                draft_vocab_size,
                dtype=torch.long,
                device=first_logits.device,
            )
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = draft_ids
            else:
                target_ids = draft_ids + mapping.to(
                    device=first_logits.device,
                    dtype=torch.long,
                )
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "stock top-2 tree drafting requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        _, top_draft_tokens = torch.topk(
            first_logits,
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        )
        flat_target_tokens = target_ids.index_select(
            0,
            top_draft_tokens.reshape(-1),
        )
        return flat_target_tokens.reshape(top_draft_tokens.shape)

    def compute_stock_top2_greedy_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Stock H1 top-2 tokens and probabilities for pruned trees.

        Complete B2D1/B6D2 controls use the token-only hook above and keep
        their existing no-softmax hot path.  Larger best-first trees need real
        H1 probabilities to decide which branches enter the fixed node budget.
        """

        if not hasattr(self, "logits_processor") or not hasattr(self, "lm_head"):
            raise NotImplementedError(
                "stock top-2 tree drafting requires the EAGLE lm_head"
            )
        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or first_logits.ndim != 2:
            raise NotImplementedError(
                "stock top-2 tree drafting requires batched draft logits"
            )
        draft_vocab_size = int(first_logits.shape[-1])
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if draft_vocab_size < 2 or target_vocab_size < 2:
            raise NotImplementedError(
                "stock top-2 tree drafting requires at least two tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            draft_ids = torch.arange(
                draft_vocab_size,
                dtype=torch.long,
                device=first_logits.device,
            )
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = draft_ids
            else:
                target_ids = draft_ids + mapping.to(
                    device=first_logits.device,
                    dtype=torch.long,
                )
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "stock top-2 tree drafting requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        top_logits, top_draft_tokens = torch.topk(
            first_logits,
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        )
        flat_target_tokens = target_ids.index_select(
            0,
            top_draft_tokens.reshape(-1),
        )
        target_tokens = flat_target_tokens.reshape(top_draft_tokens.shape)
        normalizer = torch.logsumexp(first_logits.float(), dim=-1, keepdim=True)
        probabilities = torch.exp(top_logits.float() - normalizer)
        return target_tokens, probabilities

    def compute_stock_top10_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        *,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Stock H1 top-k tokens and probabilities for EAGLE-3 D8.

        The released tree uses ten candidates.  The width-nine control uses the
        identical projection and normalization while requesting nine entries
        from the same top-k kernel.
        """

        if top_k not in {9, 10}:
            raise ValueError("stock dynamic top_k must be 9 or 10")

        if not hasattr(self, "logits_processor") or not hasattr(self, "lm_head"):
            raise NotImplementedError(
                "stock dynamic tree drafting requires the EAGLE lm_head"
            )
        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or first_logits.ndim != 2:
            raise NotImplementedError(
                "stock dynamic tree drafting requires batched draft logits"
            )
        draft_vocab_size = int(first_logits.shape[-1])
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if draft_vocab_size < top_k or target_vocab_size < top_k:
            raise NotImplementedError(
                f"stock dynamic tree drafting requires at least {top_k} tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            draft_ids = torch.arange(
                draft_vocab_size,
                dtype=torch.long,
                device=first_logits.device,
            )
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = draft_ids
            else:
                target_ids = draft_ids + mapping.to(
                    device=first_logits.device,
                    dtype=torch.long,
                )
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "stock dynamic tree drafting requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        top_logits, top_draft_tokens = torch.topk(
            first_logits,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        flat_target_tokens = target_ids.index_select(
            0,
            top_draft_tokens.reshape(-1),
        )
        target_tokens = flat_target_tokens.reshape(top_draft_tokens.shape)
        normalizer = torch.logsumexp(first_logits.float(), dim=-1, keepdim=True)
        probabilities = torch.exp(top_logits.float() - normalizer)
        return target_tokens, probabilities

    def compute_residual_head_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_residual_tree_heads():
            raise ValueError("residual tree heads are not configured")

        packed_weight = getattr(
            self, "_residual_tree_packed_independent_weight", None
        )
        if packed_weight is not None:
            later_head_count = self.residual_tree_total_heads - 1
            later_draft_logits = torch.nn.functional.linear(
                hidden_states, packed_weight
            ).view(
                *hidden_states.shape[:-1],
                later_head_count,
                self.residual_tree_draft_vocab_size,
            )
            mapped_later_logits = self._map_residual_tree_draft_logits(
                later_draft_logits
            )
            if self.residual_tree_freeze_base_head:
                return torch.cat(
                    (
                        self.compute_logits(hidden_states).unsqueeze(-2),
                        mapped_later_logits,
                    ),
                    dim=-2,
                )
            return mapped_later_logits

        logits: list[torch.Tensor] = []
        base_draft_logits = None
        if self.residual_tree_freeze_base_head:
            logits.append(self.compute_logits(hidden_states))
            if self.residual_tree_adapter_output_mode == "logit_residual":
                base_draft_logits = self.logits_processor(
                    self.lm_head, hidden_states
                )
                if base_draft_logits is None:
                    raise ValueError("base draft logits are unavailable")
        for adapter in self.residual_tree_adapters:
            if self.residual_tree_adapter_output_mode == "independent_lm_head":
                logits.append(
                    self._map_residual_tree_draft_logits(adapter(hidden_states))
                )
            elif self.residual_tree_adapter_output_mode == "logit_residual":
                if base_draft_logits is None:
                    base_draft_logits = self.logits_processor(
                        self.lm_head, hidden_states
                    )
                    if base_draft_logits is None:
                        raise ValueError("base draft logits are unavailable")
                logits.append(
                    self._map_residual_tree_draft_logits(
                        base_draft_logits + adapter(hidden_states)
                    )
                )
            else:
                adapted = hidden_states + adapter(hidden_states)
                logits.append(self.compute_logits(adapted))
        return torch.stack(logits, dim=1)

    def compute_hybrid_top9_h2_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        tree_depth: int | None = None,
        return_h1_rank_overlap: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Return H1 ranks 1-9 plus H2's best distinct dynamic candidate.

        This diagnostic keeps the released dynamic tree's width ten while
        replacing only H1 rank ten.  H2 is conditioned after all nine H1
        tokens have been removed.  Candidate probabilities remain unweighted;
        the proposer supplies the H2-to-H1 accepted-mass ratio separately.
        """

        if not self.supports_residual_greedy_candidates():
            raise NotImplementedError(
                "hybrid top9+H2 selection requires a frozen base head and one "
                "independent or logit-residual H2"
            )

        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or (
            first_logits.shape[-1] != self.residual_tree_draft_vocab_size
        ):
            raise NotImplementedError(
                "hybrid top9+H2 selection requires draft-vocabulary H1 logits"
            )
        depth_bias = getattr(self, "residual_tree_h2_tree_depth_bias", None)
        if depth_bias is not None:
            if tree_depth is None or not 0 <= tree_depth < 8:
                raise ValueError(
                    "per-depth H2 bias requires a uniform tree_depth in [0, 7]"
                )
            # Every official dynamic-tree expansion batch contains one depth.
            # Supplying its vocabulary bias to the existing linear projection
            # lets the GEMM epilogue apply it without a separate add kernel.
            second_logits = torch.nn.functional.linear(
                hidden_states,
                self.residual_tree_adapters[0].weight,
                depth_bias[tree_depth],
            )
        elif self.residual_tree_adapter_output_mode == "logit_residual":
            second_logits = first_logits + self.residual_tree_adapters[0](hidden_states)
        else:
            second_logits = self.residual_tree_adapters[0](hidden_states)
        if second_logits.shape != first_logits.shape:
            raise ValueError("hybrid H2 logits must match H1 draft-vocabulary logits")

        draft_vocab_size = int(first_logits.shape[-1])
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if draft_vocab_size < 10 or target_vocab_size < 10:
            raise NotImplementedError(
                "hybrid top9+H2 selection requires at least ten tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            draft_ids = torch.arange(
                draft_vocab_size,
                dtype=torch.long,
                device=first_logits.device,
            )
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = draft_ids
            else:
                target_ids = draft_ids + mapping.to(
                    device=first_logits.device,
                    dtype=torch.long,
                )
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "hybrid top9+H2 selection requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        top_h1_logits, top_h1_draft_tokens = torch.topk(
            first_logits,
            k=9,
            dim=-1,
            largest=True,
            sorted=True,
        )
        h1_target_tokens = target_ids.index_select(
            0,
            top_h1_draft_tokens.reshape(-1),
        ).reshape(top_h1_draft_tokens.shape)
        h1_normalizer = torch.logsumexp(first_logits.float(), dim=-1, keepdim=True)
        h1_probabilities = torch.exp(top_h1_logits.float() - h1_normalizer)

        h2_q = torch.softmax(second_logits.float(), dim=-1).clamp_min(0.0)
        h2_q = h2_q / h2_q.sum(dim=-1, keepdim=True).clamp_min(
            _RESIDUAL_TREE_GREEDY_EPS
        )
        h2_q.scatter_(1, top_h1_draft_tokens, 0.0)
        remaining_mass = h2_q.sum(dim=-1, keepdim=True)
        use_uniform_fallback = remaining_mass <= _RESIDUAL_TREE_GREEDY_EPS
        h2_q = h2_q / torch.where(
            use_uniform_fallback,
            torch.ones_like(remaining_mass),
            remaining_mass,
        )
        h2_values, h2_draft_tokens = h2_q.max(dim=-1)
        mapped_h2_tokens = target_ids.index_select(0, h2_draft_tokens)

        # At most nine target ids are blocked, so one of target ids 0..9 is
        # always available for the dense selector's uniform fallback.
        fallback_domain = torch.arange(
            10,
            dtype=torch.long,
            device=first_logits.device,
        ).expand(hidden_states.shape[0], 10)
        fallback_available = (
            fallback_domain.unsqueeze(2) != h1_target_tokens.unsqueeze(1)
        ).all(dim=2)
        fallback_tokens = fallback_available.to(torch.int64).argmax(dim=1)
        fallback_probability = 1.0 / (target_vocab_size - 9)
        use_uniform_fallback = use_uniform_fallback.squeeze(1)
        h2_tokens = torch.where(
            use_uniform_fallback,
            fallback_tokens,
            mapped_h2_tokens,
        )
        h2_probabilities = torch.where(
            use_uniform_fallback,
            torch.full_like(h2_values, fallback_probability),
            h2_values,
        )

        tokens = torch.cat((h1_target_tokens, h2_tokens.unsqueeze(1)), dim=1)
        probabilities = torch.cat(
            (h1_probabilities, h2_probabilities.unsqueeze(1)),
            dim=1,
        )
        if return_h1_rank_overlap:
            top10_h1_draft_tokens = torch.topk(
                first_logits,
                k=10,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            top10_h1_target_tokens = target_ids.index_select(
                0,
                top10_h1_draft_tokens.reshape(-1),
            ).reshape(top10_h1_draft_tokens.shape)
            rank_ids = torch.arange(
                1,
                11,
                dtype=torch.long,
                device=first_logits.device,
            )
            h1_rank_overlap = torch.where(
                h2_tokens.unsqueeze(1) == top10_h1_target_tokens,
                rank_ids,
                0,
            ).amax(dim=1, keepdim=True)
            return tokens, probabilities, h1_rank_overlap
        return tokens, probabilities

    def _compute_hybrid_topk_h2_topk_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        *,
        h1_candidate_count: int,
        tree_depth: int | None = None,
        return_h1_rank_overlap: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Return a width-10 H1/H2 allocation after excluding selected H1.

        This is a non-residual diagnostic layout: both existing output heads
        contribute multiple ranked candidates while total width remains ten.
        Candidate probabilities remain unweighted; the proposer applies the
        H2-to-H1 accepted-mass ratio to all H2 columns.
        """

        if not 1 <= h1_candidate_count <= 9:
            raise ValueError("hybrid H1 candidate count must be in [1, 9]")
        h2_candidate_count = 10 - h1_candidate_count

        if not self.supports_residual_greedy_candidates():
            raise NotImplementedError(
                "hybrid H1/H2 selection requires a frozen base head and "
                "one independent or logit-residual H2"
            )

        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or (
            first_logits.shape[-1] != self.residual_tree_draft_vocab_size
        ):
            raise NotImplementedError(
                "hybrid H1/H2 selection requires draft-vocabulary H1 logits"
            )
        depth_bias = getattr(self, "residual_tree_h2_tree_depth_bias", None)
        if depth_bias is not None:
            if tree_depth is None or not 0 <= tree_depth < 8:
                raise ValueError(
                    "per-depth H2 bias requires a uniform tree_depth in [0, 7]"
                )
            second_logits = torch.nn.functional.linear(
                hidden_states,
                self.residual_tree_adapters[0].weight,
                depth_bias[tree_depth],
            )
        elif self.residual_tree_adapter_output_mode == "logit_residual":
            second_logits = first_logits + self.residual_tree_adapters[0](hidden_states)
        else:
            second_logits = self.residual_tree_adapters[0](hidden_states)
        if second_logits.shape != first_logits.shape:
            raise ValueError("hybrid H2 logits must match H1 draft-vocabulary logits")

        draft_vocab_size = int(first_logits.shape[-1])
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if draft_vocab_size < 10 or target_vocab_size < 10:
            raise NotImplementedError(
                "hybrid H1/H2 selection requires at least ten tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            draft_ids = torch.arange(
                draft_vocab_size,
                dtype=torch.long,
                device=first_logits.device,
            )
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = draft_ids
            else:
                target_ids = draft_ids + mapping.to(
                    device=first_logits.device,
                    dtype=torch.long,
                )
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "hybrid H1/H2 selection requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        top_h1_logits, top_h1_draft_tokens = torch.topk(
            first_logits,
            k=h1_candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )
        h1_target_tokens = target_ids.index_select(
            0,
            top_h1_draft_tokens.reshape(-1),
        ).reshape(top_h1_draft_tokens.shape)
        h1_normalizer = torch.logsumexp(first_logits.float(), dim=-1, keepdim=True)
        h1_probabilities = torch.exp(top_h1_logits.float() - h1_normalizer)

        h2_q = torch.softmax(second_logits.float(), dim=-1).clamp_min(0.0)
        h2_q = h2_q / h2_q.sum(dim=-1, keepdim=True).clamp_min(
            _RESIDUAL_TREE_GREEDY_EPS
        )
        h2_q.scatter_(1, top_h1_draft_tokens, 0.0)
        remaining_mass = h2_q.sum(dim=-1, keepdim=True)
        use_uniform_fallback = remaining_mass <= _RESIDUAL_TREE_GREEDY_EPS
        conditioned_h2 = h2_q / torch.where(
            use_uniform_fallback,
            torch.ones_like(remaining_mass),
            remaining_mass,
        )
        available = torch.ones_like(conditioned_h2, dtype=torch.bool)
        available.scatter_(1, top_h1_draft_tokens, False)
        uniform_h2 = available.to(conditioned_h2.dtype) / float(
            draft_vocab_size - h1_candidate_count
        )
        conditioned_h2 = torch.where(
            use_uniform_fallback,
            uniform_h2,
            conditioned_h2,
        )
        h2_probabilities, h2_draft_tokens = torch.topk(
            conditioned_h2,
            k=h2_candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )
        h2_target_tokens = target_ids.index_select(
            0,
            h2_draft_tokens.reshape(-1),
        ).reshape(h2_draft_tokens.shape)

        tokens = torch.cat((h1_target_tokens, h2_target_tokens), dim=1)
        probabilities = torch.cat((h1_probabilities, h2_probabilities), dim=1)
        if return_h1_rank_overlap:
            top10_h1_draft_tokens = torch.topk(
                first_logits,
                k=10,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            top10_h1_target_tokens = target_ids.index_select(
                0,
                top10_h1_draft_tokens.reshape(-1),
            ).reshape(top10_h1_draft_tokens.shape)
            rank_ids = torch.arange(
                1,
                11,
                dtype=torch.long,
                device=first_logits.device,
            )
            h1_rank_overlap = torch.where(
                h2_target_tokens.unsqueeze(2) == top10_h1_target_tokens.unsqueeze(1),
                rank_ids,
                0,
            ).amax(dim=2)
            return tokens, probabilities, h1_rank_overlap
        return tokens, probabilities

    def compute_hybrid_top8_h2_top2_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        tree_depth: int | None = None,
        return_h1_rank_overlap: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        return self._compute_hybrid_topk_h2_topk_dynamic_candidates(
            hidden_states,
            h1_candidate_count=8,
            tree_depth=tree_depth,
            return_h1_rank_overlap=return_h1_rank_overlap,
        )

    def compute_hybrid_top7_h2_top3_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        tree_depth: int | None = None,
        return_h1_rank_overlap: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        return self._compute_hybrid_topk_h2_topk_dynamic_candidates(
            hidden_states,
            h1_candidate_count=7,
            tree_depth=tree_depth,
            return_h1_rank_overlap=return_h1_rank_overlap,
        )

    def compute_hybrid_top6_h2_top4_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        tree_depth: int | None = None,
        return_h1_rank_overlap: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        return self._compute_hybrid_topk_h2_topk_dynamic_candidates(
            hidden_states,
            h1_candidate_count=6,
            tree_depth=tree_depth,
            return_h1_rank_overlap=return_h1_rank_overlap,
        )

    def compute_hybrid_top5_h2_top5_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        tree_depth: int | None = None,
        return_h1_rank_overlap: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        return self._compute_hybrid_topk_h2_topk_dynamic_candidates(
            hidden_states,
            h1_candidate_count=5,
            tree_depth=tree_depth,
            return_h1_rank_overlap=return_h1_rank_overlap,
        )

    def compute_hybrid_union_top10_dynamic_candidates(
        self,
        hidden_states: torch.Tensor,
        *,
        h2_to_h1_weight: float,
        tree_depth: int | None = None,
        return_union_provenance: bool = False,
        oracle_target_tokens: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Merge native H1/H2 top tens and retain ten unique candidates.

        Identical tokens under the same parent form one tree node. Their edge
        score is the larger of H1 probability and weighted H2 probability;
        correlated head probabilities are deliberately not added. Returned
        scores already include the H2-to-H1 weight, so the tree selector must
        use unit column weights.
        """

        h2_weight = float(h2_to_h1_weight)
        if not 0.0 < h2_weight <= 1.0:
            raise ValueError("hybrid union requires 0 < H2-to-H1 weight <= 1")
        clipped_mass_calibration = getattr(
            self,
            "residual_tree_hybrid_union_clipped_mass_calibration",
            None,
        )
        calibration_row = None
        if clipped_mass_calibration is not None:
            if tree_depth is None or not 0 <= tree_depth < 8:
                raise ValueError(
                    "hybrid-union clipped-mass calibration requires tree_depth "
                    "in [0, 7]"
                )
            if not math.isclose(h2_weight, 1.0, abs_tol=1e-12):
                raise ValueError(
                    "hybrid-union clipped-mass calibration replaces the H2 scalar "
                    "and therefore requires equal H1/H2 head lambdas"
                )
            calibration_row = clipped_mass_calibration[tree_depth]
        if not self.supports_residual_greedy_candidates():
            raise NotImplementedError(
                "hybrid union selection requires a frozen base head and one "
                "independent or logit-residual H2"
            )

        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or (
            first_logits.shape[-1] != self.residual_tree_draft_vocab_size
        ):
            raise NotImplementedError(
                "hybrid union selection requires draft-vocabulary H1 logits"
            )
        depth_bias = getattr(self, "residual_tree_h2_tree_depth_bias", None)
        if depth_bias is not None:
            if tree_depth is None or not 0 <= tree_depth < 8:
                raise ValueError(
                    "per-depth H2 bias requires a uniform tree_depth in [0, 7]"
                )
            second_logits = torch.nn.functional.linear(
                hidden_states,
                self.residual_tree_adapters[0].weight,
                depth_bias[tree_depth],
            )
        elif self.residual_tree_adapter_output_mode == "logit_residual":
            second_logits = first_logits + self.residual_tree_adapters[0](hidden_states)
        else:
            second_logits = self.residual_tree_adapters[0](hidden_states)
        if second_logits.shape != first_logits.shape:
            raise ValueError("hybrid H2 logits must match H1 draft-vocabulary logits")

        draft_vocab_size = int(first_logits.shape[-1])
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if draft_vocab_size < 10 or target_vocab_size < 10:
            raise NotImplementedError(
                "hybrid union selection requires at least ten tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            draft_ids = torch.arange(
                draft_vocab_size,
                dtype=torch.long,
                device=first_logits.device,
            )
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = draft_ids
            else:
                target_ids = draft_ids + mapping.to(
                    device=first_logits.device,
                    dtype=torch.long,
                )
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "hybrid union selection requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        use_fused_union = getattr(
            self, "_residual_tree_fused_hybrid_union", True
        ) and supports_fused_hybrid_union(
            first_logits,
            second_logits,
            target_ids,
        )
        if oracle_target_tokens is not None and (
            oracle_target_tokens.shape != (hidden_states.shape[0],)
            or oracle_target_tokens.device != hidden_states.device
            or oracle_target_tokens.dtype != torch.long
        ):
            raise ValueError(
                "hybrid-union oracle targets must be int64 [batch] on the "
                "hidden-state device"
            )
        if (
            use_fused_union
            and not return_union_provenance
            and oracle_target_tokens is None
        ):
            return fused_hybrid_union_top10(
                first_logits,
                second_logits,
                target_ids,
                h2_weight=h2_weight,
                clipped_mass_calibration=calibration_row,
            )
        fused_outputs = None
        if use_fused_union:
            fused_outputs = fused_hybrid_union_top10(
                first_logits,
                second_logits,
                target_ids,
                h2_weight=h2_weight,
                clipped_mass_calibration=calibration_row,
                return_head_candidates=True,
            )
            (
                selected_target_tokens,
                selected_scores,
                fused_head_draft_tokens,
                fused_head_scores,
            ) = fused_outputs
            top_h1_draft_tokens = fused_head_draft_tokens[:, 0].to(torch.long)
            top_h2_draft_tokens = fused_head_draft_tokens[:, 1].to(torch.long)
            h1_scores = fused_head_scores[:, 0]
            h2_scores = fused_head_scores[:, 1]
        else:
            top_h1_logits, top_h1_draft_tokens = torch.topk(
                first_logits,
                k=10,
                dim=-1,
                largest=True,
                sorted=True,
            )
            top_h2_logits, top_h2_draft_tokens = torch.topk(
                second_logits,
                k=10,
                dim=-1,
                largest=True,
                sorted=True,
            )
            h1_normalizer = torch.logsumexp(first_logits.float(), dim=-1, keepdim=True)
            h2_normalizer = torch.logsumexp(second_logits.float(), dim=-1, keepdim=True)
            h1_scores = torch.exp(top_h1_logits.float() - h1_normalizer).clamp_min(
                _RESIDUAL_TREE_GREEDY_EPS
            )
            h2_scores = torch.exp(
                top_h2_logits.float() - h2_normalizer
            ).clamp_min(
                _RESIDUAL_TREE_GREEDY_EPS
            )

        overlaps = top_h2_draft_tokens.unsqueeze(2) == top_h1_draft_tokens.unsqueeze(1)
        h1_shared = overlaps.any(dim=1)
        h2_shared = overlaps.any(dim=2)
        if calibration_row is None:
            weighted_h2_scores = h2_scores * h2_weight
            matching_h2_scores = torch.where(
                overlaps,
                weighted_h2_scores.unsqueeze(2),
                torch.zeros_like(weighted_h2_scores).unsqueeze(2),
            ).amax(dim=1)
            merged_h1_scores = torch.maximum(h1_scores, matching_h2_scores)
            unique_h2_scores = torch.where(
                h2_shared,
                torch.zeros_like(weighted_h2_scores),
                weighted_h2_scores,
            )
            h1_score_contributions = h1_scores
        else:
            ranks_zero = torch.arange(10, device=h1_scores.device).unsqueeze(0)
            h1_categories = h1_shared.to(torch.long)
            h2_categories = 2 + h2_shared.to(torch.long)
            h1_factors = calibration_row[h1_categories, ranks_zero]
            h2_factors = calibration_row[h2_categories, ranks_zero]
            h1_score_contributions = h1_scores * h1_factors
            calibrated_h2_scores = h2_scores * h2_factors
            matching_h2_scores = torch.where(
                overlaps,
                calibrated_h2_scores.unsqueeze(2),
                torch.zeros_like(calibrated_h2_scores).unsqueeze(2),
            ).amax(dim=1)
            merged_h1_scores = (
                h1_score_contributions + matching_h2_scores
            ).clamp_max(1.0)
            unique_h2_scores = torch.where(
                h2_shared,
                torch.zeros_like(calibrated_h2_scores),
                calibrated_h2_scores,
            )
        union_draft_tokens = torch.cat(
            (top_h1_draft_tokens, top_h2_draft_tokens),
            dim=1,
        )
        union_scores = torch.cat((merged_h1_scores, unique_h2_scores), dim=1)
        if fused_outputs is None:
            selected_scores, selected_indices = torch.topk(
                union_scores,
                k=10,
                dim=-1,
                largest=True,
                sorted=True,
            )
            selected_draft_tokens = union_draft_tokens.gather(1, selected_indices)
            selected_target_tokens = target_ids.index_select(
                0,
                selected_draft_tokens.reshape(-1),
            ).reshape(selected_draft_tokens.shape)
        else:
            union_target_tokens = target_ids.index_select(
                0, union_draft_tokens.reshape(-1)
            ).reshape(union_draft_tokens.shape)
            selected_indices = (
                (
                    selected_target_tokens.unsqueeze(2)
                    == union_target_tokens.unsqueeze(1)
                )
                .to(torch.int32)
                .argmax(dim=2)
            )
        if oracle_target_tokens is not None:
            union_target_tokens = target_ids.index_select(
                0, union_draft_tokens.reshape(-1)
            ).reshape(union_draft_tokens.shape)
            oracle_matches = union_target_tokens.eq(
                oracle_target_tokens.unsqueeze(1)
            ) & oracle_target_tokens.unsqueeze(1).ge(0)
            oracle_available = oracle_matches.any(dim=1)
            oracle_indices = torch.where(
                oracle_matches,
                union_scores,
                torch.full_like(union_scores, -1.0),
            ).argmax(dim=1)
            selected_target_tokens = union_target_tokens.gather(
                1, selected_indices
            )
            already_selected = selected_target_tokens.eq(
                oracle_target_tokens.unsqueeze(1)
            ) & oracle_target_tokens.unsqueeze(1).ge(0)
            selected_slots = already_selected.to(torch.int64).argmax(dim=1)
            selected_slots = torch.where(
                already_selected.any(dim=1),
                selected_slots,
                torch.full_like(selected_slots, 9),
            )
            selected_indices = selected_indices.clone()
            available_rows = oracle_available.nonzero(as_tuple=False).flatten()
            selected_indices[available_rows, selected_slots[available_rows]] = (
                oracle_indices[available_rows]
            )
            selected_target_tokens = union_target_tokens.gather(
                1, selected_indices
            )
            selected_scores = union_scores.gather(1, selected_indices)
            selected_scores[available_rows, selected_slots[available_rows]] = 1.0
        if not return_union_provenance:
            return selected_target_tokens, selected_scores

        batch_size = int(hidden_states.shape[0])
        ranks = (
            torch.arange(
                1,
                11,
                dtype=torch.long,
                device=hidden_states.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        h2_rank_for_h1 = torch.where(
            overlaps,
            ranks.unsqueeze(2),
            torch.zeros_like(ranks).unsqueeze(2),
        ).amax(dim=1)
        h1_rank_for_h2 = torch.where(
            overlaps,
            ranks.unsqueeze(1),
            torch.zeros_like(ranks).unsqueeze(1),
        ).amax(dim=2)
        h1_source_masks = 1 + 2 * (h2_rank_for_h1 > 0).to(torch.long)
        h2_source_masks = torch.where(
            h1_rank_for_h2 > 0,
            torch.full_like(h1_rank_for_h2, 3),
            torch.full_like(h1_rank_for_h2, 2),
        )
        h1_score_heads = (
            matching_h2_scores > h1_score_contributions
        ).to(torch.long)
        h2_score_heads = torch.ones_like(ranks)
        union_source_masks = torch.cat((h1_source_masks, h2_source_masks), dim=1)
        union_h1_ranks = torch.cat((ranks, h1_rank_for_h2), dim=1)
        union_h2_ranks = torch.cat((h2_rank_for_h1, ranks), dim=1)
        union_score_heads = torch.cat((h1_score_heads, h2_score_heads), dim=1)
        return (
            selected_target_tokens,
            selected_scores,
            union_source_masks.gather(1, selected_indices),
            union_h1_ranks.gather(1, selected_indices),
            union_h2_ranks.gather(1, selected_indices),
            union_score_heads.gather(1, selected_indices),
        )

    def _map_residual_tree_draft_logits(
        self,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Map an independent head's draft-vocabulary logits to target IDs."""

        draft_vocab_size = self.residual_tree_draft_vocab_size
        if draft_logits.shape[-1] != draft_vocab_size:
            raise ValueError("independent head output does not match draft_vocab_size")
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        mapping = getattr(self, "draft_id_to_target_id", None)
        if mapping is None:
            if target_vocab_size != draft_vocab_size:
                raise ValueError("draft-to-target token mapping is missing")
            return draft_logits

        base = torch.arange(draft_vocab_size, device=draft_logits.device)
        target_ids = base + mapping.to(device=draft_logits.device)
        if bool(((target_ids < 0) | (target_ids >= target_vocab_size)).any()):
            raise ValueError("draft-to-target token mapping is outside target vocab")
        target_logits = draft_logits.new_full(
            (*draft_logits.shape[:-1], target_vocab_size),
            float("-inf"),
        )
        target_logits[..., target_ids] = draft_logits
        return target_logits

    def compute_residual_head_probs(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.compute_residual_head_logits(hidden_states)
        return torch.softmax(logits.float(), dim=-1)

    def supports_residual_greedy_candidates(self) -> bool:
        """Return whether compact greedy candidates are exactly available."""

        return (
            self.residual_tree_total_heads == 2
            and self.residual_tree_freeze_base_head
            and self.residual_tree_adapter_output_mode
            in {"independent_lm_head", "logit_residual"}
            and len(self.residual_tree_adapters) == 1
            and hasattr(self, "logits_processor")
            and hasattr(self, "lm_head")
        )

    def _apply_state_candidate_mass_calibration(
        self,
        hidden_states: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Scale H2-H10 selected probabilities by per-state predicted mass."""

        weight = getattr(
            self,
            "residual_tree_state_candidate_calibration_weight",
            None,
        )
        bias = getattr(
            self,
            "residual_tree_state_candidate_calibration_bias",
            None,
        )
        if weight is None and bias is None:
            return probabilities
        if weight is None or bias is None:
            raise AssertionError("state candidate-mass calibration is incomplete")
        expected_shape = (hidden_states.shape[0], self.residual_tree_total_heads)
        if probabilities.shape != expected_shape:
            raise ValueError(
                "state-calibrated probabilities must have shape "
                f"{expected_shape}"
            )
        gate_logits = torch.nn.functional.linear(
            hidden_states.to(dtype=weight.dtype),
            weight,
            bias,
        )
        gates = torch.sigmoid(gate_logits.float()).clamp_min(
            _RESIDUAL_TREE_GREEDY_EPS
        )
        calibrated = probabilities.clone()
        calibrated[:, 1:] *= gates.to(dtype=calibrated.dtype)
        return calibrated

    def compute_residual_greedy_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select compact tokens and probabilities for any residual topology.

        The legacy checkpoint contract is two ordered heads: frozen stock H1
        followed by one independent or low-rank logit-residual H2. The ordered
        stack contract extends that layout with independent H3-H10 projections.
        Every later head masks all tokens selected by its predecessors before
        choosing one token.

        Returns target-vocabulary token ids and their conditioned
        probabilities, both with shape ``[batch, num_heads]`` in ordered-head
        order. The compact probabilities preserve ``lambda*q`` node ordering
        without retaining full proposal rows. Stochastic verification continues
        to use :meth:`compute_residual_head_probs`.
        """

        ordered_head_stack = (
            self.residual_tree_total_heads > 2
            and self.residual_tree_freeze_base_head
            and self.residual_tree_adapter_output_mode
            in {"independent_lm_head", "logit_residual"}
            and len(self.residual_tree_adapters)
            == self.residual_tree_total_heads - 1
            and hasattr(self, "logits_processor")
            and hasattr(self, "lm_head")
        )
        if not (
            self.supports_residual_greedy_candidates()
            or ordered_head_stack
        ):
            raise NotImplementedError(
                "compact greedy selection requires a frozen base head followed "
                "by one residual H2 or multiple independent ordered heads"
            )

        first_logits = self.logits_processor(self.lm_head, hidden_states)
        if first_logits is None or (
            first_logits.shape[-1] != self.residual_tree_draft_vocab_size
        ):
            raise NotImplementedError(
                "compact greedy selection requires draft-vocabulary base logits"
            )
        if ordered_head_stack:
            tokens, probabilities = self._compute_ordered_independent_greedy_candidates(
                hidden_states,
                first_logits=first_logits,
            )
            probabilities = self._apply_state_candidate_mass_calibration(
                hidden_states,
                probabilities,
            )
            return tokens, probabilities
        if self.residual_tree_adapter_output_mode == "logit_residual":
            second_logits = first_logits + self.residual_tree_adapters[0](hidden_states)
        else:
            second_logits = self.residual_tree_adapters[0](hidden_states)
        if second_logits.shape[-1] != self.residual_tree_draft_vocab_size:
            raise ValueError("residual head output does not match draft_vocab_size")

        draft_vocab_size = self.residual_tree_draft_vocab_size
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if target_vocab_size < 2:
            raise NotImplementedError(
                "compact greedy selection requires at least two target tokens"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != second_logits.device
            or target_ids.shape[0] != second_logits.shape[-1]
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = torch.arange(
                    draft_vocab_size,
                    dtype=torch.long,
                    device=second_logits.device,
                )
            else:
                target_ids = torch.arange(
                    draft_vocab_size,
                    dtype=torch.long,
                    device=second_logits.device,
                ) + mapping.to(device=second_logits.device, dtype=torch.long)
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (sorted_target_ids[1:] == sorted_target_ids[:-1]).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                # The legacy scatter uses overwrite semantics for duplicate
                # target ids.  Falling back is safer than silently changing
                # which draft logit survives.
                raise NotImplementedError(
                    "compact greedy selection requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        # Use the same unexpanded draft-vocabulary projection as stock
        # LocalArgmaxMixin.  With an injective D2T map, its normalization and
        # argmax are identical to scattering into the target vocabulary, while
        # avoiding a [batch, target_vocab] allocation.
        first_q = torch.softmax(first_logits.float(), dim=-1).clamp_min(0.0)
        first_q = first_q / first_q.sum(dim=-1, keepdim=True).clamp_min(
            _RESIDUAL_TREE_GREEDY_EPS
        )
        first_values, first_draft_tokens = first_q.max(dim=-1)
        first_tokens = target_ids.index_select(0, first_draft_tokens)
        first_probabilities = first_values

        # Keep the same probability-space sequence as the legacy dense path:
        # softmax, root-row normalization, ordered-head normalization, block,
        # and renormalize.  Masking logits first is not equivalent when nearly
        # all H2 mass is assigned to H1's selected token.
        second_q = torch.softmax(second_logits.float(), dim=-1).clamp_min(0.0)
        second_q = second_q / second_q.sum(dim=-1, keepdim=True).clamp_min(
            _RESIDUAL_TREE_GREEDY_EPS
        )
        second_q = second_q.clamp_min(0.0)
        second_q = second_q / second_q.sum(dim=-1, keepdim=True).clamp_min(
            _RESIDUAL_TREE_GREEDY_EPS
        )
        second_q.scatter_(
            1,
            first_draft_tokens.unsqueeze(1),
            0.0,
        )
        remaining_mass = second_q.sum(dim=-1, keepdim=True)
        use_uniform_fallback = remaining_mass <= _RESIDUAL_TREE_GREEDY_EPS
        second_q = second_q / torch.where(
            use_uniform_fallback,
            torch.ones_like(remaining_mass),
            remaining_mass,
        )
        second_values, second_draft_tokens = second_q.max(dim=-1)
        mapped_second_tokens = target_ids.index_select(0, second_draft_tokens)

        # The dense fallback is uniform over the complete target vocabulary,
        # including target ids absent from the draft vocabulary.  Its argmax is
        # therefore the smallest target id other than the blocked H1 token.
        fallback_tokens = torch.where(
            first_tokens == 0,
            torch.ones_like(first_tokens),
            torch.zeros_like(first_tokens),
        )
        fallback_probability = 1.0 / (target_vocab_size - 1)
        use_uniform_fallback = use_uniform_fallback.squeeze(1)
        second_tokens = torch.where(
            use_uniform_fallback,
            fallback_tokens,
            mapped_second_tokens,
        )
        second_probabilities = torch.where(
            use_uniform_fallback,
            torch.full_like(second_values, fallback_probability),
            second_values,
        )
        tokens = torch.stack((first_tokens, second_tokens), dim=1)
        probabilities = torch.stack((first_probabilities, second_probabilities), dim=1)
        return tokens, probabilities

    def _compute_ordered_independent_greedy_candidates(
        self,
        hidden_states: torch.Tensor,
        *,
        first_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one token from every independent ordered head."""

        draft_vocab_size = self.residual_tree_draft_vocab_size
        target_vocab_size = int(getattr(self.config, "vocab_size", draft_vocab_size))
        if first_logits.shape[-1] != draft_vocab_size:
            raise ValueError("ordered H1 output does not match draft_vocab_size")
        if target_vocab_size < self.residual_tree_total_heads:
            raise NotImplementedError(
                "ordered compact selection requires at least one token per head"
            )

        mapping = getattr(self, "draft_id_to_target_id", None)
        mapping_version = int(mapping._version) if mapping is not None else -1
        target_ids = self._residual_tree_cached_draft_target_ids
        if (
            target_ids is None
            or target_ids.device != first_logits.device
            or target_ids.shape[0] != draft_vocab_size
            or self._residual_tree_cached_draft_mapping_version != mapping_version
        ):
            if mapping is None:
                if target_vocab_size != draft_vocab_size:
                    raise ValueError("draft-to-target token mapping is missing")
                target_ids = torch.arange(
                    draft_vocab_size,
                    dtype=torch.long,
                    device=first_logits.device,
                )
            else:
                target_ids = torch.arange(
                    draft_vocab_size,
                    dtype=torch.long,
                    device=first_logits.device,
                ) + mapping.to(device=first_logits.device, dtype=torch.long)
            sorted_target_ids = torch.sort(target_ids).values
            invalid_mapping = (
                (target_ids < 0) | (target_ids >= target_vocab_size)
            ).any()
            duplicate_mapping = (
                sorted_target_ids[1:] == sorted_target_ids[:-1]
            ).any()
            invalid, duplicate = (
                torch.stack((invalid_mapping, duplicate_mapping)).cpu().tolist()
            )
            if bool(invalid):
                raise ValueError(
                    "draft-to-target token mapping is outside target vocab"
                )
            if bool(duplicate):
                raise NotImplementedError(
                    "ordered compact selection requires an injective token mapping"
                )
            self._residual_tree_cached_draft_target_ids = target_ids
            self._residual_tree_cached_draft_mapping_version = mapping_version

        independent_head_top1 = (
            getattr(self, "residual_tree_required_candidate_selection", None)
            == "head_top1"
        )
        if not getattr(self, "_residual_tree_fused_ordered_heads", True):
            selected_draft_tokens: list[torch.Tensor] = []
            selected_target_tokens: list[torch.Tensor] = []
            selected_probabilities: list[torch.Tensor] = []
            head_logits = [first_logits]
            if self.residual_tree_adapter_output_mode == "logit_residual":
                head_logits.extend(
                    first_logits + adapter(hidden_states)
                    for adapter in self.residual_tree_adapters
                )
            else:
                head_logits.extend(
                    adapter(hidden_states)
                    for adapter in self.residual_tree_adapters
                )
            for head_index, logits in enumerate(head_logits):
                if logits.shape != first_logits.shape:
                    raise ValueError(
                        f"ordered H{head_index + 1} output does not match H1 logits"
                    )
                conditioned_logits = logits.float()
                if selected_draft_tokens and not independent_head_top1:
                    blocked = torch.stack(selected_draft_tokens, dim=1)
                    conditioned_logits.scatter_(1, blocked, float("-inf"))
                log_probabilities = torch.log_softmax(conditioned_logits, dim=1)
                draft_token = torch.argmax(log_probabilities, dim=1)
                selected_draft_tokens.append(draft_token)
                selected_target_tokens.append(
                    target_ids.index_select(0, draft_token)
                )
                selected_probabilities.append(
                    torch.exp(
                        log_probabilities.gather(
                            1, draft_token.unsqueeze(1)
                        ).squeeze(1)
                    )
                )
            return (
                torch.stack(selected_target_tokens, dim=1),
                torch.stack(selected_probabilities, dim=1),
            )

        packed_weight = getattr(
            self, "_residual_tree_packed_independent_weight", None
        )
        packed_logit_in = getattr(
            self, "_residual_tree_packed_logit_in_weight", None
        )
        packed_logit_out = getattr(
            self, "_residual_tree_packed_logit_out_weight", None
        )
        shared_logit_out = getattr(
            self, "_residual_tree_shared_logit_out_weight", None
        )
        if packed_weight is not None:
            later_logits = torch.nn.functional.linear(
                hidden_states, packed_weight
            ).view(
                hidden_states.shape[0],
                self.residual_tree_total_heads - 1,
                draft_vocab_size,
            )
        elif packed_logit_in is not None and shared_logit_out is not None:
            later_head_count = self.residual_tree_total_heads - 1
            rank = shared_logit_out.shape[1]
            low_rank_hidden = torch.nn.functional.linear(
                hidden_states, packed_logit_in
            ).view(hidden_states.shape[0], later_head_count, rank)
            later_logits = torch.nn.functional.linear(
                low_rank_hidden.reshape(-1, rank), shared_logit_out
            ).view(
                hidden_states.shape[0],
                later_head_count,
                draft_vocab_size,
            )
            later_logits = later_logits + first_logits.unsqueeze(1)
        elif packed_logit_in is not None and packed_logit_out is not None:
            later_head_count = self.residual_tree_total_heads - 1
            rank = packed_logit_out.shape[2]
            low_rank_hidden = torch.nn.functional.linear(
                hidden_states, packed_logit_in
            ).view(hidden_states.shape[0], later_head_count, rank)
            later_logits = torch.bmm(
                low_rank_hidden.transpose(0, 1),
                packed_logit_out.transpose(1, 2),
            ).transpose(0, 1)
            later_logits = later_logits + first_logits.unsqueeze(1)
        else:
            later_logits = torch.stack(
                [adapter(hidden_states) for adapter in self.residual_tree_adapters],
                dim=1,
            )
            if self.residual_tree_adapter_output_mode == "logit_residual":
                later_logits = later_logits + first_logits.unsqueeze(1)
        if later_logits.shape != (
            first_logits.shape[0],
            self.residual_tree_total_heads - 1,
            draft_vocab_size,
        ):
            raise ValueError("ordered later-head outputs do not match H1 logits")

        if getattr(self, "_residual_tree_fused_ordered_selection", False) and (
            supports_fused_ordered_selection(first_logits, later_logits)
        ):
            if independent_head_top1:
                selected_draft_token_tensor, selected_probabilities = (
                    fused_ordered_head_top1(first_logits, later_logits)
                )
            else:
                selected_draft_token_tensor, selected_probabilities = (
                    fused_ordered_distinct_top1(first_logits, later_logits)
                )
            selected_target_tokens = target_ids.index_select(
                0, selected_draft_token_tensor.reshape(-1)
            ).reshape(selected_draft_token_tensor.shape)
            return selected_target_tokens, selected_probabilities

        # Normalize every complete head row together. Independent heads keep
        # their native distribution; conditioned heads mask prior winners.
        conditioned_logits = torch.cat(
            (first_logits.unsqueeze(1), later_logits), dim=1
        ).float()
        if independent_head_top1:
            selected_draft_token_tensor = torch.argmax(
                conditioned_logits, dim=2
            )
        else:
            selected_draft_tokens: list[torch.Tensor] = []
            for head_index in range(self.residual_tree_total_heads):
                draft_token = torch.argmax(
                    conditioned_logits[:, head_index], dim=1
                )
                selected_draft_tokens.append(draft_token)
                if head_index + 1 < self.residual_tree_total_heads:
                    later_rows = conditioned_logits[:, head_index + 1 :]
                    blocked = draft_token[:, None, None].expand(
                        -1, later_rows.shape[1], 1
                    )
                    later_rows.scatter_(2, blocked, float("-inf"))
            selected_draft_token_tensor = torch.stack(
                selected_draft_tokens, dim=1
            )
        log_probabilities = torch.log_softmax(conditioned_logits, dim=2)
        selected_probabilities = torch.exp(
            log_probabilities.gather(
                2, selected_draft_token_tensor.unsqueeze(2)
            ).squeeze(2)
        )
        selected_target_tokens = target_ids.index_select(
            0, selected_draft_token_tensor.reshape(-1)
        ).reshape(selected_draft_token_tensor.shape)
        return selected_target_tokens, selected_probabilities

    def supports_residual_b2d1_greedy_candidates(self) -> bool:
        """Backward-compatible alias for older proposer integrations."""

        return self.supports_residual_greedy_candidates()

    def compute_residual_b2d1_greedy_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible alias for the topology-generic hook."""

        return self.compute_residual_greedy_candidates(hidden_states)

    def compute_residual_b2d1_greedy_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return only token ids from the exact greedy B2D1 selector."""

        tokens, _ = self.compute_residual_greedy_candidates(hidden_states)
        return tokens


def _load_residual_tree_adapter_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("residual_tree_adapter_config must contain a JSON object")
    return data


def _residual_tree_adapter_dtype(vllm_config: VllmConfig) -> torch.dtype:
    spec_config = vllm_config.speculative_config
    draft_config = getattr(spec_config, "draft_model_config", None)
    dtype = getattr(draft_config, "dtype", None)
    if dtype is None:
        dtype = vllm_config.model_config.dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"residual tree adapter dtype must be torch.dtype, got {dtype!r}"
        )
    if not (dtype.is_floating_point or dtype.is_complex):
        raise TypeError(
            f"residual tree adapter dtype must be floating point, got {dtype}"
        )
    return dtype
