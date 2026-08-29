# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import inspect
import math
import os
from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    replace,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import (
    empty_exponential_noise_like,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata, TreeSpecDecodeMetadata
from vllm.v1.spec_decode.residual_tree import (
    TreeListwiseFeatures,
    build_tree_attention_mask,
    select_batched_eagle3_dynamic_trees,
    select_batched_greedy_residual_trees,
    select_residual_tree,
    tree_to_draft_token_tree,
)
from vllm.v1.spec_decode.residual_tree_trace import (
    ROOT_VERIFIER_PROB_TRACE_ENV,
    TRAINING_FEATURE_TRACE_ENV,
    append_residual_tree_trace,
    append_residual_tree_training_features,
)
from vllm.v1.spec_decode.tree_schema import DraftTokenTree
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
    next_power_of_2,
)
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

_HYBRID_DYNAMIC_LAYOUTS: dict[str, tuple[int, int, str]] = {
    "hybrid_top9_h2_dynamic": (
        9,
        1,
        "compute_hybrid_top9_h2_dynamic_candidates",
    ),
    "hybrid_top8_h2_top2_dynamic": (
        8,
        2,
        "compute_hybrid_top8_h2_top2_dynamic_candidates",
    ),
    "hybrid_top7_h2_top3_dynamic": (
        7,
        3,
        "compute_hybrid_top7_h2_top3_dynamic_candidates",
    ),
    "hybrid_top6_h2_top4_dynamic": (
        6,
        4,
        "compute_hybrid_top6_h2_top4_dynamic_candidates",
    ),
    "hybrid_top5_h2_top5_dynamic": (
        5,
        5,
        "compute_hybrid_top5_h2_top5_dynamic_candidates",
    ),
    "hybrid_union_top10_dynamic": (
        10,
        0,
        "compute_hybrid_union_top10_dynamic_candidates",
    ),
}


@dataclasses.dataclass(frozen=True)
class _ResidualTreeDraftState:
    proposal_hidden: torch.Tensor
    transition_hidden: torch.Tensor
    position: torch.Tensor | None = None
    payload: Any | None = None


@dataclasses.dataclass(frozen=True)
class _ResidualTreeKVPayload:
    req_index: int
    root_position: int
    path_kv: tuple[tuple[torch.Tensor, ...], ...] = ()
    path_token_ids: tuple[int, ...] = ()
    path_hidden_inputs: tuple[torch.Tensor, ...] = ()


def _build_greedy_b2d1_draft_trees(
    token_rows: Sequence[Sequence[int]],
    probability_rows: Sequence[Sequence[float]],
    *,
    head_lambdas: Sequence[float],
    tree_policy: str,
) -> list[DraftTokenTree]:
    """Build legacy-ordered sibling topology without proposal tensors."""

    if len(token_rows) != len(probability_rows):
        raise ValueError("greedy B2D1 tokens and probabilities must align")
    if len(head_lambdas) != 2:
        raise ValueError("greedy B2D1 selection requires two head lambdas")
    trees: list[DraftTokenTree] = []
    for token_row, probability_row in zip(token_rows, probability_rows, strict=True):
        if len(token_row) != 2 or len(probability_row) != 2:
            raise ValueError("greedy B2D1 selection must return two candidates")
        first_token, second_token = int(token_row[0]), int(token_row[1])
        if first_token == second_token:
            raise ValueError("distinct residual heads selected a duplicate token")
        candidates = [
            (
                float(head_lambdas[head_id]) * float(probability_row[head_id]),
                head_id,
                int(token_row[head_id]),
            )
            for head_id in range(2)
        ]
        if tree_policy == "best_first":
            candidates.sort(key=lambda item: (-item[0], item[1]))
        elif tree_policy == "breadth_first":
            candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        else:
            raise ValueError(f"unsupported tree_policy: {tree_policy}")
        trees.append(
            DraftTokenTree(
                node_token_ids=[-1, *(candidate[2] for candidate in candidates)],
                parent_node_ids=[-1, 0, 0],
                node_depths=[0, 1, 1],
                node_priorities=[
                    1.0,
                    *(candidate[0] for candidate in candidates),
                ],
                child_start_indices=[0, 2, 2],
                child_end_indices=[2, 2, 2],
                child_node_ids=[1, 2],
                contributor_child_node_ids=[],
                contributor_proposal_rows=[],
                contributor_head_ids=[],
                contributor_scores=[],
                proposal_num_rows=0,
            )
        )
    return trees


@dataclasses.dataclass(frozen=True)
class _ResidualTreeTransitionPlan:
    """One topology-masked EAGLE forward for several branch transitions."""

    req_index: int
    root_position: int
    token_ids: tuple[int, ...]
    hidden_inputs: tuple[torch.Tensor, ...]
    parent_local_indices: tuple[int, ...]
    node_depths: tuple[int, ...]
    result_local_indices: tuple[int, ...]
    child_payloads: tuple[_ResidualTreeKVPayload, ...]


@dataclasses.dataclass(frozen=True)
class _ResidualTreeBatchTransitionPlan:
    """Several request-local union tries packed into one EAGLE forward."""

    request_indices: tuple[int, ...]
    root_positions: tuple[int, ...]
    query_start_locs: tuple[int, ...]
    token_ids: tuple[int, ...]
    hidden_inputs: tuple[torch.Tensor, ...]
    parent_local_indices: tuple[int, ...]
    node_depths: tuple[int, ...]
    result_local_indices: tuple[int, ...]
    child_payloads: tuple[_ResidualTreeKVPayload, ...]


class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self._share_mtp_indices = False

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # DeepSeek V4 MTP consumes the target's pre-hc_head residual stream,
        # shape (T, hc_mult * hidden_size). Expand the hidden_states buffer
        # so target_hidden_states fits; detect DeepseekV4 via draft hf_config.
        draft_hf_config = self.draft_model_config.hf_config
        if hasattr(draft_hf_config, "compress_ratios") and hasattr(
            draft_hf_config, "hc_mult"
        ):
            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        # When True, all draft steps reuse the same position as the
        # first step instead of advancing by one each iteration.
        # Used by draft models with Q-only attention that share KV
        # with the target and always predict from the same position.
        self.constant_draft_positions: bool = False

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        # Can be specialized by methods like DFlash to reduce the limit
        self.max_query_tokens = self.max_num_tokens
        self.max_positions = self.max_num_tokens

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # Cudagraph dispatcher for PIECEWISE-only dispatching in eagle.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_positions + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_positions + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_positions,
                dtype=torch.int64,
                device=device,
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Will be set when we initialize the attention backend
        self.block_size: int = -1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._warn_if_multimodal()
            self._raise_if_mrope()

        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # For draft models and parallel drafting, we need to keep track of
            # which tokens are rejected to update the slot mapping with padding slots.
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # For parallel drafting, we also need to keep track of which tokens
            # are parallel-padding tokens used to sample at later positions.
            # We populate this tensor even when using draft models for simplicity.
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
            device=device,
            with_numpy=True,
        )
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
        self._last_draft_probs: torch.Tensor | None = None
        self._residual_tree_common_attn_metadata: CommonAttentionMetadata | None = None
        self._residual_tree_kv_caches: tuple[torch.Tensor, ...] = ()
        self._residual_tree_greedy_scorer: Any | None = None
        self._residual_tree_greedy_scorer_path: str | None = None
        self._residual_tree_trace_step = 0
        self._residual_tree_training_feature_dir = os.environ.get(
            TRAINING_FEATURE_TRACE_ENV
        )
        self._residual_tree_probability_trace_dir = os.environ.get(
            ROOT_VERIFIER_PROB_TRACE_ENV
        )
        self._residual_tree_runtime_config: dict[str, Any] = {}
        self._stock_top2_rank_scores: torch.Tensor | None = None

        self._slot_mapping_buffer = torch.zeros(
            self.max_positions, dtype=torch.int64, device=device
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
            )

            # MiniMax-M3 sparse (lightning-indexer) attention. The multi-step
            # drafting machinery is shared code at num_speculative_tokens>1.
            # this just opts the metadata into the ROCm allowlist.
            from vllm.models.minimax_m3.common.sparse_attention import (
                MiniMaxM3SparseMetadata,
            )
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
                DeepseekV32IndexerMetadata,
                MiniMaxM3SparseMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

    def _raise_if_padded_drafter_batch_disabled(self):
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _warn_if_multimodal(self):
        if self.supports_mm_inputs:
            logger.warning(
                "Speculative Decoding with draft models or parallel drafting "
                "does not fully support multimodal models yet. "
                "Proceeding with text-only speculative decoding."
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        # For parallel drafting, we need the token ID to use for masked slots
        # And for EAGLE + parallel drafting, we need the hidden state tensor to use
        # for those masked slots.

        model_hf_config = self.draft_model_config.hf_config
        # DFlash stores mask_token_id in dflash_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        elif hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token`, `ptd_token_id`, or "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for the drafter.

        Only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs:
            return logits.argmax(dim=-1), None
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None

        # Parallel drafting (e.g. DFlash) samples num_speculative_tokens rows
        # per request in a single pass, so logits has batch_size * K rows while
        # the sampling metadata is per-request. The rows are request-major
        # (K consecutive slots per request), so repeat_interleave the
        # per-request temperature to match before probabilistic sampling.
        temperature = sampling_metadata.temperature
        if temperature is not None and temperature.shape[0] != logits.shape[0]:
            assert logits.shape[0] % temperature.shape[0] == 0
            factor = logits.shape[0] // temperature.shape[0]
            sampling_metadata = dataclasses.replace(
                sampling_metadata,
                temperature=temperature.repeat_interleave(factor, dim=0),
            )

        return compute_probs_and_sample_next_token(
            logits, sampling_metadata, self.use_fp64_gumbel
        )

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
        return self._sample_from_logits(logits, sampling_metadata)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        return self._last_draft_probs

    def propose_residual_tree(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
        request_ids: Sequence[str] | None = None,
    ) -> list[DraftTokenTree]:
        """Build fixed-budget residual-head draft trees.

        This is intentionally separate from ``propose`` because the linear
        EAGLE loop has one KV/attention path per request. Tree drafting needs
        an explicit model API for residual heads and ``G_theta(s_u, emb(y))``;
        otherwise running the linear loop across branches would contaminate
        non-ancestor states.
        """

        if self.parallel_drafting:
            raise NotImplementedError(
                "residual-tree drafting requires an autoregressive tree "
                "transition and does not support parallel_drafting"
            )
        if self.uses_mrope or self.uses_xdrope_dim > 0:
            raise NotImplementedError(
                "residual-tree drafting does not support M-RoPE or XD-RoPE "
                "position handling yet"
            )
        residual_model = self._unwrap_residual_tree_model()
        configured_candidate_selection = (
            self.speculative_config.residual_tree_candidate_selection
        )
        self._validate_residual_tree_model_api(residual_model)

        self.num_speculative_tokens = num_speculative_tokens
        runtime_config = self._residual_tree_runtime_config
        node_budget = int(
            runtime_config.get("node_budget", self.num_speculative_tokens)
        )
        if node_budget > int(self.num_speculative_tokens):
            raise ValueError(
                "residual-tree runtime node budget exceeds the scheduled "
                "speculative-token limit"
            )
        self._last_draft_probs = None
        self._residual_tree_common_attn_metadata = None
        self._residual_tree_kv_caches = ()
        batch_size = common_attn_metadata.batch_size()
        if request_ids is not None and len(request_ids) != batch_size:
            raise ValueError(
                "residual-tree request_ids must match the draft batch size"
            )

        if self.method in ("eagle3", "dflash"):
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        with (
            record_function_or_nullcontext("residual_tree: first_pass_forward"),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    slot_mapping_size, common_attn_metadata.slot_mapping
                ),
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        if self.num_speculative_tokens == 0:
            return [DraftTokenTree.root_only() for _ in range(batch_size)]

        proposal_hidden = last_hidden_states[token_indices_to_sample]
        transition_hidden = hidden_states[token_indices_to_sample]

        if self.uses_mrope:
            root_positions = self.mrope_positions[:, token_indices_to_sample].T
        else:
            root_positions = self.positions[token_indices_to_sample]
        self._residual_tree_common_attn_metadata = common_attn_metadata
        if self._has_model_residual_tree_transition(residual_model):
            self._residual_tree_kv_caches = ()
        else:
            self._residual_tree_kv_caches = self._collect_draft_kv_caches()
        root_states = []
        root_positions_cpu = root_positions.detach().cpu().tolist()
        for i, root_position_value in enumerate(root_positions_cpu):
            root_position = int(root_position_value)
            root_state = _ResidualTreeDraftState(
                proposal_hidden=proposal_hidden[i].detach(),
                transition_hidden=transition_hidden[i].detach(),
                position=root_positions[i].detach(),
                payload=_ResidualTreeKVPayload(
                    req_index=i,
                    root_position=root_position,
                ),
            )
            root_states.append(
                self._init_residual_tree_state(residual_model, root_state)
            )

        if configured_candidate_selection == "stock_top2":
            return self._propose_stock_top2_trees(
                residual_model,
                root_states,
                runtime_config=runtime_config,
                sampling_metadata=sampling_metadata,
            )
        if configured_candidate_selection in {
            "stock_top9_dynamic",
            "stock_top10_dynamic",
        }:
            return self._propose_stock_dynamic_trees(
                residual_model,
                root_states,
                runtime_config=runtime_config,
                sampling_metadata=sampling_metadata,
            )
        if configured_candidate_selection in _HYBRID_DYNAMIC_LAYOUTS:
            return self._propose_hybrid_dynamic_trees(
                residual_model,
                root_states,
                runtime_config=runtime_config,
                sampling_metadata=sampling_metadata,
                request_ids=request_ids,
            )

        head_lambdas = self._get_residual_tree_head_lambdas(
            residual_model,
            root_states[0],
        )
        depth_head_lambdas = runtime_config.get(
            "depth_head_lambdas",
            getattr(
                self.speculative_config,
                "residual_tree_depth_head_lambdas",
                None,
            ),
        )
        active_heads = self.speculative_config.residual_tree_active_heads
        max_depth = runtime_config.get(
            "max_depth",
            self.speculative_config.residual_tree_max_depth,
        )
        scorer_mode = runtime_config.get(
            "scorer_mode",
            getattr(
                self.speculative_config,
                "residual_tree_scorer_mode",
                "lambda_q",
            ),
        )
        listwise_scorer = (
            self._score_residual_tree_candidates
            if scorer_mode == "greedy_listwise"
            else None
        )
        if listwise_scorer is not None:
            self._load_residual_tree_greedy_scorer()
        trace_path = runtime_config.get(
            "trace_path",
            getattr(
                self.speculative_config,
                "residual_tree_trace_path",
                None,
            ),
        )
        tree_policy = runtime_config.get(
            "tree_policy",
            getattr(
                self.speculative_config,
                "residual_tree_tree_policy",
                "best_first",
            ),
        )
        batch_drafting = bool(
            runtime_config.get(
                "batch_drafting",
                getattr(
                    self.speculative_config,
                    "residual_tree_batch_drafting",
                    True,
                ),
            )
        )
        head_prior_probabilities = runtime_config.get(
            "head_prior_probabilities",
            getattr(
                self.speculative_config,
                "residual_tree_head_prior_probabilities",
                None,
            ),
        )
        scorer_top_k = int(
            runtime_config.get(
                "scorer_top_k",
                getattr(
                    self.speculative_config,
                    "residual_tree_scorer_top_k",
                    10,
                ),
            )
        )
        training_feature_dir = self._residual_tree_training_feature_dir
        probability_trace_dir = self._residual_tree_probability_trace_dir
        trace_enabled = bool(
            trace_path is not None or training_feature_dir or probability_trace_dir
        )
        trace_step = self._residual_tree_trace_step
        if trace_enabled:
            self._residual_tree_trace_step += 1
            if training_feature_dir:
                feature_request_ids = (
                    [str(request_id) for request_id in request_ids]
                    if request_ids is not None
                    else [str(index) for index in range(batch_size)]
                )
                append_residual_tree_training_features(
                    training_feature_dir,
                    step=trace_step,
                    request_ids=feature_request_ids,
                    proposal_hidden=torch.stack(
                        [state.proposal_hidden for state in root_states], dim=0
                    ),
                    positions=torch.stack(
                        [state.position.reshape(()) for state in root_states], dim=0
                    ),
                )

        if tree_policy == "eagle3_dynamic":
            if not bool(sampling_metadata.all_greedy):
                raise NotImplementedError(
                    "EAGLE-3 dynamic residual trees support greedy verification only"
                )
            if not 1 <= node_budget <= 60 or int(max_depth or 0) != 8:
                raise ValueError(
                    "EAGLE-3 dynamic residual trees require D8 with a node "
                    "budget between 1 and 60"
                )
            if scorer_mode != "lambda_q":
                raise ValueError(
                    "EAGLE-3 dynamic residual trees require lambda_q scoring"
                )
            if not batch_drafting:
                raise ValueError(
                    "EAGLE-3 dynamic residual trees require batched drafting"
                )
            if training_feature_dir or probability_trace_dir:
                raise ValueError(
                    "EAGLE-3 dynamic serving does not support training-feature "
                    "or probability traces"
                )
            if configured_candidate_selection not in {
                "head_top1",
                "distinct_head_top1",
            }:
                raise ValueError(
                    "residual EAGLE-3 dynamic trees require head_top1 or "
                    "distinct_head_top1"
                )
            canonical_replay = sampling_metadata.canonical_token_replay
            tree_oracle_enabled = (
                canonical_replay is not None
                and canonical_replay.tree_construction_oracle_batch()
            )

            def tree_oracle_targets(
                states: Sequence[_ResidualTreeDraftState],
            ) -> torch.Tensor | None:
                if not tree_oracle_enabled:
                    return None
                assert canonical_replay is not None
                target_ids: list[int] = []
                for state in states:
                    payload = state.payload
                    if not isinstance(payload, _ResidualTreeKVPayload):
                        raise ValueError(
                            "tree-construction oracle requires internal EAGLE states"
                        )
                    req_index = payload.req_index
                    config = canonical_replay.configs[req_index]
                    if config is None:
                        raise ValueError(
                            "tree-construction oracle is missing a request reference"
                        )
                    root_offset = (
                        len(canonical_replay.output_token_ids[req_index]) + 1
                    )
                    path = payload.path_token_ids
                    reference_path = config.token_ids[
                        root_offset : root_offset + len(path)
                    ]
                    target_offset = root_offset + len(path)
                    if tuple(path) != tuple(reference_path) or target_offset >= len(
                        config.token_ids
                    ):
                        target_ids.append(-1)
                    else:
                        target_ids.append(config.token_at(target_offset))
                return torch.tensor(
                    target_ids,
                    dtype=torch.long,
                    device=states[0].proposal_hidden.device,
                )

            def candidate_batch_fn(
                states: Sequence[_ResidualTreeDraftState],
            ) -> tuple[torch.Tensor, torch.Tensor]:
                tokens, probabilities = self._residual_tree_greedy_candidates_batch(
                    residual_model,
                    states,
                )
                target_ids = tree_oracle_targets(states)
                if target_ids is None:
                    return tokens, probabilities
                target_mask = target_ids[:, None].ge(0) & tokens.eq(
                    target_ids[:, None]
                )
                below_one = torch.nextafter(
                    torch.ones((), dtype=probabilities.dtype, device=probabilities.device),
                    torch.zeros((), dtype=probabilities.dtype, device=probabilities.device),
                )
                promoted = torch.where(
                    target_mask,
                    torch.ones_like(probabilities),
                    probabilities.clamp_max(below_one),
                )
                return tokens, promoted

            root_candidates = candidate_batch_fn(root_states)
            diagnostic_target_paths = None
            diagnostic_target_metadata = None
            if (
                trace_path is not None
                and canonical_replay is not None
                and canonical_replay.proposal_only_batch()
            ):
                diagnostic_target_paths = []
                diagnostic_target_metadata = []
                for root_state in root_states:
                    payload = root_state.payload
                    if not isinstance(payload, _ResidualTreeKVPayload):
                        raise ValueError(
                            "canonical target-path diagnostics require internal "
                            "EAGLE states"
                        )
                    req_index = payload.req_index
                    config = canonical_replay.configs[req_index]
                    if config is None:
                        raise ValueError(
                            "canonical target-path diagnostics are missing a "
                            "request reference"
                        )
                    root_offset = (
                        len(canonical_replay.output_token_ids[req_index]) + 1
                    )
                    diagnostic_target_paths.append(
                        config.token_ids[root_offset : root_offset + int(max_depth)]
                    )
                    diagnostic_target_metadata.append(
                        {
                            "replay_id": config.replay_id,
                            "output_offset": root_offset,
                        }
                    )
            with record_function_or_nullcontext(
                "residual_tree: select_eagle3_dynamic_trees"
            ):
                selected_trees = select_batched_eagle3_dynamic_trees(
                    root_states=root_states,
                    root_candidate_tokens=root_candidates[0],
                    root_candidate_probabilities=root_candidates[1],
                    candidate_batch_fn=candidate_batch_fn,
                    transition_batch_fn=lambda states, token_ids: (
                        self._residual_tree_transition_batch(
                            residual_model,
                            states,
                            token_ids,
                        )
                    ),
                    head_lambdas=head_lambdas,
                    depth_head_lambdas=depth_head_lambdas,
                    node_budget=node_budget,
                    max_depth=int(max_depth),
                    frontier_width=10,
                    collect_dynamic_provenance=trace_path is not None,
                    diagnostic_target_paths=diagnostic_target_paths,
                    candidate_selection=configured_candidate_selection,
                )
            if diagnostic_target_metadata is not None:
                for tree, metadata in zip(
                    selected_trees,
                    diagnostic_target_metadata,
                    strict=True,
                ):
                    if tree.dynamic_provenance is None:
                        raise AssertionError(
                            "canonical target-path provenance is unavailable"
                        )
                    tree.dynamic_provenance["canonical_target_path"].update(metadata)
            draft_trees = [
                tree_to_draft_token_tree(tree) for tree in selected_trees
            ]
            if trace_path is not None:
                for request_index, (tree, draft_tree) in enumerate(
                    zip(selected_trees, draft_trees, strict=True)
                ):
                    trace_id = f"{os.getpid()}:{trace_step}:{request_index}"
                    draft_tree.trace_id = trace_id
                    self._write_residual_tree_selection_trace(
                        trace_path,
                        trace_id=trace_id,
                        request_index=request_index,
                        request_id=(
                            request_ids[request_index]
                            if request_ids is not None
                            else None
                        ),
                        tree=tree,
                        scorer_mode=scorer_mode,
                        node_budget=node_budget,
                        max_depth=max_depth,
                        tree_policy=tree_policy,
                        batch_drafting=batch_drafting,
                    )
            return draft_trees

        configured_num_heads = int(
            getattr(residual_model, "residual_tree_total_heads", 0)
        )
        if active_heads is None:
            active_head_ids = list(range(configured_num_heads))
        elif isinstance(active_heads, torch.Tensor):
            active_head_ids = (
                [int(value) for value in active_heads.tolist()]
                if active_heads.device.type == "cpu"
                else []
            )
        else:
            active_head_ids = [int(value) for value in active_heads]
        supports_compact_candidates = getattr(
            residual_model,
            "supports_residual_greedy_candidates",
            None,
        )
        if supports_compact_candidates is None:
            supports_compact_candidates = getattr(
                residual_model,
                "supports_residual_b2d1_greedy_candidates",
                None,
            )
        if callable(supports_compact_candidates):
            supports_compact_candidates = bool(supports_compact_candidates())
        else:
            supports_compact_candidates = True
        compact_candidate_method = getattr(
            residual_model,
            "compute_residual_greedy_candidates",
            None,
        )
        if compact_candidate_method is None:
            compact_candidate_method = getattr(
                residual_model,
                "compute_residual_b2d1_greedy_candidates",
                None,
            )
        can_use_batched_greedy_trees = (
            bool(sampling_metadata.all_greedy)
            and node_budget > 0
            and configured_candidate_selection
            in {"head_top1", "distinct_head_top1"}
            and scorer_mode == "lambda_q"
            and trace_path is None
            and batch_drafting
            and configured_num_heads > 0
            and active_head_ids == list(range(configured_num_heads))
            and bool(supports_compact_candidates)
            and callable(compact_candidate_method)
        )
        if can_use_batched_greedy_trees:
            root_candidates: tuple[torch.Tensor, torch.Tensor] | None
            try:
                root_candidates = self._residual_tree_greedy_candidates_batch(
                    residual_model,
                    root_states,
                )
            except NotImplementedError:
                root_candidates = None
            if root_candidates is not None:
                with record_function_or_nullcontext(
                    "residual_tree: select_batched_greedy_trees"
                ):
                    selected_trees = select_batched_greedy_residual_trees(
                        root_states=root_states,
                        root_candidate_tokens=root_candidates[0],
                        root_candidate_probabilities=root_candidates[1],
                        candidate_batch_fn=lambda states: (
                            self._residual_tree_greedy_candidates_batch(
                                residual_model,
                                states,
                            )
                        ),
                        transition_batch_fn=lambda states, token_ids: (
                            self._residual_tree_transition_batch(
                                residual_model,
                                states,
                                token_ids,
                            )
                        ),
                        head_lambdas=head_lambdas,
                        node_budget=node_budget,
                        max_depth=max_depth,
                        active_heads=active_heads,
                        tree_policy=tree_policy,
                    )
                return [tree_to_draft_token_tree(tree) for tree in selected_trees]

        root_head_probs_by_request = None
        if node_budget > 0 and batch_drafting:
            with record_function_or_nullcontext("residual_tree: root_head_probs_batch"):
                root_head_probs_by_request = self._residual_tree_root_head_probs_batch(
                    residual_model,
                    root_states,
                )
            if (
                root_head_probs_by_request is not None
                and len(root_head_probs_by_request) != batch_size
            ):
                raise ValueError(
                    "batched residual root proposals must match the request batch"
                )

        trees: list[DraftTokenTree] = []
        proposal_batches: list[torch.Tensor] = []

        for request_index, root_state in enumerate(root_states):
            root_head_probs = (
                root_head_probs_by_request[request_index]
                if root_head_probs_by_request is not None
                else None
            )
            with record_function_or_nullcontext("residual_tree: select_request"):
                tree = select_residual_tree(
                    root_state=root_state,
                    root_head_probs=root_head_probs,
                    proposal_fn=lambda state: self._residual_tree_head_probs(
                        residual_model, state
                    ),
                    transition_fn=lambda state, token_id: (
                        self._residual_tree_transition(
                            residual_model,
                            state,
                            token_id,
                        )
                    ),
                    head_lambdas=head_lambdas,
                    node_budget=node_budget,
                    max_depth=max_depth,
                    active_heads=active_heads,
                    store_proposal_probs=True,
                    candidate_selection=configured_candidate_selection,
                    min_novel_probability_ratio=(
                        self.speculative_config.residual_tree_min_novel_probability_ratio
                    ),
                    tree_policy=tree_policy,
                    scorer_mode=scorer_mode,
                    head_prior_probabilities=head_prior_probabilities,
                    listwise_scorer=listwise_scorer,
                    scorer_top_k=scorer_top_k,
                    collect_scoring_records=trace_path is not None,
                    proposal_batch_fn=(
                        lambda states: self._residual_tree_head_probs_batch(
                            residual_model,
                            states,
                        )
                    )
                    if batch_drafting
                    else None,
                    transition_batch_fn=(
                        lambda states, token_ids: self._residual_tree_transition_batch(
                            residual_model,
                            states,
                            token_ids,
                        )
                    )
                    if batch_drafting
                    else None,
                )
            draft_tree = tree_to_draft_token_tree(tree)
            if trace_enabled:
                trace_id = f"{os.getpid()}:{trace_step}:{request_index}"
                draft_tree.trace_id = trace_id
                if trace_path is not None:
                    self._write_residual_tree_selection_trace(
                        trace_path,
                        trace_id=trace_id,
                        request_index=request_index,
                        request_id=(
                            request_ids[request_index]
                            if request_ids is not None
                            else None
                        ),
                        tree=tree,
                        scorer_mode=scorer_mode,
                        node_budget=node_budget,
                        max_depth=max_depth,
                        tree_policy=tree_policy,
                        batch_drafting=batch_drafting,
                    )
            trees.append(draft_tree)
            if tree.proposal_probs is not None:
                proposal_batches.append(tree.proposal_probs)

        if proposal_batches:
            with record_function_or_nullcontext("residual_tree: finalize"):
                self._last_draft_probs = torch.cat(proposal_batches, dim=0).contiguous()
        return trees

    def _propose_stock_top2_trees(
        self,
        model: nn.Module,
        root_states: Sequence[_ResidualTreeDraftState],
        *,
        runtime_config: dict[str, Any],
        sampling_metadata: SamplingMetadata,
    ) -> list[DraftTokenTree]:
        """Build a Stock H1 top-2 tree under any feasible binary budget."""

        node_budget = int(
            runtime_config.get("node_budget", self.num_speculative_tokens)
        )
        max_depth = runtime_config.get(
            "max_depth",
            self.speculative_config.residual_tree_max_depth,
        )
        maximum_binary_nodes = (1 << (int(max_depth) + 1)) - 2
        if node_budget < 2 or node_budget > maximum_binary_nodes:
            raise ValueError(
                "stock_top2 budget must fit the configured binary-tree depth"
            )
        if not bool(sampling_metadata.all_greedy):
            raise NotImplementedError(
                "stock_top2 tree drafting supports greedy verification only"
            )
        batch_drafting = bool(
            runtime_config.get(
                "batch_drafting",
                self.speculative_config.residual_tree_batch_drafting,
            )
        )
        if not batch_drafting:
            raise ValueError("stock_top2 requires batched tree drafting")
        trace_path = runtime_config.get(
            "trace_path",
            self.speculative_config.residual_tree_trace_path,
        )
        if (
            trace_path is not None
            or self._residual_tree_training_feature_dir
            or self._residual_tree_probability_trace_dir
        ):
            raise ValueError("stock_top2 serving does not support diagnostic traces")
        tree_policy = runtime_config.get(
            "tree_policy",
            self.speculative_config.residual_tree_tree_policy,
        )
        if tree_policy != "best_first":
            raise ValueError(
                "stock_top2 requires best_first to match the existing "
                "residual B6D2 control"
            )

        complete_fast_shape = (node_budget, max_depth) in {(2, 1), (6, 2)}
        candidate_batch_fn = (
            self._stock_top2_candidates_batch
            if complete_fast_shape
            else self._stock_top2_scored_candidates_batch
        )
        root_tokens, root_rank_scores = candidate_batch_fn(model, root_states)
        with record_function_or_nullcontext("residual_tree: select_stock_top2_trees"):
            selected_trees = select_batched_greedy_residual_trees(
                root_states=root_states,
                root_candidate_tokens=root_tokens,
                root_candidate_probabilities=root_rank_scores,
                candidate_batch_fn=lambda states: candidate_batch_fn(model, states),
                transition_batch_fn=lambda states, token_ids: (
                    self._residual_tree_transition_batch(
                        model,
                        states,
                        token_ids,
                    )
                ),
                head_lambdas=(1.0, 1.0),
                node_budget=node_budget,
                max_depth=max_depth,
                active_heads=None,
                tree_policy=tree_policy,
            )
        return [tree_to_draft_token_tree(tree) for tree in selected_trees]

    def _propose_stock_dynamic_trees(
        self,
        model: nn.Module,
        root_states: Sequence[_ResidualTreeDraftState],
        *,
        runtime_config: dict[str, Any],
        sampling_metadata: SamplingMetadata,
    ) -> list[DraftTokenTree]:
        """Build a Stock H1 width-10, at-most-60-node, depth-8 tree."""

        node_budget = int(
            runtime_config.get("node_budget", self.num_speculative_tokens)
        )
        max_depth = int(
            runtime_config.get(
                "max_depth",
                self.speculative_config.residual_tree_max_depth,
            )
        )
        tree_policy = runtime_config.get(
            "tree_policy",
            self.speculative_config.residual_tree_tree_policy,
        )
        batch_drafting = bool(
            runtime_config.get(
                "batch_drafting",
                self.speculative_config.residual_tree_batch_drafting,
            )
        )
        trace_path = runtime_config.get(
            "trace_path",
            self.speculative_config.residual_tree_trace_path,
        )
        if not 1 <= node_budget <= 60 or max_depth != 8:
            raise ValueError(
                "stock dynamic EAGLE-3 requires D8 with a node budget between 1 and 60"
            )
        if tree_policy != "eagle3_dynamic":
            raise ValueError("stock dynamic EAGLE-3 requires eagle3_dynamic policy")
        if not bool(sampling_metadata.all_greedy):
            raise NotImplementedError(
                "stock dynamic EAGLE-3 supports greedy verification only"
            )
        if not batch_drafting:
            raise ValueError("stock dynamic EAGLE-3 requires batched drafting")
        if (
            trace_path is not None
            or self._residual_tree_training_feature_dir
            or self._residual_tree_probability_trace_dir
        ):
            raise ValueError(
                "stock dynamic EAGLE-3 serving does not support diagnostic traces"
            )

        candidate_selection = getattr(
            self.speculative_config,
            "residual_tree_candidate_selection",
            "stock_top10_dynamic",
        )
        frontier_width = 9 if candidate_selection == "stock_top9_dynamic" else 10
        candidate_batch_fn = lambda selected_states: (
            self._stock_top10_dynamic_candidates_batch(
                model,
                selected_states,
                top_k=frontier_width,
            )
        )
        root_tokens, root_probabilities = candidate_batch_fn(root_states)
        with record_function_or_nullcontext(
            "residual_tree: select_stock_eagle3_dynamic_trees"
        ):
            selected_trees = select_batched_eagle3_dynamic_trees(
                root_states=root_states,
                root_candidate_tokens=root_tokens,
                root_candidate_probabilities=root_probabilities,
                candidate_batch_fn=candidate_batch_fn,
                transition_batch_fn=lambda states, token_ids: (
                    self._residual_tree_transition_batch(
                        model,
                        states,
                        token_ids,
                    )
                ),
                head_lambdas=(1.0,) * frontier_width,
                node_budget=node_budget,
                max_depth=max_depth,
                frontier_width=frontier_width,
            )
        return [tree_to_draft_token_tree(tree) for tree in selected_trees]

    def _propose_hybrid_dynamic_trees(
        self,
        model: nn.Module,
        root_states: Sequence[_ResidualTreeDraftState],
        *,
        runtime_config: dict[str, Any],
        sampling_metadata: SamplingMetadata,
        request_ids: Sequence[str] | None = None,
    ) -> list[DraftTokenTree]:
        """Build a width-10, at-most-60-node D8 H1/H2 hybrid tree."""

        node_budget = int(
            runtime_config.get("node_budget", self.num_speculative_tokens)
        )
        max_depth = int(
            runtime_config.get(
                "max_depth",
                self.speculative_config.residual_tree_max_depth,
            )
        )
        tree_policy = runtime_config.get(
            "tree_policy",
            self.speculative_config.residual_tree_tree_policy,
        )
        scorer_mode = runtime_config.get(
            "scorer_mode",
            self.speculative_config.residual_tree_scorer_mode,
        )
        batch_drafting = bool(
            runtime_config.get(
                "batch_drafting",
                self.speculative_config.residual_tree_batch_drafting,
            )
        )
        trace_path = runtime_config.get(
            "trace_path",
            self.speculative_config.residual_tree_trace_path,
        )
        if not 1 <= node_budget <= 60 or max_depth != 8:
            raise ValueError(
                "hybrid dynamic EAGLE-3 requires D8 with a node budget between 1 and 60"
            )
        if tree_policy != "eagle3_dynamic":
            raise ValueError("hybrid dynamic EAGLE-3 requires eagle3_dynamic policy")
        if scorer_mode != "lambda_q":
            raise ValueError("hybrid dynamic EAGLE-3 requires lambda_q scoring")
        if not bool(sampling_metadata.all_greedy):
            raise NotImplementedError(
                "hybrid dynamic EAGLE-3 supports greedy verification only"
            )
        if not batch_drafting:
            raise ValueError("hybrid dynamic EAGLE-3 requires batched drafting")
        if (
            self._residual_tree_training_feature_dir
            or self._residual_tree_probability_trace_dir
        ):
            raise ValueError(
                "hybrid dynamic EAGLE-3 serving does not support training-feature "
                "or probability traces"
            )

        configured_lambdas = self.speculative_config.residual_tree_head_lambdas
        if configured_lambdas is None:
            configured_lambdas = getattr(model, "residual_tree_head_lambdas", None)
        if configured_lambdas is None or len(configured_lambdas) != 2:
            raise ValueError(
                "hybrid dynamic EAGLE-3 requires H1 and H2 accepted-mass weights"
            )
        h1_lambda, h2_lambda = (float(value) for value in configured_lambdas)
        if (
            not math.isfinite(h1_lambda)
            or not math.isfinite(h2_lambda)
            or h1_lambda <= 0.0
            or h2_lambda <= 0.0
            or h2_lambda > h1_lambda
        ):
            raise ValueError(
                "hybrid dynamic EAGLE-3 requires 0 < H2 weight <= H1 weight"
            )
        # Preserve Stock's H1 path scores exactly.  Only H2 is discounted by
        # its accepted-mass weight relative to H1.
        candidate_selection = getattr(
            self.speculative_config,
            "residual_tree_candidate_selection",
            "hybrid_top9_h2_dynamic",
        )
        try:
            h1_candidate_count, h2_candidate_count, _ = _HYBRID_DYNAMIC_LAYOUTS[
                candidate_selection
            ]
        except KeyError as error:
            raise ValueError(
                f"unsupported hybrid dynamic layout: {candidate_selection}"
            ) from error
        h2_to_h1_weight = h2_lambda / h1_lambda
        is_union_merge = candidate_selection == "hybrid_union_top10_dynamic"
        try:
            frontier_h2_only_quota = int(
                os.environ.get("VLLM_RESIDUAL_TREE_FRONTIER_H2_ONLY_QUOTA", "0")
            )
            preserve_spine_count = int(
                os.environ.get("VLLM_RESIDUAL_TREE_PRESERVE_SPINE_COUNT", "0")
            )
        except ValueError as error:
            raise ValueError(
                "dynamic-tree diagnostic reservations must be integers"
            ) from error
        if frontier_h2_only_quota < 0 or preserve_spine_count < 0:
            raise ValueError(
                "dynamic-tree diagnostic reservations must be non-negative"
            )
        if frontier_h2_only_quota and (not is_union_merge or trace_path is None):
            raise ValueError(
                "H2-only frontier reservation requires traced hybrid union candidates"
            )
        if is_union_merge:
            # The model hook merges duplicate tokens and applies the H2 weight
            # before selecting ten unique candidates.
            hybrid_lambdas = (1.0,) * 10
        else:
            hybrid_lambdas = (1.0,) * h1_candidate_count + (
                h2_to_h1_weight,
            ) * h2_candidate_count
        rank_overlap_by_request: dict[
            int,
            dict[int, dict[int, dict[int, int]]],
        ] = {}
        canonical_replay = sampling_metadata.canonical_token_replay
        tree_oracle_enabled = (
            canonical_replay is not None
            and canonical_replay.tree_construction_oracle_batch()
        )
        if tree_oracle_enabled and not is_union_merge:
            raise ValueError(
                "tree-construction oracle requires hybrid_union_top10_dynamic"
            )

        def tree_oracle_targets(
            states: Sequence[_ResidualTreeDraftState],
        ) -> torch.Tensor | None:
            if not tree_oracle_enabled:
                return None
            assert canonical_replay is not None
            target_ids: list[int] = []
            for state in states:
                payload = state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "tree-construction oracle requires internal EAGLE states"
                    )
                req_index = payload.req_index
                config = canonical_replay.configs[req_index]
                if config is None:
                    raise ValueError(
                        "tree-construction oracle is missing a request reference"
                    )
                root_offset = len(canonical_replay.output_token_ids[req_index]) + 1
                path = payload.path_token_ids
                reference_path = config.token_ids[root_offset : root_offset + len(path)]
                target_offset = root_offset + len(path)
                if tuple(path) != tuple(reference_path) or target_offset >= len(
                    config.token_ids
                ):
                    target_ids.append(-1)
                else:
                    target_ids.append(config.token_at(target_offset))
            return torch.tensor(
                target_ids,
                dtype=torch.long,
                device=states[0].proposal_hidden.device,
            )

        def candidate_batch_fn(
            states: Sequence[_ResidualTreeDraftState],
        ) -> Any:
            output = self._hybrid_dynamic_candidates_batch(
                model,
                states,
                candidate_selection=candidate_selection,
                collect_h1_rank_overlap=(trace_path is not None and not is_union_merge),
                collect_union_provenance=(trace_path is not None and is_union_merge),
                h2_to_h1_weight=(h2_to_h1_weight if is_union_merge else None),
                oracle_target_tokens=(
                    tree_oracle_targets(states) if is_union_merge else None
                ),
            )
            if trace_path is None:
                return cast(tuple[torch.Tensor, torch.Tensor], output)
            if is_union_merge:
                if len(output) != 6:
                    raise AssertionError(
                        "hybrid union trace requires source/rank provenance"
                    )
                return output
            if len(output) != 3:
                raise AssertionError(
                    "hybrid trace collection requires H1-rank overlap metadata"
                )
            tokens, probabilities, rank_overlap = output
            rank_rows = rank_overlap.to(device="cpu").tolist()
            for state, rank_row in zip(states, rank_rows, strict=True):
                payload = state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "hybrid rank-overlap trace requires internal EAGLE state"
                    )
                child_depth = len(payload.path_token_ids) + 1
                depth_counts = rank_overlap_by_request.setdefault(
                    payload.req_index,
                    {},
                ).setdefault(child_depth, {})
                for h2_rank, h1_rank_value in enumerate(rank_row, start=1):
                    h1_rank = int(h1_rank_value)
                    rank_counts = depth_counts.setdefault(h2_rank, {})
                    rank_counts[h1_rank] = rank_counts.get(h1_rank, 0) + 1
            return tokens, probabilities

        root_output = candidate_batch_fn(root_states)
        root_tokens, root_probabilities = root_output[:2]
        root_candidate_provenance = (
            cast(
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                root_output[2:],
            )
            if is_union_merge and trace_path is not None
            else None
        )
        diagnostic_target_paths = None
        diagnostic_target_metadata = None
        if (
            trace_path is not None
            and canonical_replay is not None
            and canonical_replay.proposal_only_batch()
        ):
            diagnostic_target_paths = []
            diagnostic_target_metadata = []
            for root_state in root_states:
                payload = root_state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "canonical target-path diagnostics require internal "
                        "EAGLE states"
                    )
                req_index = payload.req_index
                config = canonical_replay.configs[req_index]
                if config is None:
                    raise ValueError(
                        "canonical target-path diagnostics are missing a "
                        "request reference"
                    )
                root_offset = len(canonical_replay.output_token_ids[req_index]) + 1
                diagnostic_target_paths.append(
                    config.token_ids[root_offset : root_offset + max_depth]
                )
                diagnostic_target_metadata.append(
                    {
                        "replay_id": config.replay_id,
                        "output_offset": root_offset,
                    }
                )
        with record_function_or_nullcontext(
            "residual_tree: select_hybrid_eagle3_dynamic_trees"
        ):
            selected_trees = select_batched_eagle3_dynamic_trees(
                root_states=root_states,
                root_candidate_tokens=root_tokens,
                root_candidate_probabilities=root_probabilities,
                candidate_batch_fn=candidate_batch_fn,
                transition_batch_fn=lambda states, token_ids: (
                    self._residual_tree_transition_batch(
                        model,
                        states,
                        token_ids,
                    )
                ),
                head_lambdas=hybrid_lambdas,
                node_budget=node_budget,
                max_depth=max_depth,
                frontier_width=10,
                collect_dynamic_provenance=trace_path is not None,
                root_candidate_provenance=root_candidate_provenance,
                diagnostic_target_paths=diagnostic_target_paths,
                frontier_h2_only_quota=frontier_h2_only_quota,
                preserve_spine_count=preserve_spine_count,
            )
        if diagnostic_target_metadata is not None:
            for tree, metadata in zip(
                selected_trees,
                diagnostic_target_metadata,
                strict=True,
            ):
                if tree.dynamic_provenance is None:
                    raise AssertionError(
                        "canonical target-path provenance is unavailable"
                    )
                tree.dynamic_provenance["canonical_target_path"].update(metadata)
        if trace_path is not None and not is_union_merge:
            for root_state, tree in zip(root_states, selected_trees, strict=True):
                payload = root_state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "hybrid rank-overlap trace requires internal EAGLE state"
                    )
                if tree.dynamic_provenance is None:
                    raise AssertionError("hybrid trace provenance is unavailable")
                request_counts = rank_overlap_by_request.get(payload.req_index, {})
                tree.dynamic_provenance["h2_h1_unselected_top10_overlap_by_depth"] = [
                    {
                        "depth": depth,
                        "h2_rank_counts": [
                            {
                                "h2_rank": h2_rank,
                                "total": sum(rank_counts.values()),
                                "outside_h1_top10": rank_counts.get(0, 0),
                                "h1_rank_counts": [
                                    {"h1_rank": h1_rank, "count": count}
                                    for h1_rank, count in sorted(rank_counts.items())
                                    if h1_rank > 0
                                ],
                            }
                            for h2_rank, rank_counts in sorted(h2_rank_counts.items())
                        ],
                    }
                    for depth, h2_rank_counts in sorted(request_counts.items())
                ]
        draft_trees = [tree_to_draft_token_tree(tree) for tree in selected_trees]
        if trace_path is not None:
            trace_step = self._residual_tree_trace_step
            self._residual_tree_trace_step += 1
            for request_index, (tree, draft_tree) in enumerate(
                zip(selected_trees, draft_trees, strict=True)
            ):
                trace_id = f"{os.getpid()}:{trace_step}:{request_index}"
                draft_tree.trace_id = trace_id
                self._write_residual_tree_selection_trace(
                    trace_path,
                    trace_id=trace_id,
                    request_index=request_index,
                    request_id=(
                        request_ids[request_index] if request_ids is not None else None
                    ),
                    tree=tree,
                    scorer_mode=scorer_mode,
                    node_budget=node_budget,
                    max_depth=max_depth,
                    tree_policy=tree_policy,
                    batch_drafting=batch_drafting,
                )
        return draft_trees

    def _write_residual_tree_selection_trace(
        self,
        path: str,
        *,
        trace_id: str,
        request_index: int,
        tree: Any,
        scorer_mode: str,
        request_id: str | None = None,
        node_budget: int | None = None,
        max_depth: int | None = None,
        tree_policy: str | None = None,
        batch_drafting: bool | None = None,
    ) -> None:
        candidate_states = []
        for record in tree.scoring_records:
            features = record.features
            class_probabilities = record.class_probabilities
            candidates = []
            for index, token_id in enumerate(features.candidate_token_ids):
                entered_node_id = record.entered_node_ids[index]
                candidates.append(
                    {
                        "candidate_index": index,
                        "head_id": features.head_ids[index],
                        "token_id": token_id,
                        "conditioned_proposal_probability": (
                            features.conditioned_proposal_probabilities[index]
                        ),
                        "base_probability": features.base_probabilities[index],
                        "logit_margin": features.logit_margins[index],
                        "entropy": features.entropies[index],
                        "topk_mass": features.topk_masses[index],
                        "top_k": features.top_k,
                        "candidate_priority": record.candidate_priorities[index],
                        "scorer_probability": (
                            class_probabilities[index]
                            if class_probabilities is not None
                            else None
                        ),
                        "entered_tree": entered_node_id is not None,
                        "entered_node_id": entered_node_id,
                    }
                )
            candidate_states.append(
                {
                    "parent_node_id": record.parent_id,
                    "child_depth": features.depth,
                    "parent_path_priority": record.parent_priority,
                    "candidates": candidates,
                    "none_probability": (
                        class_probabilities[-1]
                        if class_probabilities is not None
                        else None
                    ),
                }
            )

        append_residual_tree_trace(
            path,
            {
                "event": "selection",
                "trace_id": trace_id,
                "request_index": request_index,
                "request_id": request_id,
                "scorer_mode": scorer_mode,
                "tree_policy": (
                    tree_policy
                    if tree_policy is not None
                    else getattr(
                        self.speculative_config,
                        "residual_tree_tree_policy",
                        "best_first",
                    )
                ),
                "batch_drafting": (
                    batch_drafting
                    if batch_drafting is not None
                    else getattr(
                        self.speculative_config,
                        "residual_tree_batch_drafting",
                        True,
                    )
                ),
                "candidate_selection": (
                    self.speculative_config.residual_tree_candidate_selection
                ),
                "h2_candidate_columns": (
                    list(
                        range(
                            _HYBRID_DYNAMIC_LAYOUTS[
                                self.speculative_config.residual_tree_candidate_selection
                            ][0],
                            10,
                        )
                    )
                    if self.speculative_config.residual_tree_candidate_selection
                    in _HYBRID_DYNAMIC_LAYOUTS
                    else []
                ),
                "node_budget": (
                    int(node_budget)
                    if node_budget is not None
                    else int(self.num_speculative_tokens)
                ),
                "max_depth": (
                    max_depth
                    if max_depth is not None
                    else self.speculative_config.residual_tree_max_depth
                ),
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "parent_node_id": node.parent_id,
                        "token_id": node.token_id,
                        "depth": node.depth,
                        "path_priority": node.priority,
                        "contributors": [
                            {
                                "head_id": contributor.head_id,
                                "proposal_row": contributor.proposal_row,
                                "proposal_probability": (contributor.proposal_prob),
                                "lambda_weight": contributor.lambda_weight,
                                "score": contributor.score,
                            }
                            for contributor in node.contributors
                        ],
                    }
                    for node in tree.nodes
                ],
                "candidate_states": candidate_states,
                "dynamic_provenance": getattr(
                    tree,
                    "dynamic_provenance",
                    None,
                ),
            },
        )

    def _load_residual_tree_greedy_scorer(self) -> Any:
        path = getattr(self, "_residual_tree_runtime_config", {}).get(
            "greedy_scorer_path",
            getattr(
                self.speculative_config,
                "residual_tree_greedy_scorer_path",
                None,
            ),
        )
        if not path:
            raise ValueError(
                "greedy_listwise residual-tree scoring requires a scorer path"
            )
        normalized_path = str(path)
        if (
            self._residual_tree_greedy_scorer is not None
            and self._residual_tree_greedy_scorer_path == normalized_path
        ):
            return self._residual_tree_greedy_scorer
        try:
            from residual_stack.greedy_tree_scorer import load_scorer
        except ImportError as exc:
            raise ImportError(
                "greedy_listwise residual-tree scoring requires the "
                "residual_stack.greedy_tree_scorer runtime"
            ) from exc
        scorer = load_scorer(normalized_path)
        if scorer.to_dict().get("model_type") != "listwise_linear_softmax":
            raise ValueError(
                "greedy_listwise serving requires a listwise_linear_softmax scorer"
            )
        self._residual_tree_greedy_scorer = scorer
        self._residual_tree_greedy_scorer_path = normalized_path
        return scorer

    def set_residual_tree_runtime_config(
        self,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Set serving-only tree controls without reloading model weights."""

        if not self.speculative_config.residual_tree:
            raise ValueError("runtime tree controls require residual-tree mode")
        if config is None:
            self._residual_tree_runtime_config = {}
            return {}
        if not isinstance(config, dict):
            raise TypeError("residual-tree runtime config must be a dictionary")

        allowed = {
            "node_budget",
            "max_depth",
            "tree_policy",
            "batch_drafting",
            "scorer_mode",
            "head_prior_probabilities",
            "greedy_scorer_path",
            "scorer_top_k",
            "trace_path",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"unknown residual-tree runtime controls: {sorted(unknown)}"
            )

        normalized = dict(config)
        node_budget = int(
            normalized.get(
                "node_budget",
                self.speculative_config.num_speculative_tokens,
            )
        )
        maximum = int(self.speculative_config.num_speculative_tokens)
        if not 1 <= node_budget <= maximum:
            raise ValueError(
                f"residual-tree runtime node budget must be in [1, {maximum}]"
            )
        normalized["node_budget"] = node_budget

        max_depth = normalized.get(
            "max_depth",
            self.speculative_config.residual_tree_max_depth,
        )
        configured_max_depth = self.speculative_config.residual_tree_max_depth
        if max_depth is not None:
            max_depth = int(max_depth)
            if not 1 <= max_depth <= node_budget:
                raise ValueError(
                    "residual-tree runtime max depth must be in [1, node_budget]"
                )
        if configured_max_depth is not None and (
            max_depth is None or max_depth > configured_max_depth
        ):
            raise ValueError(
                "residual-tree runtime max depth cannot exceed the engine "
                f"maximum of {configured_max_depth}"
            )
        normalized["max_depth"] = max_depth

        tree_policy = normalized.get(
            "tree_policy",
            self.speculative_config.residual_tree_tree_policy,
        )
        if tree_policy not in {
            "best_first",
            "breadth_first",
            "eagle3_dynamic",
        }:
            raise ValueError(f"unknown residual-tree runtime policy: {tree_policy}")
        normalized["tree_policy"] = tree_policy

        if "batch_drafting" in normalized and not isinstance(
            normalized["batch_drafting"],
            bool,
        ):
            raise TypeError("runtime batch_drafting must be a boolean")

        scorer_mode = normalized.get(
            "scorer_mode",
            self.speculative_config.residual_tree_scorer_mode,
        )
        if scorer_mode not in {
            "lambda_q",
            "uniform",
            "head_prior",
            "greedy_listwise",
        }:
            raise ValueError(f"unknown residual-tree runtime scorer: {scorer_mode}")
        normalized["scorer_mode"] = scorer_mode

        scorer_top_k = int(
            normalized.get(
                "scorer_top_k",
                self.speculative_config.residual_tree_scorer_top_k,
            )
        )
        if scorer_top_k <= 0:
            raise ValueError("runtime scorer_top_k must be positive")
        normalized["scorer_top_k"] = scorer_top_k

        if scorer_mode == "head_prior":
            probabilities = normalized.get("head_prior_probabilities")
            if not isinstance(probabilities, (list, tuple)) or not probabilities:
                raise ValueError("head_prior runtime scorer requires probabilities")
            probabilities = [float(value) for value in probabilities]
            if any(value < 0.0 for value in probabilities) or sum(probabilities) <= 0.0:
                raise ValueError(
                    "runtime head-prior probabilities must be nonnegative "
                    "with positive mass"
                )
            normalized["head_prior_probabilities"] = probabilities
        elif normalized.get("head_prior_probabilities") is not None:
            raise ValueError(
                "runtime head-prior probabilities require head_prior scorer"
            )

        if scorer_mode == "greedy_listwise":
            scorer_path = normalized.get("greedy_scorer_path")
            if not isinstance(scorer_path, str) or not scorer_path:
                raise ValueError(
                    "greedy_listwise runtime scorer requires a scorer path"
                )
        elif normalized.get("greedy_scorer_path") is not None:
            raise ValueError(
                "runtime greedy scorer path requires greedy_listwise scorer"
            )

        trace_path = normalized.get("trace_path")
        if trace_path is not None and not isinstance(trace_path, str):
            raise TypeError("runtime trace path must be a string or None")

        self._residual_tree_runtime_config = normalized
        return dict(normalized)

    def _score_residual_tree_candidates(
        self,
        features: TreeListwiseFeatures,
    ) -> list[float]:
        from residual_stack.greedy_tree_scorer import ServingFeatureBatch

        scorer = self._load_residual_tree_greedy_scorer()
        batch = ServingFeatureBatch(
            candidate_token_ids=np.asarray(
                [features.candidate_token_ids],
                dtype=np.int64,
            ),
            conditioned_proposal_probabilities=np.asarray(
                [features.conditioned_proposal_probabilities],
                dtype=np.float64,
            ),
            base_probabilities=np.asarray(
                [features.base_probabilities],
                dtype=np.float64,
            ),
            logit_margin=np.asarray(
                [features.logit_margins],
                dtype=np.float64,
            ),
            entropy=np.asarray(
                [features.entropies],
                dtype=np.float64,
            ),
            topk_mass=np.asarray(
                [features.topk_masses],
                dtype=np.float64,
            ),
            depth=np.asarray([features.depth], dtype=np.int64),
            head_ids=tuple(head_id + 1 for head_id in features.head_ids),
            top_k=features.top_k,
        )
        probabilities = scorer.predict_proba(batch)
        if probabilities.shape != (1, len(features.head_ids) + 1):
            raise ValueError("greedy tree scorer returned an invalid shape")
        return probabilities[0].tolist()

    def _unwrap_residual_tree_model(self) -> nn.Module:
        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        return cast(nn.Module, model)

    def _validate_residual_tree_model_api(self, model: nn.Module) -> None:
        speculative_config = getattr(self, "speculative_config", None)
        candidate_selection = getattr(
            speculative_config,
            "residual_tree_candidate_selection",
            None,
        )
        if candidate_selection in {
            "stock_top2",
            "stock_top9_dynamic",
            "stock_top10_dynamic",
        }:
            hook_name = (
                "compute_stock_top2_greedy_tokens"
                if candidate_selection == "stock_top2"
                else "compute_stock_top10_dynamic_candidates"
            )
            if not callable(getattr(model, hook_name, None)):
                proposal_name = (
                    "Stock EAGLE H1 top-2 hook"
                    if candidate_selection == "stock_top2"
                    else "Stock EAGLE H1 top-10 dynamic hook"
                )
                raise NotImplementedError(
                    f"{candidate_selection} requires the {proposal_name}"
                )
            has_transition_api = self._has_model_residual_tree_transition(model)
            has_transition_api = (
                has_transition_api or self._can_use_internal_eagle_tree_transition()
            )
            if not has_transition_api:
                raise NotImplementedError(
                    "stock_top2 requires the stock EAGLE tree transition"
                )
            return
        required_selection = getattr(
            model, "residual_tree_required_candidate_selection", None
        )
        if required_selection is not None:
            actual_selection = getattr(
                self.speculative_config,
                "residual_tree_candidate_selection",
                None,
            )
            if actual_selection != required_selection:
                raise ValueError(
                    "residual-head checkpoint requires candidate selection "
                    f"{required_selection!r}, but serving configured "
                    f"{actual_selection!r}"
                )
        if candidate_selection in _HYBRID_DYNAMIC_LAYOUTS:
            h1_count, h2_count, hook_name = _HYBRID_DYNAMIC_LAYOUTS[candidate_selection]
            if not callable(getattr(model, hook_name, None)):
                raise NotImplementedError(
                    f"{candidate_selection} requires the H1 top{h1_count} plus "
                    f"H2 top{h2_count} hook"
                )
        has_configured_heads = True
        if hasattr(model, "has_residual_tree_heads"):
            has_configured_heads = bool(cast(Any, model).has_residual_tree_heads())
        has_head_api = (
            any(
                hasattr(model, name)
                for name in (
                    "compute_residual_head_probs",
                    "compute_residual_head_logits",
                    "compute_residual_tree_head_logits",
                )
            )
            and has_configured_heads
        )
        has_transition_api = self._has_model_residual_tree_transition(model)
        has_transition_api = (
            has_transition_api or self._can_use_internal_eagle_tree_transition()
        )
        if not has_head_api or not has_transition_api:
            raise NotImplementedError(
                "residual_tree drafting requires the EAGLE draft model to expose "
                "compute_residual_head_probs/logits and either a model-level "
                "residual_tree_transition/residual_tree_step or stock EAGLE "
                "KV-cache state needed for internal branch transitions. Refusing "
                "to fall back to linear drafting."
            )

    def _can_use_internal_eagle_tree_transition(self) -> bool:
        if not self.pass_hidden_states_to_model:
            return False
        if self.parallel_drafting:
            return False
        if self.uses_mrope or self.uses_xdrope_dim > 0:
            return False
        if self.vllm_config.parallel_config.decode_context_parallel_size > 1:
            return False
        return bool(self.draft_attn_groups and self._draft_attn_layer_names)

    @staticmethod
    def _has_model_residual_tree_transition(model: nn.Module) -> bool:
        return any(
            hasattr(model, name)
            for name in ("residual_tree_transition", "residual_tree_step")
        )

    def _init_residual_tree_state(
        self,
        model: nn.Module,
        state: _ResidualTreeDraftState,
    ) -> _ResidualTreeDraftState:
        init_state = getattr(model, "init_residual_tree_state", None)
        if init_state is None:
            return state
        output = self._call_with_supported_kwargs(
            init_state,
            state=state,
            proposal_hidden=state.proposal_hidden.unsqueeze(0),
            transition_hidden=state.transition_hidden.unsqueeze(0),
            position=state.position,
        )
        return self._coerce_residual_tree_state(output, fallback=state)

    def _get_residual_tree_head_lambdas(
        self,
        model: nn.Module,
        root_state: _ResidualTreeDraftState,
    ) -> torch.Tensor:
        configured = self.speculative_config.residual_tree_head_lambdas
        if configured is None:
            if hasattr(model, "get_residual_tree_head_lambdas"):
                configured = cast(Any, model).get_residual_tree_head_lambdas()
            elif hasattr(model, "residual_tree_head_lambdas"):
                configured = cast(Any, model).residual_tree_head_lambdas
        if configured is None:
            raise ValueError(
                "residual_tree_head_lambdas must be supplied in the "
                "speculative config or by the draft model. Lambdas should be "
                "estimated from clipped residual coverable mass."
            )

        lambdas = torch.as_tensor(
            configured,
            dtype=torch.float32,
            device=root_state.proposal_hidden.device,
        )
        if lambdas.ndim != 1:
            raise ValueError("residual_tree_head_lambdas must be 1-D")
        return lambdas

    def _residual_tree_head_probs(
        self,
        model: nn.Module,
        state: _ResidualTreeDraftState,
    ) -> list[torch.Tensor]:
        hidden = state.proposal_hidden.unsqueeze(0)
        with (
            record_function_or_nullcontext("residual_tree: head_probs"),
            torch.inference_mode(),
        ):
            if hasattr(model, "compute_residual_head_probs"):
                probs = self._call_with_supported_kwargs(
                    cast(Any, model).compute_residual_head_probs,
                    state=state,
                    hidden_states=hidden,
                    hidden=hidden,
                )
            elif hasattr(model, "compute_residual_head_logits"):
                logits = self._call_with_supported_kwargs(
                    cast(Any, model).compute_residual_head_logits,
                    state=state,
                    hidden_states=hidden,
                    hidden=hidden,
                )
                probs = torch.softmax(logits.float(), dim=-1)
            else:
                logits = self._call_with_supported_kwargs(
                    cast(Any, model).compute_residual_tree_head_logits,
                    state=state,
                    hidden_states=hidden,
                    hidden=hidden,
                )
                probs = torch.softmax(logits.float(), dim=-1)

        if isinstance(probs, (list, tuple)):
            rows: list[torch.Tensor] = []
            for item in probs:
                row = torch.as_tensor(item, device=hidden.device)
                if row.ndim == 2 and row.shape[0] == 1:
                    row = row[0]
                if row.ndim != 1:
                    raise ValueError("each residual head probability row must be 1-D")
                rows.append(row)
            return rows
        if probs.ndim == 3:
            if probs.shape[0] != 1:
                raise ValueError(
                    "residual head probs must have batch size 1 for one tree state"
                )
            probs = probs[0]
        if probs.ndim != 2:
            raise ValueError(
                "residual head probs/logits must have shape [num_heads, vocab] "
                "or [1, num_heads, vocab]"
            )
        return [probs[head_id] for head_id in range(probs.shape[0])]

    def _residual_tree_root_head_probs_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
    ) -> torch.Tensor | list[list[torch.Tensor]] | None:
        """Batch request roots only for a provably hidden-only head hook.

        A scalar ``state`` hook may depend on request-local payload, so passing
        a list of unrelated request states would silently change its contract.
        Such models retain the original per-request proposal path.
        """

        if hasattr(model, "compute_residual_head_probs"):
            head_method = cast(Any, model).compute_residual_head_probs
        elif hasattr(model, "compute_residual_head_logits"):
            head_method = cast(Any, model).compute_residual_head_logits
        else:
            head_method = cast(Any, model).compute_residual_tree_head_logits
        try:
            head_parameters = inspect.signature(head_method).parameters
        except (TypeError, ValueError):
            return None
        if "state" in head_parameters or "states" in head_parameters:
            return None
        if "hidden_states" not in head_parameters and "hidden" not in head_parameters:
            return None
        return self._residual_tree_head_probs_batch(
            model,
            states,
            include_state_context=False,
        )

    def _residual_tree_head_probs_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
        *,
        include_state_context: bool = True,
    ) -> torch.Tensor | list[list[torch.Tensor]]:
        """Project all residual heads for one breadth level in one call."""

        if not states:
            return []
        if hasattr(model, "compute_residual_head_probs"):
            head_method = cast(Any, model).compute_residual_head_probs
        elif hasattr(model, "compute_residual_head_logits"):
            head_method = cast(Any, model).compute_residual_head_logits
        else:
            head_method = cast(Any, model).compute_residual_tree_head_logits
        try:
            head_signature = inspect.signature(head_method)
        except (TypeError, ValueError):
            head_signature = None
        if head_signature is not None:
            head_parameters = head_signature.parameters
            accepts_hidden_batch = (
                "hidden_states" in head_parameters
                or "hidden" in head_parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in head_parameters.values()
                )
            )
            if not accepts_hidden_batch:
                return [
                    self._residual_tree_head_probs(model, state) for state in states
                ]

        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        head_kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "hidden": hidden,
        }
        if include_state_context:
            head_kwargs.update(states=states, state=states)
        with (
            record_function_or_nullcontext("residual_tree: head_probs_batch"),
            torch.inference_mode(),
        ):
            if hasattr(model, "compute_residual_head_probs"):
                probs = self._call_with_supported_kwargs(
                    cast(Any, model).compute_residual_head_probs,
                    **head_kwargs,
                )
            elif hasattr(model, "compute_residual_head_logits"):
                logits = self._call_with_supported_kwargs(
                    cast(Any, model).compute_residual_head_logits,
                    **head_kwargs,
                )
                probs = torch.softmax(logits.float(), dim=-1)
            else:
                logits = self._call_with_supported_kwargs(
                    cast(Any, model).compute_residual_tree_head_logits,
                    **head_kwargs,
                )
                probs = torch.softmax(logits.float(), dim=-1)

        batch_size = len(states)
        if isinstance(probs, (list, tuple)):
            head_rows = [torch.as_tensor(item, device=hidden.device) for item in probs]
            if all(row.ndim == 2 and row.shape[0] == batch_size for row in head_rows):
                return [
                    [row[batch_index] for row in head_rows]
                    for batch_index in range(batch_size)
                ]
            if batch_size == 1 and all(row.ndim == 1 for row in head_rows):
                return [head_rows]
            raise ValueError(
                "batched residual head probability lists must be head-major "
                "with shape [batch, vocab]"
            )

        if probs.ndim == 2 and batch_size == 1:
            probs = probs.unsqueeze(0)
        if probs.ndim != 3 or probs.shape[0] != batch_size:
            raise ValueError(
                "batched residual head probs/logits must have shape "
                "[batch, num_heads, vocab]"
            )
        return probs

    def _residual_tree_greedy_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project compact ordered-head candidates for arbitrary tree states."""

        if not states:
            raise ValueError("at least one residual-tree state is required")
        method = getattr(model, "compute_residual_greedy_candidates", None)
        if method is None:
            method = getattr(
                model,
                "compute_residual_b2d1_greedy_candidates",
                None,
            )
        if not callable(method):
            raise NotImplementedError(
                "model does not expose compact residual greedy candidates"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        with (
            record_function_or_nullcontext("residual_tree: compact_candidates_batch"),
            torch.inference_mode(),
        ):
            output = self._call_with_supported_kwargs(
                cast(Any, method),
                hidden_states=hidden,
                hidden=hidden,
            )
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            raise TypeError(
                "compact residual candidate hook must return token/prob tensors"
            )
        tokens, probabilities = output
        expected_num_heads = int(
            getattr(model, "residual_tree_total_heads", tokens.shape[-1])
        )
        expected_shape = (len(states), expected_num_heads)
        if tokens.shape != expected_shape or probabilities.shape != expected_shape:
            raise ValueError(
                f"compact residual candidates must have shape {expected_shape}"
            )
        return tokens, probabilities

    def _stock_top2_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project Stock H1 once and return its two ranked tokens per state."""

        if not states:
            raise ValueError("at least one stock-tree state is required")
        method = getattr(model, "compute_stock_top2_greedy_tokens", None)
        if not callable(method):
            raise NotImplementedError(
                "model does not expose Stock H1 top-2 token selection"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        with (
            record_function_or_nullcontext("residual_tree: stock_top2_batch"),
            torch.inference_mode(),
        ):
            tokens = self._call_with_supported_kwargs(
                cast(Any, method),
                hidden_states=hidden,
                hidden=hidden,
            )
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("Stock H1 top-2 hook must return a token tensor")
        expected_shape = (len(states), 2)
        if tokens.shape != expected_shape:
            raise ValueError(f"Stock H1 top-2 tokens must have shape {expected_shape}")
        # Complete B2D1/B6D2 trees select every level candidate, so only a
        # deterministic rank ordering is needed. Avoid a full-vocabulary
        # softmax that stock greedy drafting would not otherwise execute.
        rank_score_base = getattr(self, "_stock_top2_rank_scores", None)
        if rank_score_base is None or rank_score_base.device != tokens.device:
            rank_score_base = torch.tensor(
                ((1.0, 0.5),),
                dtype=torch.float32,
                device=tokens.device,
            )
            self._stock_top2_rank_scores = rank_score_base
        rank_scores = rank_score_base.expand(len(states), 2)
        return tokens, rank_scores

    def _stock_top2_scored_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return true H1 probabilities for best-first pruned Stock trees."""

        if not states:
            raise ValueError("at least one stock-tree state is required")
        method = getattr(model, "compute_stock_top2_greedy_candidates", None)
        if not callable(method):
            raise NotImplementedError(
                "pruned stock_top2 trees require the scored Stock H1 hook"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        with (
            record_function_or_nullcontext("residual_tree: stock_top2_scored_batch"),
            torch.inference_mode(),
        ):
            output = self._call_with_supported_kwargs(
                cast(Any, method),
                hidden_states=hidden,
                hidden=hidden,
            )
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            raise TypeError("scored Stock H1 hook must return token/prob tensors")
        tokens, probabilities = output
        expected_shape = (len(states), 2)
        if tokens.shape != expected_shape or probabilities.shape != expected_shape:
            raise ValueError(
                f"scored Stock H1 candidates must have shape {expected_shape}"
            )
        return tokens, probabilities

    def _stock_top10_dynamic_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
        *,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project Stock H1 top-k beam candidates once."""

        if not states:
            raise ValueError("at least one stock dynamic state is required")
        if top_k not in {9, 10}:
            raise ValueError("stock dynamic top_k must be 9 or 10")
        method = getattr(model, "compute_stock_top10_dynamic_candidates", None)
        if not callable(method):
            raise NotImplementedError(
                "stock dynamic trees require the Stock H1 top-10 hook"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        with (
            record_function_or_nullcontext("residual_tree: stock_top10_dynamic_batch"),
            torch.inference_mode(),
        ):
            output = self._call_with_supported_kwargs(
                cast(Any, method),
                hidden_states=hidden,
                hidden=hidden,
                top_k=top_k,
            )
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            raise TypeError(
                "stock dynamic H1 hook must return token/probability tensors"
            )
        tokens, probabilities = output
        expected_shape = (len(states), top_k)
        if tokens.shape != expected_shape or probabilities.shape != expected_shape:
            raise ValueError(
                f"stock dynamic candidates must have shape {expected_shape}"
            )
        return tokens, probabilities

    def _hybrid_dynamic_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
        *,
        candidate_selection: str,
        collect_h1_rank_overlap: bool = False,
        collect_union_provenance: bool = False,
        h2_to_h1_weight: float | None = None,
        oracle_target_tokens: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Project one configured width-10 H1/H2 candidate allocation."""

        if not states:
            raise ValueError("at least one hybrid dynamic state is required")
        try:
            _, h2_candidate_count, hook_name = _HYBRID_DYNAMIC_LAYOUTS[
                candidate_selection
            ]
        except KeyError as error:
            raise ValueError(
                f"unsupported hybrid dynamic layout: {candidate_selection}"
            ) from error
        method = getattr(model, hook_name, None)
        if not callable(method):
            raise NotImplementedError(
                f"hybrid dynamic trees require the {hook_name} hook"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        method_kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "hidden": hidden,
            "return_h1_rank_overlap": collect_h1_rank_overlap,
        }
        if candidate_selection == "hybrid_union_top10_dynamic":
            if h2_to_h1_weight is None:
                raise ValueError("hybrid union merge requires an H2-to-H1 weight")
            method_kwargs["h2_to_h1_weight"] = h2_to_h1_weight
            method_kwargs["return_union_provenance"] = collect_union_provenance
            method_kwargs["oracle_target_tokens"] = oracle_target_tokens
        needs_tree_depth = any(
            getattr(model, name, None) is not None
            for name in (
                "residual_tree_h2_tree_depth_bias",
                "residual_tree_hybrid_union_clipped_mass_calibration",
            )
        )
        if needs_tree_depth:
            depths = []
            for state in states:
                payload = state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "per-depth hybrid scoring requires the internal EAGLE "
                        "tree state"
                    )
                depths.append(len(payload.path_token_ids))
            if len(set(depths)) != 1:
                raise ValueError(
                    "per-depth hybrid scoring requires one uniform depth per "
                    "expansion batch"
                )
            method_kwargs["tree_depth"] = depths[0]
        with (
            record_function_or_nullcontext(
                f"residual_tree: {candidate_selection}_batch"
            ),
            torch.inference_mode(),
        ):
            output = self._call_with_supported_kwargs(
                cast(Any, method),
                **method_kwargs,
            )
        if collect_h1_rank_overlap and collect_union_provenance:
            raise ValueError("hybrid trace metadata modes are mutually exclusive")
        expected_values = (
            6 if collect_union_provenance else 3 if collect_h1_rank_overlap else 2
        )
        if (
            not isinstance(output, tuple)
            or len(output) != expected_values
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            suffix = (
                "/union-provenance"
                if collect_union_provenance
                else "/H1-rank"
                if collect_h1_rank_overlap
                else ""
            )
            raise TypeError(
                f"hybrid dynamic hook must return token/probability{suffix} tensors"
            )
        tokens, probabilities = output[:2]
        expected_shape = (len(states), 10)
        if tokens.shape != expected_shape or probabilities.shape != expected_shape:
            raise ValueError(
                f"hybrid dynamic candidates must have shape {expected_shape}"
            )
        if collect_h1_rank_overlap:
            rank_overlap = output[2]
            expected_overlap_shape = (len(states), h2_candidate_count)
            if rank_overlap.shape != expected_overlap_shape:
                raise ValueError(
                    f"hybrid H1-rank overlap must have shape {expected_overlap_shape}"
                )
            return tokens, probabilities, rank_overlap
        if collect_union_provenance:
            metadata = output[2:]
            if any(value.shape != expected_shape for value in metadata):
                raise ValueError(
                    "hybrid union provenance tensors must match candidate shape"
                )
            return cast(Any, output)
        return tokens, probabilities

    def _hybrid_top9_h2_dynamic_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project H1 ranks 1-9 and conditioned H2 top-1 once per state."""

        if not states:
            raise ValueError("at least one hybrid dynamic state is required")
        method = getattr(model, "compute_hybrid_top9_h2_dynamic_candidates", None)
        if not callable(method):
            raise NotImplementedError(
                "hybrid dynamic trees require the H1 top9 plus H2 hook"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        method_kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "hidden": hidden,
        }
        if getattr(model, "residual_tree_h2_tree_depth_bias", None) is not None:
            depths = []
            for state in states:
                payload = state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "per-depth H2 bias requires the internal EAGLE tree state"
                    )
                depths.append(len(payload.path_token_ids))
            if len(set(depths)) != 1:
                raise ValueError(
                    "per-depth H2 bias requires one uniform depth per expansion batch"
                )
            method_kwargs["tree_depth"] = depths[0]
        with (
            record_function_or_nullcontext(
                "residual_tree: hybrid_top9_h2_dynamic_batch"
            ),
            torch.inference_mode(),
        ):
            output = self._call_with_supported_kwargs(
                cast(Any, method),
                **method_kwargs,
            )
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            raise TypeError("hybrid dynamic hook must return token/probability tensors")
        tokens, probabilities = output
        expected_shape = (len(states), 10)
        if tokens.shape != expected_shape or probabilities.shape != expected_shape:
            raise ValueError(
                f"hybrid dynamic candidates must have shape {expected_shape}"
            )
        return tokens, probabilities

    def _hybrid_top5_h2_top5_dynamic_candidates_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project H1 top-5 and conditioned H2 top-5 once per state."""

        if not states:
            raise ValueError("at least one hybrid dynamic state is required")
        method = getattr(
            model,
            "compute_hybrid_top5_h2_top5_dynamic_candidates",
            None,
        )
        if not callable(method):
            raise NotImplementedError(
                "hybrid dynamic trees require the H1 top5 plus H2 top5 hook"
            )
        hidden = torch.stack([state.proposal_hidden for state in states], dim=0)
        method_kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "hidden": hidden,
        }
        if getattr(model, "residual_tree_h2_tree_depth_bias", None) is not None:
            depths = []
            for state in states:
                payload = state.payload
                if not isinstance(payload, _ResidualTreeKVPayload):
                    raise ValueError(
                        "per-depth H2 bias requires the internal EAGLE tree state"
                    )
                depths.append(len(payload.path_token_ids))
            if len(set(depths)) != 1:
                raise ValueError(
                    "per-depth H2 bias requires one uniform depth per expansion batch"
                )
            method_kwargs["tree_depth"] = depths[0]
        with (
            record_function_or_nullcontext(
                "residual_tree: hybrid_top5_h2_top5_dynamic_batch"
            ),
            torch.inference_mode(),
        ):
            output = self._call_with_supported_kwargs(
                cast(Any, method),
                **method_kwargs,
            )
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(value, torch.Tensor) for value in output)
        ):
            raise TypeError("hybrid dynamic hook must return token/probability tensors")
        tokens, probabilities = output
        expected_shape = (len(states), 10)
        if tokens.shape != expected_shape or probabilities.shape != expected_shape:
            raise ValueError(
                f"hybrid dynamic candidates must have shape {expected_shape}"
            )
        return tokens, probabilities

    def _residual_tree_transition(
        self,
        model: nn.Module,
        state: _ResidualTreeDraftState,
        token_id: int,
    ) -> _ResidualTreeDraftState:
        with record_function_or_nullcontext("residual_tree: transition"):
            hidden = state.transition_hidden.unsqueeze(0)
            token_ids = torch.tensor(
                [token_id], dtype=torch.int32, device=hidden.device
            )
            transition = getattr(model, "residual_tree_transition", None)
            if transition is None:
                transition = getattr(model, "residual_tree_step", None)
            if transition is None:
                return self._internal_eagle_tree_transition(state, token_id)

            with torch.inference_mode():
                output = self._call_with_supported_kwargs(
                    transition,
                    state=state,
                    token_id=token_id,
                    input_ids=token_ids,
                    hidden_states=hidden,
                    hidden=hidden,
                    position=state.position,
                )

            return self._coerce_residual_tree_state(output, fallback=state)

    def _residual_tree_transition_batch(
        self,
        model: nn.Module,
        states: Sequence[_ResidualTreeDraftState],
        token_ids: Sequence[int],
    ) -> list[_ResidualTreeDraftState]:
        """Advance one breadth level, using one tree-masked EAGLE call."""

        if len(states) != len(token_ids):
            raise ValueError("states and token_ids must have the same length")
        if not states:
            return []

        if self._has_model_residual_tree_transition(model):
            # Explicit model hooks retain their existing scalar contract. Models
            # can add a native batch hook later without changing the selector API.
            return [
                self._residual_tree_transition(model, state, token_id)
                for state, token_id in zip(states, token_ids)
            ]
        return self._internal_eagle_tree_transition_batch(states, token_ids)

    @staticmethod
    def _make_residual_tree_transition_plan(
        states: Sequence[_ResidualTreeDraftState],
        token_ids: Sequence[int],
    ) -> _ResidualTreeTransitionPlan:
        """Flatten the union of requested root-to-child paths into one trie."""

        if len(states) != len(token_ids):
            raise ValueError("states and token_ids must have the same length")
        if not states:
            raise ValueError("at least one residual-tree transition is required")

        first_payload = states[0].payload
        if not isinstance(first_payload, _ResidualTreeKVPayload):
            raise NotImplementedError(
                "batched residual-tree transition requires EAGLE KV payloads"
            )
        req_index = first_payload.req_index
        root_position = first_payload.root_position

        flat_token_ids: list[int] = []
        flat_hidden_inputs: list[torch.Tensor] = []
        parent_local_indices: list[int] = []
        node_depths: list[int] = []
        result_local_indices: list[int] = []
        child_payloads: list[_ResidualTreeKVPayload] = []
        prefix_to_local_index: dict[tuple[int, ...], int] = {}

        for state, token_id in zip(states, token_ids):
            payload = state.payload
            if not isinstance(payload, _ResidualTreeKVPayload):
                raise NotImplementedError(
                    "batched residual-tree transition requires EAGLE KV payloads"
                )
            if payload.req_index != req_index or payload.root_position != root_position:
                raise ValueError(
                    "one batched tree transition cannot mix serving requests"
                )
            if len(payload.path_token_ids) != len(payload.path_hidden_inputs):
                raise ValueError(
                    "residual-tree token and hidden-input paths must have "
                    "matching lengths"
                )

            child_path_token_ids = payload.path_token_ids + (int(token_id),)
            child_path_hidden_inputs = payload.path_hidden_inputs + (
                state.transition_hidden,
            )
            parent_local_index = -1
            for depth, (path_token_id, hidden_input) in enumerate(
                zip(child_path_token_ids, child_path_hidden_inputs),
                start=1,
            ):
                prefix = child_path_token_ids[:depth]
                local_index = prefix_to_local_index.get(prefix)
                if local_index is None:
                    local_index = len(flat_token_ids)
                    prefix_to_local_index[prefix] = local_index
                    flat_token_ids.append(path_token_id)
                    flat_hidden_inputs.append(hidden_input)
                    parent_local_indices.append(parent_local_index)
                    node_depths.append(depth)
                parent_local_index = local_index

            result_local_indices.append(parent_local_index)
            child_payloads.append(
                _ResidualTreeKVPayload(
                    req_index=req_index,
                    root_position=root_position,
                    path_token_ids=child_path_token_ids,
                    path_hidden_inputs=child_path_hidden_inputs,
                )
            )

        return _ResidualTreeTransitionPlan(
            req_index=req_index,
            root_position=root_position,
            token_ids=tuple(flat_token_ids),
            hidden_inputs=tuple(flat_hidden_inputs),
            parent_local_indices=tuple(parent_local_indices),
            node_depths=tuple(node_depths),
            result_local_indices=tuple(result_local_indices),
            child_payloads=tuple(child_payloads),
        )

    @staticmethod
    def _make_residual_tree_batch_transition_plan(
        states: Sequence[_ResidualTreeDraftState],
        token_ids: Sequence[int],
    ) -> _ResidualTreeBatchTransitionPlan:
        """Pack one union trie per serving request into a single query batch."""

        if len(states) != len(token_ids):
            raise ValueError("states and token_ids must have the same length")
        if not states:
            raise ValueError("at least one residual-tree transition is required")

        groups: dict[tuple[int, int], list[int]] = {}
        for input_index, state in enumerate(states):
            payload = state.payload
            if not isinstance(payload, _ResidualTreeKVPayload):
                raise NotImplementedError(
                    "batched residual-tree transition requires EAGLE KV payloads"
                )
            groups.setdefault(
                (payload.req_index, payload.root_position),
                [],
            ).append(input_index)

        request_indices: list[int] = []
        root_positions: list[int] = []
        query_start_locs = [0]
        flat_token_ids: list[int] = []
        flat_hidden_inputs: list[torch.Tensor] = []
        parent_local_indices: list[int] = []
        node_depths: list[int] = []
        result_local_indices = [-1] * len(states)
        child_payloads: list[_ResidualTreeKVPayload | None] = [None] * len(states)

        for (req_index, root_position), input_indices in groups.items():
            group_plan = SpecDecodeBaseProposer._make_residual_tree_transition_plan(
                [states[index] for index in input_indices],
                [token_ids[index] for index in input_indices],
            )
            node_offset = len(flat_token_ids)
            request_indices.append(req_index)
            root_positions.append(root_position)
            flat_token_ids.extend(group_plan.token_ids)
            flat_hidden_inputs.extend(group_plan.hidden_inputs)
            parent_local_indices.extend(group_plan.parent_local_indices)
            node_depths.extend(group_plan.node_depths)
            query_start_locs.append(len(flat_token_ids))
            for group_result_index, input_index in enumerate(input_indices):
                result_local_indices[input_index] = (
                    node_offset + group_plan.result_local_indices[group_result_index]
                )
                child_payloads[input_index] = group_plan.child_payloads[
                    group_result_index
                ]

        if any(index < 0 for index in result_local_indices) or any(
            payload is None for payload in child_payloads
        ):
            raise AssertionError("incomplete residual-tree batch transition plan")
        return _ResidualTreeBatchTransitionPlan(
            request_indices=tuple(request_indices),
            root_positions=tuple(root_positions),
            query_start_locs=tuple(query_start_locs),
            token_ids=tuple(flat_token_ids),
            hidden_inputs=tuple(flat_hidden_inputs),
            parent_local_indices=tuple(parent_local_indices),
            node_depths=tuple(node_depths),
            result_local_indices=tuple(result_local_indices),
            child_payloads=tuple(
                cast(_ResidualTreeKVPayload, payload) for payload in child_payloads
            ),
        )

    def _internal_eagle_tree_transition_batch(
        self,
        states: Sequence[_ResidualTreeDraftState],
        token_ids: Sequence[int],
    ) -> list[_ResidualTreeDraftState]:
        """Run a breadth level as one topology-masked EAGLE forward.

        The query contains the union trie of all requested root-to-child paths.
        Repeated ancestors are recomputed once and the attention mask hides
        siblings, so every output is equivalent to its scalar branch-local
        transition while avoiding one kernel-launch sequence per node.
        """

        common_attn_metadata = self._residual_tree_common_attn_metadata
        if common_attn_metadata is None:
            raise RuntimeError("residual-tree attention metadata is not initialized")
        if not self._residual_tree_kv_caches:
            raise NotImplementedError(
                "internal residual-tree transition requires draft KV caches"
            )

        plan = self._make_residual_tree_batch_transition_plan(states, token_ids)
        num_tree_nodes = len(plan.token_ids)
        max_depth = max(plan.node_depths)
        if max_depth > self.num_speculative_tokens:
            raise ValueError("residual-tree child depth exceeds draft token budget")

        physical_position_values: list[int] = []
        physical_request_indices: list[int] = []
        logical_position_values: list[int] = []
        for group_index, (req_index, root_position) in enumerate(
            zip(plan.request_indices, plan.root_positions, strict=True)
        ):
            group_start = plan.query_start_locs[group_index]
            group_end = plan.query_start_locs[group_index + 1]
            group_num_nodes = group_end - group_start
            physical_position_values.extend(
                range(
                    root_position + 1,
                    root_position + 1 + group_num_nodes,
                )
            )
            physical_request_indices.extend([req_index] * group_num_nodes)
            logical_position_values.extend(
                root_position + plan.node_depths[local_index]
                for local_index in range(group_start, group_end)
            )
        flat_physical_positions = torch.tensor(
            physical_position_values,
            dtype=torch.int64,
            device=self.device,
        )
        block_indices = torch.div(
            flat_physical_positions,
            self.block_size,
            rounding_mode="floor",
        )
        num_block_table_columns = common_attn_metadata.block_table_tensor.shape[1]
        if max(physical_position_values) // self.block_size >= num_block_table_columns:
            raise ValueError("residual-tree transition positions are not allocated")
        request_index_tensor = torch.tensor(
            physical_request_indices,
            dtype=torch.long,
            device=self.device,
        )
        block_ids = common_attn_metadata.block_table_tensor[
            request_index_tensor,
            block_indices,
        ]
        if bool(torch.any(block_ids < 0).item()):
            raise ValueError("residual-tree transition received an invalid block")
        slot_mapping = block_ids.to(torch.int64) * self.block_size + torch.remainder(
            flat_physical_positions, self.block_size
        )

        self.input_ids[:num_tree_nodes].copy_(
            torch.tensor(
                plan.token_ids,
                dtype=torch.int32,
                device=self.device,
            )
        )
        self.hidden_states[:num_tree_nodes].copy_(
            torch.stack(plan.hidden_inputs, dim=0)
        )
        logical_positions = torch.tensor(
            logical_position_values,
            dtype=torch.int64,
            device=self.device,
        )
        self._set_positions(num_tree_nodes, logical_positions)

        with record_function_or_nullcontext("residual_tree: transition_batch_metadata"):
            transition_attn_metadata = (
                self._make_residual_tree_transition_batch_metadata(
                    common_attn_metadata,
                    plan,
                    slot_mapping,
                )
            )
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                transition_attn_metadata,
                draft_index=max_depth,
            )
            if any(
                not isinstance(metadata, TritonAttentionMetadata)
                or metadata.tree_attn_mask is None
                for metadata in per_layer_attn_metadata.values()
            ):
                raise NotImplementedError(
                    "batched residual-tree drafting requires Triton tree attention"
                )
            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(
                    num_tree_nodes,
                    use_cudagraphs=False,
                )
            )
        if num_input_tokens != num_tree_nodes:
            raise NotImplementedError(
                "batched residual-tree transition does not support padded "
                "draft execution"
            )

        if self.supports_mm_inputs:
            self.inputs_embeds[:num_tree_nodes] = self.model.embed_input_ids(
                self.input_ids[:num_tree_nodes]
            )
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_tree_nodes]
        else:
            input_ids = self.input_ids[:num_tree_nodes]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_tree_nodes),
            "inputs_embeds": inputs_embeds,
            "hidden_states": self.hidden_states[:num_tree_nodes],
        }
        with (
            record_function_or_nullcontext("residual_tree: transition_batch_forward"),
            torch.inference_mode(),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_tree_nodes,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_tree_nodes,
                    slot_mapping,
                ),
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                transition_hidden = ret_hidden_states
            else:
                last_hidden_states, transition_hidden = ret_hidden_states

        child_states: list[_ResidualTreeDraftState] = []
        for local_index, child_payload in zip(
            plan.result_local_indices,
            plan.child_payloads,
        ):
            child_depth = len(child_payload.path_token_ids)
            child_states.append(
                _ResidualTreeDraftState(
                    proposal_hidden=last_hidden_states[local_index].detach(),
                    transition_hidden=transition_hidden[local_index].detach(),
                    position=torch.tensor(
                        child_payload.root_position + child_depth,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    payload=child_payload,
                )
            )
        return child_states

    def _internal_eagle_tree_transition(
        self,
        state: _ResidualTreeDraftState,
        token_id: int,
    ) -> _ResidualTreeDraftState:
        """Run one real EAGLE transition with branch-local KV restored."""

        payload = state.payload
        if not isinstance(payload, _ResidualTreeKVPayload):
            raise NotImplementedError(
                "internal residual-tree transition requires EAGLE KV payload"
            )
        common_attn_metadata = self._residual_tree_common_attn_metadata
        if common_attn_metadata is None:
            raise RuntimeError("residual-tree attention metadata is not initialized")
        if not self._residual_tree_kv_caches:
            raise NotImplementedError(
                "internal residual-tree transition requires draft KV caches"
            )

        if len(payload.path_token_ids) != len(payload.path_hidden_inputs) or len(
            payload.path_token_ids
        ) != len(payload.path_kv):
            raise ValueError(
                "scalar residual-tree transition requires a complete branch-local "
                "KV/token/hidden path"
            )
        child_depth = len(payload.path_token_ids) + 1
        if child_depth > self.num_speculative_tokens:
            raise ValueError("residual-tree child depth exceeds draft token budget")

        with record_function_or_nullcontext("residual_tree: transition_restore"):
            self._restore_residual_tree_path(payload, common_attn_metadata)
        child_position = payload.root_position + child_depth
        child_slot = self._tree_slot_for_position(
            common_attn_metadata,
            payload.req_index,
            child_position,
        )
        slot_mapping = torch.tensor(
            [child_slot],
            dtype=torch.int64,
            device=self.device,
        )

        self.input_ids[:1] = int(token_id)
        self.hidden_states[:1] = state.transition_hidden.view(1, -1)
        self._set_positions(
            1,
            torch.tensor(
                [child_position],
                dtype=torch.int64,
                device=self.device,
            ),
        )

        with record_function_or_nullcontext("residual_tree: transition_metadata"):
            transition_attn_metadata = self._make_residual_tree_transition_metadata(
                common_attn_metadata,
                payload.req_index,
                child_position,
                slot_mapping,
            )
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                transition_attn_metadata,
                draft_index=child_depth,
            )
            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(1, use_cudagraphs=False)
            )
        if num_input_tokens != 1:
            raise NotImplementedError(
                "internal residual-tree transition does not support padded "
                "draft execution"
            )

        if self.supports_mm_inputs:
            self.inputs_embeds[:1] = self.model.embed_input_ids(self.input_ids[:1])
            input_ids = None
            inputs_embeds = self.inputs_embeds[:1]
        else:
            input_ids = self.input_ids[:1]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(1),
            "inputs_embeds": inputs_embeds,
            "hidden_states": self.hidden_states[:1],
        }

        with (
            record_function_or_nullcontext("residual_tree: transition_forward"),
            torch.inference_mode(),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=1,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(1, slot_mapping),
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                transition_hidden = ret_hidden_states
            else:
                last_hidden_states, transition_hidden = ret_hidden_states

        with record_function_or_nullcontext("residual_tree: transition_snapshot"):
            child_kv = self._snapshot_residual_tree_slot(child_slot)
        child_payload = _ResidualTreeKVPayload(
            req_index=payload.req_index,
            root_position=payload.root_position,
            path_kv=payload.path_kv + (child_kv,),
            path_token_ids=payload.path_token_ids + (int(token_id),),
            path_hidden_inputs=payload.path_hidden_inputs + (state.transition_hidden,),
        )
        return _ResidualTreeDraftState(
            proposal_hidden=last_hidden_states[0].detach(),
            transition_hidden=transition_hidden[0].detach(),
            position=torch.tensor(
                child_position,
                dtype=torch.int64,
                device=state.transition_hidden.device,
            ),
            payload=child_payload,
        )

    def _make_residual_tree_transition_batch_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        plan: _ResidualTreeBatchTransitionPlan,
        slot_mapping: torch.Tensor,
    ) -> CommonAttentionMetadata:
        num_tree_nodes = len(plan.token_ids)
        query_start_loc_cpu = torch.tensor(
            plan.query_start_locs,
            dtype=torch.int32,
        )
        query_start_loc = query_start_loc_cpu.to(
            device=self.device,
            non_blocking=True,
        )
        group_node_counts = [
            plan.query_start_locs[index + 1] - plan.query_start_locs[index]
            for index in range(len(plan.request_indices))
        ]
        seq_len_values = [
            root_position + 1 + group_node_count
            for root_position, group_node_count in zip(
                plan.root_positions,
                group_node_counts,
                strict=True,
            )
        ]
        seq_len_cpu = torch.tensor(seq_len_values, dtype=torch.int32)
        seq_lens = seq_len_cpu.to(device=self.device, non_blocking=True)
        parent_local_indices_cpu = torch.tensor(
            plan.parent_local_indices,
            dtype=torch.int32,
        )
        node_depths_cpu = torch.tensor(plan.node_depths, dtype=torch.int32)
        max_query_len = max(group_node_counts)
        tree_attn_mask_cpu = torch.zeros(
            (num_tree_nodes, max_query_len),
            dtype=torch.bool,
        )
        for group_index, group_node_count in enumerate(group_node_counts):
            group_start = plan.query_start_locs[group_index]
            group_end = plan.query_start_locs[group_index + 1]
            tree_attn_mask_cpu[
                group_start:group_end,
                :group_node_count,
            ] = build_tree_attention_mask(
                parent_local_indices_cpu[group_start:group_end]
            )
        tree_attn_mask = tree_attn_mask_cpu.to(
            device=self.device,
            non_blocking=True,
        )
        request_index_tensor = torch.tensor(
            plan.request_indices,
            dtype=torch.long,
            device=self.device,
        )
        block_table = common_attn_metadata.block_table_tensor.index_select(
            0,
            request_index_tensor,
        )
        dcp_local_seq_lens = None
        if common_attn_metadata.dcp_local_seq_lens is not None:
            dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens.index_select(
                0,
                request_index_tensor,
            )

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=seq_len_cpu,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=seq_len_cpu,
            num_reqs=len(plan.request_indices),
            num_actual_tokens=num_tree_nodes,
            max_query_len=max_query_len,
            max_seq_len=max(seq_len_values),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=self._get_positions(num_tree_nodes),
            tree_attn_mask=tree_attn_mask,
            tree_parent_local_indices_cpu=parent_local_indices_cpu,
            tree_node_depths_cpu=node_depths_cpu,
        )

    def _make_residual_tree_transition_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        req_index: int,
        child_position: int,
        slot_mapping: torch.Tensor,
    ) -> CommonAttentionMetadata:
        query_start_loc = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=self.device,
        )
        query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
        seq_len_cpu = torch.tensor([child_position + 1], dtype=torch.int32)
        seq_lens = seq_len_cpu.to(device=self.device, non_blocking=True)
        block_table = common_attn_metadata.block_table_tensor[req_index : req_index + 1]
        dcp_local_seq_lens = None
        if common_attn_metadata.dcp_local_seq_lens is not None:
            dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens[
                req_index : req_index + 1
            ]

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=seq_len_cpu,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=seq_len_cpu,
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=child_position + 1,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=self._get_positions(1),
        )

    def _restore_residual_tree_path(
        self,
        payload: _ResidualTreeKVPayload,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        for depth_offset, kv_by_cache in enumerate(payload.path_kv, start=1):
            slot = self._tree_slot_for_position(
                common_attn_metadata,
                payload.req_index,
                payload.root_position + depth_offset,
            )
            for kv_cache, value in zip(self._residual_tree_kv_caches, kv_by_cache):
                self._write_kv_slot(kv_cache, slot, value)

    def _snapshot_residual_tree_slot(
        self,
        slot: int,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            self._read_kv_slot(kv_cache, slot)
            for kv_cache in self._residual_tree_kv_caches
        )

    def _collect_draft_kv_caches(self) -> tuple[torch.Tensor, ...]:
        kv_caches: list[torch.Tensor] = []
        seen_kv_ptrs: set[int] = set()
        for layer_name in sorted(self._draft_attn_layer_names):
            layer = self.compilation_config.static_forward_context[layer_name]
            if getattr(layer.impl, "_is_per_token_head_quant", False):
                raise NotImplementedError(
                    "internal residual-tree transition does not support "
                    "per-token-head KV quantization yet because branch snapshots "
                    "must include the side scale caches"
                )
            kv_cache = layer.kv_cache
            if not isinstance(kv_cache, torch.Tensor):
                raise NotImplementedError(
                    "internal residual-tree transition supports only tensor "
                    "draft KV caches"
                )
            if kv_cache.ndim < 3 or kv_cache.shape[1] != 2:
                raise NotImplementedError(
                    "internal residual-tree transition currently supports "
                    "Triton-style [num_blocks, 2, block_size, ...] KV caches"
                )
            ptr = kv_cache.data_ptr()
            if ptr in seen_kv_ptrs:
                continue
            seen_kv_ptrs.add(ptr)
            kv_caches.append(kv_cache)
        return tuple(kv_caches)

    def _tree_slot_for_position(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        req_index: int,
        position: int,
    ) -> int:
        block_index = position // self.block_size
        block_offset = position % self.block_size
        block_table = common_attn_metadata.block_table_tensor
        if block_index >= block_table.shape[1]:
            raise ValueError("residual-tree transition position is not allocated")
        block_id = int(block_table[req_index, block_index].detach().cpu().item())
        if block_id < 0:
            raise ValueError("residual-tree transition received an invalid block")
        return block_id * self.block_size + block_offset

    @staticmethod
    def _read_kv_slot(kv_cache: torch.Tensor, slot: int) -> torch.Tensor:
        block_size = kv_cache.shape[2]
        block = slot // block_size
        offset = slot % block_size
        return kv_cache[block, :, offset].detach().clone()

    @staticmethod
    def _write_kv_slot(
        kv_cache: torch.Tensor,
        slot: int,
        value: torch.Tensor,
    ) -> None:
        block_size = kv_cache.shape[2]
        block = slot // block_size
        offset = slot % block_size
        kv_cache[block, :, offset].copy_(value)

    @staticmethod
    def _coerce_residual_tree_state(
        output: Any,
        *,
        fallback: _ResidualTreeDraftState,
    ) -> _ResidualTreeDraftState:
        if isinstance(output, _ResidualTreeDraftState):
            return output
        if hasattr(output, "proposal_hidden") and hasattr(output, "transition_hidden"):
            proposal_hidden = cast(Any, output).proposal_hidden
            transition_hidden = cast(Any, output).transition_hidden
            position = getattr(output, "position", fallback.position)
            payload = getattr(output, "payload", output)
        elif isinstance(output, tuple):
            if len(output) == 2:
                proposal_hidden, transition_hidden = output
                position = fallback.position
                payload = fallback.payload
            elif len(output) == 3:
                proposal_hidden, transition_hidden, payload = output
                position = getattr(payload, "position", fallback.position)
            else:
                raise ValueError(
                    "residual tree transition tuple output must be "
                    "(proposal_hidden, transition_hidden[, payload])"
                )
        else:
            proposal_hidden = transition_hidden = output
            position = fallback.position
            payload = fallback.payload

        device = fallback.proposal_hidden.device
        proposal_hidden = torch.as_tensor(proposal_hidden, device=device)
        transition_hidden = torch.as_tensor(transition_hidden, device=device)
        proposal_hidden = proposal_hidden.reshape(-1, proposal_hidden.shape[-1])
        transition_hidden = transition_hidden.reshape(-1, transition_hidden.shape[-1])
        if proposal_hidden.shape[0] != 1 or transition_hidden.shape[0] != 1:
            raise ValueError("residual tree transition must return one state")

        return _ResidualTreeDraftState(
            proposal_hidden=proposal_hidden[0].detach(),
            transition_hidden=transition_hidden[0].detach(),
            position=position,
            payload=payload,
        )

    @staticmethod
    def _call_with_supported_kwargs(method: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            if "hidden_states" in kwargs and "input_ids" in kwargs:
                return method(kwargs["hidden_states"], kwargs["input_ids"])
            if "hidden_states" in kwargs:
                return method(kwargs["hidden_states"])
            return method()

        params = signature.parameters
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        ):
            return method(**kwargs)

        supported = {name: value for name, value in kwargs.items() if name in params}
        if supported:
            return method(**supported)
        if "hidden_states" in kwargs and "input_ids" in kwargs:
            return method(kwargs["hidden_states"], kwargs["input_ids"])
        if "hidden_states" in kwargs:
            return method(kwargs["hidden_states"])
        return method()

    def propose(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            model = self.model
            if isinstance(model, BreakableCUDAGraphWrapper):
                model = model.unwrap()
            assert isinstance(
                model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                    Eagle3Qwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        # Step 0 of index_share_for_mtp_iteration: let the MTP layer
        # compute its own indices (skip_topk=False) so subsequent steps
        # can reuse them.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(False)

        with (
            record_function_or_nullcontext("eagle: first_pass_forward"),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    slot_mapping_size, common_attn_metadata.slot_mapping
                ),
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        # After step 0: switch to reuse mode so steps 1+ skip the indexer
        # and read the indices that step 0 just wrote into the shared buffer.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(True)

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # No draft tokens requested (e.g. Dynamic SD decided K=0).
        # The prefill forward pass above already ran to keep the drafter
        # KV cache in sync, so just return an empty tensor.
        if self.num_speculative_tokens == 0:
            return torch.empty(
                batch_size,
                0,
                device=sample_hidden_states.device,
                dtype=torch.int64,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            # Write the sampling positions into the front of the
            # positions buffer so that subsequent loop iterations
            # (which read via _get_positions) use the correct values.
            self.positions[:batch_size] = positions

        with record_function_or_nullcontext("eagle: sample"):
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()

            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions,
                    common_attn_metadata,
                    batch_size,
                    input_batch_size,
                    block_size,
                )

            # Rebuild attention metadata. When draft positions are constant
            # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
            # loop iterations so we build once and reuse.
            if not self.constant_draft_positions or token_index == 0:
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        common_attn_metadata, draft_index=token_index + 1
                    )
                )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with (
                record_function_or_nullcontext("eagle: transition_forward"),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    num_tokens_across_dp=batch_size_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=self._get_slot_mapping(input_batch_size),
                ),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            with record_function_or_nullcontext("eagle: sample"):
                draft_token_ids, draft_probs = self._sample_draft_tokens(
                    last_hidden_states[:batch_size], sampling_metadata
                )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """Update positions, slot mappings, and sequence metadata for the
        next draft step. Returns the updated positions tensor."""
        positions_1d = positions[0] if self.uses_mrope else positions
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :batch_size]
        else:
            out_pos = self.positions[:batch_size]
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions_1d,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
        if self.uses_mrope:
            self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
            positions = self.mrope_positions[:, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                0, :batch_size
            ]
            positions = self.xdrope_positions[0, :batch_size]
        else:
            positions = self.positions[:batch_size]
        common_attn_metadata.max_seq_len = min(
            common_attn_metadata.max_seq_len + 1,
            self.max_model_len,
        )

        if common_attn_metadata._seq_lens_cpu is not None:
            common_attn_metadata._seq_lens_cpu += 1
        if common_attn_metadata._num_computed_tokens_cpu is not None:
            common_attn_metadata._num_computed_tokens_cpu += 1
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            common_attn_metadata.seq_lens_cpu_upper_bound += 1

        return positions

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            self.input_ids[token_indices_to_sample] = next_token_ids

            # copy inputs to buffer for cudagraph
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call a custom triton kernel to copy input_ids and positions
            # into the correct slots in the preallocated buffers self.input_ids,
            # self.positions.
            batch_size = cad.batch_size()
            # Since we might have to copy a lot of data for prefills, we select the
            # block size based on the max query length and limit to max 256 slots/block.
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(256, next_power_of_2(max_num_tokens_per_request))
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # Destination indices to write target_hidden_states into drafting buffer.
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel grid: one program per request (row)
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            copy_and_expand_eagle_inputs_kernel[grid](
                # (Padded) Inputs from the target model
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # sampled tokens, one per request
                # Outputs to the drafting buffers
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # Doesn't support mrope for now
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # Input metadata
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # Sizing info
                # Note that we can deduce batch_size for free from the grid size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. Update the common attention metadata with the new (meta)data
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def model_returns_tuple(self) -> bool:
        if self.method == "mtp":
            # DeepSeek-family MTP (deepseek_mtp.py) recycles the post-final-
            # norm hidden, so its forward returns (logit_hidden,
            # recycle_hidden). Other MTP families return a single tensor.
            return "DeepSeekMTPModel" in (
                self.draft_model_config.hf_config.architectures or []
            )
        return self.method not in ("mtp", "draft_model", "dflash")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute backup token IDs for discarded requests.
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata | TreeSpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        if isinstance(spec_decode_metadata, TreeSpecDecodeMetadata):
            # The tree rows have been compacted so accepted nodes form the same
            # root + accepted prefix layout as linear speculative decoding.
            # The padded EAGLE kernel wants cumulative draft slots, excluding
            # one root/bonus row per request.
            roots_seen = torch.arange(
                1,
                num_reqs + 1,
                dtype=spec_decode_metadata.cu_num_tree_nodes.dtype,
                device=device,
            )
            cu_num_draft_tokens = spec_decode_metadata.cu_num_tree_nodes - roots_seen
        else:
            cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens
        eagle_prepare_inputs_padded_kernel[grid](
            cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # upper_bound - rejected = actual post-rejection seq_lens (no D2H sync).
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu_upper_bound - num_rejected_tokens
        )

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = async_tensor_h2d(token_indices_np, device=device)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=async_tensor_h2d(new_query_start_loc_cpu, device=device),
            seq_lens=async_tensor_h2d(new_seq_lens_cpu, device=device),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        """
        spec_cfg = self.speculative_config
        base = self.vllm_config

        if spec_cfg.moe_backend is not None:
            base = replace(
                base,
                kernel_config=replace(
                    base.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )

        # Note (matt): Never inherit the attention backend from base, because there are
        # many opportunities for incompatibility, so we always independently autoselect
        # unless explicitly specified in the speculative config.
        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        return base

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        # Filter to only layers that have KV cache specs.
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        }

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Cohere2VisionForConditionalGeneration",
                "Exaone4_5_ForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "InternS2PreviewForConditionalGeneration",
                "MiMoV2OmniForCausalLM",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
                "Gemma4UnifiedForConditionalGeneration",
                "Step3p7ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)

        if (
            self.parallel_drafting
            and self.pass_hidden_states_to_model
            and self.parallel_drafting_hidden_state_tensor is not None
        ):
            flat_mask = self.model.mask_hidden.view(-1)
            if self.eagle3_use_aux_hidden_state:
                # EAGLE3: mask_hidden stores all aux hidden states,
                # project through combine_hidden_states
                self.parallel_drafting_hidden_state_tensor.copy_(
                    self.model.combine_hidden_states(flat_mask)
                )
            else:
                self.parallel_drafting_hidden_state_tensor.copy_(flat_mask)

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.
        """
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own LM head, and some may have a
        duplicate copy of the target model's LM head. In these cases, we share
        the target model's LM head with the draft model to save memory.
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and hasattr(target_language_model.lm_head, "weight")
                and hasattr(self.model.lm_head, "weight")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if hasattr(target_language_model.model, "topk_indices_buffer"):
            target_buffer = target_language_model.model.topk_indices_buffer
            if hasattr(self.model.model, "topk_indices_buffer"):
                del self.model.model.topk_indices_buffer
            self.model.model.topk_indices_buffer = target_buffer
            # Also share at per-module level so that the indexer and
            # sparse-attention backends in each MTP layer read from
            # the target model's buffer.
            for _, module in self.model.model.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer
            logger.info(
                "Detected MTP model with topk_indices_buffer. "
                "Sharing target model topk_indices_buffer with the draft model."
            )

        # Detect index_share_for_mtp_iteration: when True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        self._share_mtp_indices = getattr(
            draft_hf_config, "index_share_for_mtp_iteration", False
        )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            logger.info(
                "Using local argmax reduction for draft token generation "
                "(communication: O(2*tp_size) vs O(vocab_size))."
            )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        only_one_forward_pass = is_graph_capturing or self.parallel_drafting
        for fwd_idx in range(
            1 if only_one_forward_pass else self.num_speculative_tokens
        ):
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                self.model(**kwargs)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Find which kv_cache_group the draft layers belong to
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]

                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = sample_with_exponential_noise(probs.clone(), q)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
