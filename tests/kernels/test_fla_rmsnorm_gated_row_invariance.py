# SPDX-License-Identifier: Apache-2.0
"""GPU regression for Qwen3.5 GDN RMSNormGated row invariance.

The production no-spec call normalizes one token as 48 independent value-head
rows of width 128.  Residual trees flatten more tokens into the same call.  A
root token must therefore remain bitwise identical when its 48 rows are
embedded in any supported flattened tree size.

This test intentionally uses the captured layer-29 inputs/output and the real
checkpoint weight.  It skips when those local research artifacts are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

FLAT_ROW_COUNTS = (
    48,
    96,
    144,
    192,
    288,
    336,
    432,
    624,
    816,
    1200,
    1584,
    1920,
)
HEAD_ROWS = 48
HEAD_DIM = 128
WEIGHT_KEY = "model.language_model.layers.29.linear_attn.norm.weight"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CAPTURE = REPO_ROOT / (
    "runs/h1_j_native_speed_ultrachat_v3/"
    "b6_pos76_gdn29_outproj_v1/captures/"
    "pid3096985_call00_rows1.pt"
)
DEFAULT_MODEL = Path(
    "/home/shared/huggingface/hub/"
    "models--Qwen--Qwen3.5-27B/snapshots/"
    "fc05daec18b0a78c049392ed2e771dde82bdf654"
)


def _configured_path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser().resolve()


def _load_weight(model_path: Path, key: str) -> torch.Tensor:
    if model_path.is_dir():
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            pytest.skip(f"model index is absent: {index_path}")
        index = json.loads(index_path.read_text())
        shard_name = index.get("weight_map", {}).get(key)
        if not isinstance(shard_name, str):
            pytest.skip(f"checkpoint index does not contain {key}")
        weights_path = model_path / shard_name
    else:
        weights_path = model_path
    if not weights_path.is_file():
        pytest.skip(f"checkpoint shard is absent: {weights_path}")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        if key not in set(handle.keys()):
            pytest.skip(f"checkpoint shard does not contain {key}")
        return handle.get_tensor(key)


@pytest.fixture(scope="module")
def real_layer29_case() -> tuple[torch.Tensor, ...]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    capture_path = _configured_path("STACK_SPEC_GDN29_CAPTURE", DEFAULT_CAPTURE)
    model_path = _configured_path("STACK_SPEC_QWEN35_MODEL", DEFAULT_MODEL)
    if not capture_path.is_file():
        pytest.skip(f"layer-29 capture is absent: {capture_path}")

    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    tensors = capture.get("tensors") if isinstance(capture, dict) else None
    if not isinstance(tensors, dict):
        raise AssertionError("layer-29 capture has no tensors mapping")

    core = tensors["core_before_norm"]
    gate = tensors["z_before_norm"]
    expected = tensors["post_norm_flat"]
    weight = _load_weight(model_path, WEIGHT_KEY)
    expected_shape = (HEAD_ROWS, HEAD_DIM)
    for name, tensor in (
        ("core_before_norm", core),
        ("z_before_norm", gate),
        ("post_norm_flat", expected),
    ):
        assert tensor.shape == expected_shape, (name, tensor.shape)
        assert tensor.dtype == torch.bfloat16, (name, tensor.dtype)
    assert weight.shape == (HEAD_DIM,)
    assert weight.dtype == torch.float32
    # The checkpoint shard stores this vector as FP32, while the BF16 model
    # casts it during loading.  Reproduce the serving dtype explicitly.
    weight = weight.to(torch.bfloat16)

    device = torch.device("cuda:0")
    return tuple(tensor.to(device) for tensor in (core, gate, expected, weight))


def _tree_production(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    from vllm.model_executor.layers.fla.ops import layernorm_guard

    fn = getattr(layernorm_guard, "rmsnorm_gated_tree_fn", None)
    if fn is None:
        pytest.skip("rmsnorm_gated_tree_fn production candidate has not landed")
    output = fn(x, weight, z, 1e-6, activation)
    if isinstance(output, tuple):
        output = output[0]
    assert isinstance(output, torch.Tensor)
    return output


def _embedded_case(
    core: torch.Tensor,
    gate: torch.Tensor,
    expected: torch.Tensor,
    flat_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert flat_rows % HEAD_ROWS == 0
    repeats = flat_rows // HEAD_ROWS
    return (
        core.repeat(repeats, 1),
        gate.repeat(repeats, 1),
        expected.repeat(repeats, 1),
    )


def _cuda_kernel_names(fn) -> tuple[list[str], torch.Tensor]:
    # Warm compilation and allocator state before launch counting.
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        output = fn()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in trace.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    return names, output


@pytest.mark.parametrize("flat_rows", FLAT_ROW_COUNTS)
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_synthetic_tree_rmsnorm_matches_chunked_ordinary_rpb1_single_launch(
    flat_rows: int,
    activation: str,
) -> None:
    from vllm.model_executor.layers.fla.ops.layernorm_guard import (
        calc_rows_per_block,
        rmsnorm_fn,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    assert calc_rows_per_block(HEAD_ROWS, device) == 1
    generator = torch.Generator(device=device).manual_seed(20260804)
    x = torch.randn(
        flat_rows,
        HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.02)
    z = torch.randn(
        flat_rows,
        HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(2.0)
    weight = (
        torch.randn(
            HEAD_DIM,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        .mul_(0.1)
        .add_(1.0)
    )

    expected_chunks = []
    for start in range(0, flat_rows, HEAD_ROWS):
        expected_chunks.append(
            rmsnorm_fn(
                x[start : start + HEAD_ROWS],
                weight,
                None,
                z=z[start : start + HEAD_ROWS],
                eps=1e-6,
                norm_before_gate=True,
                activation=activation,
            )
        )
    expected = torch.cat(expected_chunks)
    kernel_names, output = _cuda_kernel_names(
        lambda: _tree_production(x, weight, z, activation)
    )

    assert output.shape == (flat_rows, HEAD_DIM)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, expected), (
        activation,
        flat_rows,
        int((output != expected).sum().item()),
        float((output.float() - expected.float()).abs().max().item()),
    )
    assert len(kernel_names) == 1, (activation, flat_rows, kernel_names)


@pytest.mark.parametrize("flat_rows", FLAT_ROW_COUNTS)
@torch.inference_mode()
def test_real_layer29_tree_rmsnorm_is_bitwise_row_invariant(
    real_layer29_case: tuple[torch.Tensor, ...],
    flat_rows: int,
) -> None:
    core, gate, expected, weight = real_layer29_case
    x, z, expected_full = _embedded_case(core, gate, expected, flat_rows)

    output = _tree_production(x, weight, z)

    assert output.shape == (flat_rows, HEAD_DIM)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, expected_full), (
        flat_rows,
        int((output != expected_full).sum().item()),
        float((output.float() - expected_full.float()).abs().max().item()),
    )
