"""Frozen ordered-head probability calibration for residual-tree ranking."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


CALIBRATOR_SCHEMA = "qwen3_32b_greedy_correction_isotonic_calibrator_v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class OrderedHeadIsotonicCalibration:
    """Ten immutable piecewise-constant monotone probability maps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_bytes())
        tables = payload.get("tables")
        if payload.get("schema") != CALIBRATOR_SCHEMA or not isinstance(
            tables, list
        ) or len(tables) != 10:
            raise ValueError("residual-tree calibration must contain H1 through H10")
        boundaries = []
        estimates = []
        maximum_blocks = 0
        parsed = []
        for expected_head, table in enumerate(tables, start=1):
            if not isinstance(table, dict) or int(table.get("head", -1)) != expected_head:
                raise ValueError("residual-tree calibration head order differs")
            blocks = table.get("blocks")
            if not isinstance(blocks, list) or not blocks:
                raise ValueError("residual-tree calibration table is empty")
            previous_score = -1.0
            previous_estimate = -1.0
            local_boundaries = []
            local_estimates = []
            for block in blocks:
                maximum = float(block["maximum_score"])
                estimate = float(block["estimate"])
                if not (
                    math.isfinite(maximum)
                    and math.isfinite(estimate)
                    and previous_score < maximum <= 1.0
                    and previous_estimate <= estimate <= 1.0
                ):
                    raise ValueError("residual-tree calibration is not monotone")
                local_boundaries.append(maximum)
                local_estimates.append(estimate)
                previous_score = maximum
                previous_estimate = estimate
            identity = dict(table)
            mapping_sha256 = identity.pop("mapping_sha256", None)
            if mapping_sha256 != _canonical_sha256(identity):
                raise ValueError("residual-tree calibration mapping identity differs")
            parsed.append((local_boundaries, local_estimates))
            maximum_blocks = max(maximum_blocks, len(local_boundaries))

        for local_boundaries, local_estimates in parsed:
            padding = maximum_blocks - len(local_boundaries)
            boundaries.append(local_boundaries + [1.0] * padding)
            estimates.append(local_estimates + [local_estimates[-1]] * padding)
        self._boundaries_cpu = torch.tensor(boundaries, dtype=torch.float64)
        self._estimates_cpu = torch.tensor(estimates, dtype=torch.float64)
        self._device_cache: dict[
            tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def __call__(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 2 or probabilities.shape[1] != 10:
            raise ValueError("ordered-head probabilities must have shape [batch,10]")
        values = probabilities.to(torch.float64)
        if not bool(torch.isfinite(values).all()) or bool(
            ((values < 0.0) | (values > 1.0)).any()
        ):
            raise ValueError("ordered-head probabilities must lie in [0,1]")
        key = (probabilities.device.type, probabilities.device.index)
        tensors = self._device_cache.get(key)
        if tensors is None:
            tensors = (
                self._boundaries_cpu.to(probabilities.device),
                self._estimates_cpu.to(probabilities.device),
            )
            self._device_cache[key] = tensors
        boundaries, estimates = tensors
        # torch.searchsorted supports one sorted row per head.  Transposing the
        # small [batch,10] score matrix applies all ten maps in one device op.
        indices = torch.searchsorted(
            boundaries,
            values.transpose(0, 1).contiguous(),
            right=False,
        ).clamp_max(boundaries.shape[1] - 1)
        calibrated = estimates.gather(1, indices).transpose(0, 1)
        return calibrated.to(probabilities.dtype)


__all__ = ["OrderedHeadIsotonicCalibration"]
