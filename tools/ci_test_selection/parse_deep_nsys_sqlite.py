#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Export launch-level test -> Python/C++ -> CUDA kernel trace records."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

CALL_PREFIX = "citest::"
AUX_PREFIXES = ("citest-setup::", "citest-teardown::")


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    return dict(connection.execute("SELECT id, value FROM StringIds"))


def _resolved(value: Any, strings: dict[int, str]) -> str | None:
    if value is None:
        return None
    return strings.get(value, str(value)) if isinstance(value, int) else str(value)


def _load_ranges(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    columns = _columns(connection, "NVTX_EVENTS")
    if not columns:
        raise SystemExit("no NVTX_EVENTS table: was --trace=nvtx enabled?")
    text = "text" if "text" in columns else "NULL"
    text_id = "textId" if "textId" in columns else "NULL"
    rows = []
    for start, end, global_tid, direct, identifier in connection.execute(
        f"SELECT start, end, globalTid, {text}, {text_id} "
        "FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        label = direct if direct is not None else strings.get(identifier)
        if label:
            rows.append(
                {
                    "end_ns": int(end),
                    "global_tid": int(global_tid),
                    "label": label,
                    "start_ns": int(start),
                }
            )
    return rows


def _test_identity(label: str) -> tuple[str, str] | None:
    if label.startswith(CALL_PREFIX):
        return label[len(CALL_PREFIX) :], "call"
    for prefix in AUX_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix) :], prefix.split("::")[0]
    return None


def _temporal_attributions(
    test_ranges: list[tuple[int, int, str, str]],
    launches: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], tuple[str, str] | bool | None]:
    ranges = sorted(test_ranges)
    ordered_launches = sorted(
        launches.items(), key=lambda item: int(item[1]["start_ns"])
    )
    active: Counter[tuple[str, str]] = Counter()
    endings: list[tuple[int, tuple[str, str]]] = []
    result: dict[tuple[int, int], tuple[str, str] | bool | None] = {}
    range_index = 0
    for key, launch in ordered_launches:
        timestamp = int(launch["start_ns"])
        while range_index < len(ranges) and ranges[range_index][0] <= timestamp:
            start, end, node_id, phase = ranges[range_index]
            identity = (node_id, phase)
            active[identity] += 1
            heappush(endings, (end, identity))
            range_index += 1
        while endings and endings[0][0] < timestamp:
            _end, identity = heappop(endings)
            active[identity] -= 1
            if not active[identity]:
                del active[identity]
        if sum(active.values()) == 1:
            result[key] = next(iter(active))
        elif active:
            result[key] = False
        else:
            result[key] = None
    return result


def _load_callchains(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> dict[int, list[dict[str, Any]]]:
    columns = _columns(connection, "CUDA_CALLCHAINS")
    required = {"id", "stackDepth", "symbol", "module"}
    if not required <= columns:
        return {}
    optional = [name for name in ("unresolved", "originalIP") if name in columns]
    select = ", ".join(["id", "stackDepth", "symbol", "module", *optional])
    chains: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(
        f"SELECT {select} FROM CUDA_CALLCHAINS ORDER BY id, stackDepth"
    ):
        identifier, depth, symbol, module, *values = row
        frame = {
            "depth": int(depth),
            "module": _resolved(module, strings),
            "symbol": _resolved(symbol, strings),
        }
        frame.update(dict(zip(optional, values)))
        chains[int(identifier)].append(frame)
    return dict(chains)


def _load_artifacts(path: Path | None) -> dict[str, set[str]]:
    artifacts: dict[str, set[str]] = defaultdict(set)
    if path is None:
        return artifacts
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row.get("source_kind") == "artifact"
                and row.get("destination_kind") == "kernel"
            ):
                artifacts[row["destination"]].add(row["source"])
    return artifacts


def _load_build_provenance(
    path: Path | None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    targets: dict[str, set[str]] = defaultdict(set)
    files: dict[str, set[str]] = defaultdict(set)
    if path is None:
        return targets, files
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row.get("source_kind") == "target"
                and row.get("destination_kind") == "artifact"
            ):
                targets[row["destination"]].add(row["source"])
            elif (
                row.get("source_kind") == "file"
                and row.get("destination_kind") == "target"
            ):
                files[row["destination"]].add(row["source"])
    return targets, files


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def export_deep_trace(
    sqlite_path: Path,
    *,
    job_key: str,
    kernel_map: Path | None,
    build_graph: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        strings = _strings(connection)
        ranges = _load_ranges(connection, strings)
        ranges_by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        test_ranges = []
        for span in ranges:
            ranges_by_tid[span["global_tid"]].append(span)
            identity = _test_identity(span["label"])
            if identity:
                node_id, phase = identity
                test_ranges.append((span["start_ns"], span["end_ns"], node_id, phase))
        for spans in ranges_by_tid.values():
            spans.sort(key=lambda span: span["start_ns"])
        starts_by_tid = {
            thread: [span["start_ns"] for span in spans]
            for thread, spans in ranges_by_tid.items()
        }

        runtime_columns = _columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
        required_runtime = {"correlationId", "globalTid", "start"}
        if not required_runtime <= runtime_columns:
            raise SystemExit("CUDA runtime table lacks launch correlation columns")
        runtime_optional = [
            name for name in ("end", "nameId", "callchainId") if name in runtime_columns
        ]
        runtime_select = ", ".join(
            ["correlationId", "globalTid", "start", *runtime_optional]
        )
        launches: dict[tuple[int, int], dict[str, Any]] = {}
        duplicates = set()
        for row in connection.execute(
            f"SELECT {runtime_select} FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            correlation_id, global_tid, start, *values = row
            key = (int(global_tid) >> 24, int(correlation_id))
            launch = {
                "correlation_id": int(correlation_id),
                "global_tid": int(global_tid),
                "process_key": int(global_tid) >> 24,
                "start_ns": int(start),
            }
            launch.update(dict(zip(runtime_optional, values)))
            if key in launches:
                duplicates.add(key)
            else:
                launches[key] = launch

        temporal = _temporal_attributions(test_ranges, launches)
        callchains = _load_callchains(connection, strings)
        artifacts_by_kernel = _load_artifacts(kernel_map)

        kernel_columns = _columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        if "mangledName" not in kernel_columns:
            raise SystemExit("kernel table has no mangledName")
        process_column = (
            "globalPid"
            if "globalPid" in kernel_columns
            else "globalTid"
            if "globalTid" in kernel_columns
            else None
        )
        if process_column is None:
            raise SystemExit("kernel table has no process identity")
        kernel_optional = [
            name
            for name in ("start", "end", "shortName", "demangledName")
            if name in kernel_columns
        ]
        kernel_select = ", ".join(
            ["correlationId", "mangledName", process_column, *kernel_optional]
        )
        rows = []
        for values in connection.execute(
            f"SELECT {kernel_select} FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            correlation_id, mangled, process_value, *optional_values = values
            key = (int(process_value) >> 24, int(correlation_id))
            if key in duplicates:
                raise SystemExit("duplicate CUDA correlation ID within one process")
            launch = launches.get(key)
            if launch is None:
                continue
            timestamp = int(launch["start_ns"])
            spans = ranges_by_tid.get(launch["global_tid"], [])
            starts = starts_by_tid.get(launch["global_tid"], [])
            active_ranges = []
            for index in range(bisect_right(starts, timestamp) - 1, -1, -1):
                span = spans[index]
                if span["end_ns"] >= timestamp:
                    active_ranges.append(span)
            active_ranges.reverse()
            exact = next(
                (
                    identity
                    for span in reversed(active_ranges)
                    if (identity := _test_identity(span["label"])) is not None
                ),
                None,
            )
            if exact:
                node_id, phase = exact
                attribution_mode = "exact_test"
            else:
                fallback = temporal.get(key)
                if fallback:
                    node_id, phase = fallback
                    attribution_mode = "temporal_test"
                else:
                    node_id, phase = None, "job"
                    attribution_mode = "job_union"

            kernel = _resolved(mangled, strings)
            optional = dict(zip(kernel_optional, optional_values))
            callchain_id = launch.get("callchainId")
            chain = (
                callchains.get(int(callchain_id), [])
                if callchain_id is not None
                else []
            )
            artifacts = sorted(artifacts_by_kernel.get(kernel or "", set()))
            rows.append(
                {
                    "active_ranges": active_ranges,
                    "artifact_class": (
                        "unmapped"
                        if not artifacts
                        else "matched"
                        if len(artifacts) == 1
                        else "ambiguous"
                    ),
                    "artifacts": artifacts,
                    "attribution_mode": attribution_mode,
                    "correlation_id": int(correlation_id),
                    "cuda_api": _resolved(launch.get("nameId"), strings),
                    "cuda_callchain": chain,
                    "cuda_callchain_id": callchain_id,
                    "global_tid": launch["global_tid"],
                    "job_key": job_key,
                    "kernel_demangled": _resolved(
                        optional.get("demangledName"), strings
                    ),
                    "kernel_end_ns": optional.get("end"),
                    "kernel_mangled": kernel,
                    "kernel_short": _resolved(optional.get("shortName"), strings),
                    "kernel_start_ns": optional.get("start"),
                    "launch_end_ns": launch.get("end"),
                    "launch_start_ns": timestamp,
                    "phase": phase,
                    "process_key": launch["process_key"],
                    "test_id": node_id,
                }
            )
    finally:
        connection.close()

    rows.sort(
        key=lambda row: (
            int(row["launch_start_ns"]),
            int(row["process_key"]),
            int(row["correlation_id"]),
            row["kernel_mangled"] or "",
        )
    )
    for index, row in enumerate(rows):
        row["launch_index"] = index

    targets_by_artifact, files_by_target = _load_build_provenance(build_graph)
    provenance = []
    kernels = {row["kernel_mangled"] for row in rows if row["kernel_mangled"]}
    for kernel in sorted(kernels):
        artifact_rows = []
        for artifact in sorted(artifacts_by_kernel.get(kernel, set())):
            target_rows = []
            for target in sorted(targets_by_artifact.get(artifact, set())):
                target_rows.append(
                    {
                        "files": sorted(files_by_target.get(target, set())),
                        "target": target,
                    }
                )
            artifact_rows.append({"artifact": artifact, "targets": target_rows})
        provenance.append({"artifacts": artifact_rows, "kernel_mangled": kernel})

    python_frames = sum(
        any(
            ".py" in str(frame.get("symbol", ""))
            or "python" in str(frame.get("module", "")).lower()
            for frame in row["cuda_callchain"]
        )
        for row in rows
    )
    summary = {
        "cuda_callchain_rate": (
            sum(bool(row["cuda_callchain"]) for row in rows) / len(rows) if rows else 0
        ),
        "job_key": job_key,
        "kernel_launch_rows": len(rows),
        "python_callchain_hint_rate": python_frames / len(rows) if rows else 0,
        "static_join_rate": (
            sum(bool(row["artifacts"]) for row in rows) / len(rows) if rows else 0
        ),
        "test_attribution_rate": (
            sum(row["test_id"] is not None for row in rows) / len(rows) if rows else 0
        ),
        "unique_kernels": len(provenance),
    }
    return rows, provenance, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--kernel-map", type=Path)
    parser.add_argument("--build-graph", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    rows, provenance, summary = export_deep_trace(
        args.sqlite,
        job_key=args.job_key,
        kernel_map=args.kernel_map,
        build_graph=args.build_graph,
    )
    _atomic_jsonl(args.out, rows)
    _atomic_jsonl(args.provenance_out, provenance)
    _atomic_json(args.summary_out, summary)
    if not rows:
        raise SystemExit("deep trace contained no correlated CUDA kernel launches")
    if not any(row["cuda_callchain"] for row in rows):
        raise SystemExit("deep trace contained no CUDA launch callchains")
    if not any(row["test_id"] for row in rows):
        raise SystemExit("deep trace contained no test-attributed CUDA launches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
