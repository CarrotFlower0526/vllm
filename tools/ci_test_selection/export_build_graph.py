#!/usr/bin/env python3
"""Build-graph exporter (MVP D1): file->target->artifact from CMake state.

Reads the CMake File API codemodel-v2 reply from an actual configured
build tree and emits JSONL edges in the frozen MVP edge shape:

    file -> target      edge_kind "member_of"   (target source membership)
    target -> target    edge_kind "required_by" (dependency -> dependent)
    target -> artifact  edge_kind "produces"    (built output path)

ELF artifact node identity is `elf-build-id:<GNU build-id>`. The build-relative
and wheel-relative paths are metadata because they name the same object
differently. This makes the cross-exporter join structural, not an assumption
that downstream ingestion will merge metadata.

Prerequisite in CI: the build must run with the File API query present
(write an empty file at <build>/.cmake/api/v1/query/codemodel-v2 before
configuring; vllm's setup.py cmake invocation picks it up transparently).

Known gap (documented, not silent): codemodel sources are the files CMake
knows about; transitively included headers need .d depfile augmentation,
which is out of MVP scope. Out-of-tree (generated) sources are counted in
the summary, not emitted as repo file nodes.

Usage:
    export_build_graph.py <build-dir> [--source-root <repo>] [--out edges.jsonl]
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from collections import defaultdict


def find_codemodel(reply_dir):
    indexes = sorted(reply_dir.glob("index-*.json"))
    if not indexes:
        raise SystemExit(
            f"no File API reply index under {reply_dir}; "
            "was the codemodel-v2 query present at configure "
            "time?"
        )
    index = json.loads(indexes[-1].read_text())
    for obj in index.get("objects", []):
        if obj.get("kind") == "codemodel":
            return json.loads((reply_dir / obj["jsonFile"]).read_text())
    raise SystemExit("File API reply has no codemodel object")


def gnu_build_id(path):
    out = subprocess.run(
        ["readelf", "-n", str(path)], capture_output=True, text=True
    ).stdout
    m = re.search(r"Build ID:\s*([0-9a-f]+)", out)
    return m.group(1) if m else None


def repo_relative(path, source_root):
    candidate = pathlib.Path(path)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    try:
        return candidate.resolve().relative_to(source_root).as_posix()
    except ValueError:
        return None


def backtrace_files(document, source_root):
    """Return repository CMake files reachable from target backtraces."""

    graph = document.get("backtraceGraph", {})
    nodes = graph.get("nodes", [])
    files = graph.get("files", [])
    roots = []
    for record in (
        [document]
        + document.get("sources", [])
        + document.get("dependencies", [])
        + document.get("artifacts", [])
        + document.get("compileGroups", [])
    ):
        if isinstance(record, dict) and isinstance(record.get("backtrace"), int):
            roots.append(record["backtrace"])

    seen = set()
    result = set()
    while roots:
        index = roots.pop()
        if index in seen or not 0 <= index < len(nodes):
            continue
        seen.add(index)
        node = nodes[index]
        file_index = node.get("file")
        if isinstance(file_index, int) and 0 <= file_index < len(files):
            relative = repo_relative(files[file_index], source_root)
            if relative is not None:
                result.add(relative)
        if isinstance(node.get("parent"), int):
            roots.append(node["parent"])
    return sorted(result)


def artifact_identity(path):
    build_id = gnu_build_id(path) if path.is_file() else None
    if build_id:
        return f"elf-build-id:{build_id}", build_id
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument(
        "--source-root",
        default=None,
        help="repo root for relativizing source paths "
        "(default: codemodel's paths.source)",
    )
    ap.add_argument("--out", default="-", help="edge JSONL output")
    args = ap.parse_args(argv)

    build_dir = pathlib.Path(args.build_dir).resolve()
    reply_dir = build_dir / ".cmake/api/v1/reply"
    codemodel = find_codemodel(reply_dir)
    source_root = pathlib.Path(
        args.source_root or codemodel["paths"]["source"]
    ).resolve()

    stats = defaultdict(int)
    seen = set()
    rows = []

    def emit(row):
        key = json.dumps(row, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        rows.append(key)

    id_to_name = {}
    target_docs = []
    for config in codemodel.get("configurations", []):
        for tref in config.get("targets", []):
            doc = json.loads((reply_dir / tref["jsonFile"]).read_text())
            target_docs.append(doc)
            id_to_name[doc.get("id", tref.get("id", doc["name"]))] = doc["name"]

    for doc in target_docs:
        name = doc["name"]
        stats["targets"] += 1
        for cmake_file in backtrace_files(doc, source_root):
            stats["cmake_file_edges"] += 1
            emit(
                {
                    "source_kind": "file",
                    "source": cmake_file,
                    "edge_kind": "configures",
                    "destination_kind": "target",
                    "destination": name,
                }
            )
        for src in doc.get("sources", []):
            rel = repo_relative(src["path"], source_root)
            if rel is None:
                stats["out_of_tree_sources"] += 1
                continue
            stats["file_edges"] += 1
            emit(
                {
                    "source_kind": "file",
                    "source": rel,
                    "edge_kind": "member_of",
                    "destination_kind": "target",
                    "destination": name,
                }
            )
        for dep in doc.get("dependencies", []):
            dep_name = id_to_name.get(
                dep.get("id", ""), dep.get("id", "").split("::@")[0]
            )
            if dep_name and dep_name != name:
                stats["target_dep_edges"] += 1
                emit(
                    {
                        "source_kind": "target",
                        "source": dep_name,
                        "edge_kind": "required_by",
                        "destination_kind": "target",
                        "destination": name,
                    }
                )
        for artifact in doc.get("artifacts", []):
            apath = artifact["path"]
            on_disk = (
                build_dir / apath
                if not pathlib.PurePosixPath(apath).is_absolute()
                else pathlib.Path(apath)
            )
            identity, build_id = artifact_identity(on_disk)
            if identity is None:
                if on_disk.is_file():
                    stats["artifacts_without_build_id"] += 1
                else:
                    stats["artifacts_missing_on_disk"] += 1
                # Static libraries and missing optional outputs cannot join to
                # runtime kernels, so they are diagnostics rather than graph
                # artifact nodes in this MVP.
                continue
            stats["artifact_edges"] += 1
            row = {
                "source_kind": "target",
                "source": name,
                "edge_kind": "produces",
                "destination_kind": "artifact",
                "destination": identity,
                "artifact_build_id": build_id,
                "artifact_path": apath,
            }
            if not on_disk.is_file():
                stats["artifacts_missing_on_disk"] += 1
            emit(row)

    rows.sort()
    if args.out == "-":
        for row in rows:
            print(row)
    else:
        output = pathlib.Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            for row in rows:
                stream.write(row + "\n")
        temporary.replace(output)
    print(json.dumps(dict(stats), indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
