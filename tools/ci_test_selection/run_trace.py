"""Run a bounded pytest pilot and export per-test Python line coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coverage import CoverageData

from . import COLLECTOR_VERSION


def _atomic_json(path: Path, document: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repo_root: Path) -> str:
    configured = os.environ.get("BUILDKITE_COMMIT")
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def normalize_repository_path(filename: str, repo_root: Path) -> str | None:
    """Map source/install paths to a repository-relative ``vllm/...`` path."""

    path = Path(filename).resolve()
    try:
        relative = path.relative_to(repo_root.resolve())
    except ValueError:
        relative = None
    if relative is not None and relative.parts and relative.parts[0] == "vllm":
        return relative.as_posix()

    parts = path.parts
    for index, part in enumerate(parts):
        if part == "vllm":
            candidate = Path(*parts[index:]).as_posix()
            if candidate.startswith("vllm/"):
                return candidate
    return None


def _node_id(context: str) -> str | None:
    if not context or context == "":
        return None
    node_id, separator, phase = context.rpartition("|")
    if separator and phase in {"run", "setup", "teardown"}:
        return node_id
    return None


def coverage_rows(
    coverage_file: Path,
    repo_root: Path,
    *,
    repository_sha: str,
    job_key: str,
) -> list[dict[str, Any]]:
    data = CoverageData(basename=str(coverage_file))
    data.read()
    rows: set[tuple[str, str, int]] = set()
    for filename in data.measured_files():
        repository_path = normalize_repository_path(filename, repo_root)
        if repository_path is None:
            continue
        for line, contexts in data.contexts_by_lineno(filename).items():
            for context in contexts:
                node_id = _node_id(context)
                if node_id:
                    rows.add((node_id, repository_path, int(line)))

    return [
        {
            "collector_version": COLLECTOR_VERSION,
            "file": file,
            "job_key": job_key,
            "line": line,
            "repository_sha": repository_sha,
            "test_id": test_id,
        }
        for test_id, file, line in sorted(rows)
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--represented-job-key", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("tests", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    tests = args.tests[1:] if args.tests[:1] == ["--"] else args.tests
    if not tests:
        raise SystemExit("at least one pytest target is required after --")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_file = output_dir / ".coverage"
    node_file = output_dir / "pytest-nodes.json"
    trace_file = output_dir / "python-trace.jsonl"
    job_file = output_dir / "job.json"
    repository_sha = _git_sha(args.repo_root)

    environment = dict(os.environ)
    environment["COVERAGE_FILE"] = str(coverage_file)
    environment["VLLM_CI_TEST_SELECTION_NODEIDS"] = str(node_file)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "tools.ci_test_selection.pytest_trace_plugin",
        "--cov=vllm",
        "--cov-context=test",
        "--cov-report=",
        *tests,
    ]
    result = subprocess.run(
        command,
        cwd=args.repo_root,
        env=environment,
        check=False,
    )

    rows = (
        coverage_rows(
            coverage_file,
            args.repo_root,
            repository_sha=repository_sha,
            job_key=args.represented_job_key,
        )
        if coverage_file.exists()
        else []
    )
    _atomic_jsonl(trace_file, rows)
    node_document = (
        json.loads(node_file.read_text(encoding="utf-8"))
        if node_file.exists()
        else {"collected": [], "exit_status": result.returncode, "outcomes": {}}
    )
    healthy = (
        result.returncode == 0 and bool(node_document["collected"]) and bool(rows)
    )
    _atomic_json(
        job_file,
        {
            "child_process_attribution": False,
            "collector_version": COLLECTOR_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "healthy": healthy,
            "image_tag": os.environ.get("IMAGE_TAG"),
            "job_key": args.job_key,
            "node_ids": node_document["collected"],
            "pytest_exit_code": result.returncode,
            "python_trace": trace_file.name,
            "python_trace_sha256": _sha256(trace_file),
            "repository_sha": repository_sha,
            "represented_job_key": args.represented_job_key,
        },
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
