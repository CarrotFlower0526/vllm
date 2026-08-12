"""Run an enrolled Buildkite pytest job under generic trace collection."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import COLLECTOR_VERSION


def _atomic_json(path: Path, document: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def decode_commands(value: str) -> list[str]:
    try:
        document = json.loads(base64.b64decode(value, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = "--commands-base64 must contain base64-encoded JSON"
        raise SystemExit(message) from error
    if not isinstance(document, list) or not document:
        raise SystemExit("trace command payload must be a non-empty JSON list")
    if not all(isinstance(command, str) and command.strip() for command in document):
        raise SystemExit("every trace command must be a non-empty string")
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--represented-job-key", required=True)
    parser.add_argument("--commands-base64", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/vllm-workspace"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-gpu", action="store_true")
    mode.add_argument("--python-only", action="store_true")
    return parser


def _encoded_command(command: str) -> str:
    return base64.b64encode(command.encode("utf-8")).decode("ascii")


def _run_command(
    command: str,
    *,
    command_index: int,
    job_key: str,
    command_cwd: Path,
    output_dir: Path,
    repo_root: Path,
    represented_job_key: str,
    capture_gpu: bool,
) -> subprocess.CompletedProcess[Any]:
    shard_dir = output_dir / "commands" / f"{command_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    runner = [
        sys.executable,
        "-m",
        "tools.ci_test_selection.run_trace",
        "--output-dir",
        str(shard_dir),
        "--job-key",
        job_key,
        "--represented-job-key",
        represented_job_key,
        "--repo-root",
        str(repo_root),
        "--command-cwd",
        str(command_cwd),
        "--command-base64",
        _encoded_command(command),
    ]
    environment = dict(os.environ)
    if capture_gpu:
        wrapper = Path(__file__).with_name("run_traced.sh")
        environment["PUBLISH_BUILD_GRAPH"] = "1" if command_index == 0 else "0"
        runner = [
            str(wrapper),
            str(shard_dir),
            represented_job_key,
            *runner,
        ]
    return subprocess.run(runner, cwd=command_cwd, env=environment, check=False)


def _job_document(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _upload_artifacts(pattern: str) -> int | None:
    if not os.environ.get("BUILDKITE"):
        return None
    if not shutil.which("buildkite-agent"):
        return 127
    result = subprocess.run(
        ["buildkite-agent", "artifact", "upload", pattern],
        check=False,
    )
    return result.returncode


def main() -> int:
    args = _parser().parse_args()
    commands = decode_commands(args.commands_base64)
    artifact_pattern = str(args.output_dir / "**/*")
    output_dir = args.output_dir.resolve()
    repo_root = args.repo_root.resolve()
    command_cwd = Path.cwd().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command_results = []
    return_code = 0
    for index, command in enumerate(commands):
        result = _run_command(
            command,
            command_index=index,
            job_key=args.job_key,
            command_cwd=command_cwd,
            output_dir=output_dir,
            repo_root=repo_root,
            represented_job_key=args.represented_job_key,
            capture_gpu=args.capture_gpu,
        )
        shard_job = _job_document(output_dir / "commands" / f"{index:03d}" / "job.json")
        command_results.append(
            {
                "command_index": index,
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "exit_code": result.returncode,
                "healthy": bool(shard_job and shard_job.get("healthy") is True),
            }
        )
        if result.returncode != 0:
            return_code = result.returncode
            break

    healthy = len(command_results) == len(commands) and all(
        result["exit_code"] == 0 and result["healthy"] for result in command_results
    )
    summary_path = output_dir / "trace-job.json"
    _atomic_json(
        summary_path,
        {
            "capture_mode": "gpu" if args.capture_gpu else "python-only",
            "collector_version": COLLECTOR_VERSION,
            "command_count": len(commands),
            "command_results": command_results,
            "created_at": datetime.now(UTC).isoformat(),
            "healthy": healthy,
            "job_key": args.job_key,
            "represented_job_key": args.represented_job_key,
        },
    )
    upload_status = _upload_artifacts(artifact_pattern)
    if not healthy and return_code == 0:
        return_code = 1
    if upload_status not in (None, 0) and return_code == 0:
        return_code = upload_status
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
