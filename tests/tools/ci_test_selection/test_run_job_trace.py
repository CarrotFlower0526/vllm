# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from pathlib import Path

import pybase64 as base64
import pytest

from tools.ci_test_selection.run_job_trace import decode_commands


def _payload(commands: list[str]) -> str:
    return base64.b64encode(json.dumps(commands).encode()).decode()


def test_decode_commands_round_trip():
    commands = ["pytest -q tests/test_one.py", "python -m pytest tests/test_two.py"]

    assert decode_commands(_payload(commands)) == commands


def test_top_level_collector_package_avoids_tests_tools_shadowing():
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "tools")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ci_test_selection.run_trace as runner; "
                "print(runner.pytest_command([]))"
            ),
        ],
        cwd=project_root / "tests",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ci_test_selection.pytest_trace_plugin" in result.stdout
    assert "ci_test_selection.nvtx_test_ranges" in result.stdout


@pytest.mark.parametrize("document", [[], [""], {"command": "pytest"}, [1]])
def test_decode_commands_rejects_invalid_documents(document):
    with pytest.raises(SystemExit):
        decode_commands(base64.b64encode(json.dumps(document).encode()).decode())


def test_python_only_job_preserves_pytest_command_and_collects_trace(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "vllm" / "sample.py"
    test_file = repo / "tests" / "test_sample.py"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    (repo / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    test_file.write_text(
        "from vllm.sample import answer\n\n"
        "def test_answer():\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace"
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["BUILDKITE_COMMIT"] = "a" * 40
    environment["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci_test_selection.run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "ci-trace-unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest -q tests/test_sample.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
        ],
        cwd=repo,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    shard = output / "commands" / "000"
    trace_rows = [
        json.loads(line)
        for line in (shard / "python-trace.jsonl").read_text().splitlines()
    ]
    assert {row["file"] for row in trace_rows} == {"vllm/sample.py"}
    assert {row["test_id"] for row in trace_rows} == {
        "tests/test_sample.py::test_answer"
    }
    assert json.loads((shard / "job.json").read_text())["healthy"] is True
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["healthy"] is True
    assert summary["capture_mode"] == "python-only"
