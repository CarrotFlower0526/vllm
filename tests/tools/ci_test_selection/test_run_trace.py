# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

from coverage import CoverageData

from tools.ci_test_selection.run_trace import (
    coverage_rows,
    normalize_repository_path,
    pytest_command,
    validate_import_environment,
)


def test_normalize_repository_path_from_checkout(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "vllm" / "attention" / "ops.py"

    assert normalize_repository_path(str(source), repo) == "vllm/attention/ops.py"


def test_normalize_repository_path_from_site_packages(tmp_path: Path):
    repo = tmp_path / "repo"
    source = tmp_path / "venv" / "site-packages" / "vllm" / "attention" / "ops.py"

    assert normalize_repository_path(str(source), repo) == "vllm/attention/ops.py"


def test_normalize_repository_path_rejects_non_vllm(tmp_path: Path):
    assert (
        normalize_repository_path(str(tmp_path / "torch" / "ops.py"), tmp_path) is None
    )


def test_coverage_rows_are_per_test_and_canonical(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "vllm" / "attention" / "ops.py"
    source.parent.mkdir(parents=True)
    source.write_text("one = 1\ntwo = 2\n", encoding="utf-8")
    coverage_file = tmp_path / ".coverage"
    data = CoverageData(basename=str(coverage_file))
    data.set_context("tests/kernels/test_ops.py::test_one|run")
    data.add_lines({str(source): {1, 2}})
    data.set_context("tests/kernels/test_ops.py::test_two|run")
    data.add_lines({str(source): {2}})
    data.write()

    rows = coverage_rows(
        coverage_file,
        repo,
        repository_sha="a" * 40,
        job_key="kernels-ops",
    )

    assert [(row["test_id"], row["line"]) for row in rows] == [
        ("tests/kernels/test_ops.py::test_one", 1),
        ("tests/kernels/test_ops.py::test_one", 2),
        ("tests/kernels/test_ops.py::test_two", 2),
    ]
    assert all(row["file"] == "vllm/attention/ops.py" for row in rows)


def test_pytest_command_loads_python_and_nvtx_plugins():
    command = pytest_command(["tests/kernels/test_ops.py"])

    assert "tools.ci_test_selection.pytest_trace_plugin" in command
    assert "tools.ci_test_selection.nvtx_test_ranges" in command
    assert command[-1] == "tests/kernels/test_ops.py"


def test_import_preflight_rejects_checkout_source_for_image_job(tmp_path: Path):
    checkout = tmp_path / "checkout"
    source_package = checkout / "vllm"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "import-environment.json"
    environment = {
        "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
        "PYTHONPATH": str(checkout),
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert document["error"] == "image job imported vllm from checkout source"
    assert document["vllm_file"] == str(source_package / "__init__.py")


def test_import_preflight_accepts_installed_package_outside_checkout(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    installed_package = tmp_path / "site-packages" / "vllm"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "import-environment.json"
    environment = {
        "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
        "PYTHONPATH": str(installed_package.parent),
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert document["error"] is None
    assert document["vllm_file"] == str(installed_package / "__init__.py")
