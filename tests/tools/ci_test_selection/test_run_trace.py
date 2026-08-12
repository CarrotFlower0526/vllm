from pathlib import Path

from coverage import CoverageData

from tools.ci_test_selection.run_trace import (
    coverage_rows,
    normalize_repository_path,
    pytest_command,
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
