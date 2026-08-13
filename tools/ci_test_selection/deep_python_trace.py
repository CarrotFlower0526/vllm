# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Record ordered repository Python call/return events for deep CI traces."""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from functools import cache
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

_OUTPUT_ENV = "VLLM_CI_TEST_SELECTION_DEEP_TRACE_DIR"
_REPO_ROOT_ENV = "VLLM_CI_TEST_SELECTION_REPO_ROOT"
_STATE: _TraceState | None = None


def _current_test_id() -> str:
    value = os.environ.get("PYTEST_CURRENT_TEST", "")
    for phase in ("setup", "call", "teardown"):
        suffix = f" ({phase})"
        if value.endswith(suffix):
            return value.removesuffix(suffix)
    return value


@cache
def _repository_path(filename: str, repo_root: Path | None) -> str | None:
    # CPython pseudo-filenames such as ``<frozen importlib._bootstrap>`` and
    # ``<string>`` resolve beneath the current working directory. Treating
    # them as repository paths produced millions of false trace events.
    if filename.startswith("<") and filename.endswith(">"):
        return None
    path = Path(filename).resolve()
    if repo_root is not None:
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            pass
        else:
            if relative.parts and relative.parts[0] in {"tests", "vllm"}:
                return relative.as_posix()

    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"tests", "vllm"}:
            candidate = Path(*parts[index:]).as_posix()
            if candidate.startswith(("tests/", "vllm/")):
                return candidate
    return None


@cache
def _function(code: CodeType, module: str | None) -> str:
    qualname = getattr(code, "co_qualname", code.co_name)
    return f"{module}.{qualname}" if module else qualname


class _TraceState:
    def __init__(self, output_dir: Path, repo_root: Path | None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (output_dir / f"python-calls.{os.getpid()}.jsonl").open(
            "w", encoding="utf-8", buffering=1024 * 1024
        )
        self._repo_root = repo_root
        self._pid = os.getpid()
        self._sequence = 0
        self._depths: dict[int, int] = {}
        self._buffer: list[str] = []
        self._buffer_limit = 4096
        self._lock = threading.Lock()
        self._closed = False

    def _metadata(self, frame: FrameType) -> tuple[str | None, str]:
        code = frame.f_code
        return (
            _repository_path(code.co_filename, self._repo_root),
            _function(code, frame.f_globals.get("__name__")),
        )

    def _flush(self) -> None:
        if self._buffer:
            self._stream.writelines(self._buffer)
            self._buffer.clear()

    def profile(self, frame: FrameType, event: str, _arg: Any) -> None:
        if event not in {"call", "return"}:
            return
        file, function = self._metadata(frame)
        if file is None:
            return

        thread_id = threading.get_native_id()
        with self._lock:
            depth = self._depths.get(thread_id, 0)
            if event == "return":
                depth = max(0, depth - 1)
                self._depths[thread_id] = depth

            caller = frame.f_back
            caller_file, caller_function = (
                self._metadata(caller) if caller is not None else (None, None)
            )
            row = {
                "caller_file": caller_file,
                "caller_function": caller_function,
                "caller_line": caller.f_lineno if caller is not None else None,
                "depth": depth,
                "event": event,
                "file": file,
                "first_line": frame.f_code.co_firstlineno,
                "function": function,
                "monotonic_ns": time.monotonic_ns(),
                "pid": self._pid,
                "sequence": self._sequence,
                "test_id": _current_test_id(),
                "thread_id": thread_id,
            }
            self._sequence += 1
            self._buffer.append(json.dumps(row, separators=(",", ":")) + "\n")
            if len(self._buffer) >= self._buffer_limit:
                self._flush()

            if event == "call":
                self._depths[thread_id] = depth + 1

    def close(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._flush()
            self._stream.close()


def install_from_environment() -> bool:
    """Install the profiler in an exec child when deep tracing is configured."""

    global _STATE
    if _STATE is not None:
        return True
    output_value = os.environ.get(_OUTPUT_ENV)
    if not output_value:
        return False
    repo_value = os.environ.get(_REPO_ROOT_ENV)
    repo_root = Path(repo_value).resolve() if repo_value else None
    _STATE = _TraceState(Path(output_value), repo_root)
    sys.setprofile(_STATE.profile)
    threading.setprofile(_STATE.profile)
    atexit.register(_STATE.close)
    return True
