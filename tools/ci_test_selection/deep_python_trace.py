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
from pathlib import Path
from types import FrameType
from typing import Any

_OUTPUT_ENV = "VLLM_CI_TEST_SELECTION_DEEP_TRACE_DIR"
_REPO_ROOT_ENV = "VLLM_CI_TEST_SELECTION_REPO_ROOT"
_STATE: _TraceState | None = None


def _repository_path(filename: str, repo_root: Path | None) -> str | None:
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


def _function(frame: FrameType) -> str:
    module = frame.f_globals.get("__name__")
    qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
    return f"{module}.{qualname}" if module else qualname


class _TraceState:
    def __init__(self, output_dir: Path, repo_root: Path | None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (output_dir / f"python-calls.{os.getpid()}.jsonl").open(
            "w", encoding="utf-8", buffering=1024 * 1024
        )
        self._repo_root = repo_root
        self._sequence = 0
        self._depths: dict[int, int] = {}
        self._lock = threading.Lock()
        self._closed = False

    def profile(self, frame: FrameType, event: str, _arg: Any) -> None:
        if event not in {"call", "return"}:
            return
        file = _repository_path(frame.f_code.co_filename, self._repo_root)
        if file is None:
            return

        thread_id = threading.get_native_id()
        with self._lock:
            depth = self._depths.get(thread_id, 0)
            if event == "return":
                depth = max(0, depth - 1)
                self._depths[thread_id] = depth

            caller = frame.f_back
            row = {
                "caller_file": (
                    _repository_path(caller.f_code.co_filename, self._repo_root)
                    if caller is not None
                    else None
                ),
                "caller_function": _function(caller) if caller is not None else None,
                "caller_line": caller.f_lineno if caller is not None else None,
                "depth": depth,
                "event": event,
                "file": file,
                "first_line": frame.f_code.co_firstlineno,
                "function": _function(frame),
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "sequence": self._sequence,
                "test_id": os.environ.get("PYTEST_CURRENT_TEST", "").removesuffix(
                    " (call)"
                ),
                "thread_id": thread_id,
            }
            self._sequence += 1
            self._stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            self._stream.write("\n")

            if event == "call":
                self._depths[thread_id] = depth + 1

    def close(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        with self._lock:
            if self._closed:
                return
            self._closed = True
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
