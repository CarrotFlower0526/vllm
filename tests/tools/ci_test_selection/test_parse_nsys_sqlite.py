# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixture-based tests for the nsys sqlite parser's join logic.

Builds a synthetic database shaped like an nsys 2024+ export (StringIds,
NVTX_EVENTS, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_KERNEL) and
checks launch-site attribution: innermost range wins, setup-phase kernels
attribute to the setup range, async kernels executing after the range ended
still attribute to their launching test, and kernels launched outside every
range are counted, not misassigned. Real-schema drift is validated in the
pilot job, not here.
"""

import json
import pathlib
import sqlite3
import subprocess
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT_DIR = HERE.parents[2] / "tools/ci_test_selection"

K_INNER, K_OUTER, K_SETUP, K_ORPHAN, K_PROC2, K_TEMPORAL = (
    "_ZinnerK",
    "_ZouterK",
    "_ZsetupK",
    "_ZorphanK",
    "_Zproc2K",
    "_ZtemporalK",
)

# nsys serializes process identity in the high bits: PID key = value >> 24
PID1, PID2 = 5, 6
TID1 = (PID1 << 24) | 7
GPID1 = PID1 << 24


def build_fixture(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute(
        "CREATE TABLE NVTX_EVENTS (start INT, end INT,"
        " globalTid INT, text TEXT, textId INT)"
    )
    con.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT,"
        " end INT, correlationId INT, globalTid INT)"
    )
    con.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT,"
        " end INT, correlationId INT, mangledName INT,"
        " globalPid INT)"
    )
    strings = {
        1: K_INNER,
        2: K_OUTER,
        3: K_SETUP,
        4: K_ORPHAN,
        5: K_PROC2,
        6: K_TEMPORAL,
    }
    con.executemany("INSERT INTO StringIds VALUES (?,?)", strings.items())
    tid = TID1
    nvtx = [
        (100, 900, tid, "citest-setup::tests/a.py::test_x", None),
        (1000, 5000, tid, "citest::tests/a.py::test_x", None),
        # nested inner range inside test_x's call phase (e.g. subtest)
        (2000, 3000, tid, "citest::tests/a.py::test_x_inner", None),
        (6000, 9000, tid, "citest::tests/a.py::test_y", None),
        # unrelated NVTX range that must be ignored
        (0, 99999, tid, "torch-internal", None),
    ]
    con.executemany("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)", nvtx)
    # (launch start, correlationId) pairs; kernel exec times irrelevant
    runtime = [
        (150, 160, 11, tid),  # inside setup range
        (2500, 2510, 12, tid),  # inside BOTH test_x and inner -> inner wins
        (4900, 4910, 13, tid),  # in test_x; kernel executes after range end
        (5500, 5510, 14, tid),  # between tests -> outside any range
    ]
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)", runtime)
    kernels = [
        (200, 300, 11, 3, GPID1),  # K_SETUP
        (2600, 2700, 12, 1, GPID1),  # K_INNER
        (7000, 8000, 13, 2, GPID1),  # K_OUTER, executes in test_y's window
        (5600, 5700, 14, 4, GPID1),  # K_ORPHAN
    ]
    con.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)", kernels
    )
    con.commit()
    con.close()


class TestJoin(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.db = base / "trace.sqlite"
        build_fixture(self.db)
        self.kmap = base / "kmap.jsonl"
        rows = [
            {
                "source_kind": "artifact",
                "source": "elf-build-id:a",
                "edge_kind": "defines_kernel",
                "destination_kind": "kernel",
                "destination": K_INNER,
            },
            {
                "source_kind": "artifact",
                "source": "elf-build-id:a",
                "edge_kind": "defines_kernel",
                "destination_kind": "kernel",
                "destination": K_OUTER,
            },
            {
                "source_kind": "artifact",
                "source": "elf-build-id:b",
                "edge_kind": "defines_kernel",
                "destination_kind": "kernel",
                "destination": K_OUTER,
            },
        ]
        self.kmap.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.out = base / "edges.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_parser(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "parse_nsys_sqlite.py"),
                str(self.db),
                "--kernel-map",
                str(self.kmap),
                "--out",
                str(self.out),
                "--job-key",
                "kernels-flashmla-test-h100",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        edges = [json.loads(line) for line in self.out.read_text().splitlines()]
        return edges, json.loads(proc.stderr)

    def test_attribution_and_classification(self):
        edges, summary = self.run_parser()
        by_kernel = {e["source"]: e for e in edges}

        self.assertEqual(by_kernel[K_SETUP]["destination"], "tests/a.py::test_x")
        self.assertEqual(by_kernel[K_SETUP]["phase"], "citest-setup")

        # innermost covering range wins
        self.assertEqual(by_kernel[K_INNER]["destination"], "tests/a.py::test_x_inner")

        # attributed by launch site even though it EXECUTED in test_y's window
        self.assertEqual(by_kernel[K_OUTER]["destination"], "tests/a.py::test_x")

        # launched between tests: retained only at conservative job precision
        self.assertEqual(by_kernel[K_ORPHAN]["destination_kind"], "job")
        self.assertEqual(
            by_kernel[K_ORPHAN]["destination"], "kernels-flashmla-test-h100"
        )
        self.assertEqual(by_kernel[K_ORPHAN]["attribution_mode"], "job_union")
        self.assertEqual(summary["outside_any_test_range"], 1)

        self.assertEqual(by_kernel[K_INNER]["artifact_class"], "matched")
        self.assertEqual(by_kernel[K_OUTER]["artifact_class"], "ambiguous")
        self.assertEqual(by_kernel[K_SETUP]["artifact_class"], "unmapped")

        self.assertEqual(summary["kernel_rows"], 4)
        self.assertEqual(summary["kernel_name_column"], "mangledName")
        self.assertTrue(summary["mangled_identity"])
        self.assertEqual(summary["unique_kernel_destination_pairs"], 4)
        self.assertEqual(by_kernel[K_INNER]["job_key"], "kernels-flashmla-test-h100")
        self.assertEqual(by_kernel[K_INNER]["test_id"], "tests/a.py::test_x_inner")
        self.assertEqual(by_kernel[K_INNER]["attribution_mode"], "exact_test")

    def test_temporal_fallback_for_child_launch_thread(self):
        child_tid = (PID1 << 24) | 19
        con = sqlite3.connect(self.db)
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4000, 4010, 21, ?)",
            (child_tid,),
        )
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4100, 4200, 21, 6, ?)",
            (GPID1,),
        )
        con.commit()
        con.close()

        edges, summary = self.run_parser()
        temporal = next(edge for edge in edges if edge["source"] == K_TEMPORAL)
        self.assertEqual(temporal["destination"], "tests/a.py::test_x")
        self.assertEqual(temporal["attribution_mode"], "temporal_test")
        self.assertEqual(summary["attribution_temporal_test"], 1)

    def test_concurrent_ranges_fall_back_to_job_union(self):
        child_tid = (PID1 << 24) | 19
        concurrent_tid = (PID1 << 24) | 20
        con = sqlite3.connect(self.db)
        con.execute(
            "INSERT INTO NVTX_EVENTS VALUES"
            " (3500, 4500, ?, 'citest::tests/b.py::test_concurrent', NULL)",
            (concurrent_tid,),
        )
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4000, 4010, 21, ?)",
            (child_tid,),
        )
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4100, 4200, 21, 6, ?)",
            (GPID1,),
        )
        con.commit()
        con.close()

        edges, summary = self.run_parser()
        fallback = next(edge for edge in edges if edge["source"] == K_TEMPORAL)
        self.assertEqual(fallback["destination_kind"], "job")
        self.assertEqual(fallback["attribution_mode"], "job_union")
        self.assertEqual(summary["ambiguous_active_test_ranges"], 1)

    def test_dedup_repeated_launches(self):
        con = sqlite3.connect(self.db)
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2520, 2530, 15, ?)",
            (TID1,),
        )
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2800, 2900, 15, 1, ?)",
            (GPID1,),
        )
        con.commit()
        con.close()
        edges, summary = self.run_parser()
        inner = [e for e in edges if e["source"] == K_INNER]
        self.assertEqual(len(inner), 1)
        self.assertEqual(summary["kernel_rows"], 5)

    def test_duplicate_correlation_across_processes(self):
        """With fork tracing, a second process reuses correlationId 12.
        Its kernel must attribute via ITS OWN launch/test range, never the
        first process's."""
        tid2 = (PID2 << 24) | 9
        gpid2 = PID2 << 24
        con = sqlite3.connect(self.db)
        con.execute(
            "INSERT INTO NVTX_EVENTS VALUES"
            " (2000, 3000, ?, 'citest::tests/b.py::test_p2', NULL)",
            (tid2,),
        )
        # same correlationId=12 as process 1's K_INNER launch
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2400, 2410, 12, ?)",
            (tid2,),
        )
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2650, 2750, 12, 5, ?)",
            (gpid2,),
        )
        con.commit()
        con.close()
        edges, summary = self.run_parser()
        by_kernel = {e["source"]: e for e in edges}
        self.assertEqual(by_kernel[K_PROC2]["destination"], "tests/b.py::test_p2")
        self.assertEqual(by_kernel[K_INNER]["destination"], "tests/a.py::test_x_inner")
        self.assertTrue(summary["process_scoped_join"])

    def test_duplicate_correlation_in_one_process_does_not_publish(self):
        self.out.write_text("last-good\n")
        con = sqlite3.connect(self.db)
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2400, 2410, 12, ?)",
            (TID1,),
        )
        con.commit()
        con.close()
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "parse_nsys_sqlite.py"),
                str(self.db),
                "--kernel-map",
                str(self.kmap),
                "--out",
                str(self.out),
                "--job-key",
                "kernels-flashmla-test-h100",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(self.out.read_text(), "last-good\n")


if __name__ == "__main__":
    unittest.main()
