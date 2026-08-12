#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stamp published JSONL edge rows with wrapper-owned provenance fields.

The raw exporters stay checkout-agnostic; this Buildkite-wrapper step adds
`repository_sha` and `created_at` (UTC RFC3339) to every row, giving the
materializer an auditable freshness bound. Rewrites in place atomically.

Usage: stamp_jsonl.py <file.jsonl> --repository-sha <40hex> \
           --created-at 2026-08-12T10:00:00Z
"""

import argparse
import json
import os
import sys
import tempfile

import regex as re


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--repository-sha", required=True)
    ap.add_argument(
        "--created-at",
        required=True,
        help="UTC RFC3339 timestamp, e.g. from $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)",
    )
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.repository_sha):
        print(
            f"stamp_jsonl: not a 40-hex sha: {args.repository_sha!r}", file=sys.stderr
        )
        return 2
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args.created_at):
        print(f"stamp_jsonl: not UTC RFC3339: {args.created_at!r}", file=sys.stderr)
        return 2

    directory = os.path.dirname(os.path.abspath(args.path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with open(args.path) as src, os.fdopen(fd, "w") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["repository_sha"] = args.repository_sha
                row["created_at"] = args.created_at
                dst.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp, args.path)
    except BaseException:
        os.unlink(tmp)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
