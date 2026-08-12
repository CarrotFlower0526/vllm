#!/usr/bin/env bash
# Pilot trace step (MVP D3): run one command under nsys with CUDA+NVTX
# tracing, export sqlite, and emit kernel->test edges + join-rate table.
#
# This is ONE trace step, not a second pytest execution: the profiled
# command is the D2 runner (python -m ci_test_selection.run_trace),
# which loads the NVTX per-test plugin in its child pytest, so Python
# coverage, node/outcome health, and GPU edges come from the same run.
#
# Designed to run INSIDE the existing vLLM test image with no image change:
# nsys is installed in-step from an exactly pinned CLI-only deb (215 MB,
# depends only on libc6 + libglib2.0-0). Bake it into the image later only
# if tracing is promoted beyond the pilot.
#
# CUDA+NVTX tracing via CUPTI needs no elevated container capabilities
# (no CAP_SYS_ADMIN, no perf_event): CPU sampling and GPU performance
# counters stay disabled (--sample=none --cpuctxsw=none, no --gpu-metrics).
#
# Outputs in <output-dir>:
#   gpu-trace.jsonl          canonical kernel->test edges (materializer input)
#   join-rate-summary.json   join-rate table (diagnostic)
#   trace.nsys-rep, trace.sqlite   raw profiles (diagnostic)
#   trace-timings.json       nsys install seconds + traced wall seconds,
#                            for the traced-vs-production delta the pilot
#                            must report
#
# Usage: run_traced.sh <output-dir> <represented-job-key> <command...>
#   e.g. run_traced.sh /tmp/trace kernels-flashmla-test-h100 \
#        python3 -m ci_test_selection.run_trace ...
set -euo pipefail

NSYS_DEB_URL="https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2026.3.1.157-3804839.deb"
NSYS_DEB_SHA256="3eb87ec08e5f8b8f153537847747bd5cfabb51b9c8793873b26a3c55dc813ad1"

if (( $# < 3 )); then
    echo "usage: $0 <output-dir> <represented-job-key> <command...>" >&2
    exit 2
fi

OUT_DIR="$1"
REPRESENTED_JOB_KEY="$2"
shift 2
mkdir -p "$OUT_DIR"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_GRAPH_DIR="${BUILD_GRAPH_DIR:-/opt/vllm-ci/build-graph}"
KERNEL_MAP="${KERNEL_MAP:-$BUILD_GRAPH_DIR/kernel-map.jsonl}"
PUBLISH_BUILD_GRAPH="${PUBLISH_BUILD_GRAPH:-0}"
DEEP_TRACE="${VLLM_CI_TEST_SELECTION_DEEP_TRACE:-0}"

for name in build-graph.jsonl kernel-map.jsonl; do
    if [[ ! -s "$BUILD_GRAPH_DIR/$name" ]]; then
        echo "missing image-build provenance: $BUILD_GRAPH_DIR/$name" >&2
        exit 1
    fi
    if [[ "$PUBLISH_BUILD_GRAPH" == "1" ]]; then
        cp "$BUILD_GRAPH_DIR/$name" "$OUT_DIR/$name"
    fi
done

install_seconds=0
if ! command -v nsys >/dev/null; then
    install_start=$(date +%s)
    deb="$(mktemp /tmp/nsys-cli-XXXX.deb)"
    curl -fsSL -o "$deb" "$NSYS_DEB_URL"
    echo "$NSYS_DEB_SHA256  $deb" | sha256sum -c -
    apt-get update -qq && apt-get install -y -qq libglib2.0-0
    dpkg -i "$deb"
    rm -f "$deb"
    install_seconds=$(( $(date +%s) - install_start ))
fi
nsys --version

# --trace-fork-before-exec: the profiled command is the D2 runner, which
# spawns pytest as a child process; without it the child's CUDA activity
# is lost. The runner loads the NVTX plugin in that child.
traced_start=$(date +%s)
profile_options=(
    --trace=cuda,nvtx --sample=none --cpuctxsw=none
    --trace-fork-before-exec=true
)
if [[ "$DEEP_TRACE" == "1" ]]; then
    profile_options=(
        --trace=cuda,nvtx --sample=process-tree --cpuctxsw=none
        --backtrace=dwarf --cudabacktrace=kernel:0 --python-backtrace=cuda
        --pytorch=functions-trace,autograd-nvtx
        --trace-fork-before-exec=true
    )
fi
set +e
nsys profile \
    "${profile_options[@]}" \
    --output "$OUT_DIR/trace" --force-overwrite=true \
    -- "$@"
profile_status=$?
set -e
traced_seconds=$(( $(date +%s) - traced_start ))

parse_status=0
if [[ -s "$OUT_DIR/trace.nsys-rep" ]]; then
    set +e
    nsys export --type sqlite --include-json true \
        --output "$OUT_DIR/trace.sqlite" \
        --force-overwrite=true "$OUT_DIR/trace.nsys-rep"
    export_status=$?
    if (( export_status == 0 )); then
        python3 "$HERE/parse_nsys_sqlite.py" "$OUT_DIR/trace.sqlite" \
            --kernel-map "$KERNEL_MAP" \
            --job-key "$REPRESENTED_JOB_KEY" \
            --out "$OUT_DIR/gpu-trace.jsonl" \
            2> "$OUT_DIR/join-rate-summary.json"
        parse_status=$?
        if (( parse_status == 0 )) && [[ "$DEEP_TRACE" == "1" ]]; then
            python3 "$HERE/parse_deep_nsys_sqlite.py" "$OUT_DIR/trace.sqlite" \
                --kernel-map "$KERNEL_MAP" \
                --build-graph "$BUILD_GRAPH_DIR/build-graph.jsonl" \
                --job-key "$REPRESENTED_JOB_KEY" \
                --out "$OUT_DIR/deep-gpu-trace.jsonl" \
                --provenance-out "$OUT_DIR/deep-native-provenance.jsonl" \
                --summary-out "$OUT_DIR/deep-trace-summary.json"
            parse_status=$?
        fi
    else
        parse_status=$export_status
        printf '{"error":"nsys export failed","exit_code":%d}\n' \
            "$export_status" > "$OUT_DIR/join-rate-summary.json"
    fi
    set -e
else
    parse_status=1
    printf '{"error":"nsys profile produced no report","exit_code":%d}\n' \
        "$profile_status" > "$OUT_DIR/join-rate-summary.json"
fi

# Runtime edge rows are stamped here. Static build-graph/kernel-map files are
# immutable image-build artifacts and are referenced by SHA-256 from
# trace-job.json rather than copied into every test job.
if [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
    created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ -s "$OUT_DIR/gpu-trace.jsonl" ]]; then
        python3 "$HERE/stamp_jsonl.py" "$OUT_DIR/gpu-trace.jsonl" \
            --repository-sha "$BUILDKITE_COMMIT" \
            --created-at "$created_at"
    fi
fi

timings_tmp="$OUT_DIR/trace-timings.json.tmp"
printf '{"nsys_install_seconds":%d,"parse_exit_code":%d,"profile_exit_code":%d,"traced_wall_seconds":%d}\n' \
    "$install_seconds" "$parse_status" "$profile_status" "$traced_seconds" \
    > "$timings_tmp"
mv "$timings_tmp" "$OUT_DIR/trace-timings.json"

cat "$OUT_DIR/join-rate-summary.json" "$OUT_DIR/trace-timings.json"

if (( profile_status != 0 )); then
    exit "$profile_status"
fi
exit "$parse_status"
