#!/usr/bin/env bash
#
# One-shot benchmark collection for the CUDA Kernel Academy repo.
#
# Runs:
#   1. 01-sgemm-tutorial standalone benchmark (all standard sizes)
#   2. 02-tensorcraft-core gemm benchmark (Google Benchmark)
#
# Output is meant to be copied straight into docs/en/benchmarks/.
# LD_PRELOAD is only needed on WSL2; it is set automatically when detected.

set -euo pipefail

# Resolve repo root from this script's location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# WSL2 exposes the host GPU driver through /usr/lib/wsl/lib/libcuda.so.1.
if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
    export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1
fi

echo "== GPU info =="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
echo
echo "== CUDA info =="
nvcc --version | tail -2
echo
echo "== Build =="
cmake --preset default
cmake --build --preset default -j
echo
echo "== 01 benchmark =="
(cd "${REPO_ROOT}/01-sgemm-tutorial" && make benchmark)
"${REPO_ROOT}/01-sgemm-tutorial/build/sgemm_benchmark" -a
echo
echo "== 02 gemm benchmark =="
"${REPO_ROOT}/build/default/bin/gemm_benchmark" \
    --benchmark_min_time=0.1s --benchmark_repetitions=3
