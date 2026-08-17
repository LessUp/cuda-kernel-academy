---
outline: [2, 3]
---

# Profiling the SGEMM Tutorial with Nsight Tools

This guide is a practical runbook for profiling the SGEMM optimization ladder in
`01-sgemm-tutorial` with NVIDIA Nsight Systems (`nsys`) and Nsight Compute
(`ncu`). It is written to be copy-pasteable: every command below runs against
the already-built `sgemm_benchmark` binary.

> **Environment caveat (WSL2):** on WSL2, the host GPU driver blocks performance
> counters, so `ncu` typically exits with `ERR_NVGPUCTRPERM` (Insufficient
> Permissions). `nsys` timing traces usually still work. Run the `ncu` steps on
> bare-metal Linux or a cloud GPU instance to get real counter data. When you
> have such a machine, paste the `ncu` report output here.

## Prerequisites

```bash
# Build the benchmark (already done in CI / `make benchmark`)
cd 01-sgemm-tutorial
make benchmark

# WSL2 only: expose the host driver to CUDA tools
export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1
```

## 1. Timeline trace with Nsight Systems

`nsys` answers the *system-level* question: where does time go (kernel launch
overhead vs GPU execution vs memory copies)?

```bash
nsys profile --stats=true -o /tmp/sgemm -f true ./build/sgemm_benchmark -s 1024
```

What to read from the `--stats=true` summary:

- **`Time (%)` / `Total Time (ns)` columns of `CUDA GPU Kernel`** — share of
  wall time spent in each kernel. For a pure-GEMM benchmark, `sgemm_*` kernels
  should dominate; anything else (e.g. `memcpy` Device-to-Host) is overhead.
- **`Instance Count`** — how many times each kernel ran (sanity check your grid
  configuration).
- **`Avg (ns)` per kernel** — the per-call latency that drives your TFLOPS
  number.
- **Gap between the end of one kernel and the start of the next** (visible in
  the trace in Nsight Systems GUI): large gaps mean launch/CPU-bound behavior,
  not kernel slowness.

For a GUI deep-dive, open the report:

```bash
nsys stats --report=cuda_gpu_kern_sum --format=csv /tmp/sgemm.nsys-rep
```

## 2. Kernel-level metrics with Nsight Compute

`ncu` answers the *kernel-level* question: what limits this kernel (memory
bandwidth, instruction issue, occupancy, stalls)?

```bash
# Profile a single launch of the tiled kernel
ncu --set basic --kernel-name regex:tiled_sgemm_kernel --launch-count 1 \
    ./build/sgemm_benchmark -s 1024
```

> Replace `tiled_sgemm_kernel` with the actual kernel name in your build
> (`--kernel-name regex:.*` lists them all). The `--set basic` section is the
> fastest useful profile; `--set full` gives the complete metric set.

How to read the key metrics:

- **`Achieved Occupancy`** (`SM Occupancy` section, 0–100%)
  - Fraction of the theoretical maximum warps resident per SM.
  - Too low (< ~25%) usually means the kernel is register/shared-memory bound
    and you are not hiding latency — check `Registers Per Thread` and
    `Shared Memory Per Block`.
  - Too high is not automatically good; high occupancy can hurt cache reuse.
- **`Memory Workload Analysis`** (memory throughput vs `Compute Workload
  Analysis`)
  - If **memory throughput** is close to the theoretical peak of the GPU and
    much higher than compute throughput, the kernel is **memory-bound**
    (the classic naive SGEMM failure mode: each thread re-reads operands from
    global memory).
  - If **compute throughput** dominates, the kernel is **compute-bound**; look
    at instruction mix and FLOP efficiency next.
- **`Scheduler Stats` / `Warp State Statistics`** (`Stall` reasons)
  - `Stall Long Scoreboard`: waiting on global memory — sign of missing/too
    small tiling or no shared-memory reuse.
  - `Stall Wait` / `Stall Barrier`: `__syncthreads()` overhead — check tile
    balance and block size.
  - `Stall Short Scoreboard`: waiting on shared memory — check bank conflicts
    (see `Shared Memory` conflict counters).
  - `Stall Not Selected`: SM has other warps ready to run — usually benign.
- **`L1/TEX` and `L2` hit rates** — how well the tiling reuses data. A tiled
  kernel should show high L1 hits for the shared-memory-managed tiles.

## 3. Turning findings into TFLOPS numbers

The benchmark prints TFLOPS directly. After profiling, map metric readings back
to the optimization ladder:

| Symptom in `ncu` | Likely cause | Fix in the ladder |
|---|---|---|
| Memory throughput ≈ peak, low compute | Global-memory-bound | Use shared-memory tiling (`sgemm-tiled`) |
| Bank conflicts (`Shared Memory` section) | Stride-1 shared access pattern | Pad shared tiles / reorder access (`sgemm-tiled`, `sgemm-double-buffer`) |
| Low occupancy, high registers/thread | Too much per-thread state | Smaller tile / `__launch_bounds__` |
| `Stall Long Scoreboard` even when tiled | Not enough loads in flight | Double buffering / more threads per tile |
| Low L2 reuse | Blocks don't share data | Increase tile size / block ordering |

## 4. One-shot collection for the docs

To record the current machine's numbers for `docs/en/benchmarks/`:

```bash
# TFLOPS table
./build/sgemm_benchmark -a

# GPU + toolchain identity (paste at the top of the benchmark doc)
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
nvcc --version | tail -2
```

## Known limitation on this machine

WSL2 reports `ERR_NVGPUCTRPERM` for `ncu` because the guest OS cannot access
the host GPU's performance counters. `nsys` timeline tracing is unaffected.
A real screenshot-backed walkthrough requires bare-metal Linux or a cloud GPU;
once available, append the `ncu` report and annotate the metrics above with the
actual numbers.
