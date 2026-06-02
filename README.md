# not-cute-zig

Small Zig/CUDA library and demo suite for building a simpler comptime layout and MMA API without pulling in CUTE or CUTLASS.

The reusable package entrypoint is `not-cute`, backed by `src/not_cute.zig`. It currently exposes:

- `layout`: comptime tensor layout, tiling, partitioning, and tensor views
- `mma`: descriptor-driven warp-level MMA backends
- `device`: device-side helpers and SM80+ primitives
- `cuda`: CUDA driver API wrappers for host allocation, module loading, launches, streams, events, and synchronization
- `flash`: host-side flash-attention launch options and validation for supported shapes
- `benchmark`: event-based runtime benchmarking helpers

The current demo binary runs:

- vector add through the CUDA driver API
- shared-memory tiled matrix transpose
- CTA/warp/lane/value ownership mapping
- one `m16n8k16` f16 -> f32 MMA matmul tile using `ldmatrix` and `mma.sync`
- batched and pipelined MMA smoke checks with CPU references
- multi-block flash-attention smoke check with a CPU reference
- optional benchmark demo

## Requirements

- Zig 0.15.2
- CUDA 12 runtime and driver
- NVIDIA GPU target compatible with the selected `-Dgpu` value
- SM80 or newer for the current MMA and `cp.async` paths

Supported compile targets are currently `sm_80`, `sm_86`, and `sm_89`.

## Package Usage

In a downstream `build.zig`, depend on this package and import the module as `not-cute`. The public API is the module set exported from `src/not_cute.zig`.

```zig
const not_cute = @import("not-cute");
const layout = not_cute.layout;
const mma = not_cute.mma;
```

## Build

Build for an Ampere GPU such as Modal A10G:

```sh
zig build -Dgpu=sm_86
```

If CUDA is not installed under `/opt/cuda`, pass the include and library directories:

```sh
zig build -Dgpu=sm_86 -Dcuda-include-dir=/usr/local/cuda/include -Dcuda-lib-dir=/usr/local/cuda/lib64
```

Run unit tests for the layout and MMA descriptor layers:

```sh
zig build test
```

Run deterministic GPU smoke demos:

```sh
zig build run -- all
```

The current flash-attention kernel computes single-head self-attention for row-major
`Q/K/V/O` tensors with explicit element strides per flattened batch/head slice.
It uses query CTAs, shared-memory K/V tiles, tensor cores for QK and PV,
lane-parallel row max/sum reductions, causal masking, and online softmax state
across K/V tiles. The flash tile shape is currently `block_m=16`, `block_n=16`
with two warps per CTA, using two `m16n8k16` MMA subtiles per K/V tile. The current implementation supports
`head_dim == 16`, `32`, or `64`; the smoke test validates padded causal/non-causal slices, and
`benchmark` includes causal/non-causal flash timing runs for `seq_len` 32, 64,
and 128.

Run explicit non-smoke demos:

```sh
zig build run -- benchmark
```

Run CUDA Toolkit baselines against not-cute on Modal:

```sh
zig build -Dgpu=sm_86
modal run modal_run.py --demo compare
```

`compare` prints JSONL records and a side-by-side summary table for not-cute and
CUDA Toolkit baselines. Current baselines include CUDA C++ vector/transpose
kernels, tiled and WMMA tensor-core CUDA flash-attention baselines with online
softmax, a materialized cuBLAS attention path (`QK^T`, softmax, `PV`), CUB
reduction, cuBLAS for the MMA/GEMM-shaped case, and PyTorch SDPA backends
(`flash`, memory-efficient, and math where available). The flash sweep includes
smoke shapes and larger supported workloads up to
`batch_heads=8, seq_len=1024, head_dim=64`.

Profile the focused large flash case with Nsight Compute on Modal:

```sh
zig build -Dgpu=sm_86
modal run modal_run.py --demo profile-flash
```

`profile-flash` runs `ncu` against only `flash_attention_fwd` for
`batch_heads=8, seq_len=1024, head_dim=64, causal=true`, collecting speed-of-light,
occupancy, scheduler, warp-state, and memory workload sections. Some hosted GPU
runtimes restrict profiler injection; if `ncu` fails, the Modal wrapper prints
the profiler error and runs the focused flash benchmark without profiling.

## Modal

This machine does not need a local GPU. Build the binary locally, then run it on Modal:

```sh
zig build -Dgpu=sm_86
modal run modal_run.py
```

`modal_run.py` mounts `zig-out/bin/not-cute-zig` into an A10G runtime container and executes it.

## Production Checks

Local checks:

```sh
zig fmt --check build.zig src/*.zig
zig build test
zig build -Dgpu=sm_80
zig build -Dgpu=sm_86
zig build -Dgpu=sm_89
```

CI runs formatting, unit tests, and the GPU compile matrix. Runtime GPU validation is intentionally separate because it requires CUDA hardware.

## Layout API

`src/layout.zig` contains the core comptime layout layer:

- `rowMajor`, `colMajor`, and custom `strided` layouts
- rank-N tuple coordinates
- nested tiling and residue-aware owner views
- `tensor(ptr, layout).get/set`
- generic `partition` helpers for CTA, warp, lane, and value ownership
- a simple XOR swizzle hook

## MMA API

`src/mma.zig` keeps architecture-specific tensor-core details out of the layout core.

The API is descriptor-driven, with compile-time backend dispatch. The first supported backend is:

```text
m16n8k16.row.col.f32.f16.f16.f32
```

It uses:

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for A
- `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` for B
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

Unsupported descriptors fail at compile time.
