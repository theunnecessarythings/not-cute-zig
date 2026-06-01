# not-cute-zig

Small Zig/CUDA library and demo suite for building a simpler comptime layout and MMA API without pulling in CUTE or CUTLASS.

The reusable package entrypoint is `not-cute`, backed by `src/not_cute.zig`. It currently exposes:

- `layout`: comptime tensor layout, tiling, partitioning, and tensor views
- `mma`: descriptor-driven warp-level MMA backends
- `device`: device-side helpers and SM80+ primitives
- `cuda`: CUDA driver API wrappers for host allocation, module loading, launches, streams, events, and synchronization
- `benchmark`: event-based runtime benchmarking helpers

The current demo binary runs:

- vector add through the CUDA driver API
- shared-memory tiled matrix transpose
- CTA/warp/lane/value ownership mapping
- one `m16n8k16` f16 -> f32 MMA matmul tile using `ldmatrix` and `mma.sync`
- batched and pipelined MMA smoke checks with CPU references
- optional benchmark and experimental flash-attention demos

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

Run explicit non-smoke demos:

```sh
zig build run -- benchmark
zig build run -- flash
```

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
