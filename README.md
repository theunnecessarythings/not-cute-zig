# not-cute-zig

Small Zig/CUDA experiments for building a simpler comptime layout and MMA API without pulling in CUTE or CUTLASS.

The current demo binary runs:

- vector add through the CUDA driver API
- shared-memory tiled matrix transpose
- CTA/warp/lane/value ownership mapping
- one `m16n8k16` f16 -> f32 MMA matmul tile using `ldmatrix` and `mma.sync`

## Build

Build for an Ampere GPU such as Modal A10G:

```sh
zig build -Dgpu=sm_86
```

Run unit tests for the layout and MMA descriptor layers:

```sh
zig build test
```

## Modal

This machine does not need a local GPU. Build the binary locally, then run it on Modal:

```sh
zig build -Dgpu=sm_86
modal run modal_run.py
```

`modal_run.py` mounts `zig-out/bin/not-cute-zig` into an A10G runtime container and executes it.

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
