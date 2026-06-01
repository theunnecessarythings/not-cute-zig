# Contributing

This project targets Zig 0.15.2 and CUDA-capable NVIDIA GPUs.

Before sending changes, run:

```sh
zig fmt --check build.zig src/*.zig
zig build test
zig build -Dgpu=sm_80
zig build -Dgpu=sm_86
zig build -Dgpu=sm_89
```

Runtime smoke checks require a GPU:

```sh
zig build -Dgpu=sm_86
zig build run -- all
```

`benchmark` and `flash` are explicit demos. They are not part of the deterministic `all` smoke path.

## Public API

The supported package entrypoint is `src/not_cute.zig`, imported by downstream projects as `not-cute`.
Modules exported from that file are treated as public API. CLI demos and experimental kernels are examples unless they are also surfaced through that entrypoint.
