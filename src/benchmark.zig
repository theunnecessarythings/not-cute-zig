const std = @import("std");
const cuda = @import("cuda.zig");
const c = cuda.c;

/// The result of a benchmark run.
pub const BenchmarkResult = struct {
    mean_ms: f32,
    min_ms: f32,
    max_ms: f32,
    throughput_gbps: ?f32 = null,
    tflops: ?f32 = null,

    pub fn print(self: BenchmarkResult, name: []const u8) void {
        std.log.info("--- Benchmark: {s} ---", .{name});
        std.log.info("  Time: mean={d:.3}ms, min={d:.3}ms, max={d:.3}ms", .{ self.mean_ms, self.min_ms, self.max_ms });
        if (self.throughput_gbps) |bw| {
            std.log.info("  Bandwidth: {d:.2} GB/s", .{bw});
        }
        if (self.tflops) |flops| {
            std.log.info("  Compute:   {d:.2} TFLOPs", .{flops});
        }
    }
};

/// Options for running a benchmark.
pub const BenchmarkOptions = struct {
    warmup_iters: usize = 3,
    iters: usize = 10,
    bytes_processed: ?usize = null,
    flops_processed: ?usize = null,
    stream: ?c.CUstream = null,
};

/// Benchmarks a compiled CUDA function with a specific launch configuration.
pub fn runKernel(
    opts: BenchmarkOptions,
    kernel: cuda.Function,
    cfg: cuda.LaunchConfig,
    args: anytype,
) !BenchmarkResult {
    var start = try cuda.Event.create();
    defer start.destroy();
    var stop = try cuda.Event.create();
    defer stop.destroy();

    // Warmup
    for (0..opts.warmup_iters) |_| {
        try kernel.launch(cfg, args);
    }

    // Ensure warmup is completely done before timing
    if (cfg.stream) |s| {
        try cuda.check(c.cuStreamSynchronize(s));
    } else {
        try cuda.check(c.cuCtxSynchronize());
    }

    var total_ms: f32 = 0;
    var min_ms: f32 = std.math.inf(f32);
    var max_ms: f32 = -std.math.inf(f32);

    for (0..opts.iters) |_| {
        try start.record(cfg.stream);
        try kernel.launch(cfg, args);
        try stop.record(cfg.stream);
        try stop.synchronize();

        const ms = try cuda.Event.elapsed(start, stop);
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    const mean_ms = total_ms / @as(f32, @floatFromInt(opts.iters));

    var result = BenchmarkResult{
        .mean_ms = mean_ms,
        .min_ms = min_ms,
        .max_ms = max_ms,
    };

    if (opts.bytes_processed) |bytes| {
        // Bandwidth calculation: (bytes / 1e9) / (mean_ms / 1000) = (bytes / 1e6) / mean_ms
        result.throughput_gbps = @as(f32, @floatFromInt(bytes)) / (mean_ms * 1_000_000.0);
    }
    if (opts.flops_processed) |flops| {
        // FLOPs calculation: (flops / 1e12) / (mean_ms / 1000) = (flops / 1e9) / mean_ms
        result.tflops = @as(f32, @floatFromInt(flops)) / (mean_ms * 1_000_000_000.0);
    }

    return result;
}

/// Sweeps over a slice of runtime LaunchConfigs for a given kernel and returns the fastest one.
pub fn tuneRuntime(
    opts: BenchmarkOptions,
    configs: []const cuda.LaunchConfig,
    kernel: cuda.Function,
    args: anytype,
) !cuda.LaunchConfig {
    if (configs.len == 0) return error.NoConfigsProvided;

    var best_cfg: cuda.LaunchConfig = configs[0];
    var best_time: f32 = std.math.inf(f32);

    for (configs) |cfg| {
        const res = try runKernel(opts, kernel, cfg, args);
        std.log.info("Tune iter: config grid=({},{},{}) block=({},{},{}) -> mean {d:.3}ms", .{
            cfg.grid_dim.x,  cfg.grid_dim.y,  cfg.grid_dim.z,
            cfg.block_dim.x, cfg.block_dim.y, cfg.block_dim.z,
            res.mean_ms,
        });
        if (res.mean_ms < best_time) {
            best_time = res.mean_ms;
            best_cfg = cfg;
        }
    }

    return best_cfg;
}

/// A wrapper for a kernel that was generated at comptime.
pub const ComptimeKernel = struct {
    name: [:0]const u8,
    function: cuda.Function,
};

/// Sweeps over multiple compiled kernel variations and returns the fastest one.
/// The `kernels` slice should be populated by querying the `cuda.Module` for each generated name.
pub fn tuneComptime(
    opts: BenchmarkOptions,
    kernels: []const ComptimeKernel,
    cfg: cuda.LaunchConfig,
    args: anytype,
) !ComptimeKernel {
    if (kernels.len == 0) return error.NoConfigsProvided;

    var best_kernel: ComptimeKernel = kernels[0];
    var best_time: f32 = std.math.inf(f32);

    for (kernels) |k| {
        const res = try runKernel(opts, k.function, cfg, args);
        std.log.info("Tune iter: kernel {s} -> mean {d:.3}ms", .{ k.name, res.mean_ms });
        if (res.mean_ms < best_time) {
            best_time = res.mean_ms;
            best_kernel = k;
        }
    }

    return best_kernel;
}
