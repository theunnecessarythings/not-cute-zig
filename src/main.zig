const std = @import("std");
const cuda = @import("cuda.zig");
const layout = @import("layout.zig");
const config = @import("config.zig");
const benchmark = @import("benchmark.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const demo_list = "vector-add, transpose, ownership, mma, streams, reduction, batched-mma, pipeline, epilogue, occupancy, benchmark, flash, all";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var args = try std.process.argsWithAllocator(alloc);
    defer args.deinit();

    _ = args.skip(); // skip exe name
    const cmd = args.next() orelse {
        std.log.err("Usage: not-cute-zig <demo_name>", .{});
        std.log.err("Available demos: {s}", .{demo_list});
        return error.MissingCommand;
    };

    try cuda.init();
    defer cuda.deinit();

    const module = try cuda.Module.loadData(@embedFile("cuda-module"));
    defer module.unload();

    if (std.mem.eql(u8, cmd, "vector-add")) {
        try runVectorAdd(module);
    } else if (std.mem.eql(u8, cmd, "transpose")) {
        try runMatrixTranspose(module);
    } else if (std.mem.eql(u8, cmd, "ownership")) {
        try runOwnershipDebug(module);
    } else if (std.mem.eql(u8, cmd, "mma")) {
        try runMmaMatmul(module);
    } else if (std.mem.eql(u8, cmd, "streams")) {
        try runStreamsDemo(module);
    } else if (std.mem.eql(u8, cmd, "reduction")) {
        try runReductionDemo(module);
    } else if (std.mem.eql(u8, cmd, "batched-mma")) {
        try runBatchedMma(module);
    } else if (std.mem.eql(u8, cmd, "pipeline")) {
        try runPipelinedMma(module);
    } else if (std.mem.eql(u8, cmd, "epilogue")) {
        try runEpilogueDemo(module);
    } else if (std.mem.eql(u8, cmd, "flash")) {
        try runFlashAttention(module);
    } else if (std.mem.eql(u8, cmd, "occupancy")) {
        try runOccupancyDemo();
    } else if (std.mem.eql(u8, cmd, "benchmark")) {
        try runBenchmarkDemo(module);
    } else if (std.mem.eql(u8, cmd, "all")) {
        try runVectorAdd(module);
        try runMatrixTranspose(module);
        try runOwnershipDebug(module);
        try runMmaMatmul(module);
        try runStreamsDemo(module);
        try runReductionDemo(module);
        try runBatchedMma(module);
        try runPipelinedMma(module);
        try runEpilogueDemo(module);
        try runOccupancyDemo();
    } else {
        std.log.err("Unknown demo: {s}", .{cmd});
        return error.UnknownCommand;
    }
}

fn runOccupancyDemo() !void {
    const dev = try cuda.getDevice();
    const calc = try cuda.OccupancyCalculator.init(dev);
    calc.printInfo();

    const block_size = 256;
    const max_blocks_per_sm = @divTrunc(calc.max_threads_per_sm, block_size);
    const total_concurrent_blocks = max_blocks_per_sm * calc.sm_count;

    std.log.info("--- Occupancy Math ---", .{});
    std.log.info("For a kernel with block size {}:", .{block_size});
    std.log.info("  Max Concurrent Blocks/SM: {}", .{max_blocks_per_sm});
    std.log.info("  Total Concurrent Blocks: {}", .{total_concurrent_blocks});
    std.log.info("  (Ideal grid size should be a multiple of {})", .{total_concurrent_blocks});
}

fn runFlashAttention(module: cuda.Module) !void {
    const seq_len = 16;
    const head_dim = 16;
    const len = seq_len * head_dim;

    var q: [len]f16 = undefined;
    var k: [len]f16 = undefined;
    var v: [len]f16 = undefined;
    var o: [len]f32 = undefined;

    for (0..len) |i| {
        q[i] = @floatFromInt(i % 5);
        k[i] = @floatFromInt(i % 3);
        v[i] = @floatFromInt(i % 2);
    }

    const d_q = try cuda.malloc(f16, len);
    defer cuda.free(d_q);

    const d_k = try cuda.malloc(f16, len);
    defer cuda.free(d_k);

    const d_v = try cuda.malloc(f16, len);
    defer cuda.free(d_v);

    const d_o = try cuda.malloc(f32, len);
    defer cuda.free(d_o);

    try cuda.memcpy(f16, d_q, &q, .host_to_device);
    try cuda.memcpy(f16, d_k, &k, .host_to_device);
    try cuda.memcpy(f16, d_v, &v, .host_to_device);

    const kernel = try module.getFunction("flash_attention_fwd");
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    try kernel.launch(
        .{
            .grid_dim = .{ .x = 1 },
            .block_dim = .{ .x = 32 },
        },
        .{ d_q.ptr, d_k.ptr, d_v.ptr, d_o.ptr, seq_len, head_dim, scale },
    );

    try cuda.memcpy(f32, &o, d_o, .device_to_host);

    std.log.warn("flash attention is experimental: ran simplified 16x16 forward pass without CPU reference validation", .{});
}

fn runEpilogueDemo(module: cuda.Module) !void {
    const k_iters = 1;
    const a_len = config.mma_m * config.mma_k * k_iters;
    const b_len = config.mma_k * config.mma_n * k_iters;
    const c_len = config.mma_m * config.mma_n;

    var a: [a_len]f16 = undefined;
    var b: [b_len]f16 = undefined;
    var c: [c_len]f32 = undefined;
    var d: [c_len]f32 = undefined;
    var expected: [c_len]f32 = undefined;

    fillPipelineInputs(k_iters, &a, &b, &c);
    computePipelineReference(k_iters, &a, &b, &c, &expected, 1.0, 0.0, .relu);

    const d_a = try cuda.malloc(f16, a_len);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f16, b_len);
    defer cuda.free(d_b);

    const d_c = try cuda.malloc(f32, c_len);
    defer cuda.free(d_c);

    const d_d = try cuda.malloc(f32, c_len);
    defer cuda.free(d_d);

    try cuda.memcpy(f16, d_a, &a, .host_to_device);
    try cuda.memcpy(f16, d_b, &b, .host_to_device);
    try cuda.memcpy(f32, d_c, &c, .host_to_device);

    const kernel = try module.getFunction("pipelined_mma_matmul");

    try kernel.launch(
        .{
            .grid_dim = .{ .x = 1, .y = 1 },
            .block_dim = .{ .x = 32 },
        },
        .{ d_a.ptr, d_b.ptr, d_c.ptr, d_d.ptr, @as(f32, 1.0), @as(f32, 0.0), a_len, b_len, c_len, k_iters, @as(u32, 1) },
    );

    try cuda.memcpy(f32, &d, d_d, .device_to_host);

    for (d, expected, 0..) |actual, want, i| {
        const diff = @abs(actual - want);
        if (diff > 0.001) {
            std.log.err("epilogue mismatch at {}: expected {}, got {}, diff {}", .{ i, want, actual, diff });
            return error.EpilogueMismatch;
        }
    }

    std.log.info("epilogue demo OK: ReLU fused successfully", .{});
}

fn runPipelinedMma(module: cuda.Module) !void {
    const k_iters = 4;
    const a_len = config.mma_m * config.mma_k * k_iters;
    const b_len = config.mma_k * config.mma_n * k_iters;
    const c_len = config.mma_m * config.mma_n; // C/D are fixed per batch tile

    var a: [a_len]f16 = undefined;
    var b: [b_len]f16 = undefined;
    var c: [c_len]f32 = undefined;
    var d: [c_len]f32 = undefined;
    var expected: [c_len]f32 = undefined;

    fillPipelineInputs(k_iters, &a, &b, &c);
    computePipelineReference(k_iters, &a, &b, &c, &expected, 1.0, 0.0, .none);

    const d_a = try cuda.malloc(f16, a_len);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f16, b_len);
    defer cuda.free(d_b);

    const d_c = try cuda.malloc(f32, c_len);
    defer cuda.free(d_c);

    const d_d = try cuda.malloc(f32, c_len);
    defer cuda.free(d_d);

    try cuda.memcpy(f16, d_a, &a, .host_to_device);
    try cuda.memcpy(f16, d_b, &b, .host_to_device);
    try cuda.memcpy(f32, d_c, &c, .host_to_device);

    const kernel = try module.getFunction("pipelined_mma_matmul");
    try kernel.launch(
        .{
            .grid_dim = .{ .x = 1, .y = 1 }, // 1 block, 1 batch
            .block_dim = .{ .x = 32 },
        },
        .{ d_a.ptr, d_b.ptr, d_c.ptr, d_d.ptr, @as(f32, 1.0), @as(f32, 0.0), a_len, b_len, c_len, k_iters, @as(u32, 0) },
    );

    try cuda.memcpy(f32, &d, d_d, .device_to_host);

    for (d, expected, 0..) |actual, want, i| {
        const diff = @abs(actual - want);
        if (diff > 0.001) {
            std.log.err("pipelined mma mismatch at {}: expected {}, got {}, diff {}", .{ i, want, actual, diff });
            return error.PipelinedMmaMismatch;
        }
    }

    std.log.info("pipelined mma OK: {} k-iterations of {}x{}x{}", .{ k_iters, config.mma_m, config.mma_n, config.mma_k });
}

fn runBatchedMma(module: cuda.Module) !void {
    const num_batches = 4;
    const a_len = config.mma_m * config.mma_k;
    const b_len = config.mma_k * config.mma_n;
    const c_len = config.mma_m * config.mma_n;

    var a: [a_len * num_batches]f16 = undefined;
    var b: [b_len * num_batches]f16 = undefined;
    var c: [c_len * num_batches]f32 = undefined;
    var d: [c_len * num_batches]f32 = undefined;
    var expected: [c_len * num_batches]f32 = undefined;

    for (0..num_batches) |batch| {
        fillMmaInputs(
            a[batch * a_len ..][0..a_len],
            b[batch * b_len ..][0..b_len],
            c[batch * c_len ..][0..c_len],
            @intCast(batch),
        );
        computeMmaReference(
            a[batch * a_len ..][0..a_len],
            b[batch * b_len ..][0..b_len],
            c[batch * c_len ..][0..c_len],
            expected[batch * c_len ..][0..c_len],
            1.0,
            0.0,
        );
    }

    const d_a = try cuda.malloc(f16, a_len * num_batches);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f16, b_len * num_batches);
    defer cuda.free(d_b);

    const d_c = try cuda.malloc(f32, c_len * num_batches);
    defer cuda.free(d_c);

    const d_d = try cuda.malloc(f32, c_len * num_batches);
    defer cuda.free(d_d);

    try cuda.memcpy(f16, d_a, &a, .host_to_device);
    try cuda.memcpy(f16, d_b, &b, .host_to_device);
    try cuda.memcpy(f32, d_c, &c, .host_to_device);

    const kernel = try module.getFunction("batched_mma_matmul");
    try kernel.launch(
        .{
            .grid_dim = .{ .x = 1, .y = num_batches },
            .block_dim = .{ .x = 32 },
        },
        .{ d_a.ptr, d_b.ptr, d_c.ptr, d_d.ptr, @as(f32, 1.0), @as(f32, 0.0), a_len, b_len, c_len },
    );

    try cuda.memcpy(f32, &d, d_d, .device_to_host);

    for (d, expected, 0..) |actual, want, i| {
        const diff = @abs(actual - want);
        if (diff > 0.001) {
            std.log.err("batched mma mismatch at {}: expected {}, got {}, diff {}", .{ i, want, actual, diff });
            return error.BatchedMmaMismatch;
        }
    }

    std.log.info("batched mma OK: {} batches of {}x{}x{}", .{ num_batches, config.mma_m, config.mma_n, config.mma_k });
}

fn runStreamsDemo(module: cuda.Module) !void {
    const num_streams = 4;
    var streams: [num_streams]cuda.Stream = undefined;
    var d_outs: [num_streams][]u32 = undefined;

    for (0..num_streams) |i| {
        streams[i] = try cuda.Stream.create();
        d_outs[i] = try cuda.malloc(u32, 1);
    }

    const kernel = try module.getFunction("stream_sleep");

    std.log.info("Launching {} streams...", .{num_streams});
    for (0..num_streams) |i| {
        try kernel.launch(.{
            .grid_dim = .{ .x = 1 },
            .block_dim = .{ .x = 32 },
            .stream = streams[i].handle,
        }, .{
            @as(u32, 500_000_000), // sleep ~0.5s per stream
            d_outs[i].ptr,
        });
    }

    for (0..num_streams) |i| {
        try streams[i].synchronize();
        streams[i].destroy();
        var out: [1]u32 = .{0};
        try cuda.memcpy(u32, &out, d_outs[i], .device_to_host);
        cuda.free(d_outs[i]);
        if (out[0] != 1) return error.StreamDemoFailed;
    }

    std.log.info("streams demo OK: concurrent sleep execution", .{});
}

fn runReductionDemo(module: cuda.Module) !void {
    const len = 1000;
    var input: [len]f32 = undefined;
    var expected_sum: f32 = 0;

    for (0..len) |i| {
        input[i] = @floatFromInt(i % 10);
        expected_sum += input[i];
    }

    const d_in = try cuda.malloc(f32, len);
    defer cuda.free(d_in);

    const block_size = 256;
    const num_blocks = (len + block_size - 1) / block_size;
    const d_out = try cuda.malloc(f32, num_blocks);
    defer cuda.free(d_out);

    try cuda.memcpy(f32, d_in, &input, .host_to_device);

    const kernel = try module.getFunction("block_reduce_sum");
    try kernel.launch(
        .{
            .grid_dim = .{ .x = num_blocks },
            .block_dim = .{ .x = block_size },
        },
        .{ d_in.ptr, d_out.ptr, len },
    );

    const block_sums = try std.heap.page_allocator.alloc(f32, num_blocks);
    defer std.heap.page_allocator.free(block_sums);
    try cuda.memcpy(f32, block_sums, d_out, .device_to_host);

    var actual_sum: f32 = 0;
    for (0..num_blocks) |i| {
        actual_sum += block_sums[i];
    }

    if (@abs(expected_sum - actual_sum) > 0.1) {
        std.log.err("reduction mismatch: expected {}, got {}", .{ expected_sum, actual_sum });
        return error.ReductionMismatch;
    }

    std.log.info("reduction demo OK: block reduce sum of {} elements", .{len});
}

fn runVectorAdd(module: cuda.Module) !void {
    var a: [config.vector_len]f32 = undefined;
    var b: [config.vector_len]f32 = undefined;
    var out: [config.vector_len]f32 = undefined;

    for (&a, &b, 0..) |*a_item, *b_item, i| {
        a_item.* = @floatFromInt(i);
        b_item.* = @floatFromInt(i * 2);
    }

    const d_a = try cuda.malloc(f32, config.vector_len);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f32, config.vector_len);
    defer cuda.free(d_b);

    const d_out = try cuda.malloc(f32, config.vector_len);
    defer cuda.free(d_out);

    try cuda.memcpy(f32, d_a, &a, .host_to_device);
    try cuda.memcpy(f32, d_b, &b, .host_to_device);

    const kernel = try module.getFunction("vector_add");
    try kernel.launch(
        .{
            .grid_dim = .{ .x = (config.vector_len + config.vector_block_size - 1) / config.vector_block_size },
            .block_dim = .{ .x = config.vector_block_size },
        },
        .{ d_a.ptr, d_b.ptr, d_out.ptr, config.vector_len },
    );

    try cuda.memcpy(f32, &out, d_out, .device_to_host);

    for (out, 0..) |actual, i| {
        const expected = a[i] + b[i];
        if (actual != expected) {
            std.log.err("mismatch at {}: expected {}, got {}", .{ i, expected, actual });
            return error.VectorAddMismatch;
        }
    }

    std.log.info("vector add OK: {} elements", .{config.vector_len});
    std.log.info("{} + {} = {}", .{ a[7], b[7], out[7] });
}

fn runMatrixTranspose(module: cuda.Module) !void {
    const input_layout = layout.rowMajor(.{ config.transpose_rows, config.transpose_cols });
    const output_layout = layout.rowMajor(.{ config.transpose_cols, config.transpose_rows });
    const len = config.transpose_rows * config.transpose_cols;

    var input: [len]f32 = undefined;
    var output: [len]f32 = undefined;

    for (&input, 0..) |*item, i| {
        item.* = @floatFromInt(i);
    }

    const d_input = try cuda.malloc(f32, len);
    defer cuda.free(d_input);

    const d_output = try cuda.malloc(f32, len);
    defer cuda.free(d_output);

    try cuda.memcpy(f32, d_input, &input, .host_to_device);

    const kernel = try module.getFunction("matrix_transpose");
    try kernel.launch(
        .{
            .grid_dim = .{
                .x = (config.transpose_cols + config.transpose_tile - 1) / config.transpose_tile,
                .y = (config.transpose_rows + config.transpose_tile - 1) / config.transpose_tile,
            },
            .block_dim = .{ .x = config.transpose_tile, .y = config.transpose_tile },
        },
        .{ d_input.ptr, d_output.ptr, config.transpose_rows, config.transpose_cols },
    );

    try cuda.memcpy(f32, &output, d_output, .device_to_host);

    for (0..config.transpose_rows) |row| {
        for (0..config.transpose_cols) |col| {
            const expected = input[input_layout.offset(.{ row, col })];
            const actual = output[output_layout.offset(.{ col, row })];
            if (actual != expected) {
                std.log.err("transpose mismatch at ({}, {}): expected {}, got {}", .{ row, col, expected, actual });
                return error.MatrixTransposeMismatch;
            }
        }
    }

    std.log.info("matrix transpose OK: {}x{} -> {}x{}", .{
        config.transpose_rows,
        config.transpose_cols,
        config.transpose_cols,
        config.transpose_rows,
    });
}

fn runOwnershipDebug(module: cuda.Module) !void {
    const matrix = layout.rowMajor(.{ config.ownership_m, config.ownership_n });
    const cta = matrix.partition(.{ config.ownership_m, config.ownership_n });
    const warp = cta.partition(.{ 32, 64 });
    const lane = warp.partition(.{ 1, config.ownership_values_per_lane_owner });
    const value = lane.partition(.{ 1, 1 });
    const len = config.ownership_m * config.ownership_n;

    var output: [len]u32 = undefined;

    const d_output = try cuda.malloc(u32, len);
    defer cuda.free(d_output);

    const kernel = try module.getFunction("ownership_debug");
    try kernel.launch(
        .{
            .grid_dim = .{ .x = @intCast(warp.ownerExtent()[0]) },
            .block_dim = .{ .x = config.ownership_warp_threads },
        },
        .{d_output.ptr},
    );

    try cuda.memcpy(u32, &output, d_output, .device_to_host);

    for (0..warp.ownerExtent()[0]) |warp_id| {
        for (0..config.ownership_warp_threads) |lane_id| {
            for (0..lane.ownerExtent()[1]) |strip_group| {
                for (0..config.ownership_values_per_lane_owner) |value_col| {
                    const coord = cta.coord(
                        .{ 0, 0 },
                        warp.coord(
                            .{ warp_id, 0 },
                            lane.coord(
                                .{ lane_id, strip_group },
                                value.coord(.{ 0, value_col }, .{ 0, 0 }),
                            ),
                        ),
                    );
                    const expected = encodeOwnership(
                        @intCast(warp_id),
                        @intCast(lane_id),
                        @intCast(strip_group * config.ownership_values_per_lane_owner + value_col),
                    );
                    const actual = output[matrix.offset(coord)];
                    if (actual != expected) {
                        std.log.err("ownership mismatch at ({}, {}): expected {}, got {}", .{
                            coord[0],
                            coord[1],
                            expected,
                            actual,
                        });
                        return error.OwnershipDebugMismatch;
                    }
                }
            }
        }
    }

    std.log.info("ownership debug OK: {}x{} CTA tile", .{ config.ownership_m, config.ownership_n });
}

fn encodeOwnership(warp_id: u32, lane_owner: u32, value_id: u32) u32 {
    return (warp_id << 24) | (lane_owner << 8) | value_id;
}

fn runBenchmarkDemo(module: cuda.Module) !void {
    const len = 10_000_000;
    const bytes = len * @sizeOf(f32) * 3; // 2 reads, 1 write

    var a: [1]f32 = .{1.0};
    const d_a = try cuda.malloc(f32, len);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f32, len);
    defer cuda.free(d_b);

    const d_out = try cuda.malloc(f32, len);
    defer cuda.free(d_out);

    // Initialization (just copy a 1 to first element to avoid empty allocations)
    try cuda.memcpy(f32, d_a[0..1], &a, .host_to_device);
    try cuda.memcpy(f32, d_b[0..1], &a, .host_to_device);

    const kernel = try module.getFunction("vector_add");
    const args = .{ d_a.ptr, d_b.ptr, d_out.ptr, len };

    std.log.info("Starting runtime parameter sweep for vector_add (len={})...", .{len});

    // Sweep block sizes
    const configs = [_]cuda.LaunchConfig{
        .{ .grid_dim = .{ .x = @intCast((len + 31) / 32) }, .block_dim = .{ .x = 32 } },
        .{ .grid_dim = .{ .x = @intCast((len + 63) / 64) }, .block_dim = .{ .x = 64 } },
        .{ .grid_dim = .{ .x = @intCast((len + 127) / 128) }, .block_dim = .{ .x = 128 } },
        .{ .grid_dim = .{ .x = @intCast((len + 255) / 256) }, .block_dim = .{ .x = 256 } },
        .{ .grid_dim = .{ .x = @intCast((len + 511) / 512) }, .block_dim = .{ .x = 512 } },
    };

    const best_cfg = try benchmark.tuneRuntime(.{
        .warmup_iters = 2,
        .iters = 5,
    }, &configs, kernel, args);

    std.log.info("Best config found: block_dim.x={}", .{best_cfg.block_dim.x});

    std.log.info("Running full benchmark on best config...", .{});
    const res = try benchmark.runKernel(.{
        .warmup_iters = 5,
        .iters = 20,
        .bytes_processed = bytes,
        .flops_processed = len, // 1 add per element
    }, kernel, best_cfg, args);

    res.print("vector_add (Best Config)");
}

fn runMmaMatmul(module: cuda.Module) !void {
    const c_layout = layout.rowMajor(.{ config.mma_m, config.mma_n });
    const a_len = config.mma_m * config.mma_k;
    const b_len = config.mma_k * config.mma_n;
    const c_len = config.mma_m * config.mma_n;
    const alpha: f32 = 1.25;
    const beta: f32 = 0.5;

    var a: [a_len]f16 = undefined;
    var b: [b_len]f16 = undefined;
    var c: [c_len]f32 = undefined;
    var d: [c_len]f32 = undefined;
    var expected: [c_len]f32 = undefined;

    fillMmaInputs(&a, &b, &c, 0);
    computeMmaReference(&a, &b, &c, &expected, alpha, beta);

    const d_a = try cuda.malloc(f16, a_len);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f16, b_len);
    defer cuda.free(d_b);

    const d_c = try cuda.malloc(f32, c_len);
    defer cuda.free(d_c);

    const d_d = try cuda.malloc(f32, c_len);
    defer cuda.free(d_d);

    try cuda.memcpy(f16, d_a, &a, .host_to_device);
    try cuda.memcpy(f16, d_b, &b, .host_to_device);
    try cuda.memcpy(f32, d_c, &c, .host_to_device);

    const kernel = try module.getFunction("mma_matmul");
    try kernel.launch(
        .{
            .grid_dim = .{ .x = 1 },
            .block_dim = .{ .x = 32 },
        },
        .{ d_a.ptr, d_b.ptr, d_c.ptr, d_d.ptr, alpha, beta },
    );

    try cuda.memcpy(f32, &d, d_d, .device_to_host);

    for (0..config.mma_m) |row| {
        for (0..config.mma_n) |col| {
            const want = expected[c_layout.offset(.{ row, col })];
            const actual = d[c_layout.offset(.{ row, col })];
            const diff = @abs(want - actual);
            if (diff > 0.001) {
                std.log.err("mma mismatch at ({}, {}): expected {}, got {}, diff {}", .{
                    row,
                    col,
                    want,
                    actual,
                    diff,
                });
                return error.MmaMatmulMismatch;
            }
        }
    }

    std.log.info("mma matmul OK: {}x{}x{}", .{ config.mma_m, config.mma_n, config.mma_k });
}

const Epilogue = enum {
    none,
    relu,
};

fn fillMmaInputs(a: []f16, b: []f16, c: []f32, seed: u32) void {
    const a_layout = layout.rowMajor(.{ config.mma_m, config.mma_k });
    const b_layout = layout.colMajor(.{ config.mma_k, config.mma_n });
    const c_layout = layout.rowMajor(.{ config.mma_m, config.mma_n });

    for (0..config.mma_m) |row| {
        for (0..config.mma_k) |col| {
            const value: f32 = @floatFromInt((row + col + seed) % 5);
            a[a_layout.offset(.{ row, col })] = @floatCast(value * 0.25);
        }
    }

    for (0..config.mma_k) |row| {
        for (0..config.mma_n) |col| {
            const value: f32 = @floatFromInt((row * 2 + col + seed) % 7);
            b[b_layout.offset(.{ row, col })] = @floatCast(value * 0.125);
        }
    }

    for (0..config.mma_m) |row| {
        for (0..config.mma_n) |col| {
            const value: f32 = @floatFromInt((row + col * 3 + seed) % 11);
            c[c_layout.offset(.{ row, col })] = value * 0.03125;
        }
    }
}

fn computeMmaReference(a: []const f16, b: []const f16, c: []const f32, d: []f32, alpha: f32, beta: f32) void {
    const a_layout = layout.rowMajor(.{ config.mma_m, config.mma_k });
    const b_layout = layout.colMajor(.{ config.mma_k, config.mma_n });
    const c_layout = layout.rowMajor(.{ config.mma_m, config.mma_n });

    for (0..config.mma_m) |row| {
        for (0..config.mma_n) |col| {
            var acc: f32 = 0;
            for (0..config.mma_k) |kk| {
                acc += @as(f32, @floatCast(a[a_layout.offset(.{ row, kk })])) *
                    @as(f32, @floatCast(b[b_layout.offset(.{ kk, col })]));
            }
            d[c_layout.offset(.{ row, col })] = alpha * acc + beta * c[c_layout.offset(.{ row, col })];
        }
    }
}

fn fillPipelineInputs(comptime k_iters: usize, a: *[config.mma_m * config.mma_k * k_iters]f16, b: *[config.mma_k * config.mma_n * k_iters]f16, c: *[config.mma_m * config.mma_n]f32) void {
    const a_tile_len = config.mma_m * config.mma_k;
    const b_tile_len = config.mma_k * config.mma_n;
    const c_len = config.mma_m * config.mma_n;

    for (0..k_iters) |tile| {
        for (0..a_tile_len) |i| {
            const value: f32 = @floatFromInt((i + tile) % 5);
            a[tile * a_tile_len + i] = @floatCast(value * 0.125);
        }
        for (0..b_tile_len) |i| {
            const value: f32 = @floatFromInt((i * 3 + tile) % 7);
            b[tile * b_tile_len + i] = @floatCast(value * 0.0625);
        }
    }

    for (0..c_len) |i| {
        c[i] = @as(f32, @floatFromInt(i % 13)) * 0.03125;
    }
}

fn computePipelineReference(comptime k_iters: usize, a: *const [config.mma_m * config.mma_k * k_iters]f16, b: *const [config.mma_k * config.mma_n * k_iters]f16, c: *const [config.mma_m * config.mma_n]f32, d: *[config.mma_m * config.mma_n]f32, alpha: f32, beta: f32, epilogue: Epilogue) void {
    const c_layout = layout.rowMajor(.{ config.mma_m, config.mma_n });
    const a_tile_len = config.mma_m * config.mma_k;
    const b_tile_len = config.mma_k * config.mma_n;

    for (0..config.mma_m) |row| {
        for (0..config.mma_n) |col| {
            var acc: f32 = 0;
            for (0..k_iters) |tile| {
                for (0..config.mma_k) |kk| {
                    const a_idx = tile * a_tile_len + row * config.mma_k + kk;
                    const b_idx = tile * b_tile_len + kk * config.mma_n + col;
                    acc += @as(f32, @floatCast(a[a_idx])) * @as(f32, @floatCast(b[b_idx]));
                }
            }

            var out = alpha * acc + beta * c[c_layout.offset(.{ row, col })];
            if (epilogue == .relu) out = @max(0.0, out);
            d[c_layout.offset(.{ row, col })] = out;
        }
    }
}
