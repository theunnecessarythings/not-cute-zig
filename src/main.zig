const std = @import("std");
const cuda = @import("cuda.zig");
const layout = @import("layout.zig");
const config = @import("config.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main() !void {
    cuda.init();

    const module = try cuda.Module.loadData(@embedFile("cuda-module"));
    defer module.unload();

    try runVectorAdd(module);
    try runMatrixTranspose(module);
    try runOwnershipDebug(module);
    try runMmaMatmul(module);
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

    cuda.memcpy(f32, d_a, &a, .host_to_device);
    cuda.memcpy(f32, d_b, &b, .host_to_device);

    const kernel = try module.getFunction("vector_add");
    kernel.launch(
        .{
            .grid_dim = .{ .x = (config.vector_len + config.vector_block_size - 1) / config.vector_block_size },
            .block_dim = .{ .x = config.vector_block_size },
        },
        .{ d_a.ptr, d_b.ptr, d_out.ptr, config.vector_len },
    );

    cuda.memcpy(f32, &out, d_out, .device_to_host);

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

    cuda.memcpy(f32, d_input, &input, .host_to_device);

    const kernel = try module.getFunction("matrix_transpose");
    kernel.launch(
        .{
            .grid_dim = .{
                .x = (config.transpose_cols + config.transpose_tile - 1) / config.transpose_tile,
                .y = (config.transpose_rows + config.transpose_tile - 1) / config.transpose_tile,
            },
            .block_dim = .{ .x = config.transpose_tile, .y = config.transpose_tile },
        },
        .{ d_input.ptr, d_output.ptr, config.transpose_rows, config.transpose_cols },
    );

    cuda.memcpy(f32, &output, d_output, .device_to_host);

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
    kernel.launch(
        .{
            .grid_dim = .{ .x = @intCast(warp.ownerExtent()[0]) },
            .block_dim = .{ .x = config.ownership_warp_threads },
        },
        .{d_output.ptr},
    );

    cuda.memcpy(u32, &output, d_output, .device_to_host);

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

fn runMmaMatmul(module: cuda.Module) !void {
    const a_layout = layout.rowMajor(.{ config.mma_m, config.mma_k });
    const b_layout = layout.colMajor(.{ config.mma_k, config.mma_n });
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

    for (0..config.mma_m) |row| {
        for (0..config.mma_k) |col| {
            const value: f32 = @floatFromInt((row + col) % 5);
            a[a_layout.offset(.{ row, col })] = @floatCast(value * 0.25);
        }
    }

    for (0..config.mma_k) |row| {
        for (0..config.mma_n) |col| {
            const value: f32 = @floatFromInt((row * 2 + col) % 7);
            b[b_layout.offset(.{ row, col })] = @floatCast(value * 0.125);
        }
    }

    for (0..config.mma_m) |row| {
        for (0..config.mma_n) |col| {
            const value: f32 = @floatFromInt((row + col * 3) % 11);
            c[c_layout.offset(.{ row, col })] = value * 0.03125;
        }
    }

    const d_a = try cuda.malloc(f16, a_len);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f16, b_len);
    defer cuda.free(d_b);

    const d_c = try cuda.malloc(f32, c_len);
    defer cuda.free(d_c);

    const d_d = try cuda.malloc(f32, c_len);
    defer cuda.free(d_d);

    cuda.memcpy(f16, d_a, &a, .host_to_device);
    cuda.memcpy(f16, d_b, &b, .host_to_device);
    cuda.memcpy(f32, d_c, &c, .host_to_device);

    const kernel = try module.getFunction("mma_matmul");
    kernel.launch(
        .{
            .grid_dim = .{ .x = 1 },
            .block_dim = .{ .x = 32 },
        },
        .{ d_a.ptr, d_b.ptr, d_c.ptr, d_d.ptr, alpha, beta },
    );

    cuda.memcpy(f32, &d, d_d, .device_to_host);

    for (0..config.mma_m) |row| {
        for (0..config.mma_n) |col| {
            var acc: f32 = 0;
            for (0..config.mma_k) |kk| {
                acc += @as(f32, @floatCast(a[a_layout.offset(.{ row, kk })])) *
                    @as(f32, @floatCast(b[b_layout.offset(.{ kk, col })]));
            }

            const expected = alpha * acc + beta * c[c_layout.offset(.{ row, col })];
            const actual = d[c_layout.offset(.{ row, col })];
            const diff = @abs(expected - actual);
            if (diff > 0.001) {
                std.log.err("mma mismatch at ({}, {}): expected {}, got {}, diff {}", .{
                    row,
                    col,
                    expected,
                    actual,
                    diff,
                });
                return error.MmaMatmulMismatch;
            }
        }
    }

    std.log.info("mma matmul OK: {}x{}x{}", .{ config.mma_m, config.mma_n, config.mma_k });
}
