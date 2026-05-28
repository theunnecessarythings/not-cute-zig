const std = @import("std");
const cuda = @import("cuda.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const n = 1024;
const block_size = 256;

pub fn main() !void {
    cuda.init();

    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    var out: [n]f32 = undefined;

    for (&a, &b, 0..) |*a_item, *b_item, i| {
        a_item.* = @floatFromInt(i);
        b_item.* = @floatFromInt(i * 2);
    }

    const d_a = try cuda.malloc(f32, n);
    defer cuda.free(d_a);

    const d_b = try cuda.malloc(f32, n);
    defer cuda.free(d_b);

    const d_out = try cuda.malloc(f32, n);
    defer cuda.free(d_out);

    cuda.memcpy(f32, d_a, &a, .host_to_device);
    cuda.memcpy(f32, d_b, &b, .host_to_device);

    const module = try cuda.Module.loadData(@embedFile("cuda-module"));
    defer module.unload();

    const kernel = try module.getFunction("vector_add");
    kernel.launch(
        .{
            .grid_dim = .{ .x = (n + block_size - 1) / block_size },
            .block_dim = .{ .x = block_size },
        },
        .{ d_a.ptr, d_b.ptr, d_out.ptr, n },
    );

    cuda.memcpy(f32, &out, d_out, .device_to_host);

    for (out, 0..) |actual, i| {
        const expected = a[i] + b[i];
        if (actual != expected) {
            std.log.err("mismatch at {}: expected {}, got {}", .{ i, expected, actual });
            return error.VectorAddMismatch;
        }
    }

    std.log.info("vector add OK: {} elements", .{n});
    std.log.info("{} + {} = {}", .{ a[7], b[7], out[7] });
}
