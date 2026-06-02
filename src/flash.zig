const std = @import("std");
const cuda = @import("cuda.zig");
const config = @import("config.zig");

pub const Options = struct {
    batch_heads: usize = 1,
    seq_len: usize,
    head_dim: usize,
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    o_stride: usize,
    scale: f32,
    causal: bool = false,

    pub fn validate(self: Options) !void {
        if (self.batch_heads == 0 or self.seq_len == 0) return error.InvalidValue;
        if (self.head_dim != 16 and self.head_dim != 32) return error.InvalidValue;

        const min_stride = self.seq_len * self.head_dim;
        if (self.q_stride < min_stride or
            self.k_stride < min_stride or
            self.v_stride < min_stride or
            self.o_stride < min_stride)
        {
            return error.InvalidValue;
        }
    }
};

pub fn launch(
    module: cuda.Module,
    q: []const f16,
    k: []const f16,
    v: []const f16,
    o: []f32,
    opts: Options,
) !void {
    try opts.validate();
    try requireLen(f16, q, opts.batch_heads * opts.q_stride);
    try requireLen(f16, k, opts.batch_heads * opts.k_stride);
    try requireLen(f16, v, opts.batch_heads * opts.v_stride);
    try requireLen(f32, o, opts.batch_heads * opts.o_stride);

    const kernel = try module.getFunction("flash_attention_fwd");
    try kernel.launch(
        .{
            .grid_dim = .{
                .x = @intCast((opts.seq_len + config.flash_block_m - 1) / config.flash_block_m),
                .y = @intCast(opts.batch_heads),
            },
            .block_dim = .{ .x = 32 },
        },
        .{
            q.ptr,
            k.ptr,
            v.ptr,
            o.ptr,
            opts.seq_len,
            opts.head_dim,
            opts.q_stride,
            opts.k_stride,
            opts.v_stride,
            opts.o_stride,
            opts.scale,
            @as(u32, if (opts.causal) 1 else 0),
        },
    );
}

fn requireLen(comptime T: type, slice: []const T, min_len: usize) !void {
    if (slice.len < min_len) return error.InvalidValue;
}
