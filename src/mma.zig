const std = @import("std");

pub const Shape = struct {
    m: usize,
    n: usize,
    k: usize,
};

pub const OperandLayout = enum {
    row,
    col,
};

pub const TypeTag = enum {
    f16,
    f32,
};

pub const Descriptor = struct {
    shape: Shape,
    a_layout: OperandLayout,
    b_layout: OperandLayout,
    a_type: TypeTag,
    b_type: TypeTag,
    c_type: TypeTag,
    d_type: TypeTag,
};

pub const m16n8k16_f16_f32 = Descriptor{
    .shape = .{ .m = 16, .n = 8, .k = 16 },
    .a_layout = .row,
    .b_layout = .col,
    .a_type = .f16,
    .b_type = .f16,
    .c_type = .f32,
    .d_type = .f32,
};

pub fn Backend(comptime desc: Descriptor) type {
    if (sameDescriptor(desc, m16n8k16_f16_f32)) return M16N8K16F16F32;
    @compileError("unsupported MMA descriptor");
}

pub const M16N8K16F16F32 = struct {
    pub const desc = m16n8k16_f16_f32;
    pub const m = 16;
    pub const n = 8;
    pub const k = 16;
    pub const lanes = 32;

    pub const AFragment = [4]u32;
    pub const BFragment = [2]u32;
    pub const Accumulator = [4]f32;

    pub fn zeroAccumulator() Accumulator {
        return .{ 0, 0, 0, 0 };
    }

    pub fn accumulatorCoord(lane_id: usize, comptime i: usize) [2]usize {
        const group_id = lane_id >> 2;
        const thread_id = lane_id & 3;
        return .{
            group_id + if (i >= 2) @as(usize, 8) else 0,
            thread_id * 2 + (i & 1),
        };
    }

    pub fn aCoord(lane_id: usize, comptime i: usize) [2]usize {
        const group_id = lane_id >> 2;
        const thread_id = lane_id & 3;
        return .{
            group_id + if ((i >= 2 and i < 4) or i >= 6) @as(usize, 8) else 0,
            thread_id * 2 + (i & 1) + if (i >= 4) @as(usize, 8) else 0,
        };
    }

    pub fn bCoord(lane_id: usize, comptime i: usize) [2]usize {
        const group_id = lane_id >> 2;
        const thread_id = lane_id & 3;
        return .{
            thread_id * 2 + (i & 1) + if (i >= 2) @as(usize, 8) else 0,
            group_id,
        };
    }

    pub fn loadA(ptr: [*]addrspace(.shared) const f16, stride: usize, lane_id: usize) AFragment {
        const tile_row = (lane_id & 7) + if ((lane_id & 8) != 0) @as(usize, 8) else 0;
        const tile_col = if ((lane_id & 16) != 0) @as(usize, 8) else 0;
        return ldmatrixX4(ptr + tile_row * stride + tile_col);
    }

    pub fn loadB(ptr: [*]addrspace(.shared) const f16, stride: usize, lane_id: usize) BFragment {
        const tile_row = (lane_id & 7) + if ((lane_id & 8) != 0) @as(usize, 8) else 0;
        return ldmatrixX2Trans(ptr + tile_row * stride);
    }

    pub fn mma(a: AFragment, b: BFragment, c: Accumulator) Accumulator {
        var d0: f32 = undefined;
        var d1: f32 = undefined;
        var d2: f32 = undefined;
        var d3: f32 = undefined;
        asm volatile (
            \\mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
            \\  {%[d0], %[d1], %[d2], %[d3]},
            \\  {%[a0], %[a1], %[a2], %[a3]},
            \\  {%[b0], %[b1]},
            \\  {%[c0], %[c1], %[c2], %[c3]};
            : [d0] "=f" (d0),
              [d1] "=f" (d1),
              [d2] "=f" (d2),
              [d3] "=f" (d3),
            : [a0] "r" (a[0]),
              [a1] "r" (a[1]),
              [a2] "r" (a[2]),
              [a3] "r" (a[3]),
              [b0] "r" (b[0]),
              [b1] "r" (b[1]),
              [c0] "f" (c[0]),
              [c1] "f" (c[1]),
              [c2] "f" (c[2]),
              [c3] "f" (c[3]),
        );
        return .{ d0, d1, d2, d3 };
    }
};

pub fn packF16x2(lo: f16, hi: f16) u32 {
    return @as(u16, @bitCast(lo)) | (@as(u32, @as(u16, @bitCast(hi))) << 16);
}

fn ldmatrixX4(ptr: [*]addrspace(.shared) const f16) [4]u32 {
    const addr = @intFromPtr(ptr);
    var d0: u32 = undefined;
    var d1: u32 = undefined;
    var d2: u32 = undefined;
    var d3: u32 = undefined;
    asm volatile (
        \\ldmatrix.sync.aligned.m8n8.x4.shared.b16
        \\  {%[d0], %[d1], %[d2], %[d3]}, [%[addr]];
        : [d0] "=r" (d0),
          [d1] "=r" (d1),
          [d2] "=r" (d2),
          [d3] "=r" (d3),
        : [addr] "l" (addr),
    );
    return .{ d0, d1, d2, d3 };
}

fn ldmatrixX2Trans(ptr: [*]addrspace(.shared) const f16) [2]u32 {
    const addr = @intFromPtr(ptr);
    var d0: u32 = undefined;
    var d1: u32 = undefined;
    asm volatile (
        \\ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
        \\  {%[d0], %[d1]}, [%[addr]];
        : [d0] "=r" (d0),
          [d1] "=r" (d1),
        : [addr] "l" (addr),
    );
    return .{ d0, d1 };
}

fn sameDescriptor(a: Descriptor, b: Descriptor) bool {
    return a.shape.m == b.shape.m and
        a.shape.n == b.shape.n and
        a.shape.k == b.shape.k and
        a.a_layout == b.a_layout and
        a.b_layout == b.b_layout and
        a.a_type == b.a_type and
        a.b_type == b.b_type and
        a.c_type == b.c_type and
        a.d_type == b.d_type;
}

test "dispatch selects m16n8k16 backend" {
    const B = Backend(m16n8k16_f16_f32);
    try std.testing.expectEqual(@as(usize, 16), B.m);
    try std.testing.expectEqual(@as(usize, 8), B.n);
    try std.testing.expectEqual(@as(usize, 16), B.k);
}

test "fragment coordinate maps match m16n8k16 formulas" {
    const B = Backend(m16n8k16_f16_f32);
    try std.testing.expectEqual([_]usize{ 0, 0 }, B.aCoord(0, 0));
    try std.testing.expectEqual([_]usize{ 8, 1 }, B.aCoord(0, 3));
    try std.testing.expectEqual([_]usize{ 0, 8 }, B.aCoord(0, 4));
    try std.testing.expectEqual([_]usize{ 0, 0 }, B.bCoord(0, 0));
    try std.testing.expectEqual([_]usize{ 8, 0 }, B.bCoord(0, 2));
    try std.testing.expectEqual([_]usize{ 15, 6 }, B.accumulatorCoord(31, 2));
}
