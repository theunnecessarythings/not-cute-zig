const std = @import("std");

pub fn rowMajor(comptime shape_tuple: anytype) Layout(rankOf(shape_tuple)) {
    const rank = comptime rankOf(shape_tuple);
    const shape = comptime tupleToArray(rank, shape_tuple);

    comptime var stride: [rank]usize = undefined;
    comptime var step: usize = 1;
    comptime var i = rank;
    inline while (i > 0) {
        i -= 1;
        stride[i] = step;
        step *= shape[i];
    }

    return .{ .shape = shape, .stride = stride };
}

pub fn colMajor(comptime shape_tuple: anytype) Layout(rankOf(shape_tuple)) {
    const rank = comptime rankOf(shape_tuple);
    const shape = comptime tupleToArray(rank, shape_tuple);

    comptime var stride: [rank]usize = undefined;
    comptime var step: usize = 1;
    inline for (0..rank) |i| {
        stride[i] = step;
        step *= shape[i];
    }

    return .{ .shape = shape, .stride = stride };
}

pub fn strided(comptime spec: anytype) Layout(rankOf(spec.shape)) {
    const rank = comptime rankOf(spec.shape);
    return .{
        .shape = tupleToArray(rank, spec.shape),
        .stride = tupleToArray(rank, spec.stride),
    };
}

pub fn Layout(comptime rank: usize) type {
    return struct {
        const Self = @This();
        pub const is_tiled = false;

        shape: [rank]usize,
        stride: [rank]usize,
        swizzle: ?XorSwizzle = null,

        pub fn extent(self: Self) [rank]usize {
            return self.shape;
        }

        pub fn partitionExtent(self: Self) [rank]usize {
            return self.extent();
        }

        pub fn offset(self: Self, coord_tuple: anytype) usize {
            const coord = tupleToArray(rank, coord_tuple);
            var result: usize = 0;
            inline for (0..rank) |i| {
                result += coord[i] * self.stride[i];
            }
            if (self.swizzle) |s| return s.apply(result);
            return result;
        }

        pub fn valid(self: Self, coord_tuple: anytype) bool {
            const coord = tupleToArray(rank, coord_tuple);
            inline for (0..rank) |i| {
                if (coord[i] >= self.shape[i]) return false;
            }
            return true;
        }

        pub fn tile(self: Self, comptime tile_shape_tuple: anytype) TiledLayout(Self, rank) {
            return .{
                .base = self,
                .tile_shape = tupleToArray(rank, tile_shape_tuple),
            };
        }

        pub fn swizzleXor(self: Self, comptime spec: XorSwizzle) Self {
            var result = self;
            result.swizzle = spec;
            return result;
        }

        pub fn partition(self: Self, comptime owner_shape_tuple: anytype) Partition(Self, rank) {
            return .{
                .base = self,
                .owner_shape = tupleToArray(rank, owner_shape_tuple),
            };
        }
    };
}

pub fn TiledLayout(comptime Base: type, comptime rank: usize) type {
    return struct {
        const Self = @This();
        pub const is_tiled = true;

        base: Base,
        tile_shape: [rank]usize,

        pub fn extent(self: Self) [rank]usize {
            return self.base.extent();
        }

        pub fn partitionExtent(self: Self) [rank]usize {
            return self.tilesExtent();
        }

        pub fn tilesExtent(self: Self) [rank]usize {
            const shape = if (Base.is_tiled) self.base.tilesExtent() else self.base.extent();
            var result: [rank]usize = undefined;
            inline for (0..rank) |i| {
                result[i] = ceilDiv(shape[i], self.tile_shape[i]);
            }
            return result;
        }

        pub fn offset(self: Self, coord_tuple: anytype) usize {
            return self.base.offset(coord_tuple);
        }

        pub fn valid(self: Self, coord_tuple: anytype) bool {
            return self.base.valid(coord_tuple);
        }

        pub fn tile(self: Self, comptime tile_shape_tuple: anytype) TiledLayout(Self, rank) {
            return .{
                .base = self,
                .tile_shape = tupleToArray(rank, tile_shape_tuple),
            };
        }

        pub fn tileView(self: Self, tile_coord_tuple: anytype) TileView(Base, rank) {
            const tile_coord = tupleToArray(rank, tile_coord_tuple);
            const shape = self.base.extent();
            var origin: [rank]usize = undefined;
            var actual: [rank]usize = undefined;

            inline for (0..rank) |i| {
                origin[i] = tile_coord[i] * self.tile_shape[i];
                actual[i] = if (origin[i] < shape[i])
                    @min(self.tile_shape[i], shape[i] - origin[i])
                else
                    0;
            }

            return .{
                .base = self.base,
                .origin = origin,
                .shape = actual,
            };
        }

        pub fn partition(self: Self, comptime owner_shape_tuple: anytype) Partition(Self, rank) {
            return .{
                .base = self,
                .owner_shape = tupleToArray(rank, owner_shape_tuple),
            };
        }
    };
}

pub fn Partition(comptime Base: type, comptime rank: usize) type {
    return struct {
        const Self = @This();

        base: Base,
        owner_shape: [rank]usize,

        pub fn extent(self: Self) [rank]usize {
            return self.base.extent();
        }

        pub fn partitionExtent(self: Self) [rank]usize {
            return self.owner_shape;
        }

        pub fn ownerShape(self: Self) [rank]usize {
            return self.owner_shape;
        }

        pub fn ownerExtent(self: Self) [rank]usize {
            const shape = self.base.partitionExtent();
            var result: [rank]usize = undefined;
            inline for (0..rank) |i| {
                result[i] = ceilDiv(shape[i], self.owner_shape[i]);
            }
            return result;
        }

        pub fn coord(self: Self, owner_coord_tuple: anytype, local_coord_tuple: anytype) [rank]usize {
            const owner_coord = tupleToArray(rank, owner_coord_tuple);
            const local_coord = tupleToArray(rank, local_coord_tuple);
            var result: [rank]usize = undefined;
            inline for (0..rank) |i| {
                result[i] = owner_coord[i] * self.owner_shape[i] + local_coord[i];
            }
            return result;
        }

        pub fn valid(self: Self, owner_coord_tuple: anytype, local_coord_tuple: anytype) bool {
            return self.base.valid(self.coord(owner_coord_tuple, local_coord_tuple));
        }

        pub fn offset(self: Self, owner_coord_tuple: anytype, local_coord_tuple: anytype) usize {
            return self.base.offset(self.coord(owner_coord_tuple, local_coord_tuple));
        }

        pub fn ownerView(self: Self, owner_coord_tuple: anytype) TileView(Base, rank) {
            const owner_coord = tupleToArray(rank, owner_coord_tuple);
            const shape = self.base.extent();
            var origin: [rank]usize = undefined;
            var actual: [rank]usize = undefined;

            inline for (0..rank) |i| {
                origin[i] = owner_coord[i] * self.owner_shape[i];
                actual[i] = if (origin[i] < shape[i])
                    @min(self.owner_shape[i], shape[i] - origin[i])
                else
                    0;
            }

            return .{
                .base = self.base,
                .origin = origin,
                .shape = actual,
            };
        }

        pub fn partition(self: Self, comptime owner_shape_tuple: anytype) Partition(Self, rank) {
            return .{
                .base = self,
                .owner_shape = tupleToArray(rank, owner_shape_tuple),
            };
        }
    };
}

pub fn TileView(comptime Base: type, comptime rank: usize) type {
    return struct {
        const Self = @This();

        base: Base,
        origin: [rank]usize,
        shape: [rank]usize,

        pub fn extent(self: Self) [rank]usize {
            return self.shape;
        }

        pub fn valid(self: Self, local_coord_tuple: anytype) bool {
            const local = tupleToArray(rank, local_coord_tuple);
            inline for (0..rank) |i| {
                if (local[i] >= self.shape[i]) return false;
            }
            return true;
        }

        pub fn offset(self: Self, local_coord_tuple: anytype) usize {
            const local = tupleToArray(rank, local_coord_tuple);
            var coord: [rank]usize = undefined;
            inline for (0..rank) |i| {
                coord[i] = self.origin[i] + local[i];
            }
            return self.base.offset(coord);
        }
    };
}

pub fn tensor(ptr: anytype, layout_value: anytype) TensorView(NormalizedPtr(@TypeOf(ptr)), @TypeOf(layout_value)) {
    return .{
        .ptr = normalizePtr(ptr),
        .layout = layout_value,
    };
}

fn NormalizedPtr(comptime Ptr: type) type {
    const info = @typeInfo(Ptr).pointer;
    if (info.size == .one and @typeInfo(info.child) == .array) {
        const array = @typeInfo(info.child).array;
        return [*]array.child;
    }
    return Ptr;
}

fn normalizePtr(ptr: anytype) NormalizedPtr(@TypeOf(ptr)) {
    const Ptr = @TypeOf(ptr);
    const info = @typeInfo(Ptr).pointer;
    if (info.size == .one and @typeInfo(info.child) == .array) {
        return ptr;
    }
    return ptr;
}

pub fn TensorView(comptime Ptr: type, comptime LayoutType: type) type {
    const info = @typeInfo(Ptr).pointer;
    if (info.size != .many and info.size != .one) {
        @compileError("TensorView requires a single-item or many-item pointer");
    }

    return struct {
        const Self = @This();
        pub const Element = info.child;

        ptr: Ptr,
        layout: LayoutType,

        pub fn ref(self: Self, coord_tuple: anytype) Ptr {
            return self.ptr + self.layout.offset(coord_tuple);
        }

        pub fn get(self: Self, coord_tuple: anytype) Element {
            return self.ref(coord_tuple)[0];
        }

        pub fn set(self: Self, coord_tuple: anytype, value: Element) void {
            self.ref(coord_tuple)[0] = value;
        }
    };
}

pub const XorSwizzle = struct {
    source_bit: u6,
    target_bit: u6,
    width: u6,

    /// 128-byte swizzle (e.g., float4 or 16-byte aligned accesses)
    pub fn swizzle128b() XorSwizzle {
        return .{ .source_bit = 4, .target_bit = 1, .width = 3 }; // shifts based on 16B alignment
    }

    /// 64-byte swizzle (e.g., float2 or 8-byte aligned accesses)
    pub fn swizzle64b() XorSwizzle {
        return .{ .source_bit = 3, .target_bit = 1, .width = 3 };
    }

    /// 32-byte swizzle (e.g., float or 4-byte aligned accesses)
    pub fn swizzle32b() XorSwizzle {
        return .{ .source_bit = 2, .target_bit = 1, .width = 3 };
    }

    pub fn apply(self: XorSwizzle, offset_value: usize) usize {
        var result = offset_value;
        var bit: u6 = 0;
        while (bit < self.width) : (bit += 1) {
            const src = self.source_bit + bit;
            const dst = self.target_bit + bit;
            const src_bit = (offset_value >> src) & 1;
            result ^= src_bit << dst;
        }
        return result;
    }
};

fn rankOf(comptime tuple: anytype) usize {
    return rankOfType(@TypeOf(tuple));
}

fn rankOfType(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| info.fields.len,
        .array => |info| info.len,
        else => @compileError("layout shape/coordinate must be a tuple or fixed array"),
    };
}

fn tupleToArray(comptime rank: usize, tuple: anytype) [rank]usize {
    comptime if (rankOfType(@TypeOf(tuple)) != rank) {
        @compileError("layout tuple rank mismatch");
    };

    var result: [rank]usize = undefined;
    inline for (0..rank) |i| {
        result[i] = tuple[i];
    }
    return result;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "row-major, column-major, and custom strides" {
    try std.testing.expectEqual(@as(usize, 17), rowMajor(.{ 3, 5, 7 }).offset(.{ 0, 2, 3 }));
    try std.testing.expectEqual(@as(usize, 15), colMajor(.{ 3, 5 }).offset(.{ 0, 5 }));
    try std.testing.expectEqual(@as(usize, 42), strided(.{
        .shape = .{ 4, 4, 4 },
        .stride = .{ 1, 10, 100 },
    }).offset(.{ 2, 4, 0 }));
}

test "rank-n validity and extents" {
    const l = rowMajor(.{ 2, 3, 4, 5 });
    try std.testing.expectEqual([_]usize{ 2, 3, 4, 5 }, l.extent());
    try std.testing.expect(l.valid(.{ 1, 2, 3, 4 }));
    try std.testing.expect(!l.valid(.{ 1, 3, 3, 4 }));
}

test "nested tile extents and residue tile view" {
    const tiled = rowMajor(.{ 130, 70 }).tile(.{ 16, 16 });
    try std.testing.expectEqual([_]usize{ 9, 5 }, tiled.tilesExtent());

    const edge = tiled.tileView(.{ 8, 4 });
    try std.testing.expectEqual([_]usize{ 2, 6 }, edge.extent());
    try std.testing.expect(edge.valid(.{ 1, 5 }));
    try std.testing.expect(!edge.valid(.{ 2, 0 }));
    try std.testing.expectEqual(rowMajor(.{ 130, 70 }).offset(.{ 128, 64 }), edge.offset(.{ 0, 0 }));

    const nested = tiled.tile(.{ 4, 2 });
    try std.testing.expectEqual([_]usize{ 3, 3 }, nested.tilesExtent());
}

test "xor swizzle transforms final offsets" {
    const base = rowMajor(.{ 8, 8 });
    const swizzled = base.swizzleXor(.{ .source_bit = 0, .target_bit = 3, .width = 1 });
    try std.testing.expectEqual(@as(usize, 1), base.offset(.{ 0, 1 }));
    try std.testing.expectEqual(@as(usize, 9), swizzled.offset(.{ 0, 1 }));
}

test "tensor view get and set use layout offsets" {
    var data = [_]usize{ 0, 1, 2, 3, 4, 5 };
    const l = rowMajor(.{ 2, 3 });
    const view = tensor(&data, l);

    try std.testing.expectEqual(@as(usize, 5), view.get(.{ 1, 2 }));
    view.set(.{ 0, 1 }, 99);
    try std.testing.expectEqual(@as(usize, 99), data[1]);
}

test "partition maps owner and local coordinates to logical coordinates" {
    const matrix = rowMajor(.{ 128, 128 });
    const cta = matrix.partition(.{ 64, 64 });

    try std.testing.expectEqual([_]usize{ 2, 2 }, cta.ownerExtent());
    try std.testing.expectEqual([_]usize{ 64, 64 }, cta.ownerShape());
    try std.testing.expectEqual([_]usize{ 65, 67 }, cta.coord(.{ 1, 1 }, .{ 1, 3 }));
    try std.testing.expectEqual(matrix.offset(.{ 65, 67 }), cta.offset(.{ 1, 1 }, .{ 1, 3 }));
}

test "partition hierarchy describes cta warp lane value ownership" {
    const cta = rowMajor(.{ 128, 128 }).partition(.{ 64, 64 });
    const warp = cta.partition(.{ 32, 64 });
    const lane = warp.partition(.{ 1, 8 });
    const value = lane.partition(.{ 1, 1 });

    try std.testing.expectEqual([_]usize{ 2, 2 }, cta.ownerExtent());
    try std.testing.expectEqual([_]usize{ 2, 1 }, warp.ownerExtent());
    try std.testing.expectEqual([_]usize{ 32, 8 }, lane.ownerExtent());
    try std.testing.expectEqual([_]usize{ 1, 8 }, value.ownerExtent());

    const cta_coord = .{ 1, 0 };
    const warp_coord = .{ 1, 0 };
    const lane_coord = .{ 7, 3 };
    const value_coord = .{ 0, 4 };

    const in_cta = cta.coord(cta_coord, warp.coord(warp_coord, lane.coord(lane_coord, value.coord(value_coord, .{ 0, 0 }))));
    try std.testing.expectEqual([_]usize{ 103, 28 }, in_cta);
    try std.testing.expectEqual(rowMajor(.{ 128, 128 }).offset(.{ 103, 28 }), cta.base.offset(in_cta));
}

test "partition owner views carry residue extents" {
    const cta = rowMajor(.{ 130, 70 }).partition(.{ 64, 64 });
    const edge = cta.ownerView(.{ 2, 1 });

    try std.testing.expectEqual([_]usize{ 3, 2 }, cta.ownerExtent());
    try std.testing.expectEqual([_]usize{ 2, 6 }, edge.extent());
    try std.testing.expect(edge.valid(.{ 1, 5 }));
    try std.testing.expect(!edge.valid(.{ 2, 0 }));
}
