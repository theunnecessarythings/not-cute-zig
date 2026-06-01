const layout = @import("layout.zig");
const mma = @import("mma.zig");
const config = @import("config.zig");

const use_swizzled_shared_transpose = false;

var transpose_shared: [config.transpose_tile * config.transpose_tile]f32 addrspace(.shared) = undefined;
var mma_shared_a: [config.mma_m * config.mma_k]f16 addrspace(.shared) = undefined;
var mma_shared_b: [config.mma_k * config.mma_n]f16 addrspace(.shared) = undefined;

pub fn vector_add(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    len: usize,
) callconv(.kernel) void {
    const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    if (i >= len) return;

    out[i] = a[i] + b[i];
}

pub fn matrix_transpose(
    input: [*]addrspace(.global) const f32,
    output: [*]addrspace(.global) f32,
    rows: usize,
    cols: usize,
) callconv(.kernel) void {
    _ = rows;
    _ = cols;

    const in_layout = layout.rowMajor(.{ config.transpose_rows, config.transpose_cols });
    const out_layout = layout.rowMajor(.{ config.transpose_cols, config.transpose_rows });
    const in_cta = in_layout.partition(.{ config.transpose_tile, config.transpose_tile });
    const out_cta = out_layout.partition(.{ config.transpose_tile, config.transpose_tile });
    const in = layout.tensor(input, in_layout);
    const out = layout.tensor(output, out_layout);

    const shared_layout = if (use_swizzled_shared_transpose)
        layout.rowMajor(.{ config.transpose_tile, config.transpose_tile }).swizzleXor(.{
            .source_bit = 0,
            .target_bit = 4,
            .width = 1,
        })
    else
        layout.rowMajor(.{ config.transpose_tile, config.transpose_tile });
    const shared_ptr: [*]addrspace(.shared) f32 = @ptrCast(&transpose_shared);
    const shared = layout.tensor(shared_ptr, shared_layout);

    const tx = @workItemId(0);
    const ty = @workItemId(1);

    const in_owner = .{ @workGroupId(1), @workGroupId(0) };
    const in_tile = in_cta.ownerView(in_owner);
    if (in_tile.valid(.{ ty, tx })) {
        shared.set(.{ ty, tx }, in.get(in_cta.coord(in_owner, .{ ty, tx })));
    }

    blockBarrier();

    const out_owner = .{ @workGroupId(0), @workGroupId(1) };
    const out_tile = out_cta.ownerView(out_owner);
    if (out_tile.valid(.{ ty, tx })) {
        out.set(out_cta.coord(out_owner, .{ ty, tx }), shared.get(.{ tx, ty }));
    }
}

pub fn ownership_debug(
    output: [*]addrspace(.global) u32,
) callconv(.kernel) void {
    const matrix = layout.rowMajor(.{ config.ownership_m, config.ownership_n });
    const cta = matrix.partition(.{ config.ownership_m, config.ownership_n });
    const warp = cta.partition(.{ 32, 64 });
    const lane = warp.partition(.{ 1, config.ownership_values_per_lane_owner });
    const value = lane.partition(.{ 1, 1 });
    const out = layout.tensor(output, matrix);

    const lane_id = @workItemId(0);
    const warp_id = @workGroupId(0);

    inline for (0..8) |strip_group| {
        const lane_owner = .{ lane_id, strip_group };

        inline for (0..config.ownership_values_per_lane_owner) |value_col| {
            const coord = cta.coord(
                .{ 0, 0 },
                warp.coord(
                    .{ warp_id, 0 },
                    lane.coord(
                        lane_owner,
                        value.coord(.{ 0, value_col }, .{ 0, 0 }),
                    ),
                ),
            );
            out.set(coord, encodeOwnership(
                @intCast(warp_id),
                @intCast(lane_id),
                @intCast(strip_group * config.ownership_values_per_lane_owner + value_col),
            ));
        }
    }
}

pub fn mma_matmul(
    a: [*]addrspace(.global) const f16,
    b: [*]addrspace(.global) const f16,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta: f32,
) callconv(.kernel) void {
    const Backend = mma.Backend(mma.m16n8k16_f16_f32);

    const a_layout = layout.rowMajor(.{ config.mma_m, config.mma_k });
    const b_layout = layout.colMajor(.{ config.mma_k, config.mma_n });
    const c_layout = layout.rowMajor(.{ config.mma_m, config.mma_n });
    const a_view = layout.tensor(a, a_layout);
    const b_view = layout.tensor(b, b_layout);
    const c_view = layout.tensor(c, c_layout);
    const d_view = layout.tensor(d, c_layout);

    const shared_a_ptr: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_a);
    const shared_b_ptr: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_b);
    const shared_a = layout.tensor(shared_a_ptr, a_layout);
    const shared_b = layout.tensor(shared_b_ptr, layout.rowMajor(.{ config.mma_k, config.mma_n }));

    const lane_id = @workItemId(0);
    var i = lane_id;
    while (i < config.mma_m * config.mma_k) : (i += Backend.lanes) {
        const row = i / config.mma_k;
        const col = i % config.mma_k;
        shared_a.set(.{ row, col }, a_view.get(.{ row, col }));
    }

    i = lane_id;
    while (i < config.mma_k * config.mma_n) : (i += Backend.lanes) {
        const row = i / config.mma_n;
        const col = i % config.mma_n;
        shared_b.set(.{ row, col }, b_view.get(.{ row, col }));
    }

    blockBarrier();

    const a_frag = Backend.loadA(shared_a_ptr, config.mma_k, lane_id);
    const b_frag = Backend.loadB(shared_b_ptr, config.mma_n, lane_id);
    const acc = Backend.mma(a_frag, b_frag, Backend.zeroAccumulator());

    inline for (0..4) |acc_i| {
        const coord = Backend.accumulatorCoord(lane_id, acc_i);
        const c_val = c_view.get(coord);
        d_view.set(coord, alpha * acc[acc_i] + beta * c_val);
    }
}

fn blockBarrier() void {
    asm volatile ("bar.sync 0;");
}

fn encodeOwnership(warp_id: u32, lane_owner: u32, value_id: u32) u32 {
    return (warp_id << 24) | (lane_owner << 8) | value_id;
}
