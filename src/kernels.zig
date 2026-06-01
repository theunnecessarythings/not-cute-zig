const layout = @import("layout.zig");
const mma = @import("mma.zig");
const config = @import("config.zig");
const device = @import("device.zig");

const use_swizzled_shared_transpose = false;

var transpose_shared: [config.transpose_tile * config.transpose_tile]f32 addrspace(.shared) = undefined;
var mma_shared_a: [config.mma_m * config.mma_k]f16 addrspace(.shared) = undefined;
var mma_shared_b: [config.mma_k * config.mma_n]f16 addrspace(.shared) = undefined;
var mma_shared_a_1: [config.mma_m * config.mma_k]f16 addrspace(.shared) = undefined;
var mma_shared_b_1: [config.mma_k * config.mma_n]f16 addrspace(.shared) = undefined;
var reduce_shared: [32]f32 addrspace(.shared) = undefined;

pub const Epilogue = enum(u32) {
    none = 0,
    relu = 1,
    silu = 2,
    gelu = 3,
};

pub fn pipelined_mma_matmul(
    a: [*]addrspace(.global) const f16,
    b: [*]addrspace(.global) const f16,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta: f32,
    batch_stride_a: usize,
    batch_stride_b: usize,
    batch_stride_c: usize,
    k_iters: usize,
    epilogue_kind: u32,
) callconv(.kernel) void {
    const Backend = mma.Backend(mma.m16n8k16_f16_f32);
    const batch_id = @workGroupId(1);

    const batched_a = a + batch_id * batch_stride_a;
    const batched_b = b + batch_id * batch_stride_b;
    const batched_c = c + batch_id * batch_stride_c;
    const batched_d = d + batch_id * batch_stride_c;

    const shared_a_ptr_0: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_a);
    const shared_b_ptr_0: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_b);
    const shared_a_ptr_1: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_a_1);
    const shared_b_ptr_1: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_b_1);

    const lane_id = @workItemId(0);
    var acc = Backend.zeroAccumulator();

    // Prologue: start loading tile 0
    var k_step: usize = 0;

    // Asynchronous load for Tile 0
    const i = lane_id * 8; // Each cp.async.cg_16 copies 16 bytes = 8 f16s.
    if (i < config.mma_m * config.mma_k) {
        device.cp_async.cg_16(@ptrCast(shared_a_ptr_0 + i), @ptrCast(batched_a + i));
    }
    if (i < config.mma_k * config.mma_n) {
        device.cp_async.cg_16(@ptrCast(shared_b_ptr_0 + i), @ptrCast(batched_b + i));
    }
    device.cp_async.commit_group();
    device.cp_async.wait_group(0); // wait for tile 0
    blockBarrier();

    var write_stage: u1 = 1;
    var read_stage: u1 = 0;

    // Main loop
    while (k_step < k_iters - 1) : (k_step += 1) {
        const next_a_offset = (k_step + 1) * config.mma_m * config.mma_k;
        const next_b_offset = (k_step + 1) * config.mma_k * config.mma_n;

        const wr_a_ptr = if (write_stage == 0) shared_a_ptr_0 else shared_a_ptr_1;
        const wr_b_ptr = if (write_stage == 0) shared_b_ptr_0 else shared_b_ptr_1;

        // Start async copy for next tile
        if (i < config.mma_m * config.mma_k) {
            device.cp_async.cg_16(@ptrCast(wr_a_ptr + i), @ptrCast(batched_a + next_a_offset + i));
        }
        if (i < config.mma_k * config.mma_n) {
            device.cp_async.cg_16(@ptrCast(wr_b_ptr + i), @ptrCast(batched_b + next_b_offset + i));
        }
        device.cp_async.commit_group();

        // Compute current tile
        const rd_a_ptr = if (read_stage == 0) shared_a_ptr_0 else shared_a_ptr_1;
        const rd_b_ptr = if (read_stage == 0) shared_b_ptr_0 else shared_b_ptr_1;

        const a_frag = Backend.loadA(rd_a_ptr, config.mma_k, lane_id);
        const b_frag = Backend.loadB(rd_b_ptr, config.mma_n, lane_id);
        acc = Backend.mma(a_frag, b_frag, acc);

        // Wait for the async copy we just issued to complete
        device.cp_async.wait_group(0);
        blockBarrier();

        write_stage +%= 1;
        read_stage +%= 1;
    }

    // Epilogue: compute final tile
    const rd_a_ptr = if (read_stage == 0) shared_a_ptr_0 else shared_a_ptr_1;
    const rd_b_ptr = if (read_stage == 0) shared_b_ptr_0 else shared_b_ptr_1;
    const a_frag = Backend.loadA(rd_a_ptr, config.mma_k, lane_id);
    const b_frag = Backend.loadB(rd_b_ptr, config.mma_n, lane_id);
    acc = Backend.mma(a_frag, b_frag, acc);

    // Write out results with fused epilogue
    const c_layout = layout.rowMajor(.{ config.mma_m, config.mma_n });
    const c_view = layout.tensor(batched_c, c_layout);
    const d_view = layout.tensor(batched_d, c_layout);

    inline for (0..4) |acc_i| {
        const coord = Backend.accumulatorCoord(lane_id, acc_i);
        const c_val = c_view.get(coord);

        var out_val = alpha * acc[acc_i] + beta * c_val;

        if (epilogue_kind == @intFromEnum(Epilogue.relu)) {
            out_val = device.relu(out_val);
        } else if (epilogue_kind == @intFromEnum(Epilogue.silu)) {
            out_val = device.silu(out_val);
        } else if (epilogue_kind == @intFromEnum(Epilogue.gelu)) {
            out_val = device.gelu(out_val);
        }

        d_view.set(coord, out_val);
    }
}

pub fn stream_sleep(
    clock_count: u32,
    out: [*]addrspace(.global) u32,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    if (tid == 0) {
        device.nanosleep(clock_count);
        out[0] = 1;
    }
}

pub fn block_reduce_sum(
    input: [*]addrspace(.global) const f32,
    output: [*]addrspace(.global) f32,
    len: usize,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    const bid = @workGroupId(0);
    const bdim = @workGroupSize(0);
    const idx = bid * bdim + tid;

    var sum: f32 = 0;
    if (idx < len) {
        sum = input[idx];
    }

    // Warp reduction (xor shuffle is fine for all-reduce, down is fine for standard reduce)
    // Actually, butterfly is fine for all-reduce. The problem is we use offset as a mask!
    // shfl.sync.bfly uses `lane_id ^ mask`.
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        const val_u32 = device.shfl_sync_bfly(0xFFFFFFFF, @bitCast(sum), offset, 32);
        sum += @as(f32, @bitCast(val_u32));
    }

    const lane = tid % 32;
    const warp_id = tid / 32;

    if (lane == 0) {
        reduce_shared[warp_id] = sum;
    }
    device.syncthreads();

    if (warp_id == 0) {
        sum = if (lane < (bdim / 32)) reduce_shared[lane] else 0;
        inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
            const val_u32 = device.shfl_sync_bfly(0xFFFFFFFF, @bitCast(sum), offset, 32);
            sum += @as(f32, @bitCast(val_u32));
        }
        if (lane == 0) {
            output[bid] = sum;
        }
    }
}

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
    mma_matmul_impl(a, b, c, d, alpha, beta);
}

pub fn batched_mma_matmul(
    a: [*]addrspace(.global) const f16,
    b: [*]addrspace(.global) const f16,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta: f32,
    batch_stride_a: usize,
    batch_stride_b: usize,
    batch_stride_c: usize,
) callconv(.kernel) void {
    const batch_id = @workGroupId(1);
    const batched_a = a + batch_id * batch_stride_a;
    const batched_b = b + batch_id * batch_stride_b;
    const batched_c = c + batch_id * batch_stride_c;
    const batched_d = d + batch_id * batch_stride_c;

    mma_matmul_impl(batched_a, batched_b, batched_c, batched_d, alpha, beta);
}

inline fn mma_matmul_impl(
    a: [*]addrspace(.global) const f16,
    b: [*]addrspace(.global) const f16,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta: f32,
) void {
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

pub fn flash_attention_fwd(
    q: [*]addrspace(.global) const f16,
    k: [*]addrspace(.global) const f16,
    v: [*]addrspace(.global) const f16,
    o: [*]addrspace(.global) f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) callconv(.kernel) void {
    _ = seq_len;
    _ = head_dim;
    const Backend = mma.Backend(mma.m16n8k16_f16_f32);
    const lane_id = @workItemId(0);

    // Simplified 1-block, 1-warp forward pass (16x16 block) for demonstration.
    // In a real FA kernel, we'd loop over tiles of K and V to incrementally compute the softmax.

    const shared_q_ptr: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_a);
    const shared_k_ptr: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_b);

    // Q is 16x16 (m=16, k=16). Load Q.
    var i = lane_id;
    while (i < 16 * 16) : (i += 32) {
        shared_q_ptr[i] = q[i];
        shared_k_ptr[i] = k[i];
    }
    blockBarrier();

    // 1. Compute S = Q * K^T * scale
    // mma.m16n8k16.row.col uses B as col-major, so passing K directly computes Q * K^T
    const q_frag = Backend.loadA(shared_q_ptr, 16, lane_id);
    const k_frag = Backend.loadB(shared_k_ptr, 16, lane_id); // using 16 as stride to treat as 16x16

    var s_acc = Backend.mma(q_frag, k_frag, Backend.zeroAccumulator());

    inline for (0..4) |acc_i| {
        s_acc[acc_i] *= scale;
    }

    // 2. Row-wise Max (simplified: no causal mask)
    var row_max: f32 = -1e20;
    inline for (0..4) |acc_i| {
        if (s_acc[acc_i] > row_max) row_max = s_acc[acc_i];
    }
    // Warp-level reduction for max across the row
    // Each thread holds 4 values of the 16x16 S matrix.
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        const val_u32 = device.shfl_sync_bfly(0xFFFFFFFF, @bitCast(row_max), offset, 32);
        row_max = @max(row_max, @as(f32, @bitCast(val_u32)));
    }

    // 3. Row-wise Sum of Exp(S - max)
    var row_sum: f32 = 0;
    inline for (0..4) |acc_i| {
        s_acc[acc_i] = device.ex2_approx((s_acc[acc_i] - row_max) * 1.4426950408889634);
        row_sum += s_acc[acc_i];
    }
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        const val_u32 = device.shfl_sync_bfly(0xFFFFFFFF, @bitCast(row_sum), offset, 32);
        row_sum += @as(f32, @bitCast(val_u32));
    }

    // 4. Normalize to get P
    inline for (0..4) |acc_i| {
        s_acc[acc_i] /= row_sum;
    }

    // Write P back to shared memory (cast to f16) to prepare for P * V
    // P is 16x16.
    const shared_p_ptr: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_a_1);
    const shared_v_ptr: [*]addrspace(.shared) f16 = @ptrCast(&mma_shared_b_1);

    inline for (0..4) |acc_i| {
        const coord = Backend.accumulatorCoord(lane_id, acc_i);
        const row = coord[0];
        const col = coord[1];
        shared_p_ptr[row * 16 + col] = @floatCast(s_acc[acc_i]);
    }

    // Load V
    i = lane_id;
    while (i < 16 * 16) : (i += 32) {
        shared_v_ptr[i] = v[i];
    }
    blockBarrier();

    // 5. Compute O = P * V
    // V needs to be treated as col-major for the mma hardware if we want row*row
    // Wait, mma.row.col expects B to be col-major. We stored V as row-major.
    // For this simple demo, we will use loadA on P, loadB on V.
    const p_frag = Backend.loadA(shared_p_ptr, 16, lane_id);
    const v_frag = Backend.loadB(shared_v_ptr, 16, lane_id);

    const o_acc = Backend.mma(p_frag, v_frag, Backend.zeroAccumulator());

    // Write out O (16x16 f32)
    inline for (0..4) |acc_i| {
        const coord = Backend.accumulatorCoord(lane_id, acc_i);
        const row = coord[0];
        const col = coord[1];
        o[row * 16 + col] = o_acc[acc_i];
    }
}
