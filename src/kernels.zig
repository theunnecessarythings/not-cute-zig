const std = @import("std");
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
var flash_shared_q: [config.flash_block_m * config.flash_max_head_dim]f16 addrspace(.shared) = undefined;
var flash_shared_k: [config.flash_block_n * config.flash_max_head_dim]f16 addrspace(.shared) = undefined;
var flash_shared_v: [config.flash_block_n * config.flash_max_head_dim]f16 addrspace(.shared) = undefined;
var flash_shared_p: [config.flash_block_m * config.flash_max_head_dim]f16 addrspace(.shared) = undefined;
var flash_scores_shared: [config.flash_block_m * config.flash_block_n]f32 addrspace(.shared) = undefined;
var flash_row_max_shared: [config.flash_block_m]f32 addrspace(.shared) = undefined;
var flash_row_sum_shared: [config.flash_block_m]f32 addrspace(.shared) = undefined;
var flash_old_scale_shared: [config.flash_block_m]f32 addrspace(.shared) = undefined;

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
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    o_stride: usize,
    scale: f32,
    causal: u32,
) callconv(.kernel) void {
    if (seq_len == 0 or (head_dim != 16 and head_dim != 32)) return;

    const Backend = mma.Backend(mma.m16n8k16_f16_f32);
    const lane_id = @workItemId(0);
    if (lane_id >= Backend.lanes) return;

    const log2e = 1.4426950408889634;
    const q_base = @workGroupId(0) * config.flash_block_m;
    const batch_id = @workGroupId(1);
    const q_batch = q + batch_id * q_stride;
    const k_batch = k + batch_id * k_stride;
    const v_batch = v + batch_id * v_stride;
    const o_batch = o + batch_id * o_stride;

    const shared_q_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_shared_q);
    const shared_k_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_shared_k);
    const shared_v_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_shared_v);
    const shared_p_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_shared_p);

    var o_acc_0 = Backend.zeroAccumulator();
    var o_acc_1 = Backend.zeroAccumulator();
    var o_acc_2 = Backend.zeroAccumulator();
    var o_acc_3 = Backend.zeroAccumulator();

    if (lane_id < config.flash_block_m) {
        flash_row_max_shared[lane_id] = -std.math.inf(f32);
        flash_row_sum_shared[lane_id] = 0;
        flash_old_scale_shared[lane_id] = 0;
    }

    blockBarrier();

    var tile_start: usize = 0;
    while (tile_start < seq_len) : (tile_start += config.flash_block_n) {
        var scores = Backend.zeroAccumulator();

        var k_start: usize = 0;
        while (k_start < head_dim) : (k_start += config.mma_k) {
            var load_i = lane_id;
            while (load_i < config.flash_block_m * config.mma_k) : (load_i += Backend.lanes) {
                const row = load_i / config.mma_k;
                const col = load_i % config.mma_k;
                const q_row = q_base + row;
                const q_col = k_start + col;
                shared_q_ptr[row * config.mma_k + col] = if (q_row < seq_len and q_col < head_dim)
                    q_batch[q_row * head_dim + q_col]
                else
                    0;
            }

            load_i = lane_id;
            while (load_i < config.mma_k * config.flash_block_n) : (load_i += Backend.lanes) {
                const kk = load_i / config.flash_block_n;
                const col = load_i % config.flash_block_n;
                const key_row = tile_start + col;
                const key_col = k_start + kk;
                shared_k_ptr[load_i] = if (key_row < seq_len and key_col < head_dim)
                    k_batch[key_row * head_dim + key_col]
                else
                    0;
            }

            blockBarrier();

            const q_frag = Backend.loadA(shared_q_ptr, config.mma_k, lane_id);
            const k_frag = Backend.loadB(shared_k_ptr, config.flash_block_n, lane_id);
            scores = Backend.mma(q_frag, k_frag, scores);

            blockBarrier();
        }

        inline for (0..4) |acc_i| {
            const coord = Backend.accumulatorCoord(lane_id, acc_i);
            const row = coord[0];
            const col = coord[1];
            const key_row = tile_start + col;
            const q_row = q_base + row;
            const visible = causal == 0 or key_row <= q_row;
            scores[acc_i] = if (q_row < seq_len and key_row < seq_len and visible)
                scores[acc_i] * scale
            else
                -std.math.inf(f32);
            flash_scores_shared[row * config.flash_block_n + col] = scores[acc_i];
        }

        blockBarrier();

        if (lane_id < config.flash_block_m * 2) {
            const row = lane_id / 2;
            const half = lane_id % 2;
            var partial_max: f32 = -std.math.inf(f32);
            inline for (0..4) |i| {
                const col = half * 4 + i;
                partial_max = @max(partial_max, flash_scores_shared[row * config.flash_block_n + col]);
            }
            reduce_shared[lane_id] = partial_max;
        }

        blockBarrier();

        if (lane_id < config.flash_block_m) {
            const row = lane_id;
            const tile_max = @max(reduce_shared[row * 2], reduce_shared[row * 2 + 1]);
            const old_m = flash_row_max_shared[row];
            const old_l = flash_row_sum_shared[row];
            const m_new = @max(old_m, tile_max);
            const old_scale = if (old_l == 0) 0 else device.ex2_approx((old_m - m_new) * log2e);
            flash_row_max_shared[row] = m_new;
            flash_old_scale_shared[row] = old_scale;
        }

        blockBarrier();

        if (lane_id < config.flash_block_m * 2) {
            const row = lane_id / 2;
            const half = lane_id % 2;
            const m_new = flash_row_max_shared[row];
            var partial_sum: f32 = 0;

            inline for (0..4) |i| {
                const col = half * 4 + i;
                const key_row = tile_start + col;
                const q_row = q_base + row;
                const score = flash_scores_shared[row * config.flash_block_n + col];
                const visible = causal == 0 or key_row <= q_row;
                const weight = if (key_row < seq_len and q_row < seq_len and visible)
                    device.ex2_approx((score - m_new) * log2e)
                else
                    0;
                partial_sum += weight;
                shared_p_ptr[row * config.mma_k + col] = @floatCast(weight);
            }
            reduce_shared[lane_id] = partial_sum;
        }

        blockBarrier();

        if (lane_id < config.flash_block_m) {
            const row = lane_id;
            const old_l = flash_row_sum_shared[row];
            const old_scale = flash_old_scale_shared[row];
            const tile_sum = reduce_shared[row * 2] + reduce_shared[row * 2 + 1];

            inline for (config.flash_block_n..config.mma_k) |col| {
                shared_p_ptr[row * config.mma_k + col] = 0;
            }

            flash_row_sum_shared[row] = old_l * old_scale + tile_sum;
        }

        blockBarrier();

        inline for (0..4) |acc_i| {
            const coord = Backend.accumulatorCoord(lane_id, acc_i);
            const row = coord[0];
            const old_scale = flash_old_scale_shared[row];
            o_acc_0[acc_i] *= old_scale;
            o_acc_1[acc_i] *= old_scale;
            o_acc_2[acc_i] *= old_scale;
            o_acc_3[acc_i] *= old_scale;
        }

        inline for (0..4) |chunk| {
            var load_i = lane_id;
            while (load_i < config.mma_k * config.flash_block_n) : (load_i += Backend.lanes) {
                const kk = load_i / config.flash_block_n;
                const col = load_i % config.flash_block_n;
                const key_row = tile_start + kk;
                const out_col = chunk * config.flash_block_n + col;
                shared_v_ptr[load_i] = if (kk < config.flash_block_n and key_row < seq_len and out_col < head_dim)
                    v_batch[key_row * head_dim + out_col]
                else
                    0;
            }

            blockBarrier();

            const p_frag = Backend.loadA(shared_p_ptr, config.mma_k, lane_id);
            const v_frag = Backend.loadB(shared_v_ptr, config.flash_block_n, lane_id);
            const pv = Backend.mma(p_frag, v_frag, Backend.zeroAccumulator());
            inline for (0..4) |acc_i| {
                if (chunk == 0) {
                    o_acc_0[acc_i] += pv[acc_i];
                } else if (chunk == 1) {
                    o_acc_1[acc_i] += pv[acc_i];
                } else if (chunk == 2) {
                    o_acc_2[acc_i] += pv[acc_i];
                } else if (chunk == 3) {
                    o_acc_3[acc_i] += pv[acc_i];
                }
            }

            blockBarrier();
        }
    }

    inline for (0..4) |acc_i| {
        const coord = Backend.accumulatorCoord(lane_id, acc_i);
        const row = coord[0];
        const col = coord[1];
        const q_row = q_base + row;
        const denom = flash_row_sum_shared[row];

        if (q_row < seq_len) {
            if (col < head_dim) {
                o_batch[q_row * head_dim + col] = o_acc_0[acc_i] / denom;
            }
            if (col + config.flash_block_n < head_dim) {
                o_batch[q_row * head_dim + col + config.flash_block_n] = o_acc_1[acc_i] / denom;
            }
            if (col + config.flash_block_n * 2 < head_dim) {
                o_batch[q_row * head_dim + col + config.flash_block_n * 2] = o_acc_2[acc_i] / denom;
            }
            if (col + config.flash_block_n * 3 < head_dim) {
                o_batch[q_row * head_dim + col + config.flash_block_n * 3] = o_acc_3[acc_i] / denom;
            }
        }
    }
}
