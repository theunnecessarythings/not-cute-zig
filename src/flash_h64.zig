const std = @import("std");
const config = @import("config.zig");
const device = @import("device.zig");
const mma = @import("mma.zig");

const head_dim_static = 64;
const output_chunks = head_dim_static / config.flash_mma_n;

var flash_h64_shared_q: [config.flash_warps * config.flash_block_m * config.mma_k]f16 addrspace(.shared) = undefined;
var flash_h64_shared_k: [config.flash_warps * config.mma_k * config.flash_mma_n]f16 addrspace(.shared) = undefined;
var flash_h64_shared_v: [config.flash_warps * config.mma_k * config.flash_mma_n]f16 addrspace(.shared) = undefined;
var flash_h64_shared_p: [config.flash_block_m * config.mma_k]f16 addrspace(.shared) = undefined;
var flash_h64_shared_p_mma: [config.flash_warps * config.flash_block_m * config.mma_k]f16 addrspace(.shared) = undefined;
var flash_h64_scores_shared: [config.flash_block_m * config.flash_block_n]f32 addrspace(.shared) = undefined;
var flash_h64_row_max_shared: [config.flash_block_m]f32 addrspace(.shared) = undefined;
var flash_h64_row_sum_shared: [config.flash_block_m]f32 addrspace(.shared) = undefined;
var flash_h64_old_scale_shared: [config.flash_block_m]f32 addrspace(.shared) = undefined;
var flash_h64_reduce_shared: [64]f32 addrspace(.shared) = undefined;

pub fn flash_attention_fwd_h64(
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
    if (seq_len == 0 or head_dim != head_dim_static) return;

    const Backend = mma.Backend(mma.m16n8k16_f16_f32);
    const thread_id = @workItemId(0);
    const warp_id = thread_id / Backend.lanes;
    const lane_id = thread_id % Backend.lanes;
    if (warp_id >= config.flash_warps) return;

    const log2e = 1.4426950408889634;
    const q_base = @workGroupId(0) * config.flash_block_m;
    const batch_id = @workGroupId(1);
    const q_batch = q + batch_id * q_stride;
    const k_batch = k + batch_id * k_stride;
    const v_batch = v + batch_id * v_stride;
    const o_batch = o + batch_id * o_stride;

    const shared_q_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_h64_shared_q);
    const shared_k_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_h64_shared_k);
    const shared_v_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_h64_shared_v);
    const shared_p_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_h64_shared_p);
    const shared_p_mma_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_h64_shared_p_mma);
    const shared_q_ptr = shared_q_all_ptr + warp_id * config.flash_block_m * config.mma_k;
    const shared_k_ptr = shared_k_all_ptr + warp_id * config.mma_k * config.flash_mma_n;
    const shared_v_ptr = shared_v_all_ptr + warp_id * config.mma_k * config.flash_mma_n;
    const shared_p_mma_ptr = shared_p_mma_all_ptr + warp_id * config.flash_block_m * config.mma_k;

    var o_acc: [output_chunks]Backend.Accumulator = undefined;
    inline for (0..output_chunks) |chunk| {
        o_acc[chunk] = Backend.zeroAccumulator();
    }

    if (thread_id < config.flash_block_m) {
        flash_h64_row_max_shared[thread_id] = -std.math.inf(f32);
        flash_h64_row_sum_shared[thread_id] = 0;
        flash_h64_old_scale_shared[thread_id] = 0;
    }

    blockBarrier();

    const tile_end = if (causal != 0)
        @min(seq_len, q_base + config.flash_block_m)
    else
        seq_len;

    var tile_start: usize = 0;
    while (tile_start < tile_end) : (tile_start += config.flash_block_n) {
        const n_part = warp_id;
        var scores = Backend.zeroAccumulator();

        var k_start: usize = 0;
        while (k_start < head_dim_static) : (k_start += config.mma_k) {
            var load_i = lane_id;
            while (load_i < config.flash_block_m * config.mma_k) : (load_i += Backend.lanes) {
                const row = load_i / config.mma_k;
                const col = load_i % config.mma_k;
                const q_row = q_base + row;
                shared_q_ptr[row * config.mma_k + col] = if (q_row < seq_len)
                    q_batch[q_row * head_dim_static + k_start + col]
                else
                    0;
            }

            load_i = lane_id;
            while (load_i < config.mma_k * config.flash_mma_n) : (load_i += Backend.lanes) {
                const kk = load_i / config.flash_mma_n;
                const col = load_i % config.flash_mma_n;
                const key_row = tile_start + n_part * config.flash_mma_n + col;
                shared_k_ptr[load_i] = if (key_row < seq_len)
                    k_batch[key_row * head_dim_static + k_start + kk]
                else
                    0;
            }

            blockBarrier();

            const q_frag = Backend.loadA(shared_q_ptr, config.mma_k, lane_id);
            const k_frag = Backend.loadB(shared_k_ptr, config.flash_mma_n, lane_id);
            scores = Backend.mma(q_frag, k_frag, scores);

            blockBarrier();
        }

        inline for (0..4) |acc_i| {
            const coord = Backend.accumulatorCoord(lane_id, acc_i);
            const row = coord[0];
            const col = coord[1];
            const block_col = n_part * config.flash_mma_n + col;
            const key_row = tile_start + block_col;
            const q_row = q_base + row;
            const visible = causal == 0 or key_row <= q_row;
            scores[acc_i] = if (q_row < seq_len and key_row < seq_len and visible)
                scores[acc_i] * scale
            else
                -std.math.inf(f32);
            flash_h64_scores_shared[row * config.flash_block_n + block_col] = scores[acc_i];
        }

        blockBarrier();

        if (thread_id < config.flash_block_m * 4) {
            const row = thread_id / 4;
            const quarter = thread_id % 4;
            var partial_max: f32 = -std.math.inf(f32);
            inline for (0..4) |i| {
                const col = quarter * 4 + i;
                partial_max = @max(partial_max, flash_h64_scores_shared[row * config.flash_block_n + col]);
            }
            flash_h64_reduce_shared[thread_id] = partial_max;
        }

        blockBarrier();

        if (thread_id < config.flash_block_m) {
            const row = thread_id;
            const tile_max = @max(@max(flash_h64_reduce_shared[row * 4], flash_h64_reduce_shared[row * 4 + 1]), @max(flash_h64_reduce_shared[row * 4 + 2], flash_h64_reduce_shared[row * 4 + 3]));
            const old_m = flash_h64_row_max_shared[row];
            const old_l = flash_h64_row_sum_shared[row];
            const m_new = @max(old_m, tile_max);
            const old_scale = if (old_l == 0) 0 else device.ex2_approx((old_m - m_new) * log2e);
            flash_h64_row_max_shared[row] = m_new;
            flash_h64_old_scale_shared[row] = old_scale;
        }

        blockBarrier();

        if (thread_id < config.flash_block_m * 4) {
            const row = thread_id / 4;
            const quarter = thread_id % 4;
            const m_new = flash_h64_row_max_shared[row];
            var partial_sum: f32 = 0;

            inline for (0..4) |i| {
                const col = quarter * 4 + i;
                const key_row = tile_start + col;
                const q_row = q_base + row;
                const score = flash_h64_scores_shared[row * config.flash_block_n + col];
                const visible = causal == 0 or key_row <= q_row;
                const weight = if (key_row < seq_len and q_row < seq_len and visible)
                    device.ex2_approx((score - m_new) * log2e)
                else
                    0;
                partial_sum += weight;
                shared_p_ptr[row * config.mma_k + col] = @floatCast(weight);
            }
            flash_h64_reduce_shared[thread_id] = partial_sum;
        }

        blockBarrier();

        if (thread_id < config.flash_block_m) {
            const row = thread_id;
            const old_l = flash_h64_row_sum_shared[row];
            const old_scale = flash_h64_old_scale_shared[row];
            const tile_sum = flash_h64_reduce_shared[row * 4] + flash_h64_reduce_shared[row * 4 + 1] + flash_h64_reduce_shared[row * 4 + 2] + flash_h64_reduce_shared[row * 4 + 3];

            flash_h64_row_sum_shared[row] = old_l * old_scale + tile_sum;
        }

        blockBarrier();

        inline for (0..4) |acc_i| {
            const row = Backend.accumulatorCoord(lane_id, acc_i)[0];
            const old_scale = flash_h64_old_scale_shared[row];
            inline for (0..output_chunks) |chunk| {
                if (chunk % config.flash_warps == warp_id) {
                    o_acc[chunk][acc_i] *= old_scale;
                }
            }
        }

        inline for (0..output_chunks) |chunk| {
            const owns_chunk = chunk % config.flash_warps == warp_id;
            if (owns_chunk) {
                var load_i = lane_id;
                while (load_i < config.mma_k * config.flash_mma_n) : (load_i += Backend.lanes) {
                    const kk = load_i / config.flash_mma_n;
                    const col = load_i % config.flash_mma_n;
                    const key_row = tile_start + kk;
                    const out_col = chunk * config.flash_mma_n + col;
                    shared_v_ptr[load_i] = if (key_row < seq_len)
                        v_batch[key_row * head_dim_static + out_col]
                    else
                        0;
                }

                var p_load_i = lane_id;
                while (p_load_i < config.flash_block_m * config.mma_k) : (p_load_i += Backend.lanes) {
                    const row = p_load_i / config.mma_k;
                    const col = p_load_i % config.mma_k;
                    shared_p_mma_ptr[row * config.mma_k + col] = shared_p_ptr[row * config.mma_k + col];
                }
            }

            blockBarrier();

            if (owns_chunk) {
                const p_frag = Backend.loadA(shared_p_mma_ptr, config.mma_k, lane_id);
                const v_frag = Backend.loadB(shared_v_ptr, config.flash_mma_n, lane_id);
                const pv = Backend.mma(p_frag, v_frag, Backend.zeroAccumulator());
                inline for (0..4) |acc_i| {
                    o_acc[chunk][acc_i] += pv[acc_i];
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
        const denom = flash_h64_row_sum_shared[row];

        if (q_row < seq_len) {
            inline for (0..output_chunks) |chunk| {
                if (chunk % config.flash_warps == warp_id) {
                    const out_col = col + config.flash_mma_n * chunk;
                    o_batch[q_row * head_dim_static + out_col] = o_acc[chunk][acc_i] / denom;
                }
            }
        }
    }
}

fn blockBarrier() void {
    asm volatile ("bar.sync 0;");
}
