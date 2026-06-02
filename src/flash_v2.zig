const std = @import("std");
const config = @import("config.zig");
const device = @import("device.zig");
const mma = @import("mma.zig");

const block_m = 16;
const block_n = 32;
const mma_n = 8;
const warps = 4;
const max_head_dim = config.flash_max_head_dim;

var flash_v2_shared_q: [block_m * max_head_dim]f16 addrspace(.shared) = undefined;
var flash_v2_shared_k: [warps * config.mma_k * mma_n]f16 addrspace(.shared) = undefined;
var flash_v2_shared_v: [config.mma_k * mma_n]f16 addrspace(.shared) = undefined;
var flash_v2_shared_p: [block_m * block_n]f16 addrspace(.shared) = undefined;
var flash_v2_shared_p_mma: [block_m * config.mma_k]f16 addrspace(.shared) = undefined;
var flash_v2_scores_shared: [block_m * block_n]f32 addrspace(.shared) = undefined;
var flash_v2_row_max_shared: [block_m]f32 addrspace(.shared) = undefined;
var flash_v2_row_sum_shared: [block_m]f32 addrspace(.shared) = undefined;
var flash_v2_old_scale_shared: [block_m]f32 addrspace(.shared) = undefined;
var flash_v2_reduce_shared: [block_m * 8]f32 addrspace(.shared) = undefined;

pub fn flash_attention_fwd_v2(
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
    if (seq_len == 0 or (head_dim != 16 and head_dim != 32 and head_dim != 64)) return;

    const Backend = mma.Backend(mma.m16n8k16_f16_f32);
    const thread_id = @workItemId(0);
    const warp_id = thread_id / Backend.lanes;
    const lane_id = thread_id % Backend.lanes;
    if (warp_id >= warps) return;

    const log2e = 1.4426950408889634;
    const q_base = @workGroupId(0) * block_m;
    const batch_id = @workGroupId(1);
    const q_batch = q + batch_id * q_stride;
    const k_batch = k + batch_id * k_stride;
    const v_batch = v + batch_id * v_stride;
    const o_batch = o + batch_id * o_stride;
    const output_chunks = (head_dim + mma_n - 1) / mma_n;

    const shared_q_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_v2_shared_q);
    const shared_k_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_v2_shared_k);
    const shared_v_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_v2_shared_v);
    const shared_p_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_v2_shared_p);
    const shared_p_mma_ptr: [*]addrspace(.shared) f16 = @ptrCast(&flash_v2_shared_p_mma);
    const shared_k_ptr = shared_k_all_ptr + warp_id * config.mma_k * mma_n;

    var o_acc: [max_head_dim / mma_n]Backend.Accumulator = undefined;
    inline for (0..max_head_dim / mma_n) |chunk| {
        o_acc[chunk] = Backend.zeroAccumulator();
    }

    var q_load_i = thread_id;
    while (q_load_i < block_m * max_head_dim) : (q_load_i += warps * Backend.lanes) {
        const row = q_load_i / max_head_dim;
        const col = q_load_i % max_head_dim;
        const q_row = q_base + row;
        shared_q_ptr[row * max_head_dim + col] = if (q_row < seq_len and col < head_dim)
            q_batch[q_row * head_dim + col]
        else
            0;
    }

    if (thread_id < block_m) {
        flash_v2_row_max_shared[thread_id] = -std.math.inf(f32);
        flash_v2_row_sum_shared[thread_id] = 0;
        flash_v2_old_scale_shared[thread_id] = 0;
    }

    blockBarrier();

    const tile_end = if (causal != 0) @min(seq_len, q_base + block_m) else seq_len;
    var tile_start: usize = 0;
    while (tile_start < tile_end) : (tile_start += block_n) {
        const n_part = warp_id;
        var scores = Backend.zeroAccumulator();

        var k_start: usize = 0;
        while (k_start < head_dim) : (k_start += config.mma_k) {
            var load_i = lane_id;
            while (load_i < config.mma_k * mma_n) : (load_i += Backend.lanes) {
                const kk = load_i / mma_n;
                const col = load_i % mma_n;
                const key_row = tile_start + n_part * mma_n + col;
                const key_col = k_start + kk;
                shared_k_ptr[load_i] = if (key_row < seq_len and key_col < head_dim)
                    k_batch[key_row * head_dim + key_col]
                else
                    0;
            }

            blockBarrier();

            const q_frag = Backend.loadA(shared_q_ptr + k_start, max_head_dim, lane_id);
            const k_frag = Backend.loadB(shared_k_ptr, mma_n, lane_id);
            scores = Backend.mma(q_frag, k_frag, scores);

            blockBarrier();
        }

        inline for (0..4) |acc_i| {
            const coord = Backend.accumulatorCoord(lane_id, acc_i);
            const row = coord[0];
            const col = coord[1];
            const block_col = n_part * mma_n + col;
            const key_row = tile_start + block_col;
            const q_row = q_base + row;
            const visible = causal == 0 or key_row <= q_row;
            const score = if (q_row < seq_len and key_row < seq_len and visible)
                scores[acc_i] * scale
            else
                -std.math.inf(f32);
            flash_v2_scores_shared[row * block_n + block_col] = score;
        }

        blockBarrier();

        if (thread_id < block_m * 8) {
            const row = thread_id / 8;
            const group = thread_id % 8;
            var partial_max: f32 = -std.math.inf(f32);
            inline for (0..4) |i| {
                const col = group * 4 + i;
                partial_max = @max(partial_max, flash_v2_scores_shared[row * block_n + col]);
            }
            flash_v2_reduce_shared[thread_id] = partial_max;
        }

        blockBarrier();

        if (thread_id < block_m) {
            const row = thread_id;
            var tile_max = flash_v2_reduce_shared[row * 8];
            inline for (1..8) |i| {
                tile_max = @max(tile_max, flash_v2_reduce_shared[row * 8 + i]);
            }
            const old_m = flash_v2_row_max_shared[row];
            const old_l = flash_v2_row_sum_shared[row];
            const m_new = @max(old_m, tile_max);
            const old_scale = if (old_l == 0) 0 else device.ex2_approx((old_m - m_new) * log2e);
            flash_v2_row_max_shared[row] = m_new;
            flash_v2_old_scale_shared[row] = old_scale;
        }

        blockBarrier();

        if (thread_id < block_m * 8) {
            const row = thread_id / 8;
            const group = thread_id % 8;
            const m_new = flash_v2_row_max_shared[row];
            var partial_sum: f32 = 0;

            inline for (0..4) |i| {
                const col = group * 4 + i;
                const key_row = tile_start + col;
                const q_row = q_base + row;
                const score = flash_v2_scores_shared[row * block_n + col];
                const visible = causal == 0 or key_row <= q_row;
                const weight = if (key_row < seq_len and q_row < seq_len and visible)
                    device.ex2_approx((score - m_new) * log2e)
                else
                    0;
                partial_sum += weight;
                shared_p_ptr[row * block_n + col] = @floatCast(weight);
            }
            flash_v2_reduce_shared[thread_id] = partial_sum;
        }

        blockBarrier();

        if (thread_id < block_m) {
            const row = thread_id;
            var tile_sum: f32 = 0;
            inline for (0..8) |i| {
                tile_sum += flash_v2_reduce_shared[row * 8 + i];
            }
            flash_v2_row_sum_shared[row] = flash_v2_row_sum_shared[row] * flash_v2_old_scale_shared[row] + tile_sum;
        }

        blockBarrier();

        if (warp_id == 0) {
            inline for (0..4) |acc_i| {
                const row = Backend.accumulatorCoord(lane_id, acc_i)[0];
                const old_scale = flash_v2_old_scale_shared[row];
                inline for (0..max_head_dim / mma_n) |chunk| {
                    if (chunk < output_chunks) {
                        o_acc[chunk][acc_i] *= old_scale;
                    }
                }
            }
        }

        inline for (0..max_head_dim / mma_n) |chunk| {
            if (chunk < output_chunks) {
                inline for (0..block_n / config.mma_k) |pv_part| {
                    if (warp_id == 0) {
                        var load_i = lane_id;
                        while (load_i < config.mma_k * mma_n) : (load_i += Backend.lanes) {
                            const kk = load_i / mma_n;
                            const col = load_i % mma_n;
                            const key_row = tile_start + pv_part * config.mma_k + kk;
                            const out_col = chunk * mma_n + col;
                            shared_v_ptr[load_i] = if (key_row < seq_len and out_col < head_dim)
                                v_batch[key_row * head_dim + out_col]
                            else
                                0;
                        }

                        var p_load_i = lane_id;
                        while (p_load_i < block_m * config.mma_k) : (p_load_i += Backend.lanes) {
                            const row = p_load_i / config.mma_k;
                            const col = p_load_i % config.mma_k;
                            shared_p_mma_ptr[row * config.mma_k + col] = shared_p_ptr[row * block_n + pv_part * config.mma_k + col];
                        }
                    }

                    blockBarrier();

                    if (warp_id == 0) {
                        const p_frag = Backend.loadA(shared_p_mma_ptr, config.mma_k, lane_id);
                        const v_frag = Backend.loadB(shared_v_ptr, mma_n, lane_id);
                        const pv = Backend.mma(p_frag, v_frag, Backend.zeroAccumulator());
                        inline for (0..4) |acc_i| {
                            o_acc[chunk][acc_i] += pv[acc_i];
                        }
                    }

                    blockBarrier();
                }
            }
        }
    }

    if (warp_id == 0) {
        inline for (0..4) |acc_i| {
            const coord = Backend.accumulatorCoord(lane_id, acc_i);
            const row = coord[0];
            const col = coord[1];
            const q_row = q_base + row;
            const denom = flash_v2_row_sum_shared[row];

            if (q_row < seq_len) {
                inline for (0..max_head_dim / mma_n) |chunk| {
                    if (chunk < output_chunks) {
                        const out_col = col + mma_n * chunk;
                        if (out_col < head_dim) {
                            o_batch[q_row * head_dim + out_col] = o_acc[chunk][acc_i] / denom;
                        }
                    }
                }
            }
        }
    }
}

fn blockBarrier() void {
    asm volatile ("bar.sync 0;");
}
