const std = @import("std");
const device = @import("device.zig");
const mma = @import("mma.zig");

const head_dim_static = 64;
const block_m = 16;
const block_n = 32;
const mma_k = 16;
const mma_n = 8;
const block_warps = 4;
const output_chunks = head_dim_static / mma_n;
const owned_output_chunks = output_chunks / block_warps;
const reduction_groups = block_n / 4;

var shared_q_all: [block_warps * block_m * mma_k]f16 addrspace(.shared) = undefined;
var shared_k_all: [block_warps * mma_k * mma_n]f16 addrspace(.shared) = undefined;
var shared_v_all: [block_warps * mma_k * mma_n]f16 addrspace(.shared) = undefined;
var shared_p: [block_m * block_n]f16 addrspace(.shared) = undefined;
var shared_p_mma_all: [block_warps * block_m * mma_k]f16 addrspace(.shared) = undefined;
var shared_scores: [block_m * block_n]f32 addrspace(.shared) = undefined;
var shared_row_max: [block_m]f32 addrspace(.shared) = undefined;
var shared_row_sum: [block_m]f32 addrspace(.shared) = undefined;
var shared_old_scale: [block_m]f32 addrspace(.shared) = undefined;
var shared_reduce: [block_m * reduction_groups]f32 addrspace(.shared) = undefined;

pub fn flash_attention_fwd_h64_bn32(
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
    if (warp_id >= block_warps) return;

    const warp_usize: usize = @intCast(warp_id);
    const log2e = 1.4426950408889634;
    const q_base = @workGroupId(0) * block_m;
    const batch_id = @workGroupId(1);
    const q_batch = q + batch_id * q_stride;
    const k_batch = k + batch_id * k_stride;
    const v_batch = v + batch_id * v_stride;
    const o_batch = o + batch_id * o_stride;

    const q_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&shared_q_all);
    const k_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&shared_k_all);
    const v_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&shared_v_all);
    const p_ptr: [*]addrspace(.shared) f16 = @ptrCast(&shared_p);
    const p_mma_all_ptr: [*]addrspace(.shared) f16 = @ptrCast(&shared_p_mma_all);
    const q_ptr = q_all_ptr + warp_usize * block_m * mma_k;
    const k_ptr = k_all_ptr + warp_usize * mma_k * mma_n;
    const v_ptr = v_all_ptr + warp_usize * mma_k * mma_n;
    const p_mma_ptr = p_mma_all_ptr + warp_usize * block_m * mma_k;

    var o_acc: [owned_output_chunks]Backend.Accumulator = undefined;
    inline for (0..owned_output_chunks) |local_chunk| {
        o_acc[local_chunk] = Backend.zeroAccumulator();
    }

    if (thread_id < block_m) {
        shared_row_max[thread_id] = -std.math.inf(f32);
        shared_row_sum[thread_id] = 0;
        shared_old_scale[thread_id] = 0;
    }

    blockBarrier();

    const tile_end = if (causal != 0)
        @min(seq_len, q_base + block_m)
    else
        seq_len;

    var tile_start: usize = 0;
    while (tile_start < tile_end) : (tile_start += block_n) {
        var scores = Backend.zeroAccumulator();

        var k_start: usize = 0;
        while (k_start < head_dim_static) : (k_start += mma_k) {
            var load_i = lane_id;
            while (load_i < block_m * mma_k) : (load_i += Backend.lanes) {
                const row = load_i / mma_k;
                const col = load_i % mma_k;
                const q_row = q_base + row;
                q_ptr[row * mma_k + col] = if (q_row < seq_len)
                    q_batch[q_row * head_dim_static + k_start + col]
                else
                    0;
            }

            load_i = lane_id;
            while (load_i < mma_k * mma_n) : (load_i += Backend.lanes) {
                const kk = load_i / mma_n;
                const col = load_i % mma_n;
                const key_row = tile_start + warp_usize * mma_n + col;
                k_ptr[load_i] = if (key_row < seq_len)
                    k_batch[key_row * head_dim_static + k_start + kk]
                else
                    0;
            }

            blockBarrier();

            const q_frag = Backend.loadA(q_ptr, mma_k, lane_id);
            const k_frag = Backend.loadB(k_ptr, mma_n, lane_id);
            scores = Backend.mma(q_frag, k_frag, scores);

            blockBarrier();
        }

        inline for (0..4) |acc_i| {
            const coord = Backend.accumulatorCoord(lane_id, acc_i);
            const row = coord[0];
            const col = coord[1];
            const block_col = warp_usize * mma_n + col;
            const key_row = tile_start + block_col;
            const q_row = q_base + row;
            const visible = causal == 0 or key_row <= q_row;
            scores[acc_i] = if (q_row < seq_len and key_row < seq_len and visible)
                scores[acc_i] * scale
            else
                -std.math.inf(f32);
            shared_scores[row * block_n + block_col] = scores[acc_i];
        }

        blockBarrier();

        if (thread_id < block_m * reduction_groups) {
            const row = thread_id / reduction_groups;
            const group = thread_id % reduction_groups;
            var partial_max: f32 = -std.math.inf(f32);
            inline for (0..4) |i| {
                const col = group * 4 + i;
                partial_max = @max(partial_max, shared_scores[row * block_n + col]);
            }
            shared_reduce[thread_id] = partial_max;
        }

        blockBarrier();

        if (thread_id < block_m) {
            const row = thread_id;
            var tile_max: f32 = -std.math.inf(f32);
            inline for (0..reduction_groups) |group| {
                tile_max = @max(tile_max, shared_reduce[row * reduction_groups + group]);
            }
            const old_m = shared_row_max[row];
            const old_l = shared_row_sum[row];
            const m_new = @max(old_m, tile_max);
            const old_scale = if (old_l == 0) 0 else device.ex2_approx((old_m - m_new) * log2e);
            shared_row_max[row] = m_new;
            shared_old_scale[row] = old_scale;
        }

        blockBarrier();

        if (thread_id < block_m * reduction_groups) {
            const row = thread_id / reduction_groups;
            const group = thread_id % reduction_groups;
            const m_new = shared_row_max[row];
            var partial_sum: f32 = 0;

            inline for (0..4) |i| {
                const col = group * 4 + i;
                const key_row = tile_start + col;
                const q_row = q_base + row;
                const score = shared_scores[row * block_n + col];
                const visible = causal == 0 or key_row <= q_row;
                const weight = if (key_row < seq_len and q_row < seq_len and visible)
                    device.ex2_approx((score - m_new) * log2e)
                else
                    0;
                partial_sum += weight;
                p_ptr[row * block_n + col] = @floatCast(weight);
            }
            shared_reduce[thread_id] = partial_sum;
        }

        blockBarrier();

        if (thread_id < block_m) {
            const row = thread_id;
            const old_l = shared_row_sum[row];
            const old_scale = shared_old_scale[row];
            var tile_sum: f32 = 0;
            inline for (0..reduction_groups) |group| {
                tile_sum += shared_reduce[row * reduction_groups + group];
            }
            shared_row_sum[row] = old_l * old_scale + tile_sum;
        }

        blockBarrier();

        inline for (0..4) |acc_i| {
            const row = Backend.accumulatorCoord(lane_id, acc_i)[0];
            const old_scale = shared_old_scale[row];
            inline for (0..owned_output_chunks) |local_chunk| {
                o_acc[local_chunk][acc_i] *= old_scale;
            }
        }

        inline for (0..owned_output_chunks) |local_chunk| {
            const chunk = local_chunk * block_warps + warp_usize;
            inline for (0..block_n / mma_k) |pv_k_part| {
                var load_i = lane_id;
                while (load_i < mma_k * mma_n) : (load_i += Backend.lanes) {
                    const kk = load_i / mma_n;
                    const col = load_i % mma_n;
                    const key_row = tile_start + pv_k_part * mma_k + kk;
                    const out_col = chunk * mma_n + col;
                    v_ptr[load_i] = if (key_row < seq_len)
                        v_batch[key_row * head_dim_static + out_col]
                    else
                        0;
                }

                var p_load_i = lane_id;
                while (p_load_i < block_m * mma_k) : (p_load_i += Backend.lanes) {
                    const row = p_load_i / mma_k;
                    const col = p_load_i % mma_k;
                    p_mma_ptr[row * mma_k + col] = p_ptr[row * block_n + pv_k_part * mma_k + col];
                }

                warpBarrier();

                const p_frag = Backend.loadA(p_mma_ptr, mma_k, lane_id);
                const v_frag = Backend.loadB(v_ptr, mma_n, lane_id);
                o_acc[local_chunk] = Backend.mma(p_frag, v_frag, o_acc[local_chunk]);

                warpBarrier();
            }
        }
    }

    inline for (0..4) |acc_i| {
        const coord = Backend.accumulatorCoord(lane_id, acc_i);
        const row = coord[0];
        const col = coord[1];
        const q_row = q_base + row;
        const denom = shared_row_sum[row];

        if (q_row < seq_len) {
            inline for (0..owned_output_chunks) |local_chunk| {
                const chunk = local_chunk * block_warps + warp_usize;
                const out_col = col + mma_n * chunk;
                o_batch[q_row * head_dim_static + out_col] = o_acc[local_chunk][acc_i] / denom;
            }
        }
    }
}

fn blockBarrier() void {
    asm volatile ("bar.sync 0;");
}

fn warpBarrier() void {
    asm volatile ("bar.warp.sync 0xffffffff;");
}
