const std = @import("std");

/// Device-only primitives for CUDA kernels.
/// This file should only be imported by files compiled for the nvptx64 target.
/// Synchronize all threads in a block.
pub inline fn syncthreads() void {
    asm volatile ("bar.sync 0;");
}

/// Shuffle sync: returns the value of `val` from the thread at `src_lane`.
pub inline fn shfl_sync_idx(mask: u32, val: u32, src_lane: u32, width: u32) u32 {
    var ret: u32 = undefined;
    const c_val = ((32 - width) << 8) | 31;
    asm volatile ("shfl.sync.idx.b32 %[ret], %[val], %[src_lane], %[c], %[mask];"
        : [ret] "=r" (ret),
        : [val] "r" (val),
          [src_lane] "r" (src_lane),
          [c] "r" (c_val),
          [mask] "r" (mask),
    );
    return ret;
}

/// Shuffle sync butterfly: returns the value of `val` from the thread at `lane_id ^ mask`.
pub inline fn shfl_sync_bfly(mask: u32, val: u32, bfly_mask: u32, width: u32) u32 {
    var ret: u32 = undefined;
    const c_val = ((32 - width) << 8) | 31;
    asm volatile ("shfl.sync.bfly.b32 %[ret], %[val], %[bfly_mask], %[c], %[mask];"
        : [ret] "=r" (ret),
        : [val] "r" (val),
          [bfly_mask] "r" (bfly_mask),
          [c] "r" (c_val),
          [mask] "r" (mask),
    );
    return ret;
}

/// Sleep for approximately `clocks` nanoseconds.
pub inline fn nanosleep(clocks: u32) void {
    asm volatile ("nanosleep.u32 %[c];"
        :
        : [c] "r" (clocks),
    );
}

/// Computes the exponential 2^x using the fast approximation instruction.
pub inline fn ex2_approx(x: f32) f32 {
    var ret: f32 = undefined;
    asm volatile ("ex2.approx.ftz.f32 %[ret], %[x];"
        : [ret] "=f" (ret),
        : [x] "f" (x),
    );
    return ret;
}

/// Applies a ReLU activation: max(0, x)
pub inline fn relu(x: f32) f32 {
    return @max(0.0, x);
}

/// Applies a Swish/SiLU activation: x * sigmoid(x)
pub inline fn silu(x: f32) f32 {
    const log2e = 1.4426950408889634; // log2(e)
    const exp_x = ex2_approx(x * log2e);
    return x * (exp_x / (1.0 + exp_x));
}

/// Applies a fast GeLU approximation activation
pub inline fn gelu(x: f32) f32 {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // using fast exp2
    const sqrt_2_over_pi = 0.7978845608;
    const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    const log2e = 1.4426950408889634; // log2(e)
    // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    const exp_2z = ex2_approx(2.0 * inner * log2e);
    const tanh_approx = (exp_2z - 1.0) / (exp_2z + 1.0);
    return 0.5 * x * (1.0 + tanh_approx);
}

/// SM80+ asynchronous memory copy from global to shared memory.
pub const cp_async = struct {
    /// Copies 16 bytes (128 bits) from global to shared memory.
    pub inline fn cg_16(dst_shared: [*]addrspace(.shared) u8, src_global: [*]addrspace(.global) const u8) void {
        const dst_ptr = @as(u32, @truncate(@intFromPtr(dst_shared)));
        const src_ptr = @intFromPtr(src_global);
        asm volatile ("cp.async.cg.shared.global [%[dst]], [%[src]], 16;"
            :
            : [dst] "r" (dst_ptr),
              [src] "l" (src_ptr),
            : .{ .memory = true });
    }

    /// Commits all prior `cp.async` instructions to the current group.
    pub inline fn commit_group() void {
        asm volatile ("cp.async.commit_group;");
    }

    /// Waits until at most `n` asynchronous copy groups are still in flight.
    pub inline fn wait_group(comptime n: u32) void {
        asm volatile ("cp.async.wait_group %[n];"
            :
            : [n] "n" (n),
            : .{ .memory = true });
    }
};
