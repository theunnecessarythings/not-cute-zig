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
