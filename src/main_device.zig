//! Entry-point wrapper for device code. This file should contain
//! everything that should be exported for the device binary.

const std = @import("std");
const kernels = @import("kernels.zig");
const flash_opt = @import("flash_opt.zig");
const flash_v2 = @import("flash_v2.zig");
const flash_h64 = @import("flash_h64.zig");
const flash_h64_causal = @import("flash_h64_causal.zig");

// Custom panic handler, to prevent stack traces etc on this target.
pub fn panic(msg: []const u8, stack_trace: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    _ = msg;
    _ = stack_trace;
    unreachable;
}

fn exportKernels(comptime namespace: type) void {
    inline for (@typeInfo(namespace).@"struct".decls) |decl| {
        const value = @field(namespace, decl.name);
        const Value = @TypeOf(value);

        switch (@typeInfo(Value)) {
            .@"fn" => |func| {
                if (std.meta.activeTag(func.calling_convention) == std.meta.activeTag(std.builtin.CallingConvention.kernel)) {
                    @export(&value, .{ .name = decl.name });
                }
            },
            .type => {
                if (@typeInfo(value) == .@"struct") {
                    exportKernels(value);
                }
            },
            else => {},
        }
    }
}

comptime {
    exportKernels(kernels);
    exportKernels(flash_opt);
    exportKernels(flash_v2);
    exportKernels(flash_h64);
    exportKernels(flash_h64_causal);
}
