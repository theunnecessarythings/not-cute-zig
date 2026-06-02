//! Public package entrypoint for not-cute.
//!
//! Import this module from downstream `build.zig` files with:
//!
//! ```zig
//! const not_cute = @import("not-cute");
//! ```
//!
//! The modules exported here are the supported library surface. Demo kernels
//! and CLI-only code live outside this entrypoint.

pub const benchmark = @import("benchmark.zig");
pub const config = @import("config.zig");
pub const cuda = @import("cuda.zig");
pub const device = @import("device.zig");
pub const flash = @import("flash.zig");
pub const layout = @import("layout.zig");
pub const mma = @import("mma.zig");

test {
    _ = benchmark;
    _ = config;
    _ = cuda;
    _ = device;
    _ = flash;
    _ = layout;
    _ = mma;
}
