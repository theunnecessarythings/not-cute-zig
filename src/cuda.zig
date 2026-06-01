const std = @import("std");
const assert = std.debug.assert;

pub const c = struct {
    pub const CUDA_SUCCESS = 0;
    pub const CUDA_ERROR_INVALID_VALUE = 1;
    pub const CUDA_ERROR_OUT_OF_MEMORY = 2;
    pub const CUDA_ERROR_NOT_INITIALIZED = 3;
    pub const CUDA_ERROR_DEINITIALIZED = 4;
    pub const CUDA_ERROR_PROFILER_DISABLED = 5;
    pub const CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6;
    pub const CUDA_ERROR_PROFILER_ALREADY_STARTED = 7;
    pub const CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8;
    pub const CUDA_ERROR_NO_DEVICE = 100;
    pub const CUDA_ERROR_INVALID_DEVICE = 101;
    pub const CUDA_ERROR_INVALID_IMAGE = 200;
    pub const CUDA_ERROR_INVALID_CONTEXT = 201;
    pub const CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202;
    pub const CUDA_ERROR_MAP_FAILED = 205;
    pub const CUDA_ERROR_UNMAP_FAILED = 206;
    pub const CUDA_ERROR_ARRAY_IS_MAPPED = 207;
    pub const CUDA_ERROR_ALREADY_MAPPED = 208;
    pub const CUDA_ERROR_NO_BINARY_FOR_GPU = 209;
    pub const CUDA_ERROR_ALREADY_ACQUIRED = 210;
    pub const CUDA_ERROR_NOT_MAPPED = 211;
    pub const CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212;
    pub const CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213;
    pub const CUDA_ERROR_ECC_UNCORRECTABLE = 214;
    pub const CUDA_ERROR_UNSUPPORTED_LIMIT = 215;
    pub const CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216;
    pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217;
    pub const CUDA_ERROR_INVALID_PTX = 218;
    pub const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219;
    pub const CUDA_ERROR_NVLINK_UNCORRECTABLE = 220;
    pub const CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221;
    pub const CUDA_ERROR_INVALID_SOURCE = 300;
    pub const CUDA_ERROR_FILE_NOT_FOUND = 301;
    pub const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302;
    pub const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303;
    pub const CUDA_ERROR_OPERATING_SYSTEM = 304;
    pub const CUDA_ERROR_INVALID_HANDLE = 400;
    pub const CUDA_ERROR_NOT_FOUND = 500;
    pub const CUDA_ERROR_NOT_READY = 600;
    pub const CUDA_ERROR_ILLEGAL_ADDRESS = 700;
    pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701;
    pub const CUDA_ERROR_LAUNCH_TIMEOUT = 702;
    pub const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703;
    pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704;
    pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705;
    pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708;
    pub const CUDA_ERROR_CONTEXT_IS_DESTROYED = 709;
    pub const CUDA_ERROR_ASSERT = 710;
    pub const CUDA_ERROR_TOO_MANY_PEERS = 711;
    pub const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712;
    pub const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713;
    pub const CUDA_ERROR_HARDWARE_STACK_ERROR = 714;
    pub const CUDA_ERROR_ILLEGAL_INSTRUCTION = 715;
    pub const CUDA_ERROR_MISALIGNED_ADDRESS = 716;
    pub const CUDA_ERROR_INVALID_ADDRESS_SPACE = 717;
    pub const CUDA_ERROR_INVALID_PC = 718;
    pub const CUDA_ERROR_LAUNCH_FAILED = 719;
    pub const CUDA_ERROR_UNKNOWN = 999;

    pub const CUresult = c_uint;
    pub const CUmodule = *opaque {};
    pub const CUfunction = *opaque {};
    pub const CUstream = *opaque {};
    pub const CUevent = *opaque {};
    pub const CUdevice = *opaque {};
    pub const CUcontext = *opaque {};

    pub extern fn cuGetErrorName(err: CUresult, msg: *[*:0]const u8) CUresult;
    pub extern fn cuInit(flags: c_uint) CUresult;
    pub extern fn cuDeviceGetCount(count: *c_int) CUresult;
    pub extern fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
    pub extern fn cuCtxCreate(context: *CUcontext, flags: c_uint, device: CUdevice) CUresult;
    pub extern fn cuCtxSynchronize() CUresult;
    pub extern fn cuStreamCreate(stream: *CUstream, flags: c_uint) CUresult;
    pub extern fn cuStreamDestroy(stream: CUstream) CUresult;
    pub extern fn cuStreamSynchronize(stream: CUstream) CUresult;
    pub extern fn cuMemAlloc(ptr: **anyopaque, size: usize) CUresult;
    pub extern fn cuMemFree(ptr: *anyopaque) CUresult;
    pub extern fn cuMemcpyHtoD(dst_dev: *anyopaque, src_host: *const anyopaque, size: usize) CUresult;
    pub extern fn cuMemcpyDtoH(dst_host: *anyopaque, src_dev: *const anyopaque, size: usize) CUresult;
    pub extern fn cuEventCreate(event: *CUevent, flags: c_uint) CUresult;
    pub extern fn cuEventDestroy(event: CUevent) CUresult;
    pub extern fn cuEventRecord(event: CUevent, stream: ?CUstream) CUresult;
    pub extern fn cuEventSynchronize(event: CUevent) CUresult;
    pub extern fn cuEventElapsedTime(result: *f32, a: CUevent, b: CUevent) CUresult;
    pub extern fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
    pub extern fn cuModuleUnload(module: CUmodule) CUresult;
    pub extern fn cuModuleGetFunction(function: *CUfunction, module: CUmodule, name: [*:0]const u8) CUresult;
    pub extern fn cuLaunchKernel(
        function: CUfunction,
        gdx: c_uint,
        gdy: c_uint,
        gdz: c_uint,
        bdx: c_uint,
        bdy: c_uint,
        bdz: c_uint,
        shmem: c_uint,
        stream: ?CUstream,
        params: ?[*]?*anyopaque,
        extra: ?[*]?*anyopaque,
    ) CUresult;
};

/// High-level Zig error set mapped to CUDA driver API errors.
pub const CudaError = error{
    OutOfMemory,
    SharedObjectInitFailed,
    NotFound,
    InvalidValue,
    InvalidContext,
    InvalidDevice,
    InvalidImage,
    ContextIsDestroyed,
    LaunchFailed,
    LaunchOutOfResources,
    LaunchTimeout,
    IllegalAddress,
    NoDevice,
    UnknownError,
};

/// Validates a CUDA result code and maps it to a standard Zig CudaError.
pub fn check(err: c.CUresult) CudaError!void {
    if (err == c.CUDA_SUCCESS) return;
    var msg: [*:0]const u8 = undefined;
    _ = c.cuGetErrorName(err, &msg);
    std.log.err("CUDA error: {s} ({})", .{ msg, err });
    return switch (err) {
        c.CUDA_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        c.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => error.SharedObjectInitFailed,
        c.CUDA_ERROR_NOT_FOUND => error.NotFound,
        c.CUDA_ERROR_INVALID_VALUE => error.InvalidValue,
        c.CUDA_ERROR_INVALID_CONTEXT => error.InvalidContext,
        c.CUDA_ERROR_INVALID_DEVICE => error.InvalidDevice,
        c.CUDA_ERROR_INVALID_IMAGE => error.InvalidImage,
        c.CUDA_ERROR_CONTEXT_IS_DESTROYED => error.ContextIsDestroyed,
        c.CUDA_ERROR_LAUNCH_FAILED => error.LaunchFailed,
        c.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => error.LaunchOutOfResources,
        c.CUDA_ERROR_LAUNCH_TIMEOUT => error.LaunchTimeout,
        c.CUDA_ERROR_ILLEGAL_ADDRESS => error.IllegalAddress,
        c.CUDA_ERROR_NO_DEVICE => error.NoDevice,
        else => error.UnknownError,
    };
}

/// Initializes the CUDA driver API. Must be called before any other CUDA functions.
pub fn init() !void {
    try check(c.cuInit(0));

    var count: c_int = undefined;
    try check(c.cuDeviceGetCount(&count));

    var device: c.CUdevice = undefined;
    try check(c.cuDeviceGet(&device, 0));

    var context: c.CUcontext = undefined;
    try check(c.cuCtxCreate(&context, 0, device));
}

/// Allocates `n` elements of type `T` on the device.
pub fn malloc(comptime T: type, n: usize) ![]T {
    var result: usize = 0; // cuda driver does not write to the upper bytes, so initialize as zero!
    try check(c.cuMemAlloc(@ptrCast(&result), n * @sizeOf(T)));
    return @as([*]T, @ptrFromInt(result))[0..n];
}

/// Frees memory allocated on the device.
pub fn free(ptr: anytype) void {
    const actual_ptr = switch (@typeInfo(@TypeOf(ptr)).pointer.size) {
        .slice => ptr.ptr,
        else => ptr,
    };
    check(c.cuMemFree(actual_ptr)) catch |err| {
        std.log.warn("Failed to free CUDA memory: {}", .{err});
    };
}

pub const CopyDir = enum {
    host_to_device,
    device_to_host,
};

/// Copies memory between host and device.
pub fn memcpy(comptime T: type, dst: []T, src: []const T, direction: CopyDir) !void {
    assert(dst.len >= src.len);
    switch (direction) {
        .host_to_device => try check(c.cuMemcpyHtoD(dst.ptr, src.ptr, @sizeOf(T) * src.len)),
        .device_to_host => try check(c.cuMemcpyDtoH(dst.ptr, src.ptr, @sizeOf(T) * src.len)),
    }
}

/// Represents a loaded CUDA module (typically a PTX or cubin file).
pub const Module = struct {
    handle: c.CUmodule,

    /// Loads a CUDA module from a data buffer (e.g., embedded PTX).
    pub fn loadData(image: *const anyopaque) !Module {
        var module: Module = undefined;
        try check(c.cuModuleLoadData(&module.handle, image));
        return module;
    }

    /// Unloads a loaded CUDA module.
    pub fn unload(self: Module) void {
        check(c.cuModuleUnload(self.handle)) catch |err| {
            std.log.warn("Failed to unload CUDA module: {}", .{err});
        };
    }

    /// Retrieves a kernel function by name from the module.
    pub fn getFunction(self: Module, name: [*:0]const u8) !Function {
        var function: Function = undefined;
        try check(c.cuModuleGetFunction(&function.handle, self.handle, name));
        return function;
    }
};

/// CUDA Stream for concurrent execution.
pub const Stream = struct {
    handle: c.CUstream,

    /// Creates a new asynchronous stream.
    pub fn create() !Stream {
        var stream: Stream = undefined;
        // CUDA_STREAM_NON_BLOCKING = 1
        try check(c.cuStreamCreate(&stream.handle, 1));
        return stream;
    }

    /// Destroys a stream.
    pub fn destroy(self: Stream) void {
        check(c.cuStreamDestroy(self.handle)) catch |err| {
            std.log.warn("Failed to destroy CUDA stream: {}", .{err});
        };
    }

    /// Blocks the host until the stream has completed all operations.
    pub fn synchronize(self: Stream) !void {
        try check(c.cuStreamSynchronize(self.handle));
    }
};

/// Configuration for launching a CUDA kernel.
pub const LaunchConfig = struct {
    grid_dim: Dim3 = .{},
    block_dim: Dim3 = .{},
    shared_mem_per_block: u32 = 0,
    stream: ?c.CUstream = null,
};

/// Represents 3D dimensions for thread blocks and grids.
pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,
};

/// Represents a CUDA kernel function.
pub const Function = struct {
    handle: c.CUfunction,

    /// Launches the kernel with the specified configuration and arguments.
    pub fn launch(self: Function, cfg: LaunchConfig, args: anytype) !void {
        const Args = blk: {
            comptime var fields: [args.len]type = undefined;
            inline for (&fields, 0..) |*field, i| {
                field.* = switch (@typeInfo(@TypeOf(args[i]))) {
                    .comptime_int => usize,
                    .comptime_float => f64,
                    else => @TypeOf(args[i]),
                };
            }
            break :blk std.meta.Tuple(&fields);
        };

        var kernel_args: Args = args;
        var args_buf: [args.len]?*anyopaque = undefined;
        inline for (&args_buf, 0..) |*arg_buf, i| {
            arg_buf.* = @ptrCast(&kernel_args[i]);
        }

        try check(c.cuLaunchKernel(
            self.handle,
            cfg.grid_dim.x,
            cfg.grid_dim.y,
            cfg.grid_dim.z,
            cfg.block_dim.x,
            cfg.block_dim.y,
            cfg.block_dim.z,
            cfg.shared_mem_per_block,
            cfg.stream,
            &args_buf,
            null,
        ));
    }
};

/// CUDA Event for timing and synchronization.
pub const Event = struct {
    handle: c.CUevent,

    /// Creates a new CUDA event.
    pub fn create() !Event {
        var event: Event = undefined;
        try check(c.cuEventCreate(&event.handle, 0));
        return event;
    }

    /// Destroys a CUDA event.
    pub fn destroy(self: Event) void {
        check(c.cuEventDestroy(self.handle)) catch |err| {
            std.log.warn("Failed to destroy CUDA event: {}", .{err});
        };
    }

    /// Records the event in the specified stream (or default stream if null).
    pub fn record(self: Event, stream: ?c.CUstream) !void {
        try check(c.cuEventRecord(self.handle, stream));
    }

    /// Blocks until the event has completed.
    pub fn synchronize(self: Event) !void {
        try check(c.cuEventSynchronize(self.handle));
    }

    /// Calculates the elapsed time in milliseconds between two events.
    pub fn elapsed(start: Event, stop: Event) !f32 {
        var result: f32 = undefined;
        try check(c.cuEventElapsedTime(&result, start.handle, stop.handle));
        return result;
    }
};
