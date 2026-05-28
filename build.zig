const std = @import("std");

pub fn build(b: *std.Build) void {
    const host_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build final executable
    const exe = b.addExecutable(.{
        .name = "vector-add-minimal",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = host_target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const nvptx_mcpu = b.option([]const u8, "gpu", "Target GPU features to add or subtract") orelse "sm_80";
    const nvptx_target = b.resolveTargetQuery(std.Build.parseTargetQuery(.{
        .arch_os_abi = "nvptx64-cuda-none",
        .cpu_features = nvptx_mcpu,
    }) catch unreachable);

    const nvptx_code = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "vector-add-kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main_device.zig"),
            .target = nvptx_target,
            .optimize = .ReleaseFast,
        }),
    });
    nvptx_code.linker_allow_shlib_undefined = false;
    nvptx_code.bundle_compiler_rt = false;

    const nvptx_module = nvptx_code.getEmittedAsm();

    exe.addIncludePath(.{ .cwd_relative = "/opt/cuda/include" });
    exe.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    exe.linkSystemLibrary("cuda");
    exe.root_module.addAnonymousImport("cuda-module", .{
        .root_source_file = nvptx_module,
    });
}
