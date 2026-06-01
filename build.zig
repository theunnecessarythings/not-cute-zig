const std = @import("std");

pub fn build(b: *std.Build) void {
    const host_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const cuda_include_dir = b.option([]const u8, "cuda-include-dir", "CUDA include directory") orelse "/opt/cuda/include";
    const cuda_lib_dir = b.option([]const u8, "cuda-lib-dir", "CUDA library directory") orelse "/opt/cuda/lib64";

    const not_cute_mod = b.addModule("not-cute", .{
        .root_source_file = b.path("src/not_cute.zig"),
        .target = host_target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Build final executable
    const exe = b.addExecutable(.{
        .name = "not-cute-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = host_target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.addImport("not-cute", not_cute_mod);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const layout_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/layout.zig"),
            .target = host_target,
            .optimize = optimize,
        }),
    });

    const mma_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mma.zig"),
            .target = host_target,
            .optimize = optimize,
        }),
    });

    const package_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/not_cute.zig"),
            .target = host_target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(layout_tests).step);
    test_step.dependOn(&b.addRunArtifact(mma_tests).step);
    test_step.dependOn(&b.addRunArtifact(package_tests).step);

    const docs_step = b.step("docs", "Generate HTML documentation");
    const docs_install = b.addInstallDirectory(.{
        .source_dir = package_tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&docs_install.step);

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

    exe.addIncludePath(.{ .cwd_relative = cuda_include_dir });
    exe.addLibraryPath(.{ .cwd_relative = cuda_lib_dir });
    exe.linkSystemLibrary("cuda");
    exe.root_module.addAnonymousImport("cuda-module", .{
        .root_source_file = nvptx_module,
    });
}
