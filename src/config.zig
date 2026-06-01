pub const app_name = "not-cute-zig";

pub const vector_len = 1024;
pub const vector_block_size = 256;

pub const transpose_rows = 130;
pub const transpose_cols = 70;
pub const transpose_tile = 16;

pub const ownership_m = 64;
pub const ownership_n = 64;
pub const ownership_warp_threads = 32;
pub const ownership_values_per_lane_owner = 8;

pub const mma_m = 16;
pub const mma_n = 8;
pub const mma_k = 16;
