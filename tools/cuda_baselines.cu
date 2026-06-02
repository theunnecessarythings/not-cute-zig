#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

#define CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t err__ = (expr); \
    if (err__ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err__)); \
        std::exit(1); \
    } \
} while (0)

struct Timing {
    float mean_ms = 0;
    float min_ms = 0;
    float max_ms = 0;
};

struct DeviceInfo {
    int ordinal = 0;
    char name[256]{};
    int sm_major = 0;
    int sm_minor = 0;
    int sm_count = 0;
};

DeviceInfo device_info() {
    DeviceInfo info{};
    CUDA_CHECK(cudaGetDevice(&info.ordinal));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, info.ordinal));
    std::snprintf(info.name, sizeof(info.name), "%s", prop.name);
    info.sm_major = prop.major;
    info.sm_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    return info;
}

template <typename Fn>
Timing time_kernel(int warmups, int iters, Fn fn) {
    for (int i = 0; i < warmups; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    Timing t{};
    t.min_ms = *std::min_element(samples.begin(), samples.end());
    t.max_ms = *std::max_element(samples.begin(), samples.end());
    double total = 0;
    for (float ms : samples) total += ms;
    t.mean_ms = static_cast<float>(total / samples.size());
    return t;
}

void emit_json(
    const DeviceInfo& dev,
    const char* kernel,
    const char* impl,
    const char* shape,
    bool correct,
    Timing t,
    double bytes,
    double flops,
    int warmups,
    int iters
) {
    const double gbps = bytes > 0 ? bytes / (t.mean_ms * 1.0e6) : 0;
    const double tflops = flops > 0 ? flops / (t.mean_ms * 1.0e9) : 0;
    std::printf(
        "{\"device\":\"%s\",\"sm\":\"%d%d\",\"sm_count\":%d,"
        "\"kernel\":\"%s\",\"implementation\":\"%s\",\"shape\":\"%s\","
        "\"correct\":%s,\"warmups\":%d,\"iterations\":%d,"
        "\"mean_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,"
        "\"gbps\":%.6f,\"tflops\":%.6f}\n",
        dev.name, dev.sm_major, dev.sm_minor, dev.sm_count,
        kernel, impl, shape, correct ? "true" : "false", warmups, iters,
        t.mean_ms, t.min_ms, t.max_ms, gbps, tflops
    );
}

void emit_json_with_error(
    const DeviceInfo& dev,
    const char* kernel,
    const char* impl,
    const char* shape,
    bool correct,
    Timing t,
    double bytes,
    double flops,
    int warmups,
    int iters,
    float max_abs_error
) {
    const double gbps = bytes > 0 ? bytes / (t.mean_ms * 1.0e6) : 0;
    const double tflops = flops > 0 ? flops / (t.mean_ms * 1.0e9) : 0;
    std::printf(
        "{\"device\":\"%s\",\"sm\":\"%d%d\",\"sm_count\":%d,"
        "\"kernel\":\"%s\",\"implementation\":\"%s\",\"shape\":\"%s\","
        "\"correct\":%s,\"warmups\":%d,\"iterations\":%d,"
        "\"mean_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,"
        "\"gbps\":%.6f,\"tflops\":%.6f,\"max_abs_error\":%.6f}\n",
        dev.name, dev.sm_major, dev.sm_minor, dev.sm_count,
        kernel, impl, shape, correct ? "true" : "false", warmups, iters,
        t.mean_ms, t.min_ms, t.max_ms, gbps, tflops, max_abs_error
    );
}

void emit_json_with_error_and_workspace(
    const DeviceInfo& dev,
    const char* kernel,
    const char* impl,
    const char* shape,
    bool correct,
    Timing t,
    double bytes,
    double flops,
    int warmups,
    int iters,
    float max_abs_error,
    size_t workspace_bytes
) {
    const double gbps = bytes > 0 ? bytes / (t.mean_ms * 1.0e6) : 0;
    const double tflops = flops > 0 ? flops / (t.mean_ms * 1.0e9) : 0;
    std::printf(
        "{\"device\":\"%s\",\"sm\":\"%d%d\",\"sm_count\":%d,"
        "\"kernel\":\"%s\",\"implementation\":\"%s\",\"shape\":\"%s\","
        "\"correct\":%s,\"warmups\":%d,\"iterations\":%d,"
        "\"mean_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,"
        "\"gbps\":%.6f,\"tflops\":%.6f,\"max_abs_error\":%.6f,"
        "\"workspace_bytes\":%zu}\n",
        dev.name, dev.sm_major, dev.sm_minor, dev.sm_count,
        kernel, impl, shape, correct ? "true" : "false", warmups, iters,
        t.mean_ms, t.min_ms, t.max_ms, gbps, tflops, max_abs_error,
        workspace_bytes
    );
}

__global__ void vector_add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[16][17];
    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    __syncthreads();
    int ox = blockIdx.y * 16 + threadIdx.x;
    int oy = blockIdx.x * 16 + threadIdx.y;
    if (ox < rows && oy < cols) output[oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
}

__inline__ __device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void flash_tiled_kernel(
    const half* q,
    const half* k,
    const half* v,
    float* o,
    int batch_heads,
    int seq_len,
    int head_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int o_stride,
    float scale,
    int causal
) {
    constexpr int block_m = 16;
    constexpr int block_n = 8;
    __shared__ half smem_k[block_n][64];
    __shared__ half smem_v[block_n][64];
    __shared__ float smem_score[block_m][block_n];
    __shared__ float smem_weight[block_m][block_n];
    __shared__ float smem_m[block_m];
    __shared__ float smem_l[block_m];
    __shared__ float smem_old_scale[block_m];
    __shared__ float smem_out[block_m][64];

    const int lane = threadIdx.x;
    const int local_row = threadIdx.y;
    const int q_row = blockIdx.x * block_m + local_row;
    const int batch = blockIdx.y;
    if (batch >= batch_heads || local_row >= block_m) return;

    const half* qb = q + batch * q_stride;
    const half* kb = k + batch * k_stride;
    const half* vb = v + batch * v_stride;
    float* ob = o + batch * o_stride;

    if (lane == 0) {
        smem_m[local_row] = -INFINITY;
        smem_l[local_row] = 0.0f;
    }
    for (int col = lane; col < head_dim; col += 32) {
        smem_out[local_row][col] = 0.0f;
    }
    __syncthreads();

    for (int tile = 0; tile < seq_len; tile += block_n) {
        for (int idx = local_row * 32 + lane; idx < block_n * 64; idx += block_m * 32) {
            const int r = idx / 64;
            const int c = idx % 64;
            const int key_row = tile + r;
            smem_k[r][c] = (key_row < seq_len && c < head_dim) ? kb[key_row * head_dim + c] : __float2half(0.0f);
            smem_v[r][c] = (key_row < seq_len && c < head_dim) ? vb[key_row * head_dim + c] : __float2half(0.0f);
        }
        __syncthreads();

        float tile_max = -INFINITY;
        for (int key = 0; key < block_n; ++key) {
            const int key_row = tile + key;
            float partial = 0.0f;
            for (int c = lane; c < head_dim; c += 32) {
                if (q_row < seq_len && key_row < seq_len) {
                    partial += __half2float(qb[q_row * head_dim + c]) * __half2float(smem_k[key][c]);
                }
            }
            float score = warp_sum(partial);
            if (lane == 0) {
                const bool visible = !causal || key_row <= q_row;
                score = (q_row < seq_len && key_row < seq_len && visible) ? score * scale : -INFINITY;
                smem_score[local_row][key] = score;
                tile_max = fmaxf(tile_max, score);
            }
        }

        if (lane == 0) {
            const float old_m = smem_m[local_row];
            const float old_l = smem_l[local_row];
            const float m_new = fmaxf(old_m, tile_max);
            const float old_scale = old_l == 0.0f ? 0.0f : expf(old_m - m_new);
            float tile_sum = 0.0f;
            for (int key = 0; key < block_n; ++key) {
                const float score = smem_score[local_row][key];
                const float weight = isfinite(score) ? expf(score - m_new) : 0.0f;
                smem_weight[local_row][key] = weight;
                tile_sum += weight;
            }
            smem_m[local_row] = m_new;
            smem_l[local_row] = old_l * old_scale + tile_sum;
            smem_old_scale[local_row] = old_scale;
        }
        __syncthreads();

        const float old_scale = smem_old_scale[local_row];
        for (int col = lane; col < head_dim; col += 32) {
            float acc = smem_out[local_row][col] * old_scale;
            for (int key = 0; key < block_n; ++key) {
                acc += smem_weight[local_row][key] * __half2float(smem_v[key][col]);
            }
            smem_out[local_row][col] = acc;
        }
        __syncthreads();
    }

    const float denom = smem_l[local_row];
    for (int col = lane; col < head_dim; col += 32) {
        if (q_row < seq_len) ob[q_row * head_dim + col] = smem_out[local_row][col] / denom;
    }
}

__global__ void flash_wmma_kernel(
    const half* q,
    const half* k,
    const half* v,
    float* o,
    int batch_heads,
    int seq_len,
    int head_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int o_stride,
    float scale,
    int causal
) {
    constexpr int block_m = 16;
    constexpr int block_n = 8;
    __shared__ half smem_q[16 * 16];
    __shared__ half smem_k[16 * 16];
    __shared__ half smem_p[16 * 16];
    __shared__ half smem_v[16 * 16];
    __shared__ float smem_scores[16 * 16];
    __shared__ float smem_o[16 * 64];
    __shared__ float smem_m[16];
    __shared__ float smem_l[16];
    __shared__ float smem_old_scale[16];

    const int lane = threadIdx.x;
    const int q_base = blockIdx.x * block_m;
    const int batch = blockIdx.y;
    if (lane >= 32 || batch >= batch_heads) return;

    const half* qb = q + batch * q_stride;
    const half* kb = k + batch * k_stride;
    const half* vb = v + batch * v_stride;
    float* ob = o + batch * o_stride;

    for (int i = lane; i < 16; i += 32) {
        smem_m[i] = -INFINITY;
        smem_l[i] = 0.0f;
    }
    for (int i = lane; i < 16 * 64; i += 32) smem_o[i] = 0.0f;
    __syncthreads();

    for (int tile = 0; tile < seq_len; tile += block_n) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_frag;
        wmma::fill_fragment(score_frag, 0.0f);
        for (int k_base = 0; k_base < head_dim; k_base += 16) {
            for (int i = lane; i < 16 * 16; i += 32) {
                const int row = i / 16;
                const int col = i % 16;
                const int q_row = q_base + row;
                const int head_col = k_base + col;
                smem_q[row * 16 + col] = (q_row < seq_len && head_col < head_dim)
                    ? qb[q_row * head_dim + head_col]
                    : __float2half(0.0f);

                const int key_row = tile + row;
                smem_k[col * 16 + row] = (row < block_n && key_row < seq_len && head_col < head_dim)
                    ? kb[key_row * head_dim + head_col]
                    : __float2half(0.0f);
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(q_frag, smem_q, 16);
            wmma::load_matrix_sync(k_frag, smem_k, 16);
            wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
            __syncthreads();
        }

        wmma::store_matrix_sync(smem_scores, score_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (int row = lane; row < 16; row += 32) {
            const int q_row = q_base + row;
            float tile_max = -INFINITY;
            for (int key = 0; key < block_n; ++key) {
                const int key_row = tile + key;
                const bool visible = !causal || key_row <= q_row;
                float score = (q_row < seq_len && key_row < seq_len && visible)
                    ? smem_scores[row * 16 + key] * scale
                    : -INFINITY;
                smem_scores[row * 16 + key] = score;
                tile_max = fmaxf(tile_max, score);
            }

            const float old_m = smem_m[row];
            const float old_l = smem_l[row];
            const float m_new = fmaxf(old_m, tile_max);
            const float old_scale = old_l == 0.0f ? 0.0f : expf(old_m - m_new);
            float tile_sum = 0.0f;
            for (int key = 0; key < block_n; ++key) {
                const float score = smem_scores[row * 16 + key];
                const float weight = isfinite(score) ? expf(score - m_new) : 0.0f;
                smem_p[row * 16 + key] = __float2half(weight);
                tile_sum += weight;
            }
            for (int key = block_n; key < 16; ++key) smem_p[row * 16 + key] = __float2half(0.0f);
            smem_m[row] = m_new;
            smem_l[row] = old_l * old_scale + tile_sum;
            smem_old_scale[row] = old_scale;
        }
        __syncthreads();

        for (int col_chunk = 0; col_chunk < 4; ++col_chunk) {
            for (int i = lane; i < 16 * 16; i += 32) {
                const int row = i / 16;
                const int col = i % 16;
                const int out_col = col_chunk * 16 + col;
                const int key_row = tile + row;
                smem_v[col * 16 + row] = (row < block_n && key_row < seq_len && out_col < head_dim)
                    ? vb[key_row * head_dim + out_col]
                    : __float2half(0.0f);
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
            wmma::fill_fragment(pv_frag, 0.0f);
            wmma::load_matrix_sync(p_frag, smem_p, 16);
            wmma::load_matrix_sync(v_frag, smem_v, 16);
            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            wmma::store_matrix_sync(smem_scores, pv_frag, 16, wmma::mem_row_major);
            __syncthreads();

            for (int i = lane; i < 16 * 16; i += 32) {
                const int row = i / 16;
                const int col = i % 16;
                const int out_col = col_chunk * 16 + col;
                if (out_col < head_dim) {
                    smem_o[row * 64 + out_col] = smem_o[row * 64 + out_col] * smem_old_scale[row] + smem_scores[i];
                }
            }
            __syncthreads();
        }
    }

    for (int i = lane; i < 16 * 64; i += 32) {
        const int row = i / 64;
        const int col = i % 64;
        const int q_row = q_base + row;
        if (q_row < seq_len && col < head_dim) {
            ob[q_row * head_dim + col] = smem_o[i] / smem_l[row];
        }
    }
}

__global__ void attention_softmax_kernel(
    const float* scores,
    half* probs,
    int batch_heads,
    int seq_len,
    float scale,
    int causal
) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= seq_len || batch >= batch_heads) return;

    const float* row_scores = scores + (static_cast<size_t>(batch) * seq_len + row) * seq_len;
    half* row_probs = probs + (static_cast<size_t>(batch) * seq_len + row) * seq_len;

    float local_max = -INFINITY;
    for (int col = lane; col < seq_len; col += blockDim.x) {
        const bool visible = !causal || col <= row;
        const float score = visible ? row_scores[col] * scale : -INFINITY;
        local_max = fmaxf(local_max, score);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    __shared__ float row_max;
    if (lane == 0) row_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = lane; col < seq_len; col += blockDim.x) {
        const bool visible = !causal || col <= row;
        const float weight = visible ? expf(row_scores[col] * scale - row_max) : 0.0f;
        row_probs[col] = __float2half(weight);
        local_sum += weight;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
    }
    __shared__ float row_sum;
    if (lane == 0) row_sum = local_sum;
    __syncthreads();

    const float inv_sum = 1.0f / row_sum;
    for (int col = lane; col < seq_len; col += blockDim.x) {
        row_probs[col] = __float2half(__half2float(row_probs[col]) * inv_sum);
    }
}

bool check_close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

void run_vector(const DeviceInfo& dev, int warmups, int iters) {
    constexpr size_t n = 10'000'000;
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<float>(i % 17);
    float *a, *b, *out;
    CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(256), grid((n + block.x - 1) / block.x);
    auto t = time_kernel(warmups, iters, [&] { vector_add_kernel<<<grid, block>>>(a, b, out, n); });
    CUDA_CHECK(cudaGetLastError());
    emit_json(dev, "vector_add", "cuda_cpp", "n=10000000", true, t, n * sizeof(float) * 3.0, n, warmups, iters);
    cudaFree(a); cudaFree(b); cudaFree(out);
}

void run_transpose(const DeviceInfo& dev, int warmups, int iters) {
    constexpr int rows = 130, cols = 70;
    constexpr size_t n = rows * cols;
    std::vector<float> h(n), got(n), expected(n);
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<float>(i);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) expected[c * rows + r] = h[r * cols + c];
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16), grid((cols + 15) / 16, (rows + 15) / 16);
    auto t = time_kernel(warmups, iters, [&] { transpose_kernel<<<grid, block>>>(in, out, rows, cols); });
    CUDA_CHECK(cudaMemcpy(got.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
    emit_json(dev, "transpose", "cuda_cpp", "130x70", check_close(got, expected, 0), t, n * sizeof(float) * 2.0, 0, warmups, iters);
    cudaFree(in); cudaFree(out);
}

void run_reduction(const DeviceInfo& dev, int warmups, int iters) {
    constexpr size_t n = 1000;
    std::vector<float> h(n);
    float expected = 0;
    for (size_t i = 0; i < n; ++i) { h[i] = static_cast<float>(i % 10); expected += h[i]; }
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    void* temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, in, out, n));
    CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
    auto reduce = [&] { CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, in, out, n)); };
    auto t = time_kernel(warmups, iters, reduce);
    float got = 0;
    CUDA_CHECK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));
    emit_json(dev, "reduction", "cub", "n=1000", std::fabs(got - expected) < 0.1f, t, n * sizeof(float), n, warmups, iters);
    cudaFree(temp);
    cudaFree(in); cudaFree(out);
}

void run_gemm(const DeviceInfo& dev, int warmups, int iters) {
    constexpr int m = 16, n = 8, kdim = 16;
    std::vector<half> A(m * kdim), B(kdim * n);
    std::vector<float> C(m * n, 0), got(m * n), expected(m * n);
    for (int i = 0; i < m * kdim; ++i) A[i] = __float2half(static_cast<float>(i % 5) * 0.25f);
    for (int i = 0; i < kdim * n; ++i) B[i] = __float2half(static_cast<float>(i % 7) * 0.125f);
    for (int r = 0; r < m; ++r) for (int c = 0; c < n; ++c) {
        float acc = 0;
        for (int kk = 0; kk < kdim; ++kk) acc += __half2float(A[r * kdim + kk]) * __half2float(B[kk * n + c]);
        expected[r * n + c] = acc;
    }
    half *dA, *dB; float *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, C.size() * sizeof(float)));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    auto launch = [&] {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, kdim,
            &alpha,
            dB, CUDA_R_16F, n,
            dA, CUDA_R_16F, kdim,
            &beta,
            dC, CUDA_R_32F, n,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto t = time_kernel(warmups, iters, launch);
    CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    emit_json(dev, "mma_gemm", "cublas", "16x8x16", check_close(got, expected, 0.01f), t, (A.size()+B.size())*sizeof(half)+C.size()*sizeof(float), 2.0*m*n*kdim, warmups, iters);
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void run_flash_case(const DeviceInfo& dev, int warmups, int iters, int batch_heads, int seq_len, int head_dim, bool causal) {
    const int stride = seq_len * head_dim;
    const size_t storage = static_cast<size_t>(batch_heads) * stride;
    std::vector<half> q(storage), k(storage), v(storage);
    std::vector<float> o(storage), expected(storage);
    for (size_t i = 0; i < storage; ++i) {
        q[i] = __float2half(static_cast<float>((i * 3 + 1) % 17) * 0.125f);
        k[i] = __float2half(static_cast<float>((i * 5 + 2) % 17) * 0.0625f);
        v[i] = __float2half(static_cast<float>((i * 7 + 3) % 17) * 0.03125f);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int b = 0; b < batch_heads; ++b) {
        const size_t base = static_cast<size_t>(b) * stride;
        for (int row = 0; row < seq_len; ++row) {
            float row_max = -INFINITY;
            for (int key = 0; key < seq_len; ++key) {
                if (causal && key > row) continue;
                float score = 0.0f;
                for (int c = 0; c < head_dim; ++c) {
                    score += __half2float(q[base + row * head_dim + c]) * __half2float(k[base + key * head_dim + c]);
                }
                row_max = std::max(row_max, score * scale);
            }
            float row_sum = 0.0f;
            for (int key = 0; key < seq_len; ++key) {
                if (causal && key > row) continue;
                float score = 0.0f;
                for (int c = 0; c < head_dim; ++c) {
                    score += __half2float(q[base + row * head_dim + c]) * __half2float(k[base + key * head_dim + c]);
                }
                const float weight = std::exp(score * scale - row_max);
                row_sum += weight;
                for (int c = 0; c < head_dim; ++c) {
                    expected[base + row * head_dim + c] += weight * __half2float(v[base + key * head_dim + c]);
                }
            }
            for (int c = 0; c < head_dim; ++c) expected[base + row * head_dim + c] /= row_sum;
        }
    }
    half *dq, *dk, *dv; float *dout;
    CUDA_CHECK(cudaMalloc(&dq, storage * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dk, storage * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dv, storage * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dout, storage * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), storage * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), storage * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), storage * sizeof(half), cudaMemcpyHostToDevice));
    dim3 block(32, 16), grid((seq_len + 15) / 16, batch_heads);
    auto tiled_t = time_kernel(warmups, iters, [&] {
        flash_tiled_kernel<<<grid, block>>>(dq, dk, dv, dout, batch_heads, seq_len, head_dim, stride, stride, stride, stride, scale, causal ? 1 : 0);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(o.data(), dout, storage * sizeof(float), cudaMemcpyDeviceToHost));
    char shape[128];
    std::snprintf(shape, sizeof(shape), "batch_heads=%d,seq=%d,head=%d,causal=%s", batch_heads, seq_len, head_dim, causal ? "true" : "false");
    const double effective_seq = causal ? (static_cast<double>(seq_len) * (seq_len + 1) / 2.0) : static_cast<double>(seq_len) * seq_len;
    const float tiled_error = max_abs_error(o, expected);
    emit_json_with_error(dev, "flash_attention", "cuda_tiled", shape, tiled_error <= 0.02f, tiled_t, storage * (sizeof(half) * 3.0 + sizeof(float)), batch_heads * 4.0 * effective_seq * head_dim, warmups, iters, tiled_error);

    dim3 wmma_block(32), wmma_grid((seq_len + 15) / 16, batch_heads);
    auto wmma_t = time_kernel(warmups, iters, [&] {
        flash_wmma_kernel<<<wmma_grid, wmma_block>>>(dq, dk, dv, dout, batch_heads, seq_len, head_dim, stride, stride, stride, stride, scale, causal ? 1 : 0);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(o.data(), dout, storage * sizeof(float), cudaMemcpyDeviceToHost));
    const float wmma_error = max_abs_error(o, expected);
    emit_json_with_error(dev, "flash_attention", "cuda_wmma", shape, wmma_error <= 0.08f, wmma_t, storage * (sizeof(half) * 3.0 + sizeof(float)), batch_heads * 4.0 * effective_seq * head_dim, warmups, iters, wmma_error);

    float* dscores;
    half* dprobs;
    const size_t scores_len = static_cast<size_t>(batch_heads) * seq_len * seq_len;
    CUDA_CHECK(cudaMalloc(&dscores, scores_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dprobs, scores_len * sizeof(half)));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const long long qkv_stride = static_cast<long long>(stride);
    const long long score_stride = static_cast<long long>(seq_len) * seq_len;
    dim3 softmax_grid(seq_len, batch_heads);
    dim3 softmax_block(32);
    auto materialized = [&] {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len,
            seq_len,
            head_dim,
            &alpha,
            dk,
            CUDA_R_16F,
            head_dim,
            qkv_stride,
            dq,
            CUDA_R_16F,
            head_dim,
            qkv_stride,
            &beta,
            dscores,
            CUDA_R_32F,
            seq_len,
            score_stride,
            batch_heads,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        attention_softmax_kernel<<<softmax_grid, softmax_block>>>(dscores, dprobs, batch_heads, seq_len, scale, causal ? 1 : 0);
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim,
            seq_len,
            seq_len,
            &alpha,
            dv,
            CUDA_R_16F,
            head_dim,
            qkv_stride,
            dprobs,
            CUDA_R_16F,
            seq_len,
            score_stride,
            &beta,
            dout,
            CUDA_R_32F,
            head_dim,
            qkv_stride,
            batch_heads,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto cublas_attention_t = time_kernel(warmups, iters, materialized);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(o.data(), dout, storage * sizeof(float), cudaMemcpyDeviceToHost));
    const float cublas_attention_error = max_abs_error(o, expected);
    const size_t workspace_bytes = scores_len * (sizeof(float) + sizeof(half));
    const double materialized_bytes = storage * (sizeof(half) * 3.0 + sizeof(float)) + static_cast<double>(workspace_bytes);
    emit_json_with_error_and_workspace(dev, "flash_attention", "cublas_materialized", shape, cublas_attention_error <= 0.08f, cublas_attention_t, materialized_bytes, batch_heads * 4.0 * effective_seq * head_dim, warmups, iters, cublas_attention_error, workspace_bytes);
    cublasDestroy(handle);
    cudaFree(dscores);
    cudaFree(dprobs);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}

} // namespace

int main(int argc, char** argv) {
    int warmups = 5;
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) warmups = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    DeviceInfo dev = device_info();
    run_vector(dev, warmups, iters);
    run_transpose(dev, warmups, iters);
    run_reduction(dev, warmups, iters);
    run_gemm(dev, warmups, iters);
    run_flash_case(dev, warmups, iters, 2, 32, 16, false);
    run_flash_case(dev, warmups, iters, 2, 32, 32, true);
    run_flash_case(dev, warmups, iters, 2, 64, 16, false);
    run_flash_case(dev, warmups, iters, 2, 64, 32, true);
    run_flash_case(dev, warmups, iters, 2, 128, 32, true);
    run_flash_case(dev, warmups, iters, 8, 128, 16, false);
    run_flash_case(dev, warmups, iters, 8, 128, 32, true);
    run_flash_case(dev, warmups, iters, 8, 256, 16, false);
    run_flash_case(dev, warmups, iters, 8, 256, 32, true);
    run_flash_case(dev, warmups, iters, 16, 256, 32, false);
    run_flash_case(dev, warmups, iters, 16, 512, 32, true);
    run_flash_case(dev, warmups, iters, 8, 256, 64, false);
    run_flash_case(dev, warmups, iters, 8, 512, 64, true);
    run_flash_case(dev, warmups, iters, 16, 512, 64, false);
    run_flash_case(dev, warmups, iters, 8, 1024, 64, true);
    return 0;
}
