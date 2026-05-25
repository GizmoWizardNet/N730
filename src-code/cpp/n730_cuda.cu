/*
 * n730_cuda.cu — N730 Kernel
 * ==========================================
 *
 * GT 730 constraints(certain values may change) noted:
 *   - Compute capability 3.5 (Kepler)
 *   - 2GB VRAM DDR3
 *   - 384 CUDA cores
 *   - No FP16 tensor cores (those are Volta+)
 *   - cuBLAS SGEMM is the fast path
 *   - Max shared memory: 48KB per SM
 *
 * Build:
 *   nvcc -O3 -arch=sm_35 -shared -Xcompiler -fPIC \
 *        -lcublas -o n730_cuda.so n730_cuda.cu
 *
 *   Windows (x64 Developer Prompt):
 *   nvcc -O3 -arch=sm_35 --shared -lcublas \
 *        -o n730_cuda.dll n730_cuda.cu
 *
 * Internal usage only.
 *
 * Minimum CUDA version(IDK why the fuck you would use this but whatever bruh): 9.0 (last version supporting sm_35)
 * Recommended CUDA version: 11.4 (for least pain)
 *
 * REQUIRED Visual Studio Setup: C++ Desktop workload, MSVC v142 and VS 2019 as optimal version
 * Publicly available VS2019 Community download: https://aka.ms/vs/16/release/vs_community.exe
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef _WIN32
#define N730_API extern "C" __declspec(dllexport)
#else
#define N730_API extern "C" __attribute__((visibility("default")))
#endif

// ─── Error handling ───────────────────────────────────────────────────────────

#define CUDA_CHECK(x)                                                     \
    do                                                                    \
    {                                                                     \
        cudaError_t e = (x);                                              \
        if (e != cudaSuccess)                                             \
        {                                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                               \
            return N730_CUDA_ERR;                                         \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(x)                                                         \
    do                                                                          \
    {                                                                           \
        cublasStatus_t s = (x);                                                 \
        if (s != CUBLAS_STATUS_SUCCESS)                                         \
        {                                                                       \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
            return N730_CUDA_ERR;                                               \
        }                                                                       \
    } while (0)

static const int N730_OK = 0;
static const int N730_CUDA_ERR = -10;
static const int N730_OOM = -11;
static const int N730_NULL = -12;

// ─── GPU context ─────────────────────────────────────────────────────────────

struct N730CudaCtx
{
    cublasHandle_t cublas;

    // Persistent VRAM buffers — allocated once, reused every layer
    float *d_weights;     // dequantized weight matrix (max layer size)
    float *d_activations; // current hidden states  (seq * hidden)
    float *d_attn_out;    // attention output buffer
    float *d_mlp_out;     // MLP output buffer
    float *d_qkv;         // Q/K/V projections     (seq * 3 * hidden)
    float *d_scores;      // attention scores      (seq * seq * heads)
    float *d_norm_buf;    // RMSNorm workspace

    // Sizes
    int max_weight_elements; // largest layer's rows*cols
    int max_seq;
    int hidden_size;
    int num_heads;
    int head_dim;
    int vocab_size;

    // Staging: pinned host memory for fast DMA
    float *h_weights_pinned; // pinned host buffer for weight transfers
    uint8_t *h_quant_pinned; // pinned host buffer for raw quantized bytes
    int pinned_bytes;

    // Dedicated device staging buffer for raw quantized bytes during upload.
    // Sized to hold the raw bytes of the largest INT4 layer (max_weight_elements/2).
    // Kept separate from d_norm_buf so weight uploads never clobber active data.
    uint8_t *d_quant_staging;
    int quant_staging_bytes;
};

// ─── Dequantization kernels — run on GPU ─────────────────────────────────────

/*
 * INT4 dequant: each byte holds two 4-bit values (lo nibble, hi nibble).
 * Launch with n_elements/2 threads (each thread handles one byte = 2 values).
 */
__global__ void dequant_int4_kernel(
    const uint8_t *__restrict__ src,
    float *__restrict__ dst,
    int n_elements,
    float scale,
    float zero_point)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_bytes = (n_elements + 1) / 2;
    if (i >= n_bytes)
        return;

    uint8_t byte = src[i];
    float lo = ((float)(byte & 0x0F) - zero_point) * scale;
    float hi = ((float)((byte >> 4) & 0x0F) - zero_point) * scale;

    int out0 = i * 2;
    int out1 = out0 + 1;
    dst[out0] = lo;
    if (out1 < n_elements)
        dst[out1] = hi;
}

/*
 * INT8 dequant: one thread per element.
 * Simplest kernel — compiler will vectorize loads.
 */
__global__ void dequant_int8_kernel(
    const uint8_t *__restrict__ src,
    float *__restrict__ dst,
    int n_elements,
    float scale,
    float zero_point)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    dst[i] = ((float)src[i] - zero_point) * scale;
}

/*
 * RMSNorm kernel: normalize x by RMS, multiply by weight vector w.
 * One block per row (one token position), up to 256 threads (8 warps).
 * Reduction: warp shuffle within each warp, then shared memory across warps.
 *
 * hidden_size for this model is 1536. With 128 threads each thread covers
 * 12 elements in the accumulation loop, then we reduce 4 warps via smem.
 */
__global__ void rmsnorm_kernel(
    float *__restrict__ x,       // (seq, hidden) — modified in place
    const float *__restrict__ w, // (hidden,) norm weights
    int seq,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    if (row >= seq)
        return;

    float *xrow = x + row * hidden;

    // Each thread accumulates partial sum of squares over its strided elements
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        float v = xrow[i];
        sum_sq += v * v;
    }

    // Step 1: warp-level reduction (handles threads within same warp)
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down(sum_sq, offset);

    // Step 2: write each warp's result to shared memory
    // (up to 8 warps for blockDim.x=256; we use blockDim.x/32 slots)
    extern __shared__ float warp_sums[]; // blockDim.x/32 floats
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    // Step 3: first warp reduces the warp partial sums
    int n_warps = blockDim.x >> 5;
    if (warp_id == 0)
    {
        sum_sq = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down(sum_sq, offset);
        if (lane == 0)
            warp_sums[0] = rsqrtf(sum_sq / hidden + eps);
    }
    __syncthreads();

    float rms_inv = warp_sums[0];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        xrow[i] = xrow[i] * rms_inv * w[i];
}

/*
 * SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Applied element-wise to gate projection output.
 */
__global__ void silu_kernel(
    float *__restrict__ gate, // modified in place
    int n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    float x = gate[i];
    gate[i] = x / (1.0f + expf(-x));
}

/*
 * Elementwise multiply: gate *= up (SwiGLU merge step)
 */
__global__ void elemwise_mul_kernel(
    float *__restrict__ gate, // modified in place: gate = gate * up
    const float *__restrict__ up,
    int n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    gate[i] *= up[i];
}

/*
 * Residual add: x += delta
 */
__global__ void residual_add_kernel(
    float *__restrict__ x,
    const float *__restrict__ delta,
    int n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    x[i] += delta[i];
}

/*
 * Softmax over last dimension.
 * One block per (head * query) row, seq_total columns.
 * Uses proper multi-warp reduction via shared memory.
 */
__global__ void softmax_kernel(
    float *__restrict__ scores, // (n_rows, seq) modified in place
    int n_rows,
    int seq)
{
    int row = blockIdx.x;
    if (row >= n_rows)
        return;
    float *s = scores + row * seq;

    extern __shared__ float smem[]; // 2 * n_warps floats: [0..n_warps-1]=max, [n_warps..]=sum
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int n_warps = blockDim.x >> 5;
    float *smem_max = smem;
    float *smem_sum = smem + n_warps;

    // ── Pass 1: find max ──────────────────────────────────────────────────
    float mx = -1e20f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        mx = fmaxf(mx, s[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        mx = fmaxf(mx, __shfl_down(mx, offset));
    if (lane == 0)
        smem_max[warp_id] = mx;
    __syncthreads();
    if (warp_id == 0)
    {
        mx = (lane < n_warps) ? smem_max[lane] : -1e20f;
        for (int offset = 16; offset > 0; offset >>= 1)
            mx = fmaxf(mx, __shfl_down(mx, offset));
        if (lane == 0)
            smem_max[0] = mx;
    }
    __syncthreads();
    mx = smem_max[0];

    // ── Pass 2: exp and partial sum ───────────────────────────────────────
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
    {
        s[i] = expf(fmaxf(s[i] - mx, -50.0f));
        sum += s[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0)
        smem_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0)
    {
        sum = (lane < n_warps) ? smem_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0)
            smem_sum[0] = sum + 1e-9f;
    }
    __syncthreads();
    sum = smem_sum[0];

    // ── Pass 3: normalize ─────────────────────────────────────────────────
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        s[i] /= sum;
}

/*
 * Causal mask: set scores[q, k] = -1e4 where k > q + offset
 * (offset = number of previously cached tokens)
 */
__global__ void causal_mask_kernel(
    float *__restrict__ scores, // (n_heads, seq_q, seq_total)
    int n_heads,
    int seq_q,
    int seq_total,
    int cache_offset)
{
    int h = blockIdx.z;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads || q >= seq_q || k >= seq_total)
        return;

    int abs_q = cache_offset + q;
    if (k > abs_q)
    {
        scores[h * seq_q * seq_total + q * seq_total + k] = -1e4f;
    }
}

// ─── RoPE kernel ─────────────────────────────────────────────────────────────

/*
 * Apply rotary position embeddings in-place.
 * x: (seq, n_heads, head_dim)
 * cos/sin: (max_seq, head_dim/2) precomputed on host, stored on device
 */
__global__ void rope_kernel(
    float *__restrict__ x,           // (seq, n_heads, head_dim)
    const float *__restrict__ cos_f, // (max_seq, head_dim/2)
    const float *__restrict__ sin_f,
    int seq,
    int n_heads,
    int head_dim,
    int offset // position offset for KV cache
)
{
    int s = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x; // iterates over head_dim/2
    if (s >= seq || h >= n_heads || i >= head_dim / 2)
        return;

    float *xsh = x + s * n_heads * head_dim + h * head_dim;
    int pos = s + offset;
    float c = cos_f[pos * (head_dim / 2) + i];
    float sv = sin_f[pos * (head_dim / 2) + i];

    float x0 = xsh[i];
    float x1 = xsh[i + head_dim / 2];
    xsh[i] = x0 * c - x1 * sv;
    xsh[i + head_dim / 2] = x0 * sv + x1 * c;
}

// ─── Public API ──────────────────────────────────────────────────────────────

N730_API int n730_cuda_init(
    int hidden_size,
    int num_heads,
    int head_dim,
    int vocab_size,
    int max_seq,
    int max_weight_elements,
    void **out_ctx)
{
    N730CudaCtx *ctx = new N730CudaCtx{};
    ctx->hidden_size = hidden_size;
    ctx->num_heads = num_heads;
    ctx->head_dim = head_dim;
    ctx->vocab_size = vocab_size;
    ctx->max_seq = max_seq;
    ctx->max_weight_elements = max_weight_elements;

    // Init cuBLAS
    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS)
    {
        delete ctx;
        return N730_CUDA_ERR;
    }

    // Allocate persistent VRAM buffers
    size_t wbytes = (size_t)max_weight_elements * sizeof(float);
    size_t abytes = (size_t)max_seq * hidden_size * sizeof(float);
    size_t qkv = (size_t)max_seq * 3 * hidden_size * sizeof(float);
    size_t scores = (size_t)num_heads * max_seq * max_seq * sizeof(float);

    if (cudaMalloc(&ctx->d_weights, wbytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_activations, abytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_attn_out, abytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_mlp_out, abytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_qkv, qkv) != cudaSuccess ||
        cudaMalloc(&ctx->d_scores, scores) != cudaSuccess ||
        cudaMalloc(&ctx->d_norm_buf, abytes) != cudaSuccess)
    {
        delete ctx;
        return N730_OOM;
    }

    // Pinned host memory for fast H2D transfers
    int pinned = max_weight_elements * 4; // enough for FP32 or INT8
    ctx->pinned_bytes = pinned;
    if (cudaMallocHost(&ctx->h_weights_pinned, pinned) != cudaSuccess ||
        cudaMallocHost(&ctx->h_quant_pinned, pinned) != cudaSuccess)
    {
        delete ctx;
        return N730_OOM;
    }

    // Dedicated device staging for quantized bytes — INT4 worst case is n_elem/2 bytes.
    // Allocate as max_weight_elements bytes (covers INT8 too, and INT4 easily).
    ctx->quant_staging_bytes = max_weight_elements;
    if (cudaMalloc(&ctx->d_quant_staging, (size_t)max_weight_elements) != cudaSuccess)
    {
        delete ctx;
        return N730_OOM;
    }

    *out_ctx = ctx;
    printf("N730 CUDA ready: hidden=%d heads=%d vocab=%d\n",
           hidden_size, num_heads, vocab_size);
    printf("VRAM allocated: weights=%.1fMB activations=%.1fMB\n",
           wbytes / 1048576.0f, abytes / 1048576.0f);
    return N730_OK;
}

N730_API void n730_cuda_destroy(void *ctx_ptr)
{
    if (!ctx_ptr)
        return;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;
    cublasDestroy(ctx->cublas);
    cudaFree(ctx->d_weights);
    cudaFree(ctx->d_activations);
    cudaFree(ctx->d_attn_out);
    cudaFree(ctx->d_mlp_out);
    cudaFree(ctx->d_qkv);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_norm_buf);
    cudaFreeHost(ctx->h_weights_pinned);
    cudaFreeHost(ctx->h_quant_pinned);
    cudaFree(ctx->d_quant_staging);
    delete ctx;
}

/*
 * n730_load_activations
 * Copy host float32 activations (embedding lookup result) into VRAM.
 * Called once per forward pass with the embedded token(s).
 */
N730_API int n730_load_activations(
    void *ctx_ptr,
    const float *host_activations,
    int seq_len,
    int hidden_size)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;
    size_t bytes = (size_t)seq_len * hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ctx->d_activations, host_activations, bytes,
                          cudaMemcpyHostToDevice));
    return N730_OK;
}

/*
 * n730_get_activations
 * Copy VRAM activations back to host (for final norm + lm_head on CPU,
 * or for debugging). Only last token position needed for generation.
 */
N730_API int n730_get_activations(
    void *ctx_ptr,
    float *host_out,
    int seq_len,
    int hidden_size)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;
    size_t bytes = (size_t)seq_len * hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(host_out, ctx->d_activations, bytes,
                          cudaMemcpyDeviceToHost));
    return N730_OK;
}

/*
 * n730_upload_weight
 * DMA a quantized weight matrix from host RAM → VRAM, dequantize in place.
 * This is the hot path: called once per layer per token.
 *
 * raw_bytes: INT4 or INT8 packed bytes (from n730core / scheduler)
 * prec_id:   4 = INT4, 8 = INT8
 * n_elements: rows * cols of the weight matrix
 */
N730_API int n730_upload_weight(
    void *ctx_ptr,
    const uint8_t *raw_bytes,
    int prec_id,
    int n_elements,
    float scale,
    float zero_point)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;

    int raw_bytes_count = (prec_id == 4) ? (n_elements + 1) / 2 : n_elements;

    // Use pre-allocated pinned staging buffer — avoids cudaMalloc/cudaFree per call.
    // The pinned buffer was sized to max_weight_elements * 4 bytes at init, which
    // is always >= raw_bytes_count for INT4/INT8/FP16.
    memcpy(ctx->h_quant_pinned, raw_bytes, raw_bytes_count);

    // DMA from pinned host → dedicated device staging buffer (never overlaps with
    // d_norm_buf or any active activation buffer).
    CUDA_CHECK(cudaMemcpy(ctx->d_quant_staging, ctx->h_quant_pinned, raw_bytes_count,
                          cudaMemcpyHostToDevice));

    // Dequantize on GPU: d_quant_staging (raw bytes) → d_weights (float32)
    int threads = 256;
    if (prec_id == 4)
    {
        int n_bytes = (n_elements + 1) / 2;
        int blocks = (n_bytes + threads - 1) / threads;
        dequant_int4_kernel<<<blocks, threads>>>(
            ctx->d_quant_staging, ctx->d_weights, n_elements, scale, zero_point);
    }
    else
    {
        int blocks = (n_elements + threads - 1) / threads;
        dequant_int8_kernel<<<blocks, threads>>>(
            ctx->d_quant_staging, ctx->d_weights, n_elements, scale, zero_point);
    }

    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

/*
 * n730_upload_norm_weight
 * Upload a 1D RMSNorm weight vector to a caller-provided device buffer.
 * Returns the device pointer via out_ptr.
 * Caller is responsible for freeing with n730_free_device_buf.
 */
N730_API int n730_upload_norm_weight(
    const float *host_w,
    int n_elements,
    void **out_ptr)
{
    float *d_w;
    size_t bytes = n_elements * sizeof(float);
    if (cudaMalloc(&d_w, bytes) != cudaSuccess)
        return N730_OOM;
    if (cudaMemcpy(d_w, host_w, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(d_w);
        return N730_CUDA_ERR;
    }
    *out_ptr = d_w;
    return N730_OK;
}

N730_API void n730_free_device_buf(void *ptr)
{
    if (ptr)
        cudaFree(ptr);
}

/*
 * n730_rmsnorm_inplace
 * Apply RMSNorm to d_activations using a device-side norm weight vector.
 * Result written to d_norm_buf (leaves d_activations unchanged for residual).
 */
N730_API int n730_rmsnorm(
    void *ctx_ptr,
    const float *d_norm_w, // device pointer to norm weights
    int seq_len,
    float eps)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;

    // Copy activations → norm_buf, then normalize in place
    size_t bytes = (size_t)seq_len * ctx->hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(ctx->d_norm_buf, ctx->d_activations, bytes,
                          cudaMemcpyDeviceToDevice));

    // 128 threads = 4 warps; shared memory = 4 floats (one per warp)
    int threads = 128;
    int smem = (threads / 32) * sizeof(float);
    rmsnorm_kernel<<<seq_len, threads, smem>>>(
        ctx->d_norm_buf, d_norm_w, seq_len, ctx->hidden_size, eps);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

/*
 * n730_linear
 * SGEMM: out = norm_buf @ W^T
 * W is already in d_weights (uploaded + dequantized).
 * out_buf: device pointer to output buffer (caller allocated).
 *
 * cuBLAS SGEMM: C = alpha*A*B + beta*C
 * We want: out(seq, out_dim) = norm_buf(seq, hidden) @ W(out_dim, hidden)^T
 * In column-major (cuBLAS default):
 *   A = W^T → (hidden, out_dim) col-major = W (out_dim, hidden) row-major
 *   B = norm_buf^T → (hidden, seq) col-major = norm_buf (seq, hidden) row-major
 *   C = out^T → (out_dim, seq)
 */
N730_API int n730_linear(
    void *ctx_ptr,
    float *d_out, // pre-allocated device output buffer
    int seq_len,
    int in_dim,
    int out_dim)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;

    const float alpha = 1.0f, beta = 0.0f;
    // d_out(seq, out_dim) = d_norm_buf(seq, in_dim) @ d_weights(out_dim, in_dim)^T
    // cuBLAS col-major: sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // m=out_dim, n=seq, k=in_dim
    // A = d_weights (out_dim x in_dim, row-major = in_dim x out_dim col-major), lda=in_dim, transa=N
    // B = d_norm_buf (seq x in_dim, row-major = in_dim x seq col-major), ldb=in_dim, transb=T
    // C = d_out (out_dim x seq col-major = seq x out_dim row-major), ldc=out_dim
    CUBLAS_CHECK(cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_T, // A^T: weights are row-major, cuBLAS wants col-major
        CUBLAS_OP_N, // B: activations
        out_dim, seq_len, in_dim,
        &alpha,
        ctx->d_weights, in_dim,  // A
        ctx->d_norm_buf, in_dim, // B
        &beta,
        d_out, out_dim // C
        ));
    return N730_OK;
}

/*
 * n730_residual_add
 * x += delta — adds sub-result back to residual stream.
 */
N730_API int n730_residual_add(
    void *ctx_ptr,
    const float *d_delta,
    int seq_len)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;
    int n = seq_len * ctx->hidden_size;
    int threads = 256, blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads>>>(ctx->d_activations, d_delta, n);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

/*
 * n730_swiglu
 * SwiGLU activation: gate = silu(gate) * up
 * gate and up are both (seq, intermediate_size) in device memory.
 * Result stored in gate buffer.
 */
N730_API int n730_swiglu(
    float *d_gate,
    float *d_up,
    int seq_len,
    int intermediate_size)
{
    int n = seq_len * intermediate_size;
    int threads = 256, blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(d_gate, n);
    elemwise_mul_kernel<<<blocks, threads>>>(d_gate, d_up, n);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

/*
 * n730_apply_rope
 * Apply rotary embeddings to Q or K tensor already in device memory.
 */
N730_API int n730_apply_rope(
    float *d_x,         // (seq, n_heads, head_dim) device
    const float *d_cos, // (max_seq, head_dim/2) device
    const float *d_sin,
    int seq_len,
    int n_heads,
    int head_dim,
    int position_offset)
{
    dim3 blocks(seq_len, n_heads);
    int threads = head_dim / 2;
    rope_kernel<<<blocks, threads>>>(d_x, d_cos, d_sin,
                                     seq_len, n_heads, head_dim,
                                     position_offset);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

/*
 * n730_rope_precompute
 * Build cos/sin tables on device. Called once at model init.
 */
N730_API int n730_rope_precompute(
    int max_seq,
    int head_dim,
    float theta,
    void **d_cos_out,
    void **d_sin_out)
{
    // Build on host first
    int h2 = head_dim / 2;
    float *h_cos = new float[max_seq * h2];
    float *h_sin = new float[max_seq * h2];

    for (int pos = 0; pos < max_seq; pos++)
    {
        for (int i = 0; i < h2; i++)
        {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            h_cos[pos * h2 + i] = cosf(angle);
            h_sin[pos * h2 + i] = sinf(angle);
        }
    }

    size_t bytes = (size_t)max_seq * h2 * sizeof(float);
    float *d_cos, *d_sin;
    if (cudaMalloc(&d_cos, bytes) != cudaSuccess ||
        cudaMalloc(&d_sin, bytes) != cudaSuccess)
    {
        delete[] h_cos;
        delete[] h_sin;
        return N730_OOM;
    }
    cudaMemcpy(d_cos, h_cos, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, bytes, cudaMemcpyHostToDevice);

    delete[] h_cos;
    delete[] h_sin;
    *d_cos_out = d_cos;
    *d_sin_out = d_sin;
    return N730_OK;
}

/*
 * n730_softmax_scores
 * Apply causal mask + softmax to attention score matrix.
 * scores: (n_heads, seq_q, seq_total) device memory
 */
N730_API int n730_softmax_scores(
    float *d_scores,
    int n_heads,
    int seq_q,
    int seq_total,
    int cache_offset)
{
    // Apply causal mask
    if (seq_q > 1)
    {
        dim3 threads(16, 16);
        dim3 blocks(
            (seq_total + 15) / 16,
            (seq_q + 15) / 16,
            n_heads);
        causal_mask_kernel<<<blocks, threads>>>(
            d_scores, n_heads, seq_q, seq_total, cache_offset);
        CUDA_CHECK(cudaGetLastError());
    }

    // Softmax over each (head, query) row — 64 threads = 2 warps, smem = 2*2 floats
    int n_rows = n_heads * seq_q;
    int sf_threads = 64;
    int sf_smem = 2 * (sf_threads / 32) * sizeof(float);
    softmax_kernel<<<n_rows, sf_threads, sf_smem>>>(
        d_scores, n_rows, seq_total);
    CUDA_CHECK(cudaGetLastError());
    return N730_OK;
}

/*
 * n730_attention_forward
 * Full attention forward pass entirely on GPU.
 *
 * Inputs (all device pointers):
 *   d_q        : (seq, n_heads, head_dim)  — already RoPE'd
 *   d_k_cache  : (total_seq, n_kv_heads, head_dim) — full KV cache for this layer
 *   d_v_cache  : (total_seq, n_kv_heads, head_dim)
 *   d_out      : (seq, n_heads * head_dim) — output buffer
 *
 * Uses cuBLAS SGEMM for Q@K^T and attn@V instead of looping on CPU.
 * GQA (grouped query attention) handled by repeating KV heads on the fly.
 *
 * For the GT 730 (sm_35, 384 cores) this is the critical path:
 * cuBLAS SGEMM beats numpy CPU einsum by ~20-50x for the sizes used here.
 */
N730_API int n730_attention_forward(
    void *ctx_ptr,
    const float *d_q,       // (seq, n_heads, head_dim) device
    const float *d_k_cache, // (total_seq, n_kv_heads, head_dim) device
    const float *d_v_cache, // (total_seq, n_kv_heads, head_dim) device
    float *d_out,           // (seq, n_heads * head_dim) device — output
    int seq_q,              // number of query tokens (1 in decode, >1 in prefill)
    int seq_total,          // total KV length (cache + new tokens)
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int cache_offset // number of previously cached tokens
)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;

    const float scale = 1.0f / sqrtf((float)head_dim);
    const float alpha1 = scale;
    const float beta0 = 0.0f;
    const float alpha1f = 1.0f;

    int grp = n_heads / n_kv_heads; // GQA group size

    // d_scores is pre-allocated: (n_heads * max_seq * max_seq) floats
    // We'll use it as (n_heads, seq_q, seq_total) — must fit
    // (already guaranteed by init: max_seq * max_seq * n_heads)

    // ── Step 1: scores[h, q, k] = Q[q,h,:] @ K[k,kv_h,:]^T * scale ───────
    // For each head h, kv_head = h / grp
    // cuBLAS SGEMM is column-major; we work around by using transposes.
    //
    // For decode (seq_q=1): loop over heads, SGEMM is a GEMV (fast).
    // For prefill (seq_q>1): loop over heads, SGEMM is (seq_q, seq_total).
    //
    // Q layout: (seq_q, n_heads, head_dim) row-major
    //   → head h starts at d_q + h * head_dim, stride = n_heads * head_dim
    // K layout: (seq_total, n_kv_heads, head_dim) row-major
    //   → kv-head kv_h starts at d_k_cache + kv_h * head_dim, stride = n_kv_heads * head_dim

    for (int h = 0; h < n_heads; h++)
    {
        int kv_h = h / grp;

        // Q slice: (seq_q, head_dim), strides n_heads*head_dim in leading dim
        // K slice: (seq_total, head_dim), strides n_kv_heads*head_dim in leading dim
        // score[h]: (seq_q, seq_total)

        // cuBLAS col-major SGEMM: score = Q * K^T, shape (seq_q, seq_total)
        //   m=seq_total, n=seq_q, k=head_dim
        // K slice for kv_h: (seq_total, head_dim) row-major, row-stride = n_kv_heads*head_dim.
        //   As col-major: (head_dim x seq_total) with lda = n_kv_heads*head_dim.
        //   CUBLAS_OP_T transposes it → op(K) = (seq_total x head_dim) = K.
        // Q slice for h:  (seq_q, head_dim) row-major, row-stride = n_heads*head_dim.
        //   As col-major: (head_dim x seq_q) with ldb = n_heads*head_dim.
        //   CUBLAS_OP_N: op(Q) = (head_dim x seq_q) = Q^T.
        // C = (seq_total x seq_q) col-major = score (seq_q x seq_total) row-major.

        float *d_score_h = ctx->d_scores + h * seq_q * seq_total;

        CUBLAS_CHECK(cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_T, // K: (head_dim x seq_total) col-major with lda=n_kv*head_dim → op(K)=(seq_total x head_dim)
            CUBLAS_OP_N, // Q: (head_dim x seq_q)  col-major with ldb=n_heads*head_dim → op(Q)=(head_dim x seq_q)
            seq_total, seq_q, head_dim,
            &alpha1,
            d_k_cache + kv_h * head_dim, n_kv_heads * head_dim, // K (strided)
            d_q + h * head_dim, n_heads * head_dim,             // Q (strided)
            &beta0,
            d_score_h, seq_total));
    }

    // ── Step 2: causal mask + softmax ──────────────────────────────────────
    if (seq_q > 1)
    {
        dim3 threads(16, 16);
        dim3 blocks(
            (seq_total + 15) / 16,
            (seq_q + 15) / 16,
            n_heads);
        causal_mask_kernel<<<blocks, threads>>>(
            ctx->d_scores, n_heads, seq_q, seq_total, cache_offset);
        CUDA_CHECK(cudaGetLastError());
    }

    int n_rows = n_heads * seq_q;
    int sf_threads = 64;
    int sf_smem = 2 * (sf_threads / 32) * sizeof(float);
    softmax_kernel<<<n_rows, sf_threads, sf_smem>>>(ctx->d_scores, n_rows, seq_total);
    CUDA_CHECK(cudaGetLastError());

    // ── Step 3: attn_out[h] = scores[h] @ V[kv_h] ────────────────────────
    // score[h]: (seq_q, seq_total)
    // V[kv_h]: (seq_total, head_dim), stride n_kv_heads * head_dim
    // out[h]:  (seq_q, head_dim), stride n_heads * head_dim

    for (int h = 0; h < n_heads; h++)
    {
        int kv_h = h / grp;
        float *d_score_h = ctx->d_scores + h * seq_q * seq_total;
        float *d_out_h = d_out + h * head_dim;

        // attn_out(seq_q, head_dim) = score(seq_q, seq_total) @ V(seq_total, head_dim)
        // cuBLAS col-major: m=head_dim, n=seq_q, k=seq_total
        //
        // Same layout reasoning as K above: V slice for kv_h is (seq_total, head_dim)
        // row-major with row-stride n_kv_heads*head_dim.  As col-major this looks like
        // (head_dim x seq_total) with lda = n_kv_heads*head_dim — already V^T, use CUBLAS_OP_N.
        //
        // B = score (seq_total, seq_q) col-major = score (seq_q, seq_total) row-major, ldb=seq_total
        // C = out (head_dim, seq_q) col-major = out (seq_q, head_dim) row-major, ldc=n_heads*head_dim

        CUBLAS_CHECK(cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_N, // V is already V^T in col-major
            CUBLAS_OP_N, // scores
            head_dim, seq_q, seq_total,
            &alpha1f,
            d_v_cache + kv_h * head_dim, n_kv_heads * head_dim, // V (strided, col-major = V^T)
            d_score_h, seq_total,
            &beta0,
            d_out_h, n_heads * head_dim // output (strided by n_heads*head_dim)
            ));
    }

    return N730_OK;
}

/*
 * n730_linear_from_buf
 * Same as n730_linear but reads input from an arbitrary device buffer
 * instead of d_norm_buf. Used for the O-projection (input = attn_out)
 * and down-projection (input = gate after SwiGLU).
 */
N730_API int n730_linear_from_buf(
    void *ctx_ptr,
    const float *d_in, // (seq, in_dim) device — arbitrary input
    float *d_out,      // (seq, out_dim) device — output
    int seq_len,
    int in_dim,
    int out_dim)
{
    if (!ctx_ptr)
        return N730_NULL;
    N730CudaCtx *ctx = (N730CudaCtx *)ctx_ptr;

    const float alpha = 1.0f, beta = 0.0f;
    // d_out(seq, out_dim) = d_in(seq, in_dim) @ d_weights(out_dim, in_dim)^T
    CUBLAS_CHECK(cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, seq_len, in_dim,
        &alpha,
        ctx->d_weights, in_dim,
        d_in, in_dim,
        &beta,
        d_out, out_dim));
    return N730_OK;
}

/*
 * n730_device_alloc / n730_device_free
 * Let Python allocate/free device buffers for KV cache and projections.
 */
N730_API int n730_device_alloc(int n_floats, void **out_ptr)
{
    float *p;
    if (cudaMalloc(&p, n_floats * sizeof(float)) != cudaSuccess)
        return N730_OOM;
    *out_ptr = p;
    return N730_OK;
}

N730_API void n730_device_free(void *ptr)
{
    if (ptr)
        cudaFree(ptr);
}

/*
 * n730_memcpy_d2d / n730_memcpy_h2d / n730_memcpy_d2h
 * Raw memory copy helpers for Python to orchestrate.
 */
N730_API int n730_memcpy_d2d(void *dst, const void *src, int n_floats)
{
    CUDA_CHECK(cudaMemcpy(dst, src, n_floats * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return N730_OK;
}
N730_API int n730_memcpy_h2d(void *dst, const void *src, int n_floats)
{
    CUDA_CHECK(cudaMemcpy(dst, src, n_floats * sizeof(float),
                          cudaMemcpyHostToDevice));
    return N730_OK;
}
N730_API int n730_memcpy_d2h(void *dst, const void *src, int n_floats)
{
    CUDA_CHECK(cudaMemcpy(dst, src, n_floats * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return N730_OK;
}

N730_API int n730_sync()
{
    CUDA_CHECK(cudaDeviceSynchronize());
    return N730_OK;
}

N730_API const char *n730_cuda_version()
{
    return "N730 CUDA Kernel 0.1.0 / Project Bombakla / sm_35";
}
