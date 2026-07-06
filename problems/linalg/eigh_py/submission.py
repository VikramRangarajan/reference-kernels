#!POPCORN leaderboard eigh
#!POPCORN gpu B200

from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

torch.randn(3).cuda()

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <vector>
#include <mutex>
#include <algorithm>
#include <cfloat>

#define CUDA_CHECK(call) do {                                                      \
    cudaError_t err = call;                                                        \
    if (err != cudaSuccess) {                                                      \
        throw std::runtime_error(std::string("CUDA error: ") +                    \
                                 cudaGetErrorString(err));                         \
    }                                                                              \
} while(0)

#define CUSOLVER_CHECK(call) do {                                                  \
    cusolverStatus_t err = call;                                                   \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                          \
        throw std::runtime_error("cuSOLVER error: " +                             \
                                 std::to_string((int)err));                        \
    }                                                                              \
} while(0)

static cusolverDnHandle_t g_handle = nullptr;
static std::mutex g_mutex;

static void ensure_handle() {
    if (!g_handle) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_handle) {
            CUSOLVER_CHECK(cusolverDnCreate(&g_handle));
        }
    }
}

template <typename scalar_t>
cudaDataType cuda_data_type();

template <>
cudaDataType cuda_data_type<double>() { return CUDA_R_64F; }

template <>
cudaDataType cuda_data_type<float>() { return CUDA_R_32F; }

static void* g_dev_work = nullptr;
static size_t g_dev_work_bytes = 0;
static void* g_d_D = nullptr;
static size_t g_d_D_bytes = 0;
static void* g_d_E = nullptr;
static size_t g_d_E_bytes = 0;
static void* g_d_tau = nullptr;
static size_t g_d_tau_bytes = 0;
static void* g_d_T = nullptr;
static size_t g_d_T_bytes = 0;
static int* g_d_info = nullptr;
static void* g_d_scratch = nullptr;
static size_t g_d_scratch_bytes = 0;
static void* g_d_maxval = nullptr;

static void ensure_tridiag_workspace(size_t work_bytes, int64_t n, int elem_size, size_t scratch_bytes) {
    if (work_bytes > g_dev_work_bytes) {
        if (g_dev_work) cudaFree(g_dev_work);
        CUDA_CHECK(cudaMalloc(&g_dev_work, work_bytes));
        g_dev_work_bytes = work_bytes;
    }
    size_t n_bytes = (size_t)n * elem_size;
    if (n_bytes > g_d_D_bytes) {
        if (g_d_D) cudaFree(g_d_D);
        CUDA_CHECK(cudaMalloc(&g_d_D, n_bytes));
        g_d_D_bytes = n_bytes;
    }
    if (n_bytes > g_d_E_bytes) {
        if (g_d_E) cudaFree(g_d_E);
        CUDA_CHECK(cudaMalloc(&g_d_E, n_bytes));
        g_d_E_bytes = n_bytes;
    }
    if (n_bytes > g_d_tau_bytes) {
        if (g_d_tau) cudaFree(g_d_tau);
        CUDA_CHECK(cudaMalloc(&g_d_tau, n_bytes));
        g_d_tau_bytes = n_bytes;
    }
    size_t T_bytes = (size_t)n * n * elem_size;
    if (T_bytes > g_d_T_bytes) {
        if (g_d_T) cudaFree(g_d_T);
        CUDA_CHECK(cudaMalloc(&g_d_T, T_bytes));
        g_d_T_bytes = T_bytes;
    }
    if (scratch_bytes > g_d_scratch_bytes) {
        if (g_d_scratch) cudaFree(g_d_scratch);
        CUDA_CHECK(cudaMalloc(&g_d_scratch, scratch_bytes));
        g_d_scratch_bytes = scratch_bytes;
    }
    if (!g_d_info) {
        CUDA_CHECK(cudaMalloc(&g_d_info, sizeof(int)));
    }
    if (!g_d_maxval) {
        CUDA_CHECK(cudaMalloc(&g_d_maxval, sizeof(float)));
    }
}

template <typename scalar_t>
__global__ void fill_tridiagonal_kernel(scalar_t* T, int n, const scalar_t* d, const scalar_t* e) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        scalar_t val = 0;
        if (row == col) {
            val = d[row];
        } else if (row == col + 1) {
            val = e[col];
        } else if (col == row + 1) {
            val = e[row];
        }
        T[col * n + row] = val;
    }
}

template <typename scalar_t>
__global__ void max_abs_kernel(const scalar_t* A, int64_t total, float* block_max) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + tid;
    float local_max = 0.0f;
    while (gid < total) {
        float val = (float)A[gid];
        if (val < 0) val = -val;
        if (val > local_max) local_max = val;
        gid += (int64_t)gridDim.x * blockDim.x;
    }
    shared[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) shared[tid] = shared[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = shared[0];
}

__global__ void reduce_max_kernel(const float* block_max, int n_blocks, float* result) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float val = 0.0f;
    if (tid < n_blocks) val = block_max[tid];
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) shared[tid] = shared[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *result = shared[0];
}

template <typename scalar_t>
__global__ void scale_matrix_kernel(scalar_t* A, int64_t total, scalar_t scale) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        A[tid] *= scale;
    }
}

template <typename scalar_t>
__global__ void unscale_w_kernel(scalar_t* W, int64_t total, scalar_t inv_scale) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        W[tid] *= inv_scale;
    }
}

static float compute_scale(cusolverDnHandle_t handle, float* d_A, int64_t total_elems) {
    // Compute max absolute value across the entire batch
    int block_size = 256;
    int n_blocks = std::min((int64_t)4096, (total_elems + block_size - 1) / block_size);
    size_t smem = block_size * sizeof(float);
    
    float* d_block_max = (float*)g_d_scratch;
    if ((size_t)n_blocks * sizeof(float) > g_d_scratch_bytes) {
        if (g_d_scratch) cudaFree(g_d_scratch);
        CUDA_CHECK(cudaMalloc(&g_d_scratch, (size_t)n_blocks * sizeof(float)));
        g_d_scratch_bytes = (size_t)n_blocks * sizeof(float);
    }
    
    max_abs_kernel<<<n_blocks, block_size, smem>>>(d_A, total_elems, d_block_max);
    
    reduce_max_kernel<<<1, block_size, block_size * sizeof(float)>>>(d_block_max, n_blocks, (float*)g_d_maxval);
    
    float max_val = 0.0f;
    CUDA_CHECK(cudaMemcpy(&max_val, g_d_maxval, sizeof(float), cudaMemcpyDeviceToHost));
    
    if (max_val < 1.0f || !std::isfinite(max_val)) return 1.0f;
    
    // Threshold: avoid overflow when squares are summed.
    // For n up to 4096, safe max is roughly sqrt(FLT_MAX / 4096) ~ 9e16
    float threshold = 1e15f;
    if (max_val > threshold) {
        return 1.0f / max_val;
    }
    return 1.0f;
}

template <typename scalar_t>
void batched_eigh_kernel(cusolverDnHandle_t handle,
                          cusolverEigMode_t jobz, cublasFillMode_t uplo,
                          int64_t n, scalar_t* d_A, int64_t lda,
                          scalar_t* d_W, int64_t batchSize) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int n_int = (int)n;

    int lwork_sytrd = 0, lwork_syevd = 0, lwork_ormtr = 0;

    if constexpr (std::is_same_v<scalar_t, double>) {
        CUSOLVER_CHECK(cusolverDnDsytrd_bufferSize(
            handle, uplo, n_int, nullptr, n_int, nullptr, nullptr, nullptr, &lwork_sytrd));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
            handle, jobz, uplo, n_int, nullptr, n_int, nullptr, &lwork_syevd));
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
            CUSOLVER_CHECK(cusolverDnDormtr_bufferSize(
                handle, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N,
                n_int, n_int, nullptr, n_int, nullptr, nullptr, n_int, &lwork_ormtr));
        }
    } else {
        CUSOLVER_CHECK(cusolverDnSsytrd_bufferSize(
            handle, uplo, n_int, nullptr, n_int, nullptr, nullptr, nullptr, &lwork_sytrd));
        CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
            handle, jobz, uplo, n_int, nullptr, n_int, nullptr, &lwork_syevd));
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
            CUSOLVER_CHECK(cusolverDnSormtr_bufferSize(
                handle, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N,
                n_int, n_int, nullptr, n_int, nullptr, nullptr, n_int, &lwork_ormtr));
        }
    }

    int max_lwork = std::max({lwork_sytrd, lwork_syevd, lwork_ormtr});
    int alloc_lwork = std::max(max_lwork, 100000);
    size_t work_bytes = (size_t)alloc_lwork * sizeof(scalar_t);
    size_t scratch_bytes = 4096 * sizeof(float);

    ensure_tridiag_workspace(work_bytes, n, (int)sizeof(scalar_t), scratch_bytes);

    // Compute and apply scaling to prevent overflow in sytrd's Householder norms
    int64_t total_elems = batchSize * n * n;
    float scale = 1.0f;
    if constexpr (std::is_same_v<scalar_t, float>) {
        scale = compute_scale(handle, d_A, total_elems);
        if (scale != 1.0f) {
            int64_t flat_size = total_elems;
            int tb = 256;
            int tg = (flat_size + tb - 1) / tb;
            tg = std::min(tg, 65535);
            scale_matrix_kernel<<<tg, tb>>>(d_A, flat_size, (scalar_t)scale);
        }
    }

    dim3 block(16, 16);
    dim3 grid((n_int + 15) / 16, (n_int + 15) / 16);

    for (int64_t b = 0; b < batchSize; b++) {
        scalar_t* A_b = d_A + b * n * lda;
        scalar_t* W_b = d_W + b * n;

        scalar_t* d_D = (scalar_t*)g_d_D;
        scalar_t* d_E = (scalar_t*)g_d_E;
        scalar_t* d_tau = (scalar_t*)g_d_tau;
        scalar_t* d_T = (scalar_t*)g_d_T;

        if constexpr (std::is_same_v<scalar_t, double>) {
            CUSOLVER_CHECK(cusolverDnDsytrd(
                handle, uplo, n_int, A_b, (int)lda,
                d_D, d_E, d_tau,
                (scalar_t*)g_dev_work, lwork_sytrd, g_d_info));
        } else {
            CUSOLVER_CHECK(cusolverDnSsytrd(
                handle, uplo, n_int, A_b, (int)lda,
                d_D, d_E, d_tau,
                (scalar_t*)g_dev_work, lwork_sytrd, g_d_info));
        }

        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, g_d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("sytrd failed at batch " + std::to_string(b) +
                                     " with info=" + std::to_string(h_info));
        }

        fill_tridiagonal_kernel<<<grid, block>>>(d_T, n_int, d_D, d_E);

        if constexpr (std::is_same_v<scalar_t, double>) {
            CUSOLVER_CHECK(cusolverDnDsyevd(
                handle, jobz, uplo, n_int, d_T, n_int,
                W_b, (scalar_t*)g_dev_work, lwork_syevd, g_d_info));
        } else {
            CUSOLVER_CHECK(cusolverDnSsyevd(
                handle, jobz, uplo, n_int, d_T, n_int,
                W_b, (scalar_t*)g_dev_work, lwork_syevd, g_d_info));
        }

        CUDA_CHECK(cudaMemcpy(&h_info, g_d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("syevd failed at batch " + std::to_string(b) +
                                     " with info=" + std::to_string(h_info));
        }

        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
            if constexpr (std::is_same_v<scalar_t, double>) {
                CUSOLVER_CHECK(cusolverDnDormtr(
                    handle, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N,
                    n_int, n_int, A_b, (int)lda, d_tau,
                    d_T, n_int, (scalar_t*)g_dev_work, lwork_ormtr, g_d_info));
            } else {
                CUSOLVER_CHECK(cusolverDnSormtr(
                    handle, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N,
                    n_int, n_int, A_b, (int)lda, d_tau,
                    d_T, n_int, (scalar_t*)g_dev_work, lwork_ormtr, g_d_info));
            }

            CUDA_CHECK(cudaMemcpy(&h_info, g_d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_info != 0) {
                throw std::runtime_error("ormtr failed at batch " + std::to_string(b) +
                                         " with info=" + std::to_string(h_info));
            }

            CUDA_CHECK(cudaMemcpy(A_b, d_T, (size_t)n * n * sizeof(scalar_t),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    // Unscale eigenvalues
    if (scale != 1.0f) {
        scalar_t inv_scale = (scalar_t)(1.0f / scale);
        int64_t flat_size = batchSize * n;
        int tb = 256;
        int tg = (flat_size + tb - 1) / tb;
        tg = std::min(tg, 65535);
        unscale_w_kernel<<<tg, tb>>>(d_W, flat_size, inv_scale);
    }
}

std::vector<torch::Tensor> syevd_cuda(torch::Tensor A, bool compute_eigenvectors) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be 2 or 3-dimensional");
    TORCH_CHECK(A.size(-1) == A.size(-2), "A must be square");
    TORCH_CHECK(A.scalar_type() == torch::kFloat64 ||
                A.scalar_type() == torch::kFloat32,
                "A must be float32 or float64");

    auto A_contig = A.contiguous();
    int64_t n = A_contig.size(-1);
    int64_t lda = n;
    int64_t batch = A_contig.dim() == 2 ? 1 : A_contig.size(0);

    ensure_handle();

    torch::Tensor eigenvectors = torch::empty_like(A_contig);
    eigenvectors.copy_(A_contig);
    torch::Tensor eigenvalues = torch::empty({batch, n}, A_contig.options());

    cusolverEigMode_t jobz = compute_eigenvectors
        ? CUSOLVER_EIG_MODE_VECTOR
        : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    auto dtype = A_contig.scalar_type();

    AT_DISPATCH_FLOATING_TYPES(dtype, "syevd_cuda", [&] {
        batched_eigh_kernel<scalar_t>(
            g_handle, jobz, uplo, n,
            eigenvectors.data_ptr<scalar_t>(), lda,
            eigenvalues.data_ptr<scalar_t>(), batch);
    });

    return {eigenvectors.transpose(1, 2).contiguous(), eigenvalues};
}
"""

CPP_SRC = """
std::vector<torch::Tensor> syevd_cuda(torch::Tensor A, bool compute_eigenvectors);
"""

Path("build").mkdir(exist_ok=True)
_syevd_module = load_inline(
    name="syevd_module",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["syevd_cuda"],
    verbose=True,
    extra_ldflags=["-lcusolver", "-lcublas"],
    build_directory="build",
)


def custom_kernel(data: input_t) -> output_t:
    result = _syevd_module.syevd_cuda(data, True)
    return (result[0], result[1])
