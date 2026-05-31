#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void row_layernorm(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += x[row * cols + col];
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared[0] / cols;

    float local_var_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float diff = x[row * cols + col] - mean;
        local_var_sum += diff * diff;
    }

    shared[tid] = local_var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float variance = shared[0] / cols;
    float inv_std = rsqrtf(variance + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        float normalized = (x[idx] - mean) * inv_std;
        y[idx] = normalized * gamma[col] + beta[col];
    }
}

int main() {
    const int rows = 128;
    const int cols = 512;
    const float eps = 1e-5f;
    const int n = rows * cols;
    const size_t bytes = n * sizeof(float);
    const size_t param_bytes = cols * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    std::vector<float> h_y(n);
    std::vector<float> h_ref(n);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            h_x[row * cols + col] = static_cast<float>((row * 3 + col) % 37) / 11.0f;
        }
    }
    for (int col = 0; col < cols; ++col) {
        h_gamma[col] = 1.0f + static_cast<float>(col % 5) * 0.01f;
        h_beta[col] = static_cast<float>(col % 7) * 0.001f;
    }

    for (int row = 0; row < rows; ++row) {
        float mean = 0.0f;
        for (int col = 0; col < cols; ++col) {
            mean += h_x[row * cols + col];
        }
        mean /= cols;

        float variance = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float diff = h_x[row * cols + col] - mean;
            variance += diff * diff;
        }
        variance /= cols;

        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            float normalized = (h_x[idx] - mean) * inv_std;
            h_ref[idx] = normalized * h_gamma[col] + h_beta[col];
        }
    }

    float* d_x = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, param_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), param_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), param_bytes, cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = rows;
    const size_t shared_bytes = threads_per_block * sizeof(float);
    row_layernorm<<<blocks, threads_per_block, shared_bytes>>>(
        d_x,
        d_gamma,
        d_beta,
        d_y,
        rows,
        cols,
        eps
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_y[i] - h_ref[i]) > 1e-4f) {
            std::cerr << "mismatch at " << i << ": got " << h_y[i]
                      << ", expected " << h_ref[i] << std::endl;
            ok = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));

    if (!ok) {
        return 1;
    }

    std::cout << "row_layernorm passed with shape=(" << rows << ", " << cols << ")"
              << ", blocks=" << blocks
              << ", threads_per_block=" << threads_per_block << std::endl;
    return 0;
}

