#include <cmath>
#include <iostream>
#include <limits>
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

__global__ void row_softmax(
    const float* x,
    float* y,
    int rows,
    int cols
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float local_max = -std::numeric_limits<float>::infinity();
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, x[row * cols + col]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = expf(x[row * cols + col] - row_max);
        y[row * cols + col] = value;
        local_sum += value;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = shared[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row * cols + col] /= row_sum;
    }
}

int main() {
    const int rows = 128;
    const int cols = 512;
    const int n = rows * cols;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_ref(n);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            h_x[row * cols + col] = static_cast<float>((row + col) % 31) / 7.0f;
        }
    }

    for (int row = 0; row < rows; ++row) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, h_x[row * cols + col]);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = std::exp(h_x[row * cols + col] - row_max);
            h_ref[row * cols + col] = value;
            row_sum += value;
        }

        for (int col = 0; col < cols; ++col) {
            h_ref[row * cols + col] /= row_sum;
        }
    }

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = rows;
    const size_t shared_bytes = threads_per_block * sizeof(float);
    row_softmax<<<blocks, threads_per_block, shared_bytes>>>(d_x, d_y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_y[i] - h_ref[i]) > 1e-5f) {
            std::cerr << "mismatch at " << i << ": got " << h_y[i]
                      << ", expected " << h_ref[i] << std::endl;
            ok = false;
            break;
        }
    }

    for (int row = 0; row < rows && ok; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += h_y[row * cols + col];
        }
        if (std::fabs(sum - 1.0f) > 1e-4f) {
            std::cerr << "row " << row << " sum mismatch: " << sum << std::endl;
            ok = false;
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    if (!ok) {
        return 1;
    }

    std::cout << "row_softmax passed with shape=(" << rows << ", " << cols << ")"
              << ", blocks=" << blocks
              << ", threads_per_block=" << threads_per_block << std::endl;
    return 0;
}

