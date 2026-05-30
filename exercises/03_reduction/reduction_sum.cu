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

__global__ void reduce_sum_blocks(
    const float* x,
    float* partial_sums,
    int n
) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

int main() {
    const int n = 1 << 20;
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    const size_t input_bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>((i % 7) + 1);
    }

    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_sum += h_x[i];
    }

    float* d_x = nullptr;
    float* d_tmp1 = nullptr;
    float* d_tmp2 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp1, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp2, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), input_bytes, cudaMemcpyHostToDevice));

    size_t shared_bytes = threads_per_block * sizeof(float);

    float* current = d_x;
    float* next = d_tmp1;
    int current_n = n;
    int passes = 0;

    while (current_n > 1) {
        int current_blocks = (current_n + threads_per_block - 1) / threads_per_block;
        reduce_sum_blocks<<<current_blocks, threads_per_block, shared_bytes>>>(
            current,
            next,
            current_n
        );
        CUDA_CHECK(cudaGetLastError());

        current = next;
        next = (next == d_tmp1) ? d_tmp2 : d_tmp1;
        current_n = current_blocks;
        ++passes;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, current, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_tmp1, 0, blocks * sizeof(float)));
    reduce_sum_blocks<<<blocks, threads_per_block, shared_bytes>>>(
        d_x,
        d_tmp1,
        n
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partial(blocks);
    CUDA_CHECK(cudaMemcpy(
        h_partial.data(),
        d_tmp1,
        blocks * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float cpu_combined_partial_sum = 0.0f;
    for (float value : h_partial) {
        cpu_combined_partial_sum += value;
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));

    if (std::fabs(gpu_sum - cpu_sum) > 1e-3f) {
        std::cerr << "sum mismatch: got " << gpu_sum
                  << ", expected " << cpu_sum << std::endl;
        return 1;
    }
    if (std::fabs(cpu_combined_partial_sum - cpu_sum) > 1e-3f) {
        std::cerr << "partial sum mismatch: got " << cpu_combined_partial_sum
                  << ", expected " << cpu_sum << std::endl;
        return 1;
    }

    std::cout << "reduction_sum passed with n=" << n
              << ", blocks=" << blocks
              << ", threads_per_block=" << threads_per_block
              << ", gpu_passes=" << passes
              << ", sum=" << gpu_sum << std::endl;
    return 0;
}
