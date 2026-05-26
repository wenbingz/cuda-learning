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

__global__ void matrix_add(
    const float* a,
    const float* b,
    float* c,
    int height,
    int width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int height = 1024;
    const int width = 2048;
    const int n = height * width;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = row * width + col;
            h_a[idx] = static_cast<float>(row);
            h_b[idx] = static_cast<float>(col);
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    matrix_add<<<grid, block>>>(d_a, d_b, d_c, height, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = row * width + col;
            float expected = h_a[idx] + h_b[idx];
            if (std::fabs(h_c[idx] - expected) > 1e-5f) {
                std::cerr << "mismatch at (" << row << ", " << col
                          << "): got " << h_c[idx]
                          << ", expected " << expected << std::endl;
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (!ok) {
        return 1;
    }

    std::cout << "matrix_add passed with shape=(" << height << ", " << width << ")"
              << ", grid=(" << grid.x << ", " << grid.y << ")"
              << ", block=(" << block.x << ", " << block.y << ")" << std::endl;
    return 0;
}

