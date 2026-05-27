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

__global__ void matmul_naive(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;
        for (int inner = 0; inner < k; ++inner) {
            acc += a[row * k + inner] * b[inner * n + col];
        }
        c[row * n + col] = acc;
    }
}

int main() {
    const int m = 256;
    const int k = 512;
    const int n = 384;

    const size_t a_bytes = m * k * sizeof(float);
    const size_t b_bytes = k * n * sizeof(float);
    const size_t c_bytes = m * n * sizeof(float);

    std::vector<float> h_a(m * k);
    std::vector<float> h_b(k * n);
    std::vector<float> h_c(m * n);
    std::vector<float> h_ref(m * n);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            h_a[row * k + col] = static_cast<float>((row + col) % 13) / 13.0f;
        }
    }
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            h_b[row * n + col] = static_cast<float>((row - col) % 17) / 17.0f;
        }
    }

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                acc += h_a[row * k + inner] * h_b[inner * n + col];
            }
            h_ref[row * n + col] = acc;
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );

    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int idx = row * n + col;
            if (std::fabs(h_c[idx] - h_ref[idx]) > 1e-3f) {
                std::cerr << "mismatch at (" << row << ", " << col
                          << "): got " << h_c[idx]
                          << ", expected " << h_ref[idx] << std::endl;
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

    std::cout << "matmul_naive passed with A=(" << m << ", " << k << ")"
              << ", B=(" << k << ", " << n << ")"
              << ", C=(" << m << ", " << n << ")"
              << ", grid=(" << grid.x << ", " << grid.y << ")"
              << ", block=(" << block.x << ", " << block.y << ")" << std::endl;
    return 0;
}

