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

__global__ void compute_scores(
    const float* q,
    const float* k,
    float* scores,
    int seq_len,
    int head_dim,
    bool causal
) {
    int key_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int query_pos = blockIdx.y * blockDim.y + threadIdx.y;

    if (query_pos >= seq_len || key_pos >= seq_len) {
        return;
    }

    int score_idx = query_pos * seq_len + key_pos;
    if (causal && key_pos > query_pos) {
        scores[score_idx] = -std::numeric_limits<float>::infinity();
        return;
    }

    float acc = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        acc += q[query_pos * head_dim + dim] * k[key_pos * head_dim + dim];
    }

    scores[score_idx] = acc * rsqrtf(static_cast<float>(head_dim));
}

__global__ void row_softmax(
    const float* scores,
    float* weights,
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
        local_max = fmaxf(local_max, scores[row * cols + col]);
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
        float value = expf(scores[row * cols + col] - row_max);
        weights[row * cols + col] = value;
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
        weights[row * cols + col] /= row_sum;
    }
}

__global__ void compute_output(
    const float* weights,
    const float* v,
    float* out,
    int seq_len,
    int head_dim
) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    int query_pos = blockIdx.y * blockDim.y + threadIdx.y;

    if (query_pos >= seq_len || dim >= head_dim) {
        return;
    }

    float acc = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
        acc += weights[query_pos * seq_len + key_pos] * v[key_pos * head_dim + dim];
    }
    out[query_pos * head_dim + dim] = acc;
}

int main() {
    const int seq_len = 8;
    const int head_dim = 16;
    const bool causal = true;

    const size_t qkv_bytes = seq_len * head_dim * sizeof(float);
    const size_t score_bytes = seq_len * seq_len * sizeof(float);

    std::vector<float> h_q(seq_len * head_dim);
    std::vector<float> h_k(seq_len * head_dim);
    std::vector<float> h_v(seq_len * head_dim);
    std::vector<float> h_scores(seq_len * seq_len);
    std::vector<float> h_weights(seq_len * seq_len);
    std::vector<float> h_out(seq_len * head_dim);
    std::vector<float> h_ref(seq_len * head_dim);

    for (int row = 0; row < seq_len; ++row) {
        for (int dim = 0; dim < head_dim; ++dim) {
            int idx = row * head_dim + dim;
            h_q[idx] = static_cast<float>((row + dim) % 7 - 3) / 7.0f;
            h_k[idx] = static_cast<float>((row * 2 + dim) % 11 - 5) / 11.0f;
            h_v[idx] = static_cast<float>((row - dim) % 13) / 13.0f;
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int query_pos = 0; query_pos < seq_len; ++query_pos) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
            int score_idx = query_pos * seq_len + key_pos;
            if (causal && key_pos > query_pos) {
                h_scores[score_idx] = -std::numeric_limits<float>::infinity();
            } else {
                float acc = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim) {
                    acc += h_q[query_pos * head_dim + dim] * h_k[key_pos * head_dim + dim];
                }
                h_scores[score_idx] = acc * scale;
            }
            row_max = std::max(row_max, h_scores[score_idx]);
        }

        float row_sum = 0.0f;
        for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
            int score_idx = query_pos * seq_len + key_pos;
            float value = std::exp(h_scores[score_idx] - row_max);
            h_weights[score_idx] = value;
            row_sum += value;
        }

        for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
            h_weights[query_pos * seq_len + key_pos] /= row_sum;
        }

        for (int dim = 0; dim < head_dim; ++dim) {
            float acc = 0.0f;
            for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
                acc += h_weights[query_pos * seq_len + key_pos] * h_v[key_pos * head_dim + dim];
            }
            h_ref[query_pos * head_dim + dim] = acc;
        }
    }

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_scores = nullptr;
    float* d_weights = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_k, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, qkv_bytes));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), qkv_bytes, cudaMemcpyHostToDevice));

    dim3 score_block(16, 16);
    dim3 score_grid(
        (seq_len + score_block.x - 1) / score_block.x,
        (seq_len + score_block.y - 1) / score_block.y
    );
    compute_scores<<<score_grid, score_block>>>(d_q, d_k, d_scores, seq_len, head_dim, causal);
    CUDA_CHECK(cudaGetLastError());

    const int softmax_threads = 256;
    const size_t shared_bytes = softmax_threads * sizeof(float);
    row_softmax<<<seq_len, softmax_threads, shared_bytes>>>(d_scores, d_weights, seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());

    dim3 out_block(16, 16);
    dim3 out_grid(
        (head_dim + out_block.x - 1) / out_block.x,
        (seq_len + out_block.y - 1) / out_block.y
    );
    compute_output<<<out_grid, out_block>>>(d_weights, d_v, d_out, seq_len, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, qkv_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int row = 0; row < seq_len; ++row) {
        for (int dim = 0; dim < head_dim; ++dim) {
            int idx = row * head_dim + dim;
            if (std::fabs(h_out[idx] - h_ref[idx]) > 1e-5f) {
                std::cerr << "mismatch at (" << row << ", " << dim
                          << "): got " << h_out[idx]
                          << ", expected " << h_ref[idx] << std::endl;
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_out));

    if (!ok) {
        return 1;
    }

    std::cout << "attention_naive passed with seq_len=" << seq_len
              << ", head_dim=" << head_dim
              << ", causal=" << causal
              << ", score_grid=(" << score_grid.x << ", " << score_grid.y << ")"
              << ", score_block=(" << score_block.x << ", " << score_block.y << ")"
              << std::endl;
    return 0;
}
