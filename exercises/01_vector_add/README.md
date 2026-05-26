# 01 Vector Add

目标：写第一个 CUDA kernel。

任务：

```text
C[i] = A[i] + B[i]
```

需要理解：

- 一个 thread 负责一个元素
- `blockIdx.x` / `threadIdx.x`
- 越界检查
- CPU 结果对照

伪代码：

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
vector_add passed with n=1048576, blocks=4096, threads_per_block=256
```

## 代码阅读重点

- `cudaMalloc`：在 GPU global memory 分配数组。
- `cudaMemcpyHostToDevice`：把 CPU 数据拷到 GPU。
- `vector_add<<<blocks, threads_per_block>>>`：启动 kernel。
- `blockIdx.x * blockDim.x + threadIdx.x`：计算当前 thread 的全局编号。
- `if (i < n)`：防止最后一个 block 的多余 thread 越界。
- `cudaDeviceSynchronize`：等待 GPU kernel 执行完成。
- `cudaMemcpyDeviceToHost`：把 GPU 结果拷回 CPU 校验。
