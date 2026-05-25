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

