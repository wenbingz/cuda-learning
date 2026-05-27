# 04 Naive Matmul

目标：写最朴素的矩阵乘法。

任务：

```text
C[M, N] = A[M, K] @ B[K, N]
```

策略：

```text
一个 thread 计算一个 C[row, col]
```

每个 thread：

```text
for k in 0..K-1:
    acc += A[row, k] * B[k, col]
```

思考：

- 每个 `C[row, col]` 读了多少 global memory？
- 相邻 thread 访问 B 是否连续？
- 为什么这个版本会比 cuBLAS 慢很多？

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
matmul_naive passed with A=(256, 512), B=(512, 384), C=(256, 384), grid=(24, 16), block=(16, 16)
```

## 代码阅读重点

每个 thread 负责一个输出元素：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

如果 `row < m && col < n`，这个 thread 计算：

```cpp
float acc = 0.0f;
for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
}
c[row * n + col] = acc;
```

这里的 `inner` 就是矩阵乘法里的 K 维。

## 为什么叫 naive

这个版本每个 thread 都直接从 global memory 读取 A 的一行和 B 的一列，没有复用数据。

后面的 tiled matmul 会把 A/B 的小块先读到 shared memory，让同一个 block 内的 thread 复用这些数据。
