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

