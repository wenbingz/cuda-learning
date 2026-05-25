# 矩阵乘法原理笔记

矩阵乘法：

```text
C[M, N] = A[M, K] @ B[K, N]
```

单个元素：

```text
C[i, j] = sum(A[i, k] * B[k, j] for k in 0..K-1)
```

## Naive Matmul

最直接的方法：

```text
一个 thread 计算一个 C[i, j]
每个 thread 从 global memory 读取 A 的一行和 B 的一列
```

问题：

- global memory 重复读取很多
- B 的列访问可能不连续
- 数据复用差

## Tiled Matmul

把矩阵切成 tile：

```text
每个 block 负责 C 的一个 tile
A tile 和 B tile 先加载到 shared memory
block 内多个 thread 复用这些数据
```

优势：

- 减少 global memory 读取
- 提高数据复用
- 更接近 GPU 的计算方式

## 思考问题

- tile size 太小会怎样？
- tile size 太大会怎样？
- shared memory 占用如何影响 occupancy？

