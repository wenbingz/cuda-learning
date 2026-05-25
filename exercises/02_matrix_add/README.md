# 02 Matrix Add

目标：理解二维数据如何映射到 CUDA grid。

任务：

```text
C[row, col] = A[row, col] + B[row, col]
```

需要理解：

- 2D grid / 2D block
- row-major memory layout
- `(row, col)` 到一维地址的转换

地址计算：

```text
idx = row * width + col
```

