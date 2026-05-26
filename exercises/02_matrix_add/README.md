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

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
matrix_add passed with shape=(1024, 2048), grid=(128, 64), block=(16, 16)
```

## 代码阅读重点

```cpp
dim3 block(16, 16);
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
```

- `block.x` 覆盖列方向。
- `block.y` 覆盖行方向。
- `grid.x` 是列方向需要多少个 block。
- `grid.y` 是行方向需要多少个 block。

kernel 内部：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

这两行把 block 坐标和 thread 坐标转换成矩阵元素坐标。
