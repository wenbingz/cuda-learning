# 05 Tiled Matmul

目标：用 shared memory 优化矩阵乘法。

核心思想：

```text
把 A 和 B 的一小块 tile 读入 shared memory
block 内多个 thread 重复使用这些 tile
```

典型流程：

```text
for each tile along K:
    load A tile to shared memory
    load B tile to shared memory
    __syncthreads()
    compute partial dot product
    __syncthreads()
```

思考：

- tile size 如何选择？
- shared memory 为什么比 global memory 快？
- `__syncthreads()` 为什么必须存在？

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
matmul_tiled passed with A=(256, 512), B=(512, 384), C=(256, 384), tile_size=16, grid=(24, 16), block=(16, 16)
```

## 代码阅读重点

一个 block 负责 C 的一个 `16 x 16` tile：

```cpp
int row = blockIdx.y * TILE_SIZE + threadIdx.y;
int col = blockIdx.x * TILE_SIZE + threadIdx.x;
```

每轮沿 K 维加载一块 A 和一块 B：

```cpp
tile_a[threadIdx.y][threadIdx.x] = a[row * k + a_col];
tile_b[threadIdx.y][threadIdx.x] = b[b_row * n + col];
```

然后 block 内所有 thread 等待 tile 加载完成：

```cpp
__syncthreads();
```

再用 shared memory 里的 tile 做部分点积：

```cpp
for (int inner = 0; inner < TILE_SIZE; ++inner) {
    acc += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
}
```

最后再同步一次，确保所有 thread 都用完当前 tile，然后进入下一轮加载。

## 为什么比 naive 更好

naive matmul 中，每个 thread 都直接从 global memory 读取 A/B。

tiled matmul 中，一个 block 里的 thread 协作把 A/B 的 tile 读到 shared memory，然后共同复用：

```text
global memory 读取次数减少
shared memory 访问更快
同一块数据被 16x16 个输出元素复用
```
