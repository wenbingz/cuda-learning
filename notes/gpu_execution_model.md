# GPU 执行模型笔记

## 核心概念

CUDA kernel 启动时会创建一个 grid：

```text
grid
└── blocks
    └── threads
```

每个 thread 有自己的索引：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

常见映射方式：

```text
一维数组：一个 thread 处理一个元素
二维矩阵：一个 thread 处理一个元素或一个 tile 中的一个元素
reduction：一个 thread 处理多个元素，再在 block 内合并
```

## Warp

NVIDIA GPU 通常以 warp 为单位调度，一个 warp 有 32 个 thread。

需要重点关注：

- warp 内 thread 执行同一条指令
- 分支不一致会造成 divergence
- 连续 thread 访问连续地址更容易 coalescing

## Memory Hierarchy

从快到慢大致是：

```text
register
shared memory
L2 cache
global memory / HBM
```

CUDA 优化常常是在做一件事：

```text
少读 global memory，多复用 shared memory/register 中的数据。
```

