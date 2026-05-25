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

