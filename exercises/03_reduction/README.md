# 03 Reduction

目标：理解并行规约。

任务：

```text
sum(x)
max(x)
row-wise sum
row-wise max
```

重点：

- 每个 thread 先算局部结果
- block 内用 shared memory 合并
- 每轮 active thread 数减半
- 使用 `__syncthreads()`

思考：

- 为什么 reduction 通常容易 memory-bound？
- 如何减少分支和同步？

