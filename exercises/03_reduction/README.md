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

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
reduction_sum passed with n=1048576, blocks=4096, threads_per_block=256, sum=4.1943e+06
```

## 实现

代码里包含两种路径：

```text
1. 全 GPU 多 kernel reduction：
   input -> partial sums -> partial sums -> ... -> single value

2. 教学对照路径：
   input -> one kernel partial sums -> CPU 合并 partial sums
```

第一条路径是真正的多 kernel reduction。第二条路径用于对照，帮助你看到第一轮 partial sums 是什么。

## 核心代码

动态 shared memory：

```cpp
extern __shared__ float shared[];
```

kernel launch 时指定大小：

```cpp
size_t shared_bytes = threads_per_block * sizeof(float);
reduce_sum_blocks<<<blocks, threads_per_block, shared_bytes>>>(...);
```

每个 thread 先加载一个元素：

```cpp
shared[tid] = (i < n) ? x[i] : 0.0f;
__syncthreads();
```

然后每轮 active thread 数减半：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    __syncthreads();
}
```

最后 `threadIdx.x == 0` 写出这个 block 的 partial sum：

```cpp
if (tid == 0) {
    partial_sums[blockIdx.x] = shared[0];
}
```

## 为什么需要多 kernel

`__syncthreads()` 只能同步一个 block 内的 thread，不能同步不同 block。

所以每个 block 先输出一个 partial sum。全 GPU 版本会把 partial sums 当成下一轮输入，继续启动同一个 reduction kernel：

```cpp
while (current_n > 1) {
    int current_blocks = (current_n + threads_per_block - 1) / threads_per_block;
    reduce_sum_blocks<<<current_blocks, threads_per_block, shared_bytes>>>(
        current,
        next,
        current_n
    );
    current = next;
    next = (next == d_tmp1) ? d_tmp2 : d_tmp1;
    current_n = current_blocks;
}
```

这里 `d_tmp1` 和 `d_tmp2` 是 ping-pong buffers：

```text
第 1 轮写 d_tmp1
第 2 轮写 d_tmp2
第 3 轮写 d_tmp1
...
```

两个 kernel launch 之间天然形成全局同步边界。
