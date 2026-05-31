# 07 LayerNorm

目标：实现 LayerNorm forward。

公式：

```text
mean = average(x)
var = average((x - mean)^2)
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

重点：

- row-wise mean
- row-wise variance
- normalize
- affine transform

连接 Transformer：

```text
LayerNorm 稳定每层 hidden states 的分布。
```

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
row_layernorm passed with shape=(128, 512), blocks=128, threads_per_block=256
```

## 第一版实现

第一版使用：

```text
一个 block 处理一行 hidden vector
block 内 threads 共同处理 cols
```

当 `cols > blockDim.x` 时，每个 thread 用 stride loop 处理多个列：

```cpp
for (int col = tid; col < cols; col += blockDim.x) {
    ...
}
```

## 核心步骤

1. 每个 thread 累加自己负责的列，得到 `local_sum`。
2. block 内 sum reduction 得到 `mean`。
3. 每个 thread 累加 `(x - mean)^2`，得到 `local_var_sum`。
4. block 内 sum reduction 得到 `variance`。
5. 每个 thread 写：

```cpp
float normalized = (x[idx] - mean) * rsqrtf(variance + eps);
y[idx] = normalized * gamma[col] + beta[col];
```

## 和 Transformer 的关系

Transformer 里的 LayerNorm/RMSNorm 都是对每个 token 的 hidden vector 做归一化。

如果 hidden states 是：

```text
batch x time x hidden_dim
```

那么可以把：

```text
rows = batch * time
cols = hidden_dim
```

每一行就是一个 token 的 hidden vector。
