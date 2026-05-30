# 06 Softmax

目标：实现 row-wise softmax。

公式：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

为什么减 max：

```text
防止 exp 溢出，提高数值稳定性。
```

步骤：

```text
1. row-wise max
2. exp(x - max)
3. row-wise sum
4. divide
```

连接 Transformer：

```text
attention scores -> mask -> softmax -> attention weights
```

## 运行

在有 NVIDIA CUDA 环境的机器上：

```bash
make run
```

期望输出类似：

```text
row_softmax passed with shape=(128, 512), blocks=128, threads_per_block=256
```

## 第一版实现

第一版使用：

```text
一个 block 处理一行
block 内 threads 共同处理这一行的 cols
```

当 `cols > blockDim.x` 时，每个 thread 用 stride loop 处理多个列：

```cpp
for (int col = tid; col < cols; col += blockDim.x) {
    ...
}
```

## 核心步骤

1. 每个 thread 扫描自己负责的列，得到 `local_max`。
2. block 内 reduction 得到 `row_max`。
3. 每个 thread 计算 `exp(x - row_max)`，同时累加 `local_sum`。
4. block 内 reduction 得到 `row_sum`。
5. 每个 thread 写：

```cpp
y[row * cols + col] /= row_sum;
```

## 为什么和 attention 有关

attention 里每一行 scores 都要做 softmax：

```text
scores[row, :] -> attention_weights[row, :]
```

所以 row-wise softmax 是理解 attention kernel 的基础。
