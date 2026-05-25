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

