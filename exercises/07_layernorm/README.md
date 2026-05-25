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

