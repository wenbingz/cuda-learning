# 08 Attention

目标：把 matmul 和 softmax 串起来理解 attention。

公式：

```text
scores = Q @ K^T / sqrt(head_dim)
weights = softmax(scores)
out = weights @ V
```

需要观察：

- `scores` shape 是 `seq_len x seq_len`
- causal mask 如何屏蔽未来 token
- 中间矩阵为什么吃显存

进阶：

```text
理解 FlashAttention 为什么避免显式存完整 scores/weights。
```

