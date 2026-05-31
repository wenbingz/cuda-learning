# 08 Naive Attention

目标：把 matmul 和 softmax 串起来理解 attention。

公式：

```text
scores = Q @ K^T / sqrt(head_dim)
weights = softmax(scores)
out = weights @ V
```

需要观察：

- `scores` shape 是 `seq_len x seq_len`
- `scores[query_pos, key_pos]` 是一个 query token 和一个 key token 的点积
- causal mask 如何屏蔽未来 token
- 中间矩阵为什么吃显存

本练习故意拆成 3 个 kernel：

```text
compute_scores : QK^T / sqrt(head_dim)
row_softmax    : 对每个 query 的 scores 做 softmax
compute_output : weights @ V
```

运行：

```bash
make run
```

进阶：

```text
理解 FlashAttention 为什么避免显式存完整 scores/weights。
```
