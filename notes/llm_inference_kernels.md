# LLM 推理内核笔记

## Prefill

prefill 一次性处理完整 prompt：

```text
input shape: batch x prompt_len
attention: prompt_len x prompt_len
```

特点：

- 大矩阵计算多
- GPU 利用率通常较高
- 影响 TTFT

## Decode

decode 每步只处理一个新 token：

```text
input shape: batch x 1
attention: 1 x cached_seq_len
```

特点：

- 每步计算小
- 需要持续读取 KV cache
- 容易受 memory bandwidth 和调度影响
- 影响 TPOT

## KV Cache

KV cache 粗略大小：

```text
batch_size
* seq_len
* n_layer
* 2
* n_kv_heads
* head_dim
* bytes_per_value
```

它决定：

- 上下文长度能开多大
- 并发能开多高
- decode 阶段要读多少历史 K/V

## FlashAttention

普通 attention 显式产生：

```text
QK^T: seq_len x seq_len
```

FlashAttention 的核心思想：

```text
分块计算 attention
避免把完整 attention matrix 写入 HBM
用 online softmax 保持数值正确
```

目标：

```text
减少 HBM 读写，而不改变 attention 数学结果。
```

