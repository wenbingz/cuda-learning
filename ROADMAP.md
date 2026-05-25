# CUDA/GPU 矩阵计算学习计划

## 第 1 周：GPU 执行模型

目标：

- 理解 CPU 和 GPU 的差异
- 理解 thread、warp、block、grid、SM
- 理解 global memory、shared memory、register
- 能画出一个 kernel launch 后 thread 如何覆盖数组

练习：

- `01_vector_add`
- `02_matrix_add`

验收问题：

- 为什么 GPU 适合大量相同操作？
- 为什么分支发散会慢？
- 为什么访存合并很重要？

## 第 2 周：Reduction

目标：

- 理解 parallel reduction tree
- 理解 block 内同步
- 理解 row-wise sum/max 的并行方式

练习：

- `03_reduction`

验收问题：

- 一个 block 如何把多个 thread 的局部结果合并？
- reduction 为什么常常受 memory bandwidth 影响？

## 第 3 周：矩阵乘法

目标：

- 手写 naive matmul
- 理解 tiled matmul
- 理解 shared memory 如何减少 global memory 读取

练习：

- `04_matmul_naive`
- `05_matmul_tiled`

验收问题：

- naive matmul 每个 `C[i, j]` 需要读多少数据？
- tiled matmul 为什么更快？
- block size 如何影响 occupancy 和 shared memory 使用？

## 第 4 周：Transformer 基础算子

目标：

- 理解 softmax = max + exp + sum + divide
- 理解 layernorm = mean + variance + normalize
- 理解 attention = QK^T + softmax + V

练习：

- `06_softmax`
- `07_layernorm`
- `08_attention`

验收问题：

- softmax 为什么要先减 max？
- attention 中间矩阵为什么是 `seq_len x seq_len`？
- 哪些步骤是 compute-bound，哪些是 memory-bound？

## 第 5 周：Triton

目标：

- 用 Triton 写 vector add、matmul、softmax
- 理解 program id、block pointer、mask
- 连接 CUDA 思维和深度学习 kernel 写法

练习：

- `09_triton_matmul`

验收问题：

- Triton 的 program 和 CUDA block 有什么相似？
- 为什么 Triton 适合深度学习 kernel 原型开发？

## 第 6 周：LLM 推理性能

目标：

- 理解 prefill 和 decode 的性能差异
- 理解 KV cache 读写
- 理解量化为什么能降低显存和带宽压力
- 能读懂 FlashAttention 的核心动机

阅读：

- `notes/llm_inference_kernels.md`

验收问题：

- 为什么 prefill 更像 GEMM，decode 更像 GEMV？
- 为什么 batch size 影响 GPU 利用率？
- FlashAttention 为什么减少 HBM 读写？

