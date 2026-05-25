# cuda-learning

用来系统学习 CUDA 编程、GPU 矩阵计算和大模型推理内核原理的练习仓库。

学习目标不是只会写几个 kernel，而是逐步建立下面这条理解链：

```text
GPU 硬件执行模型
-> CUDA thread/block/grid
-> memory hierarchy
-> matrix multiply
-> reduction / softmax / layernorm
-> attention
-> FlashAttention / Triton
-> LLM prefill/decode 性能分析
```

## 学习路线

1. GPU 执行模型和 CUDA 基础
2. 基础 kernel：vector add、matrix add、transpose
3. reduction：sum、max、row-wise reduction
4. GEMM：naive matmul、tiled matmul、shared memory
5. Transformer 算子：softmax、layernorm、attention
6. Triton：用 Python 风格写 GPU kernel
7. LLM 推理性能：prefill、decode、KV cache、量化

## 目录结构

```text
cuda-learning/
|- README.md
|- ROADMAP.md
|- notes/
|  |- gpu_execution_model.md
|  |- matmul_principles.md
|  |- llm_inference_kernels.md
|- exercises/
|  |- 01_vector_add/
|  |- 02_matrix_add/
|  |- 03_reduction/
|  |- 04_matmul_naive/
|  |- 05_matmul_tiled/
|  |- 06_softmax/
|  |- 07_layernorm/
|  |- 08_attention/
|  |- 09_triton_matmul/
```

## 推荐节奏

每个练习都按同一套问题复盘：

```text
每个 thread 算什么？
thread block 如何覆盖数据？
读写了哪些 memory？
访存是否 coalesced？
瓶颈是算力还是访存？
和 PyTorch/cuBLAS 相比慢在哪里？
```

## 环境建议

优先使用 Linux + NVIDIA GPU：

```bash
nvcc --version
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

如果当前机器没有 NVIDIA GPU，也可以先读 notes 和写 CPU/PyTorch 对照，之后到有 CUDA 的机器上跑 kernel。

