# 09 Triton Matmul

目标：用 Triton 写一个 matmul kernel。

需要理解：

- `@triton.jit`
- program id
- block pointer
- mask
- `tl.dot`

为什么学 Triton：

```text
它比 CUDA 更接近 Python，但仍然能表达高性能 GPU kernel。
很多深度学习自定义 kernel 会优先用 Triton 原型开发。
```

