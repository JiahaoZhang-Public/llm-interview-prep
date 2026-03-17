# 面试追问

## 基础理解
- 为什么中间层通常比 hidden_size 更宽（4x）？这有什么理论依据？
- "Position-wise" 是什么意思？它和全局 FFN 的区别？
- FFN 本质上等价于什么操作？（提示：1x1 卷积）

## 激活函数
- ReLU → GELU → SwiGLU 的演变路线？每步改进了什么？
- SwiGLU 的公式和参数量？为什么 SwiGLU 的 intermediate_size 通常用 8/3 × hidden？
- GELU 的近似公式？为什么它比 ReLU 更适合 Transformer？

## 工程问题
- FFN 占模型总参数的比例大约是多少？
- 如何在 FFN 层做 tensor parallelism？
- Mixture of Experts 中 FFN 如何被替换？
