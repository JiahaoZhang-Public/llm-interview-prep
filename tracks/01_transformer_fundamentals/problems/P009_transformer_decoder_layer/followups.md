# 面试追问

## 基础理解
- Decoder layer 和 Encoder block (P004) 在代码上有什么区别？
- QKV projection 是否应该融合成一个矩阵？好处和坏处？
- decoder-only 和 encoder-decoder 的 decoder 有什么不同？

## KV Cache 集成
- 推理时 KV Cache 应该接在哪里？forward 的签名要怎么改？
- 有了 KV Cache 后，每步 Q 的 shape 是什么？K/V 的 shape 呢？
- KV Cache 如何和 causal mask 配合？

## 架构选择
- Pre-LN 为什么比 Post-LN 训练更稳定？
- 现代 LLM (LLaMA) 的 decoder layer 和你的实现有什么区别？（RoPE, RMSNorm, SwiGLU）
- 如何在 decoder layer 中加入 cross-attention 子层？
