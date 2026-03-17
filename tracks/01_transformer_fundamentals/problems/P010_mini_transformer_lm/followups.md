# 面试追问

## 基础理解
- 如何做 embedding 和 LM head 的权重共享（weight tying）？好处和坏处？
- 如果加入 RoPE，需要改哪些代码？position embedding 还需要吗？
- 在此之上如何计算 causal LM loss？logits 和 labels 如何对齐？

## 训练相关
- 如何估算这个模型的参数量？给出公式。
- 训练一个 1B 参数的模型大约需要多少 FLOPs？（Chinchilla scaling law）
- 如何做 mixed precision training？哪些层需要保持 fp32？

## 架构演变
- 你的实现和 GPT-2 的区别？和 LLaMA 的区别？
- 如何扩展到多层（N 个 block）？
- 如何加入 cross-attention 变成 encoder-decoder 模型？
