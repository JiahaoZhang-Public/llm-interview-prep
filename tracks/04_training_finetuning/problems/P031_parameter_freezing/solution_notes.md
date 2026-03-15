# P031 Parameter Freezing 题解

## 核心思路

1. 先把所有参数的 requires_grad 设为 False（全部冻结）
2. 遍历 named_parameters，如果名字匹配 trainable_patterns 中的任何模式，设为 True
3. 返回可训练参数的名字列表

## 面试口述模板

> "Parameter freezing 是微调时减少计算量和防止灾难性遗忘的基本手段。
> 先冻结所有参数，再只解冻匹配特定模式的参数。
> 比如 LoRA 微调时只训练 lora_A 和 lora_B，
> 或者 linear probing 时只训练最后的分类头。
> 冻结的参数不计算梯度，显著减少显存占用和计算量。"

## 常见坑

- **冻结顺序**：先全冻结再选择性解冻，不是只冻结不匹配的
- **BN/LN 参数**：有时候需要特别处理 normalization 层的参数
- **optimizer 要跟 requires_grad 同步**：只把可训练参数传给 optimizer
- **模式匹配粒度**：用 `in` 做子串匹配还是 regex，要和调用方约定清楚

## 复杂度

- 时间：O(N × P)，N = pattern 数，P = 参数数
