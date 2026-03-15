# P015 Padding & Attention Mask 题解

## 核心思路

1. 找到 batch 中最长序列的长度 max_len
2. 短序列在末尾补 pad_token_id 到 max_len
3. 生成 attention mask：真实 token 位置为 1，padding 位置为 0
4. mask 告诉模型哪些位置不参与 attention 计算

## 面试口述模板

> "Batch 推理时不同序列长度不一，需要 padding 对齐到最长。
> Padding 位置填特殊 token（通常 id=0），同时生成 attention mask——
> 1 表示真实 token，0 表示 padding。
> 在 attention 计算时，mask 为 0 的位置会被忽略（通常转成 -inf additive mask），
> 保证 padding 不影响输出。这在 batch inference 和 training 中都是必要步骤。"

## 常见坑

- **左 padding vs 右 padding**：decoder 模型通常左 padding（保证最后一个 token 对齐），encoder 通常右 padding
- **mask 维度**：attention mask 通常需要扩展到 (B, 1, 1, T) 或 (B, 1, T, T) 才能 broadcast
- **padding token id 选择**：要和模型 config 一致，不能随意选
- **忘记在 loss 计算中也用 mask**：padding 位置的 loss 不应该参与反向传播

## 复杂度

- 时间：O(B × L)，B 为 batch size，L 为 max_len
- 空间：O(B × L) 存 padded batch + mask
