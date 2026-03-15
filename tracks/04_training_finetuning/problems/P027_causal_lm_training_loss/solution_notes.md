# P027 Causal LM Training Loss 题解

## 核心思路

1. 将 logits 和 labels 错位：logits[:, :-1] 预测 labels[:, 1:]
2. 即位置 i 的 logits 预测位置 i+1 的 token
3. reshape 为 2D 后用标准 cross_entropy
4. ignore_index=-100 跳过 padding 位置

## 面试口述模板

> "Causal LM 的训练 loss 是 next-token prediction 的 cross entropy。
> 关键是 shift：logits 的第 i 个位置预测第 i+1 个 token，
> 所以 logits 取 [:,:-1]，labels 取 [:,1:]，然后 flatten 成 2D 做 cross_entropy。
> padding 位置用 ignore_index=-100 标记，不参与 loss 计算。
> 这就是 GPT 系列模型的标准训练目标。"

## 常见坑

- **忘记 shift**：不 shift 的话每个位置在预测自己，不是 next token
- **shift 方向搞反**：logits 去掉最后一步，labels 去掉第一步
- **ignore_index 不传**：padding token 会算入 loss，影响训练
- **contiguous()**：slice 后 tensor 可能不连续，view 前要 contiguous

## 复杂度

- 时间：O(B × T × V)，B = batch，T = seq_len，V = vocab
