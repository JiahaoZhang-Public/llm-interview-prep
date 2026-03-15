# P022 Logits Processor (Repetition Penalty) 题解

## 核心思路

1. 遍历已生成的 token id 集合
2. 对出现过的 token 的 logits 施加惩罚：正值除以 penalty，负值乘以 penalty
3. 这使得重复 token 的概率降低，减少退化重复

## 面试口述模板

> "Repetition penalty 是一种 logits processor，用于减少生成文本中的重复。
> 对已经出现过的 token，如果其 logit > 0 就除以 penalty，< 0 就乘以 penalty，
> 效果都是降低该 token 被再次选中的概率。
> penalty=1.0 无效果，>1.0 抑制重复，<1.0 反而鼓励重复。
> 这是 Hugging Face generate() 中常用的参数之一。"

## 常见坑

- **正负 logit 的处理不同**：正值要除，负值要乘，统一操作方向会适得其反
- **penalty 范围**：通常 1.0-1.5 之间，太大会让输出跑偏
- **和 frequency penalty 的区别**：repetition penalty 是二元的（出现过/没出现），frequency penalty 按出现次数累加

## 复杂度

- 时间：O(|generated_ids|)
- 空间：O(V) 存调整后的 logits
