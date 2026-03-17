# P022 Logits Processor (Repetition Penalty) 题解

## 核心思路

1. 遍历已生成 token id 的**去重集合**
2. 正值 logit 除以 penalty，负值 logit 乘以 penalty
3. 效果一致：降低重复 token 被选中的概率
4. 未出现过的 token logits 不变

## 面试口述模板

> "Repetition penalty 在 softmax 前修改 logits 来减少重复。
> 正 logit 除以 penalty，负 logit 乘以 penalty，都是降低概率。
> penalty=1.0 无效果，>1.0 抑制重复，通常用 1.1-1.5。
> 注意和 frequency penalty 区别：repetition penalty 是二元的（出现过/没出现），
> frequency penalty 按出现次数线性惩罚。
> 正负 logit 处理不同是最容易出错的点。"

## 常见坑

- **正负 logit 处理不同**：统一操作会适得其反
- **去重处理**：`set(generated_ids)` 因为只看是否出现过
- **EOS 处理**：EOS 被过度惩罚会导致无法停止
- **不修改原列表**：返回新列表

## 复杂度

- 时间：O(|unique_generated_ids|)
- 空间：O(V)
