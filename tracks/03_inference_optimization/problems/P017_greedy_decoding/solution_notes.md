# P017 Greedy Decoding 题解

## 核心思路

1. 对 logits 取 argmax，返回概率最高的 token id
2. 最简单的解码策略，完全确定性，一次遍历即可
3. **不需要 softmax**：softmax 是单调递增变换，不改变 argmax 的结果

## 面试口述模板

> "Greedy decoding 每步选 logits 最大的 token，就是 argmax。
> 不需要算 softmax，因为 softmax 是单调变换，不会改变最大值的位置。
> 优点是简单快速且确定性；缺点是每步贪心不等于全局最优——
> 当前步选的最好的词，可能让后续步的选择变差。
> 另一个问题是 repetition degeneration：greedy 容易反复生成同一个短语。
> 这就是为什么实际系统会用 beam search 或 sampling。"

## 常见坑

- **和 argmax 后 softmax 混淆**：greedy 只需要 argmax，不需要算概率
- **无多样性**：相同输入永远产生相同输出
- **重复退化**：极易产生 "I think that I think that..." 这样的退化
- **并列处理**：多个 logit 相同时返回第一个，面试要提
- **batch 支持**：实际用 `torch.argmax(logits, dim=-1)`

## 复杂度

- 时间：O(V)，V 为 vocab size
- 空间：O(1)
