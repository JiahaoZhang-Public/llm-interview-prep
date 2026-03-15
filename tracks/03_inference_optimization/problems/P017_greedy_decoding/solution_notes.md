# P017 Greedy Decoding 题解

## 核心思路

1. 对 logits 取 argmax，返回概率最高的 token id
2. 最简单的解码策略，完全确定性

## 面试口述模板

> "Greedy decoding 就是每步选 logits 最大的 token，即 argmax。
> 优点是简单快速且确定性；缺点是容易陷入局部最优——
> 因为当前最优 token 组成的序列不一定是全局最优的。
> 这就是为什么有 beam search（保留多条路径）和 sampling（引入随机性）。"

## 常见坑

- **和 argmax 后 softmax 混淆**：greedy 只需要 argmax，不需要算概率
- **无多样性**：相同输入永远产生相同输出，不适合创意生成
- **重复退化**：greedy 容易反复生成同一个词或短语

## 复杂度

- 时间：O(V)，V 为 vocab size（一次遍历）
- 空间：O(1)
