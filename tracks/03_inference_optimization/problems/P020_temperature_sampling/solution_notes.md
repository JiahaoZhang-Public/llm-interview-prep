# P020 Temperature Sampling 题解

## 核心思路

1. 将 logits 除以 temperature：`scaled_logits = logits / T`
2. 对缩放后的 logits 做 softmax 得到概率分布
3. 按概率随机采样
4. T < 1 让分布更尖锐，T > 1 更平坦，T = 1 不变

**数学直觉**：T 缩小了 logits 之间的差异。T < 1 → 差异放大 → 更尖锐。

## 面试口述模板

> "Temperature 是控制 LLM 生成随机性最基础的旋钮。
> 把 logits 除以 T 后做 softmax：T < 1 放大差异，输出更确定；
> T > 1 缩小差异，输出更随机。T → 0 退化为 greedy，T → ∞ 均匀分布。
> 实际使用 0-0.2 用于代码生成，0.7-1.0 用于创意写作。
> 注意：T=0 要特殊处理为 argmax，低温度下要减 max 防溢出。"

## 常见坑

- **temperature = 0 除零**：需特殊处理为 greedy
- **先 scale 再 softmax**：不是先 softmax 再 scale
- **数值稳定性**：低温度放大 logits，exp 可能溢出 → 减 max
- **和 top-k/top-p 组合顺序**：先 temperature scale，再 top-k/top-p

## 复杂度

- 时间：O(V)
- 空间：O(V)
