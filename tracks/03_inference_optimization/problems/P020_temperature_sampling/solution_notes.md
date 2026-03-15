# P020 Temperature Sampling 题解

## 核心思路

1. 将 logits 除以 temperature 参数
2. 对缩放后的 logits 做 softmax 得到概率
3. 按概率随机采样
4. temperature < 1 让分布更尖锐（趋向 greedy），> 1 让分布更平坦（更随机）

## 面试口述模板

> "Temperature sampling 通过缩放 logits 来控制生成的随机性。
> 把 logits 除以 temperature 后再做 softmax：
> T < 1 时，高概率 token 变得更高、低概率更低，输出更确定；
> T > 1 时，分布变平坦，输出更多样；T → 0 退化为 greedy。
> 这是最基础的采样超参数，通常和 top-k/top-p 配合使用。"

## 常见坑

- **temperature = 0 除零**：需要特殊处理为 greedy
- **先 scale 再 softmax**：不是先 softmax 再 scale
- **数值稳定性**：低温度时 logits 放大可能导致 overflow，要减 max

## 复杂度

- 时间：O(V)
- 空间：O(V)
