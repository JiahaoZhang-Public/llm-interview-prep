# P018 Top-K Sampling 题解

## 核心思路

1. 把 logits 按值从大到小排序，只保留前 k 个
2. 对这 k 个 logits 做 softmax 得到概率分布
3. 按概率随机采样一个 token

Top-K 限制了候选集大小，避免低概率 token 被采到产生垃圾输出。

## 面试口述模板

> "Top-K sampling 就是在采样前先把 logits 排序，只保留概率最高的 K 个候选 token，
> 把其余的概率设为零，然后对这 K 个重新归一化做随机采样。
> 好处是避免长尾低概率 token 产生不合理的输出。
> 缺点是 K 是固定的——有时概率分布很集中只需要 2-3 个候选，
> 有时很平坦需要更多候选。这就是 Top-P (nucleus) sampling 要解决的问题。"

## 常见坑

- **忘记对 top-k 后的 logits 重新归一化**：直接用原始 logits 做 softmax 没问题，但采样范围一定要限制在 top-k 内
- **k=1 退化为 greedy**：面试时提一句展示理解
- **logits vs probabilities**：先 softmax 再取 top-k 和先取 top-k 再 softmax 结果不同，标准做法是先取 top-k 再 softmax
- **数值稳定性**：softmax 前减去 max logit 防止 exp 溢出

## 复杂度

- 时间：O(V log V) 排序，V 为 vocab size；或用 partial sort O(V + K log K)
- 空间：O(V)
