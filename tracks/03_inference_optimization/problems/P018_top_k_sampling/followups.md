# 面试追问

## 基础理解
- 先 softmax 再取 top-k 和先取 top-k 再 softmax 结果一样吗？
- K=1 和 greedy 完全等价吗？
- "减去 max" 的技巧为什么不改变结果？

## 对比分析
- Top-K 固定 K 值有什么问题？用具体例子说明。
- Top-K 和 Top-P 可以同时使用吗？先做哪个？
- Top-K + Temperature 如何组合？

## 工程与优化
- 排序 O(V log V) 如何用 partial sort 优化到 O(V + K log K)？
- GPU 上如何高效实现 top-k？（`torch.topk`）
- logits 中有 -inf 或 nan 怎么处理？
