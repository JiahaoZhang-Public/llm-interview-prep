# 面试追问

## 基础理解
- 为什么必须从概率最高的开始累加？
- p=0.9 和 p=0.95 差别大吗？为什么 OpenAI 默认 p=1？
- 均匀分布时 nucleus 大小是多少？

## 对比分析
- Top-P 解决了 Top-K 的什么问题？用具体场景举例。
- Top-K + Top-P 组合的执行顺序？
- Top-P + 低 Temperature 时 nucleus 只有 1 个 token，等价于 greedy 吗？

## 工程与论文
- Nucleus sampling 论文的核心发现？
- HuggingFace `generate()` 中 `top_p` 和 `do_sample` 的关系？
- GPU 上如何高效实现？（cumsum → 二分查找）
