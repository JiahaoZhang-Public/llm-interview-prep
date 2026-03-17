# 面试追问

## 基础理解
- 从数学上证明：为什么 T < 1 让分布更尖锐？
- T = 0 时怎么处理？
- T = 1 时和普通 softmax sampling 等价吗？

## 与其他策略组合
- Temperature + Top-K：先 scale 还是先 top-k？
- Temperature + Top-P：低 T 后 nucleus 怎么变？
- ChatGPT API 中 temperature 和 top_p 同时设置时的行为？

## 工程与实践
- 代码生成推荐什么 temperature？创意写作呢？
- 什么是 temperature annealing？
- batch 推理中如何给不同请求设不同 temperature？
