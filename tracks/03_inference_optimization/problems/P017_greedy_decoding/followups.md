# 面试追问

## 基础理解
- Greedy decoding 为什么是确定性的？什么场景是优势/劣势？
- 为什么不需要先做 softmax？argmax 在 softmax 前后结果一样吗？
- 哪些任务适合 greedy？哪些容易失败？

## 对比分析
- Greedy vs Beam Search：beam search 如何避免局部最优？
- Greedy vs Sampling：什么场景下 sampling 更好？
- beam_size=1 和 greedy 完全等价吗？

## 工程问题
- 如何在 EOS 处正确停止？
- 如何实现 batch 版本？
- 重复退化是什么？greedy 为什么特别容易出现？
