# 面试追问

## 基础理解
- 为什么正 logit 除而负 logit 乘？统一操作会怎样？
- penalty = 1.0 时效果？< 1.0 呢？
- logit = 0 施加 penalty 后会怎样？

## 概念辨析
- Repetition penalty vs Frequency penalty vs Presence penalty 有什么区别？
- 三种 penalty 可以组合吗？

## 工程实现
- 多个 logits processor 的执行顺序？
- HuggingFace `LogitsProcessorList` 如何自定义？
- EOS token 要不要被惩罚？
- batch 推理中如何对每个请求应用不同 penalty？
