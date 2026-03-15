# P026 Cross Entropy Loss 题解

## 核心思路

1. 对 logits 做 softmax 得到概率分布
2. 取 target class 的概率，求负对数：-log(p[target])
3. 对 batch 求平均
4. PyTorch 的 F.cross_entropy 直接接受 raw logits（内部做 log_softmax + nll_loss）

## 面试口述模板

> "Cross entropy loss 衡量预测分布和真实分布的差异。
> 对于分类任务，真实分布是 one-hot，所以简化为 -log(p[target])。
> PyTorch 的 F.cross_entropy 直接接受 logits（不需要先 softmax），
> 内部用 log_softmax + NLL loss 实现，数值更稳定。"

## 常见坑

- **先 softmax 再传入**：F.cross_entropy 接受 raw logits，先 softmax 会算错
- **logits shape**：(B, C) 不是 (B, C, 1)
- **数值稳定性**：手写时要用 log_softmax trick 而非先 softmax 再 log

## 复杂度

- 时间：O(B × C)，B = batch，C = 类别数
