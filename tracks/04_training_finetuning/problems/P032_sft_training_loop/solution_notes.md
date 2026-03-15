# P032 SFT Training Loop 题解

## 核心思路

1. **zero_grad**：清空上一步残留的梯度
2. **forward**：模型前向传播得到 logits
3. **loss**：计算 cross_entropy loss
4. **backward**：反向传播计算梯度
5. **step**：优化器更新参数
6. 返回 loss 的标量值

## 面试口述模板

> "标准的 SFT training step 就是经典的 PyTorch 训练循环五步：
> zero_grad → forward → loss → backward → step。
> SFT 全称 Supervised Fine-Tuning，用带标注的 (instruction, response) 对
> 做 next-token prediction 微调。
> 和 pretraining 的区别是数据质量更高、通常只训练 response 部分的 loss。"

## 常见坑

- **zero_grad 位置**：必须在 forward 之前，否则梯度会累积
- **loss.item()**：返回 Python float，不是 tensor；记录用 item()，反传用 loss 本身
- **混合精度**：生产中通常配合 autocast + GradScaler
- **只算 response 部分的 loss**：SFT 中 instruction 部分通常用 ignore_index mask 掉

## 复杂度

- 取决于模型大小和 batch 大小
- 单步：前向 O(forward) + 反向 O(~2× forward) + 优化器 O(P)
