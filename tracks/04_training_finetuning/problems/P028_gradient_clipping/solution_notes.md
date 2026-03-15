# P028 Gradient Clipping 题解

## 核心思路

1. 计算所有参数梯度的全局 L2 norm
2. 如果超过 max_norm，按比例缩放所有梯度
3. 返回裁剪前的原始 norm
4. PyTorch 提供 `clip_grad_norm_` 一步完成

## 面试口述模板

> "Gradient clipping 是防止梯度爆炸的标准技术。
> 计算所有参数梯度的全局 L2 norm，如果超过阈值就按比例缩放，
> 使得 norm 恰好等于 max_norm。这保持了梯度方向不变，只限制大小。
> 在 Transformer 训练中几乎是标配，通常 max_norm 设为 1.0。"

## 常见坑

- **局部 vs 全局 norm**：要先算所有参数的总 norm，再统一缩放；不是每个参数独立 clip
- **clip 时机**：在 backward 之后、step 之前
- **返回值**：返回的是 clip 之前的 norm，用于监控训练状态
- **和 gradient scaling 的关系**：混合精度训练中先 unscale 再 clip

## 复杂度

- 时间：O(P)，P = 总参数量（遍历一次计算 norm，一次缩放）
