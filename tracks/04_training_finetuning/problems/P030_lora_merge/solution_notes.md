# P030 LoRA Merge 题解

## 核心思路

1. LoRA 训练后得到 A (rank × in) 和 B (out × rank)
2. 合并公式：W_merged = W_base + B @ A × (alpha / rank)
3. 合并后模型等价于原始模型 + LoRA，但推理时无额外计算

## 面试口述模板

> "LoRA merge 就是把训练好的低秩矩阵合并回原始权重。
> W_merged = W_base + B × A × scaling，其中 scaling = alpha / rank。
> 合并后模型变回一个普通的 dense 模型，推理没有任何额外开销。
> 这是 LoRA 相比 Adapter 等方法的一大优势——
> Adapter 在推理时还有额外的前向计算，LoRA 可以完全消除。"

## 常见坑

- **scaling 忘了**：必须乘 alpha/rank，否则 LoRA 的贡献大小不对
- **矩阵乘法顺序**：B @ A 是 (out × rank) @ (rank × in) = (out × in)，和 W 同 shape
- **合并后不可逆**：一旦合并就分不出 base 和 LoRA 了，要保留原始权重的备份
- **多个 LoRA 合并**：多个 adapter 可以按比例加权合并（model merging）

## 复杂度

- 合并计算：O(out × rank × in)
- 合并后推理：与原模型完全相同，零额外开销
