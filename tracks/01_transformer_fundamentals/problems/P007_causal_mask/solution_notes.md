# P007 Causal Mask 题解

## 核心思路

1. 生成一个 (seq_len, seq_len) 的矩阵
2. 下三角（含对角线）填 0，上三角填 -inf
3. 加到 attention scores 上后，softmax 将上三角位置的概率推向 0
4. 效果：位置 i 只能 attend 到位置 0..i（保证自回归性质）

## 面试口述模板

> "Causal mask 是保证自回归生成的关键机制。
> 它是一个上三角为 -inf、下三角为 0 的矩阵。
> 加到 attention scores 上后，softmax 让未来位置的权重为 0，
> 这样位置 i 的输出只依赖于位置 0 到 i 的输入，
> 保证了推理时的因果性——模型不能偷看未来的 token。"

## 常见坑

- **triu vs tril**：`torch.triu(diagonal=1)` 取上三角不含对角线，这才是要 mask 掉的部分
- **fill_value**：必须是 -inf 而非 0 或很大的负数（有些实现用 -1e9 也行，但不精确）
- **加法 vs 乘法**：additive mask 加到 scores 上，不是乘到 weights 上
- **batch 维度**：实际使用时 mask 需要 broadcast 到 (B, H, T, T)

## 复杂度

- 时间：O(n²) 生成
- 空间：O(n²)
- 可以预计算并缓存
