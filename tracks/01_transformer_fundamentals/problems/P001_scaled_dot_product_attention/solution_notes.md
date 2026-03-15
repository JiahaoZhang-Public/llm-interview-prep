# P001 Scaled Dot-Product Attention 题解

## 核心思路

1. 计算 Q·K^T 得到 attention scores
2. 除以 √d_k 做 scaling，防止 softmax 进入梯度消失区
3. 如果有 mask，以加法方式应用（-inf 位置 softmax 后趋近 0）
4. softmax 归一化后乘 V 得到输出

关键公式：`Attention(Q,K,V) = softmax(QK^T / √d_k) · V`

## 面试口述模板

> "Scaled dot-product attention 首先用 Q 和 K 的转置做矩阵乘得到 raw scores，
> 然后除以 √d_k 来 scale——这是因为当 d_k 较大时，点积的方差会线性增长，
> 导致 softmax 输出接近 one-hot，梯度几乎为零。
> Mask 以 additive 方式加到 scores 上（用 -inf），softmax 之后自然变成 0。
> 最后用 weights 乘 V 做加权求和。"

## 常见坑

- **忘记 scale**：不除 √d_k，大维度下 softmax 饱和、训练不稳
- **mask 用乘法而非加法**：乘 0 后 softmax 前的值是 0 而不是 -inf，还会分到概率
- **transpose 维度搞错**：K 要 transpose 最后两个维度 (-2, -1)，不是整个 tensor
- **softmax 轴写错**：必须在最后一维（seq_len 的 key 维度）做 softmax

## 复杂度

- 时间：O(n² · d)，n 为序列长度，d 为 head 维度
- 空间：O(n²) 存 attention weights 矩阵
- 这也是为什么长序列需要 FlashAttention 等 IO-aware 优化
