# P001 实现 Scaled Dot-Product Attention

## 背景

**Scaled dot-product attention** 是 Transformer 的核心运算，出自 "Attention Is All You Need" (Vaswani et al., 2017)。

公式：`Attention(Q, K, V) = softmax(Q K^T / √d_k) · V`

```
Q: (*, seq_q, d_k)     — Query
K: (*, seq_k, d_k)     — Key
V: (*, seq_k, d_v)     — Value

scores = Q @ K^T     → (*, seq_q, seq_k)     # 原始相似度
scores = scores / √d_k                        # 缩放，防止 softmax 饱和
if mask: scores += mask                        # additive mask（-inf 位置 softmax 后→0）
weights = softmax(scores, dim=-1)              # 归一化为概率
output = weights @ V  → (*, seq_q, d_v)       # 加权求和
```

**为什么要除以 √d_k？** 当 d_k 较大时，Q·K^T 点积的方差约为 d_k，值很大时 softmax 输出接近 one-hot，梯度几乎为零。除以 √d_k 将方差稳定在 1。

**Mask 的两种风格：**
- **Additive mask**：在 scores 上加 0 / -inf（本题使用）
- **Boolean mask**：True = 被屏蔽位置

## 题目目标

实现 scaled dot-product attention 函数。

## 接口规范

```python
def scaled_dot_product_attention(
    q: Tensor,          # (*, seq_q, d_k)
    k: Tensor,          # (*, seq_k, d_k)
    v: Tensor,          # (*, seq_k, d_v)
    mask: Tensor = None # (*, seq_q, seq_k) additive mask, 0 或 -inf
) -> Tensor:            # (*, seq_q, d_v)
```

## 示例

```python
q = torch.tensor([[[1.0, 0.0]]])           # (1, 1, 2)
k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # (1, 2, 2)
v = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]) # (1, 2, 2)
mask = torch.tensor([[0.0, float("-inf")]])

output = scaled_dot_product_attention(q, k, v, mask=mask)
# output ≈ [[2.0, 0.0]]  （mask 屏蔽了第 2 个 key，全部 attend 到第 1 个）
```

## 约束

| 条件 | 说明 |
|------|------|
| mask = None | 不做 mask |
| mask 类型 | additive（0 / -inf），不是 boolean |
| d_k | 从 q.size(-1) 获取 |
| softmax 维度 | dim=-1（对 key 维度做归一化） |
| 输出 shape | (*, seq_q, d_v) |

## 练习建议

1. 按公式分步实现：matmul → scale → mask → softmax → matmul
2. 用上面的示例验证结果
3. 口头说明"为什么要除以 √d_k"
