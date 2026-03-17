# P007 实现 Causal Mask

## 背景

**Causal mask**（因果掩码）保证了自回归生成的核心性质：**位置 i 只能看到位置 0..i，不能看到未来的 token。**

```
seq_len = 4:
         k0    k1    k2    k3
q0  [  0,   -inf, -inf, -inf ]    ← q0 只能看 k0
q1  [  0,    0,   -inf, -inf ]    ← q1 能看 k0, k1
q2  [  0,    0,    0,   -inf ]    ← q2 能看 k0, k1, k2
q3  [  0,    0,    0,    0   ]    ← q3 能看所有
```

加到 attention scores 上后，-inf 位置经过 softmax 变为 0，实现了遮挡。

## 接口规范

```python
def causal_mask(seq_len: int, fill_value: float = float('-inf')) -> Tensor:
    """返回 (seq_len, seq_len) 的 additive causal mask
    下三角（含对角线）= 0，上三角 = fill_value（默认 -inf）"""
```

## 示例

```python
mask = causal_mask(3)
# tensor([[  0., -inf, -inf],
#         [  0.,   0., -inf],
#         [  0.,   0.,   0.]])
```

## 约束

| 条件 | 说明 |
|------|------|
| fill_value | 默认 -inf，上三角用此值填充 |
| 对角线 | 属于下三角，值为 0 |
| 实现方式 | `torch.triu(fill, diagonal=1)` |

## 练习建议

1. 用 torch.triu 一行实现
2. 验证 shape 和对角线
3. 口头说明为什么用加法 mask 而不是乘法 mask
