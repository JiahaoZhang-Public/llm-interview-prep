# P006 实现 RoPE Position Embedding

## 背景

**RoPE（Rotary Position Embedding）** 是目前 LLM 最常用的位置编码方式（LLaMA, Mistral, Qwen 等均使用）。

**核心思想：** 把向量的每两个相邻维度视为一个 2D 平面，根据位置和频率做旋转。

```
对维度对 (d_{2i}, d_{2i+1})：
频率: θ_i = 1 / base^{2i/d}
角度: angle = position × θ_i

旋转:
[d_{2i}  ]     [cos(angle)  -sin(angle)] [d_{2i}  ]
[d_{2i+1}]  =  [sin(angle)   cos(angle)] [d_{2i+1}]
```

**关键优势：** Q 和 K 旋转后的点积只依赖于**相对位置差** (m-n)，天然编码了相对位置信息。

## 接口规范

```python
def apply_rope(
    q: Tensor,              # (*, seq_len, dim)
    k: Tensor,              # (*, seq_len, dim)
    position_ids: Tensor = None,  # (seq_len,) 可选
    base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """对 Q 和 K 应用 RoPE，返回旋转后的 (q, k)"""
```

## 约束

| 条件 | 说明 |
|------|------|
| 维度配对 | x[..., 0::2] 和 x[..., 1::2] |
| 只对 Q 和 K | V 不需要 RoPE |
| shape 不变 | 输出 shape == 输入 shape |
| base | 默认 10000，可调整做长度外推 |

## 练习建议

1. 分步实现：计算频率 → 计算角度 → cos/sin → 旋转
2. 验证 shape 不变
3. 口头说明 RoPE 如何编码相对位置
