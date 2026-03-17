# P016 实现 KV Cache

## 背景

在 Transformer 自回归生成中，每生成一个新 token 都需要对**所有已生成的 token** 执行 attention。
如果不做缓存，生成第 t 个 token 时要重新计算前 t-1 步的 Key 和 Value，导致整体复杂度为 O(n^3 d)。

**KV Cache** 的核心思想：把每一步算出的 K_t, V_t 缓存起来，下一步直接拼接到历史缓存上。
这样每步只需计算新 token 的 K, V，总复杂度降为 O(n^2 d)。

```
Step 1: Q1 @ [K1]^T → attn → [V1] → output1          (cache = [K1, V1])
Step 2: Q2 @ [K1,K2]^T → attn → [V1,V2] → output2    (cache = [K1K2, V1V2])
Step 3: Q3 @ [K1,K2,K3]^T → attn → ...               (cache grows)
```

## 题目目标

实现一个简洁的 KV Cache 类，支持增量追加和完整读取。

## 接口规范

```python
class KVCache:
    def __init__(self): """初始化空缓存"""
    def append(self, key, value): """追加一步的 key/value"""
    def get(self): """返回 (keys, values)"""
    def reset(self): """清空缓存"""
```

## 示例

```python
cache = KVCache()
cache.append([1.0, 2.0], [10.0, 20.0])
cache.append([3.0, 4.0], [30.0, 40.0])
keys, values = cache.get()  # len(keys) == 2

cache.reset()
keys, values = cache.get()  # keys == [], values == []
```

## 约束 & 边界

| 条件 | 说明 |
|------|------|
| 空缓存 | `get()` 应返回两个空列表 |
| `reset()` 后 | 状态回到初始化 |
| 多次 append | 保持插入顺序 |
| key/value 类型 | 本题用 Python list 即可 |

## 进阶思考

- 真实模型中 KV Cache shape：`(batch, num_heads, seq_len, head_dim)`
- 生产环境用预分配 Tensor buffer + index pointer，避免反复 concat
- 显存估算：`2 * num_layers * seq_len * num_heads * head_dim * dtype_bytes`

## 练习建议

1. 先用 list 实现基本功能
2. 尝试用 `torch.cat` 改写（选做）
3. 口头说明"为什么 KV Cache 能把 O(n^3) 降到 O(n^2)"
