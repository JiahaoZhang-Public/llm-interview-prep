# P016 KV Cache 题解

## 核心思路

1. 自回归生成时，每步只新增一个 token 的 Q，但需要和所有历史 token 的 K、V 做 attention
2. KV Cache 把之前步骤算过的 K、V 缓存下来，每步只计算新 token 的 K、V 并 append
3. 这样每步的 attention 计算从 O(n²d) 降到 O(nd)，总生成从 O(n³d) 降到 O(n²d)

## 面试口述模板

> "KV Cache 是自回归推理的核心优化。
> 没有 cache 的话，生成第 t 个 token 需要对前 t 个位置重新算一遍 K 和 V，总复杂度是 O(n³)。
> 有了 KV Cache，我们把每步新算的 key 和 value 向量 append 到缓存里，
> 下一步只需要算新 token 的 q、k、v，然后让 q 和整个缓存的 K 做 attention。
> 这样每步计算量从 O(n²) 降到 O(n)，总推理从 O(n³) 降到 O(n²)。
> 代价是额外的显存占用：每层要存 2 × n × d 的 float。"

## 常见坑

- **reset 时机**：每个新 prompt 要清空 cache，否则前一个 prompt 的上下文会泄露
- **多层 cache**：实际模型每个 transformer 层都有独立的 KV Cache
- **显存估算**：KV Cache 大小 = 2 × num_layers × seq_len × hidden_size × dtype_bytes
  - 例如 70B 模型 80 层 8192 seq_len fp16：约 20GB
- **Tensor 拼接 vs 预分配**：反复 concat 很慢，生产中预分配固定长度 buffer 用 index 写入

## 复杂度

- append：O(1)（均摊）
- get：O(1)
- 显存：O(n × d) per layer，n 为已生成长度
