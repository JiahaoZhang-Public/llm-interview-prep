# P024 Prefix Caching 题解

## 核心思路

1. 维护一个 prefix → value 的映射（key 是 token tuple）
2. add_prefix 存入缓存
3. lookup 查找输入 token 序列的最长前缀匹配
4. 如果多个 prefix 都匹配，返回最长那个的 value

## 面试口述模板

> "Prefix caching 利用了多个请求可能共享相同前缀（如系统 prompt）的特点。
> 把前缀的 KV cache 缓存起来，后续请求如果有相同前缀就直接复用，
> 避免重复计算 prefill。实现上用 prefix token 序列作为 key，
> lookup 时找最长匹配的已缓存前缀。
> 这在 system prompt 固定、多轮对话等场景下能显著降低首 token 延迟。"

## 常见坑

- **最长匹配 vs 精确匹配**：要找最长前缀，不是完全一致才命中
- **缓存失效**：system prompt 变化时要清理对应缓存
- **内存管理**：缓存太多前缀会占大量显存，需要 LRU 淘汰
- **Trie 优化**：用 Trie 结构可以 O(L) 查找最长前缀，比暴力遍历快

## 复杂度

- add：O(1)（hash table 存储）
- lookup：O(N × L) 暴力 / O(L) Trie，N 为缓存条目数，L 为查询长度
