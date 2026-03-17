# P024 Prefix Caching 题解

## 核心思路

1. dict 存 prefix → value 映射
2. lookup 遍历找最长前缀匹配
3. 无匹配返回 None

## 面试口述模板

> "Prefix caching 利用多个请求共享相同前缀（如 system prompt）的特点。
> 把前缀的 KV cache 缓存起来，后续请求 lookup 最长匹配，命中后跳过对应 prefill。
> 我的实现用 dict + 暴力遍历找最长匹配，O(N*L)。Trie 可优化到 O(L)。
> 实际系统（SGLang RadixAttention）还需 LRU 淘汰、block 级粒度、跨 GPU 缓存共享。"

## 常见坑

- **最长匹配 vs 精确匹配**
- **查询比缓存短**：[1,2,3] 缓存，查 [1,2] → 不匹配
- **缓存失效**：system prompt 变化
- **内存管理**：需要 LRU 淘汰
- **cache stampede**：多个请求同时 miss 同一前缀

## 复杂度

- add：O(L)
- lookup 暴力：O(N*L)
- lookup Trie：O(L)
