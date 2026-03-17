# P024 实现 Prefix Caching

## 背景

大量请求共享相同 system prompt。**Prefix caching** 缓存共享前缀的 KV cache，命中时跳过 prefill，降低 TTFT。

查找策略：**最长前缀匹配**。

## 接口规范

```python
class PrefixCache:
    def __init__(self): ...
    def add_prefix(self, prefix_tokens, value): """存入前缀"""
    def lookup(self, prefix_tokens): """最长前缀匹配，返回 value 或 None"""
```

## 约束

| 条件 | 说明 |
|------|------|
| 最长匹配 | 多个前缀匹配时返回最长的 |
| 无匹配 | 返回 None |

## 进阶思考

- 暴力 O(N*L)，Trie 优化到 O(L)
- 需要 LRU 淘汰控制显存
- SGLang 的 RadixAttention

## 练习建议

1. 用 dict + 暴力遍历实现
2. 思考 Trie 优化
3. 口头说明多轮对话场景的收益
