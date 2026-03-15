# P048 LLM Cache System 题解

## 核心思路

1. 对 prompt 做哈希（如 SHA-256）作为 cache key，缓存 LLM 响应
2. 数据结构：dict 映射 hash(prompt + params) → response
3. 查询时先检查缓存命中，命中则直接返回，未命中则调用 LLM 并写入缓存
4. 使用 LRU 淘汰策略控制缓存大小（OrderedDict 或 functools.lru_cache）
5. cache key 应包含影响输出的所有参数：prompt、model、temperature、max_tokens 等

## 面试口述模板

> "LLM Cache 的思路很直观：把 prompt 和生成参数拼接后做 SHA-256 哈希作为 key，value 是 LLM 的响应。用 OrderedDict 实现 LRU——每次命中把 key 移到末尾，容量满时弹出头部最久未用的条目。查询流程是 get-or-compute：先查缓存，命中返回；未命中调 LLM，写入缓存。注意 temperature > 0 时 LLM 输出不确定，可以选择不缓存或接受缓存旧结果。"

## 常见坑

- **temperature > 0 的缓存**：非零温度输出有随机性，是否缓存取决于业务需求
- **hash 碰撞**：SHA-256 碰撞概率极低但理论存在，生产环境可存原始 key 做二次校验
- **参数遗漏**：cache key 必须包含所有影响输出的参数，漏掉 model 版本会导致脏缓存
- **并发写入**：多线程同时 miss 可能重复调用 LLM，可用锁或 singleflight 模式

## 复杂度

- 时间：缓存命中 O(1)，未命中 O(C_llm)，淘汰 O(1)
- 空间：O(N · R)，N 为缓存条目数，R 为平均响应大小
