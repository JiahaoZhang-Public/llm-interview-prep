# P049 Embedding Cache 题解

## 核心思路

1. 缓存文本到 embedding 的映射，避免重复调用 embedding API
2. cache key = hash(text)，value = embedding vector
3. 批量请求时先过滤已缓存的文本，只对未命中的调用 embedding API
4. 合并缓存结果和新计算结果，保持原始顺序返回
5. 使用 LRU 或 TTL 策略控制缓存大小，embedding 向量占用内存较大

## 面试口述模板

> "Embedding Cache 的核心价值是节省 API 调用成本和延迟。实现上我用 dict 存 text_hash → embedding。批量编码时先遍历输入文本列表，把命中缓存的和未命中的分开。未命中的批量发给 embedding API，拿到结果后写入缓存。最后按原始顺序拼装所有结果返回。关键优化是批量过滤——如果 100 个文本有 80 个已缓存，只需对 20 个调 API，节省 80% 开销。"

## 常见坑

- **文本归一化**：相同语义但不同空白/大小写的文本会产生不同 hash，需先做 normalize
- **embedding 模型版本**：模型升级后旧缓存失效，cache key 应包含 model_name
- **内存压力**：每个 embedding 1536 维 float32 占 6KB，百万条约 6GB，需设合理上限
- **批量接口顺序**：API 返回顺序必须和请求顺序一致，否则映射错乱

## 复杂度

- 时间：O(n) 查缓存 + O(m · d) 调 API（m 为未命中数），均摊远优于全量调用
- 空间：O(N · d)，N 为缓存条目数，d 为向量维度
