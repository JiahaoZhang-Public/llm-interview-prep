# P035 Embedding Vector Store 题解

## 核心思路

1. 维护一个列表存储 (id, text, embedding_vector) 三元组
2. `add(doc_id, text, embedding)` 将文档向量插入存储
3. `search(query_embedding, top_k)` 遍历所有向量，计算 query 与每条记录的余弦相似度，取 top-k
4. 余弦相似度 = dot(a, b) / (||a|| * ||b||)，用 numpy 向量化可大幅加速
5. 可用最小堆维护 top-k，或直接 argsort 后取前 k 个

## 面试口述模板

> "我会实现一个内存向量存储。核心数据结构是一个列表保存文档 id、文本和 embedding。搜索时对 query embedding 和所有文档 embedding 计算余弦相似度——先做矩阵乘法得到点积，再除以范数乘积。然后用 argsort 或 heapq.nlargest 取 top-k 结果返回。这是暴力搜索，适合小规模场景；大规模需要 FAISS 或 HNSW 等近似最近邻索引。"

## 常见坑

- **零向量**：范数为 0 时除法报错，需特判返回相似度 0
- **未归一化**：如果 embedding 已经 L2 归一化，余弦相似度退化为点积，可以省掉除法
- **重复 id**：插入相同 id 应更新而非重复添加，用 dict 比 list 更安全
- **top_k > 文档数**：返回所有文档即可，不要报错

## 复杂度

- 时间：add O(1)；search O(n·d)，n 为文档数，d 为向量维度
- 空间：O(n·d) 存储所有 embedding
