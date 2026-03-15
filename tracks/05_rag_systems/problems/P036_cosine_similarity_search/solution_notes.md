# P036 Cosine Similarity Search 题解

## 核心思路

1. 给定 query 向量和文档向量矩阵，计算 query 与每个文档的余弦相似度
2. cosine_sim(a, b) = dot(a, b) / (||a|| × ||b||)
3. 批量计算：scores = (doc_matrix @ query) / (doc_norms * query_norm)
4. 对 scores 做 argsort 降序，取前 k 个索引作为检索结果
5. 返回 (doc_id, score) 列表，按相似度降序排列

## 面试口述模板

> "余弦相似度搜索的关键是向量化计算。我先把所有文档 embedding 堆成矩阵，然后用矩阵乘法一次算出 query 和所有文档的点积。再分别算出文档范数向量和 query 范数，做逐元素除法得到余弦相似度数组。最后 argsort 取 top-k。整个过程 numpy 向量化，避免 Python 循环，效率高。"

## 常见坑

- **范数为零**：全零向量的范数为 0，需加 epsilon（如 1e-8）防止除零
- **相似度 > 1**：浮点误差可能导致结果略大于 1，必要时 clip 到 [-1, 1]
- **argsort 方向**：numpy argsort 默认升序，取 top-k 要用 `[::-1][:k]` 或 `np.argpartition`
- **稀疏 vs 稠密**：如果向量很稀疏，用 scipy sparse 能节省内存和计算

## 复杂度

- 时间：O(n·d) 计算相似度 + O(n log k) 取 top-k（用 argpartition 可做到 O(n)）
- 空间：O(n) 存储相似度分数数组
