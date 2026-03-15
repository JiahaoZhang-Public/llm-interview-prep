# P039 Reranker 题解

## 核心思路

1. 第一阶段用 bi-encoder（向量检索）快速召回 top-N 候选文档
2. 第二阶段用 cross-encoder 对 (query, doc) 对逐一打分，精排得到 top-k
3. cross-encoder 将 query 和 doc 拼接为 "[CLS] query [SEP] doc [SEP]"，输出单一相关性分数
4. 实现 rerank 函数：接收 query、候选列表、scorer 函数，返回按分数降序排列的结果
5. scorer 函数签名：`score(query, doc) -> float`，可以是真实模型或模拟函数

## 面试口述模板

> "Reranker 是 RAG 中典型的两阶段检索架构。第一阶段 bi-encoder 独立编码 query 和文档，用向量相似度快速召回候选。第二阶段 cross-encoder 把 query 和文档拼在一起过 Transformer，能捕获更细粒度的交互特征，但计算成本高。所以实践中先召回几十到几百个候选，再用 cross-encoder 精排取 top-k。实现上就是对每个候选调用 scorer 得到分数，然后排序截断。"

## 常见坑

- **cross-encoder 不能预计算文档 embedding**：每次 query 变化都需要重新计算，这是和 bi-encoder 的关键区别
- **候选数量控制**：cross-encoder 是 O(N) 次推理，N 太大延迟不可接受
- **分数归一化**：不同 scorer 的分数范围可能不同，排序用原始分数即可，不要随意归一化
- **空候选列表**：直接返回空列表，不要报错

## 复杂度

- 时间：O(N · L²)，N 个候选，每个经过 cross-encoder 推理，L 为序列长度
- 空间：O(N) 存储分数，cross-encoder 推理时 O(L²) attention 矩阵
