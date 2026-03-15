# P037 Simple RAG Pipeline 题解

## 核心思路

1. **Retrieve**：用 retriever 对 query 做语义检索，取回 top-k 相关文档
2. **Augment**：把检索到的文档拼接成 context，填入 prompt template
3. **Generate**：把组装好的 prompt 送给 LLM 生成最终回答

RAG = Retrieval-Augmented Generation，让 LLM 基于外部知识回答问题。

## 面试口述模板

> "RAG pipeline 分三步：Retrieve、Augment、Generate。
> 首先用 retriever（通常基于向量检索）对用户 query 做语义搜索，拿到最相关的 top-k 文档。
> 然后把检索到的文档内容拼接成 context，和 query 一起填入 prompt template。
> 最后把这个 augmented prompt 送给 LLM 生成回答。
> RAG 的好处是不需要把所有知识微调进模型——外部知识库可以随时更新，
> 减少幻觉，还能提供溯源（回答基于哪些文档）。"

## 常见坑

- **context 过长超出上下文窗口**：top-k 太大或文档太长会截断，需要做 chunking 和长度控制
- **检索质量差**：garbage in, garbage out。embedding 模型和 chunking 策略直接影响最终效果
- **prompt template 格式**：query 和 context 的位置、分隔符影响 LLM 理解
- **多文档冲突**：检索到的文档互相矛盾时，LLM 可能给出不一致的回答

## 复杂度

- 检索：O(N) 暴力 / O(log N) 向量索引（HNSW、IVF 等）
- 生成：取决于 LLM 和 prompt 长度
- 端到端延迟 = 检索延迟 + LLM 生成延迟（通常检索 <100ms，生成是瓶颈）
