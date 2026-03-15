# 知识卡片

## 为什么 chunk overlap 很重要？

它保留边界上下文，减少关键信息被切断。

## MMR 优化的目标是什么？

在相关性和多样性之间做平衡。

## 为什么向量召回后还需要 reranker？

向量召回覆盖广但排序不一定精确，reranker 用来提升排序质量。

## multi-query retrieval 主要修复什么问题？

避免单一 query 表达覆盖不足导致漏召回。
