# P014 Tokenize & Detokenize Round-Trip 题解

## 核心思路

1. **fit**：扫描语料，为每个唯一 word 分配一个 id，建立双向映射
2. **encode**：按空格切词，查表得到 id 序列
3. **decode**：id 序列查反向表，用空格拼接还原文本
4. 关键约束：`decode(encode(text)) == text` 必须成立

## 面试口述模板

> "这是一个最简单的词级 tokenizer，核心是保证 round-trip 可逆性。
> fit 阶段按空格分词并建立 word↔id 双向映射。
> encode 查正向表，decode 查反向表用空格拼接。
> 实际场景中，词级 tokenizer 的问题是 OOV——任何训练时没见过的词都处理不了。
> 这就是为什么现代 NLP 都转向了 subword tokenizer（BPE/WordPiece/SentencePiece）。"

## 常见坑

- **空格处理**：encode 和 decode 都按空格切分/拼接，多空格或前后空格需要统一处理
- **OOV**：简单词级 tokenizer 没有 UNK 回退机制
- **id 分配确定性**：按遍历顺序分配，相同输入要产生相同映射

## 复杂度

- fit：O(N)，N 为语料总词数
- encode/decode：O(L)，L 为文本词数
