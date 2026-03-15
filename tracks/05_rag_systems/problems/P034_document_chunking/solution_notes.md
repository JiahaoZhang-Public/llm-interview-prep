# P034 Document Chunking 题解

## 核心思路

1. 按固定 `chunk_size` 切分文本，相邻 chunk 之间保留 `overlap` 个字符的重叠区域，防止语义断裂
2. 用滑动窗口实现：起始位置 `start` 每次前进 `chunk_size - overlap`，截取 `text[start:start+chunk_size]`
3. 当剩余文本不足 `chunk_size` 时，将尾部作为最后一个 chunk（避免丢失末尾信息）
4. 返回 chunk 列表，每个 chunk 可附带元数据（起始偏移、chunk 编号）

## 面试口述模板

> "Document Chunking 的核心是滑动窗口切分。我会用 start 指针从 0 开始，每次取 chunk_size 长度的子串，然后 start 前进 chunk_size - overlap 步。overlap 保证相邻 chunk 在边界处有重复内容，这样检索时不会因为一句话被截断而丢失语义。最后不足 chunk_size 的尾部直接作为最后一个 chunk 返回。时间复杂度 O(n)，n 是文本总长度。"

## 常见坑

- **overlap >= chunk_size**：会导致窗口不前进或倒退，需要校验 `overlap < chunk_size`
- **空文本或极短文本**：直接返回空列表或整段作为单个 chunk，不要进入死循环
- **按字符 vs 按 token 切分**：面试时要确认单位；按 token 切分更精确但需要 tokenizer
- **末尾 chunk 过短**：可选择将过短的尾部合并到上一个 chunk，避免产生信息碎片

## 复杂度

- 时间：O(n)，n 为文本长度，每个字符最多被访问常数次
- 空间：O(n)，存储所有 chunk 的总字符量约为 n × chunk_size / (chunk_size - overlap)
