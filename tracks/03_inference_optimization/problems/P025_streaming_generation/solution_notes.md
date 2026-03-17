# P025 Streaming Generation 题解

## 核心思路

1. Python generator（yield）逐个输出新 token
2. 每步：step_fn → append → yield
3. 遇 EOS 或达 max_new_tokens 停止
4. 调用方边生成边处理（SSE 推送）

## 面试口述模板

> "Streaming generation 用 generator 逐 token 输出。
> 循环中 step_fn 得 next token，先 append 到上下文，然后 yield。
> 遇 EOS 或达 max tokens 时停止。
> 实际系统中包装成 SSE 推送给前端。
> 注意事项：detokenization 的 subword 边界、客户端断连时取消生成、
> EOS token 是否 yield 取决于需求。"

## 常见坑

- **EOS 检查时机**：yield 后再 break
- **token 追加到上下文**：yield 前要 append
- **detokenization**：subword 可能是半个字
- **取消机制**：检测客户端断连
- **max_new_tokens=0**：直接返回空 generator

## 复杂度

- 时间：O(T*C)，T = 生成 token 数，C = step_fn 开销
- 空间：O(T) 上下文序列
