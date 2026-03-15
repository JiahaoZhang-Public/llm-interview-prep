# P025 Streaming Generation 题解

## 核心思路

1. 用 generator（yield）逐个输出新生成的 token
2. 每步调用 step_fn 获取下一个 token，append 到序列，yield 出去
3. 遇到 EOS 或达到 max_new_tokens 时停止
4. 调用方可以边生成边处理（如 SSE 推送给前端）

## 面试口述模板

> "Streaming generation 用 Python generator 实现逐 token 输出。
> 每步调用 step_fn 得到 next token 后 yield 出去，
> 调用方可以立即处理（比如通过 SSE 发给前端显示），
> 不需要等待整个序列生成完毕。
> 这对用户体验至关重要——用户看到文字逐渐出现而非等几秒后一次性显示。
> 实际系统中还要处理 detokenization 的边界（token 可能跨字符边界）。"

## 常见坑

- **EOS 检查时机**：yield 之后再 break，确保 EOS token 也被输出（或按需求不输出）
- **token 拼接到上下文**：yield 前要把 token append 到序列，不然 step_fn 看不到历史
- **detokenization**：subword token 可能只是一个字符片段，需要 buffer 直到形成完整字符
- **取消机制**：实际系统需要支持客户端断连时停止生成

## 复杂度

- 时间：O(T × C)，T = 生成 token 数，C = step_fn 的开销
- 空间：O(T) 累积的 token 序列
