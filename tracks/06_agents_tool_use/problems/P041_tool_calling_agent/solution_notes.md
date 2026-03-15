# P041 Tool-Calling Agent 题解

## 核心思路

1. Agent 循环：prompt LLM → 解析输出 → 判断是否需要调用工具 → 执行工具 → 将结果注入上下文 → 继续循环
2. LLM 输出中包含结构化的工具调用指令（如 JSON 格式的 function_call）
3. 解析工具名和参数，在注册的工具字典中查找并执行
4. 将工具返回值作为 observation 拼回 messages，再次调用 LLM
5. 当 LLM 输出最终答案（不含工具调用）时终止循环

## 面试口述模板

> "Tool-Calling Agent 的核心是一个 while 循环。每次迭代把当前 messages 发给 LLM，检查返回中是否有 tool_call。如果有，就从工具注册表中找到对应函数，用解析出的参数执行，把结果包装成 tool message 追加到 messages。如果没有 tool_call，说明 LLM 给出了最终回答，退出循环。关键是要设置最大迭代次数防止死循环，并做好异常处理——工具执行失败时把错误信息返回给 LLM 让它自行修正。"

## 常见坑

- **无限循环**：必须设 max_iterations，LLM 可能反复调用同一工具
- **参数解析失败**：LLM 生成的 JSON 可能格式错误，需 try-catch 并将错误反馈给 LLM
- **工具不存在**：LLM 可能幻觉一个不存在的工具名，需返回错误提示
- **上下文膨胀**：每轮工具调用都增加 messages 长度，需注意 token 限制

## 复杂度

- 时间：O(T · C_llm)，T 为迭代轮数，C_llm 为单次 LLM 调用开销
- 空间：O(T · L)，L 为每轮 message 平均长度，messages 列表线性增长
