# P045 ReAct Agent Loop 题解

## 核心思路

1. ReAct = Reasoning + Acting，每步包含三个阶段：Thought → Action → Observation
2. Thought：LLM 输出推理过程（自然语言思考当前状态和下一步策略）
3. Action：LLM 决定调用哪个工具及参数（结构化输出）
4. Observation：执行工具后返回结果，拼回 prompt 供下一轮使用
5. 循环直到 LLM 输出 "Final Answer" 或达到最大轮次

## 面试口述模板

> "ReAct 是 Agent 的经典范式。每一轮我让 LLM 先输出 Thought 解释推理过程，然后输出 Action 指定工具调用，我执行工具后把 Observation 拼回上下文。这样 LLM 能看到完整的推理链条，在下一轮做出更好的决策。和纯 tool-calling 的区别是 Thought 阶段的 chain-of-thought 让模型显式推理，减少错误调用。我会用正则或特定格式解析 Thought/Action/Observation，并设最大轮次防止死循环。"

## 常见坑

- **格式解析**：LLM 输出可能不严格遵循 Thought/Action 格式，需要鲁棒的解析逻辑
- **Thought 泄漏**：Thought 中可能直接给出答案而不调用工具，需检测 Final Answer 模式
- **Observation 过长**：工具返回内容可能很长，需截断避免撑爆上下文
- **early stop**：如果连续多轮 Thought 重复或 Action 相同，应提前终止

## 复杂度

- 时间：O(T · (C_llm + C_tool))，T 为轮次，C_llm 为 LLM 推理开销，C_tool 为工具执行开销
- 空间：O(T · L)，L 为每轮 Thought + Action + Observation 的平均 token 数
