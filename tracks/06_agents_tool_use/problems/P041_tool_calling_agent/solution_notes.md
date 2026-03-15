# P041 Tool Calling Agent — Solution Notes

## Core Approach

1. Agent loop: send user message to LLM, check if the response contains a tool call
2. If ToolCall is returned: parse the function name and arguments, execute the tool, feed the result back to the LLM
3. If no ToolCall (plain text response): return the final answer to the user
4. Loop until the LLM produces a final text response or max iterations reached
5. The LLM decides when and which tool to call based on the available function schemas

## Interview Oral Template

> "A tool-calling agent runs in a loop. I send the conversation to the LLM
> along with function schemas describing available tools. If the LLM returns
> a tool call, I parse the function name and arguments, execute the tool,
> and append the result to the conversation as a tool message. Then I call
> the LLM again with the updated conversation. This repeats until the LLM
> produces a final text response instead of a tool call. The key design
> decisions are: which tools to expose, how to format tool results, and
> setting a max iteration limit to prevent infinite loops."

## Common Pitfalls

- **Infinite loop**: The LLM might keep calling tools forever -- must have a max_iterations guard
- **Error handling**: Tool execution can fail -- must catch exceptions and feed error messages back to the LLM so it can recover
- **Conversation context growth**: Each tool call/result pair adds to the context -- can exceed the context window in long interactions
- **Tool schema quality**: Vague or incorrect function descriptions cause the LLM to misuse tools or hallucinate parameters

## Complexity

- Time: O(iterations * (LLM_call + tool_execution)) -- dominated by LLM inference latency
- Space: O(conversation_length) for the growing message history
- Typically 1-5 iterations for simple tasks; complex tasks may need more
