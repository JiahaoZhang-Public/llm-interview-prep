# P045 ReAct Agent Loop — Solution Notes

## Core Approach

1. ReAct alternates between three phases: Thought, Action, Observation
2. **Thought**: The LLM reasons about what to do next (chain-of-thought)
3. **Action**: The LLM specifies a tool call based on the reasoning
4. **Observation**: Execute the tool and return the result to the LLM
5. Loop continues until the LLM produces a "Final Answer" instead of an action
6. The full Thought-Action-Observation trace is appended to context at each iteration

## Interview Oral Template

> "ReAct combines reasoning and acting in a single loop. At each step, the
> LLM first generates a Thought -- an explicit reasoning trace about what
> it knows and what it needs to do next. Then it chooses an Action -- a
> specific tool call. The tool is executed and the result is returned as
> an Observation. This triplet is appended to the conversation and the
> loop repeats. The explicit Thought step is what distinguishes ReAct from
> plain tool calling -- it improves accuracy by forcing the model to reason
> before acting. The loop ends when the model outputs a Final Answer."

## Common Pitfalls

- **Skipping the Thought step**: Without explicit reasoning, the agent makes more errors -- the Thought step is not optional, it's the key innovation
- **Context explosion**: Each iteration adds Thought + Action + Observation to context -- long traces can exceed the context window
- **Parsing errors**: Must robustly parse the LLM's output to extract Thought, Action, and action arguments -- the model may not always follow the format exactly
- **Max iterations**: Must cap iterations to prevent infinite loops where the agent reasons in circles

## Complexity

- Time: O(iterations * (LLM_call + tool_execution)) -- typically 3-8 iterations
- Space: O(iterations * (thought_length + observation_length)) for the growing trace
- The explicit reasoning trace makes debugging much easier compared to opaque tool-calling agents
