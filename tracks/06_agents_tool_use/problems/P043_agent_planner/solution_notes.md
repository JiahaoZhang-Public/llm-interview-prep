# P043 Agent Planner — Solution Notes

## Core Approach

1. Given a user task and available tools, decompose the task into an ordered sequence of steps
2. Each step specifies: which tool to call, what arguments to use, and dependencies on previous steps
3. The planner analyzes the task, identifies required information, and determines the correct tool order
4. Output is a plan (list of steps) that an executor can run sequentially
5. Steps may depend on outputs of previous steps (data flow between tools)

## Interview Oral Template

> "An agent planner breaks down a complex task into an ordered sequence of
> tool calls. Given the user's goal and the available tool descriptions,
> the planner identifies what information is needed and which tools provide
> it, then arranges them in dependency order. For example, to answer
> 'what's the weather in the user's city', the plan would be: step 1 --
> call get_location to find the city, step 2 -- call get_weather with the
> city from step 1. The planner can be LLM-based (generate the plan in
> natural language or JSON) or rule-based for constrained domains."

## Common Pitfalls

- **Ignoring dependencies**: Steps that depend on previous outputs must be ordered correctly -- parallel execution is only safe for independent steps
- **Over-planning**: Breaking a simple task into too many steps adds latency and error surface -- sometimes a single tool call suffices
- **Missing error recovery**: The plan should handle what to do if a step fails -- retry, skip, or abort
- **Tool hallucination**: The planner might reference tools that don't exist -- must validate each step against the actual tool registry

## Complexity

- Time: O(LLM_call) for generating the plan + O(steps * tool_execution) for running it
- Space: O(steps) for the plan representation plus intermediate results
- Planning overhead is typically one LLM call; execution dominates total time
