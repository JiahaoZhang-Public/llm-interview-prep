def react_loop(question: str, llm, tools, max_steps: int = 5):
    tool_map = {tool.name: tool for tool in tools}
    context = f"Question: {question}\n"

    for _ in range(max_steps):
        response = llm(context)

        if isinstance(response, dict) and "final_answer" in response:
            return response["final_answer"]

        if isinstance(response, dict) and "action" in response:
            action = response["action"]
            arguments = response.get("arguments", {})
            tool = tool_map[action]
            observation = tool.call(arguments)
            context += f"Action: {action}\nObservation: {observation}\n"

    return None
