def plan_steps(task: str, available_tools):
    steps = []
    for tool in available_tools:
        steps.append({"tool": tool, "reason": f"Use {tool} for: {task}"})
    return steps
