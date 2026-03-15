class ToolCallingAgent:
    def __init__(self, llm, tools):
        raise NotImplementedError("Initialize this class in the starter.")

    def run(self, user_query: str):
        raise NotImplementedError("Implement P041.run().")
