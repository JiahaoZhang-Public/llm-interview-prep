from llm_prep.types import ToolCall


class ToolCallingAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.messages = []

    def run(self, user_query: str):
        self.messages.append({"role": "user", "content": user_query})

        while True:
            response = self.llm.generate(self.messages, tools=list(self.tools.values()))

            if isinstance(response, ToolCall):
                tool = self.tools[response.name]
                result = tool.call(response.arguments)
                self.messages.append({"role": "tool", "name": response.name, "content": str(result)})
            else:
                return response
