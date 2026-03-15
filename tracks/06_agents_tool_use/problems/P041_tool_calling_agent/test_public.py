from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented
from llm_prep.types import ToolCall


class EchoTool:
    name = "echo"
    description = "Echoes the provided text."

    def call(self, arguments):
        return arguments["text"]


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ToolCall(name="echo", arguments={"text": "tool result"})
        return "final answer"


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.agent
@pytest.mark.difficulty_4
def test_tool_calling_agent_runs_tool_then_returns_answer() -> None:
    agent = starter.ToolCallingAgent(llm=DummyLLM(), tools=[EchoTool()])
    try:
        answer = agent.run("say hi")
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert answer == "final answer"
