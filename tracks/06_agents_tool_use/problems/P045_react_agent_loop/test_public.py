from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


class SearchTool:
    name = "search"

    def call(self, arguments):
        return f"obs::{arguments['query']}"


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return {"action": "search", "arguments": {"query": "llm"}}
        return {"final_answer": "done"}


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.agent
@pytest.mark.difficulty_4
def test_react_loop_returns_final_answer() -> None:
    try:
        answer = starter.react_loop("what is llm", DummyLLM(), [SearchTool()], max_steps=3)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert answer == "done"
