from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.agent
@pytest.mark.difficulty_2
def test_short_term_memory_keeps_recent_window() -> None:
    memory = starter.ShortTermMemory(max_items=2)
    try:
        memory.append("a")
        memory.append("b")
        memory.append("c")
        recent = memory.recent()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert recent == ["b", "c"]
