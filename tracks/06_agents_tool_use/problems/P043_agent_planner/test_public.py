from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.agent
@pytest.mark.difficulty_3
def test_plan_steps_returns_ordered_list() -> None:
    try:
        steps = starter.plan_steps("answer a support ticket", ["retrieve_docs", "draft_reply"])
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(steps, list)
    assert len(steps) >= 1
