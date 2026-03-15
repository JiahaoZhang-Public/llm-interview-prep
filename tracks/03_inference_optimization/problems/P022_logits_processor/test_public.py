from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_repetition_penalty_changes_seen_tokens() -> None:
    logits = [1.0, 2.0, 3.0]
    try:
        adjusted = starter.apply_repetition_penalty(logits, [1], penalty=1.2)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(adjusted) == 3
    assert adjusted[1] != logits[1]
