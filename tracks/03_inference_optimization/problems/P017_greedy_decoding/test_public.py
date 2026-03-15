from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_greedy_decode_returns_argmax_index() -> None:
    try:
        token_id = starter.greedy_decode([0.1, 0.9, 0.2])
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert token_id == 1
