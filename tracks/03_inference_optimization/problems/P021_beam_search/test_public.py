from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


def step_fn(tokens):
    last = tokens[-1]
    if last == 2:
        return [(-0.1, 2)]
    return [(-0.1, last + 1), (-0.5, 2)]


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
@pytest.mark.top10
def test_beam_search_returns_sequence() -> None:
    try:
        sequence = starter.beam_search(step_fn, start_tokens=[0], beam_size=2, max_new_tokens=3, eos_token_id=2)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(sequence, list)
    assert sequence[0] == 0
