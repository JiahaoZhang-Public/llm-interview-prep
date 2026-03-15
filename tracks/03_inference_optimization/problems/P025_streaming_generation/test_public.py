from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


def step_fn(tokens):
    return len(tokens) + 10


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_stream_generate_yields_tokens() -> None:
    try:
        output = list(starter.stream_generate(step_fn, [1, 2], max_new_tokens=3))
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(output) == 3
