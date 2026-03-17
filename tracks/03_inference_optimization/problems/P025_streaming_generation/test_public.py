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


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_stream_generate_is_generator() -> None:
    import types
    try:
        gen = starter.stream_generate(step_fn, [1, 2], max_new_tokens=3)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(gen, types.GeneratorType), "stream_generate should return a generator"


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_stream_generate_eos_early_stop() -> None:
    def eos_step(tokens):
        return 99 if len(tokens) >= 4 else len(tokens)

    try:
        output = list(starter.stream_generate(eos_step, [0, 1], max_new_tokens=100, eos_token_id=99))
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert 99 in output, "EOS token should be yielded"
    assert len(output) <= 10, "Should stop early after EOS"


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_stream_generate_context_grows() -> None:
    seen_lengths = []

    def tracking_step(tokens):
        seen_lengths.append(len(tokens))
        return 42

    try:
        list(starter.stream_generate(tracking_step, [1, 2], max_new_tokens=3))
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert seen_lengths == [2, 3, 4]


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_stream_generate_max_zero() -> None:
    try:
        output = list(starter.stream_generate(step_fn, [1, 2], max_new_tokens=0))
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert output == []
