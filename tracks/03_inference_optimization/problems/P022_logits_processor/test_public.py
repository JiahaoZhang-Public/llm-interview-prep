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


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_repetition_penalty_does_not_change_unseen() -> None:
    logits = [1.0, 2.0, 3.0]
    try:
        adjusted = starter.apply_repetition_penalty(logits, [1], penalty=1.2)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert adjusted[0] == 1.0
    assert adjusted[2] == 3.0


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_repetition_penalty_positive_logit_decreases() -> None:
    logits = [1.0, 2.0, 3.0]
    try:
        adjusted = starter.apply_repetition_penalty(logits, [1], penalty=1.5)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert adjusted[1] < 2.0


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_repetition_penalty_negative_logit_more_negative() -> None:
    logits = [1.0, -2.0, 3.0]
    try:
        adjusted = starter.apply_repetition_penalty(logits, [1], penalty=1.5)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert adjusted[1] < -2.0


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_repetition_penalty_empty_generated() -> None:
    logits = [1.0, 2.0, 3.0]
    try:
        adjusted = starter.apply_repetition_penalty(logits, [], penalty=1.5)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert adjusted == logits


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_repetition_penalty_one_is_noop() -> None:
    logits = [1.0, 2.0, 3.0]
    try:
        adjusted = starter.apply_repetition_penalty(logits, [0, 1, 2], penalty=1.0)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    for i in range(3):
        assert abs(adjusted[i] - logits[i]) < 1e-9
