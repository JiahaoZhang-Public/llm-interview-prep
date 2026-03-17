from pathlib import Path
import pytest
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

def step_fn(tokens):
    last = tokens[-1]
    if last == 2: return [(-0.1, 2)]
    return [(-0.1, last + 1), (-0.5, 2)]

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
@pytest.mark.top10
def test_beam_search_returns_sequence() -> None:
    try:
        sequence = starter.beam_search(step_fn, start_tokens=[0], beam_size=2, max_new_tokens=3, eos_token_id=2)
    except Exception as exc:
        skip_not_implemented(exc)
    assert isinstance(sequence, list)
    assert sequence[0] == 0

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_beam_search_beam1_is_greedy() -> None:
    def greedy_step(tokens):
        return [(-0.1, 10), (-0.5, 20)]
    try:
        seq = starter.beam_search(greedy_step, [0], beam_size=1, max_new_tokens=3)
    except Exception as exc:
        skip_not_implemented(exc)
    assert seq == [0, 10, 10, 10]

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_beam_search_eos_early_stop() -> None:
    def counting_step(tokens):
        return [(-0.1, 99)]
    try:
        seq = starter.beam_search(counting_step, [0], beam_size=2, max_new_tokens=100, eos_token_id=99)
    except Exception as exc:
        skip_not_implemented(exc)
    assert len(seq) <= 5
    assert 99 in seq

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_beam_search_preserves_start_tokens() -> None:
    try:
        seq = starter.beam_search(step_fn, [5, 6], beam_size=2, max_new_tokens=2, eos_token_id=2)
    except Exception as exc:
        skip_not_implemented(exc)
    assert seq[0] == 5
    assert seq[1] == 6

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_beam_search_no_eos() -> None:
    def simple_step(tokens):
        return [(-0.1, 1), (-0.5, 2)]
    try:
        seq = starter.beam_search(simple_step, [0], beam_size=2, max_new_tokens=4, eos_token_id=None)
    except Exception as exc:
        skip_not_implemented(exc)
    assert len(seq) == 5
