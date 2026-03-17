import random
from pathlib import Path
import pytest
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_temperature_sample_returns_valid_index() -> None:
    random.seed(0)
    try:
        token_id = starter.temperature_sample([1.0, 2.0, 3.0], temperature=0.7)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_temperature_low_favors_argmax() -> None:
    counts = {0: 0, 1: 0, 2: 0}
    try:
        for seed in range(100):
            random.seed(seed)
            tid = starter.temperature_sample([1.0, 2.0, 10.0], temperature=0.1)
            counts[tid] += 1
    except Exception as exc:
        skip_not_implemented(exc)
    assert counts[2] > 90

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_temperature_high_more_uniform() -> None:
    counts = {0: 0, 1: 0, 2: 0}
    try:
        for seed in range(300):
            random.seed(seed)
            tid = starter.temperature_sample([1.0, 2.0, 3.0], temperature=10.0)
            counts[tid] += 1
    except Exception as exc:
        skip_not_implemented(exc)
    assert all(c > 50 for c in counts.values())

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_temperature_one_preserves_distribution() -> None:
    random.seed(0)
    try:
        token_id = starter.temperature_sample([1.0, 2.0, 3.0], temperature=1.0)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}
