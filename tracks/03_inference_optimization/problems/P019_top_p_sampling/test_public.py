import random
from pathlib import Path
import pytest
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
@pytest.mark.top10
def test_top_p_sample_returns_valid_index() -> None:
    random.seed(0)
    try:
        token_id = starter.top_p_sample([3.0, 2.0, 0.1], p=0.8)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_top_p_sharp_distribution() -> None:
    results = set()
    try:
        for seed in range(20):
            random.seed(seed)
            results.add(starter.top_p_sample([100.0, 0.0, 0.0], p=0.5))
    except Exception as exc:
        skip_not_implemented(exc)
    assert results == {0}

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_top_p_full_distribution() -> None:
    results = set()
    try:
        for seed in range(100):
            random.seed(seed)
            results.add(starter.top_p_sample([1.0, 1.0, 1.0], p=1.0))
    except Exception as exc:
        skip_not_implemented(exc)
    assert len(results) >= 2

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_top_p_with_negative_logits() -> None:
    random.seed(42)
    try:
        token_id = starter.top_p_sample([-1.0, -2.0, -10.0], p=0.9)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}
