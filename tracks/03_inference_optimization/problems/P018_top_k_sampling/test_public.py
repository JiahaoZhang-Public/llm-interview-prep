import random
from pathlib import Path
import pytest
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
@pytest.mark.top10
def test_top_k_sample_only_returns_from_top_k() -> None:
    random.seed(0)
    try:
        token_id = starter.top_k_sample([0.1, 0.9, 0.8, 0.0], k=2)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {1, 2}

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_top_k_with_k1_is_greedy() -> None:
    random.seed(42)
    try:
        for _ in range(5):
            token_id = starter.top_k_sample([0.1, 0.9, 0.8, 0.0], k=1)
            assert token_id == 1
    except Exception as exc:
        skip_not_implemented(exc)

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_top_k_never_returns_outside_k() -> None:
    logits = [1.0, 10.0, 9.0, 0.5, 0.1]
    results = set()
    try:
        for seed in range(50):
            random.seed(seed)
            results.add(starter.top_k_sample(logits, k=2))
    except Exception as exc:
        skip_not_implemented(exc)
    assert results.issubset({1, 2})

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_top_k_with_negative_logits() -> None:
    random.seed(0)
    try:
        token_id = starter.top_k_sample([-10.0, -1.0, -2.0, -20.0], k=2)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {1, 2}

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_top_k_full_vocab() -> None:
    random.seed(0)
    try:
        token_id = starter.top_k_sample([1.0, 2.0, 3.0], k=3)
    except Exception as exc:
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}
