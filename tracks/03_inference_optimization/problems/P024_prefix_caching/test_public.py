from pathlib import Path
import pytest
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_prefix_cache_returns_longest_match() -> None:
    cache = starter.PrefixCache()
    try:
        cache.add_prefix((1, 2), "short")
        cache.add_prefix((1, 2, 3), "long")
        value = cache.lookup((1, 2, 3, 4))
    except Exception as exc:
        skip_not_implemented(exc)
    assert value == "long"

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_prefix_cache_shorter_match() -> None:
    cache = starter.PrefixCache()
    try:
        cache.add_prefix((1, 2), "short")
        cache.add_prefix((1, 2, 3), "long")
        value = cache.lookup((1, 2, 5))
    except Exception as exc:
        skip_not_implemented(exc)
    assert value == "short"

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_prefix_cache_no_match() -> None:
    cache = starter.PrefixCache()
    try:
        cache.add_prefix((1, 2), "cached")
        value = cache.lookup((3, 4, 5))
    except Exception as exc:
        skip_not_implemented(exc)
    assert value is None

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_prefix_cache_exact_match() -> None:
    cache = starter.PrefixCache()
    try:
        cache.add_prefix((1, 2, 3), "exact")
        value = cache.lookup((1, 2, 3))
    except Exception as exc:
        skip_not_implemented(exc)
    assert value == "exact"

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_prefix_cache_empty_cache_returns_none() -> None:
    cache = starter.PrefixCache()
    try:
        value = cache.lookup((1, 2, 3))
    except Exception as exc:
        skip_not_implemented(exc)
    assert value is None

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_prefix_cache_query_shorter_than_all_cached() -> None:
    cache = starter.PrefixCache()
    try:
        cache.add_prefix((1, 2, 3, 4, 5), "very_long")
        value = cache.lookup((1, 2))
    except Exception as exc:
        skip_not_implemented(exc)
    assert value is None
