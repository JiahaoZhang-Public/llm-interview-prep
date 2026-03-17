from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
@pytest.mark.top10
def test_kv_cache_appends_incrementally() -> None:
    cache = starter.KVCache()
    try:
        cache.append([1], [10])
        cache.append([2], [20])
        keys, values = cache.get()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(keys) == 2
    assert len(values) == 2


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_kv_cache_reset_clears_state() -> None:
    cache = starter.KVCache()
    try:
        cache.append([1], [10])
        cache.append([2], [20])
        cache.reset()
        keys, values = cache.get()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(keys) == 0
    assert len(values) == 0


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_kv_cache_empty_get() -> None:
    cache = starter.KVCache()
    try:
        keys, values = cache.get()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert keys == []
    assert values == []


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_kv_cache_preserves_insertion_order() -> None:
    cache = starter.KVCache()
    try:
        cache.append("a", "x")
        cache.append("b", "y")
        cache.append("c", "z")
        keys, values = cache.get()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert keys == ["a", "b", "c"]
    assert values == ["x", "y", "z"]


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
def test_kv_cache_reuse_after_reset() -> None:
    """Cache should be usable after reset"""
    cache = starter.KVCache()
    try:
        cache.append([1], [10])
        cache.reset()
        cache.append([99], [99])
        keys, values = cache.get()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(keys) == 1
    assert keys[0] == [99]
