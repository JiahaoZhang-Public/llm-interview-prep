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
