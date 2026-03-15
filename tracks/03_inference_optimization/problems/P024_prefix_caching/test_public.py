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
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert value == "long"
