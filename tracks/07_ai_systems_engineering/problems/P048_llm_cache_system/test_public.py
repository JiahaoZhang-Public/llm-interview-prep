from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.systems
@pytest.mark.difficulty_2
def test_prompt_cache_hits_by_prompt() -> None:
    cache = starter.PromptCache()
    try:
        cache.put("hello", {"text": "world"})
        value = cache.get("hello")
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert value == {"text": "world"}
