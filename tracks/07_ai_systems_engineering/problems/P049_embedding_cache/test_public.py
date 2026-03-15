from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return [float(len(text))]


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.systems
@pytest.mark.difficulty_2
def test_embedding_cache_avoids_duplicate_work() -> None:
    embedder = CountingEmbedder()
    cache = starter.EmbeddingCache(embedder)
    try:
        first = cache.get_embedding("hello")
        second = cache.get_embedding("hello")
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert first == second
    assert embedder.calls == 1
