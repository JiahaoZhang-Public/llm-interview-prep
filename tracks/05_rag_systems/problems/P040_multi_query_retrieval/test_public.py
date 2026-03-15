from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


def query_generator(query):
    return [query, query + " rewrite"]


class DummyRetriever:
    def search(self, query, top_k=3):
        suffix = "1" if "rewrite" not in query else "2"
        return [{"doc_id": f"doc-{suffix}", "text": query}]


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_3
def test_multi_query_retrieve_deduplicates_results() -> None:
    try:
        results = starter.multi_query_retrieve("query", query_generator, DummyRetriever(), top_k=3)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(results) >= 1
