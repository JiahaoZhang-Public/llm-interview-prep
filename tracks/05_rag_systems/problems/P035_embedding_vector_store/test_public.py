from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


def embedder(text):
    return [float(len(text)), 1.0]


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_3
def test_vector_store_adds_and_searches_documents() -> None:
    store = starter.InMemoryVectorStore(embedder=embedder)
    try:
        store.add_document("doc-1", "short")
        store.add_document("doc-2", "longer text")
        results = store.search("short", top_k=1)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(results) == 1
