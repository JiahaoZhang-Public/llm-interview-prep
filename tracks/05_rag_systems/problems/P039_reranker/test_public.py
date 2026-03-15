from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


def scorer(query, doc):
    return len(set(query.split()) & set(doc.split()))


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_2
def test_rerank_documents_sorts_descending() -> None:
    docs = ["llm systems", "systems", "tokens"]
    try:
        ordered = starter.rerank_documents("llm systems", docs, scorer)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert ordered[0] == "llm systems"
