from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


class DummyRetriever:
    def search(self, query, top_k=3):
        return [{"doc_id": "doc-1", "text": "retrieved context"}]


def dummy_llm(prompt):
    return f"answer::{prompt}"


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_4
@pytest.mark.top10
def test_simple_rag_pipeline_returns_final_answer() -> None:
    try:
        answer = starter.run_simple_rag("what is rag", DummyRetriever(), dummy_llm, "Question: {query}\nContext: {context}")
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert answer.startswith("answer::")
