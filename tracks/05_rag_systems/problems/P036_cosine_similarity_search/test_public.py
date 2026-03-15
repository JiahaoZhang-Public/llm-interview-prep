from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_2
@pytest.mark.top10
def test_cosine_similarity_search_returns_best_id() -> None:
    try:
        results = starter.cosine_similarity_search([1.0, 0.0], {"a": [1.0, 0.0], "b": [0.0, 1.0]}, top_k=1)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    top_id = results[0][0] if isinstance(results[0], tuple) else results[0]["doc_id"]
    assert top_id == "a"
