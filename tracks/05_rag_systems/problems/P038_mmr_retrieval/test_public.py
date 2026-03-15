from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_3
def test_mmr_select_returns_requested_count() -> None:
    doc_embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    try:
        selected = starter.mmr_select([1.0, 0.0], doc_embeddings, lambda_mult=0.5, top_k=2)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(selected) == 2
