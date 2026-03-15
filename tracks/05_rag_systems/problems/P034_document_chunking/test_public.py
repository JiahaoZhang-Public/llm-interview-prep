from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.rag
@pytest.mark.difficulty_1
def test_chunk_document_respects_overlap() -> None:
    try:
        chunks = starter.chunk_document("abcdefghij", chunk_size=4, overlap=1)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert chunks[0] == "abcd"
    assert chunks[1].startswith("d")
