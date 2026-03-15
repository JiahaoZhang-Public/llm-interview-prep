from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.tokenization
@pytest.mark.difficulty_2
def test_round_trip_tokenizer_is_reversible() -> None:
    tokenizer = starter.RoundTripTokenizer()
    try:
        tokenizer.fit(["hello world"])
        token_ids = tokenizer.encode("hello world")
        text = tokenizer.decode(token_ids)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert text == "hello world"
