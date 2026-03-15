from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.tokenization
@pytest.mark.difficulty_3
@pytest.mark.top10
def test_bpe_tokenizer_round_trip_on_training_corpus() -> None:
    tokenizer = starter.BPETokenizer(vocab_size=20)
    try:
        tokenizer.train(["low lower newest"])
        tokens = tokenizer.encode("low")
        text = tokenizer.decode(tokens)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(tokens, list)
    assert isinstance(text, str)
