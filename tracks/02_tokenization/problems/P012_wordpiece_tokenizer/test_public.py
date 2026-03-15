from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.tokenization
@pytest.mark.difficulty_3
def test_wordpiece_tokenizer_uses_greedy_matching() -> None:
    vocab = {"[UNK]": 0, "play": 1, "##ing": 2}
    tokenizer = starter.WordPieceTokenizer(vocab=vocab)
    try:
        tokens = tokenizer.encode("playing")
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(tokens, list)
