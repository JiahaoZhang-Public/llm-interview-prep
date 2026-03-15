from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.tokenization
@pytest.mark.difficulty_2
def test_train_subword_vocab_returns_dict() -> None:
    try:
        vocab = starter.train_subword_vocab(["a aa aaa"], vocab_size=8)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(vocab, dict)
    assert len(vocab) <= 8
