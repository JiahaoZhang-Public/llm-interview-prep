from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
@pytest.mark.top10
def test_mini_transformer_lm_returns_vocab_logits() -> None:
    try:
        model = starter.MiniTransformerLM(vocab_size=32, hidden_size=16, num_heads=4, intermediate_size=32)
        logits = model(torch.randint(0, 32, (2, 5)))
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(logits.shape) == (2, 5, 32)
