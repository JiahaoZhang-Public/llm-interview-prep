from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_2
def test_causal_lm_loss_returns_scalar() -> None:
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
    try:
        loss = starter.causal_lm_loss(logits, labels)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert loss.ndim == 0
