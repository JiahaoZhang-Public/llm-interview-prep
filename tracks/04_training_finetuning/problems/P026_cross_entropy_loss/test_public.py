from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_2
def test_cross_entropy_loss_returns_scalar() -> None:
    logits = torch.tensor([[2.0, 1.0]])
    targets = torch.tensor([0])
    try:
        loss = starter.cross_entropy_loss(logits, targets)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert loss.ndim == 0
