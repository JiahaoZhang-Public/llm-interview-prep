from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_1
def test_clip_gradients_returns_preclip_norm() -> None:
    model = ToyModel()
    model.weight.grad = torch.tensor([3.0, 4.0])
    try:
        norm = starter.clip_gradients(model.parameters(), max_norm=1.0)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert norm >= 5.0
