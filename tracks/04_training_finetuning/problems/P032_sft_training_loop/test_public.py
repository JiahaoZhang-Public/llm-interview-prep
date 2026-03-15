from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_3
def test_sft_train_step_returns_float_loss() -> None:
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {"inputs": torch.randn(2, 4), "labels": torch.tensor([0, 1])}
    try:
        loss = starter.sft_train_step(model, batch, optimizer)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert isinstance(loss, float)
