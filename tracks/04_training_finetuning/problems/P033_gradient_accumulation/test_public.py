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
@pytest.mark.difficulty_2
def test_gradient_accumulation_returns_step_count() -> None:
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        {"inputs": torch.randn(2, 4), "labels": torch.tensor([0, 1])},
        {"inputs": torch.randn(2, 4), "labels": torch.tensor([1, 0])},
    ]
    try:
        step_count = starter.run_gradient_accumulation(model, batches, optimizer, accumulation_steps=2)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert step_count == 1
