from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)


starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_1
def test_freeze_parameters_keeps_matching_names_trainable() -> None:
    model = ToyModel()
    try:
        trainable = starter.freeze_parameters(model, ["head"])
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert any("head" in name for name in trainable)
