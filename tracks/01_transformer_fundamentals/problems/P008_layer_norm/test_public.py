from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_layer_norm_normalizes_last_dimension() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = torch.ones(3)
    beta = torch.zeros(3)
    try:
        out = starter.layer_norm(x, gamma, beta)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(1), atol=1e-4)
