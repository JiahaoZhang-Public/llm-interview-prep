from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_3
@pytest.mark.top10
def test_lora_linear_returns_projected_shape() -> None:
    try:
        layer = starter.LoRALinear(in_features=4, out_features=3, rank=2)
        output = layer(torch.randn(2, 4))
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 3)
