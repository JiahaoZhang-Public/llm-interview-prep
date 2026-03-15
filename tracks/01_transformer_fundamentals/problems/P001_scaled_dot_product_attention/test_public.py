from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
@pytest.mark.top10
def test_scaled_dot_product_attention_supports_mask() -> None:
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    mask = torch.tensor([[0.0, float("-inf")]])
    try:
        output = starter.scaled_dot_product_attention(q, k, v, mask=mask)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 1, 2)
    assert torch.allclose(output[0, 0], torch.tensor([2.0, 0.0]), atol=1e-4)
