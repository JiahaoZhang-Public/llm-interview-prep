from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
@pytest.mark.top10
def test_multi_head_attention_shape() -> None:
    try:
        module = starter.MultiHeadAttention(hidden_size=8, num_heads=2)
        output = module(torch.randn(2, 4, 8))
    except Exception as exc:  # pragma: no cover - practice starter path
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 4, 8)
