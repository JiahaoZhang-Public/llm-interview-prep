from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.training
@pytest.mark.difficulty_2
def test_merge_lora_weights_preserves_dense_shape() -> None:
    base_weight = torch.zeros(3, 4)
    lora_a = torch.ones(2, 4)
    lora_b = torch.ones(3, 2)
    try:
        merged = starter.merge_lora_weights(base_weight, lora_a, lora_b, alpha=2.0)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(merged.shape) == (3, 4)
