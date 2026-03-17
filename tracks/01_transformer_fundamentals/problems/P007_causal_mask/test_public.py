from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_1
def test_causal_mask_is_lower_triangular() -> None:
    try:
        mask = starter.causal_mask(4)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(mask.shape) == (4, 4)
    assert torch.isfinite(torch.diag(mask)).all()
    assert mask[0, 3] < 0

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_1
def test_causal_mask_diagonal_is_zero() -> None:
    try:
        mask = starter.causal_mask(5)
    except Exception as exc:
        skip_not_implemented(exc)
    assert torch.allclose(torch.diag(mask), torch.zeros(5))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_1
def test_causal_mask_upper_triangle_is_neg_inf() -> None:
    try:
        mask = starter.causal_mask(3)
    except Exception as exc:
        skip_not_implemented(exc)
    assert mask[0, 1] == float('-inf')
    assert mask[0, 2] == float('-inf')
    assert mask[1, 2] == float('-inf')

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_1
def test_causal_mask_lower_triangle_is_zero() -> None:
    try:
        mask = starter.causal_mask(3)
    except Exception as exc:
        skip_not_implemented(exc)
    assert mask[1, 0] == 0.0
    assert mask[2, 0] == 0.0
    assert mask[2, 1] == 0.0

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_1
def test_causal_mask_size_one() -> None:
    try:
        mask = starter.causal_mask(1)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(mask.shape) == (1, 1)
    assert mask[0, 0] == 0.0
