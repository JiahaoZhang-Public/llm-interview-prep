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
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 4, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_multi_head_attention_single_head() -> None:
    """num_heads=1 应该等价于单头"""
    try:
        module = starter.MultiHeadAttention(hidden_size=8, num_heads=1)
        output = module(torch.randn(1, 3, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 3, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_multi_head_attention_with_mask() -> None:
    """带 mask 时不应报错"""
    try:
        module = starter.MultiHeadAttention(hidden_size=8, num_heads=2)
        mask = torch.zeros(4, 4)
        mask[0, 1:] = float('-inf')
        output = module(torch.randn(1, 4, 8), mask=mask)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 4, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_multi_head_attention_has_projections() -> None:
    """应该包含 Q/K/V/O 四组投影"""
    try:
        module = starter.MultiHeadAttention(hidden_size=16, num_heads=4)
    except Exception as exc:
        skip_not_implemented(exc)
    assert hasattr(module, 'q_proj')
    assert hasattr(module, 'out_proj')
