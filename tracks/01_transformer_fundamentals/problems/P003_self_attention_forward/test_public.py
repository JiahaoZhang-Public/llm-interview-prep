from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_self_attention_forward_shape() -> None:
    try:
        module = starter.SelfAttention(hidden_size=8)
        output = module(torch.randn(2, 4, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 4, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_self_attention_single_token() -> None:
    """单 token 序列 attention 应正常工作"""
    try:
        module = starter.SelfAttention(hidden_size=8)
        output = module(torch.randn(1, 1, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 1, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_self_attention_has_three_projections() -> None:
    """应该有 Q/K/V 三个独立的投影"""
    try:
        module = starter.SelfAttention(hidden_size=16)
    except Exception as exc:
        skip_not_implemented(exc)
    assert hasattr(module, 'q_proj')
    assert hasattr(module, 'k_proj')
    assert hasattr(module, 'v_proj')

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_self_attention_with_mask() -> None:
    try:
        module = starter.SelfAttention(hidden_size=8)
        mask = torch.zeros(3, 3)
        mask = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)
        output = module(torch.randn(1, 3, 8), mask=mask)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 3, 8)
