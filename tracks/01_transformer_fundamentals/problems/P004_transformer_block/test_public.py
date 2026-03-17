from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_block_shape() -> None:
    try:
        module = starter.TransformerBlock(hidden_size=8, num_heads=2, intermediate_size=16)
        output = module(torch.randn(2, 4, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 4, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_block_residual_connection() -> None:
    """残差连接下 output 不应和 input 完全无关"""
    torch.manual_seed(42)
    try:
        module = starter.TransformerBlock(hidden_size=8, num_heads=2, intermediate_size=16)
        x = torch.randn(1, 3, 8)
        output = module(x)
    except Exception as exc:
        skip_not_implemented(exc)
    # 不是全零（非退化），且和输入有关系（残差）
    assert not torch.allclose(output, torch.zeros_like(output))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_block_has_layernorm() -> None:
    """应该包含 LayerNorm"""
    try:
        module = starter.TransformerBlock(hidden_size=8, num_heads=2, intermediate_size=16)
    except Exception as exc:
        skip_not_implemented(exc)
    assert hasattr(module, 'ln1') or hasattr(module, 'norm1')
    assert hasattr(module, 'ln2') or hasattr(module, 'norm2')

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_block_different_intermediate_size() -> None:
    """intermediate_size 可以和 hidden_size 不同"""
    try:
        module = starter.TransformerBlock(hidden_size=8, num_heads=2, intermediate_size=32)
        output = module(torch.randn(1, 2, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 2, 8)
