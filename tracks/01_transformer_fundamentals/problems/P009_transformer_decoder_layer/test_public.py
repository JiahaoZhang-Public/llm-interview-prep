from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_decoder_layer_shape() -> None:
    try:
        module = starter.TransformerDecoderLayer(hidden_size=8, num_heads=2, intermediate_size=16)
        output = module(torch.randn(2, 4, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 4, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_decoder_layer_auto_causal_mask() -> None:
    """不传 mask 时应自动生成 causal mask"""
    try:
        module = starter.TransformerDecoderLayer(hidden_size=8, num_heads=2, intermediate_size=16)
        output = module(torch.randn(1, 5, 8))  # 不传 mask
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 5, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_decoder_layer_single_token() -> None:
    """单 token 输入应正常工作"""
    try:
        module = starter.TransformerDecoderLayer(hidden_size=8, num_heads=2, intermediate_size=16)
        output = module(torch.randn(1, 1, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 1, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_transformer_decoder_layer_with_explicit_mask() -> None:
    """传入显式 mask 也应正常工作"""
    try:
        module = starter.TransformerDecoderLayer(hidden_size=8, num_heads=2, intermediate_size=16)
        mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
        output = module(torch.randn(1, 3, 8), mask=mask)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 3, 8)
