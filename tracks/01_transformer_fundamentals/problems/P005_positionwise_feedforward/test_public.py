from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_positionwise_feedforward_shape() -> None:
    try:
        module = starter.PositionwiseFeedForward(hidden_size=8, intermediate_size=16)
        output = module(torch.randn(2, 4, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 4, 8)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_positionwise_feedforward_expansion() -> None:
    """intermediate 可以是 hidden 的任意倍数"""
    try:
        module = starter.PositionwiseFeedForward(hidden_size=4, intermediate_size=32)
        output = module(torch.randn(1, 2, 4))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 2, 4)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_positionwise_feedforward_nonlinearity() -> None:
    """FFN 应包含非线性激活"""
    try:
        module = starter.PositionwiseFeedForward(hidden_size=8, intermediate_size=16)
    except Exception as exc:
        skip_not_implemented(exc)
    assert hasattr(module, 'relu') or hasattr(module, 'activation') or hasattr(module, 'act')

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_positionwise_feedforward_single_token() -> None:
    try:
        module = starter.PositionwiseFeedForward(hidden_size=8, intermediate_size=16)
        output = module(torch.randn(1, 1, 8))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 1, 8)
