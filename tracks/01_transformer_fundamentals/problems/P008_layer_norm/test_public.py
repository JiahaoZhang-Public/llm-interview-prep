from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_layer_norm_normalizes_last_dimension() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = torch.ones(3)
    beta = torch.zeros(3)
    try:
        out = starter.layer_norm(x, gamma, beta)
    except Exception as exc:
        skip_not_implemented(exc)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(1), atol=1e-4)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_layer_norm_unit_variance() -> None:
    """gamma=1, beta=0 时输出方差应接近 1"""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    gamma = torch.ones(5)
    beta = torch.zeros(5)
    try:
        out = starter.layer_norm(x, gamma, beta)
    except Exception as exc:
        skip_not_implemented(exc)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(1), atol=1e-3)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_layer_norm_affine_transform() -> None:
    """gamma 和 beta 应该正确应用"""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = torch.tensor([2.0, 2.0, 2.0])
    beta = torch.tensor([1.0, 1.0, 1.0])
    try:
        out = starter.layer_norm(x, gamma, beta)
    except Exception as exc:
        skip_not_implemented(exc)
    # 归一化后 scale by 2 + shift by 1，mean 应为 1.0
    assert torch.allclose(out.mean(dim=-1), torch.tensor([1.0]), atol=1e-4)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_layer_norm_matches_pytorch() -> None:
    """和 PyTorch 内置 LayerNorm 结果对齐"""
    x = torch.randn(2, 4, 8)
    gamma = torch.randn(8)
    beta = torch.randn(8)
    try:
        our = starter.layer_norm(x, gamma, beta)
    except Exception as exc:
        skip_not_implemented(exc)
    ln = torch.nn.LayerNorm(8, elementwise_affine=False)
    expected = gamma * ln(x) + beta
    assert torch.allclose(our, expected, atol=1e-4)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_layer_norm_preserves_shape() -> None:
    x = torch.randn(3, 5, 16)
    gamma = torch.ones(16)
    beta = torch.zeros(16)
    try:
        out = starter.layer_norm(x, gamma, beta)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(out.shape) == (3, 5, 16)
