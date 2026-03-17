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


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_scaled_dot_product_attention_no_mask() -> None:
    """不带 mask 时所有 key 都应参与"""
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    try:
        output = starter.scaled_dot_product_attention(q, k, v)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(output.shape) == (1, 1, 2)
    # 两个 key 都有权重，output 应该是两个 v 的加权和
    assert output[0, 0, 0] > 0 and output[0, 0, 1] > 0


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_scaled_dot_product_attention_output_shape() -> None:
    """验证多 batch、多 query 的 shape"""
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    try:
        output = starter.scaled_dot_product_attention(q, k, v)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(output.shape) == (2, 3, 8)


@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_2
def test_scaled_dot_product_attention_weights_sum_to_one() -> None:
    """attention weights 应该归一化（间接验证 softmax 正确）"""
    q = torch.randn(1, 1, 4)
    k = torch.randn(1, 3, 4)
    v = torch.ones(1, 3, 4)  # 全 1 的 V
    try:
        output = starter.scaled_dot_product_attention(q, k, v)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    # 如果 weights 加和为 1，那 output = weights @ ones = 全 1
    assert torch.allclose(output, torch.ones(1, 1, 4), atol=1e-4)
