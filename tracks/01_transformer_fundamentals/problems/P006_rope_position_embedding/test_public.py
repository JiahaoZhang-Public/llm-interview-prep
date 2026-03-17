from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_apply_rope_preserves_shape() -> None:
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 3, 8)
    try:
        q_out, k_out = starter.apply_rope(q, k)
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(q_out.shape) == tuple(q.shape)
    assert tuple(k_out.shape) == tuple(k.shape)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_apply_rope_position_zero_is_identity() -> None:
    """位置 0 时旋转角度为 0，cos=1 sin=0，应近似 identity"""
    q = torch.ones(1, 1, 4)
    k = torch.ones(1, 1, 4)
    try:
        q_out, k_out = starter.apply_rope(q, k, position_ids=torch.tensor([0]))
    except Exception as exc:
        skip_not_implemented(exc)
    # 位置0：angle=0, cos=1, sin=0 → 旋转不变
    assert torch.allclose(q_out, q, atol=1e-5)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_apply_rope_different_positions_give_different_results() -> None:
    """不同位置应该给出不同的旋转结果"""
    q = torch.ones(1, 2, 4)
    k = torch.ones(1, 2, 4)
    try:
        q_out, k_out = starter.apply_rope(q, k)
    except Exception as exc:
        skip_not_implemented(exc)
    # 位置 0 和位置 1 的旋转结果应不同
    assert not torch.allclose(q_out[0, 0], q_out[0, 1])

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_3
def test_apply_rope_preserves_norm_approximately() -> None:
    """旋转应大致保持向量范数"""
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    try:
        q_out, k_out = starter.apply_rope(q, k)
    except Exception as exc:
        skip_not_implemented(exc)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-4)
