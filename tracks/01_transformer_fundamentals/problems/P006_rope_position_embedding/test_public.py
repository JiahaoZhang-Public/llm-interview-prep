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
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(q_out.shape) == tuple(q.shape)
    assert tuple(k_out.shape) == tuple(k.shape)
