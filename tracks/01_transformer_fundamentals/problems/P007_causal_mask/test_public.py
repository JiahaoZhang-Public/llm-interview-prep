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
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert tuple(mask.shape) == (4, 4)
    assert torch.isfinite(torch.diag(mask)).all()
    assert mask[0, 3] < 0
