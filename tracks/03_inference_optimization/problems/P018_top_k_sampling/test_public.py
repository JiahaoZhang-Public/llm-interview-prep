import random
from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
@pytest.mark.top10
def test_top_k_sample_only_returns_from_top_k() -> None:
    random.seed(0)
    try:
        token_id = starter.top_k_sample([0.1, 0.9, 0.8, 0.0], k=2)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert token_id in {1, 2}
