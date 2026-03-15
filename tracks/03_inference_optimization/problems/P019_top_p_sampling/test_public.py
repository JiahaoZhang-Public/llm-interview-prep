import random
from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_2
@pytest.mark.top10
def test_top_p_sample_returns_valid_index() -> None:
    random.seed(0)
    try:
        token_id = starter.top_p_sample([3.0, 2.0, 0.1], p=0.8)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}
