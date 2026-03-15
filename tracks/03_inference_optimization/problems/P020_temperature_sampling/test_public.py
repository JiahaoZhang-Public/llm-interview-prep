import random
from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_1
def test_temperature_sample_returns_valid_index() -> None:
    random.seed(0)
    try:
        token_id = starter.temperature_sample([1.0, 2.0, 3.0], temperature=0.7)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert token_id in {0, 1, 2}
