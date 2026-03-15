from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.systems
@pytest.mark.difficulty_2
def test_token_rate_limiter_respects_budget() -> None:
    limiter = starter.TokenRateLimiter(tokens_per_second=10.0, burst=10.0)
    try:
        first = limiter.allow(6, now=0.0)
        second = limiter.allow(6, now=0.0)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert first is True
    assert second is False
