from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.tokenization
@pytest.mark.difficulty_1
def test_pad_sequences_returns_padded_batch_and_mask() -> None:
    try:
        padded, mask = starter.pad_sequences([[1, 2, 3], [4]], pad_token_id=0)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(padded) == 2
    assert len(mask[0]) == 3
    assert mask[1][-1] == 0
