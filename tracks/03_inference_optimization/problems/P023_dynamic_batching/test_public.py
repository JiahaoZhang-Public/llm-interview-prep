from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_dynamic_batcher_flushes_in_enqueue_order() -> None:
    batcher = starter.DynamicBatcher(max_batch_size=4)
    try:
        batcher.enqueue({"id": "a"})
        batcher.enqueue({"id": "b"})
        batch = batcher.flush()
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert [item["id"] for item in batch] == ["a", "b"]
