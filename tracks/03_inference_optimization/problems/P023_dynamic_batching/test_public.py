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
    except Exception as exc:
        skip_not_implemented(exc)
    assert [item["id"] for item in batch] == ["a", "b"]

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_dynamic_batcher_respects_max_size() -> None:
    batcher = starter.DynamicBatcher(max_batch_size=2)
    try:
        for i in range(5):
            batcher.enqueue({"id": i})
        batch = batcher.flush()
    except Exception as exc:
        skip_not_implemented(exc)
    assert len(batch) == 2
    assert batch[0]["id"] == 0

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_dynamic_batcher_empty_flush() -> None:
    batcher = starter.DynamicBatcher(max_batch_size=4)
    try:
        batch = batcher.flush()
    except Exception as exc:
        skip_not_implemented(exc)
    assert batch == []

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_dynamic_batcher_multiple_flushes() -> None:
    batcher = starter.DynamicBatcher(max_batch_size=2)
    try:
        for i in range(5):
            batcher.enqueue({"id": i})
        batch1 = batcher.flush()
        batch2 = batcher.flush()
        batch3 = batcher.flush()
    except Exception as exc:
        skip_not_implemented(exc)
    assert [r["id"] for r in batch1] == [0, 1]
    assert [r["id"] for r in batch2] == [2, 3]
    assert [r["id"] for r in batch3] == [4]

@pytest.mark.practice
@pytest.mark.inference
@pytest.mark.difficulty_3
def test_dynamic_batcher_enqueue_after_flush() -> None:
    batcher = starter.DynamicBatcher(max_batch_size=10)
    try:
        batcher.enqueue("first")
        batch1 = batcher.flush()
        batcher.enqueue("second")
        batch2 = batcher.flush()
    except Exception as exc:
        skip_not_implemented(exc)
    assert batch1 == ["first"]
    assert batch2 == ["second"]
