from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.systems
@pytest.mark.difficulty_4
def test_model_parallel_scheduler_returns_stage_plan() -> None:
    scheduler = starter.ModelParallelScheduler(shard_ids=["gpu0", "gpu1"])
    try:
        plan = scheduler.schedule("req-1", ["embed", "decode"])
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert len(plan) == 2
