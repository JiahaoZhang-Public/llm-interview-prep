from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.smoke
def test_problem_count_is_stable() -> None:
    regular_problem_meta = list(Path("tracks").glob("0[1-7]_*/problems/*/meta.yaml"))
    mock_problem_meta = list(Path("tracks/90_mock_interviews/problems").glob("*/meta.yaml"))
    assert len(regular_problem_meta) == 50
    assert len(mock_problem_meta) == 5


@pytest.mark.smoke
def test_required_docs_exist() -> None:
    required = [
        Path("docs/how_to_practice.md"),
        Path("docs/interview_rounds.md"),
        Path("docs/high_frequency_top10.md"),
        Path("docs/roadmaps/zero_to_one_8week.md"),
        Path("docs/roadmaps/top10_7day_sprint.md"),
    ]
    for path in required:
        assert path.exists(), f"Missing required document: {path}"
