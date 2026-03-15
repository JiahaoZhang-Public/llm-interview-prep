from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.practice
@pytest.mark.agent
@pytest.mark.difficulty_2
def test_parse_function_schema_extracts_required_fields() -> None:
    schema = {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    try:
        parsed = starter.parse_function_schema(schema)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    assert parsed["name"] == "search"
    assert "query" in parsed["required"]
