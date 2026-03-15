from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
_ = fastapi

from fastapi.testclient import TestClient
from llm_prep.utils.testing import load_local_starter, skip_not_implemented

starter = load_local_starter(Path(__file__).with_name("starter.py"))


def generator(prompt):
    return f"generated::{prompt}"


@pytest.mark.practice
@pytest.mark.systems
@pytest.mark.difficulty_3
def test_create_app_exposes_generate_endpoint() -> None:
    try:
        app = starter.create_app(generator)
    except Exception as exc:  # pragma: no cover
        skip_not_implemented(exc)
    client = TestClient(app)
    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 200
    assert response.json()["generated_text"] == "generated::hello"
