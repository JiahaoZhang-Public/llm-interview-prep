from pathlib import Path

import pytest

from llm_prep.utils.testing import load_local_starter

starter = load_local_starter(Path(__file__).with_name("starter.py"))


@pytest.mark.smoke
@pytest.mark.mock
def test_training_finetuning_round_pack_shape() -> None:
    pack = starter.get_mock_pack()
    assert pack["id"] == "M003"
    assert len(pack["problems"]) == 4
