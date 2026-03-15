from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_local_starter(path: str | Path) -> ModuleType:
    starter_path = Path(path)
    spec = importlib.util.spec_from_file_location(f"starter_{starter_path.stem}", starter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import starter module from {starter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def skip_not_implemented(exc: Exception) -> None:
    if isinstance(exc, NotImplementedError):
        pytest.skip("Implement starter.py first, then rerun the public tests.")
    raise exc

