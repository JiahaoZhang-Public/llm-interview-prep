PYTHON ?= python
PYTEST ?= pytest

.PHONY: install smoke test scaffold-main scaffold-zh scaffold-en count-problems

install:
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	$(PYTEST) -m smoke

test:
	$(PYTEST)

scaffold-main:
	$(PYTHON) scripts/scaffold_repo.py --locale main

scaffold-zh:
	$(PYTHON) scripts/scaffold_repo.py --locale zh-cn

scaffold-en:
	$(PYTHON) scripts/scaffold_repo.py --locale en-us

count-problems:
	$(PYTHON) -c "from pathlib import Path; print(len(list(Path('tracks').glob('*/problems/*/meta.yaml'))))"

