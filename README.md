# LLM Interview Prep Repo

`main` is the technical hub branch for this repository. It contains the shared Python package, tests, scaffolding tools, and sync rules for the bilingual practice branches.

## Branches

| Branch | Purpose |
| --- | --- |
| `codex/zh-cn` | Chinese practice edition |
| `codex/en-us` | English practice edition |
| `codex/zh-cn-solutions` | Chinese solutions branch |
| `codex/en-us-solutions` | English solutions branch |

## Sync Rules

1. Code, interfaces, and public tests land on `main` first.
2. User-facing study content is localized on `codex/zh-cn` and `codex/en-us`.
3. Solutions branches only inherit from their paired language branches.
4. `starter.py`, `test_public.py`, and `src/llm_prep/**` must stay identical across all study branches.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
make smoke
```

## Regeneration

Use the scaffold script when you need to regenerate localized study content:

```bash
python scripts/scaffold_repo.py --locale main
python scripts/scaffold_repo.py --locale zh-cn
python scripts/scaffold_repo.py --locale en-us
```
