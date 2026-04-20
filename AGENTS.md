# AGENTS.md — air-lab-os

Read this before any other file in the repo.

## What this is

A domain-agnostic self-improving engine. It runs patterns against
datasets, scores results, and promotes what works. It knows nothing
about any specific domain — fraud, marketing, credit risk are all
plugins. This repo contains zero domain-specific logic.

## Sibling repo layout

```
~/LocalDocuments/
  air-lab-os/            ← this repo
  bb_datasets/           ← example dataset plugin (read-only)
  external-datasets/     ← future dataset store (read-only, Phase 2)
```

## Rules — never break these

1. **Never modify `scoring_policy.yaml`.**
   It is the locked evaluation contract. `evaluator.py` reads it.
   Nothing else writes to it. If a weight should change, tell the human.

2. **`registry.json` is derived state.**
   Rebuilt from `memory/runs.json` on every startup. If they conflict,
   `runs.json` wins. Never write `registry.json` directly — only via
   `PatternRegistry.save()`.

3. **`memory/runs.json` is append-only.**
   Never delete or modify existing entries. Only append new run objects.

4. **The eval split is fixed.**
   `DatasetHandle.eval_df()` always returns the same slice. Never
   change the split logic inside a DatasetHandle implementation.

5. **The engine is domain-agnostic.**
   No domain-specific imports, logic, or hardcoded column names belong
   in `runtime/`, `lab/`, `evaluation/`, or `memory/`. Domain logic
   lives in pattern files and DatasetHandle implementations only.

6. **Run tests before every commit.**
   `uv run pytest tests/ -q` must pass with zero failures.

7. **One commit per kept experiment.**
   `git commit -m "exp: <pattern_name> score=<score> tier=<tier>"`
   Discarded experiments: `git reset --soft HEAD~1`

8. **Never install packages outside pyproject.toml.**
   Stop and ask the human if a new dependency is needed.

## File ownership

| File/dir              | Owner      | Codex may…              |
|-----------------------|------------|-------------------------------|
| scoring_policy.yaml   | Human      | Read only                     |
| AGENTS.md             | Human      | Read only                     |
| SPEC.md               | Human      | Read only                     |
| memory/runs.json      | System     | Append only                   |
| registry.json         | System     | Overwrite via registry.save() |
| patterns/scratch/     | Codex| Read + write                  |
| patterns/working/     | Codex| Read + write                  |
| patterns/stable/      | Codex| Read + write                  |
| lab/experiments/      | Codex| Write (run logs)              |
| runtime/**            | Codex| Read + write                  |
| evaluation/**         | Codex| Read + write                  |
| datasets/**           | Codex| Read + write                  |
| tests/**              | Codex| Read + write                  |

## How to run

```bash
# Run one experiment
uv run python main.py --pattern <name> --dataset <dataset_id>

# Run the full autonomous loop
uv run python main.py --loop --dataset <dataset_id>

# Run tests
uv run pytest tests/ -q
```

## Session startup checklist

1. Read AGENTS.md
2. Read SPEC.md
3. `uv run pytest tests/ -q` — confirm baseline passes
4. Read `memory/runs.json` for experiment history
5. Read `registry.json` for current pattern state
