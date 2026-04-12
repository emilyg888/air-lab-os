# SPEC.md — air-lab-os: Phase 1 Foundation

> **For Claude Code.** Read this file in full before writing any code.
> Build everything in this spec, in the order listed. Do not proceed to
> the next section until the current section's deliverables pass.
>
> air-lab-os is a **domain-agnostic self-improving engine**. It knows
> nothing about fraud, marketing, or any other domain. It knows how to:
> run a pattern against a dataset, score the result, compare patterns,
> and track what improved. Use cases are plugins. This file contains
> zero domain-specific logic — any domain reference is an example only.

---

## What air-lab-os is

A compounding experiment loop. At startup you plug in:

1. A **dataset** (anything implementing `DatasetHandle`)
2. A **set of patterns** (anything implementing `PatternHandler`)
3. A **scoring policy** (a locked YAML file)

The engine runs the loop autonomously:

```
QWEN plans → playground runs → evaluator scores →
arena compares → registry updates → repeat
```

Patterns that improve move up through `scratch → working → stable`.
The engine never knows what domain it is operating in.

---

## Repo layout

The repo is already scaffolded. Fill in the empty files exactly as
specified. Do not create new top-level directories.

```
air-lab-os/
  CLAUDE.md                        ← standing instructions (write this)
  SPEC.md                          ← this file
  main.py                          ← engine entry point
  scoring_policy.yaml              ← locked weights (write this)
  runtime/
    __init__.py
    config.py                      ← ScoringPolicy loader
    llm.py                         ← QWEN plan layer
  lab/
    __init__.py
    playground.py                  ← run_experiment()
    arena.py                       ← compare_patterns()
    experiments/                   ← per-run JSON logs (git-ignored)
  evaluation/
    __init__.py
    evaluator.py                   ← evaluate() — locked scoring layer
  memory/
    __init__.py
    registry.py                    ← PatternRegistry
    runs.json                      ← append-only run log
  patterns/
    __init__.py
    base.py                        ← PatternHandler ABC + RunResult
    scratch/                       ← tier 1: exploration patterns
    working/                       ← tier 2: improving patterns
    stable/                        ← tier 3: promoted patterns
  datasets/
    __init__.py
    base.py                        ← DatasetHandle ABC + DatasetMeta
  tests/
    __init__.py
    test_evaluator.py
    test_registry.py
    test_playground.py
    test_arena.py
```

Sibling repos (read-only, never modify):

```
../bb_datasets/                    ← example dataset provider
../external-datasets/              ← future dataset store (Phase 2)
```

---

## CLAUDE.md

Write this content exactly:

```markdown
# CLAUDE.md — air-lab-os

Read this before any other file in the repo.

## What this is

A domain-agnostic self-improving engine. It runs patterns against
datasets, scores results, and promotes what works. It knows nothing
about any specific domain — fraud, marketing, credit risk are all
plugins. This repo contains zero domain-specific logic.

## Sibling repo layout
```

~/LocalDocuments/
air-lab-os/ ← this repo
bb_datasets/ ← example dataset plugin (read-only)
external-datasets/ ← future dataset store (read-only, Phase 2)

````

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

| File/dir              | Owner      | Claude Code may…              |
|-----------------------|------------|-------------------------------|
| scoring_policy.yaml   | Human      | Read only                     |
| CLAUDE.md             | Human      | Read only                     |
| SPEC.md               | Human      | Read only                     |
| memory/runs.json      | System     | Append only                   |
| registry.json         | System     | Overwrite via registry.save() |
| patterns/scratch/     | Claude Code| Read + write                  |
| patterns/working/     | Claude Code| Read + write                  |
| patterns/stable/      | Claude Code| Read + write                  |
| lab/experiments/      | Claude Code| Write (run logs)              |
| runtime/**            | Claude Code| Read + write                  |
| evaluation/**         | Claude Code| Read + write                  |
| datasets/**           | Claude Code| Read + write                  |
| tests/**              | Claude Code| Read + write                  |

## How to run

```bash
# Run one experiment
uv run python main.py --pattern <name> --dataset <dataset_id>

# Run the full autonomous loop
uv run python main.py --loop --dataset <dataset_id>

# Run tests
uv run pytest tests/ -q
````

## Session startup checklist

1. Read CLAUDE.md
2. Read SPEC.md
3. `uv run pytest tests/ -q` — confirm baseline passes
4. Read `memory/runs.json` for experiment history
5. Read `registry.json` for current pattern state

````

---

## scoring_policy.yaml

Write this content exactly. Do not modify during the build:

```yaml
# air-lab-os scoring policy
# Locked — human-owned. Do not modify.
# Weights must sum to 1.0.
# This is the DEFAULT policy. Use-case plugins may provide their own.

version: "1.0"

weights:
  primary_metric:  0.40   # domain metric (f1, accuracy, etc) — provided by RunResult
  explainability:  0.25   # fraction of positive flags with non-empty explanation
  latency:         0.20   # inverse-scaled: 0ms=1.0, max_ms=0.0, linear
  cost:            0.15   # inverse-scaled: $0=1.0, max_per_1k=0.0, linear

promotion:
  working_threshold: 0.65  # min score to reach working tier (was: silver)
  stable_threshold:  0.78  # min score to become promotion candidate (was: gold)
  min_runs:          3     # min experiments before promotion eligible

latency:
  max_ms: 500              # latency at which latency_score = 0.0

cost:
  max_per_1k: 0.10         # USD per 1k rows at which cost_score = 0.0
````

---

## pyproject.toml

```toml
[project]
name = "air-lab-os"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "pyyaml>=6.0",
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
pythonpath = ["."]
```

---

## Component 1: Plugin contracts (ABCs)

These two abstract base classes are the entire interface between the
engine and any use case. The engine only ever calls these methods.

### `datasets/base.py`

```python
"""
DatasetHandle — abstract contract for all dataset plugins.

Every dataset a use case provides must implement this interface.
The engine calls these methods and never touches raw data files.

The eval split MUST be fixed and deterministic — same rows every
call, regardless of when the dataset was generated or loaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetMeta:
    """
    Metadata every dataset must declare about itself.

    The engine uses this to log experiments correctly and to
    determine which scoring metric to treat as the primary metric.
    """
    name: str                        # unique dataset id, e.g. "bb_fraud_v3"
    domain: str                      # use-case domain, e.g. "fraud"
    tier: str                        # "bronze" | "silver" | "gold" | "test"
    version: str                     # e.g. "1.0", "2.0"
    label_column: str                # ground truth column in eval_df
    primary_metric: str              # metric name the evaluator treats as primary
                                     # e.g. "f1_score", "accuracy", "conversion_rate"
    row_count: int                   # total rows in the full dataset
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # domain-specific metadata


class DatasetHandle(ABC):
    """
    Abstract interface for all dataset plugins.

    Implementations live in use-case repos (e.g. bb_datasets/fraud/load.py).
    The engine imports a DatasetHandle — never raw data files.

    The 80/20 train/eval split is fixed inside each implementation.
    Never expose the split ratio as a parameter — it must be immutable.
    """

    @property
    @abstractmethod
    def meta(self) -> DatasetMeta:
        """Return dataset metadata. Called once at startup for logging."""
        ...

    @abstractmethod
    def eval_df(self) -> pd.DataFrame:
        """
        Return the fixed evaluation slice (last 20% sorted by primary key).

        This slice is immutable — same rows every call. The engine scores
        all patterns against this slice so results are comparable.
        """
        ...

    @abstractmethod
    def train_df(self) -> pd.DataFrame:
        """
        Return the fixed training slice (first 80% sorted by primary key).

        Patterns that need fitting (e.g. ML models) call this internally.
        The engine does not call train_df directly.
        """
        ...

    @abstractmethod
    def labels(self) -> list[bool]:
        """
        Ground truth labels for the eval slice.

        Must be the same length as eval_df() and in the same row order.
        Derived from eval_df()[meta.label_column].astype(bool).
        """
        ...

    def summary(self) -> dict:
        """Optional: return a summary dict for dashboard display."""
        return {
            "name":           self.meta.name,
            "domain":         self.meta.domain,
            "tier":           self.meta.tier,
            "rows":           self.meta.row_count,
            "label_column":   self.meta.label_column,
            "primary_metric": self.meta.primary_metric,
        }
```

### `patterns/base.py`

```python
"""
PatternHandler — abstract contract for all pattern plugins.

Every pattern the engine runs must implement this interface.
The engine calls detect() and never imports domain-specific code.

Pattern files live in patterns/scratch/, patterns/working/, or
patterns/stable/ depending on their current promotion tier.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from datasets.base import DatasetHandle


@dataclass
class RunResult:
    """
    Standardised output of every pattern run.

    All three lists must be the same length as the eval DataFrame.

    flags       — True if the row is a positive detection
    scores      — float in [0.0, 1.0], confidence or probability
    explanation — human-readable reason string, "" if none
    latency_ms  — wall-clock ms for the full detect() call (set by base)
    cost_per_1k — estimated USD per 1000 rows (0.0 for local patterns)
    primary_metric_value — the domain metric for this run (e.g. f1 score).
                           Set by the engine after scoring, not by the pattern.
    """
    flags: list[bool]
    scores: list[float]
    explanation: list[str]
    latency_ms: float = 0.0
    cost_per_1k: float = 0.0
    primary_metric_value: float = 0.0
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.flags)
        assert len(self.scores) == n, \
            f"scores length {len(self.scores)} must match flags length {n}"
        assert len(self.explanation) == n, \
            f"explanation length {len(self.explanation)} must match flags length {n}"
        assert all(0.0 <= s <= 1.0 for s in self.scores), \
            "all scores must be in [0.0, 1.0]"


class PatternHandler(ABC):
    """
    Abstract base for all pattern implementations.

    Subclasses implement run(). The base class wraps run() to record
    wall-clock latency automatically via detect().

    Pattern files implementing this class live in:
      patterns/scratch/   — exploration, unproven
      patterns/working/   — improving, above working_threshold
      patterns/stable/    — promoted, certified on gold dataset

    Usage:
        handle  = MyDatasetHandle()
        pattern = MyPattern(threshold=0.5)
        result  = pattern.detect(handle)
    """

    name: str = "base"                   # unique pattern identifier
    version: str = "0.1"                 # pattern version string

    def detect(self, handle: DatasetHandle) -> RunResult:
        """
        Public entry point. Times the call, injects latency_ms.
        Do NOT override — override run() instead.
        """
        t0 = time.perf_counter()
        result = self.run(handle)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, handle: DatasetHandle) -> RunResult:
        """
        Implement pattern logic here.

        Args:
            handle: DatasetHandle providing eval_df(), train_df(), labels().
                    The pattern must not assume any specific columns exist
                    beyond what DatasetHandle.meta declares.

        Returns:
            RunResult with flags, scores, explanation the same length as
            handle.eval_df(). Do NOT set latency_ms — base class does it.
        """
        ...

    def describe(self) -> dict:
        """
        Return hyperparameter dict for run logging.
        Override to include pattern-specific config.
        """
        return {"pattern": self.name, "version": self.version}
```

---

## Component 2: Evaluation engine

### `evaluation/evaluator.py`

```python
"""
Evaluator — locked scoring layer.

evaluate(result, handle, policy) → EvalMetrics

Reads weights from scoring_policy.yaml via ScoringPolicy.
Never hardcodes metric names or weights.
The primary_metric value comes from RunResult — set by the domain
pattern, not by the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from datasets.base import DatasetHandle
from patterns.base import RunResult


POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


@dataclass
class ScoringPolicy:
    primary_metric_weight:  float
    explainability_weight:  float
    latency_weight:         float
    cost_weight:            float
    latency_max_ms:         float
    cost_max_per_1k:        float
    working_threshold:      float
    stable_threshold:       float
    min_runs:               int

    def validate(self) -> None:
        total = (
            self.primary_metric_weight
            + self.explainability_weight
            + self.latency_weight
            + self.cost_weight
        )
        assert abs(total - 1.0) < 1e-6, (
            f"Weights must sum to 1.0, got {total:.6f}. "
            f"Check scoring_policy.yaml."
        )


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        primary_metric_weight = w["primary_metric"],
        explainability_weight  = w["explainability"],
        latency_weight         = w["latency"],
        cost_weight            = w["cost"],
        latency_max_ms         = raw["latency"]["max_ms"],
        cost_max_per_1k        = raw["cost"]["max_per_1k"],
        working_threshold      = raw["promotion"]["working_threshold"],
        stable_threshold       = raw["promotion"]["stable_threshold"],
        min_runs               = raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


@dataclass
class EvalMetrics:
    """
    Full scoring breakdown for one experiment run.
    `score` is the weighted composite used for registry promotion.
    `primary_metric_value` is domain-specific (e.g. f1, accuracy).
    """
    primary_metric_value:  float    # from RunResult (domain sets this)
    primary_metric_score:  float    # normalised [0,1] — same value here
    explainability_score:  float    # fraction of flags with explanation
    latency_score:         float    # 1.0 = instant, 0.0 = max_ms
    cost_score:            float    # 1.0 = free, 0.0 = max cost
    score:                 float    # weighted composite [0, 1]
    extra: dict = None              # domain-specific metrics from RunResult


def evaluate(
    result: RunResult,
    handle: DatasetHandle,
    policy: ScoringPolicy | None = None,
) -> EvalMetrics:
    """
    Score a RunResult against ground truth labels.

    Args:
        result: output of PatternHandler.detect()
        handle: DatasetHandle providing labels() for ground truth
        policy: loaded ScoringPolicy; loads from yaml if None

    Returns:
        EvalMetrics with per-dimension scores and weighted composite
    """
    if policy is None:
        policy = load_policy()

    labels = handle.labels()
    assert len(result.flags) == len(labels), (
        f"RunResult has {len(result.flags)} flags "
        f"but dataset has {len(labels)} labels"
    )

    # --- Primary metric ---
    # The pattern sets primary_metric_value on RunResult.
    # The evaluator treats it as a [0, 1] score directly.
    # Domain patterns are responsible for computing their own metric
    # (e.g. f1_score) and setting it before returning RunResult.
    primary_score = float(result.primary_metric_value)
    primary_score = max(0.0, min(1.0, primary_score))

    # --- Explainability ---
    # Fraction of flagged rows that have a non-empty explanation string.
    # If nothing is flagged, score is 1.0 (nothing to explain).
    flagged = [i for i, f in enumerate(result.flags) if f]
    if flagged:
        explained    = sum(1 for i in flagged if result.explanation[i].strip())
        expl_score   = explained / len(flagged)
    else:
        expl_score   = 1.0

    # --- Latency ---
    latency_score = max(0.0, 1.0 - result.latency_ms / policy.latency_max_ms)

    # --- Cost ---
    cost_score = max(0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k)

    # --- Weighted composite ---
    composite = (
        policy.primary_metric_weight  * primary_score
        + policy.explainability_weight * expl_score
        + policy.latency_weight        * latency_score
        + policy.cost_weight           * cost_score
    )

    return EvalMetrics(
        primary_metric_value  = round(primary_score, 4),
        primary_metric_score  = round(primary_score, 4),
        explainability_score  = round(expl_score, 4),
        latency_score         = round(latency_score, 4),
        cost_score            = round(cost_score, 4),
        score                 = round(composite, 4),
        extra                 = result.extra_metrics or {},
    )
```

---

## Component 3: Pattern registry

### `memory/registry.py`

```python
"""
PatternRegistry — live system state rebuilt from memory/runs.json.

Two concepts are deliberately separated:

  status — outcome of the most recent run: "keep" | "discard" | "crash"
  tier   — promotion level of the pattern:  "scratch" | "working" | "stable"

Status changes every run. Tier only changes on promotion.
Never write registry.json directly — always use PatternRegistry.save().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


RUNS_PATH     = Path(__file__).parent / "runs.json"
REGISTRY_PATH = Path(__file__).parent.parent / "registry.json"


@dataclass
class RegistryEntry:
    # Identity
    pattern:      str              # unique pattern name
    domain:       str              # use-case domain (set by plugin)
    version:      str              # pattern version string
    pattern_path: str              # relative path to pattern file

    # Tier — promotion level, matches directory name
    tier:         str              # "scratch" | "working" | "stable"

    # Run outcome of the last experiment
    last_status:  str              # "keep" | "discard" | "crash"

    # Performance tracking
    confidence:   float            # best composite score across all keep runs
    runs:         int              # total experiment attempts
    last_commit:  str
    last_updated: int              # unix timestamp
    last_dataset: str              # dataset_id of the best scoring run

    # Promotion
    promotion_candidate: bool      # True if stable_threshold met and runs >= min_runs

    # Best run metrics — domain-agnostic dict
    best_metrics: dict[str, Any] = field(default_factory=dict)
    # e.g. {"f1": 0.82, "precision": 0.87, "recall": 0.76}  ← fraud domain
    # e.g. {"conversion_rate": 0.14, "roi": 2.8}             ← marketing domain

    description: str = ""


class PatternRegistry:
    """
    Rebuilt from runs.json on every startup.
    registry.json is derived — never authoritative.

    Usage:
        registry = PatternRegistry.load()
        entry    = registry.get("my_pattern")
        registry.save()
    """

    # Thresholds — loaded from scoring policy on PatternRegistry.load()
    WORKING_THRESHOLD: float = 0.65
    STABLE_THRESHOLD:  float = 0.78
    MIN_RUNS:          int   = 3

    def __init__(self, entries: dict[str, RegistryEntry]):
        self._entries = entries

    @classmethod
    def load(
        cls,
        runs_path:     Path = RUNS_PATH,
        registry_path: Path = REGISTRY_PATH,
    ) -> "PatternRegistry":
        """
        Rebuild registry from runs.json.
        Stable (gold) tier statuses are preserved from registry.json
        across rebuilds — they represent human promotion decisions.
        """
        # Load promotion thresholds from scoring policy
        try:
            from runtime.config import load_policy
            policy = load_policy()
            cls.WORKING_THRESHOLD = policy.working_threshold
            cls.STABLE_THRESHOLD  = policy.stable_threshold
            cls.MIN_RUNS          = policy.min_runs
        except Exception:
            pass  # use class defaults if policy unavailable

        # Preserve stable tier from previous registry.json
        # Stable = human-approved. Never downgrade automatically.
        stable_patterns: set[str] = set()
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                stable_patterns = {
                    k for k, v in data.items()
                    if isinstance(v, dict) and v.get("tier") == "stable"
                }
            except Exception:
                pass

        entries: dict[str, RegistryEntry] = {}

        if not runs_path.exists():
            return cls(entries)

        # runs.json is a list of run objects, newest last
        try:
            runs = json.loads(runs_path.read_text())
        except (json.JSONDecodeError, ValueError):
            runs = []

        for run in runs:
            pattern = run.get("pattern", "")
            score   = float(run.get("score", 0.0))
            status  = run.get("status", "discard")
            ts      = int(run.get("timestamp", 0))
            domain  = run.get("domain", "")
            dataset = run.get("dataset_id", "")

            if not pattern:
                continue

            if pattern not in entries:
                entries[pattern] = RegistryEntry(
                    pattern      = pattern,
                    domain       = domain,
                    version      = run.get("version", "0.1"),
                    pattern_path = run.get("pattern_path", ""),
                    tier         = "scratch",
                    last_status  = status,
                    confidence   = 0.0,
                    runs         = 0,
                    last_commit  = run.get("commit", ""),
                    last_updated = ts,
                    last_dataset = dataset,
                    promotion_candidate = False,
                    description  = run.get("description", ""),
                )

            e = entries[pattern]
            e.runs        += 1
            e.last_status  = status
            e.last_updated = max(e.last_updated, ts)
            e.last_commit  = run.get("commit", e.last_commit)
            e.last_dataset = dataset or e.last_dataset
            if domain:
                e.domain = domain

            if status == "keep" and score > e.confidence:
                e.confidence  = score
                e.best_metrics = run.get("metrics", {})
                e.description  = run.get("description", e.description)

        # Compute tiers
        for pattern, e in entries.items():
            if pattern in stable_patterns:
                e.tier = "stable"
            elif e.confidence >= cls.WORKING_THRESHOLD:
                e.tier = "working"
            else:
                e.tier = "scratch"

            e.promotion_candidate = (
                e.confidence >= cls.STABLE_THRESHOLD
                and e.runs >= cls.MIN_RUNS
                and e.tier != "stable"
            )

        return cls(entries)

    def get(self, pattern: str) -> Optional[RegistryEntry]:
        return self._entries.get(pattern)

    def all(self) -> list[RegistryEntry]:
        """All entries sorted by confidence descending."""
        return sorted(
            self._entries.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )

    def by_tier(self, tier: str) -> list[RegistryEntry]:
        """Filter entries by tier: 'scratch' | 'working' | 'stable'."""
        return [e for e in self._entries.values() if e.tier == tier]

    def best_score(self, pattern: str) -> Optional[float]:
        e = self._entries.get(pattern)
        return e.confidence if e else None

    def promotion_candidates(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.promotion_candidate]

    def save(self, path: Path = REGISTRY_PATH) -> None:
        """Write registry.json. Derived — safe to overwrite."""
        data = {p: asdict(e) for p, e in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))


def append_run(run: dict, path: Path = RUNS_PATH) -> None:
    """
    Append one run record to runs.json.
    Creates the file as an empty list if it does not exist.

    run must contain at minimum:
      pattern, domain, dataset_id, score, status, timestamp, commit
    """
    if path.exists():
        try:
            runs = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            runs = []
    else:
        runs = []

    runs.append(run)
    path.write_text(json.dumps(runs, indent=2))
```

---

## Component 4: Experiment runner

### `lab/playground.py`

```python
"""
Playground — run_experiment() is the core loop entry point.

run_experiment(pattern, handle, config) → ExperimentResult

The engine never knows what domain it is in. It calls
pattern.detect(handle), passes the result to evaluate(), and
records everything to runs.json.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets.base import DatasetHandle
from evaluation.evaluator import EvalMetrics, evaluate, load_policy
from memory.registry import PatternRegistry, RegistryEntry, append_run
from patterns.base import PatternHandler, RunResult

RUNS_PATH     = Path("memory/runs.json")
REGISTRY_PATH = Path("registry.json")


@dataclass
class ExperimentResult:
    """Full output of one experiment run. Written to runs.json."""
    pattern:     str
    domain:      str
    version:     str
    dataset_id:  str
    pattern_path: str
    config:      dict[str, Any]
    metrics:     EvalMetrics | None   # None on crash
    score:       float
    commit:      str
    status:      str                  # "keep" | "discard" | "crash"
    tier:        str                  # tier at time of run
    description: str
    timestamp:   int                  # unix seconds


def _short_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def run_experiment(
    pattern:       PatternHandler,
    handle:        DatasetHandle,
    description:   str = "",
    runs_path:     Path = RUNS_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> ExperimentResult:
    """
    Run one experiment end-to-end.

    Args:
        pattern:       PatternHandler instance to evaluate
        handle:        DatasetHandle providing data and labels
        description:   short label for the run log
        runs_path:     path to runs.json
        registry_path: path to registry.json

    Returns:
        ExperimentResult

    Side effects:
        Appends one record to runs.json.
        Rebuilds and saves registry.json.
    """
    commit    = _short_commit()
    timestamp = int(time.time())
    meta      = handle.meta

    # Determine current tier from registry
    registry = PatternRegistry.load(runs_path, registry_path)
    entry    = registry.get(pattern.name)
    tier     = entry.tier if entry else "scratch"

    try:
        # Run the pattern
        result: RunResult = pattern.detect(handle)

        # Score
        policy  = load_policy()
        metrics = evaluate(result, handle, policy)
        score   = metrics.score

        # Keep if this is the best score for this pattern
        current_best = registry.best_score(pattern.name)
        status = "keep" if (current_best is None or score > current_best) else "discard"

        exp = ExperimentResult(
            pattern      = pattern.name,
            domain       = meta.domain,
            version      = pattern.version,
            dataset_id   = meta.name,
            pattern_path = _find_pattern_path(pattern.name),
            config       = pattern.describe(),
            metrics      = metrics,
            score        = score,
            commit       = commit,
            status       = status,
            tier         = tier,
            description  = description or f"{pattern.name} v{pattern.version}",
            timestamp    = timestamp,
        )

    except Exception as exc:
        exp = ExperimentResult(
            pattern      = pattern.name,
            domain       = meta.domain,
            version      = getattr(pattern, "version", "?"),
            dataset_id   = meta.name,
            pattern_path = _find_pattern_path(pattern.name),
            config       = pattern.describe() if hasattr(pattern, "describe") else {},
            metrics      = None,
            score        = 0.0,
            commit       = commit,
            status       = "crash",
            tier         = tier,
            description  = f"CRASH: {type(exc).__name__}: {exc}",
            timestamp    = timestamp,
        )

    _write_run(exp, runs_path)
    registry = PatternRegistry.load(runs_path, registry_path)
    registry.save(registry_path)

    return exp


def _find_pattern_path(name: str) -> str:
    """Locate a pattern file across tier directories."""
    for tier_dir in ["patterns/scratch", "patterns/working", "patterns/stable"]:
        candidate = Path(tier_dir) / f"{name}.py"
        if candidate.exists():
            return str(candidate)
    return f"patterns/scratch/{name}.py"  # default for new patterns


def _write_run(exp: ExperimentResult, path: Path) -> None:
    """Serialise ExperimentResult and append to runs.json."""
    m = exp.metrics
    run = {
        "pattern":      exp.pattern,
        "domain":       exp.domain,
        "version":      exp.version,
        "dataset_id":   exp.dataset_id,
        "pattern_path": exp.pattern_path,
        "config":       exp.config,
        "score":        exp.score,
        "status":       exp.status,
        "tier":         exp.tier,
        "commit":       exp.commit,
        "timestamp":    exp.timestamp,
        "description":  exp.description,
        "metrics": {
            "primary_metric_value": m.primary_metric_value,
            "explainability_score": m.explainability_score,
            "latency_score":        m.latency_score,
            "cost_score":           m.cost_score,
            "score":                m.score,
            **(m.extra or {}),
        } if m else {},
    }
    append_run(run, path)
```

### `lab/arena.py`

```python
"""
Arena — compare_patterns() runs multiple patterns on the same dataset.

compare_patterns(patterns, handle) → ArenaResult

Returns a ranked list of patterns by composite score.
Used by the experiment loop to pick the best performer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets.base import DatasetHandle
from lab.playground import ExperimentResult, run_experiment
from patterns.base import PatternHandler

RUNS_PATH     = Path("memory/runs.json")
REGISTRY_PATH = Path("registry.json")


@dataclass
class ArenaResult:
    """Ranked output of a multi-pattern comparison run."""
    rankings:   list[ExperimentResult]   # sorted best → worst by score
    winner:     ExperimentResult | None  # highest scoring non-crash result
    dataset_id: str


def compare_patterns(
    patterns:      list[PatternHandler],
    handle:        DatasetHandle,
    description:   str = "arena run",
    runs_path:     Path = RUNS_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> ArenaResult:
    """
    Run all patterns against the same dataset and rank by score.

    Each pattern gets one experiment run. All results are logged
    to runs.json individually. The arena result surfaces the winner
    but does not automatically promote — promotion is the registry's job.

    Args:
        patterns:  list of PatternHandler instances to compare
        handle:    DatasetHandle — same dataset for all patterns
        description: label for all runs in this arena session

    Returns:
        ArenaResult with ranked list and winner
    """
    results: list[ExperimentResult] = []

    for pattern in patterns:
        result = run_experiment(
            pattern       = pattern,
            handle        = handle,
            description   = f"{description} | {pattern.name}",
            runs_path     = runs_path,
            registry_path = registry_path,
        )
        results.append(result)

    # Sort: non-crash results by score desc, crashes last
    ranked = sorted(
        results,
        key=lambda r: (r.status != "crash", r.score),
        reverse=True,
    )

    winner = next((r for r in ranked if r.status != "crash"), None)

    return ArenaResult(
        rankings   = ranked,
        winner     = winner,
        dataset_id = handle.meta.name,
    )
```

---

## Component 5: QWEN plan layer

### `runtime/config.py`

```python
"""ScoringPolicy loader — reads scoring_policy.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.evaluator import ScoringPolicy

POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        primary_metric_weight = w["primary_metric"],
        explainability_weight  = w["explainability"],
        latency_weight         = w["latency"],
        cost_weight            = w["cost"],
        latency_max_ms         = raw["latency"]["max_ms"],
        cost_max_per_1k        = raw["cost"]["max_per_1k"],
        working_threshold      = raw["promotion"]["working_threshold"],
        stable_threshold       = raw["promotion"]["stable_threshold"],
        min_runs               = raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy
```

### `runtime/llm.py`

````python
"""
QWEN plan layer — generates the next experiment queue.

plan(registry, runs, goal) → list[ExperimentPlan]

Calls a local QWEN model via Ollama (default: localhost:11434).
Returns a ranked list of experiments to run next.
Falls back to a rule-based ranker if QWEN is unavailable or
returns malformed JSON.

The planner is domain-agnostic. It reasons about patterns by
reading the registry and run history — it never imports domain code.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

from memory.registry import PatternRegistry, RegistryEntry


OLLAMA_URL   = "http://localhost:11434/api/generate"
QWEN_MODEL   = "qwen2.5:14b"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3


@dataclass
class ExperimentPlan:
    """One planned experiment, as returned by the QWEN planner."""
    pattern_name: str          # which pattern to run
    rationale:    str          # why QWEN chose this pattern
    config:       dict         # hyperparameter suggestions
    priority:     int          # 1 = highest priority


def plan(
    registry:   PatternRegistry,
    run_history: list[dict],
    goal:        str = "maximise composite score",
    n_plans:     int = 3,
) -> list[ExperimentPlan]:
    """
    Generate the next N experiments to run.

    Tries QWEN first. Falls back to rule-based ranker on any failure.

    Args:
        registry:    current PatternRegistry state
        run_history: recent run records from runs.json (last 20)
        goal:        natural language goal for this session
        n_plans:     how many experiments to plan

    Returns:
        list[ExperimentPlan] sorted by priority ascending (1 = run first)
    """
    try:
        return _qwen_plan(registry, run_history, goal, n_plans)
    except Exception:
        return _rule_based_plan(registry, n_plans)


def _build_prompt(
    registry:    PatternRegistry,
    run_history: list[dict],
    goal:        str,
    n_plans:     int,
) -> str:
    entries = registry.all()
    registry_summary = [
        {
            "pattern":   e.pattern,
            "tier":      e.tier,
            "confidence": e.confidence,
            "runs":      e.runs,
            "last_status": e.last_status,
            "promotion_candidate": e.promotion_candidate,
        }
        for e in entries
    ]

    recent_runs = run_history[-20:] if len(run_history) > 20 else run_history

    return f"""You are the planning layer of an autonomous experiment engine.

Goal: {goal}

Current pattern registry:
{json.dumps(registry_summary, indent=2)}

Recent experiment history (last {len(recent_runs)} runs):
{json.dumps(recent_runs, indent=2)}

Task: Suggest the next {n_plans} experiments to run.

Rules:
- Prioritise patterns with few runs and high potential
- Avoid repeating configs that already failed
- Suggest hyperparameter variations for promising patterns
- Consider patterns in 'scratch' tier that have never been tried
- Return ONLY valid JSON — no preamble, no markdown, no explanation

Return exactly this JSON structure:
{{
  "plans": [
    {{
      "pattern_name": "<name>",
      "rationale": "<one sentence why>",
      "config": {{}},
      "priority": 1
    }}
  ]
}}"""


def _qwen_plan(
    registry:    PatternRegistry,
    run_history: list[dict],
    goal:        str,
    n_plans:     int,
) -> list[ExperimentPlan]:
    prompt = _build_prompt(registry, run_history, goal, n_plans)

    payload = json.dumps({
        "model":  QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    raw = body.get("response", "")
    # Strip any accidental markdown fences
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)

    return [
        ExperimentPlan(
            pattern_name = p["pattern_name"],
            rationale    = p.get("rationale", ""),
            config       = p.get("config", {}),
            priority     = p.get("priority", i + 1),
        )
        for i, p in enumerate(parsed.get("plans", []))
    ]


def _rule_based_plan(
    registry: PatternRegistry,
    n_plans:  int,
) -> list[ExperimentPlan]:
    """
    Fallback planner when QWEN is unavailable.

    Strategy:
      1. Patterns with zero runs (never tried) — highest priority
      2. Patterns below working_threshold with fewest runs — try more
      3. Patterns near stable_threshold — push over the line
    """
    entries = registry.all()
    plans: list[ExperimentPlan] = []

    # Never-tried patterns first
    for e in entries:
        if e.runs == 0:
            plans.append(ExperimentPlan(
                pattern_name = e.pattern,
                rationale    = "never been run — establish baseline",
                config       = {},
                priority     = len(plans) + 1,
            ))

    # Scratch patterns with fewest runs
    scratch = sorted(registry.by_tier("scratch"), key=lambda e: e.runs)
    for e in scratch:
        if len(plans) >= n_plans:
            break
        if not any(p.pattern_name == e.pattern for p in plans):
            plans.append(ExperimentPlan(
                pattern_name = e.pattern,
                rationale    = f"scratch tier, only {e.runs} runs — needs more data",
                config       = {},
                priority     = len(plans) + 1,
            ))

    # Working patterns near stable threshold
    working = sorted(
        registry.by_tier("working"),
        key=lambda e: e.confidence,
        reverse=True,
    )
    for e in working:
        if len(plans) >= n_plans:
            break
        plans.append(ExperimentPlan(
            pattern_name = e.pattern,
            rationale    = f"working tier, score {e.confidence:.3f} — push toward stable",
            config       = {},
            priority     = len(plans) + 1,
        ))

    return plans[:n_plans]
````

---

## Component 6: Engine entry point

### `main.py`

```python
"""
air-lab-os — main entry point.

Usage:
    uv run python main.py --pattern <name> --dataset <dataset_id>
    uv run python main.py --loop --dataset <dataset_id> --domain <domain>
    uv run python main.py --arena --dataset <dataset_id>

The engine is domain-agnostic. Use-case plugins are imported
dynamically via --domain. Patterns are discovered from the
patterns/ directory tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from memory.registry import PatternRegistry, RUNS_PATH, REGISTRY_PATH


def _load_runs() -> list[dict]:
    if RUNS_PATH.exists():
        try:
            return json.loads(RUNS_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _discover_patterns() -> dict[str, Path]:
    """Find all pattern .py files across tier directories."""
    found = {}
    for tier_dir in ["patterns/scratch", "patterns/working", "patterns/stable"]:
        for p in Path(tier_dir).glob("*.py"):
            if p.name.startswith("_"):
                continue
            found[p.stem] = p
    return found


def cmd_status() -> None:
    """Print current registry state."""
    registry = PatternRegistry.load()
    entries  = registry.all()
    if not entries:
        print("Registry is empty — no experiments run yet.")
        return
    print(f"\n{'Pattern':<30} {'Tier':<10} {'Score':>7} {'Runs':>5} {'Status':<10}")
    print("-" * 68)
    for e in entries:
        cand = " [promote?]" if e.promotion_candidate else ""
        print(
            f"{e.pattern:<30} {e.tier:<10} "
            f"{e.confidence:>7.4f} {e.runs:>5} "
            f"{e.last_status:<10}{cand}"
        )
    candidates = registry.promotion_candidates()
    if candidates:
        print(f"\n  Promotion candidates: {[e.pattern for e in candidates]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="air-lab-os — domain-agnostic self-improving engine"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Status
    subparsers.add_parser("status", help="Print registry state")

    # Single experiment
    run_p = subparsers.add_parser("run", help="Run one experiment")
    run_p.add_argument("--pattern",     required=True, help="Pattern name")
    run_p.add_argument("--dataset",     required=True, help="Dataset plugin module")
    run_p.add_argument("--description", default="",    help="Run description")

    # Arena
    arena_p = subparsers.add_parser("arena", help="Compare all patterns")
    arena_p.add_argument("--dataset", required=True, help="Dataset plugin module")

    # Autonomous loop
    loop_p = subparsers.add_parser("loop", help="Run autonomous experiment loop")
    loop_p.add_argument("--dataset", required=True, help="Dataset plugin module")
    loop_p.add_argument("--goal",    default="maximise composite score")
    loop_p.add_argument("--rounds",  type=int, default=0,
                         help="Max rounds (0 = run until interrupted)")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        cmd_status()
        return

    # All other commands need a dataset
    # Dataset is loaded by importing the module specified by --dataset
    # e.g. --dataset bb_datasets.fraud.handle → imports and calls get_handle()
    dataset_module = importlib.import_module(args.dataset)
    handle = dataset_module.get_handle()

    if args.command == "run":
        from lab.playground import run_experiment
        patterns = _discover_patterns()
        if args.pattern not in patterns:
            print(f"Pattern '{args.pattern}' not found. Available: {list(patterns)}")
            return
        module = importlib.import_module(
            patterns[args.pattern].as_posix()
            .replace("/", ".").removesuffix(".py")
        )
        pattern = module.get_pattern()
        result  = run_experiment(pattern, handle, description=args.description)
        print(f"\n--- Run result ---")
        print(f"pattern:  {result.pattern}")
        print(f"status:   {result.status}")
        print(f"tier:     {result.tier}")
        print(f"score:    {result.score:.4f}")
        if result.metrics:
            m = result.metrics
            print(f"primary:  {m.primary_metric_value:.4f}")
            print(f"expl:     {m.explainability_score:.4f}")
            print(f"latency:  {m.latency_score:.4f}")
        print(f"commit:   {result.commit}")

    elif args.command == "arena":
        from lab.arena import compare_patterns
        patterns = _discover_patterns()
        pattern_instances = []
        for name, path in patterns.items():
            mod = importlib.import_module(
                path.as_posix().replace("/", ".").removesuffix(".py")
            )
            pattern_instances.append(mod.get_pattern())
        arena = compare_patterns(pattern_instances, handle)
        print(f"\n--- Arena results ({handle.meta.name}) ---")
        for i, r in enumerate(arena.rankings, 1):
            crash = " [CRASH]" if r.status == "crash" else ""
            print(f"  {i}. {r.pattern:<28} score={r.score:.4f}{crash}")
        if arena.winner:
            print(f"\n  Winner: {arena.winner.pattern}")

    elif args.command == "loop":
        from lab.playground import run_experiment
        from runtime.llm import plan
        round_n = 0
        print(f"Starting autonomous loop. Goal: {args.goal}")
        print("Press Ctrl+C to stop.\n")
        try:
            while args.rounds == 0 or round_n < args.rounds:
                round_n += 1
                registry = PatternRegistry.load()
                runs     = _load_runs()
                plans    = plan(registry, runs, goal=args.goal)
                if not plans:
                    print("Planner returned no experiments. Stopping.")
                    break
                for p in plans:
                    patterns = _discover_patterns()
                    if p.pattern_name not in patterns:
                        print(f"  [skip] {p.pattern_name} — not found in patterns/")
                        continue
                    mod = importlib.import_module(
                        patterns[p.pattern_name].as_posix()
                        .replace("/", ".").removesuffix(".py")
                    )
                    pattern = mod.get_pattern()
                    print(f"  Round {round_n}: {p.pattern_name} — {p.rationale}")
                    result = run_experiment(pattern, handle, description=p.rationale)
                    print(
                        f"    score={result.score:.4f} "
                        f"status={result.status} "
                        f"tier={result.tier}"
                    )
        except KeyboardInterrupt:
            print("\nLoop interrupted by user.")
        cmd_status()


if __name__ == "__main__":
    main()
```

---

## Tests

### `tests/test_evaluator.py`

```python
"""Evaluator tests — no domain imports, pure engine logic."""

import pytest
from evaluation.evaluator import evaluate, load_policy, EvalMetrics
from patterns.base import RunResult
from datasets.base import DatasetHandle, DatasetMeta
import pandas as pd


class StubHandle(DatasetHandle):
    """Minimal DatasetHandle for testing."""
    def __init__(self, n=10, n_positive=3):
        self._n = n
        self._n_pos = n_positive

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="stub", domain="test", tier="bronze", version="0.1",
            label_column="label", primary_metric="f1_score",
            row_count=self._n,
        )

    def eval_df(self) -> pd.DataFrame:
        return pd.DataFrame({"label": [True]*self._n_pos + [False]*(self._n-self._n_pos)})

    def train_df(self) -> pd.DataFrame:
        return self.eval_df()

    def labels(self) -> list[bool]:
        return [True]*self._n_pos + [False]*(self._n-self._n_pos)


@pytest.fixture
def policy():
    return load_policy()


@pytest.fixture
def handle():
    return StubHandle()


def _result(primary=1.0, with_expl=True, latency_ms=10.0, n=10, n_pos=3):
    flags = [True]*n_pos + [False]*(n-n_pos)
    scores = [0.9]*n_pos + [0.1]*(n-n_pos)
    expl = (["high signal"]*n_pos + [""]*( n-n_pos)) if with_expl else [""]*n
    r = RunResult(flags=flags, scores=scores, explanation=expl, latency_ms=latency_ms)
    r.primary_metric_value = primary
    return r


def test_score_in_range(policy, handle):
    m = evaluate(_result(), handle, policy)
    assert 0.0 <= m.score <= 1.0


def test_perfect_primary_metric(policy, handle):
    m = evaluate(_result(primary=1.0), handle, policy)
    assert m.primary_metric_value == pytest.approx(1.0)


def test_zero_primary_metric(policy, handle):
    m = evaluate(_result(primary=0.0), handle, policy)
    assert m.primary_metric_value == pytest.approx(0.0)


def test_full_explainability(policy, handle):
    m = evaluate(_result(with_expl=True), handle, policy)
    assert m.explainability_score == pytest.approx(1.0)


def test_zero_explainability(policy, handle):
    m = evaluate(_result(with_expl=False), handle, policy)
    assert m.explainability_score == pytest.approx(0.0)


def test_latency_penalised(policy, handle):
    fast = evaluate(_result(latency_ms=0.0), handle, policy)
    slow = evaluate(_result(latency_ms=500.0), handle, policy)
    assert fast.latency_score > slow.latency_score
    assert slow.latency_score == pytest.approx(0.0)


def test_no_flags_explainability_is_one(policy, handle):
    r = RunResult(flags=[False]*10, scores=[0.0]*10, explanation=[""]*10)
    r.primary_metric_value = 0.0
    m = evaluate(r, handle, policy)
    assert m.explainability_score == pytest.approx(1.0)


def test_weights_sum_to_one(policy):
    total = (
        policy.primary_metric_weight
        + policy.explainability_weight
        + policy.latency_weight
        + policy.cost_weight
    )
    assert abs(total - 1.0) < 1e-6
```

### `tests/test_registry.py`

```python
"""Registry tests — verifies status/tier separation."""

import json
import pytest
from pathlib import Path
from memory.registry import PatternRegistry, append_run


def _run(pattern, score, status, timestamp, domain="test", dataset_id="stub"):
    return {
        "pattern":      pattern,
        "domain":       domain,
        "version":      "0.1",
        "dataset_id":   dataset_id,
        "pattern_path": f"patterns/scratch/{pattern}.py",
        "config":       {},
        "score":        score,
        "status":       status,
        "tier":         "scratch",
        "commit":       "abc1234",
        "timestamp":    timestamp,
        "description":  f"{pattern} test run",
        "metrics":      {"primary_metric_value": score},
    }


def _write_runs(path: Path, runs: list[dict]) -> None:
    path.write_text(json.dumps(runs, indent=2))


def test_status_and_tier_are_separate(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.72, "keep", 1000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    e = reg.get("pattern_a")
    assert e.last_status == "keep"     # run outcome
    assert e.tier == "working"         # promotion level (0.72 >= 0.65)


def test_discard_status_does_not_affect_tier(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.72, "keep",    1000),
        _run("pattern_a", 0.60, "discard", 2000),  # worse run — discarded
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    e = reg.get("pattern_a")
    assert e.last_status == "discard"  # last run was discarded
    assert e.tier == "working"         # tier unchanged — confidence still 0.72
    assert e.confidence == pytest.approx(0.72)
    assert e.runs == 2


def test_scratch_tier_below_threshold(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_b", 0.50, "keep", 1000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_b").tier == "scratch"


def test_stable_tier_preserved_across_rebuild(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.82, "keep", 1000),
    ])
    # Human promoted to stable — persist in registry.json
    registry_path.write_text(json.dumps({
        "pattern_a": {"tier": "stable", "confidence": 0.82}
    }))
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_a").tier == "stable"


def test_promotion_candidate_requires_min_runs(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    # One run above stable_threshold — not yet a candidate (min_runs=3)
    _write_runs(runs_path, [
        _run("pattern_a", 0.82, "keep", 1000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_a").promotion_candidate is False


def test_promotion_candidate_after_min_runs(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    runs = [_run("pattern_a", 0.82, "keep", i) for i in range(3)]
    _write_runs(runs_path, runs)
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_a").promotion_candidate is True


def test_best_metrics_are_domain_agnostic(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    run = _run("pattern_a", 0.72, "keep", 1000)
    run["metrics"] = {"primary_metric_value": 0.72, "custom_kpi": 0.88}
    _write_runs(runs_path, [run])
    reg = PatternRegistry.load(runs_path, registry_path)
    e = reg.get("pattern_a")
    assert "custom_kpi" in e.best_metrics


def test_append_run_creates_file(tmp_path):
    runs_path = tmp_path / "runs.json"
    append_run({"pattern": "p", "score": 0.5}, runs_path)
    assert runs_path.exists()
    data = json.loads(runs_path.read_text())
    assert len(data) == 1


def test_append_run_is_append_only(tmp_path):
    runs_path = tmp_path / "runs.json"
    append_run({"pattern": "p", "score": 0.5}, runs_path)
    append_run({"pattern": "p", "score": 0.6}, runs_path)
    data = json.loads(runs_path.read_text())
    assert len(data) == 2
    assert data[0]["score"] == pytest.approx(0.5)  # first entry preserved
```

### `tests/test_playground.py`

```python
"""Playground tests — mocks PatternHandler and DatasetHandle."""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from datasets.base import DatasetHandle, DatasetMeta
from patterns.base import PatternHandler, RunResult
from lab.playground import run_experiment


class StubHandle(DatasetHandle):
    @property
    def meta(self):
        return DatasetMeta(
            name="stub", domain="test", tier="bronze", version="0.1",
            label_column="label", primary_metric="f1_score", row_count=10,
        )
    def eval_df(self):
        return pd.DataFrame({"label": [True]*3 + [False]*7})
    def train_df(self):
        return self.eval_df()
    def labels(self):
        return [True]*3 + [False]*7


class StubPattern(PatternHandler):
    name    = "stub_pattern"
    version = "0.1"

    def __init__(self, primary_metric=0.8):
        self._primary = primary_metric

    def run(self, handle):
        df = handle.eval_df()
        n  = len(df)
        r  = RunResult(
            flags       = [True]*3 + [False]*7,
            scores      = [0.9]*3 + [0.1]*7,
            explanation = ["high signal"]*3 + [""]*7,
        )
        r.primary_metric_value = self._primary
        return r

    def describe(self):
        return {"pattern": self.name, "primary_metric": self._primary}


class CrashPattern(PatternHandler):
    name    = "crash_pattern"
    version = "0.1"

    def run(self, handle):
        raise RuntimeError("intentional crash")

    def describe(self):
        return {"pattern": self.name}


@pytest.fixture
def tmp_paths(tmp_path):
    return {
        "runs":     tmp_path / "runs.json",
        "registry": tmp_path / "registry.json",
    }


def test_successful_run_writes_to_runs_json(tmp_paths):
    run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert tmp_paths["runs"].exists()
    data = json.loads(tmp_paths["runs"].read_text())
    assert len(data) == 1
    assert data[0]["pattern"] == "stub_pattern"
    assert data[0]["status"] in ("keep", "discard")


def test_crash_is_logged_not_raised(tmp_paths):
    result = run_experiment(
        CrashPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert result.status == "crash"
    assert tmp_paths["runs"].exists()


def test_result_has_separate_status_and_tier(tmp_paths):
    result = run_experiment(
        StubPattern(primary_metric=0.8), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert result.status in ("keep", "discard", "crash")
    assert result.tier in ("scratch", "working", "stable")


def test_registry_updated_after_run(tmp_paths):
    run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert tmp_paths["registry"].exists()
    data = json.loads(tmp_paths["registry"].read_text())
    assert "stub_pattern" in data


def test_second_run_status_discard_if_no_improvement(tmp_paths):
    run_experiment(
        StubPattern(primary_metric=0.8), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    r2 = run_experiment(
        StubPattern(primary_metric=0.8), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert r2.status == "discard"
```

### `tests/test_arena.py`

```python
"""Arena tests."""

import pytest
import pandas as pd
from pathlib import Path

from datasets.base import DatasetHandle, DatasetMeta
from patterns.base import PatternHandler, RunResult
from lab.arena import compare_patterns


class StubHandle(DatasetHandle):
    @property
    def meta(self):
        return DatasetMeta(
            name="stub", domain="test", tier="bronze", version="0.1",
            label_column="label", primary_metric="f1_score", row_count=10,
        )
    def eval_df(self):
        return pd.DataFrame({"label": [True]*3 + [False]*7})
    def train_df(self):
        return self.eval_df()
    def labels(self):
        return [True]*3 + [False]*7


def _make_pattern(name, score):
    class P(PatternHandler):
        def run(self, handle):
            df = handle.eval_df()
            n  = len(df)
            r  = RunResult(
                flags=[True]*3+[False]*7,
                scores=[score]*3+[0.1]*7,
                explanation=["sig"]*3+[""]*7,
            )
            r.primary_metric_value = score
            return r
        def describe(self):
            return {"pattern": self.name}
    P.name    = name
    P.version = "0.1"
    return P()


@pytest.fixture
def tmp_paths(tmp_path):
    return {"runs": tmp_path/"runs.json", "registry": tmp_path/"registry.json"}


def test_arena_ranks_by_score(tmp_paths):
    patterns = [
        _make_pattern("low",  0.5),
        _make_pattern("high", 0.9),
        _make_pattern("mid",  0.7),
    ]
    arena = compare_patterns(patterns, StubHandle(),
                             runs_path=tmp_paths["runs"],
                             registry_path=tmp_paths["registry"])
    scores = [r.score for r in arena.rankings]
    assert scores == sorted(scores, reverse=True)


def test_arena_winner_is_highest(tmp_paths):
    patterns = [_make_pattern("a", 0.6), _make_pattern("b", 0.85)]
    arena = compare_patterns(patterns, StubHandle(),
                             runs_path=tmp_paths["runs"],
                             registry_path=tmp_paths["registry"])
    assert arena.winner.pattern == "b"


def test_arena_logs_all_patterns(tmp_paths):
    import json
    patterns = [_make_pattern(f"p{i}", 0.5+i*0.1) for i in range(3)]
    compare_patterns(patterns, StubHandle(),
                     runs_path=tmp_paths["runs"],
                     registry_path=tmp_paths["registry"])
    runs = json.loads(tmp_paths["runs"].read_text())
    assert len(runs) == 3
```

---

## registry.json schema

The updated schema — note `status` and `tier` are separate fields:

```json
{
  "pattern_name": {
    "pattern": "pattern_name",
    "domain": "fraud",
    "version": "0.1",
    "pattern_path": "patterns/scratch/pattern_name.py",

    "tier": "scratch",
    "last_status": "keep",

    "confidence": 0.721,
    "runs": 4,
    "last_commit": "38378b8",
    "last_updated": 1775715431,
    "last_dataset": "bb_fraud_v3",

    "promotion_candidate": false,

    "best_metrics": {
      "primary_metric_value": 0.721,
      "explainability_score": 0.95,
      "latency_score": 0.98
    },

    "description": "rule_v1 spike_threshold=20000"
  }
}
```

Key changes from the sample registry.json you provided:

- `status: "silver"` → split into `tier: "working"` + `last_status: "keep"`
- `best_precision/recall/f1` → `best_metrics: dict` (domain-agnostic)
- Added `domain`, `version`, `pattern_path`, `last_dataset`

---

## Deliverables checklist

Claude Code must confirm all before Phase 1 is complete:

- [ ] `uv run pytest tests/ -q` passes with zero failures
- [ ] `uv run python main.py status` runs without error (empty registry is fine)
- [ ] `memory/runs.json` does not exist yet (created on first run)
- [ ] `registry.json` does not exist yet (created on first run)
- [ ] `scoring_policy.yaml` weights sum to exactly 1.0
- [ ] `CLAUDE.md` is present and unmodified
- [ ] Zero domain-specific imports in `runtime/`, `lab/`, `evaluation/`, `memory/`
- [ ] `RegistryEntry.tier` and `RegistryEntry.last_status` are separate fields
- [ ] `best_metrics` is a `dict` not individual precision/recall/f1 fields

## What Claude Code must NOT do in Phase 1

- Do not import from `bb_datasets` or any domain repo — the engine is generic
- Do not hardcode column names (`fraud_flag`, `txn_id`, etc.) anywhere in the engine
- Do not hardcode metric names (`f1_score`, `precision`, etc.) in the evaluator
- Do not create any pattern files in `patterns/scratch/` — that is the use case's job
- Do not build the dashboard or API
- Do not add dependencies not in pyproject.toml
- Do not modify `scoring_policy.yaml`

---

_End of Phase 1 spec._
_Phase 2: fraud detection use case plugin + external-datasets layer._
_Phase 3: dashboard + SSE log stream._
_Phase 4: auto-promote loop + notifications._
