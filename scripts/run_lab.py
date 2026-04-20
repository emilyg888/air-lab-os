from __future__ import annotations

from pathlib import Path

import yaml

from core.arena import select_candidates
from core.loop import run_lab
from core.mode import Mode
from core.registry import load_registry
from use_cases.fraud.patterns.ml_logistic import MlLogistic
from use_cases.fraud.patterns.rule_spike import RuleSpike
from use_cases.fraud.patterns.rule_velocity import RuleVelocity


def load_policy():
    with open(Path(__file__).resolve().parents[1] / "scoring_policy.yaml") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    mode = Mode.EXPLORATION

    # -----------------------------
    # 1. Load policy
    # -----------------------------
    policy = load_policy()

    # -----------------------------
    # 2. Select patterns (arena-driven)
    # -----------------------------
    registry = load_registry()

    if registry:
        selected = select_candidates(mode, top_n=3)
        pattern_names = [p["pattern"] for p in selected]
    else:
        pattern_names = ["rule_velocity", "rule_spike", "ml_logistic"]

    # -----------------------------
    # 3. Instantiate patterns
    # -----------------------------
    pattern_map = {
        "rule_velocity": RuleVelocity(),
        "rule_spike": RuleSpike(),
        "ml_logistic": MlLogistic(),
    }

    patterns = [pattern_map[name] for name in pattern_names if name in pattern_map]

    # -----------------------------
    # 4. Run loop (TRAIN + VALIDATE)
    # -----------------------------
    results = run_lab(
        patterns=patterns,
        explore_dataset="fraud_v1",
        validate_dataset="fraud_gold",
        policy=policy,
        mode=mode,
    )

    # -----------------------------
    # 5. Print results
    # -----------------------------
    print("\n=== RESULTS ===")
    for result in results:
        print(result)
