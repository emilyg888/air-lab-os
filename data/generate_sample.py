"""
Generate data/sample_transactions.csv for Phase 1 experiments.

Properties:
- 2000 rows
- Columns: transaction_id, amount, user_id, timestamp, merchant_id, is_fraud
- Fraud rate: ~8%
- Fraud signal 1: users with >6 transactions in any 1-hour window are more
  likely to be fraud (velocity signal for VelocityDetector)
- Fraud signal 2: higher amounts (>$1000) slightly more likely to be fraud
  (feature signal for LogisticDetector)
"""

from __future__ import annotations
import csv
import random
from pathlib import Path


N_ROWS = 2000
N_USERS = 50
N_MERCHANTS = 20
WEEK_SECONDS = 7 * 24 * 3600
TARGET_FRAUD_RATE = 0.08
SEED = 42

OUTPUT = Path(__file__).parent / "sample_transactions.csv"


def main() -> None:
    rng = random.Random(SEED)

    # Generate base transactions
    rows = []
    for i in range(N_ROWS):
        txn_id = f"T{i+1:04d}"
        amount = round(rng.uniform(1.0, 5000.0), 2)
        user_id = f"U{rng.randint(1, N_USERS):03d}"
        timestamp = rng.randint(0, WEEK_SECONDS)
        merchant_id = f"M{rng.randint(1, N_MERCHANTS):03d}"
        rows.append({
            "transaction_id": txn_id,
            "amount": amount,
            "user_id": user_id,
            "timestamp": timestamp,
            "merchant_id": merchant_id,
            "is_fraud": False,
        })

    # Pick a handful of users and cram several of their txns into 1h windows
    # to create the velocity signal.
    fraud_users = rng.sample(
        [f"U{i:03d}" for i in range(1, N_USERS + 1)], k=8
    )
    for uid in fraud_users:
        # Pick a cluster center and squash ~8 of this user's txns near it
        user_rows = [r for r in rows if r["user_id"] == uid]
        if len(user_rows) < 8:
            continue
        cluster = rng.sample(user_rows, k=8)
        center = rng.randint(0, WEEK_SECONDS - 3600)
        for r in cluster:
            r["timestamp"] = center + rng.randint(0, 3500)

    # Sort by timestamp so the velocity detector sees realistic ordering
    rows.sort(key=lambda r: r["timestamp"])

    # Label fraud. Two signals combined:
    #   - velocity: >6 txns by same user in a 1h window → high fraud weight
    #   - amount: >$1000 → small fraud bump
    # We assign probabilistically, then trim/expand to hit ~8%.
    # Pre-compute velocity counts per row.
    by_user: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        by_user.setdefault(r["user_id"], []).append(idx)

    fraud_weights = [0.0] * len(rows)
    for uid, indices in by_user.items():
        ts_sorted = sorted(indices, key=lambda i: rows[i]["timestamp"])
        for i, idx in enumerate(ts_sorted):
            t = rows[idx]["timestamp"]
            count = sum(
                1 for j in ts_sorted
                if t - 3600 <= rows[j]["timestamp"] <= t
            )
            if count > 6:
                fraud_weights[idx] += 0.6
        # amount bump
        for idx in indices:
            if rows[idx]["amount"] > 1000:
                fraud_weights[idx] += 0.08

    # Add base noise
    for i in range(len(rows)):
        fraud_weights[i] += rng.uniform(0.0, 0.03)

    # Take top ~8% as fraud
    target_count = int(N_ROWS * TARGET_FRAUD_RATE)
    ranked = sorted(range(len(rows)), key=lambda i: fraud_weights[i], reverse=True)
    fraud_indices = set(ranked[:target_count])
    for i in fraud_indices:
        rows[i]["is_fraud"] = True

    # Restore transaction_id to be monotonic T0001..T2000 over current order
    for new_i, r in enumerate(rows):
        r["transaction_id"] = f"T{new_i+1:04d}"

    # Write CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["transaction_id", "amount", "user_id", "timestamp", "merchant_id", "is_fraud"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    n_fraud = sum(1 for r in rows if r["is_fraud"])
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Fraud rate: {n_fraud}/{len(rows)} = {n_fraud/len(rows):.2%}")


if __name__ == "__main__":
    main()
