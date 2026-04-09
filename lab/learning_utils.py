"""
lab/learning_utils.py
---------------------
Flattened from learning/extractor.py + learning/validator.py.

Contents:
  - extract():  build a PromotionCandidate from raw engine outputs (deterministic)
  - Validator:  novelty scoring + deduplication against the registry

Architecture invariants:
  - INV-2: Zero LLM calls — extraction, novelty, and dedup are deterministic
  - INV-1: Pipelines not prompts — extractor is one pipeline stage
  - INV-7: Tier decisions are NOT made here — that is criteria.py's job
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

from runtime.config import PromotionConfig
from runtime.promotion import PromotionCandidate

log = logging.getLogger(__name__)


# ===========================================================================
# Extractor
# ===========================================================================
#
# Ownership of extracted fields:
#   - content       ← built from result + reflection + decomposition
#   - title         ← result.summary or tasks[:2]
#   - domain        ← inferred from patterns_used and task text
#   - pattern_type  ← inferred from result.summary
#   - tags          ← mode + patterns_used
#   - lineage       ← patterns_used
#   - scores        ← directly from evaluation dict
#   - novelty       ← set to 0.0 here; filled by Validator.score_novelty
# ===========================================================================

def extract(
    result:        dict,
    evaluation:    dict,
    reflection:    dict,
    decomposition: dict,
    mode:          str,
    run_id:        str,
    iteration:     int,
) -> PromotionCandidate:
    """
    Build a PromotionCandidate from the outputs of one engine iteration.

    Called by PromotionPipeline.process() before novelty scoring.
    Returns a fully-typed candidate with novelty=0.0 (to be filled later).
    """
    content = _build_content(result, reflection, decomposition)
    tags    = _build_tags(result, mode)

    candidate = PromotionCandidate(
        source_run_id    = run_id,
        source_iteration = iteration,
        mode             = mode,
        title            = _make_title(result, decomposition),
        content          = content,
        domain           = _infer_domain(result, decomposition),
        tags             = tags,
        pattern_type     = _infer_pattern_type(result),
        config           = {},
        composite_score  = evaluation.get("score", 0.0),
        correctness      = evaluation.get("correctness", 0.0),
        governance       = evaluation.get("governance", 0.0),
        alignment        = evaluation.get("alignment", 0.0),
        completeness     = evaluation.get("completeness", 0.0),
        novelty          = 0.0,   # filled by Validator.score_novelty
        lineage          = list(result.get("patterns_used", [])),
    )
    log.debug(
        "extractor: candidate title=%r domain=%r type=%r score=%.3f",
        candidate.title, candidate.domain, candidate.pattern_type, candidate.composite_score,
    )
    return candidate


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------

def _build_content(result: dict, reflection: dict, decomposition: dict) -> str:
    """
    Assemble structured, human-readable pattern content from engine outputs.
    Order matters — most important information first.
    """
    parts: list[str] = []

    if summary := result.get("summary", "").strip():
        parts.append(f"Summary: {summary}")

    if outputs := result.get("outputs"):
        if outputs:  # skip empty dicts
            parts.append(f"Outputs: {json.dumps(outputs)}")

    if trade_offs := result.get("trade_offs"):
        parts.append(f"Trade-offs: {', '.join(str(t) for t in trade_offs)}")

    if what_worked := reflection.get("what_worked"):
        parts.append(f"What worked: {', '.join(str(w) for w in what_worked)}")

    if strategy := reflection.get("strategy_update", "").strip():
        parts.append(f"Strategy: {strategy}")

    if tasks := decomposition.get("tasks"):
        parts.append(f"Tasks: {', '.join(str(t) for t in tasks)}")

    if approaches := decomposition.get("approaches"):
        parts.append(f"Approaches: {', '.join(str(a) for a in approaches)}")

    if constraints := decomposition.get("constraints"):
        parts.append(f"Constraints: {', '.join(str(c) for c in constraints)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Field inference
# ---------------------------------------------------------------------------

def _make_title(result: dict, decomposition: dict) -> str:
    summary = result.get("summary", "").strip()
    if summary and summary not in ("stub execution complete", "execution completed (stub)"):
        return summary[:72]
    tasks = decomposition.get("tasks", [])
    if tasks:
        return f"Pattern: {', '.join(str(t) for t in tasks[:2])}"[:72]
    return "Auto-promoted pattern"


def _build_tags(result: dict, mode: str) -> list[str]:
    """Tags: deduped set of mode + patterns_used + standard labels."""
    return sorted({
        "promoted",
        "auto-generated",
        mode,
        *result.get("patterns_used", []),
    })


def _infer_domain(result: dict, decomposition: dict) -> str:
    """
    Infer domain from patterns used and task text.
    Falls back to 'ai-architecture' — the most common domain in this system.
    """
    patterns_used = result.get("patterns_used", [])
    # Coaching is more specific than rag — check it first.
    if any("coach" in p.lower() for p in patterns_used):
        return "coaching"
    if any("rag" in p.lower() for p in patterns_used):
        return "retrieval"

    task_text = " ".join(str(t) for t in decomposition.get("tasks", []))
    if re.search(r"\brag\b|retrieval", task_text, re.IGNORECASE):
        return "retrieval"
    if re.search(r"\beval", task_text, re.IGNORECASE):
        return "ai-evaluation"
    if re.search(r"\bcoach", task_text, re.IGNORECASE):
        return "coaching"
    return "ai-architecture"


def _infer_pattern_type(result: dict) -> str:
    summary = result.get("summary", "").lower()
    if re.search(r"\beval|\bjudge", summary):
        return "evaluation"
    if re.search(r"\breflect|root.cause", summary):
        return "reflection"
    if re.search(r"\bpipeline|\barchitecture", summary):
        return "pipeline"
    return "generic"


# ===========================================================================
# Validator (novelty + deduplication)
# ===========================================================================
#
# Novelty algorithm:
#   novelty = 1.0 - max(jaccard(candidate, existing)) for all existing patterns
#   Jaccard computed on tokenised content (lowercase alphanumeric tokens)
#   Range: 0.0 (exact duplicate) → 1.0 (completely novel)
#
# Deduplication:
#   Pass 1 — SHA-256 hash of content (fast, exact)
#   Pass 2 — Jaccard >= similarity_threshold (near-duplicate)
# ===========================================================================

def _tokenize(text: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens. frozenset for set operations."""
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class Validator:
    """
    Scores novelty and detects duplicates against the current registry.

    Usage (called by promoter.py):
        validator = Validator(config.promotion)
        candidate = validator.score_novelty(candidate, existing_records)
        dup_id    = validator.find_duplicate(candidate, existing_records)
    """

    def __init__(self, config: PromotionConfig) -> None:
        self._cfg = config

    # -----------------------------------------------------------------------
    # Novelty scoring
    # -----------------------------------------------------------------------

    def score_novelty(
        self,
        candidate: PromotionCandidate,
        existing:  list[dict],
    ) -> PromotionCandidate:
        """
        Compute novelty and attach it to the candidate (mutates in place).
        Returns the same candidate object for fluent chaining.

        novelty = 1.0 when no existing patterns exist (first promotion).
        novelty = 0.0 when content is identical to an existing pattern.
        """
        if not existing:
            candidate.novelty = 1.0
            log.debug("validator: novelty=1.0 (empty registry)")
            return candidate

        cand_tokens = _tokenize(candidate.content)
        max_sim     = 0.0

        for record in existing:
            existing_tokens = _tokenize(record.get("content", ""))
            sim = _jaccard(cand_tokens, existing_tokens)
            if sim > max_sim:
                max_sim = sim

        candidate.novelty = round(1.0 - max_sim, 4)
        log.debug(
            "validator: novelty=%.4f (max_similarity=%.4f against %d patterns)",
            candidate.novelty, max_sim, len(existing),
        )
        return candidate

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    def find_duplicate(
        self,
        candidate: PromotionCandidate,
        existing:  list[dict],
    ) -> str | None:
        """
        Return the id of a near-duplicate existing pattern, or None.

        Two-pass algorithm:
          Pass 1 — SHA-256 hash match (O(n), no token computation needed)
          Pass 2 — Token Jaccard >= similarity_threshold
        """
        threshold   = self._cfg.similarity_threshold
        cand_hash   = _hash_content(candidate.content)
        cand_tokens = _tokenize(candidate.content)

        for record in existing:
            # Pass 1: exact content hash
            existing_hash = _hash_content(record.get("content", ""))
            if existing_hash == cand_hash:
                log.info(
                    "validator: exact duplicate of %s (hash=%s)",
                    record["id"], cand_hash,
                )
                return record["id"]

            # Pass 2: near-duplicate by Jaccard
            sim = _jaccard(cand_tokens, _tokenize(record.get("content", "")))
            if sim >= threshold:
                log.info(
                    "validator: near-duplicate of %s (jaccard=%.4f >= %.4f)",
                    record["id"], sim, threshold,
                )
                return record["id"]

        return None
