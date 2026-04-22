"""
probabilistic_match.py
======================
Fuzzy entity matching using Jaro-Winkler, token-set ratio, jurisdiction, and
address similarity.  Combines features via a calibrated logistic regression
model to produce match probabilities.

Uses real LegalEntity data from Neo4j (loaded by ch03_fin/ch04_fin).
High-confidence matches (>= 0.99) are auto-accepted; lower-confidence
matches are queued for human review.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
import numpy as np

try:
    from rapidfuzz import fuzz
    from jellyfish import jaro_winkler_similarity
except ImportError:
    fuzz = None
    jaro_winkler_similarity = None

from ChaptersFinancial._platform.providers.graph import GraphProvider


# Thresholds
AUTO_ACCEPT_THRESHOLD = 0.99
REVIEW_THRESHOLD = 0.70


def _compute_features(a: dict, b: dict) -> list[float]:
    """Compute similarity features between two entity records."""
    name_a = (a.get("name") or "").strip().upper()
    name_b = (b.get("name") or "").strip().upper()

    # Jaro-Winkler on names
    jw = jaro_winkler_similarity(name_a, name_b) if jaro_winkler_similarity else 0.0

    # Token-set ratio (handles reordering like "APPLE INC" vs "INC APPLE")
    tsr = fuzz.token_set_ratio(name_a, name_b) / 100.0 if fuzz else 0.0

    # Jurisdiction match (binary)
    jur_a = (a.get("jurisdiction") or "").upper()
    jur_b = (b.get("jurisdiction") or "").upper()
    jur_match = 1.0 if jur_a and jur_b and jur_a == jur_b else 0.0

    # Address similarity
    addr_a = (a.get("registeredAddress") or "").upper()
    addr_b = (b.get("registeredAddress") or "").upper()
    addr_sim = 0.0
    if addr_a and addr_b:
        addr_sim = jaro_winkler_similarity(addr_a, addr_b) if jaro_winkler_similarity else 0.0

    # CIK match (if both have it)
    cik_match = 0.0
    if a.get("cik") and b.get("cik"):
        cik_match = 1.0 if a["cik"] == b["cik"] else -1.0

    return [jw, tsr, jur_match, addr_sim, cik_match]


def _logistic_score(features: list[float]) -> float:
    """
    Simple logistic regression scoring with hand-tuned weights.
    In production, train on the golden crosswalk set.
    """
    # Weights: [jw, tsr, jur_match, addr_sim, cik_match]
    weights = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
    bias = -0.40
    z = np.dot(weights, features) + bias
    return float(1.0 / (1.0 + np.exp(-5.0 * z)))  # scaled sigmoid


def run_probabilistic_matching():
    """
    Pull entity pairs from Neo4j that share jurisdiction but not LEI,
    compute similarity features, and create Crosswalk nodes for matches
    above the review threshold.
    """
    print("[ch05_fin] Probabilistic Entity Matching")
    print("=" * 60)

    gp = GraphProvider()

    # Get entities grouped by jurisdiction for blocking
    print("\n1. Loading entities by jurisdiction …")
    entities = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.name IS NOT NULL AND le.jurisdiction IS NOT NULL
        RETURN le.lei AS lei, le.name AS name,
               le.jurisdiction AS jurisdiction,
               le.registeredAddress AS registeredAddress,
               le.cik AS cik
        ORDER BY le.jurisdiction, le.name
    """)
    print(f"   Loaded {len(entities)} entities.")

    if not (fuzz and jaro_winkler_similarity):
        print("\n  [WARN] rapidfuzz / jellyfish not installed. Skipping probabilistic matching.")
        print("  Run: pip install rapidfuzz jellyfish")
        gp.close()
        return

    # Block by jurisdiction
    from collections import defaultdict
    blocks: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        jur = e.get("jurisdiction", "UNKNOWN")
        blocks[jur].append(e)

    matches = []
    reviewed = 0
    print(f"\n2. Comparing within {len(blocks)} jurisdiction blocks …")
    for jur, group in blocks.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, min(i + 50, len(group))):  # limit comparisons
                feats = _compute_features(group[i], group[j])
                score = _logistic_score(feats)
                reviewed += 1
                if score >= REVIEW_THRESHOLD:
                    matches.append({
                        "leiA": group[i]["lei"],
                        "leiB": group[j]["lei"],
                        "nameA": group[i]["name"],
                        "nameB": group[j]["name"],
                        "confidence": round(score, 4),
                        "autoAccept": score >= AUTO_ACCEPT_THRESHOLD,
                    })

    print(f"   Compared {reviewed} pairs, found {len(matches)} potential matches.")

    # Write matches to Neo4j as Crosswalk nodes
    print("\n3. Writing crosswalk nodes …")
    auto_count = 0
    review_count = 0
    for m in matches:
        query = """
        MERGE (cw:Crosswalk {
            idA: $leiA, idTypeA: 'LEI',
            idB: $leiB, idTypeB: 'LEI'
        })
        SET cw.matchType = 'PROBABILISTIC',
            cw.confidence = $confidence,
            cw.nameA = $nameA,
            cw.nameB = $nameB,
            cw.needsReview = NOT $autoAccept,
            cw.ingestedAt = datetime()
        """
        gp.run(query, m)
        if m["autoAccept"]:
            auto_count += 1
        else:
            review_count += 1

    print(f"   Auto-accepted: {auto_count}")
    print(f"   Queued for review: {review_count}")

    # Export review queue
    review_queue = [m for m in matches if not m["autoAccept"]]
    if review_queue:
        queue_path = Path(__file__).parent.parent.parent.parent / "data_fin" / "review_queue.json"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(json.dumps(review_queue, indent=2))
        print(f"\n4. Review queue exported to {queue_path}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run_probabilistic_matching()
