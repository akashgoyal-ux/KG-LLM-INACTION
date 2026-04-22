"""
contradiction_check.py
======================
Compare numeric claims in LLM answers against StatementItem values in Neo4j.
Flags mismatches between asserted values and authoritative XBRL data.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider

_NUMBER_RE = re.compile(r"\$?([\d,]+(?:\.\d+)?)\s*(million|billion|trillion|M|B|T)?", re.IGNORECASE)

_MULTIPLIERS = {
    "million": 1e6, "m": 1e6,
    "billion": 1e9, "b": 1e9,
    "trillion": 1e12, "t": 1e12,
}


def _parse_amount(text: str) -> float | None:
    """Parse a monetary amount from text."""
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    num = float(match.group(1).replace(",", ""))
    mult = match.group(2)
    if mult:
        num *= _MULTIPLIERS.get(mult.lower(), 1.0)
    return num


def check_contradictions(answer_text: str, entity_name: str, gp: GraphProvider) -> list[dict]:
    """
    Check if numeric claims in the answer contradict XBRL data in Neo4j.
    Returns list of {claim, actual, concept, mismatch_pct}.
    """
    contradictions = []

    # Find numeric claims in the answer
    claims = _NUMBER_RE.findall(answer_text)
    if not claims:
        return []

    # Get actual StatementItem values for the entity
    actuals = gp.run("""
        MATCH (le:LegalEntity)
        WHERE toLower(le.name) CONTAINS toLower($name)
        MATCH (f:Filing)-[:REPORTS_ON]->(le)
        MATCH (si:StatementItem)-[:FROM_FILING]->(f)
        RETURN si.concept AS concept, si.value AS value,
               si.period AS period, si.unit AS unit
        ORDER BY si.period DESC
        LIMIT 20
    """, {"name": entity_name})

    if not actuals:
        return []

    # Compare
    for claim_match in _NUMBER_RE.finditer(answer_text):
        claim_val = _parse_amount(claim_match.group())
        if claim_val is None:
            continue

        for actual in actuals:
            if actual.get("value") and actual["value"] != 0:
                mismatch = abs(claim_val - actual["value"]) / abs(actual["value"])
                if 0.05 < mismatch < 10.0:  # plausible comparison range
                    contradictions.append({
                        "claim": claim_match.group(),
                        "claim_value": claim_val,
                        "actual_value": actual["value"],
                        "concept": actual["concept"],
                        "period": actual.get("period"),
                        "mismatch_pct": round(mismatch * 100, 1),
                    })

    return contradictions
