"""
Listing 01 – Zero-Shot Financial Entity Extraction
===================================================
Demonstrates zero-shot entity extraction from a financial text using the
platform LLM provider.  Uses the canonical entity_extraction.schema.json.

Run:
    cd <repo_root>
    LLM_PROVIDER=mock python -m ChaptersFinancial.ch02_fin.listings.01_zero_shot_entity_extraction

    # With a real model:
    LLM_PROVIDER=openai OPENAI_API_KEY=sk-... \
        python -m ChaptersFinancial.ch02_fin.listings.01_zero_shot_entity_extraction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from repo root
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ChaptersFinancial._platform.providers.llm import LLMProvider

# ------------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a financial named-entity recognition (NER) system.
Your task: extract all financial entities from the given text.
Return ONLY a valid JSON object matching the provided schema.
Do NOT add any commentary outside the JSON."""

_USER_PROMPT_TEMPLATE = """Extract all financial entities from the following text.

TEXT:
{text}

Return a JSON object with the key "entities" containing an array of objects,
each with "text" (exact span), "label" (one of: ORG, PERSON, TICKER, ISIN,
EXCHANGE, CURRENCY, METRIC, LOCATION), "normalized" (canonical form),
and "confidence" (0.0–1.0).
"""

# ------------------------------------------------------------------
# Schema (loaded from file)
# ------------------------------------------------------------------
_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "entity_extraction.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

# ------------------------------------------------------------------
# Example texts
# ------------------------------------------------------------------
EXAMPLE_TEXTS = [
    # Text 1: Simple news headline
    "Goldman Sachs (GS) downgraded NVIDIA Corporation (NVDA) from Buy to Neutral, "
    "cutting its price target to $105 from $135 due to AI chip demand uncertainty.",

    # Text 2: Earnings snippet — first 500 chars keeps inference fast on local CPU models
    (Path(__file__).parent.parent / "golden" / "earnings_call_excerpt.txt").read_text()[:500],
]


def extract_entities(text: str, provider: LLMProvider) -> dict:
    prompt = _USER_PROMPT_TEMPLATE.format(text=text[:500])  # cap to keep inference fast on local CPU models
    return provider.complete_json(prompt, schema=_SCHEMA, system=_SYSTEM_PROMPT)


def main():
    provider = LLMProvider()
    print(f"Provider: {provider._provider}\n")

    for i, text in enumerate(EXAMPLE_TEXTS, start=1):
        print(f"{'=' * 60}")
        print(f"Example {i}")
        print(f"{'=' * 60}")
        preview = text[:200].replace("\n", " ")
        print(f"Text (truncated): {preview}...\n")

        result = extract_entities(text, provider)

        entities = result.get("entities", [])
        print(f"Extracted {len(entities)} entities:")
        for ent in entities:
            label      = ent.get("label", "?")
            text_span  = ent.get("text", "")
            normalized = ent.get("normalized", "")
            conf       = ent.get("confidence", "?")
            print(f"  [{label:12s}]  {text_span:<35s}  -> {normalized}  (conf={conf})")

        print(f"\nUsage so far: {provider.usage_summary()}\n")


if __name__ == "__main__":
    main()
