"""
Listing 02 – Few-Shot Financial Event Extraction
=================================================
Demonstrates few-shot prompting for event extraction.
Adding labelled examples inside the prompt significantly improves precision
for rare event types (SANCTIONS, REGULATORY_ACTION, etc.).

Run:
    cd <repo_root>
    LLM_PROVIDER=mock python -m ChaptersFinancial.ch02_fin.listings.02_few_shot_event_extraction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ChaptersFinancial._platform.providers.llm import LLMProvider

_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "event_extraction.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

# ------------------------------------------------------------------
# Few-shot examples embedded in the system message
# ------------------------------------------------------------------
_FEW_SHOT_EXAMPLES = [
    {
        "text": "Moody's lowered its rating on XYZ Bank to Ba1 from Baa3, citing deteriorating asset quality.",
        "output": {
            "events": [{
                "event_type": "RATING_CHANGE",
                "description": "Moody's downgraded XYZ Bank from Baa3 to Ba1",
                "entities_involved": ["XYZ Bank", "Moody's"],
                "sentiment": "NEGATIVE",
                "occurred_at": None,
                "confidence": 0.97
            }]
        }
    },
    {
        "text": "Acme Corp announced it will acquire Beta Ltd for $2.4 billion in an all-stock deal expected to close in Q1 2025.",
        "output": {
            "events": [{
                "event_type": "MA_ANNOUNCEMENT",
                "description": "Acme Corp announced acquisition of Beta Ltd for $2.4 billion",
                "entities_involved": ["Acme Corp", "Beta Ltd"],
                "sentiment": "POSITIVE",
                "occurred_at": None,
                "confidence": 0.99
            }]
        }
    }
]


def _build_few_shot_system() -> str:
    examples_text = ""
    for i, ex in enumerate(_FEW_SHOT_EXAMPLES, start=1):
        examples_text += (
            f"\nExample {i}:\n"
            f"TEXT: {ex['text']}\n"
            f"OUTPUT: {json.dumps(ex['output'])}\n"
        )
    return (
        "You are a financial event extraction system.\n"
        "Extract financial events from text and return a JSON object matching the schema.\n"
        "Here are labelled examples to guide your output:\n"
        + examples_text
        + "\nNow extract events from the user-provided text using the same format."
    )


_SYSTEM_FEW_SHOT = _build_few_shot_system()

_USER_PROMPT_TEMPLATE = """Extract all financial events from the following text.

TEXT:
{text}

Return JSON with key "events" containing an array of event objects.
Each event must have: event_type, description, entities_involved, sentiment, occurred_at, confidence.
"""

# ------------------------------------------------------------------
# Test texts
# ------------------------------------------------------------------
TEST_TEXTS = [
    # Multi-event text
    (
        "The Office of Foreign Assets Control (OFAC) sanctioned three Russian energy firms "
        "on Tuesday. Separately, CEO Jane Doe of FutureBank announced her resignation, "
        "effective March 31, amid an ongoing SEC investigation into derivative mis-selling."
    ),
    # Earnings + guidance — first 500 chars keeps inference fast on cloud models
    (Path(__file__).parent.parent / "golden" / "earnings_call_excerpt.txt").read_text()[:500],
]


def extract_events(text: str, provider: LLMProvider) -> dict:
    prompt = _USER_PROMPT_TEMPLATE.format(text=text[:500])
    return provider.complete_json(prompt, schema=_SCHEMA, system=_SYSTEM_FEW_SHOT)


def main():
    provider = LLMProvider()
    print(f"Provider: {provider._provider}\n")

    for i, text in enumerate(TEST_TEXTS, start=1):
        print(f"{'=' * 60}")
        print(f"Example {i}")
        print(f"{'=' * 60}")
        preview = text[:200].replace("\n", " ")
        print(f"Text: {preview}...\n")

        result = extract_events(text, provider)
        events = result.get("events", [])
        print(f"Extracted {len(events)} event(s):")
        for ev in events:
            print(f"  [{ev.get('event_type', '?'):25s}]  {ev.get('description', '')[:80]}")
            print(f"    entities: {ev.get('entities_involved', [])}")
            print(f"    sentiment={ev.get('sentiment','?')}  confidence={ev.get('confidence','?')}\n")

    print(f"Usage: {provider.usage_summary()}")


if __name__ == "__main__":
    main()
