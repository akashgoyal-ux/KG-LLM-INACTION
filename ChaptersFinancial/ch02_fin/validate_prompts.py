"""
validate_prompts.py
===================
Validates the golden expected_entities.json against all three JSON schemas
and runs each listing against the LLM in mock mode, checking that the output
is schema-valid.

Usage:
    cd <repo_root>
    LLM_PROVIDER=mock python -m ChaptersFinancial.ch02_fin.validate_prompts

Exit code 0 = all valid.  Exit code 1 = validation failures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import jsonschema  # type: ignore
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


_CH02 = Path(__file__).parent
_SCHEMAS_DIR = _CH02 / "schemas"
_GOLDEN_DIR  = _CH02 / "golden"

RESULTS: list[tuple[str, bool, str]] = []   # (name, passed, message)


def validate(name: str, instance: dict, schema_file: str) -> bool:
    schema_path = _SCHEMAS_DIR / schema_file
    if not schema_path.exists():
        RESULTS.append((name, False, f"Schema file not found: {schema_file}"))
        return False

    schema = json.loads(schema_path.read_text())

    if not _HAS_JSONSCHEMA:
        # Fallback: only check required keys
        required = schema.get("required", [])
        missing = [k for k in required if k not in instance]
        if missing:
            RESULTS.append((name, False, f"Missing required keys: {missing}"))
            return False
        RESULTS.append((name, True, "OK (jsonschema not installed; required-keys check only)"))
        return True

    try:
        jsonschema.validate(instance=instance, schema=schema)
        RESULTS.append((name, True, "VALID"))
        return True
    except jsonschema.ValidationError as exc:
        RESULTS.append((name, False, str(exc.message)))
        return False


def check_golden_entities():
    """Validate the golden expected_entities.json against entity + event schemas."""
    golden_path = _GOLDEN_DIR / "expected_entities.json"
    if not golden_path.exists():
        RESULTS.append(("golden_entities.json exists", False, "File not found"))
        return

    data = json.loads(golden_path.read_text())

    # Wrap in the schema's expected envelope
    entity_envelope = {"entities": data.get("entities", [])}
    validate("golden / entity_extraction schema", entity_envelope, "entity_extraction.schema.json")

    event_envelope = {"events": data.get("events", [])}
    validate("golden / event_extraction schema", event_envelope, "event_extraction.schema.json")


def check_listing_mock_output(listing_name: str, func, schema_file: str):
    """Run a listing function in mock mode and validate the output."""
    import os
    os.environ["LLM_PROVIDER"] = "mock"
    try:
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        provider = LLMProvider(provider="mock")
        result = func(provider)
        validate(f"{listing_name} (mock output)", result, schema_file)
    except Exception as exc:
        RESULTS.append((f"{listing_name} (mock output)", False, str(exc)))


def check_listing01():
    from ChaptersFinancial.ch02_fin.listings._01_zero_shot_entity_extraction_fn import extract_entities_fn
    check_listing_mock_output(
        "listing01_zero_shot",
        lambda p: extract_entities_fn("Apple reported $94.9B revenue", p),
        "entity_extraction.schema.json",
    )


def check_listing02():
    from ChaptersFinancial.ch02_fin.listings._02_few_shot_event_extraction_fn import extract_events_fn
    check_listing_mock_output(
        "listing02_few_shot_events",
        lambda p: extract_events_fn("Moody's downgraded XYZ Bank.", p),
        "event_extraction.schema.json",
    )


def check_schemas_exist():
    for fname in ["entity_extraction.schema.json", "event_extraction.schema.json", "filing_summary.schema.json"]:
        path = _SCHEMAS_DIR / fname
        RESULTS.append((f"schema exists: {fname}", path.exists(), "" if path.exists() else "MISSING"))


def print_results():
    print(f"\n{'=' * 65}")
    print("  validate_prompts.py – results")
    print(f"{'=' * 65}")
    failures = 0
    for name, passed, msg in RESULTS:
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {icon}  {name}")
        if not passed:
            print(f"         {msg}")
            failures += 1
    print(f"{'=' * 65}")
    print(f"  {len(RESULTS) - failures}/{len(RESULTS)} checks passed")
    return failures


if __name__ == "__main__":
    check_schemas_exist()
    check_golden_entities()

    # Inline mock-based listing checks (import the functional cores)
    # Listings are self-contained; we import them and test the core function
    import os
    os.environ["LLM_PROVIDER"] = "ollama"

    # Import listing modules and call their core extraction functions directly
    sys.path.insert(0, str(_REPO_ROOT))
    from ChaptersFinancial._platform.providers.llm import LLMProvider as _LP

    # Listing 01
    try:
        from ChaptersFinancial.ch02_fin.listings.l01_zero_shot_entity_extraction import (  # noqa
            extract_entities, _SCHEMA as _S01
        )
        r01 = extract_entities("Apple reported $94.9B revenue.", _LP(provider="mock"))
        validate("listing01 extract_entities (mock)", r01, "entity_extraction.schema.json")
    except Exception as exc:
        RESULTS.append(("listing01 extract_entities (mock)", False, str(exc)))

    # Listing 02
    try:
        from ChaptersFinancial.ch02_fin.listings.l02_few_shot_event_extraction import (  # noqa
            extract_events
        )
        r02 = extract_events("Moody's downgraded XYZ Bank.", _LP(provider="mock"))
        validate("listing02 extract_events (mock)", r02, "event_extraction.schema.json")
    except Exception as exc:
        RESULTS.append(("listing02 extract_events (mock)", False, str(exc)))

    # Listing 03
    try:
        from ChaptersFinancial.ch02_fin.listings.l03_chain_of_verification_filings import (  # noqa
            step1_extract, _FILING_EXCERPT
        )
        r03 = step1_extract(_FILING_EXCERPT, _LP(provider="mock"))
        validate("listing03 step1_extract (mock)", r03, "filing_summary.schema.json")
    except Exception as exc:
        RESULTS.append(("listing03 step1_extract (mock)", False, str(exc)))

    failures = print_results()
    sys.exit(1 if failures > 0 else 0)
