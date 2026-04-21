"""
tests/test_schemas.py
=====================
pytest suite for ch02_fin: JSON schemas, golden data, listings, guardrails.

Tests are split into two groups:
  - Unit/schema tests: always fast, no network required.
  - @pytest.mark.llm: call deepseek-v3.2:cloud via Ollama; skipped if Ollama
    is not reachable at http://localhost:11434.

Run all:         pytest ChaptersFinancial/ch02_fin/tests/test_schemas.py -v
Run unit only:   pytest ChaptersFinancial/ch02_fin/tests/test_schemas.py -v -m "not llm"
Run LLM tests:   pytest ChaptersFinancial/ch02_fin/tests/test_schemas.py -v -m llm
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_CH02      = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_OLLAMA_MODEL = "deepseek-v3.2:cloud"


def _ollama_available() -> bool:
    try:
        import httpx  # type: ignore
        r = httpx.get("http://localhost:11434/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


llm_mark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable at http://localhost:11434"
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_listing(filename: str):
    """Import a listing module by file path (handles digit-prefixed names)."""
    path = _CH02 / "listings" / filename
    spec = importlib.util.spec_from_file_location(f"listing_{filename}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _schema(name: str) -> dict:
    return json.loads((_CH02 / "schemas" / name).read_text())


# --------------------------------------------------------------------------
# Schema file existence
# --------------------------------------------------------------------------
class TestSchemaFiles:
    @pytest.mark.parametrize("fname", [
        "entity_extraction.schema.json",
        "event_extraction.schema.json",
        "filing_summary.schema.json",
    ])
    def test_schema_exists(self, fname):
        assert (_CH02 / "schemas" / fname).exists(), f"Missing schema: {fname}"

    def test_entity_schema_has_required_fields(self):
        s = _schema("entity_extraction.schema.json")
        assert "required" in s
        assert "entities" in s["required"]

    def test_event_schema_has_required_fields(self):
        s = _schema("event_extraction.schema.json")
        assert "entities" not in s.get("required", [])
        assert "events" in s.get("required", [])

    def test_filing_schema_has_required_fields(self):
        s = _schema("filing_summary.schema.json")
        assert "filing_type" in s.get("required", [])
        assert "key_facts" in s.get("required", [])


# --------------------------------------------------------------------------
# Golden data
# --------------------------------------------------------------------------
class TestGoldenData:
    def test_golden_entities_json_parseable(self):
        path = _CH02 / "golden" / "expected_entities.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "entities" in data
        assert len(data["entities"]) > 0

    def test_golden_has_expected_entity_labels(self):
        data = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        labels = {e["label"] for e in data["entities"]}
        assert "ORG"      in labels
        assert "TICKER"   in labels
        assert "METRIC"   in labels
        assert "CURRENCY" in labels

    def test_golden_issuer_has_real_lei(self):
        """The golden file must contain Apple's real GLEIF LEI."""
        data = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        assert data["issuer"]["lei"] == "54930084UKLVMY22DS16"

    def test_golden_issuer_has_real_figi(self):
        data = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        assert data["issuer"]["figi"] == "BBG000B9XRY4"

    def test_golden_issuer_has_real_isin(self):
        data = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        assert data["issuer"]["isin"] == "US0378331005"

    def test_golden_issuer_has_fibo_class(self):
        data = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        fibo = data["issuer"]["fibo_class"]
        assert fibo.startswith("https://spec.edmcouncil.org/fibo/")

    def test_golden_has_events(self):
        data = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        assert "events" in data
        assert len(data["events"]) > 0

    def test_earnings_call_excerpt_exists(self):
        assert (_CH02 / "golden" / "earnings_call_excerpt.txt").exists()

    def test_earnings_call_contains_apple(self):
        text = (_CH02 / "golden" / "earnings_call_excerpt.txt").read_text()
        assert "Apple" in text

    def test_golden_entities_validate_against_schema(self):
        """Validate a stripped entity envelope (schema-required fields only)."""
        pytest.importorskip("jsonschema")
        import jsonschema  # type: ignore

        data   = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        schema = _schema("entity_extraction.schema.json")
        # Strip to schema-defined entity fields (additionalProperties: false)
        allowed = {"text", "label", "normalized", "confidence", "context"}
        stripped = [
            {k: v for k, v in e.items() if k in allowed}
            for e in data["entities"]
        ]
        jsonschema.validate(instance={"entities": stripped}, schema=schema)

    def test_golden_events_validate_against_schema(self):
        pytest.importorskip("jsonschema")
        import jsonschema  # type: ignore

        data   = json.loads((_CH02 / "golden" / "expected_entities.json").read_text())
        schema = _schema("event_extraction.schema.json")
        # Strip to schema-defined event fields
        allowed = {"event_type", "description", "entities_involved",
                   "sentiment", "occurred_at", "confidence"}
        stripped = [
            {k: v for k, v in e.items() if k in allowed}
            for e in data["events"]
        ]
        jsonschema.validate(instance={"events": stripped}, schema=schema)


# --------------------------------------------------------------------------
# Listing 01 – Zero-shot entity extraction
# --------------------------------------------------------------------------
@pytest.mark.llm
class TestListing01ZeroShot:
    @pytest.fixture(scope="class")
    def mod(self):
        return _load_listing("01_zero_shot_entity_extraction.py")

    @pytest.fixture(autouse=True)
    def require_ollama_fixture(self):
        if not _ollama_available():
            pytest.skip("Ollama not reachable")

    @llm_mark
    def test_extract_entities_returns_dict(self, mod):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p = LLMProvider(provider="ollama")
        result = mod.extract_entities(
            "Goldman Sachs (GS) downgraded NVIDIA (NVDA) on 2024-01-15.", p
        )
        assert isinstance(result, dict)

    @llm_mark
    def test_extract_entities_has_entities_key(self, mod):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p = LLMProvider(provider="ollama")
        result = mod.extract_entities(
            "Apple Inc. (AAPL, LEI: 54930084UKLVMY22DS16) revenue $94.9B Q3 2024.", p
        )
        assert "entities" in result

    @llm_mark
    def test_extract_apple_from_real_excerpt(self, mod):
        """LLM must recognise Apple and AAPL from a short golden sentence."""
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        # Use a single sentence — fast for the 27B CPU model
        text = "Apple Inc. (ticker: AAPL) reported iPhone revenue of $46.0 billion in Q3 FY2024."
        p = LLMProvider(provider="ollama")
        result = mod.extract_entities(text, p)
        texts = [str(e.get("text", "")).lower() for e in result.get("entities", [])]
        assert any("apple" in t or "aapl" in t for t in texts), \
            f"Apple/AAPL not found in: {texts}"

    def test_example_texts_defined(self, mod):
        assert hasattr(mod, "EXAMPLE_TEXTS")
        assert len(mod.EXAMPLE_TEXTS) >= 2


# --------------------------------------------------------------------------
# Listing 02 – Few-shot event extraction
# --------------------------------------------------------------------------
@pytest.mark.llm
class TestListing02FewShot:
    @pytest.fixture(scope="class")
    def mod(self):
        return _load_listing("02_few_shot_event_extraction.py")

    @pytest.fixture(autouse=True)
    def require_ollama_fixture(self):
        if not _ollama_available():
            pytest.skip("Ollama not reachable")

    @llm_mark
    def test_extract_events_returns_dict(self, mod):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p = LLMProvider(provider="ollama")
        result = mod.extract_events(
            "Moody's downgraded Goldman Sachs (LEI: 784F5XWPLTWKTBV3E584) from A1 to A2.",
            p,
        )
        assert isinstance(result, dict)

    @llm_mark
    def test_extract_events_has_events_key(self, mod):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p = LLMProvider(provider="ollama")
        result = mod.extract_events(
            "JPMorgan Chase (LEI: 8I5DZWZKVSZI1NUHU748) announced a $30B share buyback.", p
        )
        assert "events" in result

    @llm_mark
    def test_rating_change_event_detected(self, mod):
        """A clear rating-change sentence must produce a RATING_CHANGE event."""
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p = LLMProvider(provider="ollama")
        result = mod.extract_events(
            "S&P lowered Microsoft (LEI: INR2EJN1ERAN0W5ZP974) credit rating from AAA to AA+.",
            p,
        )
        types = [e.get("event_type", "") for e in result.get("events", [])]
        assert any("RATING" in t.upper() for t in types), \
            f"Expected RATING_CHANGE event, got: {types}"

    def test_few_shot_examples_defined(self, mod):
        assert hasattr(mod, "_FEW_SHOT_EXAMPLES")
        assert len(mod._FEW_SHOT_EXAMPLES) >= 2


# --------------------------------------------------------------------------
# Listing 03 – Chain-of-verification
# --------------------------------------------------------------------------
@pytest.mark.llm
class TestListing03ChainOfVerification:
    @pytest.fixture(scope="class")
    def mod(self):
        return _load_listing("03_chain_of_verification_filings.py")

    @pytest.fixture(autouse=True)
    def require_ollama_fixture(self):
        if not _ollama_available():
            pytest.skip("Ollama not reachable")

    @llm_mark
    def test_step1_extract_returns_dict(self, mod):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p = LLMProvider(provider="ollama")
        result = mod.step1_extract(mod._FILING_EXCERPT, p)
        assert isinstance(result, dict)

    @llm_mark
    def test_step2_verify_returns_dict(self, mod):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        p  = LLMProvider(provider="ollama")
        s1 = mod.step1_extract(mod._FILING_EXCERPT, p)
        v  = mod.step2_verify(mod._FILING_EXCERPT, s1, p)
        assert isinstance(v, dict)
        assert "verified" in v or "issues" in v or "corrected" in v or v == {}

    def test_filing_excerpt_non_empty(self, mod):
        assert len(mod._FILING_EXCERPT.strip()) > 100


# --------------------------------------------------------------------------
# Listing 04 – Compliance guardrails
# (guardrail logic is pure Python; only safe_prompt_succeeds calls Ollama)
# --------------------------------------------------------------------------
class TestListing04ComplianceGuardrails:
    @pytest.fixture(scope="class")
    def mod(self):
        return _load_listing("04_compliance_guardrails.py")

    @pytest.mark.llm
    @llm_mark
    def test_safe_prompt_succeeds(self, mod):
        """A normal financial text should pass the guardrail and call Ollama."""
        if not _ollama_available():
            pytest.skip("Ollama not reachable")
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        inner = LLMProvider(provider="ollama")
        provider = mod.GuardedLLMProvider(inner=inner)
        text = "Microsoft (MSFT) Q2 FY2024 revenue $61.9B."
        prompt = f"Extract entities.\n\nTEXT:\n{text}"
        result = provider.complete_json(prompt)
        assert isinstance(result, dict)

    def test_forbidden_price_target_blocked(self, mod):
        # Pure regex – no LLM call needed
        provider = mod.GuardedLLMProvider()
        with pytest.raises(mod.ComplianceError):
            provider.complete_json("What is the price target for NVDA?")

    def test_forbidden_buy_recommendation_blocked(self, mod):
        provider = mod.GuardedLLMProvider()
        with pytest.raises(mod.ComplianceError):
            provider.complete_json("Should I buy Apple stock?")

    def test_disclaimer_always_attached(self, mod):
        # _check_output is pure Python; inject a result dict directly
        provider = mod.GuardedLLMProvider()
        result = provider._check_output({"entities": []})
        assert "disclaimer" in result

    def test_speculative_output_flagged(self, mod):
        provider = mod.GuardedLLMProvider()
        mock_result = {
            "entities": [{"text": "Apple Inc.", "label": "ORG", "confidence": 0.80}],
            "source_excerpt": "Apple revenue is expected to grow 20% next year."
        }
        processed = provider._check_output(mock_result)
        assert processed.get("speculative_language_detected")

    def test_high_confidence_without_context_capped(self, mod):
        provider = mod.GuardedLLMProvider()
        mock_result = {"entities": [{"text": "Apple Inc.", "label": "ORG", "confidence": 0.99}]}
        processed = provider._check_output(mock_result)
        assert processed["entities"][0]["confidence"] <= 0.95

    def test_forbidden_pattern_set_non_empty(self, mod):
        assert len(mod._FORBIDDEN_INPUT_PATTERNS) >= 3
