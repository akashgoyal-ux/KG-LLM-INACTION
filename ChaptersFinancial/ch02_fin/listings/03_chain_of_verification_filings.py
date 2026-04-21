"""
Listing 03 – Chain-of-Verification for Filing Summaries
=========================================================
Demonstrates a two-step chain-of-verification (CoV) pattern:
  Step 1 – Initial extraction: extract key facts from a filing excerpt.
  Step 2 – Verification pass: ask the model to check its own output for
            numerical accuracy, contradictions, and forward-looking statement flags.

This pattern catches hallucinated numbers before facts enter the KG.

Run:
    cd <repo_root>
    LLM_PROVIDER=mock python -m ChaptersFinancial.ch02_fin.listings.03_chain_of_verification_filings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ChaptersFinancial._platform.providers.llm import LLMProvider

_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "filing_summary.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

# ------------------------------------------------------------------
# Step 1 – Extraction prompt
# ------------------------------------------------------------------
_EXTRACT_SYSTEM = (
    "You are a financial analyst extracting structured data from SEC filings. "
    "Return ONLY valid JSON. Do not include narrative outside the JSON."
)

_EXTRACT_PROMPT = """
Analyse the following filing excerpt and extract a structured summary.

FILING EXCERPT:
{text}

Return a JSON object with the following keys:
- filing_type (e.g. 10-K, 10-Q, 8-K)
- period (reporting period end date, YYYY-MM-DD or YYYY-Q#)
- issuer (object with name, ticker, cik if available)
- key_facts (array, max 10, each with: fact, category, numeric_value, unit)
- risks (array of risk factor strings)
- forward_looking_statements (array of FLS strings)
- compliance_flags (array, e.g. going concern, restatement)
"""

# ------------------------------------------------------------------
# Step 2 – Verification prompt
# ------------------------------------------------------------------
_VERIFY_SYSTEM = (
    "You are a compliance analyst verifying a structured filing summary for accuracy. "
    "Return ONLY valid JSON with a 'verified' boolean, 'issues' list, and 'corrected' object."
)

_VERIFY_PROMPT = """
You previously extracted the following structured summary from a filing excerpt.
Verify its accuracy against the original text.

ORIGINAL TEXT:
{text}

EXTRACTED SUMMARY:
{summary}

Check for:
1. Numerical accuracy (are figures correct as stated in the original text?)
2. Contradictions between extracted fields
3. Missed forward-looking statements (FLS) that should be flagged
4. Any compliance flags that were overlooked

Return JSON:
{{
  "verified": <true if no issues, false otherwise>,
  "issues": ["<description of issue 1>", ...],
  "corrected": <the corrected summary object, or null if no corrections needed>
}}
"""


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------
_FILING_EXCERPT = """
FORM 10-Q – QUARTERLY REPORT
Registrant: TechCorp Inc.   CIK: 0001234567   Ticker: TCRP
Period of Report: September 30, 2024

FINANCIAL HIGHLIGHTS (UNAUDITED)
Revenue for the quarter ended September 30, 2024 was $1.24 billion,
compared to $1.18 billion for the prior-year quarter, an increase of 5.1%.
Operating income was $312 million (operating margin: 25.2%).
Earnings per diluted share were $0.87, versus $0.79 in the prior-year period.

RISK FACTORS
The Company is subject to significant competition in cloud services.
Foreign exchange fluctuations could materially impact results.
Ongoing litigation with Vendor Corp may result in material losses.

FORWARD-LOOKING STATEMENTS
Management expects revenue for fiscal year 2025 to be in the range of
$5.0 billion to $5.3 billion, representing growth of 8–14% over fiscal 2024.
These statements are subject to risks and uncertainties.
"""


def step1_extract(text: str, provider: LLMProvider) -> dict:
    prompt = _EXTRACT_PROMPT.format(text=text.strip())
    return provider.complete_json(prompt, schema=_SCHEMA, system=_EXTRACT_SYSTEM)


def step2_verify(text: str, summary: dict, provider: LLMProvider) -> dict:
    verify_schema = {
        "type": "object",
        "properties": {
            "verified": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "corrected": {}
        }
    }
    prompt = _VERIFY_PROMPT.format(
        text=text.strip(),
        summary=json.dumps(summary, indent=2)
    )
    return provider.complete_json(prompt, schema=verify_schema, system=_VERIFY_SYSTEM)


def main():
    provider = LLMProvider()
    print(f"Provider: {provider._provider}\n")
    print("STEP 1 – Extraction")
    print("-" * 50)

    summary = step1_extract(_FILING_EXCERPT, provider)
    print(json.dumps(summary, indent=2))

    print("\nSTEP 2 – Chain-of-Verification")
    print("-" * 50)

    verification = step2_verify(_FILING_EXCERPT, summary, provider)
    print(json.dumps(verification, indent=2))

    verified = verification.get("verified", False)
    issues   = verification.get("issues", [])
    print(f"\nVerification result: {'PASS' if verified else 'FAIL'}")
    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues detected.")

    print(f"\nUsage: {provider.usage_summary()}")


if __name__ == "__main__":
    main()
