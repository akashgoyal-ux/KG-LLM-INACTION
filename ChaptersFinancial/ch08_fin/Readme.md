# ch08_fin — LLM-Driven KG Extraction from Filings

## Objective
Extract structured facts (entities, events, KPIs) from SEC EDGAR filings
using LLMs with strict JSON schemas, then merge into the canonical KG.

## Data Sources (real-time)
- **SEC EDGAR XBRL companyfacts**: Real financial statement data
- **SEC EDGAR full-text index**: Recent filing texts
- Uses ch02_fin JSON schemas and prompt templates

## Pipeline
1. `ingest_filings.py` — Fetch real SEC filings from EDGAR
2. `extract_with_llm.py` — LLM extraction with JSON schema validation
3. `normalize_and_merge.py` — Normalize and merge into KG with provenance

## Make Targets
- `make ingest` — fetch and chunk filings
- `make extract` — run LLM extraction
- `make merge` — normalize and merge to KG
- `make all` — full pipeline
