# ch06_fin — Financial News NLP + Enrichment

## Objective
NER + enrichment over a finance news corpus from SEC RSS feeds and central
bank press releases, populating Document, Chunk, Mention nodes and linking
to LegalEntity/Instrument.

## Data Sources (real-time, no dummy data)
- **SEC EDGAR RSS**: Latest filing press releases
- **Federal Reserve Press Releases**: Public monetary policy statements
- **ECB Press Releases**: European Central Bank public statements

## Pipeline
1. `step1_ingest_news.py` — Fetch real news from SEC RSS + central bank feeds
2. `step2_run_ner.py` — spaCy NER + custom financial entity ruler
3. `step3_enrich_via_apis.py` — GLEIF/OpenFIGI API enrichment

## Make Targets
- `make ingest` — fetch and store documents
- `make ner` — run NER pipeline
- `make enrich` — enrich entities via APIs
- `make all` — full pipeline
