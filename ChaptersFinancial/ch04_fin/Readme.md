# ch04_fin — Multi-Source Financial Graph + Community Analysis

Mirrors ch04 (multi-source biomedical KG + Louvain/WCC + DWPC) for the financial domain.

## What It Does

1. **Exchanges** — Imports trading venues from the ISO 10383 MIC registry (live CSV).
2. **Instruments (OpenFIGI)** — Maps tickers to FIGI identifiers via the OpenFIGI API.
3. **Listings** — Links instruments to exchanges and issuers (LegalEntities from ch03_fin).
4. **Ownership (SEC 13F)** — Seeds `OWNS` edges from real SEC 13F filings.
5. **Sector Mapping** — Classifies entities by sector using SEC SIC codes.
6. **Community Detection** — Louvain communities on the ownership/parent graph.
7. **Centrality** — PageRank on the entity relationship network.

## Prerequisites

- ch03_fin imported (LegalEntity + OntologyClass nodes must exist).
- Neo4j 5.x with GDS plugin (for Louvain/PageRank) or the analysis scripts
  fall back to Cypher-only approximations.
- Python packages: `neo4j`, `httpx`, `pyyaml`, `tqdm`, `pycountry`.

## Usage

```bash
cd ChaptersFinancial/ch04_fin

# Import all data sources
make import

# Run community detection + centrality analysis
make analysis

# Full pipeline
make all

# Verify
make verify
```

## Configuration

Override via env vars:
- `FIGI_TICKERS` — comma-separated tickers to look up (default: top US tickers)
- `SEC_13F_CIK` — CIK of a 13F filer to import holdings from
