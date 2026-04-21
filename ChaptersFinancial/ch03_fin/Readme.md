# ch03_fin — Financial Ontology Bootstrapping

Mirrors ch03 (HPO ontology via Neosemantics) for the financial domain.

## What It Does

1. **FIBO Ontology** — Loads a subset of the FIBO (Financial Industry Business Ontology)
   RDF into Neo4j via Neosemantics (n10s), creating `OntologyClass` nodes.
2. **GLEIF LEI** — Streams real Legal Entity Identifier records from the GLEIF API
   and merges them as `LegalEntity` nodes with full provenance.
3. **ISO 4217 Currencies** — Loads all active ISO 4217 currency codes as
   `OntologyClass` nodes (source=ISO4217).
4. **Constraints** — Bootstraps the canonical schema constraints/indices.

## Prerequisites

- Neo4j 5.x running with the **Neosemantics (n10s)** plugin installed.
- Python packages: `neo4j`, `httpx`, `pyyaml`, `tqdm`, `pycountry`.

## Usage

```bash
cd ChaptersFinancial/ch03_fin

# Bootstrap constraints + n10s config
make init

# Download & import FIBO ontology subset
make import-fibo

# Import GLEIF LEI records (live API — configurable country/count)
make import-gleif

# Import ISO 4217 currencies
make import-currencies

# Run everything
make import

# Verify counts
make verify
```

## Configuration

All settings come from `../.env` and `../_platform/config/provider_config.yaml`.
GLEIF import defaults: 100 entities from US jurisdiction.  Override via env vars:

```bash
GLEIF_COUNTRY=GB GLEIF_PAGE_SIZE=200 make import-gleif
```
