# ch05_fin — Entity Resolution and Reconciliation

## Objective
Reconcile entities across LEI / CIK / FIGI / ISIN / vendor IDs and across name variants in news/filings.

## Architecture
- **Deterministic matching**: exact join on canonical IDs (LEI↔CIK, FIGI↔ISIN)
- **Probabilistic matching**: Jaro-Winkler + token-set ratio + jurisdiction + address similarity
- **Scoring**: calibrated logistic regression blender
- **Review queue**: low-confidence matches queued for manual review

## Outputs
- `:Crosswalk` nodes linking authoritative IDs
- `:Mention -[:RESOLVED_TO]-> :LegalEntity|:Instrument` for textual aliases
- Manual-review queue table

## Make Targets
- `make install` — install dependencies
- `make reconcile` — run deterministic + probabilistic matching
- `make eval` — evaluate against golden crosswalk set

## Definition of Done
- Precision ≥ 0.99 at recall ≥ 0.85 on golden set
- Review queue produced for low-confidence rows
