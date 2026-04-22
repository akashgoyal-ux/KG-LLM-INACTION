# ch16_fin — Integration, Governance, Test Harness

## Objective
Make the whole stack releasable: data contracts, schema migrations,
lineage tracking, and regression tests.

## Components
- **contracts/** — Pydantic-style data contracts for core node types
- **migrations/** — Versioned Cypher migrations with tracking
- **lineage/** — Data lineage emitter
- **tests/** — Integration and regression tests
