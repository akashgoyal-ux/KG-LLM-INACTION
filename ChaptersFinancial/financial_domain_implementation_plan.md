# Financial Domain Implementation Plan (Chapter-by-Chapter)

Date: 2026-04-18
Repository: knowledge-graphs-and-llms-in-action

## 1) Goal
Build a financial-domain counterpart for each chapter (ch02-ch17) by preserving the original architectural progression:
1. Ontology-first KG creation
2. Multi-source ingestion and reconciliation
3. NLP + NED + LLM extraction
4. Graph analytics and GNN modeling
5. Graph RAG and interactive QA applications

This plan is implementation-oriented and aligned with existing code patterns (`util/base_importer.py`, `util/graphdb_base.py`, Makefile orchestration, Neo4j-first pipelines).

## 2) Financial Ontology Stack
Use a layered ontology strategy rather than a single ontology.

### 2.1 Core Ontologies / Standards
- FIBO (Financial Industry Business Ontology): legal entities, instruments, contracts, markets, roles
- CFI / ISO 10962: instrument classification
- ISO 4217: currencies
- LEI (GLEIF): legal entity identifiers and corporate hierarchy
- FIGI / OpenFIGI mapping: tradable instrument identifiers
- XBRL + IFRS / US-GAAP taxonomies: financial reporting concepts
- ACTUS (algorithmic contract terms) for contract event semantics (optional advanced layer)

### 2.2 Entity Model (Canonical)
- `LegalEntity` (LEI, jurisdiction, legal form)
- `Instrument` (ISIN/CUSIP/FIGI/CFI, asset class)
- `Market` / `Venue` / `Exchange`
- `Issuer`, `Underwriter`, `Servicer`, `BeneficialOwner`
- `Filing`, `StatementItem`, `ReportingPeriod`
- `Transaction` / `Position` / `Exposure`
- `Event` (M&A, downgrade, litigation, sanctions, guidance)
- `Document` (news, filing, transcript, report)

### 2.3 Relationship Model (Canonical)
- `(:LegalEntity)-[:ISSUES]->(:Instrument)`
- `(:Instrument)-[:TRADED_ON]->(:Exchange)`
- `(:LegalEntity)-[:PARENT_OF]->(:LegalEntity)`
- `(:LegalEntity)-[:OWNS {pct, asOf}]->(:LegalEntity)`
- `(:Filing)-[:REPORTS_ON]->(:LegalEntity)`
- `(:StatementItem)-[:FROM_FILING]->(:Filing)`
- `(:Document)-[:MENTIONS]->(:LegalEntity|:Instrument|:Event)`
- `(:Event)-[:AFFECTS]->(:Instrument|:LegalEntity)`
- `(:Counterparty)-[:EXPOSED_TO]->(:Counterparty)`

## 3) Shared Engineering Template (apply to all chapters)
- Keep `BaseImporter` inheritance and batch UNWIND ingestion pattern.
- Keep `make init`, `make import`, `make analysis`/`make app` lifecycle.
- Add a new top-level namespace for financial equivalents: `ChaptersFinancial/chXX_fin/`.
- Add environment-driven provider abstraction for LLMs (`OPENAI_*`, `AZURE_*`, `OLLAMA_*`).
- Add deterministic caching (`data/cache_fin_llm/`, `data/cache_fin_api/`).
- Enforce reproducibility: schema migration scripts + deterministic sample datasets.

## 4) Chapter-by-Chapter Financial Implementation

## ch02 -> Financial Prompting Foundations
Objective:
- Replace generic prompt examples with financial prompting and compliance-safe reasoning examples.

Implementation:
- Add prompt templates for:
  - entity extraction from earnings-call text
  - financial event extraction (rating change, guidance revision)
  - contradiction checks across two filings
- Include strict output schema examples (JSON mode).

Deliverables:
- `ChaptersFinancial/ch02_fin/listings/` prompt files
- baseline prompts + few-shot variants + validation rubric

---

## ch03 -> Financial Ontology to KG Bootstrapping
Objective:
- Equivalent of HPO import using FIBO + LEI subset.

Implementation:
- Build `import_fibo_lei.py` importer:
  - create constraints/indexes on `LEI`, `ISIN`, `FIGI`
  - ingest FIBO classes/relations (selected modules only to keep scope manageable)
  - ingest GLEIF sample data for legal entities and parent links
- Apply labeling/typing pass similar to ch03 labeling phase.

Data Sources:
- FIBO ontology exports (RDF/OWL)
- GLEIF monthly concatenated files

Deliverables:
- `ChaptersFinancial/ch03_fin/importer/import_fibo_lei.py`
- `Makefile` with `init/import`
- ontology load documentation

---

## ch04 -> Multi-Source Financial Graph + Community Analysis
Objective:
- Equivalent of biomedical multi-omics graph: market structure + ownership + supply-chain/sector links.

Implementation:
- `import_seed.py` to merge:
  - LEI entity hierarchy
  - security master subset (OpenFIGI mappings)
  - exchange listings and sector classifications (GICS/NAICS proxy)
  - event feeds (downgrades, sanctions, bankruptcies)
- `analysis/louvain_cluster_analysis.py` on ownership and co-movement subgraphs.
- `analysis/systemic_risk_analysis.py` (financial adaptation of pathway analysis).

Deliverables:
- seeded financial graph database
- cluster-level risk concentration report

---

## ch05 -> Structured Financial Data Reconciliation
Objective:
- Equivalent of miRNA reconciliation: unify entities across LEI, ticker, ISIN, CIK, vendor IDs.

Implementation:
- Importers for each source ID system.
- Reconciliation module:
  - deterministic matching (`LEI`, `ISIN`, `CIK`)
  - probabilistic matching (name + jurisdiction + website + address similarity)
  - confidence scoring and manual-review queue
- Introduce `EntityResolutionDecision` nodes for auditability.

Deliverables:
- `reconciliation/reconciliate_financial_entities.py`
- metrics: precision/recall on golden crosswalk set

---

## ch06 -> Financial News NLP + Enrichment
Objective:
- Equivalent of BBC NER pipeline using finance news and company metadata.

Implementation:
- Ingest finance corpora (Reuters sample, SEC press releases, central bank statements).
- NER + span classification for:
  - `LegalEntity`, `Instrument`, `Metric`, `Event`, `PolicyAction`
- Enrichment via APIs:
  - OpenFIGI, GLEIF, optionally Wikidata/DBpedia fallback
- Caching strategy mirrored from ch06.

Deliverables:
- `step1__import_fin_news.py`
- `step2__enrich_entities.py`
- `step3__ownership_enrichment.py`

---

## ch07 -> Embeddings for Financial Concepts
Objective:
- Keep conceptual chapter but use finance-specific embedding tasks.

Implementation:
- Example scripts:
  - embed issuer profiles and compare nearest peers
  - compare earnings-call paragraph embeddings over time
  - evaluate model drift under market regime changes

Deliverables:
- `listings/` scripts with reproducible sample corpora

---

## ch08 -> LLM-Driven Financial KG Extraction
Objective:
- Equivalent diary pipeline for financial documents (10-K, 10-Q, earnings calls, macro reports).

Implementation:
- `ingest_and_process.py` adaptation:
  - ingest document pages/chunks
  - prompt LLM for JSON extraction of entities/relations
  - normalize aliases/tickers/legal names
  - resolve to canonical ontology IDs
- Add strict schema validator and retry policy.
- Keep response cache and processed flags.

Deliverables:
- `importer/ingest_and_process_fin.py`
- prompt templates + JSON schema + cache folder

---

## ch09 -> Financial NED with Ontology Linking
Objective:
- Equivalent SNOMED/UMLS linking for financial ontologies.

Implementation:
- Build financial ontology linker:
  - mention -> candidate entities from LEI/OpenFIGI/FIBO classes
  - rank candidates using context (instrument type, exchange, jurisdiction, period)
  - attach confidence and provenance
- Import OCR annual reports and scanned announcements as source docs.

Deliverables:
- `disambiguation/entity_extractor.py`
- `disambiguation/disambiguator.py`
- `disambiguation/ontology_linking.py`

---

## ch10 -> Open-Model Financial NED (Ollama)
Objective:
- Replicate ch09 with local/open models for cost control and data residency.

Implementation:
- Keep same NED pipeline abstractions.
- Swap remote model calls with local Ollama endpoints.
- Benchmark quality/cost/latency vs ch09 model.

Deliverables:
- `disambiguation/main.py` with model provider strategy
- benchmark report script

---

## ch11 -> Financial Graph Embeddings + Classification
Objective:
- Node embeddings for tasks like issuer risk tiering or fraud ring detection.

Implementation:
- Node2Vec/GraphSAGE embeddings on entity-ownership-transaction graphs.
- Classification examples:
  - high-risk issuer classification
  - suspicious intermediary detection

Deliverables:
- `analysis/simple_classification_example.py`
- `analysis/simple_clustering_example.py`

---

## ch12 -> Financial Feature Engineering
Objective:
- Compute interpretable graph features for risk and compliance models.

Implementation:
- Node features: centralities, egonet stats, temporal activity volatility.
- Edge features: exposure amount, recency decay, relation multiplexity.
- Path features: DWPC across risk-relevant metapaths.
- Train baseline sklearn models with feature importance reports.

Deliverables:
- listing-style scripts mirroring ch12 progression
- feature store export (CSV/Parquet)

---

## ch13 -> Financial GNN Foundations
Objective:
- Adapt toy GNN examples to financial mini-graphs and then production slices.

Implementation:
- Start with synthetic counterparty graph.
- Progress to real subgraph for default-propagation or AML pattern detection.
- Provide architecture comparison (GCN/GAT/SAGE/GIN).

Deliverables:
- `listings/GNN_all_in_one_fin.py`
- reproducible train/eval config

---

## ch14 -> Financial Node Classification + Link Prediction
Objective:
- Full PyG training pipelines for production-style tasks.

Implementation:
- Node classification tasks:
  - AML suspicious node label
  - distress probability class
- Link prediction tasks:
  - likely counterparty relationship
  - potential ownership link or board interlock
- Include temporal split strategy to avoid leakage.

Deliverables:
- `train_for_classification.py`
- `train_for_link_prediction.py`
- `eval/` and `plot/` modules with calibration and PR-AUC reporting

---

## ch15 -> Financial Graph RAG
Objective:
- Hybrid retrieval over filings/news + KG facts for analyst-style Q&A.

Implementation:
- Vector index over documents/chunks with financial metadata filters.
- KG tools:
  - schema-aware Cypher generation
  - entity-centric evidence assembly
  - contradiction detection between textual and structured evidence
- Add citation tracking and answer provenance chain.

Deliverables:
- `code/tools.py` financial adaptation
- `prompts/prompt_structured_fin.txt`

---

## ch16 -> Reserved Integration Chapter
Objective:
- Since repository has no current ch16 implementation, use it as integration and governance bridge.

Implementation:
- Data contracts and ontology versioning strategy.
- CI checks for schema drift.
- Test harness for importer/NER/NED/RAG regression.

Deliverables:
- `ChaptersFinancial/ch16_fin/README.md`
- governance and test playbooks

---

## ch17 -> Financial Investigative Copilot App
Objective:
- Adapt Streamlit + LangGraph app for financial investigations and due diligence.

Implementation:
- UI panels:
  - entity profile
  - ownership map
  - event timeline
  - exposure paths and what-if scenario probe
- Agent tools:
  - schema inspector
  - Cypher generator with safety guardrails
  - report composer with evidence citations
- Support OpenAI/Azure/Ollama via config.

Deliverables:
- `app.py` financial UI
- `chains/investigator.py` finance prompt/logic
- `tools/schema.py` with finance ontology descriptors

## 5) Implementation Phases

Phase 1 (Foundation): ch02-ch05
- Stand up finance ontology core, entity resolution, and deterministic imports.

Phase 2 (NLP + NED): ch06-ch10
- Add document intelligence, disambiguation, and open-model alternative.

Phase 3 (Graph ML): ch11-ch14
- Add embeddings, handcrafted features, and GNN training pipelines.

Phase 4 (Applications): ch15-ch17
- Deliver Graph RAG and interactive financial investigator app.

## 6) Data and Licensing Notes
- Prefer openly licensed subsets for reproducible examples.
- Keep proprietary feeds optional behind adapters.
- Track source and license per dataset in `DATA_SOURCES.md`.

## 7) Engineering Guardrails
- Versioned schema migrations for Neo4j constraints/indexes.
- Provenance fields on all extracted facts (`sourceDocId`, `span`, `extractor`, `timestamp`).
- Evaluation at each stage:
  - NER F1
  - NED accuracy@k
  - Graph ML PR-AUC/ROC-AUC
  - RAG faithfulness + citation precision

## 8) Suggested Folder Blueprint

```text
ChaptersFinancial/
  ch02_fin/
  ch03_fin/
  ch04_fin/
  ch05_fin/
  ch06_fin/
  ch07_fin/
  ch08_fin/
  ch09_fin/
  ch10_fin/
  ch11_fin/
  ch12_fin/
  ch13_fin/
  ch14_fin/
  ch15_fin/
  ch16_fin/
  ch17_fin/
  DATA_SOURCES.md
  financial_domain_implementation_plan.md
```

## 9) Immediate Next Steps
1. Scaffold `ch03_fin`, `ch04_fin`, `ch05_fin` first (ontology import + reconciliation baseline).
2. Freeze canonical ID policy (`LEI`, `FIGI`, `ISIN`, `CIK`) and matching precedence.
3. Implement shared `financial_importer_base.py` and reusable constraint/index setup.
4. Create a 5k-document pilot corpus for ch06-ch10 before scaling.
5. Define evaluation datasets for NER/NED and risk-label tasks before model training.
