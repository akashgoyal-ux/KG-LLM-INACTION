# Financial Domain Implementation Plan v2 (In-Depth)

Date: 2026-04-18
Repository: knowledge-graphs-and-llms-in-action
Supersedes: `financial_domain_implementation_plan.md` (kept for history)
Scope: Chapter-by-chapter financial-domain re-implementation of the book's KG + LLM stack, with concrete schemas, modules, Cypher, prompts, evaluation, and execution steps.

---

## 0. Why a v2 Plan

The v1 plan was a high-level outline. v2 is built to be *executable*:
- Maps every chapter to concrete files, classes, Cypher constraints, evaluation metrics, and runtime commands.
- Locks down a single canonical financial schema reused across chapters so chapters compose into one stack instead of being parallel demos.
- Defines provider-agnostic abstractions for LLMs, vector stores, and graph DB, so OpenAI / Azure / Ollama and Neo4j / Memgraph swaps are config changes, not rewrites.
- Adds a real evaluation harness (NER F1, NED accuracy@k, link-prediction PR-AUC, RAG faithfulness) instead of "we will evaluate".
- Adds operational concerns the book chapters under-specify: schema migrations, provenance, lineage, idempotency, rate limiting, secrets, cost tracking.

If anything in v1 conflicts with v2, v2 wins.

---

## 1. Architectural North Star

### 1.1 Layered Architecture

```
                 +------------------------------------------------+
   Layer 5  App  |  Streamlit Investigator + Graph RAG endpoints  |   ch15, ch17
                 +------------------------------------------------+
   Layer 4  ML   |  GNNs, embeddings, feature store, eval harness |   ch11-ch14
                 +------------------------------------------------+
   Layer 3  IE   |  NER, NED, LLM extraction, ontology linking    |   ch06, ch08-ch10
                 +------------------------------------------------+
   Layer 2  KG   |  Canonical Financial KG (Neo4j) + provenance   |   ch03-ch05
                 +------------------------------------------------+
   Layer 1  Data |  Filings, news, market refdata, GLEIF, FIBO    |   raw + cache
                 +------------------------------------------------+
   Layer 0  Plat |  util/, providers, secrets, observability      |   shared
                 +------------------------------------------------+
```

Every chapter lives in exactly one layer and *must* consume from layers below via stable interfaces (no chapter reaches into another chapter's internals).

### 1.2 Canonical Repository Layout

```text
ChaptersFinancial/
  _platform/                    # Layer 0: shared, reusable
    __init__.py
    fin_importer_base.py        # subclass of util.base_importer.BaseImporter
    schema/
      constraints.cypher        # all uniqueness / index DDL
      migrations/               # versioned cypher migrations
    providers/
      llm.py                    # OpenAI / Azure / Ollama unified client
      vector.py                 # Neo4j vector / FAISS adapter
      graph.py                  # Neo4j driver wrapper
    ontology/
      fibo_loader.py            # subset RDF loader via Neosemantics
      gleif_loader.py
      figi_loader.py
      xbrl_loader.py
    eval/
      ner_eval.py
      ned_eval.py
      ml_eval.py
      rag_eval.py
    obs/
      cost_tracker.py
      run_logger.py
    config/
      schema_config.yaml        # canonical labels + props + descriptions
      provider_config.yaml
  ch02_fin/ ... ch17_fin/
  data_fin/                     # raw + cache (gitignored except samples)
    raw/
    cache_llm/
    cache_api/
    samples/
  DATA_SOURCES.md
  README.md
  Makefile                      # umbrella, delegates to chapter Makefiles
```

### 1.3 Canonical Neo4j Schema (single source of truth)

Node labels (PascalCase), properties (camelCase). All facts carry provenance.

| Label | Key | Core Properties |
|---|---|---|
| `LegalEntity` | `lei` (unique) | `name`, `legalForm`, `jurisdiction`, `status`, `registeredAddress`, `cik`, `aliases[]` |
| `Instrument` | `figi` (unique) | `isin`, `cusip`, `ticker`, `cfiCode`, `assetClass`, `currency`, `issuerLei` |
| `Exchange` | `mic` (unique) | `name`, `country`, `operatingMic` |
| `Filing` | `filingId` | `formType`, `filedAt`, `periodEnd`, `cik`, `sourceUrl` |
| `StatementItem` | `(filingId, concept, period)` | `value`, `unit`, `xbrlConcept`, `period` |
| `Document` | `docId` | `type`, `publishedAt`, `source`, `title`, `lang`, `hash` |
| `Chunk` | `chunkId` | `docId`, `ord`, `text`, `embedding` (vector) |
| `Mention` | `mentionId` | `text`, `start`, `end`, `chunkId`, `score`, `extractor` |
| `Event` | `eventId` | `type`, `occurredAt`, `confidence`, `source` |
| `Transaction` | `txId` | `amount`, `currency`, `valueDate`, `direction` |
| `Position` | `(holderLei, instrFigi, asOf)` | `quantity`, `marketValue` |
| `OntologyClass` | `iri` | `label`, `source` (FIBO/CFI) |

Relationships:

```
(:LegalEntity)-[:ISSUES]->(:Instrument)
(:Instrument)-[:LISTED_ON]->(:Exchange)
(:LegalEntity)-[:PARENT_OF {asOf}]->(:LegalEntity)
(:LegalEntity)-[:OWNS {pct, asOf, source}]->(:LegalEntity)
(:LegalEntity)-[:CONTROLS {basis, asOf}]->(:LegalEntity)
(:Filing)-[:REPORTS_ON]->(:LegalEntity)
(:StatementItem)-[:FROM_FILING]->(:Filing)
(:Mention)-[:IN_CHUNK]->(:Chunk)-[:OF_DOC]->(:Document)
(:Mention)-[:RESOLVED_TO {confidence, linker}]->(:LegalEntity|:Instrument|:Event)
(:Document)-[:MENTIONS]->(:LegalEntity|:Instrument|:Event)
(:Event)-[:AFFECTS]->(:LegalEntity|:Instrument)
(:LegalEntity)-[:CLASSIFIED_AS]->(:OntologyClass)
(:Instrument)-[:CLASSIFIED_AS]->(:OntologyClass)
(:LegalEntity)-[:EXPOSED_TO {amount, currency, asOf}]->(:LegalEntity)
```

Provenance pattern: every write attaches `{sourceId, sourceType, ingestedAt, runId, extractor}` either on the relationship, on a `:Provenance` node linked to the fact, or on the node itself for raw imports.

### 1.4 Canonical IDs and Resolution Precedence

When the same real-world entity is referenced multiple ways:

1. `lei` (GLEIF) — authoritative for legal entities.
2. `cik` (SEC) — authoritative for US filers.
3. `figi` — authoritative for tradable instruments.
4. `isin` then `cusip` — fallback for instruments.
5. Name + jurisdiction — last resort, must yield a `Mention -> RESOLVED_TO` with confidence < 1.

A `:Crosswalk` node is created when two authoritative IDs map to the same entity, allowing audit without destructive merges.

### 1.5 Cross-Cutting Engineering Rules

- All importers extend `_platform/fin_importer_base.py` (which extends `util.base_importer.BaseImporter`).
- All Cypher writes are idempotent (`MERGE` on keys, `SET` on properties, never `CREATE` for entities).
- Every job emits a `Run` node (`runId`, `chapter`, `module`, `gitSha`, `startedAt`, `endedAt`, `status`, `metrics`).
- Secrets only via environment; never committed.
- Rate-limited external calls go through `_platform/providers/*` with retry + exponential backoff + on-disk cache keyed by request hash.
- `make lint test` must pass before `make import`.

---

## 2. Financial Ontology Stack (Concrete Usage)

| Ontology / Standard | Used For | Where Loaded | Notes |
|---|---|---|---|
| FIBO | Class hierarchy of legal entities, instruments, contracts | ch03_fin via Neosemantics | Load only `BE`, `FBC`, `SEC`, `IND` modules to bound size |
| GLEIF LEI | LegalEntity master | ch03_fin, ch05_fin | Use Level 1 (entity) + Level 2 (parent) concatenated files |
| ISO 10962 (CFI) | Instrument classification | ch03_fin | Encode 6-letter CFI as `OntologyClass` nodes |
| ISO 4217 | Currency normalization | ch03_fin | Lookup table only |
| FIGI / OpenFIGI | Tradable instrument identity | ch04_fin enrichment | Map ticker+exchange -> FIGI |
| SEC EDGAR | Filings + CIKs | ch04_fin, ch08_fin | Use submissions + facts JSON |
| XBRL US-GAAP / IFRS | StatementItem concepts | ch08_fin | Use `companyfacts` JSON |
| GICS / NAICS | Sector classification | ch04_fin | Vendor or proxy mapping |
| ACTUS (optional) | Contract event semantics | ch12_fin advanced | Defer until ch12 features need it |

A short rationale per chapter sits in §4.

---

## 3. Shared Engineering Template (Layer 0 details)

### 3.1 `fin_importer_base.py` (sketch)

Responsibilities beyond `BaseImporter`:
- Inject `runId` into every batch parameter automatically.
- Standardize constraint creation (`ensure_constraints(self, names: list[str])` reading `config/schema_config.yaml`).
- Standardize provenance attachment helper.
- Provide `merge_legal_entity`, `merge_instrument`, `merge_document`, `merge_mention` Cypher snippets so chapters do not redefine them.

### 3.2 `providers/llm.py`

Abstract over:
- OpenAI Chat Completions (JSON mode, tool calls).
- Azure OpenAI.
- Ollama local.
- Anthropic / Bedrock as optional.

Methods:
- `complete_json(prompt, schema, *, temperature, model)` returns parsed dict validated against `schema` (pydantic).
- `embed(texts, *, model)` returns `np.ndarray`.

Adds: per-call cost tracking, request hash caching at `data_fin/cache_llm/<hash>.json`.

### 3.3 `providers/vector.py`

- Neo4j vector index adapter (preferred to keep stack single-DB).
- Local FAISS fallback for offline development.

### 3.4 `eval/`

- `ner_eval.py`: precision/recall/F1 vs gold spans.
- `ned_eval.py`: accuracy@1/@5, MRR vs gold IDs.
- `ml_eval.py`: ROC-AUC, PR-AUC, calibration, fairness slices.
- `rag_eval.py`: faithfulness (claims supported by retrieved evidence), citation precision, answer correctness vs gold.

### 3.5 `obs/`

- `run_logger.py` writes a `Run` node + JSON log to `data_fin/runs/`.
- `cost_tracker.py` aggregates token + API costs per run.

---

## 4. Chapter-by-Chapter Implementation

For each chapter:
- **Source mapping** (which book chapter pattern is reused).
- **Objective**.
- **Inputs / Outputs**.
- **Concrete files**.
- **Schema deltas**.
- **Algorithms / prompts / models**.
- **Evaluation**.
- **Make targets**.
- **Definition of Done**.

---

### ch02_fin — Financial Prompting Foundations

Source mapping: ch02 (concept + prompt examples).
Objective: Establish the prompting style, JSON schemas, and safety/compliance guardrails used by every later LLM-driven chapter.

Inputs: none beyond example texts (10-K excerpt, earnings-call snippet, news headline).
Outputs: a prompt library + JSON schemas + golden examples used by ch08, ch15, ch17.

Files:
```
ch02_fin/
  listings/
    01_zero_shot_entity_extraction.md
    02_few_shot_event_extraction.md
    03_chain_of_verification_filings.md
    04_compliance_guardrails.md
  schemas/
    entity_extraction.schema.json
    event_extraction.schema.json
    filing_summary.schema.json
  golden/
    earnings_call_excerpt.txt
    expected_entities.json
```

Algorithms / prompts:
- Zero-shot vs few-shot vs chain-of-verification, each with the same JSON schema target.
- Guardrails: forbid speculative price targets, require citations, refuse PII generation.

Evaluation: schema-validity rate on a 50-example test set; manual quality rubric.

Make targets: `make validate` (run schemas against goldens via JSON Schema validator).

DoD: every prompt produces schema-valid JSON in ≥ 95% of runs against the golden set, with two LLM providers.

---

### ch03_fin — Financial Ontology Bootstrapping

Source mapping: ch03 (HPO via Neosemantics).
Objective: Load FIBO subset + GLEIF + CFI + ISO 4217 into Neo4j as the canonical schema substrate.

Inputs:
- FIBO RDF (Production, modules: BE, FBC, SEC, IND).
- GLEIF Level 1 + Level 2 sample (e.g. one country first, then full).
- CFI codes table (CSV).
- ISO 4217 currency table (CSV).

Outputs: Neo4j database `fin_core` with `OntologyClass`, `LegalEntity`, `Currency`, baseline constraints.

Files:
```
ch03_fin/
  Makefile
  Readme.md
  importer/
    import_fibo.py            # uses Neosemantics n10s.rdf.import.fetch
    import_gleif.py           # streams GLEIF zip, batches MERGE
    import_cfi.py
    import_iso4217.py
  schema/
    constraints.cypher
```

Schema deltas: introduces all base constraints listed in §1.3.

Algorithms:
- FIBO: load via Neosemantics, then label nodes from `Resource` to `OntologyClass`; tag `source='FIBO'`.
- GLEIF: stream-parse CSV with `csv.DictReader`, `MERGE (le:LegalEntity {lei:$lei})`, then `MERGE (parent)-[:PARENT_OF]->(child)` from Level 2.

Evaluation:
- Row counts match source files.
- Spot-check 20 LEIs vs GLEIF API.

Make targets: `init`, `download`, `import`, `verify`.

DoD: querying `MATCH (le:LegalEntity) RETURN count(le)` returns expected count; FIBO class hierarchy reachable from `OntologyClass`.

---

### ch04_fin — Multi-Source Financial Graph + Community Analysis

Source mapping: ch04 (multi-source biomedical KG + Louvain/WCC + DWPC).
Objective: Build the working graph used by all later chapters: legal entities, instruments, exchanges, sector classifications, ownership; then run community + centrality analyses.

Inputs:
- Output of ch03_fin.
- OpenFIGI mapping for a chosen instrument universe (e.g. S&P 500 + EuroStoxx 50).
- Exchange master (MIC codes from ISO 10383).
- Sector mapping (GICS-like proxy).
- A simple ownership feed (e.g. SEC 13F sample) to seed `OWNS` edges.

Outputs:
- Populated `fin_core` graph.
- Community labels on `LegalEntity` (`communityLouvain`).
- Centrality scores (`pagerank`, `betweenness`).

Files:
```
ch04_fin/
  Makefile
  importer/
    import_exchanges.py
    import_instruments_figi.py
    import_listings.py
    import_ownership_13f.py
    import_sector_mapping.py
  analysis/
    project_graph.py            # GDS projection
    community_louvain.py
    centrality_pagerank.py
    metapath_dwpc.py            # e.g. (LE)-[:OWNS*1..3]->(LE)-[:ISSUES]->(I)
```

Algorithms:
- GDS projection of `(:LegalEntity)-[:OWNS|PARENT_OF|CONTROLS]-(:LegalEntity)` for community detection.
- DWPC on metapaths to score systemic linkage between issuers.

Evaluation:
- Sanity: known conglomerates land in the same Louvain community.
- DWPC top-K manually inspected.

Make targets: `init`, `import`, `analysis`.

DoD: at least one notebook/markdown report showing top-10 communities and top-10 PageRank issuers with commentary.

---

### ch05_fin — Entity Resolution and Reconciliation

Source mapping: ch05 (miRNA reconciliation with scispacy).
Objective: Reconcile entities across LEI / CIK / FIGI / ISIN / vendor IDs and across name variants in news/filings.

Inputs:
- All identifiers ingested in ch03/ch04.
- A vendor-name CSV with messy aliases.
- A small golden crosswalk set (~500 mappings) for evaluation.

Outputs:
- `:Crosswalk` nodes linking authoritative IDs.
- `:Mention -[:RESOLVED_TO]-> :LegalEntity|:Instrument` for textual aliases.
- Manual-review queue table.

Files:
```
ch05_fin/
  reconciliation/
    deterministic_match.py     # by LEI/CIK/FIGI/ISIN
    probabilistic_match.py     # name + jurisdiction + address similarity
    scoring.py                 # logistic blender with calibrated probabilities
    review_queue.py
  eval/
    eval_reconciliation.py
```

Algorithms:
- Deterministic: exact join on canonical IDs.
- Probabilistic: Jaro-Winkler on name + token-set ratio + jurisdiction match + address Levenshtein; combine with calibrated logistic regression trained on the golden set.
- Threshold tuning to hit target precision (e.g. P>=0.99 for auto-accept, otherwise queue).

Evaluation: precision/recall on golden crosswalk; ROC of the blender.

Make targets: `init`, `reconcile`, `eval`.

DoD: precision ≥ 0.99 at recall ≥ 0.85 on golden set; review queue produced for low-confidence rows.

---

### ch06_fin — Financial News NLP + Enrichment

Source mapping: ch06 (BBC + spaCy + Wikidata).
Objective: NER + enrichment over a finance news corpus, populating `Document`, `Chunk`, `Mention`, and linking to `LegalEntity`/`Instrument`.

Inputs:
- A finance news sample (Reuters/Bloomberg public excerpts, SEC press releases, central bank statements).
- ch04_fin graph for enrichment lookups.

Outputs:
- Document/Chunk/Mention subgraph.
- Enriched `LegalEntity` properties (sector, website) where missing.

Files:
```
ch06_fin/
  importer/
    step1_ingest_news.py
    step2_run_ner.py            # spaCy + custom finance ruler
    step3_enrich_via_openfigi_gleif.py
  ner/
    finance_entity_ruler.py     # tickers, ISINs, currencies, monetary amounts
```

Algorithms:
- spaCy `en_core_web_trf` + EntityRuler with regexes for `TICKER`, `ISIN`, `MONEY`, `PCT`, `RATING`.
- Enrichment: GLEIF API by name; OpenFIGI API by ticker; cache to `data_fin/cache_api/`.

Evaluation:
- NER F1 on 200 manually labeled headlines.
- Enrichment hit rate.

Make targets: `init`, `ingest`, `ner`, `enrich`.

DoD: NER F1 ≥ 0.80 on the labeled set; ≥ 70% enrichment coverage.

---

### ch07_fin — Embeddings for Financial Concepts

Source mapping: ch07 (embedding examples).
Objective: Practical embedding workflows on financial text + tabular features.

Files:
```
ch07_fin/
  listings/
    01_embed_issuer_profiles.py
    02_peer_search_by_embedding.py
    03_drift_across_quarters.py
    04_hybrid_text_plus_features.py
```

Algorithms:
- Sentence embeddings (OpenAI / `bge`/`e5` local) over issuer profile text.
- Cosine peer search; compare to GICS peers.
- Drift: embed earnings-call paragraphs per quarter; track centroid movement.

Evaluation: peer overlap @10 with GICS; qualitative drift report.

Make targets: `init`, `run-listings`.

DoD: notebook outputs reproducible from cache.

---

### ch08_fin — LLM-Driven KG Extraction from Filings

Source mapping: ch08 (Warren Weaver diaries via ChatGPT).
Objective: Extract structured facts (entities, events, KPIs) from 10-K/10-Q/8-K and earnings-call transcripts using LLMs with strict JSON schemas, then merge into the canonical KG.

Inputs:
- A pilot set of 500 SEC filings (mix of 10-K/10-Q/8-K) + transcripts.
- ch02_fin schemas + prompts.

Outputs:
- New `Event`, `StatementItem` nodes.
- `Document`/`Chunk`/`Mention` linked to extracted entities.
- Provenance with `chunkId`, `model`, `promptVersion`.

Files:
```
ch08_fin/
  importer/
    ingest_filings.py
    chunk_and_embed.py
    extract_with_llm.py        # uses _platform/providers/llm.py
    normalize_and_merge.py
  prompts/
    extract_entities.txt
    extract_events.txt
    extract_kpis.txt
  cache/                       # symlink to data_fin/cache_llm
```

Algorithms:
- Chunk filings to ~1.5k tokens with overlap; embed with provider abstraction.
- For each chunk, call LLM with JSON schema; validate; retry on schema fail with reflexion prompt.
- Normalize aliases via ch05_fin reconciler before merging.

Evaluation:
- Schema-validity rate.
- Spot precision on 100 manually checked extractions.
- Cost per filing.

Make targets: `init`, `ingest`, `extract`, `merge`, `eval`.

DoD: ≥ 95% schema validity, spot precision ≥ 0.85, deterministic re-runs from cache.

---

### ch09_fin — Financial NED with Ontology Linking

Source mapping: ch09 (SNOMED/UMLS NED).
Objective: Link textual mentions to canonical financial entities/instruments with calibrated confidence and ontology classes.

Inputs:
- Mentions produced by ch06_fin/ch08_fin.
- ch03_fin ontology classes; ch04_fin entity universe.

Outputs:
- `Mention -[:RESOLVED_TO {confidence, linker, candidates}]-> LegalEntity|Instrument`.
- Ontology classification edges via FIBO/CFI.

Files:
```
ch09_fin/
  disambiguation/
    candidate_generator.py     # ticker/alias dictionary + dense retrieval
    contextual_ranker.py       # BM25 + cross-encoder reranker
    ontology_linker.py
    main.py
  eval/
    eval_ned.py
```

Algorithms:
- Candidate generation: alias dictionary (built from GLEIF + ticker tables) + dense retrieval over entity profile embeddings.
- Ranking: cross-encoder over (mention context, candidate profile); calibration via isotonic regression on a labeled dev set.

Evaluation: accuracy@1/@5, MRR; per-asset-class breakdown.

Make targets: `init`, `disambiguate`, `eval`.

DoD: accuracy@1 ≥ 0.85 on dev set; calibration ECE ≤ 0.05.

---

### ch10_fin — Open-Model NED (Ollama) and Cost/Quality Benchmark

Source mapping: ch10 (Ollama-based NED).
Objective: Reproduce ch09_fin pipeline with local models; benchmark vs ch09_fin.

Files:
```
ch10_fin/
  disambiguation/
    main.py                    # provider=ollama
  benchmark/
    run_benchmark.py
    report.md
```

Algorithms: same pipeline, swap LLM provider; report quality, latency, $ / 1k mentions.

Make targets: `init`, `disambiguate`, `benchmark`.

DoD: benchmark report with at least two models compared on a fixed dev set.

---

### ch11_fin — Graph Embeddings + Classification/Clustering

Source mapping: ch11 (Node2Vec + sklearn).
Objective: Produce node embeddings for downstream tasks (issuer risk tiering, ring detection).

Files:
```
ch11_fin/
  analysis/
    node2vec_issuers.py
    cluster_issuers.py
    classify_high_risk.py
```

Algorithms:
- Node2Vec on `(:LegalEntity)-[:OWNS|CONTROLS|EXPOSED_TO]-(:LegalEntity)`.
- KMeans + silhouette + qualitative inspection.
- Logistic regression / gradient boosting on top of embeddings for high-risk classification using a labeled subset (sanctions list + adverse media).

Evaluation: silhouette; classification ROC-AUC on stratified split.

DoD: ROC-AUC ≥ 0.85 on labeled set with embeddings vs ≥ 0.75 baseline (handcrafted features only).

---

### ch12_fin — Graph Feature Engineering for Risk/Compliance

Source mapping: ch12 (handcrafted graph features).
Objective: Build interpretable features per node/edge/path for risk + AML.

Files:
```
ch12_fin/
  features/
    node_features.py            # degree, triangles, egonet density, centralities
    edge_features.py            # exposure amount, recency decay, multiplexity
    metapath_dwpc.py
    refex.py
  store/
    export_parquet.py
  models/
    train_rf_baseline.py
```

Algorithms:
- Compute features in Cypher + pandas; persist to `data_fin/feature_store/`.
- ReFeX recursive features for structural roles.

Evaluation: feature importance; PR-AUC on AML-style labels.

DoD: feature store reproducible from a fresh import; baseline RF beats logistic on PR-AUC.

---

### ch13_fin — Financial GNN Foundations

Source mapping: ch13 (Karate Club + GNN intro).
Objective: Establish PyG pipeline on a small synthetic counterparty graph, then on a real subgraph.

Files:
```
ch13_fin/
  listings/
    01_pyg_intro_synthetic_counterparty.py
    02_compare_gcn_gat_sage_gin.py
```

Evaluation: accuracy on synthetic node-classification; convergence curves.

DoD: side-by-side comparison plot of architectures.

---

### ch14_fin — Node Classification + Link Prediction (PyG)

Source mapping: ch14 (PyG training pipelines).
Objective: Production-style training for AML node classification and ownership/board-link prediction with temporal splits.

Files (mirroring book layout):
```
ch14_fin/
  train_for_classification.py
  train_for_link_prediction.py
  model/
    gnn_model.py
    hetero_model.py
    task_model.py
    util_model.py
  eval/
    eval_funcs.py
    eval_reports.py
  plot/
    plot_metrics.py
    plot_conf_mtx.py
  data/
    build_pyg_dataset.py        # from Neo4j -> PyG HeteroData
```

Algorithms:
- HeteroGNN over `(:LegalEntity)`, `(:Instrument)`, `(:Transaction)` with `OWNS`, `ISSUES`, `EXPOSED_TO`.
- Temporal split by `asOf`/`valueDate`; no future leakage.
- Calibration (Platt / isotonic) for production-grade probabilities.

Evaluation: PR-AUC, ROC-AUC, calibration ECE; link-prediction Hits@K and MRR.

DoD: PR-AUC ≥ 0.7 on AML label; Hits@10 ≥ 0.4 on link prediction; calibrated outputs.

---

### ch15_fin — Financial Graph RAG

Source mapping: ch15 (vector + KG RAG with LangChain).
Objective: Hybrid retrieval that fuses dense vector search over `Chunk` with schema-aware Cypher generation, returning answers with citations and contradiction flags.

Files:
```
ch15_fin/
  code/
    tools.py                   # vector_search, kg_reader, kg_doc_selector
    definitions.py             # schema description for prompt
    contradiction_check.py
  prompts/
    prompt_structured_fin.txt
    cypher_generator.txt
  listings/
    01_conversational_agent.py
    02_chunk_embed_pipeline.py
    03_react_agent_with_graph_rag.py
```

Algorithms:
- Vector search over `Chunk.embedding` with metadata filters (asOf range, formType, issuer LEI).
- Schema-aware Cypher generation with allow-list of labels/rels; AST-validated before execution.
- Contradiction check: compare numeric claims in answer to `StatementItem` values; flag mismatches.
- Citation tracking: every claim links back to `Chunk` or `StatementItem` IDs.

Evaluation:
- Faithfulness: % of claims supported by retrieved evidence.
- Citation precision.
- Answer correctness vs gold.

DoD: faithfulness ≥ 0.9, citation precision ≥ 0.95 on a 100-question gold set.

---

### ch16_fin — Integration, Governance, Test Harness (new)

Source mapping: none in book (filling the gap).
Objective: Make the whole stack releasable: data contracts, schema migrations, lineage, regression tests.

Files:
```
ch16_fin/
  contracts/
    legal_entity.contract.yaml
    instrument.contract.yaml
    filing.contract.yaml
  migrations/
    20260418_001_init.cypher
    ...
  lineage/
    lineage_emitter.py
  tests/
    test_importers.py
    test_ner_ned_regression.py
    test_rag_regression.py
  ci/
    github_actions.yml
```

Algorithms:
- Contracts validated at import time (pydantic).
- Migrations applied via a small runner that records `applied_migrations` in a `:Migration` node.
- Lineage emitted to `data_fin/lineage/` and optionally OpenLineage.

DoD: `make ci` runs lint + tests + a small end-to-end import on a sample dataset, in under 10 minutes locally.

---

### ch17_fin — Financial Investigative Copilot (Streamlit + LangGraph)

Source mapping: ch17 (LangGraph investigator app).
Objective: An analyst-facing app that combines KG investigation, Graph RAG, ownership maps, event timelines, and what-if exposure probes.

Files:
```
ch17_fin/
  app.py
  Makefile
  app.config.yaml
  importer/
    import_seed.py             # if running standalone
  chains/
    investigator.py
    chain_config.yaml
    templates/
      system.txt
      tools.txt
  tools/
    schema.py                  # consumes _platform/config/schema_config.yaml
    cypher_safety.py           # parser + allow-list
    map_renderer.py
    graph_renderer.py
    timeline_renderer.py
```

UI panels:
- Entity profile (LE/Instrument).
- Ownership map (folium / pydeck).
- Event timeline (st_timeline).
- Exposure path explorer with adjustable hop limit.
- Q&A panel powered by ch15_fin Graph RAG with citations.

Safety:
- Cypher generated by the LLM is parsed; only allow-listed labels/rels/clauses execute.
- Read-only role on Neo4j for the app.

DoD: end-to-end demo answering 5 scripted analyst questions with citations and graph visualizations.

---

## 5. Phased Delivery Plan

Phase 1 — Foundation
- `_platform/`, ch02_fin, ch03_fin, ch04_fin, ch05_fin.
- Outcome: a populated `fin_core` graph with reconciled entities and ontologies.

Phase 2 — Information Extraction
- ch06_fin, ch07_fin, ch08_fin, ch09_fin, ch10_fin.
- Outcome: documents, mentions, events, NED with calibrated confidence.

Phase 3 — Graph ML
- ch11_fin, ch12_fin, ch13_fin, ch14_fin.
- Outcome: feature store, embeddings, GNN models with eval reports.

Phase 4 — Applications + Governance
- ch15_fin, ch16_fin, ch17_fin.
- Outcome: Graph RAG service, governed releases, analyst app.

Each phase ends with an integration checkpoint: a notebook that exercises the whole stack and emits a metrics report.

---

## 6. Cross-Chapter Evaluation Harness

Single CLI: `python -m ChaptersFinancial._platform.eval.run --suite all`.

Suites:
- `ner` (ch06): F1 against `eval/gold_ner.jsonl`.
- `ned` (ch09/ch10): accuracy@k, MRR, ECE.
- `ml` (ch11/ch12/ch14): ROC-AUC, PR-AUC, Hits@K, calibration.
- `rag` (ch15): faithfulness, citation precision, answer correctness.
- `import` (ch03-ch05): row-count and constraint checks.

All suite outputs append to `data_fin/runs/metrics.jsonl` and update a `Run` node in Neo4j.

---

## 7. Data, Licensing, and Reproducibility

- Each dataset has an entry in `DATA_SOURCES.md` with: URL, license, refresh cadence, sample size, gitignore rule.
- Pilot uses *only* openly licensed subsets (GLEIF, SEC EDGAR, ISO codes, FIBO, central bank pages).
- Proprietary feeds (Bloomberg/Refinitiv/IDC) are pluggable via adapters and never required for the public flow.
- `data_fin/samples/` ships a tiny deterministic sample so a fresh clone can run `make demo` end to end.

---

## 8. Operational Concerns (often missed)

- Secrets via `.env` + `direnv`; never `config.ini`.
- Rate limiting at the provider layer; surfaced as metrics.
- Cost tracking per `Run` (LLM tokens, API calls, Neo4j query time).
- Idempotent re-runs: every importer accepts `--since` and uses MERGE.
- Migrations are forward-only and recorded.
- Backups: `make backup` dumps `fin_core` to `data_fin/backups/<runId>.dump`.

---

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Ontology bloat from full FIBO load | Whitelist modules; lazy-load classes referenced by data |
| LLM hallucination in extraction | JSON schema + retry + reconciler + spot eval gate before merge |
| NED drift across vendors | Calibration set rebuilt quarterly; monitor ECE per run |
| Data leakage in GNN training | Strict temporal splits + frozen splits committed to repo |
| Cost blowups | Provider-level budget caps; cache-first; nightly cost report |
| Cypher injection via LLM | AST parser + allow-list + read-only DB role for app |
| Schema drift breaking apps | ch16_fin contracts + migrations + CI |

---

## 10. Definition of Done (Stack-Level)

The financial stack is "done" when:
1. `make demo` on a clean clone completes Phase 1-4 on the sample dataset in under 30 minutes locally.
2. Eval harness reports meet the per-chapter DoD thresholds.
3. The Streamlit app answers the 5 scripted analyst questions with citations and visualizations.
4. CI (`ch16_fin/ci`) is green on `main`.
5. `DATA_SOURCES.md` and `README.md` allow a new contributor to bootstrap without help.

---

## 11. Immediate Next Actions (this week)

1. Create `ChaptersFinancial/_platform/` skeleton with `fin_importer_base.py`, `providers/llm.py`, `providers/graph.py`, and `config/schema_config.yaml`.
2. Implement ch03_fin importers (FIBO subset + GLEIF sample) and verify constraints.
3. Stand up ch04_fin enrichment for a 50-issuer pilot universe; run Louvain.
4. Draft ch05_fin reconciliation with golden set; lock acceptance thresholds.
5. Wire `_platform/eval/` and a single `Run` node convention before any LLM work begins.
