// Financial Domain – Canonical Neo4j Constraints & Indices
// Safe to re-run: uses IF NOT EXISTS (Neo4j 4.4+ / 5.x)
// Run via: FinImporterBase.ensure_constraints()

// ----------------------------------------------------------------
// LegalEntity
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_legal_entity_lei IF NOT EXISTS
  FOR (le:LegalEntity) REQUIRE le.lei IS UNIQUE;

CREATE INDEX fin_legal_entity_name IF NOT EXISTS
  FOR (le:LegalEntity) ON (le.name);

CREATE INDEX fin_legal_entity_cik IF NOT EXISTS
  FOR (le:LegalEntity) ON (le.cik);

// ----------------------------------------------------------------
// Instrument
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_instrument_figi IF NOT EXISTS
  FOR (i:Instrument) REQUIRE i.figi IS UNIQUE;

CREATE INDEX fin_instrument_isin IF NOT EXISTS
  FOR (i:Instrument) ON (i.isin);

CREATE INDEX fin_instrument_ticker IF NOT EXISTS
  FOR (i:Instrument) ON (i.ticker);

// ----------------------------------------------------------------
// Exchange
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_exchange_mic IF NOT EXISTS
  FOR (ex:Exchange) REQUIRE ex.mic IS UNIQUE;

// ----------------------------------------------------------------
// Filing
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_filing_id IF NOT EXISTS
  FOR (f:Filing) REQUIRE f.filingId IS UNIQUE;

// ----------------------------------------------------------------
// Document / Chunk / Mention
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_document_id IF NOT EXISTS
  FOR (d:Document) REQUIRE d.docId IS UNIQUE;

CREATE CONSTRAINT fin_chunk_id IF NOT EXISTS
  FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE;

CREATE CONSTRAINT fin_mention_id IF NOT EXISTS
  FOR (m:Mention) REQUIRE m.mentionId IS UNIQUE;

// ----------------------------------------------------------------
// Event / Transaction / OntologyClass
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_event_id IF NOT EXISTS
  FOR (ev:Event) REQUIRE ev.eventId IS UNIQUE;

CREATE CONSTRAINT fin_transaction_id IF NOT EXISTS
  FOR (tx:Transaction) REQUIRE tx.txId IS UNIQUE;

CREATE CONSTRAINT fin_ontology_iri IF NOT EXISTS
  FOR (oc:OntologyClass) REQUIRE oc.iri IS UNIQUE;

// ----------------------------------------------------------------
// Run (observability)
// ----------------------------------------------------------------
CREATE CONSTRAINT fin_run_id IF NOT EXISTS
  FOR (r:Run) REQUIRE r.runId IS UNIQUE
