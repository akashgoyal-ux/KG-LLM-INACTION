"""
FinImporterBase
===============
Extends util.base_importer.BaseImporter with financial-domain conveniences:
  - run_id injected automatically into every batch parameter dict
  - provenance helper (attach source, extractor, timestamp to node/rel)
  - merge helpers for the canonical schema nodes
  - constraint bootstrap from schema_config.yaml
"""

from __future__ import annotations

import os
import uuid
import datetime
import yaml
from pathlib import Path
from typing import Iterable, Iterator

from util.base_importer import BaseImporter

_SCHEMA_CONFIG = Path(__file__).parent / "config" / "schema_config.yaml"


def _load_schema() -> dict:
    with _SCHEMA_CONFIG.open() as f:
        return yaml.safe_load(f)


class FinImporterBase(BaseImporter):
    """
    Base class for all financial-domain importers.

    Usage
    -----
    class MyImporter(FinImporterBase):
        def __init__(self, argv):
            super().__init__(argv=argv)
            self._database = "fin_core"
            self.run_id = self.new_run_id("my_module")

        def run(self):
            rows = [{"lei": "529900...", "name": "Acme Ltd"}]
            self.merge_legal_entities(rows)
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self, command=None, argv=None):
        super().__init__(command=command, argv=argv)
        self._database = os.getenv("NEO4J_DATABASE", "fin_core")
        self._run_id: str | None = None
        self._chapter: str = "unknown"
        self._schema = _load_schema()

    # ------------------------------------------------------------------
    # Run-ID helpers
    # ------------------------------------------------------------------
    def new_run_id(self, chapter: str) -> str:
        """Generate and store a unique run ID for this import job."""
        self._chapter = chapter
        self._run_id = str(uuid.uuid4())
        return self._run_id

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            self._run_id = str(uuid.uuid4())
        return self._run_id

    @run_id.setter
    def run_id(self, value: str):
        self._run_id = value

    def _inject_run_id(self, params: Iterable[dict]) -> Iterator[dict]:
        """Inject runId + ingestedAt into every parameter dict in the stream."""
        ts = datetime.datetime.utcnow().isoformat()
        for row in params:
            row = dict(row)
            row.setdefault("runId", self.run_id)
            row.setdefault("ingestedAt", ts)
            yield row

    def batch_store(self, query: str, parameters_iterator: Iterable, size: int = None,
                    strategy: str = "aggregate", desc: str = ""):
        """Wrap parent batch_store, auto-injecting runId."""
        enriched = self._inject_run_id(parameters_iterator)
        super().batch_store(query, enriched, size=size, strategy=strategy, desc=desc)

    # ------------------------------------------------------------------
    # Constraint bootstrap
    # ------------------------------------------------------------------
    def ensure_constraints(self):
        """
        Create uniqueness constraints + indices declared in schema_config.yaml.
        Safe to re-run (uses IF NOT EXISTS).
        """
        constraints_file = Path(__file__).parent / "schema" / "constraints.cypher"
        statements = constraints_file.read_text().split(";")
        with self._driver.session(database=self._database) as session:
            for stmt in statements:
                stmt = stmt.strip()
                if stmt:
                    try:
                        session.run(stmt)
                    except Exception as exc:  # pragma: no cover
                        print(f"[WARN] constraint skipped: {exc}")

    # ------------------------------------------------------------------
    # Canonical merge helpers (idempotent MERGE + SET)
    # ------------------------------------------------------------------
    _MERGE_LEGAL_ENTITY = """
    UNWIND $batch AS row
    MERGE (le:LegalEntity {lei: row.lei})
    SET le.name         = coalesce(row.name, le.name),
        le.legalForm    = coalesce(row.legalForm, le.legalForm),
        le.jurisdiction = coalesce(row.jurisdiction, le.jurisdiction),
        le.status       = coalesce(row.status, le.status),
        le.cik          = coalesce(row.cik, le.cik),
        le.runId        = row.runId,
        le.ingestedAt   = row.ingestedAt
    """

    _MERGE_INSTRUMENT = """
    UNWIND $batch AS row
    MERGE (i:Instrument {figi: row.figi})
    SET i.isin       = coalesce(row.isin, i.isin),
        i.cusip      = coalesce(row.cusip, i.cusip),
        i.ticker     = coalesce(row.ticker, i.ticker),
        i.cfiCode    = coalesce(row.cfiCode, i.cfiCode),
        i.assetClass = coalesce(row.assetClass, i.assetClass),
        i.currency   = coalesce(row.currency, i.currency),
        i.issuerLei  = coalesce(row.issuerLei, i.issuerLei),
        i.runId      = row.runId,
        i.ingestedAt = row.ingestedAt
    """

    _MERGE_DOCUMENT = """
    UNWIND $batch AS row
    MERGE (d:Document {docId: row.docId})
    SET d.type        = coalesce(row.type, d.type),
        d.publishedAt = coalesce(row.publishedAt, d.publishedAt),
        d.source      = coalesce(row.source, d.source),
        d.title       = coalesce(row.title, d.title),
        d.lang        = coalesce(row.lang, d.lang),
        d.runId       = row.runId,
        d.ingestedAt  = row.ingestedAt
    """

    _MERGE_MENTION = """
    UNWIND $batch AS row
    MERGE (m:Mention {mentionId: row.mentionId})
    SET m.text      = row.text,
        m.start     = row.start,
        m.end       = row.end,
        m.chunkId   = row.chunkId,
        m.score     = coalesce(row.score, 0.0),
        m.extractor = coalesce(row.extractor, 'unknown'),
        m.runId     = row.runId
    """

    def merge_legal_entities(self, rows: list[dict]):
        self.batch_store(self._MERGE_LEGAL_ENTITY, iter(rows), size=len(rows),
                         desc="Merging LegalEntities")

    def merge_instruments(self, rows: list[dict]):
        self.batch_store(self._MERGE_INSTRUMENT, iter(rows), size=len(rows),
                         desc="Merging Instruments")

    def merge_documents(self, rows: list[dict]):
        self.batch_store(self._MERGE_DOCUMENT, iter(rows), size=len(rows),
                         desc="Merging Documents")

    def merge_mentions(self, rows: list[dict]):
        self.batch_store(self._MERGE_MENTION, iter(rows), size=len(rows),
                         desc="Merging Mentions")

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------
    def provenance_props(self, source_id: str = "", source_type: str = "", extractor: str = "") -> dict:
        return {
            "sourceId": source_id,
            "sourceType": source_type,
            "extractor": extractor,
            "runId": self.run_id,
            "ingestedAt": datetime.datetime.utcnow().isoformat(),
        }
