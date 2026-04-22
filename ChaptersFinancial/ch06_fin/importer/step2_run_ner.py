"""
step2_run_ner.py
================
Run NER on Document/Chunk text using spaCy + a custom financial EntityRuler.

Produces :Mention nodes linked to :Chunk via IN_CHUNK.
Custom patterns detect: TICKER, ISIN, CUSIP, MONEY, PCT, RATING.
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

# Financial entity patterns for spaCy EntityRuler
_FINANCE_PATTERNS = [
    # ISIN: 2 letter country + 9 alphanum + 1 check digit
    {"label": "ISIN", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$"}}]},
    # CUSIP: 9 characters
    {"label": "CUSIP", "pattern": [{"TEXT": {"REGEX": r"^[0-9]{3}[A-Z0-9]{5}[0-9]$"}}]},
    # Ticker symbols (e.g., $AAPL, $MSFT)
    {"label": "TICKER", "pattern": [{"TEXT": {"REGEX": r"^\$[A-Z]{1,5}$"}}]},
    # Percentage values
    {"label": "PCT", "pattern": [{"LIKE_NUM": True}, {"TEXT": {"REGEX": r"^%$|^percent$|^pct$"}}]},
    # Credit ratings
    {"label": "RATING", "pattern": [{"TEXT": {"REGEX": r"^(AAA|AA\+|AA|AA-|A\+|A|A-|BBB\+|BBB|BBB-|BB\+|BB|BB-|B\+|B|B-|CCC|CC|C|D)$"}}]},
]

# Regex patterns for post-processing
_ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")
_TICKER_RE = re.compile(r"\$[A-Z]{1,5}\b")
_MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|T)?", re.IGNORECASE)


class NERRunner(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch06_fin.run_ner")

    def _load_spacy(self):
        """Load spaCy model with custom financial entity ruler."""
        try:
            import spacy
        except ImportError:
            raise ImportError("pip install spacy && python -m spacy download en_core_web_sm")

        try:
            nlp = spacy.load("en_core_web_trf")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Run: python -m spacy download en_core_web_sm"
                )

        # Add financial entity ruler
        if "finance_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", name="finance_ruler", before="ner")
            ruler.add_patterns(_FINANCE_PATTERNS)

        return nlp

    def _get_chunks(self) -> list[dict]:
        """Load unprocessed chunks from Neo4j."""
        with self._driver.session(database=self._database) as session:
            return session.run("""
                MATCH (c:Chunk)
                WHERE c.nerProcessed IS NULL OR c.nerProcessed = false
                RETURN c.chunkId AS chunkId, c.text AS text
                ORDER BY c.chunkId
                LIMIT 500
            """).data()

    def _extract_mentions_regex(self, text: str, chunk_id: str) -> list[dict]:
        """Extract financial entities using regex patterns."""
        mentions = []
        for match in _ISIN_RE.finditer(text):
            mentions.append({
                "mentionId": f"{chunk_id}_isin_{match.start()}",
                "text": match.group(), "start": match.start(), "end": match.end(),
                "chunkId": chunk_id, "label": "ISIN", "score": 0.95,
                "extractor": "regex",
            })
        for match in _TICKER_RE.finditer(text):
            mentions.append({
                "mentionId": f"{chunk_id}_tick_{match.start()}",
                "text": match.group(), "start": match.start(), "end": match.end(),
                "chunkId": chunk_id, "label": "TICKER", "score": 0.90,
                "extractor": "regex",
            })
        for match in _MONEY_RE.finditer(text):
            mentions.append({
                "mentionId": f"{chunk_id}_money_{match.start()}",
                "text": match.group(), "start": match.start(), "end": match.end(),
                "chunkId": chunk_id, "label": "MONEY", "score": 0.85,
                "extractor": "regex",
            })
        return mentions

    def _store_mentions(self, mentions: list[dict]):
        """Store Mention nodes and link to Chunks."""
        if not mentions:
            return
        query = """
        UNWIND $batch AS row
        MERGE (m:Mention {mentionId: row.mentionId})
        SET m.text      = row.text,
            m.start     = row.start,
            m.end       = row.end,
            m.chunkId   = row.chunkId,
            m.label     = row.label,
            m.score     = row.score,
            m.extractor = row.extractor,
            m.runId     = row.runId,
            m.ingestedAt = row.ingestedAt
        WITH m, row
        MATCH (c:Chunk {chunkId: row.chunkId})
        MERGE (m)-[:IN_CHUNK]->(c)
        """
        self.batch_store(query, iter(mentions), size=len(mentions), desc="Storing mentions")

    def _mark_chunks_processed(self, chunk_ids: list[str]):
        """Mark chunks as NER-processed."""
        with self._driver.session(database=self._database) as session:
            session.run(
                "UNWIND $ids AS cid MATCH (c:Chunk {chunkId: cid}) SET c.nerProcessed = true",
                ids=chunk_ids,
            )

    def run(self):
        print("[ch06_fin] NER Pipeline")
        print("=" * 60)

        self.ensure_constraints()

        print("\n1. Loading spaCy model …")
        nlp = self._load_spacy()

        print("\n2. Loading unprocessed chunks …")
        chunks = self._get_chunks()
        print(f"   Found {len(chunks)} chunks to process.")

        if not chunks:
            print("   No new chunks. Done.")
            return

        print("\n3. Running NER …")
        all_mentions = []
        processed_ids = []

        for chunk in chunks:
            text = chunk.get("text") or ""
            chunk_id = chunk["chunkId"]

            # spaCy NER
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PERSON", "GPE", "MONEY", "DATE", "PERCENT",
                                   "ISIN", "TICKER", "CUSIP", "RATING", "PCT"):
                    all_mentions.append({
                        "mentionId": f"{chunk_id}_sp_{ent.start_char}",
                        "text": ent.text, "start": ent.start_char, "end": ent.end_char,
                        "chunkId": chunk_id, "label": ent.label_, "score": 0.80,
                        "extractor": "spacy",
                    })

            # Regex-based extraction for financial patterns
            all_mentions.extend(self._extract_mentions_regex(text, chunk_id))
            processed_ids.append(chunk_id)

        print(f"   Extracted {len(all_mentions)} mentions from {len(chunks)} chunks.")

        print("\n4. Storing mentions …")
        self._store_mentions(all_mentions)

        print("\n5. Marking chunks processed …")
        self._mark_chunks_processed(processed_ids)

        # Link ORG mentions to existing LegalEntities using APOC fuzzy matching
        print("\n6. Linking ORG mentions to LegalEntities …")
        link_query = """
        MATCH (m:Mention {label: 'ORG'})
        WHERE NOT (m)-[:RESOLVED_TO]->()
        MATCH (le:LegalEntity)
        WHERE le.name IS NOT NULL
          AND apoc.text.jaroWinklerDistance(
                toLower(m.text), toLower(le.name)) > 0.85
        WITH m, le,
             apoc.text.jaroWinklerDistance(toLower(m.text), toLower(le.name)) AS sim
        ORDER BY sim DESC
        WITH m, collect({entity: le, score: sim})[0] AS best
        MERGE (m)-[r:RESOLVED_TO]->(best.entity)
        SET r.confidence = best.score,
            r.linker = 'apoc_jaro_winkler'
        RETURN count(r) AS linked
        """
        try:
            with self._driver.session(database=self._database) as session:
                cnt = session.run(link_query).single()["linked"]
                print(f"   Linked {cnt} ORG mentions to LegalEntities.")
        except Exception as exc:
            print(f"   [WARN] APOC linking skipped: {exc}")

        print("\n7. Verification …")
        with self._driver.session(database=self._database) as session:
            m_cnt = session.run("MATCH (m:Mention) RETURN count(m) AS cnt").single()["cnt"]
            linked = session.run(
                "MATCH (m:Mention)-[:RESOLVED_TO]->(le:LegalEntity) RETURN count(m) AS cnt"
            ).single()["cnt"]
            print(f"   Total mentions: {m_cnt}")
            print(f"   Resolved to entities: {linked}")

        print("\nDone.")


if __name__ == "__main__":
    runner = NERRunner(argv=sys.argv[1:])
    try:
        runner.run()
    finally:
        runner.close()
