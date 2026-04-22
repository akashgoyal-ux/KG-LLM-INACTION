"""
deterministic_match.py
======================
Exact-join reconciliation across canonical identifiers:
  LEI ↔ CIK, FIGI ↔ ISIN/CUSIP, ticker ↔ instrument name.

Creates :Crosswalk nodes linking pairs of authoritative IDs and
MERGE (:LegalEntity)-[:SAME_AS]->(:LegalEntity) where CIK maps to LEI.

Uses real data already in Neo4j from ch03_fin/ch04_fin imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase


class DeterministicMatcher(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch05_fin.deterministic_match")

    # ------------------------------------------------------------------
    # Crosswalk: LEI ↔ CIK
    # ------------------------------------------------------------------
    def match_lei_cik(self):
        """
        Find LegalEntity nodes that have both lei and cik populated
        (from GLEIF + SEC data) and create :Crosswalk nodes.
        """
        print("  Matching LEI ↔ CIK …")
        query = """
        MATCH (le:LegalEntity)
        WHERE le.lei IS NOT NULL AND le.cik IS NOT NULL
        MERGE (cw:Crosswalk {
            idA: le.lei, idTypeA: 'LEI',
            idB: le.cik, idTypeB: 'CIK'
        })
        SET cw.matchType = 'DETERMINISTIC',
            cw.confidence = 1.0,
            cw.runId = $runId,
            cw.ingestedAt = datetime()
        MERGE (cw)-[:LINKS]->(le)
        RETURN count(cw) AS cnt
        """
        with self._driver.session(database=self._database) as session:
            cnt = session.run(query, runId=self.run_id).single()["cnt"]
            print(f"    Created {cnt} LEI↔CIK crosswalks.")

    # ------------------------------------------------------------------
    # Crosswalk: FIGI ↔ ISIN
    # ------------------------------------------------------------------
    def match_figi_isin(self):
        """Link instruments that have both FIGI and ISIN."""
        print("  Matching FIGI ↔ ISIN …")
        query = """
        MATCH (i:Instrument)
        WHERE i.figi IS NOT NULL AND i.isin IS NOT NULL
        MERGE (cw:Crosswalk {
            idA: i.figi, idTypeA: 'FIGI',
            idB: i.isin, idTypeB: 'ISIN'
        })
        SET cw.matchType = 'DETERMINISTIC',
            cw.confidence = 1.0,
            cw.runId = $runId,
            cw.ingestedAt = datetime()
        MERGE (cw)-[:LINKS]->(i)
        RETURN count(cw) AS cnt
        """
        with self._driver.session(database=self._database) as session:
            cnt = session.run(query, runId=self.run_id).single()["cnt"]
            print(f"    Created {cnt} FIGI↔ISIN crosswalks.")

    # ------------------------------------------------------------------
    # Crosswalk: FIGI ↔ CUSIP
    # ------------------------------------------------------------------
    def match_figi_cusip(self):
        """Link instruments that have both FIGI and CUSIP."""
        print("  Matching FIGI ↔ CUSIP …")
        query = """
        MATCH (i:Instrument)
        WHERE i.figi IS NOT NULL AND i.cusip IS NOT NULL
        MERGE (cw:Crosswalk {
            idA: i.figi, idTypeA: 'FIGI',
            idB: i.cusip, idTypeB: 'CUSIP'
        })
        SET cw.matchType = 'DETERMINISTIC',
            cw.confidence = 1.0,
            cw.runId = $runId,
            cw.ingestedAt = datetime()
        MERGE (cw)-[:LINKS]->(i)
        RETURN count(cw) AS cnt
        """
        with self._driver.session(database=self._database) as session:
            cnt = session.run(query, runId=self.run_id).single()["cnt"]
            print(f"    Created {cnt} FIGI↔CUSIP crosswalks.")

    # ------------------------------------------------------------------
    # Issuer ↔ Instrument linkage via issuerLei
    # ------------------------------------------------------------------
    def link_issuers_to_instruments(self):
        """Create ISSUES edges where Instrument.issuerLei matches LegalEntity.lei."""
        print("  Linking issuers to instruments …")
        query = """
        MATCH (i:Instrument), (le:LegalEntity)
        WHERE i.issuerLei IS NOT NULL AND i.issuerLei = le.lei
        MERGE (le)-[:ISSUES]->(i)
        RETURN count(*) AS cnt
        """
        with self._driver.session(database=self._database) as session:
            cnt = session.run(query).single()["cnt"]
            print(f"    Created/confirmed {cnt} ISSUES edges.")

    # ------------------------------------------------------------------
    # Duplicate detection via APOC
    # ------------------------------------------------------------------
    def detect_duplicate_entities(self):
        """
        Use APOC text similarity to find potential duplicate LegalEntity nodes
        with very similar names but different LEIs.
        """
        print("  Detecting potential duplicates via name similarity …")
        query = """
        MATCH (a:LegalEntity), (b:LegalEntity)
        WHERE id(a) < id(b)
          AND a.jurisdiction = b.jurisdiction
          AND a.name IS NOT NULL AND b.name IS NOT NULL
          AND apoc.text.jaroWinklerDistance(
                apoc.text.clean(a.name),
                apoc.text.clean(b.name)
              ) > 0.92
        MERGE (cw:Crosswalk {
            idA: a.lei, idTypeA: 'LEI',
            idB: b.lei, idTypeB: 'LEI'
        })
        SET cw.matchType = 'DETERMINISTIC_NAME',
            cw.confidence = apoc.text.jaroWinklerDistance(
                apoc.text.clean(a.name), apoc.text.clean(b.name)),
            cw.runId = $runId,
            cw.ingestedAt = datetime(),
            cw.needsReview = true
        RETURN count(cw) AS cnt
        """
        try:
            with self._driver.session(database=self._database) as session:
                cnt = session.run(query, runId=self.run_id).single()["cnt"]
                print(f"    Found {cnt} potential duplicates (queued for review).")
        except Exception as exc:
            print(f"    [WARN] APOC text similarity not available: {exc}")
            print("    Install neo4j-apoc-core for duplicate detection.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        print("[ch05_fin] Deterministic Entity Reconciliation")
        print("=" * 60)

        print("\n1. Ensuring constraints …")
        self.ensure_constraints()
        # Add Crosswalk constraint
        with self._driver.session(database=self._database) as session:
            try:
                session.run(
                    "CREATE CONSTRAINT crosswalk_unique IF NOT EXISTS "
                    "FOR (cw:Crosswalk) REQUIRE (cw.idA, cw.idTypeA, cw.idB, cw.idTypeB) IS UNIQUE"
                )
            except Exception:
                pass

        print("\n2. LEI ↔ CIK matching …")
        self.match_lei_cik()

        print("\n3. FIGI ↔ ISIN matching …")
        self.match_figi_isin()

        print("\n4. FIGI ↔ CUSIP matching …")
        self.match_figi_cusip()

        print("\n5. Issuer ↔ Instrument linkage …")
        self.link_issuers_to_instruments()

        print("\n6. Duplicate detection …")
        self.detect_duplicate_entities()

        # Summary
        print("\n7. Summary …")
        with self._driver.session(database=self._database) as session:
            cnt = session.run("MATCH (cw:Crosswalk) RETURN count(cw) AS cnt").single()["cnt"]
            print(f"    Total Crosswalk nodes: {cnt}")
            review = session.run(
                "MATCH (cw:Crosswalk {needsReview: true}) RETURN count(cw) AS cnt"
            ).single()["cnt"]
            print(f"    Needing review: {review}")

        print("\nDone.")


if __name__ == "__main__":
    matcher = DeterministicMatcher(argv=sys.argv[1:])
    try:
        matcher.run()
    finally:
        matcher.close()
