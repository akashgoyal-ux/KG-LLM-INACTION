"""
import_fibo.py
==============
Load a subset of the FIBO (Financial Industry Business Ontology) into Neo4j
using Neosemantics (n10s).

Modules loaded: Business Entities (BE), Financial Business & Commerce (FBC),
Securities (SEC), Indices (IND).  These provide the class hierarchy for
legal-entity types, instrument types, and market concepts.

The resulting `OntologyClass` nodes are tagged with source='FIBO' and carry
their full IRI, rdfs:label, and rdfs:comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from ch03_fin/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

# FIBO Production RDF URLs (OWL/RDF-XML).  We load the "About" entry points
# which transitively pull the module contents.
_FIBO_MODULES = {
    "BE": "https://spec.edmcouncil.org/fibo/ontology/BE/MetadataBE/BEDomain",
    "FBC": "https://spec.edmcouncil.org/fibo/ontology/FBC/MetadataFBC/FBCDomain",
    "SEC": "https://spec.edmcouncil.org/fibo/ontology/SEC/MetadataSEC/SECDomain",
    "IND": "https://spec.edmcouncil.org/fibo/ontology/IND/MetadataIND/INDDomain",
}


class FIBOImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch03_fin.import_fibo")

    # ------------------------------------------------------------------
    # n10s initialisation
    # ------------------------------------------------------------------
    def _init_n10s(self):
        """Configure Neosemantics graph config (safe to re-run)."""
        with self._driver.session(database=self._database) as session:
            # Check if graph config already exists
            result = session.run(
                "CALL n10s.graphconfig.show() YIELD param RETURN count(param) AS cnt"
            ).single()
            if result["cnt"] == 0:
                session.run(
                    "CALL n10s.graphconfig.init({"
                    "  handleVocabUris: 'MAP',"
                    "  handleMultival: 'ARRAY',"
                    "  handleRDFTypes: 'LABELS_AND_NODES',"
                    "  keepLangTag: false,"
                    "  keepCustomDataTypes: false"
                    "})"
                )
                print("  n10s graph config initialised.")
            else:
                print("  n10s graph config already exists — skipping.")

            # Ensure n10s uniqueness constraint on Resource.uri
            try:
                session.run(
                    "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS "
                    "FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Load FIBO modules
    # ------------------------------------------------------------------
    def _load_fibo_module(self, name: str, url: str):
        """Fetch one FIBO module via n10s.rdf.import.fetch."""
        print(f"  Loading FIBO module {name} from {url} …")
        with self._driver.session(database=self._database) as session:
            result = session.run(
                "CALL n10s.rdf.import.fetch($url, 'RDF/XML') "
                "YIELD terminationStatus, triplesLoaded, extraInfo "
                "RETURN terminationStatus, triplesLoaded, extraInfo",
                url=url,
            ).single()
            status = result["terminationStatus"]
            triples = result["triplesLoaded"]
            extra = result["extraInfo"]
            print(f"    {name}: status={status}, triples={triples}")
            if extra:
                print(f"    extra: {extra}")

    # ------------------------------------------------------------------
    # Relabel to OntologyClass
    # ------------------------------------------------------------------
    def _relabel_to_ontology_class(self):
        """
        Tag every Resource that is an owl:Class with the :OntologyClass label,
        and set source='FIBO'.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run("""
                MATCH (r:Resource)
                WHERE r.uri STARTS WITH 'https://spec.edmcouncil.org/fibo/'
                SET r:OntologyClass, r.source = 'FIBO', r.iri = r.uri,
                    r.label = CASE
                      WHEN r.`rdfs__label` IS NOT NULL
                        AND size(r.`rdfs__label`) > 0
                        AND r.`rdfs__label`[0] <> ''
                        THEN r.`rdfs__label`[0]
                      ELSE
                        CASE
                          WHEN split(r.uri, '/')[-1] <> ''
                            THEN split(r.uri, '/')[-1]
                          ELSE split(r.uri, '/')[-2]
                        END
                    END
                RETURN count(r) AS cnt
            """).single()
            print(f"  Labelled {result['cnt']} FIBO resources as OntologyClass.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        print("[ch03_fin] FIBO Ontology Import")
        print("=" * 60)

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        print("\n2. Initialising n10s …")
        self._init_n10s()

        print("\n3. Loading FIBO modules …")
        for name, url in _FIBO_MODULES.items():
            try:
                self._load_fibo_module(name, url)
            except Exception as exc:
                print(f"  [WARN] Could not load {name}: {exc}")
                print(f"  Skipping {name} — FIBO servers may be temporarily unavailable.")

        print("\n4. Relabelling to OntologyClass …")
        self._relabel_to_ontology_class()

        print("\n5. Verification …")
        self._verify()

        print("\nDone.")

    def _verify(self):
        with self._driver.session(database=self._database) as session:
            cnt = session.run(
                "MATCH (oc:OntologyClass {source: 'FIBO'}) RETURN count(oc) AS cnt"
            ).single()["cnt"]
            print(f"  OntologyClass nodes (FIBO): {cnt}")

            # Show a few sample classes
            sample = session.run(
                "MATCH (oc:OntologyClass {source: 'FIBO'}) "
                "RETURN oc.label AS label, oc.iri AS iri "
                "ORDER BY oc.label LIMIT 10"
            )
            for rec in sample:
                lbl = rec['label']
                if isinstance(lbl, list):
                    lbl = lbl[0] if lbl else ""
                print(f"    {str(lbl):<40s} {rec['iri']}")


if __name__ == "__main__":
    importer = FIBOImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
