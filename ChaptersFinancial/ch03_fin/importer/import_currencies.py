"""
import_currencies.py
====================
Load ISO 4217 currency codes as OntologyClass nodes (source='ISO4217').
Uses the ``pycountry`` package which bundles the official ISO 4217 table.

Each currency becomes:
  (:OntologyClass {iri: 'iso4217:<code>', label: '<name>', source: 'ISO4217',
                   numericCode: '<num>', currencyCode: '<code>'})
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pycountry
from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

_MERGE_CURRENCY = """
UNWIND $batch AS row
MERGE (oc:OntologyClass {iri: row.iri})
SET oc.label        = row.label,
    oc.source       = 'ISO4217',
    oc.currencyCode = row.currencyCode,
    oc.numericCode  = row.numericCode,
    oc.runId        = row.runId,
    oc.ingestedAt   = row.ingestedAt
"""


class CurrencyImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch03_fin.import_currencies")

    def _build_rows(self) -> list[dict]:
        rows = []
        for cur in pycountry.currencies:
            rows.append({
                "iri": f"iso4217:{cur.alpha_3}",
                "label": cur.name,
                "currencyCode": cur.alpha_3,
                "numericCode": getattr(cur, "numeric", ""),
            })
        return rows

    def run(self):
        print("[ch03_fin] ISO 4217 Currency Import")
        print("=" * 60)

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        rows = self._build_rows()
        print(f"\n2. Merging {len(rows)} currencies as OntologyClass nodes …")
        self.batch_store(_MERGE_CURRENCY, iter(rows), size=len(rows), desc="Currencies")

        print("\n3. Verification …")
        with self._driver.session(database=self._database) as session:
            cnt = session.run(
                "MATCH (oc:OntologyClass {source: 'ISO4217'}) RETURN count(oc) AS cnt"
            ).single()["cnt"]
            print(f"  Currency OntologyClass nodes: {cnt}")

            sample = session.run(
                "MATCH (oc:OntologyClass {source: 'ISO4217'}) "
                "RETURN oc.currencyCode AS code, oc.label AS name "
                "ORDER BY oc.currencyCode LIMIT 10"
            )
            for rec in sample:
                print(f"    {rec['code']}  {rec['name']}")

        print("\nDone.")


if __name__ == "__main__":
    importer = CurrencyImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
