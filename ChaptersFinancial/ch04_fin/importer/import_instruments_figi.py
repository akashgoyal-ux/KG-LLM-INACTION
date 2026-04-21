"""
import_instruments_figi.py
==========================
Map a set of tickers to FIGI identifiers using the OpenFIGI API (free, no key
required for small volumes).

For each resolved ticker, creates:
  (:Instrument {figi, ticker, assetClass, currency, ...})
  (:Instrument)-[:ISSUES]->(:LegalEntity)        (if issuer LEI found)
  (:Instrument)-[:LISTED_ON]->(:Exchange)         (if MIC found)

Ticker universe: Configured via FIGI_TICKERS env var (comma-separated) or
defaults to a broad set of major US-listed companies.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

# Default: a diverse set of major tickers spanning sectors.
_DEFAULT_TICKERS = (
    "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,BRK.B,JPM,V,"
    "JNJ,UNH,PG,HD,MA,XOM,CVX,PFE,ABBV,KO,"
    "PEP,MRK,LLY,BAC,WFC,COST,TMO,CSCO,ACN,AVGO,"
    "MCD,ABT,TXN,DHR,NEE,PM,UNP,LOW,RTX,HON,"
    "SCHW,AMGN,GS,BLK,AXP,SYK,ISRG,MDLZ,ADI,GILD"
)

# OpenFIGI returns exchCode (short code) in responses, but micCode is often empty
# for composite-level queries.  This table maps exchCode → ISO 10383 MIC so we
# can create LISTED_ON edges even without an explicit micCode field.
_EXCHCODE_TO_MIC: dict[str, str] = {
    # US venues
    "UN": "XNYS",   # NYSE
    "UW": "XNAS",   # NASDAQ Global Select Market
    "UQ": "XNAS",   # NASDAQ Global Market
    "UR": "XNCM",   # NASDAQ Capital Market
    "UM": "XNMS",   # NASDAQ Global Market (alt code)
    "UA": "XASE",   # NYSE American (AMEX)
    "UT": "XNAS",   # NASDAQ SmallCap
    "UF": "XNYS",   # NYSE Arca fallback
    "UP": "ARCX",   # NYSE Arca
    "UC": "XCBO",   # CBOE
    # European
    "LN": "XLON",   # London Stock Exchange
    "GY": "XETR",   # XETRA (Frankfurt)
    "FP": "XPAR",   # Euronext Paris
    "NA": "XAMS",   # Euronext Amsterdam
    "BB": "XBRU",   # Euronext Brussels
    "IM": "XMIL",   # Borsa Italiana
    "SM": "XMAD",   # Bolsa de Madrid
    "SS": "XSTO",   # NASDAQ Stockholm
    "HO": "XHEL",   # Nasdaq Helsinki
    "DC": "XCSE",   # Nasdaq Copenhagen
    # Asia-Pacific
    "JP": "XTKS",   # Tokyo Stock Exchange
    "HK": "XHKG",   # Hong Kong Stock Exchange
    "AU": "XASX",   # ASX
    "CH": "XSWX",   # SIX Swiss Exchange
    "CN": "XSHG",   # Shanghai Stock Exchange
    "CG": "XSHE",   # Shenzhen Stock Exchange
}

# Priority order for selecting the best US listing from OpenFIGI results.
# UW = NASDAQ Global Select, UN = NYSE, UA = NYSE American, UP = NYSE Arca, UQ/UR = NASDAQ tiers
_US_PRIMARY_EXCHCODES = ["UW", "UN", "UA", "UP", "UQ", "UR", "UM"]


def _pick_best_listing(data_list: list[dict]) -> dict:
    """
    From multiple OpenFIGI listings for one ticker, select the canonical US
    primary-exchange Common Stock entry.  Priority:
      1. Common Stock on a preferred US primary exchange (UW, UN, UA, UP, …)
      2. Any Common Stock
      3. First result as last resort
    """
    common_us = [
        d for d in data_list
        if d.get("securityType2") == "Common Stock"
        and d.get("exchCode") in _US_PRIMARY_EXCHCODES
    ]
    if common_us:
        # Respect the priority order
        for code in _US_PRIMARY_EXCHCODES:
            for d in common_us:
                if d.get("exchCode") == code:
                    return d
    common = [d for d in data_list if d.get("securityType2") == "Common Stock"]
    return common[0] if common else data_list[0]

_MERGE_INSTRUMENT = """
UNWIND $batch AS row
MERGE (i:Instrument {figi: row.figi})
SET i.ticker     = row.ticker,
    i.name       = row.name,
    i.assetClass = row.assetClass,
    i.currency   = row.currency,
    i.exchCode   = row.exchCode,
    i.mic        = row.mic,
    i.secType    = row.secType,
    i.runId      = row.runId,
    i.ingestedAt = row.ingestedAt
"""

_LINK_LISTED_ON = """
UNWIND $batch AS row
MATCH (i:Instrument {figi: row.figi})
MATCH (ex:Exchange {mic: row.mic})
MERGE (i)-[:LISTED_ON]->(ex)
"""


class FIGIImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch04_fin.import_instruments_figi")
        raw = os.getenv("FIGI_TICKERS", _DEFAULT_TICKERS)
        self.tickers = [t.strip() for t in raw.split(",") if t.strip()]

    def _lookup_batch_for_exchange(
        self, client: httpx.Client, tickers: list[str], exch_code: str | None
    ) -> dict[str, dict]:
        """
        Probe one specific exchCode (or no filter) for a batch of tickers.
        Returns a dict {ticker: instrument_dict} for tickers that resolved.
        """
        if exch_code:
            payload = [
                {"idType": "TICKER", "idValue": t, "exchCode": exch_code}
                for t in tickers
            ]
        else:
            payload = [
                {"idType": "TICKER", "idValue": t, "marketSecDes": "Equity"}
                for t in tickers
            ]
        resp = client.post(_OPENFIGI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        results = resp.json()

        resolved: dict[str, dict] = {}
        for i, item in enumerate(results):
            data_list = item.get("data", [])
            if not data_list:
                continue
            d = _pick_best_listing(data_list) if exch_code is None else data_list[0]
            ec = d.get("exchCode", "")
            mic = d.get("micCode", "") or _EXCHCODE_TO_MIC.get(ec, "")
            # Use compositeFIGI as the stable node key — it is the same across
            # all exchange-specific listings of the same security.
            figi = d.get("compositeFIGI") or d.get("figi", "")
            resolved[tickers[i]] = {
                "figi": figi,
                "ticker": tickers[i],
                "name": d.get("name", ""),
                "assetClass": d.get("securityType2", d.get("securityType", "")),
                "currency": d.get("marketSector", ""),
                "exchCode": ec,
                "mic": mic,
                "secType": d.get("securityType", ""),
            }
        return resolved

    def _probe_pass(
        self,
        client: httpx.Client,
        tickers: list[str],
        exch_code: str | None,
        resolved: dict[str, dict],
        label: str,
    ) -> None:
        """Run one probing pass for unresolved tickers at a given exchange."""
        remaining = [t for t in tickers if t not in resolved]
        if not remaining:
            return
        batches = [remaining[i:i + 10] for i in range(0, len(remaining), 10)]
        for batch in tqdm(batches, desc=label):
            resolved.update(self._lookup_batch_for_exchange(client, batch, exch_code))
            time.sleep(0.5)

    def _fetch_all(self) -> list[dict]:
        """
        Look up tickers in three sequential passes to maximise MIC resolution:
          Pass 1: exchCode=UW (NASDAQ Global Select) — resolves NASDAQ stocks
          Pass 2: exchCode=UN (NYSE)               — resolves NYSE stocks
          Pass 3: no exchCode filter + best-listing  — catches anything else
        """
        resolved: dict[str, dict] = {}
        with httpx.Client() as client:
            self._probe_pass(client, self.tickers, "UW", resolved, "OpenFIGI NASDAQ (UW)")
            self._probe_pass(client, self.tickers, "UN", resolved, "OpenFIGI NYSE (UN)")
            self._probe_pass(client, self.tickers, None, resolved, "OpenFIGI fallback")
        return list(resolved.values())

    def run(self):
        print("[ch04_fin] Instrument (OpenFIGI) Import")
        print("=" * 60)
        print(f"  Tickers to resolve: {len(self.tickers)}")

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        print("\n2. Resolving tickers via OpenFIGI …")
        instruments = self._fetch_all()
        print(f"  Resolved {len(instruments)} / {len(self.tickers)} tickers.")

        if not instruments:
            print("  No instruments resolved.")
            return

        print("\n3. Merging Instrument nodes …")
        self.batch_store(_MERGE_INSTRUMENT, iter(instruments), size=len(instruments),
                         desc="Instruments")

        print("\n4. Linking instruments to exchanges …")
        with_mic = [i for i in instruments if i.get("mic")]
        if with_mic:
            self.batch_store(_LINK_LISTED_ON, iter(with_mic), size=len(with_mic),
                             desc="LISTED_ON")
            print(f"  Created {len(with_mic)} LISTED_ON relationships.")
        else:
            print("  No MIC codes to link.")

        print("\n5. Verification …")
        with self._driver.session(database=self._database) as session:
            cnt = session.run("MATCH (i:Instrument) RETURN count(i) AS cnt").single()["cnt"]
            print(f"  Instrument nodes: {cnt}")
            listed = session.run(
                "MATCH ()-[r:LISTED_ON]->() RETURN count(r) AS cnt"
            ).single()["cnt"]
            print(f"  LISTED_ON relationships: {listed}")
            for rec in session.run(
                "MATCH (i:Instrument) "
                "RETURN i.figi AS figi, i.ticker AS ticker, i.name AS name "
                "ORDER BY i.ticker LIMIT 10"
            ):
                print(f"    {rec['figi']}  {rec['ticker']:<6s}  {rec['name']}")

        print("\nDone.")


if __name__ == "__main__":
    importer = FIGIImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
