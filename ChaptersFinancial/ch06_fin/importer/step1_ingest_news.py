"""
step1_ingest_news.py
====================
Fetch real financial news from public feeds and store as Document/Chunk nodes.

Data sources (all public, no API key required):
  - SEC EDGAR RSS: latest filings feed
  - Federal Reserve press releases
  - ECB press releases
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx

try:
    import feedparser
except ImportError:
    feedparser = None

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

# Public RSS / Atom feeds
_FEEDS = {
    "SEC_EDGAR": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=40&search_text=&start=0&output=atom",
    "FED_PRESS": "https://www.federalreserve.gov/feeds/press_all.xml",
}

_HEADERS = {
    "User-Agent": "KG-LLM-INACTION/1.0 (financial-kg-research; +https://github.com/neo4j)",
    "Accept": "application/xml, application/atom+xml, text/xml",
}


class NewsIngester(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch06_fin.ingest_news")

    def _fetch_feed(self, name: str, url: str) -> list[dict]:
        """Fetch an RSS/Atom feed and return parsed entries."""
        print(f"  Fetching {name} …")
        try:
            resp = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            entries = []
            for entry in feed.entries[:20]:  # limit to 20 per feed
                doc_id = hashlib.sha256(
                    (entry.get("id") or entry.get("link", str(uuid.uuid4()))).encode()
                ).hexdigest()[:16]
                entries.append({
                    "docId": f"{name}_{doc_id}",
                    "title": entry.get("title", ""),
                    "text": entry.get("summary", entry.get("description", "")),
                    "publishedAt": entry.get("published", datetime.utcnow().isoformat()),
                    "source": name,
                    "type": "NEWS",
                    "lang": "en",
                    "link": entry.get("link", ""),
                })
            print(f"    Got {len(entries)} entries.")
            return entries
        except Exception as exc:
            print(f"    [WARN] Could not fetch {name}: {exc}")
            return []

    def _store_documents(self, docs: list[dict]):
        """Merge Document nodes into Neo4j."""
        query = """
        UNWIND $batch AS row
        MERGE (d:Document {docId: row.docId})
        SET d.title       = row.title,
            d.publishedAt = row.publishedAt,
            d.source      = row.source,
            d.type        = row.type,
            d.lang        = row.lang,
            d.sourceUrl   = row.link,
            d.hash        = row.hash,
            d.runId       = row.runId,
            d.ingestedAt  = row.ingestedAt
        """
        for doc in docs:
            doc["hash"] = hashlib.sha256((doc.get("text") or "").encode()).hexdigest()
        self.batch_store(query, iter(docs), size=len(docs), desc="Storing documents")

    def _chunk_and_store(self, docs: list[dict]):
        """Split documents into chunks and store as Chunk nodes."""
        chunks = []
        for doc in docs:
            text = doc.get("text", "")
            # Simple chunking: split at ~500 chars with 100 char overlap
            chunk_size, overlap = 500, 100
            for i, start in enumerate(range(0, max(len(text), 1), chunk_size - overlap)):
                chunk_text = text[start:start + chunk_size]
                if not chunk_text.strip():
                    continue
                chunks.append({
                    "chunkId": f"{doc['docId']}_c{i}",
                    "docId": doc["docId"],
                    "ord": i,
                    "text": chunk_text,
                })
        if not chunks:
            return

        query = """
        UNWIND $batch AS row
        MERGE (c:Chunk {chunkId: row.chunkId})
        SET c.docId = row.docId, c.ord = row.ord, c.text = row.text,
            c.runId = row.runId, c.ingestedAt = row.ingestedAt
        WITH c, row
        MATCH (d:Document {docId: row.docId})
        MERGE (c)-[:OF_DOC]->(d)
        """
        self.batch_store(query, iter(chunks), size=len(chunks), desc="Storing chunks")

    def run(self):
        print("[ch06_fin] News Ingestion from Public Feeds")
        print("=" * 60)

        if feedparser is None:
            print("[ERROR] feedparser not installed. Run: pip install feedparser")
            return

        self.ensure_constraints()

        all_docs = []
        for name, url in _FEEDS.items():
            all_docs.extend(self._fetch_feed(name, url))

        if not all_docs:
            print("\n  No documents fetched. Check network connectivity.")
            return

        print(f"\n  Total documents: {len(all_docs)}")

        print("\n2. Storing documents …")
        self._store_documents(all_docs)

        print("\n3. Chunking and storing …")
        self._chunk_and_store(all_docs)

        print("\n4. Verification …")
        with self._driver.session(database=self._database) as session:
            doc_cnt = session.run("MATCH (d:Document) RETURN count(d) AS cnt").single()["cnt"]
            chunk_cnt = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
            print(f"   Documents: {doc_cnt}")
            print(f"   Chunks: {chunk_cnt}")

        print("\nDone.")


if __name__ == "__main__":
    ingester = NewsIngester(argv=sys.argv[1:])
    try:
        ingester.run()
    finally:
        ingester.close()
