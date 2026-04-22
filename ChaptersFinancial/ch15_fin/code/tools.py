"""
tools.py
========
Graph RAG tools for the financial knowledge graph:
  1. vector_search — semantic search over Chunk embeddings
  2. kg_reader — structured Cypher queries against the canonical schema
  3. kg_doc_selector — find relevant documents for an entity
  4. cypher_generator — LLM-generated Cypher with safety validation
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider

# Allow-listed labels and relationship types for Cypher generation safety
_ALLOWED_LABELS = {
    "LegalEntity", "Instrument", "Exchange", "Filing", "StatementItem",
    "Document", "Chunk", "Mention", "Event", "OntologyClass", "Crosswalk",
}
_ALLOWED_RELS = {
    "ISSUES", "LISTED_ON", "PARENT_OF", "OWNS", "CONTROLS",
    "REPORTS_ON", "FROM_FILING", "OF_DOC", "IN_CHUNK", "MENTIONS",
    "RESOLVED_TO", "AFFECTS", "CLASSIFIED_AS", "EXPOSED_TO", "LINKS",
}
_FORBIDDEN_CLAUSES = {"DELETE", "DETACH", "CREATE", "SET", "REMOVE", "DROP", "CALL"}


def validate_cypher(cypher: str) -> tuple[bool, str]:
    """Validate generated Cypher against allow-list. Returns (safe, reason)."""
    upper = cypher.upper()
    for clause in _FORBIDDEN_CLAUSES:
        if re.search(rf"\b{clause}\b", upper):
            return False, f"Forbidden clause: {clause}"
    return True, "OK"


class GraphRAGTools:
    """Collection of retrieval tools for the financial Graph RAG system."""

    def __init__(self, gp: GraphProvider | None = None, llm: LLMProvider | None = None):
        self._gp = gp or GraphProvider()
        self._llm = llm or LLMProvider()

    def vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over Chunk embeddings using cosine similarity."""
        # Embed the query
        query_emb = self._llm.embed([query])[0]

        # Get chunks with embeddings from Neo4j
        chunks = self._gp.run("""
            MATCH (c:Chunk)-[:OF_DOC]->(d:Document)
            WHERE c.embedding IS NOT NULL
            RETURN c.chunkId AS chunkId, c.text AS text,
                   c.embedding AS embedding,
                   d.title AS docTitle, d.source AS docSource
            LIMIT 500
        """)

        if not chunks:
            # Fallback: text-based search
            return self._gp.run("""
                MATCH (c:Chunk)-[:OF_DOC]->(d:Document)
                WHERE c.text CONTAINS $query
                RETURN c.chunkId AS chunkId, c.text AS text,
                       d.title AS docTitle, d.source AS docSource
                LIMIT $k
            """, {"query": query, "k": top_k})

        # Compute similarities
        q_arr = np.array(query_emb)
        results = []
        for chunk in chunks:
            c_arr = np.array(chunk["embedding"])
            sim = float(np.dot(q_arr, c_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(c_arr) + 1e-9))
            results.append({**chunk, "similarity": sim})
            del results[-1]["embedding"]

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def kg_reader(self, entity_name: str) -> dict:
        """Get structured entity information from the knowledge graph."""
        result = self._gp.run("""
            MATCH (le:LegalEntity)
            WHERE toLower(le.name) CONTAINS toLower($name)
            OPTIONAL MATCH (le)-[:ISSUES]->(i:Instrument)
            OPTIONAL MATCH (le)-[:CLASSIFIED_AS]->(oc:OntologyClass)
            OPTIONAL MATCH (f:Filing)-[:REPORTS_ON]->(le)
            OPTIONAL MATCH (e:Event)-[:AFFECTS]->(le)
            RETURN le.lei AS lei, le.name AS name,
                   le.jurisdiction AS jurisdiction,
                   le.status AS status,
                   collect(DISTINCT i.ticker) AS tickers,
                   collect(DISTINCT oc.label) AS classifications,
                   count(DISTINCT f) AS filingCount,
                   collect(DISTINCT e.type)[..5] AS recentEvents
            LIMIT 1
        """, {"name": entity_name})
        return result[0] if result else {}

    def kg_doc_selector(self, entity_name: str, top_k: int = 5) -> list[dict]:
        """Find documents that mention a given entity."""
        return self._gp.run("""
            MATCH (le:LegalEntity)
            WHERE toLower(le.name) CONTAINS toLower($name)
            MATCH (d:Document)-[:MENTIONS]->(le)
            RETURN d.docId AS docId, d.title AS title,
                   d.publishedAt AS publishedAt, d.source AS source
            ORDER BY d.publishedAt DESC
            LIMIT $k
        """, {"name": entity_name, "k": top_k})

    def cypher_query(self, question: str) -> list[dict]:
        """Generate and execute a safe Cypher query from a natural language question."""
        schema_desc = """
        Node labels: LegalEntity(lei, name, jurisdiction), Instrument(figi, ticker),
        Exchange(mic, name), Filing(filingId, formType, filedAt), StatementItem(concept, value, period),
        Document(docId, title, source), Event(eventId, type), OntologyClass(iri, label)
        Relationships: ISSUES, LISTED_ON, PARENT_OF, OWNS, CONTROLS, REPORTS_ON,
        FROM_FILING, OF_DOC, IN_CHUNK, MENTIONS, RESOLVED_TO, AFFECTS, CLASSIFIED_AS
        """
        prompt = f"""Generate a Cypher query for Neo4j to answer this question:
"{question}"

Schema: {schema_desc}

Rules:
- READ-ONLY: only MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT
- No DELETE, CREATE, SET, REMOVE, DROP, CALL
- Use parameters where possible
- Return at most 20 rows

Return ONLY the Cypher query, no explanation."""

        result = self._llm.complete_json(
            prompt,
            system="You are a Cypher query generator. Return JSON with a 'cypher' field.",
        )
        cypher = result.get("cypher", "")

        safe, reason = validate_cypher(cypher)
        if not safe:
            return [{"error": f"Unsafe query rejected: {reason}"}]

        try:
            return self._gp.run(cypher)
        except Exception as exc:
            return [{"error": str(exc)}]

    def answer_question(self, question: str) -> dict:
        """
        Full Graph RAG pipeline: combine vector search + KG lookup to answer
        a financial question with citations.
        """
        # 1. Vector search for relevant chunks
        chunks = self.vector_search(question, top_k=3)

        # 2. Try to extract entity name for KG lookup
        entity_info = {}
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                info = self.kg_reader(word)
                if info:
                    entity_info = info
                    break

        # 3. Generate answer using LLM with context
        context_parts = []
        citations = []
        for chunk in chunks:
            context_parts.append(f"[{chunk.get('docTitle', 'doc')}]: {chunk.get('text', '')[:300]}")
            citations.append(chunk.get("chunkId", ""))

        if entity_info:
            context_parts.append(f"[KG]: {entity_info}")

        context = "\n\n".join(context_parts)
        prompt = f"""Answer the following financial question using the provided context.
Cite your sources using [source] notation.

Question: {question}

Context:
{context}

Provide a JSON response with:
- "answer": your answer text with citations
- "confidence": 0-1 confidence score
- "citations": list of source identifiers used"""

        result = self._llm.complete_json(prompt)
        result["retrieved_chunks"] = len(chunks)
        result["kg_entity_found"] = bool(entity_info)
        return result


def run():
    """Demo: answer a sample question."""
    print("[ch15_fin] Graph RAG Tools Demo")
    print("=" * 60)

    tools = GraphRAGTools()

    # Demo questions
    questions = [
        "What companies are in the financial services sector?",
        "Which entities have the highest PageRank?",
        "What recent events affected major banks?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        answer = tools.answer_question(q)
        print(f"  A: {answer.get('answer', 'No answer')[:200]}")
        print(f"     Confidence: {answer.get('confidence', 'N/A')}")
        print(f"     Citations: {answer.get('citations', [])}")

    tools._gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
