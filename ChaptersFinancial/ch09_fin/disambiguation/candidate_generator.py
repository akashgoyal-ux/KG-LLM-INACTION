"""
candidate_generator.py
======================
Generate candidate entities for mention resolution using:
  1. Alias dictionary built from GLEIF names + ticker tables in Neo4j
  2. Dense retrieval over entity profile embeddings (from ch07_fin)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider


class CandidateGenerator:
    """Generate candidate entities for a given mention text."""

    def __init__(self, gp: GraphProvider, llm: LLMProvider | None = None):
        self._gp = gp
        self._llm = llm
        self._alias_dict: dict[str, list[dict]] | None = None

    def _build_alias_dict(self) -> dict[str, list[dict]]:
        """Build alias dictionary from Neo4j entity names + aliases."""
        if self._alias_dict is not None:
            return self._alias_dict

        records = self._gp.run("""
            MATCH (le:LegalEntity)
            WHERE le.name IS NOT NULL
            RETURN le.lei AS lei, le.name AS name,
                   le.aliases AS aliases, le.jurisdiction AS jurisdiction
        """)
        alias_dict: dict[str, list[dict]] = {}
        for rec in records:
            # Index by lowercase name
            key = rec["name"].strip().lower()
            entry = {"lei": rec["lei"], "name": rec["name"],
                     "jurisdiction": rec.get("jurisdiction")}
            alias_dict.setdefault(key, []).append(entry)
            # Also index aliases
            for alias in (rec.get("aliases") or []):
                if alias:
                    alias_dict.setdefault(alias.strip().lower(), []).append(entry)

        # Add ticker → instrument mapping
        instruments = self._gp.run("""
            MATCH (i:Instrument)
            WHERE i.ticker IS NOT NULL
            RETURN i.figi AS figi, i.ticker AS ticker, i.name AS name
        """)
        for inst in instruments:
            key = inst["ticker"].strip().lower()
            entry = {"figi": inst["figi"], "name": inst.get("name") or inst["ticker"]}
            alias_dict.setdefault(key, []).append(entry)
            key_dollar = f"${key}"
            alias_dict.setdefault(key_dollar, []).append(entry)

        self._alias_dict = alias_dict
        return alias_dict

    def generate_candidates(self, mention_text: str, top_k: int = 10) -> list[dict]:
        """
        Return up to top_k candidate entities for the given mention text.
        Combines alias dictionary lookup with dense retrieval.
        """
        candidates = []

        # 1. Alias dictionary exact/prefix match
        alias_dict = self._build_alias_dict()
        key = mention_text.strip().lower()
        exact = alias_dict.get(key, [])
        for c in exact[:top_k]:
            c["source"] = "alias_exact"
            c["score"] = 1.0
        candidates.extend(exact[:top_k])

        # Prefix match
        if len(candidates) < top_k:
            for alias_key, entries in alias_dict.items():
                if alias_key.startswith(key) and alias_key != key:
                    for e in entries[:2]:
                        e["source"] = "alias_prefix"
                        e["score"] = 0.8
                        candidates.append(e)
                    if len(candidates) >= top_k:
                        break

        # 2. Dense retrieval via embedding similarity
        if self._llm and len(candidates) < top_k:
            try:
                mention_emb = self._llm.embed([mention_text])[0]
                # Get entities with embeddings from Neo4j
                entities = self._gp.run("""
                    MATCH (le:LegalEntity)
                    WHERE le.profileEmbedding IS NOT NULL
                    RETURN le.lei AS lei, le.name AS name,
                           le.profileEmbedding AS embedding
                    LIMIT 500
                """)
                # Compute cosine similarities
                mention_arr = np.array(mention_emb)
                sims = []
                for ent in entities:
                    ent_arr = np.array(ent["embedding"])
                    cos_sim = float(
                        np.dot(mention_arr, ent_arr) /
                        (np.linalg.norm(mention_arr) * np.linalg.norm(ent_arr) + 1e-9)
                    )
                    sims.append((ent, cos_sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                for ent, sim in sims[:top_k - len(candidates)]:
                    candidates.append({
                        "lei": ent["lei"], "name": ent["name"],
                        "source": "dense_retrieval", "score": sim,
                    })
            except Exception:
                pass

        return candidates[:top_k]
