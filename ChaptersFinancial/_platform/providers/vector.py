"""
VectorProvider
==============
Abstraction over vector similarity search.
Backends: Neo4j vector index (default) or local FAISS (offline/dev).

Usage
-----
from ChaptersFinancial._platform.providers.vector import VectorProvider

vp = VectorProvider()
vp.upsert("chunk-001", [0.1, 0.2, ...], {"docId": "doc-001", "text": "..."})
results = vp.search([0.1, 0.2, ...], top_k=5)
# [{"id": "chunk-001", "score": 0.97, "metadata": {...}}, ...]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "provider_config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


class VectorProvider:
    def __init__(self, backend: str | None = None):
        cfg = _load_config()
        self._cfg = cfg.get("vector", {})
        self._backend = (
            backend
            or os.getenv("VECTOR_BACKEND")
            or self._cfg.get("default_backend", "neo4j")
        ).lower()
        self._index_name = self._cfg.get("index_name", "fin_embeddings")
        self._dim = int(self._cfg.get("embedding_dim", 1536))

        if self._backend == "faiss":
            self._init_faiss()
        elif self._backend == "neo4j":
            self._init_neo4j()
        else:
            raise ValueError(f"Unknown vector backend: {self._backend}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert(self, doc_id: str, vector: list[float], metadata: dict | None = None) -> None:
        if self._backend == "faiss":
            self._faiss_upsert(doc_id, vector, metadata or {})
        else:
            self._neo4j_upsert(doc_id, vector, metadata or {})

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        if self._backend == "faiss":
            return self._faiss_search(query_vector, top_k)
        return self._neo4j_search(query_vector, top_k)

    # ------------------------------------------------------------------
    # Neo4j backend
    # ------------------------------------------------------------------
    def _init_neo4j(self):
        from ChaptersFinancial._platform.providers.graph import GraphProvider  # noqa: PLC0415
        self._graph = GraphProvider()

    def _neo4j_upsert(self, doc_id: str, vector: list[float], metadata: dict) -> None:
        cypher = """
        MERGE (c:Chunk {chunkId: $docId})
        SET c.embedding = $vector,
            c.docId     = $docId,
            c.text      = $text,
            c.metadata  = $meta
        """
        self._graph.run(cypher, {
            "docId": doc_id,
            "vector": vector,
            "text": metadata.get("text", ""),
            "meta": json.dumps(metadata),
        })

    def _neo4j_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        cypher = """
        CALL db.index.vector.queryNodes($index, $k, $vector)
        YIELD node, score
        RETURN node.chunkId AS id, score, node.text AS text, node.docId AS docId
        ORDER BY score DESC
        """
        rows = self._graph.run(cypher, {
            "index": self._index_name,
            "k": top_k,
            "vector": query_vector,
        })
        return [{"id": r["id"], "score": r["score"], "metadata": {"text": r["text"], "docId": r["docId"]}}
                for r in rows]

    # ------------------------------------------------------------------
    # FAISS backend (offline / dev)
    # ------------------------------------------------------------------
    def _init_faiss(self):
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
            self._faiss = faiss
            self._np = np
        except ImportError as exc:
            raise ImportError("pip install faiss-cpu numpy to use the faiss backend") from exc

        faiss_cfg = self._cfg.get("faiss", {})
        repo_root = Path(__file__).parent.parent.parent.parent
        self._index_file = repo_root / faiss_cfg.get("index_file", "data_fin/vector_index/faiss.index")
        self._meta_file  = repo_root / faiss_cfg.get("meta_file",  "data_fin/vector_index/faiss_meta.jsonl")
        self._index_file.parent.mkdir(parents=True, exist_ok=True)

        if self._index_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            self._meta: list[dict] = [
                json.loads(line) for line in self._meta_file.read_text().splitlines() if line
            ]
        else:
            self._index = faiss.IndexFlatIP(self._dim)   # inner-product (cosine after normalise)
            self._meta = []

    def _faiss_upsert(self, doc_id: str, vector: list[float], metadata: dict) -> None:
        import numpy as np  # type: ignore
        import faiss  # type: ignore

        vec = np.array([vector], dtype="float32")
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._meta.append({"id": doc_id, "metadata": metadata})
        faiss.write_index(self._index, str(self._index_file))
        with self._meta_file.open("a") as f:
            f.write(json.dumps({"id": doc_id, "metadata": metadata}) + "\n")

    def _faiss_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        import numpy as np  # type: ignore
        import faiss  # type: ignore

        if self._index.ntotal == 0:
            return []
        vec = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._meta):
                entry = self._meta[idx]
                results.append({"id": entry["id"], "score": float(score), "metadata": entry["metadata"]})
        return results
