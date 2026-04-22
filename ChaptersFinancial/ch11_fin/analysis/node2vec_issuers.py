"""
node2vec_issuers.py
===================
Compute Node2Vec embeddings on the entity-ownership graph using Neo4j GDS.
Falls back to a Cypher-based random walk + Word2Vec if GDS is not available.

Projects: (:LegalEntity)-[:OWNS|CONTROLS|PARENT_OF]-(:LegalEntity)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def _try_gds_node2vec(gp: GraphProvider) -> bool:
    """Attempt to run Node2Vec via GDS plugin."""
    try:
        # Create graph projection
        gp.run("""
            CALL gds.graph.project(
                'issuer-ownership',
                'LegalEntity',
                {
                    OWNS: {orientation: 'UNDIRECTED'},
                    CONTROLS: {orientation: 'UNDIRECTED'},
                    PARENT_OF: {orientation: 'UNDIRECTED'}
                }
            )
        """)
        print("   GDS graph projected.")

        # Run Node2Vec
        gp.run("""
            CALL gds.node2vec.write('issuer-ownership', {
                embeddingDimension: 64,
                walkLength: 20,
                walksPerNode: 10,
                inOutFactor: 1.0,
                returnFactor: 1.0,
                writeProperty: 'node2vecEmbedding'
            })
        """)
        print("   Node2Vec embeddings written.")

        # Drop projection
        gp.run("CALL gds.graph.drop('issuer-ownership')")
        return True
    except Exception as exc:
        # Clean up projection if it was created
        try:
            gp.run("CALL gds.graph.drop('issuer-ownership')")
        except Exception:
            pass
        print(f"   [INFO] GDS Node2Vec not available: {exc}")
        return False


def _cypher_random_walks(gp: GraphProvider) -> list[dict]:
    """
    Fallback: generate random walks using Cypher + APOC path expander,
    then use simple co-occurrence for embeddings.
    """
    print("   Using Cypher-based random walk fallback …")

    # Get adjacency data
    edges = gp.run("""
        MATCH (a:LegalEntity)-[r:OWNS|CONTROLS|PARENT_OF]-(b:LegalEntity)
        RETURN a.lei AS source, b.lei AS target
    """)

    if not edges:
        print("   No ownership edges found.")
        return []

    # Build adjacency list
    from collections import defaultdict
    import random

    adj: dict[str, list[str]] = defaultdict(list)
    all_nodes = set()
    for e in edges:
        adj[e["source"]].append(e["target"])
        adj[e["target"]].append(e["source"])
        all_nodes.update([e["source"], e["target"]])

    print(f"   Graph: {len(all_nodes)} nodes, {len(edges)} edges")

    # Random walks
    walks = []
    for node in all_nodes:
        for _ in range(10):  # walks per node
            walk = [node]
            current = node
            for _ in range(20):  # walk length
                neighbors = adj.get(current, [])
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            walks.append(walk)

    # Simple co-occurrence embedding (SVD on co-occurrence matrix)
    node_list = sorted(all_nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    dim = min(64, n)

    cooccurrence = np.zeros((n, n))
    window = 5
    for walk in walks:
        for i, node in enumerate(walk):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    cooccurrence[node_idx[node]][node_idx[walk[j]]] += 1

    # SVD for dimensionality reduction
    from numpy.linalg import svd
    U, S, _ = svd(cooccurrence, full_matrices=False)
    embeddings = U[:, :dim] * np.sqrt(S[:dim])

    # Store embeddings in Neo4j
    results = []
    for i, node in enumerate(node_list):
        emb = embeddings[i].tolist()
        gp.run(
            "MATCH (le:LegalEntity {lei: $lei}) SET le.node2vecEmbedding = $emb",
            {"lei": node, "emb": emb},
        )
        results.append({"lei": node, "embedding": emb})

    print(f"   Computed {len(results)} embeddings (dim={dim}).")
    return results


def run():
    print("[ch11_fin] Node2Vec Embeddings for Issuers")
    print("=" * 60)

    gp = GraphProvider()

    print("\n1. Attempting GDS Node2Vec …")
    if not _try_gds_node2vec(gp):
        print("\n2. Falling back to Cypher random walks …")
        _cypher_random_walks(gp)

    # Verify
    print("\n3. Verification …")
    cnt = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.node2vecEmbedding IS NOT NULL
        RETURN count(le) AS cnt
    """)[0]["cnt"]
    print(f"   Entities with Node2Vec embeddings: {cnt}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
