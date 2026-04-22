"""
export_parquet.py
=================
Export entity feature store to Parquet for offline ML pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch12_fin] Export Feature Store to Parquet")
    print("=" * 60)

    try:
        import pandas as pd
    except ImportError:
        print("  [ERROR] pandas not installed. Run: pip install pandas pyarrow")
        return

    gp = GraphProvider()

    entities = gp.run("""
        MATCH (le:LegalEntity)
        RETURN le.lei AS lei, le.name AS name,
               le.jurisdiction AS jurisdiction,
               le.status AS status,
               coalesce(le.feat_totalDegree, 0) AS totalDegree,
               coalesce(le.feat_inDegree, 0) AS inDegree,
               coalesce(le.feat_outDegree, 0) AS outDegree,
               coalesce(le.feat_filingCount, 0) AS filingCount,
               coalesce(le.feat_mentionCount, 0) AS mentionCount,
               coalesce(le.feat_eventCount, 0) AS eventCount,
               coalesce(le.pagerank, 0.0) AS pagerank,
               le.communityLouvain AS community,
               le.clusterNode2vec AS clusterNode2vec,
               coalesce(le.feat_maxMultiplex, 0) AS maxMultiplex
    """)
    gp.close()

    if not entities:
        print("  No entities found.")
        return

    df = pd.DataFrame(entities)
    out_dir = Path(__file__).resolve().parents[3] / "data_fin" / "feature_store"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entity_features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n  Exported {len(df)} entities to {out_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")

    print("\nDone.")


if __name__ == "__main__":
    run()
