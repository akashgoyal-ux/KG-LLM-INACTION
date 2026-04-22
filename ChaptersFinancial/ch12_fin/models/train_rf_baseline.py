"""
train_rf_baseline.py
====================
Train a Random Forest baseline for risk classification using
handcrafted graph features computed in ch12_fin.features.
Compare RF vs LogisticRegression on PR-AUC.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch12_fin] Random Forest Baseline for Risk Classification")
    print("=" * 60)

    gp = GraphProvider()

    # Load features
    entities = gp.run("""
        MATCH (le:LegalEntity)
        RETURN le.lei AS lei, le.name AS name,
               coalesce(le.feat_totalDegree, 0) AS totalDegree,
               coalesce(le.feat_filingCount, 0) AS filingCount,
               coalesce(le.feat_mentionCount, 0) AS mentionCount,
               coalesce(le.feat_eventCount, 0) AS eventCount,
               coalesce(le.pagerank, 0.0) AS pagerank,
               le.status AS status,
               coalesce(le.feat_maxMultiplex, 0) AS maxMultiplex
    """)
    print(f"\n  Entities: {len(entities)}")
    gp.close()

    if len(entities) < 10:
        print("  Need more entities. Run ch12_fin features first.")
        return

    # Build feature matrix
    feature_names = ["totalDegree", "filingCount", "mentionCount", "eventCount",
                     "pagerank", "maxMultiplex"]
    X = np.array([[e.get(f, 0) or 0 for f in feature_names] for e in entities], dtype=float)

    # Proxy labels (same as ch11)
    y = np.array([
        1 if (e.get("totalDegree", 0) > 5 or e.get("status") == "INACTIVE") else 0
        for e in entities
    ])

    print(f"  Features: {feature_names}")
    print(f"  Positive labels: {y.sum()}/{len(y)}")

    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        print("  Insufficient label diversity.")
        return

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    cv = min(3, max(2, y.sum(), len(y) - y.sum()))

    # Logistic Regression baseline
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring="average_precision")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="average_precision")

    print(f"\n  Results ({cv}-fold CV PR-AUC):")
    print(f"  {'Model':<25s} {'PR-AUC':>10s}")
    print(f"  {'-'*37}")
    print(f"  {'LogisticRegression':<25s} {lr_scores.mean():>10.3f}")
    print(f"  {'RandomForest':<25s} {rf_scores.mean():>10.3f}")

    # Feature importance (from RF trained on full data)
    rf.fit(X, y)
    importance = rf.feature_importances_
    print(f"\n  Feature Importance (Random Forest):")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        print(f"    {name:<20s} {imp:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
