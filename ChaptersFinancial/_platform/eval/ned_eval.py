"""
NED Evaluation (ch09_fin, ch10_fin)
=====================================
Named Entity Disambiguation accuracy@k, MRR, and ECE.

Gold format (JSONL):
  {"mentionId": "m001", "goldId": "LEI:529900T8BM49AURSDO55", "candidates": [
      {"id": "LEI:529900T8BM49AURSDO55", "score": 0.92},
      {"id": "LEI:XYZ...",               "score": 0.74}
  ]}

Usage
-----
from ChaptersFinancial._platform.eval.ned_eval import NEDEval

ev = NEDEval(gold_path="data_fin/eval/gold_ned.jsonl")
report = ev.evaluate()   # reads gold candidates + gold IDs
print(report)
"""

from __future__ import annotations

import json
import math
from pathlib import Path


class NEDEval:
    def __init__(self, gold_path: str | Path):
        self._gold: list[dict] = []
        for line in Path(gold_path).read_text().splitlines():
            line = line.strip()
            if line:
                self._gold.append(json.loads(line))

    def evaluate(self, ks: list[int] | None = None) -> dict:
        if ks is None:
            ks = [1, 3, 5]
        accuracy_at_k: dict[int, float] = {k: 0.0 for k in ks}
        rr_sum = 0.0
        ece_bins: list[tuple[float, bool]] = []  # (confidence, correct)

        for item in self._gold:
            gold_id = item["goldId"]
            candidates = item.get("candidates", [])
            # Sort by descending score
            ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            ranked_ids = [c["id"] for c in ranked]

            # Accuracy@k
            for k in ks:
                if gold_id in ranked_ids[:k]:
                    accuracy_at_k[k] += 1

            # MRR
            if gold_id in ranked_ids:
                rank = ranked_ids.index(gold_id) + 1
                rr_sum += 1.0 / rank

            # ECE data: use top candidate confidence
            if ranked:
                conf = float(ranked[0].get("score", 0.0))
                correct = ranked_ids[0] == gold_id if ranked else False
                ece_bins.append((conf, correct))

        n = len(self._gold) or 1
        result = {
            "n": len(self._gold),
            "mrr": round(rr_sum / n, 4),
            "ece": round(_ece(ece_bins), 4),
        }
        for k in ks:
            result[f"accuracy@{k}"] = round(accuracy_at_k[k] / n, 4)
        return result


def _ece(bins: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform binning)."""
    if not bins:
        return 0.0
    bucket_conf = [0.0] * n_bins
    bucket_acc  = [0.0] * n_bins
    bucket_n    = [0]   * n_bins
    for conf, correct in bins:
        idx = min(int(conf * n_bins), n_bins - 1)
        bucket_conf[idx] += conf
        bucket_acc[idx]  += int(correct)
        bucket_n[idx]    += 1
    ece = 0.0
    total = len(bins)
    for i in range(n_bins):
        if bucket_n[i] > 0:
            avg_conf = bucket_conf[i] / bucket_n[i]
            avg_acc  = bucket_acc[i]  / bucket_n[i]
            ece += (bucket_n[i] / total) * abs(avg_conf - avg_acc)
    return ece
