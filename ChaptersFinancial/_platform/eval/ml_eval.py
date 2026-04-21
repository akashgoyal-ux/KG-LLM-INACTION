"""
ML Evaluation (ch11_fin – ch14_fin)
=====================================
ROC-AUC, PR-AUC, calibration ECE, Hits@K, MRR for graph-ML tasks.

Usage
-----
from ChaptersFinancial._platform.eval.ml_eval import MLEval

report = MLEval.classification_report(y_true, y_score)
report = MLEval.link_prediction_report(pos_scores, neg_scores, ks=[5, 10])
"""

from __future__ import annotations

import math
from typing import Sequence


class MLEval:
    # ------------------------------------------------------------------
    # Node classification
    # ------------------------------------------------------------------
    @staticmethod
    def classification_report(
        y_true: Sequence[int],
        y_score: Sequence[float],
    ) -> dict:
        """
        Parameters
        ----------
        y_true  : binary labels (0/1)
        y_score : predicted positive probability

        Returns
        -------
        dict with roc_auc, pr_auc, ece, n_positive, n_negative
        """
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install scikit-learn to use MLEval") from exc

        roc = roc_auc_score(y_true, y_score)
        pr  = average_precision_score(y_true, y_score)
        ece = MLEval._ece(list(zip(y_score, y_true)))

        return {
            "roc_auc": round(float(roc), 4),
            "pr_auc":  round(float(pr), 4),
            "ece":     round(ece, 4),
            "n_positive": int(sum(y_true)),
            "n_negative": int(len(y_true) - sum(y_true)),
        }

    # ------------------------------------------------------------------
    # Link prediction
    # ------------------------------------------------------------------
    @staticmethod
    def link_prediction_report(
        pos_scores: Sequence[float],
        neg_scores: Sequence[float],
        ks: list[int] | None = None,
    ) -> dict:
        """
        pos_scores : scores for true (positive) edges
        neg_scores : scores for negative edges
        Returns Hits@K and MRR.
        """
        if ks is None:
            ks = [1, 5, 10]

        all_scores = list(pos_scores) + list(neg_scores)
        labels = [1] * len(pos_scores) + [0] * len(neg_scores)

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore
            roc = float(roc_auc_score(labels, all_scores))
            pr  = float(average_precision_score(labels, all_scores))
        except ImportError:
            roc = pr = float("nan")

        # Hits@K and MRR: for each positive edge, how many negatives rank higher?
        hits: dict[int, float] = {k: 0.0 for k in ks}
        rr_sum = 0.0
        for ps in pos_scores:
            n_above = sum(1 for ns in neg_scores if ns >= ps)
            rank = n_above + 1
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1

        n = len(pos_scores) or 1
        result: dict = {
            "roc_auc": round(roc, 4),
            "pr_auc":  round(pr, 4),
            "mrr":     round(rr_sum / n, 4),
        }
        for k in ks:
            result[f"hits@{k}"] = round(hits[k] / n, 4)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ece(bins: list[tuple[float, int]], n_bins: int = 10) -> float:
        bucket_conf = [0.0] * n_bins
        bucket_acc  = [0.0] * n_bins
        bucket_n    = [0]   * n_bins
        for conf, label in bins:
            idx = min(int(conf * n_bins), n_bins - 1)
            bucket_conf[idx] += conf
            bucket_acc[idx]  += label
            bucket_n[idx]    += 1
        ece = 0.0
        total = len(bins) or 1
        for i in range(n_bins):
            if bucket_n[i] > 0:
                avg_conf = bucket_conf[i] / bucket_n[i]
                avg_acc  = bucket_acc[i]  / bucket_n[i]
                ece += (bucket_n[i] / total) * abs(avg_conf - avg_acc)
        return ece
