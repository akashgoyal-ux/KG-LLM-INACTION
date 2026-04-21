"""
NER Evaluation (ch06_fin)
=========================
Compute precision, recall and F1 for named entity recognition
against a gold standard set.

Gold format (JSONL):
  {"text": "Apple Inc. reported ...", "entities": [{"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10}]}

Prediction format: same structure, list of dicts with text/label/start/end.

Usage
-----
from ChaptersFinancial._platform.eval.ner_eval import NEREval

ev = NEREval(gold_path="data_fin/eval/gold_ner.jsonl")
report = ev.evaluate(predictions)
print(report)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


class NEREval:
    def __init__(self, gold_path: str | Path):
        self._gold: list[dict] = []
        for line in Path(gold_path).read_text().splitlines():
            line = line.strip()
            if line:
                self._gold.append(json.loads(line))

    # ------------------------------------------------------------------
    # Evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(self, predictions: list[list[dict]]) -> dict:
        """
        Parameters
        ----------
        predictions : list of list of entity dicts, one inner list per document.
                      Each entity dict: {text, label, start, end}

        Returns
        -------
        dict with overall + per-label precision, recall, F1.
        """
        if len(predictions) != len(self._gold):
            raise ValueError(
                f"Expected {len(self._gold)} prediction lists, got {len(predictions)}"
            )

        label_tp: dict[str, int] = defaultdict(int)
        label_fp: dict[str, int] = defaultdict(int)
        label_fn: dict[str, int] = defaultdict(int)

        for gold_doc, pred_doc in zip(self._gold, predictions):
            gold_spans = {(e["start"], e["end"], e["label"]) for e in gold_doc["entities"]}
            pred_spans = {(e["start"], e["end"], e["label"]) for e in pred_doc}

            for span in pred_spans:
                if span in gold_spans:
                    label_tp[span[2]] += 1
                else:
                    label_fp[span[2]] += 1

            for span in gold_spans:
                if span not in pred_spans:
                    label_fn[span[2]] += 1

        results: dict = {}
        all_tp = all_fp = all_fn = 0

        all_labels = set(label_tp) | set(label_fp) | set(label_fn)
        for label in sorted(all_labels):
            tp = label_tp[label]
            fp = label_fp[label]
            fn = label_fn[label]
            p, r, f1 = _prf(tp, fp, fn)
            results[label] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
            all_tp += tp
            all_fp += fp
            all_fn += fn

        p, r, f1 = _prf(all_tp, all_fp, all_fn)
        results["overall"] = {"precision": p, "recall": r, "f1": f1,
                               "tp": all_tp, "fp": all_fp, "fn": all_fn}
        return results


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)
