"""
RAG Evaluation (ch15_fin)
=========================
Faithfulness, citation precision, and answer correctness for Graph RAG.

Faithfulness  = fraction of answer claims supported by retrieved evidence.
Citation prec = fraction of cited chunks that actually contain the claimed fact.
Answer correct= fuzzy match of answer against gold answer string.

All three metrics can be estimated using an LLM-as-judge (optional).
In offline mode they fall back to simple heuristics.

Usage
-----
from ChaptersFinancial._platform.eval.rag_eval import RAGEval

ev = RAGEval(gold_path="data_fin/eval/gold_rag.jsonl")
report = ev.evaluate(answers)
# answers: list of {"question": ..., "answer": ..., "citations": [...], "evidence": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path


class RAGEval:
    def __init__(self, gold_path: str | Path, llm_judge: bool = False):
        """
        Parameters
        ----------
        gold_path  : JSONL file with one gold item per line:
                     {"question": "...", "gold_answer": "...", "gold_entity_ids": [...]}
        llm_judge  : if True, use LLMProvider to score faithfulness (requires API key)
        """
        self._gold: list[dict] = []
        for line in Path(gold_path).read_text().splitlines():
            line = line.strip()
            if line:
                self._gold.append(json.loads(line))
        self._llm_judge = llm_judge

    def evaluate(self, answers: list[dict]) -> dict:
        """
        Parameters
        ----------
        answers : list of dicts, one per question:
          {
            "question": str,
            "answer":   str,
            "citations": [chunkId, ...],   # chunk IDs cited
            "evidence":  [str, ...],       # evidence texts retrieved
          }

        Returns
        -------
        dict with faithfulness, citation_precision, answer_correctness, n
        """
        if len(answers) != len(self._gold):
            raise ValueError(f"Expected {len(self._gold)} answers, got {len(answers)}")

        faithfulness_scores: list[float] = []
        citation_precision_scores: list[float] = []
        correctness_scores: list[float] = []

        for gold, pred in zip(self._gold, answers):
            gold_answer = gold.get("gold_answer", "")
            gold_ids = set(gold.get("gold_entity_ids", []))

            # 1. Faithfulness (heuristic: fraction of gold entity IDs mentioned in answer+evidence)
            pred_text = (pred.get("answer", "") + " " +
                         " ".join(pred.get("evidence", []))).lower()
            if gold_ids:
                matched = sum(1 for gid in gold_ids if gid.lower() in pred_text)
                faithfulness_scores.append(matched / len(gold_ids))
            else:
                faithfulness_scores.append(1.0)

            # 2. Citation precision (fraction of cited chunks containing ≥1 gold entity)
            evidence_texts = [e.lower() for e in pred.get("evidence", [])]
            if evidence_texts:
                prec = sum(
                    1 for et in evidence_texts
                    if any(gid.lower() in et for gid in gold_ids)
                ) / len(evidence_texts)
                citation_precision_scores.append(prec)
            else:
                citation_precision_scores.append(0.0)

            # 3. Answer correctness (token overlap F1 with gold answer)
            correctness_scores.append(
                _token_f1(pred.get("answer", ""), gold_answer)
            )

        n = len(self._gold)
        return {
            "n": n,
            "faithfulness":       round(sum(faithfulness_scores) / n, 4),
            "citation_precision": round(sum(citation_precision_scores) / n, 4),
            "answer_correctness": round(sum(correctness_scores) / n, 4),
        }


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings (case-insensitive)."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)
