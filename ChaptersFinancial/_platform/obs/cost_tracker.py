"""
CostTracker
===========
Accumulates LLM token costs across one import session and persists to disk.

Usage
-----
from ChaptersFinancial._platform.obs.cost_tracker import CostTracker

tracker = CostTracker(chapter="ch08_fin")
tracker.record(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=80)
tracker.record(model="gpt-4o-mini", prompt_tokens=150, completion_tokens=60)
print(tracker.summary())
tracker.save()
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path


# USD per 1k tokens – update as pricing changes
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":          {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":     {"input": 0.00015,"output": 0.0006},
    "gpt-4":           {"input": 0.03,   "output": 0.06},
    "gpt-3.5-turbo":   {"input": 0.001,  "output": 0.002},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "llama3.1:latest": {"input": 0.0,    "output": 0.0},  # local = free
}
_DEFAULT_PRICING = {"input": 0.005, "output": 0.015}


class CostTracker:
    def __init__(self, chapter: str = "unknown", run_id: str | None = None):
        self.chapter = chapter
        self.run_id = run_id or "unknown"
        self._records: list[dict] = []

        repo_root = Path(__file__).parent.parent.parent.parent
        self._out_dir = repo_root / "data_fin" / "costs"
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
        """Record a single API call and return its USD cost."""
        pricing = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (
            prompt_tokens * pricing["input"] / 1000.0
            + completion_tokens * pricing["output"] / 1000.0
        )
        self._records.append({
            "model": model,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "costUsd": cost,
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        return cost

    def summary(self) -> dict:
        total_cost = sum(r["costUsd"] for r in self._records)
        total_calls = len(self._records)
        total_prompt = sum(r["promptTokens"] for r in self._records)
        total_completion = sum(r["completionTokens"] for r in self._records)
        by_model: dict[str, dict] = {}
        for r in self._records:
            m = r["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "promptTokens": 0, "completionTokens": 0, "costUsd": 0.0}
            by_model[m]["calls"] += 1
            by_model[m]["promptTokens"] += r["promptTokens"]
            by_model[m]["completionTokens"] += r["completionTokens"]
            by_model[m]["costUsd"] += r["costUsd"]
        return {
            "chapter": self.chapter,
            "runId": self.run_id,
            "totalCalls": total_calls,
            "totalPromptTokens": total_prompt,
            "totalCompletionTokens": total_completion,
            "totalCostUsd": round(total_cost, 6),
            "byModel": by_model,
        }

    def save(self) -> Path:
        out_file = self._out_dir / f"{self.chapter}_{self.run_id[:8]}.json"
        out_file.write_text(json.dumps(self.summary(), indent=2))
        return out_file
