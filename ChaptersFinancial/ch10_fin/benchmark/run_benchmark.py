"""
run_benchmark.py
================
Compare NED quality + cost across different LLM providers.
Runs the same mention set through OpenAI and Ollama, reporting:
  - Accuracy (resolution rate)
  - Latency (seconds per mention)
  - Cost (tokens / API calls)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider


def _benchmark_provider(provider_name: str, sample_texts: list[str]) -> dict:
    """Run NED-style extraction with a given provider and measure metrics."""
    print(f"\n  Benchmarking provider: {provider_name} …")
    try:
        llm = LLMProvider(provider=provider_name)
    except Exception as exc:
        print(f"    [SKIP] Cannot init {provider_name}: {exc}")
        return {"provider": provider_name, "error": str(exc)}

    results = []
    start = time.time()
    for text in sample_texts:
        try:
            result = llm.complete_json(
                f"Extract all organization names from this text as a JSON array of strings: {text}",
                timeout=60,
            )
            results.append(result)
        except Exception as exc:
            results.append({"error": str(exc)})
    elapsed = time.time() - start

    successful = sum(1 for r in results if "error" not in r)
    return {
        "provider": provider_name,
        "total": len(sample_texts),
        "successful": successful,
        "accuracy": successful / max(len(sample_texts), 1),
        "total_time_s": round(elapsed, 2),
        "avg_time_per_mention_s": round(elapsed / max(len(sample_texts), 1), 3),
        "usage": llm.usage_summary(),
    }


def run():
    print("[ch10_fin] NED Provider Benchmark")
    print("=" * 60)

    gp = GraphProvider()

    # Get sample mention texts from Neo4j
    mentions = gp.run("""
        MATCH (m:Mention {label: 'ORG'})
        RETURN m.text AS text
        LIMIT 20
    """)
    sample_texts = [m["text"] for m in mentions if m.get("text")]
    gp.close()

    if not sample_texts:
        print("\n  No mention texts found. Run ch06_fin or ch08_fin first.")
        # Use fallback sample texts
        sample_texts = [
            "Apple Inc. reported strong quarterly earnings",
            "Microsoft Corporation announced a new acquisition",
            "Goldman Sachs Group increased its dividend",
            "JPMorgan Chase reported record profits",
            "Tesla Inc. expanded its manufacturing capacity",
        ]

    print(f"\n  Sample size: {len(sample_texts)} mentions")

    # Benchmark each available provider
    providers = ["mock", "ollama", "openai"]
    results = []
    for p in providers:
        result = _benchmark_provider(p, sample_texts)
        results.append(result)

    # Report
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Provider':<15s} {'Accuracy':>10s} {'Avg Time':>10s} {'Calls':>8s}")
    print("-" * 45)
    for r in results:
        if "error" in r:
            print(f"{r['provider']:<15s} {'SKIP':>10s}")
            continue
        print(
            f"{r['provider']:<15s} "
            f"{r['accuracy']:>9.1%} "
            f"{r['avg_time_per_mention_s']:>9.3f}s "
            f"{r['usage'].get('total_calls', 0):>7d}"
        )

    # Save report
    report_path = Path(__file__).parent / "report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Report saved to {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    run()
