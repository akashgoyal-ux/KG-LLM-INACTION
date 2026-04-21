"""
test_connections.py
===================
Validates that the platform can reach Neo4j and the configured LLM provider.
Run with:  python -m ChaptersFinancial.ch01_fin.setup.test_connections [--live]

--live  : also sends a real LLM request (requires API key / Ollama running).
Without --live, only the graph connection is tested (LLM uses mock).
"""

from __future__ import annotations

import sys
import os


def check_neo4j() -> bool:
    print("──── Neo4j connection ────")
    try:
        from ChaptersFinancial._platform.providers.graph import GraphProvider
        gp = GraphProvider()
        ok = gp.ping()
        if ok:
            print("  ✓ Neo4j reachable")
        else:
            print("  ✗ Neo4j not reachable (is Neo4j running on bolt://localhost:7687?)")
        gp.close()
        return ok
    except Exception as exc:
        print(f"  ✗ Neo4j error: {exc}")
        return False


def check_llm(live: bool = False) -> bool:
    print("──── LLM provider (Ollama) ────")
    try:
        import httpx  # type: ignore
        r = httpx.get("http://localhost:11434/api/version", timeout=5)
        version = r.json().get("version", "unknown")
        print(f"  ✓ Ollama server reachable  (version {version})")
    except Exception as exc:
        print(f"  ✗ Ollama not reachable: {exc}")
        return False

    if not live:
        print("  (skipping live inference – pass --live to test)")
        return True

    try:
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        lp = LLMProvider(provider="ollama")
        result = lp.complete_json(
            'Return a JSON object with key "status" set to "ok".',
        )
        if isinstance(result, dict):
            print(f"  ✓ Ollama inference OK: {result}")
            return True
        print(f"  ~ Unexpected response: {result}")
        return True
    except Exception as exc:
        print(f"  ✗ Ollama inference error: {exc}")
        return False


def check_platform_imports() -> bool:
    print("──── Platform imports ────")
    errors = []
    modules = [
        "ChaptersFinancial._platform.fin_importer_base",
        "ChaptersFinancial._platform.providers.llm",
        "ChaptersFinancial._platform.providers.graph",
        "ChaptersFinancial._platform.providers.vector",
        "ChaptersFinancial._platform.obs.run_logger",
        "ChaptersFinancial._platform.obs.cost_tracker",
        "ChaptersFinancial._platform.eval.ner_eval",
        "ChaptersFinancial._platform.eval.ned_eval",
        "ChaptersFinancial._platform.eval.ml_eval",
        "ChaptersFinancial._platform.eval.rag_eval",
    ]
    for mod in modules:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except Exception as exc:
            print(f"  ✗ {mod}: {exc}")
            errors.append(mod)
    return len(errors) == 0


def check_config_files() -> bool:
    print("──── Config files ────")
    from pathlib import Path
    root = Path(__file__).parent.parent.parent
    files = [
        "_platform/config/schema_config.yaml",
        "_platform/config/provider_config.yaml",
        "_platform/schema/constraints.cypher",
    ]
    ok = True
    for f in files:
        path = root / f
        if path.exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ missing: {f}")
            ok = False
    return ok


if __name__ == "__main__":
    live = "--live" in sys.argv
    results = {
        "platform_imports": check_platform_imports(),
        "config_files":     check_config_files(),
        "neo4j":            check_neo4j(),
        "llm":              check_llm(live=live),
    }
    print("\n──── Summary ────")
    all_ok = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  {status}  {check}")

    if not all_ok:
        print("\nSome checks failed. Review output above.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
