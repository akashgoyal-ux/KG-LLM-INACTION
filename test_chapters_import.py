"""
test_chapters_import.py
========================
Validates that every financial chapter module can be imported without error.
Also runs lightweight tests that don't require network or GPU.
"""
from __future__ import annotations

import importlib
import sys
import os
import traceback
from pathlib import Path

# ── repo root on path ──────────────────────────────────────────────────────
REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

# Load .env so providers initialise without crashing
from dotenv import load_dotenv
load_dotenv(REPO / "ChaptersFinancial" / ".env", override=True)

# ── helpers ───────────────────────────────────────────────────────────────
PASS: list[str] = []
FAIL: list[tuple[str, str]] = []

def check(label: str, fn):
    """Run fn(); record PASS or FAIL."""
    try:
        fn()
        PASS.append(label)
        print(f"  [OK]  {label}")
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        FAIL.append((label, tb))
        print(f"  [ERR] {label}\n        {exc}")

def imp(module: str):
    """Return a zero-arg callable that imports module."""
    return lambda: importlib.import_module(module)

# ── _platform ─────────────────────────────────────────────────────────────
print("\n=== _platform ===")
check("_platform.__init__",          imp("ChaptersFinancial._platform"))
check("_platform.fin_importer_base", imp("ChaptersFinancial._platform.fin_importer_base"))
check("_platform.providers.graph",   imp("ChaptersFinancial._platform.providers.graph"))
check("_platform.providers.llm",     imp("ChaptersFinancial._platform.providers.llm"))
check("_platform.providers.vector",  imp("ChaptersFinancial._platform.providers.vector"))
check("_platform.obs.cost_tracker",  imp("ChaptersFinancial._platform.obs.cost_tracker"))
check("_platform.obs.run_logger",    imp("ChaptersFinancial._platform.obs.run_logger"))
check("_platform.eval.ner_eval",     imp("ChaptersFinancial._platform.eval.ner_eval"))
check("_platform.eval.ned_eval",     imp("ChaptersFinancial._platform.eval.ned_eval"))
check("_platform.eval.rag_eval",     imp("ChaptersFinancial._platform.eval.rag_eval"))
check("_platform.eval.ml_eval",      imp("ChaptersFinancial._platform.eval.ml_eval"))

# ── ch01_fin ──────────────────────────────────────────────────────────────
print("\n=== ch01_fin ===")
check("ch01_fin.__init__",                imp("ChaptersFinancial.ch01_fin"))
check("ch01_fin.setup.__init__",          imp("ChaptersFinancial.ch01_fin.setup"))

# ── ch02_fin ──────────────────────────────────────────────────────────────
print("\n=== ch02_fin ===")
check("ch02_fin.__init__",               imp("ChaptersFinancial.ch02_fin"))
check("ch02_fin.validate_prompts",       imp("ChaptersFinancial.ch02_fin.validate_prompts"))
check("ch02_fin.tests.test_schemas",     imp("ChaptersFinancial.ch02_fin.tests.test_schemas"))
check("ch02_fin.listings.01_zero_shot",  lambda: importlib.util.spec_from_file_location(
    "ch02_listing01",
    REPO / "ChaptersFinancial/ch02_fin/listings/01_zero_shot_entity_extraction.py"))

# ── ch03_fin ──────────────────────────────────────────────────────────────
print("\n=== ch03_fin ===")
check("ch03_fin.__init__",                   imp("ChaptersFinancial.ch03_fin"))
check("ch03_fin.importer.import_gleif",      imp("ChaptersFinancial.ch03_fin.importer.import_gleif"))
check("ch03_fin.importer.import_currencies", imp("ChaptersFinancial.ch03_fin.importer.import_currencies"))
check("ch03_fin.importer.import_fibo",       imp("ChaptersFinancial.ch03_fin.importer.import_fibo"))
check("ch03_fin.importer.verify",            imp("ChaptersFinancial.ch03_fin.importer.verify"))
check("ch03_fin.analysis.ontology_analysis", imp("ChaptersFinancial.ch03_fin.analysis.ontology_analysis"))

# ── ch04_fin ──────────────────────────────────────────────────────────────
print("\n=== ch04_fin ===")
check("ch04_fin.__init__",                       imp("ChaptersFinancial.ch04_fin"))
check("ch04_fin.importer.import_exchanges",      imp("ChaptersFinancial.ch04_fin.importer.import_exchanges"))
check("ch04_fin.importer.import_instruments_figi", imp("ChaptersFinancial.ch04_fin.importer.import_instruments_figi"))
check("ch04_fin.importer.import_ownership_13f",  imp("ChaptersFinancial.ch04_fin.importer.import_ownership_13f"))
check("ch04_fin.importer.import_sector_mapping", imp("ChaptersFinancial.ch04_fin.importer.import_sector_mapping"))
check("ch04_fin.analysis.graph_summary",         imp("ChaptersFinancial.ch04_fin.analysis.graph_summary"))
check("ch04_fin.analysis.centrality_pagerank",   imp("ChaptersFinancial.ch04_fin.analysis.centrality_pagerank"))
check("ch04_fin.analysis.community_louvain",     imp("ChaptersFinancial.ch04_fin.analysis.community_louvain"))

# ── ch05_fin ──────────────────────────────────────────────────────────────
print("\n=== ch05_fin ===")
check("ch05_fin.__init__",                           imp("ChaptersFinancial.ch05_fin"))
check("ch05_fin.reconciliation.deterministic_match", imp("ChaptersFinancial.ch05_fin.reconciliation.deterministic_match"))
check("ch05_fin.reconciliation.probabilistic_match", imp("ChaptersFinancial.ch05_fin.reconciliation.probabilistic_match"))
check("ch05_fin.eval.eval_reconciliation",           imp("ChaptersFinancial.ch05_fin.eval.eval_reconciliation"))

# ── ch06_fin ──────────────────────────────────────────────────────────────
print("\n=== ch06_fin ===")
check("ch06_fin.__init__",                  imp("ChaptersFinancial.ch06_fin"))
check("ch06_fin.importer.step1_ingest_news", imp("ChaptersFinancial.ch06_fin.importer.step1_ingest_news"))
check("ch06_fin.importer.step2_run_ner",     imp("ChaptersFinancial.ch06_fin.importer.step2_run_ner"))
check("ch06_fin.importer.step3_enrich_via_apis", imp("ChaptersFinancial.ch06_fin.importer.step3_enrich_via_apis"))

# ── ch07_fin ──────────────────────────────────────────────────────────────
print("\n=== ch07_fin ===")
check("ch07_fin.__init__",                      imp("ChaptersFinancial.ch07_fin"))
check("ch07_fin.listings.embed_issuer_profiles", imp("ChaptersFinancial.ch07_fin.listings.embed_issuer_profiles"))
check("ch07_fin.listings.peer_search_by_embedding", imp("ChaptersFinancial.ch07_fin.listings.peer_search_by_embedding"))

# ── ch08_fin ──────────────────────────────────────────────────────────────
print("\n=== ch08_fin ===")
check("ch08_fin.__init__",                      imp("ChaptersFinancial.ch08_fin"))
check("ch08_fin.importer.ingest_filings",       imp("ChaptersFinancial.ch08_fin.importer.ingest_filings"))
check("ch08_fin.importer.extract_with_llm",     imp("ChaptersFinancial.ch08_fin.importer.extract_with_llm"))
check("ch08_fin.importer.normalize_and_merge",  imp("ChaptersFinancial.ch08_fin.importer.normalize_and_merge"))

# ── ch09_fin ──────────────────────────────────────────────────────────────
print("\n=== ch09_fin ===")
check("ch09_fin.__init__",                        imp("ChaptersFinancial.ch09_fin"))
check("ch09_fin.disambiguation.candidate_generator", imp("ChaptersFinancial.ch09_fin.disambiguation.candidate_generator"))
check("ch09_fin.disambiguation.ontology_linker",  imp("ChaptersFinancial.ch09_fin.disambiguation.ontology_linker"))
check("ch09_fin.disambiguation.main",             imp("ChaptersFinancial.ch09_fin.disambiguation.main"))
check("ch09_fin.eval.eval_ned",                   imp("ChaptersFinancial.ch09_fin.eval.eval_ned"))

# ── ch10_fin ──────────────────────────────────────────────────────────────
print("\n=== ch10_fin ===")
check("ch10_fin.__init__",                    imp("ChaptersFinancial.ch10_fin"))
check("ch10_fin.benchmark.run_benchmark",     imp("ChaptersFinancial.ch10_fin.benchmark.run_benchmark"))
check("ch10_fin.disambiguation.main",         imp("ChaptersFinancial.ch10_fin.disambiguation.main"))

# ── ch11_fin ──────────────────────────────────────────────────────────────
print("\n=== ch11_fin ===")
check("ch11_fin.__init__",                    imp("ChaptersFinancial.ch11_fin"))
check("ch11_fin.analysis.node2vec_issuers",   imp("ChaptersFinancial.ch11_fin.analysis.node2vec_issuers"))
check("ch11_fin.analysis.cluster_issuers",    imp("ChaptersFinancial.ch11_fin.analysis.cluster_issuers"))
check("ch11_fin.analysis.classify_high_risk", imp("ChaptersFinancial.ch11_fin.analysis.classify_high_risk"))

# ── ch12_fin ──────────────────────────────────────────────────────────────
print("\n=== ch12_fin ===")
check("ch12_fin.__init__",                 imp("ChaptersFinancial.ch12_fin"))
check("ch12_fin.features.node_features",   imp("ChaptersFinancial.ch12_fin.features.node_features"))
check("ch12_fin.features.edge_features",   imp("ChaptersFinancial.ch12_fin.features.edge_features"))
check("ch12_fin.models.train_rf_baseline", imp("ChaptersFinancial.ch12_fin.models.train_rf_baseline"))
check("ch12_fin.store.export_parquet",     imp("ChaptersFinancial.ch12_fin.store.export_parquet"))

# ── ch13_fin ──────────────────────────────────────────────────────────────
print("\n=== ch13_fin ===")
check("ch13_fin.__init__",                          imp("ChaptersFinancial.ch13_fin"))
check("ch13_fin.listings.pyg_intro_counterparty",   imp("ChaptersFinancial.ch13_fin.listings.pyg_intro_counterparty"))
check("ch13_fin.listings.compare_architectures",    imp("ChaptersFinancial.ch13_fin.listings.compare_architectures"))

# ── ch14_fin ──────────────────────────────────────────────────────────────
print("\n=== ch14_fin ===")
check("ch14_fin.__init__",                    imp("ChaptersFinancial.ch14_fin"))
check("ch14_fin.data.build_pyg_dataset",      imp("ChaptersFinancial.ch14_fin.data.build_pyg_dataset"))
check("ch14_fin.train_for_classification",    imp("ChaptersFinancial.ch14_fin.train_for_classification"))
check("ch14_fin.train_for_link_prediction",   imp("ChaptersFinancial.ch14_fin.train_for_link_prediction"))

# ── ch15_fin ──────────────────────────────────────────────────────────────
print("\n=== ch15_fin ===")
check("ch15_fin.__init__",                        imp("ChaptersFinancial.ch15_fin"))
check("ch15_fin.code.tools",                      imp("ChaptersFinancial.ch15_fin.code.tools"))
check("ch15_fin.code.contradiction_check",        imp("ChaptersFinancial.ch15_fin.code.contradiction_check"))
check("ch15_fin.listings.react_agent_with_graph_rag", imp("ChaptersFinancial.ch15_fin.listings.react_agent_with_graph_rag"))
check("ch15_fin.listings.conversational_agent",   imp("ChaptersFinancial.ch15_fin.listings.conversational_agent"))

# ── ch16_fin ──────────────────────────────────────────────────────────────
print("\n=== ch16_fin ===")
check("ch16_fin.__init__",                    imp("ChaptersFinancial.ch16_fin"))
check("ch16_fin.lineage.lineage_emitter",     imp("ChaptersFinancial.ch16_fin.lineage.lineage_emitter"))
check("ch16_fin.migrations.apply_migrations", imp("ChaptersFinancial.ch16_fin.migrations.apply_migrations"))
check("ch16_fin.contracts.validate_contracts", imp("ChaptersFinancial.ch16_fin.contracts.validate_contracts"))
check("ch16_fin.tests.test_integration",      imp("ChaptersFinancial.ch16_fin.tests.test_integration"))

# ── ch17_fin ──────────────────────────────────────────────────────────────
print("\n=== ch17_fin ===")
check("ch17_fin.__init__", imp("ChaptersFinancial.ch17_fin"))
# app.py has streamlit at module level — just check syntax via compile
def _check_ch17_app():
    src = (REPO / "ChaptersFinancial/ch17_fin/app.py").read_text()
    compile(src, "ch17_fin/app.py", "exec")
check("ch17_fin.app (syntax)", _check_ch17_app)

# ── ch02 validate_prompts lightweight run ─────────────────────────────────
print("\n=== ch02_fin: validate_prompts (mock mode) ===")
def _run_ch02_validate():
    os.environ["LLM_PROVIDER"] = "mock"
    import ChaptersFinancial.ch02_fin.validate_prompts as vp
    # Re-run the validation function if it exposes one
    if hasattr(vp, "run_all"):
        vp.run_all()
    else:
        # run as script in-process via exec
        src = (REPO / "ChaptersFinancial/ch02_fin/validate_prompts.py").read_text()
        ns: dict = {"__name__": "__main__", "__file__": str(REPO / "ChaptersFinancial/ch02_fin/validate_prompts.py")}
        exec(compile(src, "validate_prompts.py", "exec"), ns)
check("ch02_fin.validate_prompts (mock)", _run_ch02_validate)

# ── ch05 reconciliation unit test ─────────────────────────────────────────
print("\n=== ch05_fin: deterministic_match unit test ===")
def _test_det_match():
    from ChaptersFinancial.ch05_fin.reconciliation.deterministic_match import DeterministicMatcher
    dm = DeterministicMatcher.__new__(DeterministicMatcher)
    # Basic sanity: the class must exist and have a match method
    assert hasattr(dm, "match") or hasattr(DeterministicMatcher, "run"), \
        "DeterministicMatcher missing 'match' or 'run' method"
check("ch05_fin: DeterministicMatcher exists", _test_det_match)

# ── _platform LLMProvider._parse_json ─────────────────────────────────────
print("\n=== _platform: LLMProvider unit tests ===")
def _test_parse_json():
    from ChaptersFinancial._platform.providers.llm import LLMProvider
    assert LLMProvider._parse_json('{"a":1}') == {"a": 1}
    assert LLMProvider._parse_json('```json\n{"b":2}\n```') == {"b": 2}
    assert LLMProvider._parse_json("some text {\"c\": 3} more") == {"c": 3}
check("_platform: LLMProvider._parse_json", _test_parse_json)

# ── summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print("\nFailed details:")
    for label, tb in FAIL:
        print(f"\n-- {label} --")
        print(tb)
    sys.exit(1)
else:
    print("All chapter imports OK!")
    sys.exit(0)
