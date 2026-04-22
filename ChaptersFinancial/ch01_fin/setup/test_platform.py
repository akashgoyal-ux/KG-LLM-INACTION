"""
test_platform.py
================
pytest unit tests for all _platform modules.

Tests are split into two groups:
  - Unit tests (no external deps): always fast, run without Ollama or Neo4j.
  - LLM tests (@pytest.mark.llm): call the locally-installed Ollama model
    deepseek-v3.2:cloud; skipped automatically if Ollama is not reachable.

Run all:          pytest ChaptersFinancial/ch01_fin/setup/test_platform.py -v
Run unit only:    pytest ChaptersFinancial/ch01_fin/setup/test_platform.py -v -m "not llm"
Run LLM tests:    pytest ChaptersFinancial/ch01_fin/setup/test_platform.py -v -m llm
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        import httpx  # type: ignore
        r = httpx.get("http://localhost:11434/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


_OLLAMA_MODEL = "deepseek-v3.2:cloud"

# Skip marker: applied to any test that calls the real Ollama endpoint
llm_mark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable at http://localhost:11434"
)


# ==========================================================================
# LLMProvider – unit (pure Python, no network)
# ==========================================================================
class TestLLMProviderUnit:
    def test_parse_json_plain(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        raw = '{"key": "value"}'
        assert LLMProvider._parse_json(raw) == {"key": "value"}

    def test_parse_json_strips_fences(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        raw = '```json\n{"key": "value"}\n```'
        assert LLMProvider._parse_json(raw) == {"key": "value"}

    def test_parse_json_strips_think_tags(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        raw = '<think>\nLet me reason...\n</think>\n{"entities": []}'
        result = LLMProvider._parse_json(raw)
        assert result == {"entities": []}

    def test_parse_json_embedded_in_prose(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        raw = 'Sure, here is the result: {"status": "ok"} Hope that helps.'
        assert LLMProvider._parse_json(raw) == {"status": "ok"}

    def test_cache_roundtrip(self, tmp_path):
        """Second call with identical prompt must return cached result."""
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        lp = LLMProvider(provider="mock")
        lp._cache_dir = tmp_path
        # Manually seed cache so test is fully offline
        prompt = '{"seed": 1}'
        key    = LLMProvider._cache_key(prompt, None, lp._default_model())
        cached = {"entities": [], "_cached": True}
        (tmp_path / f"{key}.json").write_text(json.dumps(cached))
        result = lp.complete_json(prompt)
        assert result == cached

    def test_unknown_provider_raises(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        lp = LLMProvider(provider="ollama")
        lp._provider = "nonexistent_provider"
        with pytest.raises(ValueError, match="Unknown provider"):
            lp._call_chat("p", system="s", temperature=0, model="m")

    def test_usage_summary_structure(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        lp = LLMProvider(provider="ollama")
        s = lp.usage_summary()
        assert "provider" in s and "total_calls" in s


# ==========================================================================
# LLMProvider – Ollama live tests
# ==========================================================================
@pytest.mark.llm
class TestLLMProviderOllama:
    @pytest.fixture(autouse=True)
    def require_ollama(self):
        """Skip the whole class if Ollama is unreachable at the start of each test."""
        if not _ollama_available():
            pytest.skip("Ollama not reachable at localhost:11434")

    @llm_mark
    def test_complete_json_returns_dict(self):
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        lp = LLMProvider(provider="ollama")
        result = lp.complete_json(
            'Return a JSON object with a single key "status" set to "ok".',
            model=_OLLAMA_MODEL,
            timeout=90,
        )
        assert isinstance(result, dict)

    @llm_mark
    def test_entity_extraction_from_real_text(self):
        """Extract entities from a short, real financial sentence (fast for 27B CPU)."""
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        # Keep to one sentence so the 27B model can respond within 60-90s on CPU
        text = "Apple Inc. (AAPL, LEI: 54930084UKLVMY22DS16) reported revenue of $94.9 billion in Q3 FY2024."
        lp = LLMProvider(provider="ollama")
        result = lp.complete_json(
            f'Extract financial entities as JSON with key "entities" '
            f'(array of {{"text": str, "label": str}}).\n\nTEXT:\n{text}',
            model=_OLLAMA_MODEL,
            timeout=90,
        )
        assert isinstance(result, dict)
        assert "entities" in result
        texts = [str(e.get("text", "")).lower() for e in result.get("entities", [])]
        assert any("apple" in t or "aapl" in t for t in texts), \
            f"Expected Apple/AAPL in extracted entities, got: {texts}"

    @llm_mark
    def test_cache_used_on_second_call(self, tmp_path):
        """First call goes to Ollama; second call hits cache (no second network req)."""
        from ChaptersFinancial._platform.providers.llm import LLMProvider
        lp = LLMProvider(provider="ollama")
        lp._cache_dir = tmp_path
        prompt = 'Respond with JSON: {"ping": "pong"}'
        r1 = lp.complete_json(prompt, model=_OLLAMA_MODEL, timeout=90)
        r2 = lp.complete_json(prompt, model=_OLLAMA_MODEL, timeout=90)  # must hit cache
        assert r1 == r2


# ==========================================================================
# CostTracker tests
# ==========================================================================
class TestCostTracker:
    def test_record_and_summary(self, tmp_path):
        from ChaptersFinancial._platform.obs.cost_tracker import CostTracker
        ct = CostTracker(chapter="test", run_id="abc")
        ct._out_dir = tmp_path
        ct.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=200)
        ct.record("gpt-4o-mini", prompt_tokens=500,  completion_tokens=100)
        s = ct.summary()
        assert s["totalCalls"] == 2
        assert s["totalPromptTokens"] == 1500
        assert s["totalCompletionTokens"] == 300
        assert s["totalCostUsd"] > 0

    def test_save_produces_file(self, tmp_path):
        from ChaptersFinancial._platform.obs.cost_tracker import CostTracker
        ct = CostTracker(chapter="test", run_id="xyz12345")
        ct._out_dir = tmp_path
        ct.record("gpt-4o-mini", 100, 50)
        out = ct.save()
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["chapter"] == "test"

    def test_free_local_model(self, tmp_path):
        from ChaptersFinancial._platform.obs.cost_tracker import CostTracker
        ct = CostTracker(chapter="ch10", run_id="abc")
        ct._out_dir = tmp_path
        cost = ct.record("llama3.1:latest", 5000, 2000)
        assert cost == 0.0


# ==========================================================================
# RunLogger tests
# ==========================================================================
class TestRunLogger:
    def test_context_manager_completed(self, tmp_path):
        from ChaptersFinancial._platform.obs.run_logger import RunLogger
        rl = RunLogger(chapter="test", module="unit", log_dir=tmp_path)
        with rl as run:
            run.set_metric("rows", 42)
        assert rl.status == "COMPLETED"
        log_file = tmp_path / "runs.jsonl"
        assert log_file.exists()
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l]
        last = lines[-1]
        assert last["status"] == "COMPLETED"
        assert last["metrics"]["rows"] == 42

    def test_context_manager_failed(self, tmp_path):
        from ChaptersFinancial._platform.obs.run_logger import RunLogger
        rl = RunLogger(chapter="test", module="unit", log_dir=tmp_path)
        with pytest.raises(ValueError):
            with rl:
                raise ValueError("boom")
        assert rl.status == "FAILED"

    def test_increment(self, tmp_path):
        from ChaptersFinancial._platform.obs.run_logger import RunLogger
        rl = RunLogger(chapter="test", module="unit", log_dir=tmp_path)
        rl.increment("docs", 3)
        rl.increment("docs", 7)
        assert rl._metrics["docs"] == 10


# ==========================================================================
# NEREval tests
# ==========================================================================
class TestNEREval:
    def test_perfect_prediction(self, tmp_path):
        from ChaptersFinancial._platform.eval.ner_eval import NEREval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "text": "Apple Inc. announced...",
            "entities": [{"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10}]
        }))
        ev = NEREval(gold_path=gold)
        preds = [[{"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10}]]
        report = ev.evaluate(preds)
        assert report["overall"]["f1"] == 1.0

    def test_zero_prediction(self, tmp_path):
        from ChaptersFinancial._platform.eval.ner_eval import NEREval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "text": "Apple Inc. announced...",
            "entities": [{"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10}]
        }))
        ev = NEREval(gold_path=gold)
        report = ev.evaluate([[]])
        assert report["overall"]["recall"] == 0.0

    def test_partial_prediction(self, tmp_path):
        from ChaptersFinancial._platform.eval.ner_eval import NEREval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps({"text": "...", "entities": [
                {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                {"text": "London", "label": "LOC", "start": 10, "end": 16},
            ]})
        )
        ev = NEREval(gold_path=gold)
        preds = [[{"text": "Apple", "label": "ORG", "start": 0, "end": 5}]]
        report = ev.evaluate(preds)
        assert report["overall"]["recall"] < 1.0


# ==========================================================================
# NEDEval tests
# ==========================================================================
class TestNEDEval:
    def test_all_correct(self, tmp_path):
        from ChaptersFinancial._platform.eval.ned_eval import NEDEval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "mentionId": "m001",
            "goldId": "LEI:ABC",
            "candidates": [{"id": "LEI:ABC", "score": 0.9}, {"id": "LEI:XYZ", "score": 0.5}]
        }))
        ev = NEDEval(gold_path=gold)
        r = ev.evaluate()
        assert r["accuracy@1"] == 1.0

    def test_all_wrong(self, tmp_path):
        from ChaptersFinancial._platform.eval.ned_eval import NEDEval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "mentionId": "m001",
            "goldId": "LEI:ABC",
            "candidates": [{"id": "LEI:XYZ", "score": 0.9}, {"id": "LEI:OTHER", "score": 0.5}]
        }))
        ev = NEDEval(gold_path=gold)
        r = ev.evaluate()
        assert r["accuracy@1"] == 0.0
        assert r["mrr"] == 0.0


# ==========================================================================
# MLEval tests
# ==========================================================================
class TestMLEval:
    def test_classification_perfect(self):
        pytest.importorskip("sklearn")
        from ChaptersFinancial._platform.eval.ml_eval import MLEval
        y_true  = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.1, 0.2]
        r = MLEval.classification_report(y_true, y_score)
        assert r["roc_auc"] == 1.0
        assert r["pr_auc"]  == 1.0

    def test_link_prediction_hits(self):
        pytest.importorskip("sklearn")
        from ChaptersFinancial._platform.eval.ml_eval import MLEval
        r = MLEval.link_prediction_report(
            pos_scores=[0.9, 0.8, 0.7],
            neg_scores=[0.1, 0.2, 0.3],
            ks=[1, 3],
        )
        assert r["hits@3"] == 1.0


# ==========================================================================
# RAGEval tests
# ==========================================================================
class TestRAGEval:
    def test_perfect_answer(self, tmp_path):
        from ChaptersFinancial._platform.eval.rag_eval import RAGEval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "question": "Who is the CEO?",
            "gold_answer": "Tim Cook is the CEO",
            "gold_entity_ids": ["lei:apple"]
        }))
        ev = RAGEval(gold_path=gold)
        answers = [{
            "question": "Who is the CEO?",
            "answer": "Tim Cook is the CEO",
            "citations": ["chunk1"],
            "evidence": ["lei:apple is a company. Tim Cook is the CEO."]
        }]
        r = ev.evaluate(answers)
        assert r["faithfulness"] == 1.0
        assert r["answer_correctness"] == 1.0

    def test_empty_answer(self, tmp_path):
        from ChaptersFinancial._platform.eval.rag_eval import RAGEval
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "question": "Q",
            "gold_answer": "The answer",
            "gold_entity_ids": ["lei:x"]
        }))
        ev = RAGEval(gold_path=gold)
        r = ev.evaluate([{"question": "Q", "answer": "", "citations": [], "evidence": []}])
        assert r["answer_correctness"] == 0.0


# ==========================================================================
# Config / Schema file tests
# ==========================================================================
class TestConfigFiles:
    def test_schema_config_loadable(self):
        import yaml
        config_path = _REPO_ROOT / "ChaptersFinancial/_platform/config/schema_config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert "nodes" in data
        assert "LegalEntity" in data["nodes"]
        assert "Instrument" in data["nodes"]
        assert "relationships" in data

    def test_provider_config_loadable(self):
        import yaml
        config_path = _REPO_ROOT / "ChaptersFinancial/_platform/config/provider_config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert "llm" in data
        assert "graph" in data
        assert "vector" in data

    def test_constraints_file_exists(self):
        path = _REPO_ROOT / "ChaptersFinancial/_platform/schema/constraints.cypher"
        assert path.exists()
        content = path.read_text()
        assert "LegalEntity" in content
        assert "Instrument"  in content
