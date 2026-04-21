# ch01_fin — Platform Setup & Validation

## Purpose
This chapter sets up the shared `_platform` infrastructure used by every subsequent financial chapter. It contains:
- Connection validators for Neo4j and the LLM provider
- pytest unit tests for all platform modules (no API key required)

## Prerequisites
- Python 3.9+
- (Optional) Neo4j 4.4+ or 5.x running locally on bolt://localhost:7687
- (Optional) OpenAI API key or local Ollama instance for live LLM tests

## Quick Start

```bash
cd ChaptersFinancial/ch01_fin

# Install dependencies
make install

# Run all unit tests (fully offline, no API key required)
make test

# Check connectivity (Neo4j ping + mock LLM response)
make connections

# Check with a live LLM call (requires OPENAI_API_KEY env var or Ollama running)
make connections-live
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `openai` / `azure` / `ollama` / `mock` |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `AZURE_OPENAI_API_KEY` | — | Required for Azure provider |
| `AZURE_OPENAI_ENDPOINT` | — | Azure endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | — | Deployment name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `NEO4J_DATABASE` | `fin_core` | Target database |

Copy `.env.example` at the repo root to `.env` and fill in your values.

## What is Tested
- All `_platform` module imports
- LLM provider (mock mode): JSON completion, embed, cache roundtrip
- CostTracker: record, summary, save
- RunLogger: context manager, metrics, JSONL output
- NEREval, NEDEval, MLEval, RAGEval: correctness of evaluation logic
- Config YAML files: parseable and complete
- Constraints Cypher file: exists and references key labels
