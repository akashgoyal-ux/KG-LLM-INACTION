PYTHONPATH ?= ../../
PYTHON     ?= python
PIP        ?= pip

.PHONY: install test test-unit test-llm connections connections-live

install:
	$(PIP) install -r requirements.txt

# Run fast unit tests only (no LLM/Neo4j required)
test-unit:
	PYTHONPATH=$(PYTHONPATH) \
	    pytest setup/test_platform.py -v --tb=short -m "not llm"

# Run all tests including Ollama LLM tests
test:
	PYTHONPATH=$(PYTHONPATH) \
	    pytest setup/test_platform.py -v --tb=short

# Run only LLM tests (requires Ollama + deepseek-v3.2:cloud)
test-llm:
	PYTHONPATH=$(PYTHONPATH) \
	    pytest setup/test_platform.py -v --tb=short -m llm

# Check connectivity (Neo4j ping + Ollama version check)
connections:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch01_fin.setup.test_connections

# Check connectivity including a live Ollama LLM call
connections-live:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch01_fin.setup.test_connections --live
