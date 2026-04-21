PYTHONPATH ?= ../../
PYTHON     ?= python
PIP        ?= pip

.PHONY: install test test-unit test-llm validate run-01 run-02 run-03 run-04 run-live

install:
	$(PIP) install -r requirements.txt

# Fast unit tests – schema, golden data, guardrail regex (no LLM)
test-unit:
	PYTHONPATH=$(PYTHONPATH) \
	    pytest tests/test_schemas.py -v --tb=short -m "not llm"

# All tests including Ollama LLM tests
test:
	PYTHONPATH=$(PYTHONPATH) \
	    pytest tests/test_schemas.py -v --tb=short

# LLM tests only
test-llm:
	PYTHONPATH=$(PYTHONPATH) \
	    pytest tests/test_schemas.py -v --tb=short -m llm

# Validate schemas + golden data
validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) validate_prompts.py

# Run individual listings (Ollama)
run-01:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) listings/01_zero_shot_entity_extraction.py

run-02:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) listings/02_few_shot_event_extraction.py

run-03:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) listings/03_chain_of_verification_filings.py

run-04:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) listings/04_compliance_guardrails.py
