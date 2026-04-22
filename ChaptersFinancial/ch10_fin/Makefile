PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: disambiguate benchmark

disambiguate:
	LLM_PROVIDER=ollama PYTHONPATH=$(PYTHONPATH) \
	    $(PYTHON) -m ChaptersFinancial.ch10_fin.disambiguation.main

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch10_fin.benchmark.run_benchmark
