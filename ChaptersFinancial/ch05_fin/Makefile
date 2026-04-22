PYTHONPATH ?= ../../
PYTHON     ?= python
PIP        ?= pip

.PHONY: install init reconcile eval test

install:
	$(PIP) install -r requirements.txt

init:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch05_fin.reconciliation.deterministic_match

reconcile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch05_fin.reconciliation.deterministic_match
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch05_fin.reconciliation.probabilistic_match

eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch05_fin.eval.eval_reconciliation

test:
	PYTHONPATH=$(PYTHONPATH) pytest eval/ -v --tb=short
