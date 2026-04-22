PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: init disambiguate eval

disambiguate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch09_fin.disambiguation.main

eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch09_fin.eval.eval_ned
