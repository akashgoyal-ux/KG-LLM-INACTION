PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: node2vec cluster classify

node2vec:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch11_fin.analysis.node2vec_issuers

cluster:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch11_fin.analysis.cluster_issuers

classify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch11_fin.analysis.classify_high_risk
