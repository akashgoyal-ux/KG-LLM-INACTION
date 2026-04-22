PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: features train export

features:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch12_fin.features.node_features
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch12_fin.features.edge_features

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch12_fin.models.train_rf_baseline

export:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch12_fin.store.export_parquet
