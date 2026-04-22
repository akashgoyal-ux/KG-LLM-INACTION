PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: build-dataset train-classify train-link

build-dataset:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch14_fin.data.build_pyg_dataset

train-classify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch14_fin.train_for_classification

train-link:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch14_fin.train_for_link_prediction
