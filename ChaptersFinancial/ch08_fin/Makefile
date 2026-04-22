PYTHONPATH ?= ../../
PYTHON     ?= python
PIP        ?= pip

.PHONY: install init ingest extract merge eval all

install:
	$(PIP) install -r requirements.txt

ingest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch08_fin.importer.ingest_filings

extract:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch08_fin.importer.extract_with_llm

merge:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch08_fin.importer.normalize_and_merge

all: ingest extract merge
