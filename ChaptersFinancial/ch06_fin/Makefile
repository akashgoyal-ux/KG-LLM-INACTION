PYTHONPATH ?= ../../
PYTHON     ?= python
PIP        ?= pip

.PHONY: install init ingest ner enrich all

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_trf

init: install

ingest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch06_fin.importer.step1_ingest_news

ner:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch06_fin.importer.step2_run_ner

enrich:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch06_fin.importer.step3_enrich_via_apis

all: ingest ner enrich
