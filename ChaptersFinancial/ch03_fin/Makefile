.PHONY: init import-fibo import-gleif import-currencies import verify analysis

PYTHONPATH := $(shell cd ../.. && pwd)
RUN        := PYTHONPATH=$(PYTHONPATH) uv run --active python

init:
	$(RUN) -c "\
from ChaptersFinancial._platform.fin_importer_base import FinImporterBase; \
i = FinImporterBase(); i.ensure_constraints(); i.close(); \
print('Constraints created.')"

import-fibo:
	$(RUN) importer/import_fibo.py

import-gleif:
	$(RUN) importer/import_gleif.py

import-currencies:
	$(RUN) importer/import_currencies.py

import: init import-fibo import-gleif import-currencies
	@echo "\n✓ ch03_fin import complete."

analysis:
	$(RUN) analysis/ontology_analysis.py

verify:
	$(RUN) importer/verify.py
