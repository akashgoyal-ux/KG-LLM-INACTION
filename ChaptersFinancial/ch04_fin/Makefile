.PHONY: init import-exchanges import-instruments import-ownership import-sectors import analysis verify all

PYTHONPATH := $(shell cd ../.. && pwd)
RUN        := PYTHONPATH=$(PYTHONPATH) uv run --active python

init:
	$(RUN) -c "\
from ChaptersFinancial._platform.fin_importer_base import FinImporterBase; \
i = FinImporterBase(); i.ensure_constraints(); i.close(); \
print('Constraints created.')"

import-exchanges:
	$(RUN) importer/import_exchanges.py

import-instruments:
	$(RUN) importer/import_instruments_figi.py

import-ownership:
	$(RUN) importer/import_ownership_13f.py

import-sectors:
	$(RUN) importer/import_sector_mapping.py

import: init import-exchanges import-instruments import-ownership import-sectors
	@echo "\n✓ ch04_fin import complete."

community:
	$(RUN) analysis/community_louvain.py

pagerank:
	$(RUN) analysis/centrality_pagerank.py

analysis: community pagerank
	@echo "\n✓ ch04_fin analysis complete."

verify:
	$(RUN) analysis/graph_summary.py

all: import analysis verify
