PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: migrate validate test ci

migrate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch16_fin.migrations.apply_migrations

validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch16_fin.contracts.validate_contracts

test:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -v --tb=short

ci: migrate validate test
