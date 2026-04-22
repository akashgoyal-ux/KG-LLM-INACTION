PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: install run

install:
	$(PIP) install -r requirements.txt

run:
	PYTHONPATH=$(PYTHONPATH) streamlit run app.py
