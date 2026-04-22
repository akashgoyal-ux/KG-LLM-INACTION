PYTHONPATH ?= ../../
PYTHON     ?= python

.PHONY: embed-chunks run-rag agent

embed-chunks:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch15_fin.code.chunk_embed_pipeline

run-rag:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch15_fin.code.tools

agent:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ChaptersFinancial.ch15_fin.listings.react_agent_with_graph_rag
