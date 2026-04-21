PYTHON     ?= python
PIP        ?= pip

# Umbrella – delegates to each chapter's Makefile.
# Usage:
#   make install     – install all chapter dependencies
#   make test        – run all offline tests
#   make test-ch01   – run ch01_fin tests only
#   make test-ch02   – run ch02_fin tests only
#   make import-ch03 – run ch03_fin ontology import
#   make import-ch04 – run ch04_fin multi-source import + analysis

.PHONY: install test test-ch01 test-ch02 import-ch03 import-ch04 verify-ch03 verify-ch04

install:
	cd ch01_fin && $(MAKE) install
	cd ch02_fin && $(MAKE) install

test: test-ch01 test-ch02

test-ch01:
	cd ch01_fin && $(MAKE) test

test-ch02:
	cd ch02_fin && $(MAKE) test

import-ch03:
	cd ch03_fin && $(MAKE) import

verify-ch03:
	cd ch03_fin && $(MAKE) verify

import-ch04:
	cd ch04_fin && $(MAKE) all

verify-ch04:
	cd ch04_fin && $(MAKE) verify
