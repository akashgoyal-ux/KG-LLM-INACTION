"""
ChaptersFinancial._platform
============================
Layer 0 – shared infrastructure for every Financial chapter.

Exports the most commonly used symbols so callers can do:
    from ChaptersFinancial._platform import FinImporterBase, LLMProvider
"""

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase  # noqa: F401
from ChaptersFinancial._platform.providers.llm import LLMProvider           # noqa: F401
from ChaptersFinancial._platform.providers.graph import GraphProvider        # noqa: F401
