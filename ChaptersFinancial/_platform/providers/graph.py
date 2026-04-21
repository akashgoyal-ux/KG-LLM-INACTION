"""
GraphProvider
=============
Thin Neo4j driver wrapper that reads from provider_config.yaml / env vars.
Chapters import this instead of touching util.graphdb_base directly, so the
database name and connection details stay in one place.

Usage
-----
from ChaptersFinancial._platform.providers.graph import GraphProvider

gp = GraphProvider()
with gp.session() as s:
    records = s.run("MATCH (le:LegalEntity) RETURN le.name LIMIT 5").data()
gp.close()
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from ChaptersFinancial/ (two levels up from this file's directory)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "provider_config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


class GraphProvider:
    """
    Read-oriented Neo4j driver abstraction.
    For write-heavy import jobs use FinImporterBase (which uses the neo4j driver directly).
    """

    def __init__(self):
        cfg = _load_config()["graph"]
        uri = os.getenv("NEO4J_URI") or cfg.get("uri", "neo4j://localhost:7687")
        print(f"Connecting to Neo4j at... {uri}...")
        user = os.getenv("NEO4J_USER") or cfg.get("user", "neo4j")
        password = os.getenv("NEO4J_PASSWORD") or cfg.get("password", "")
        self._database = os.getenv("NEO4J_DATABASE") or cfg.get("database", "neo4j")

        encrypted = int(cfg.get("encrypted", 0))

        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(
                uri, auth=(user, password), encrypted=bool(encrypted)
            )
        except ImportError as exc:
            raise ImportError("pip install neo4j to use GraphProvider") from exc

    def session(self, **kwargs):
        return self._driver.session(database=self._database, **kwargs)

    def run(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self.session() as s:
            return s.run(cypher, params or {}).data()

    def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self.session() as s:
                s.run("RETURN 1").single()
            return True
        except Exception:
            return False

    def close(self):
        self._driver.close()
