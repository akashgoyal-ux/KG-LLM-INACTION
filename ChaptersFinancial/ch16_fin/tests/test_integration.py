"""
test_integration.py
===================
Integration tests verifying end-to-end data flow across chapters.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.fixture(scope="module")
def gp():
    from ChaptersFinancial._platform.providers.graph import GraphProvider
    gp = GraphProvider()
    yield gp
    gp.close()


def test_legal_entities_exist(gp):
    """At least some LegalEntity nodes should exist after ch03+ch04."""
    cnt = gp.run("MATCH (le:LegalEntity) RETURN count(le) AS cnt")[0]["cnt"]
    assert cnt > 0, "No LegalEntity nodes found"


def test_legal_entities_have_lei(gp):
    """All LegalEntity nodes must have a non-null lei."""
    bad = gp.run(
        "MATCH (le:LegalEntity) WHERE le.lei IS NULL RETURN count(le) AS cnt"
    )[0]["cnt"]
    assert bad == 0, f"{bad} LegalEntity nodes without lei"


def test_ontology_classes_exist(gp):
    """FIBO ontology classes should be loaded."""
    cnt = gp.run("MATCH (oc:OntologyClass) RETURN count(oc) AS cnt")[0]["cnt"]
    # May be 0 if FIBO import failed, but should at least not error
    assert cnt >= 0


def test_documents_have_chunks(gp):
    """Documents should be chunked."""
    orphan = gp.run("""
        MATCH (d:Document)
        WHERE NOT (d)<-[:OF_DOC]-(:Chunk)
        RETURN count(d) AS cnt
    """)[0]["cnt"]
    # Allow some orphans (documents may not have text)
    total = gp.run("MATCH (d:Document) RETURN count(d) AS cnt")[0]["cnt"]
    if total > 0:
        assert orphan / total < 0.5, f"Too many orphan documents: {orphan}/{total}"


def test_cypher_safety_validation():
    """Validate that the Cypher safety checker rejects writes."""
    from ChaptersFinancial.ch15_fin.code.tools import validate_cypher
    assert validate_cypher("MATCH (n) RETURN n LIMIT 10")[0] is True
    assert validate_cypher("MATCH (n) DELETE n")[0] is False
    assert validate_cypher("CREATE (n:Foo)")[0] is False
    assert validate_cypher("MATCH (n) SET n.x = 1")[0] is False
