"""
app.py — Financial Investigative Copilot (ch17_fin)
====================================================
Streamlit application for interactive financial knowledge graph exploration.

Panels:
  1. Entity Search & Profile
  2. Ownership Network
  3. Event Timeline
  4. Exposure Path Explorer
  5. Graph RAG Q&A with Citations
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider
from ChaptersFinancial.ch15_fin.code.tools import GraphRAGTools, validate_cypher


@st.cache_resource
def get_providers():
    gp = GraphProvider()
    llm = LLMProvider()
    tools = GraphRAGTools(gp, llm)
    return gp, llm, tools


def main():
    st.set_page_config(page_title="Financial Investigative Copilot", layout="wide")
    st.title("Financial Investigative Copilot")
    st.caption("Powered by Neo4j Knowledge Graph + LLM Graph RAG")

    gp, llm, tools = get_providers()

    # Sidebar: Entity Search
    st.sidebar.header("Entity Search")
    search_term = st.sidebar.text_input("Search entity by name", "")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Entity Profile", "Ownership Network", "Event Timeline",
        "Exposure Paths", "Q&A (Graph RAG)"
    ])

    # --- Tab 1: Entity Profile ---
    with tab1:
        st.header("Entity Profile")
        if search_term:
            profile = gp.run("""
                MATCH (le:LegalEntity)
                WHERE toLower(le.name) CONTAINS toLower($name)
                OPTIONAL MATCH (le)-[:ISSUES]->(i:Instrument)
                OPTIONAL MATCH (le)-[:CLASSIFIED_AS]->(oc:OntologyClass)
                OPTIONAL MATCH (f:Filing)-[:REPORTS_ON]->(le)
                RETURN le.lei AS lei, le.name AS name,
                       le.jurisdiction AS jurisdiction,
                       le.status AS status,
                       le.legalForm AS legalForm,
                       le.pagerank AS pagerank,
                       collect(DISTINCT i.ticker) AS tickers,
                       collect(DISTINCT oc.label) AS classifications,
                       count(DISTINCT f) AS filingCount
                LIMIT 5
            """, {"name": search_term})

            if profile:
                for p in profile:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Entity", p["name"])
                        st.text(f"LEI: {p['lei']}")
                        st.text(f"Jurisdiction: {p.get('jurisdiction', 'N/A')}")
                        st.text(f"Status: {p.get('status', 'N/A')}")
                    with col2:
                        st.metric("Filings", p["filingCount"])
                        st.text(f"PageRank: {p.get('pagerank', 'N/A')}")
                        st.text(f"Tickers: {', '.join(p.get('tickers', []))}")
                        st.text(f"Classes: {', '.join(p.get('classifications', []))}")
                    st.divider()
            else:
                st.info("No entities found.")
        else:
            st.info("Enter an entity name in the sidebar to search.")

    # --- Tab 2: Ownership Network ---
    with tab2:
        st.header("Ownership Network")
        if search_term:
            ownership = gp.run("""
                MATCH (le:LegalEntity)
                WHERE toLower(le.name) CONTAINS toLower($name)
                OPTIONAL MATCH (le)-[r:OWNS|CONTROLS|PARENT_OF]-(related:LegalEntity)
                RETURN le.name AS entity,
                       type(r) AS relType,
                       related.name AS related,
                       r.pct AS ownershipPct
                LIMIT 50
            """, {"name": search_term})

            if ownership:
                st.dataframe(ownership)
            else:
                st.info("No ownership relationships found.")

    # --- Tab 3: Event Timeline ---
    with tab3:
        st.header("Event Timeline")
        if search_term:
            events = gp.run("""
                MATCH (le:LegalEntity)
                WHERE toLower(le.name) CONTAINS toLower($name)
                MATCH (e:Event)-[:AFFECTS]->(le)
                RETURN e.type AS eventType, e.description AS description,
                       e.occurredAt AS date, e.confidence AS confidence
                ORDER BY e.occurredAt DESC
                LIMIT 20
            """, {"name": search_term})

            if events:
                st.dataframe(events)
            else:
                st.info("No events found for this entity.")

    # --- Tab 4: Exposure Paths ---
    with tab4:
        st.header("Exposure Path Explorer")
        col1, col2 = st.columns(2)
        entity_a = col1.text_input("From entity", search_term)
        entity_b = col2.text_input("To entity", "")
        max_hops = st.slider("Max hops", 1, 6, 3)

        if entity_a and entity_b:
            paths = gp.run("""
                MATCH (a:LegalEntity), (b:LegalEntity)
                WHERE toLower(a.name) CONTAINS toLower($a)
                  AND toLower(b.name) CONTAINS toLower($b)
                MATCH path = shortestPath((a)-[:OWNS|CONTROLS|PARENT_OF|EXPOSED_TO*..%d]-(b))
                RETURN [n IN nodes(path) | n.name] AS nodes,
                       [r IN relationships(path) | type(r)] AS rels,
                       length(path) AS hops
                LIMIT 5
            """ % max_hops, {"a": entity_a, "b": entity_b})

            if paths:
                for p in paths:
                    st.write(f"**Path ({p['hops']} hops):** {' → '.join(p['nodes'])}")
                    st.caption(f"Relationships: {', '.join(p['rels'])}")
            else:
                st.info("No paths found between these entities.")

    # --- Tab 5: Graph RAG Q&A ---
    with tab5:
        st.header("Q&A — Graph RAG")
        question = st.text_area("Ask a question about the financial knowledge graph")

        if st.button("Ask") and question:
            with st.spinner("Thinking…"):
                answer = tools.answer_question(question)

            st.subheader("Answer")
            st.write(answer.get("answer", "No answer generated."))

            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{answer.get('confidence', 0):.2f}")
            col2.metric("Sources Used", answer.get("retrieved_chunks", 0))

            if answer.get("citations"):
                st.subheader("Citations")
                for c in answer["citations"]:
                    st.caption(c)


if __name__ == "__main__":
    main()
