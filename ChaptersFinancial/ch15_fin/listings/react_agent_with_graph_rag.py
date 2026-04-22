"""
react_agent_with_graph_rag.py
==============================
ReAct-style agent that iterates: Thought → Action → Observation
using Graph RAG tools (vector_search, kg_reader, cypher_query).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider
from ChaptersFinancial.ch15_fin.code.tools import GraphRAGTools, validate_cypher

REACT_SYSTEM = """You are a financial analyst assistant with access to a Knowledge Graph.

Available tools:
- vector_search(query): semantic search over document chunks
- kg_reader(entity_name): look up entity in KG
- cypher_query(cypher): run a read-only Cypher query

Follow this format:
Thought: (your reasoning)
Action: tool_name(args)
Observation: (tool output - provided by system)
... repeat as needed ...
Answer: (final answer with citations)

Only generate safe, read-only Cypher. Never use CREATE, DELETE, SET, REMOVE."""

MAX_STEPS = 5


def run_react_agent(question: str, gp: GraphProvider = None, llm: LLMProvider = None):
    gp = gp or GraphProvider()
    llm = llm or LLMProvider()
    tools = GraphRAGTools(gp, llm)

    messages = [{"role": "system", "content": REACT_SYSTEM}]
    messages.append({"role": "user", "content": question})

    for step in range(MAX_STEPS):
        resp = llm.complete_json(
            f"{REACT_SYSTEM}\n\nConversation so far:\n"
            + "\n".join(m["content"] for m in messages[1:])
            + "\n\nContinue with Thought/Action or Answer.",
        )
        text = str(resp)

        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
            return {"answer": answer, "steps": step + 1}

        # Parse action
        if "Action:" in text:
            action_line = text.split("Action:")[-1].split("\n")[0].strip()
            observation = _execute_action(action_line, tools, gp)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "assistant", "content": text})

    return {"answer": "Reached max steps without answer.", "steps": MAX_STEPS}


def _execute_action(action: str, tools: GraphRAGTools, gp: GraphProvider) -> str:
    try:
        if action.startswith("vector_search("):
            query = action.split("(", 1)[1].rstrip(")")
            results = tools.vector_search(query)
            return str(results[:3])
        elif action.startswith("kg_reader("):
            entity = action.split("(", 1)[1].rstrip(")")
            return str(tools.kg_reader(entity))
        elif action.startswith("cypher_query("):
            cypher = action.split("(", 1)[1].rstrip(")")
            ok, msg = validate_cypher(cypher)
            if not ok:
                return f"BLOCKED: {msg}"
            return str(gp.run(cypher)[:5])
        else:
            return "Unknown tool."
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    result = run_react_agent("What are the most connected entities in the financial graph?")
    print(f"Answer: {result['answer']}")
    print(f"Steps: {result['steps']}")
