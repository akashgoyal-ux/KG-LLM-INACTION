"""
conversational_agent.py
========================
Multi-turn conversational agent with Graph RAG memory.
Maintains conversation history and can reference prior answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider
from ChaptersFinancial.ch15_fin.code.tools import GraphRAGTools


class ConversationalAgent:
    def __init__(self, gp: GraphProvider = None, llm: LLMProvider = None):
        self.gp = gp or GraphProvider()
        self.llm = llm or LLMProvider()
        self.tools = GraphRAGTools(self.gp, self.llm)
        self.history: list[dict] = []

    def ask(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})

        # Get Graph RAG answer
        rag_result = self.tools.answer_question(question)
        answer = rag_result.get("answer", "I could not find an answer.")

        # Enhance with conversation context
        if len(self.history) > 2:
            context = "\n".join(
                f"{m['role']}: {m['content'][:200]}"
                for m in self.history[-6:]
            )
            enhanced = self.llm.complete_json(
                f"Given this conversation:\n{context}\n\n"
                f"And this Graph RAG answer: {answer}\n\n"
                f"Provide a coherent response. Return {{\"answer\": ...}}"
            )
            if isinstance(enhanced, dict) and "answer" in enhanced:
                answer = enhanced["answer"]

        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self):
        self.history.clear()


if __name__ == "__main__":
    agent = ConversationalAgent()
    questions = [
        "What entities are in the knowledge graph?",
        "Which of those have the most filings?",
        "Tell me more about the top one.",
    ]
    for q in questions:
        print(f"\nUser: {q}")
        print(f"Agent: {agent.ask(q)}")
