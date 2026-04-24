"""RAG agent — OWNED BY MEMBER 1 (feature/rag branch).

This file is a placeholder so `orchestrator.graph` imports cleanly before the
Member 1 PR lands. Member 1 will replace the body. Signature is fixed:

    def run(state: AgentState) -> dict
"""
from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import append_trace
from state import AgentState


def run(state: AgentState) -> Dict[str, Any]:
    return {
        "rag_result": "[placeholder] RAG agent not implemented yet (Member 1).",
        "sources": [],
        "trace_log": append_trace("RAG agent: placeholder hit (Member 1 pending)"),
    }


if __name__ == "__main__":
    print(run({"query": "What was Apple's revenue in 2024?"}))
