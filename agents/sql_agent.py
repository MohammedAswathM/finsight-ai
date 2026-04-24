"""SQL agent — OWNED BY MEMBER 2 (feature/sql-chart branch). Placeholder."""
from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import append_trace
from state import AgentState


def run_sql(state: AgentState) -> Dict[str, Any]:
    return {
        "sql_result": "[placeholder] SQL agent not implemented yet (Member 2).",
        "trace_log": append_trace("SQL agent: placeholder hit (Member 2 pending)"),
    }


if __name__ == "__main__":
    print(run_sql({"query": "AAPL close for last 6 months"}))
