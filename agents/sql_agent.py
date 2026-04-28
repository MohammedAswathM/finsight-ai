"""SQL agent — OWNED BY MEMBER 2 (feature/sql-chart branch). Placeholder."""
from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import append_trace
from state import AgentState


def run_sql(state: AgentState) -> Dict[str, Any]:
    # Populate transaction_features for fraud detection from SQL agent output.
    # This enables the fraud branch to execute end-to-end in the current workflow.
    sample_transaction_features = {
        "Time": 10000.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311139,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110053,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62,
    }

    return {
        "sql_result": "SQL agent: provided transaction_features for fraud detection.",
        "transaction_features": sample_transaction_features,
        "trace_log": append_trace(
            "SQL agent: populated transaction_features for fraud prediction"
        ),
    }


if __name__ == "__main__":
    print(run_sql({"query": "AAPL close for last 6 months"}))
