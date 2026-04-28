"""
state.py — Shared AgentState TypedDict for FinSight AI orchestrator.
All agents read/write from this single state object.
DO NOT MODIFY WITHOUT NOTIFYING ALL MEMBERS.
"""

from typing import TypedDict, Optional, List, Dict


class AgentState(TypedDict):
    # INPUT
    query: str                        # original user question — never modify this

    # PLANNING
    plan: Optional[List[str]]         # planner's list of sub-tasks
    agents_to_call: Optional[List[str]]  # which agents the planner decided to invoke

    # AGENT OUTPUTS
    rag_result: Optional[str]         # cited text from SEC filings
    sources: Optional[List[str]]      # list of source citations from RAG

    sql_result: Optional[str]         # structured data from SQLite query
    chart_path: Optional[str]         # file path to generated chart PNG

    sentiment_result: Optional[str]   # sentiment score + summary string
    image_data: Optional[str]         # base64 encoded user-uploaded image

    # ML MODEL OUTPUTS (AIML Infra course)
    fraud_score: Optional[Dict]       # {"fraud_probability": 0.87, "risk_level": "HIGH", "is_fraud": True}
    forecast: Optional[Dict]          # {"direction": "UP", "confidence": 0.74, "days_ahead": 5}

    # EVALUATION + CONTROL FLOW
    eval_score: Optional[float]       # critic's quality score 0.0–1.0
    eval_feedback: Optional[str]      # critic's explanation of what was weak
    retry_count: int                  # how many reflection loops have run (start at 0)

    # FINAL OUTPUT
    final_report: Optional[str]       # synthesized final answer shown to user

    # TRACE LOG (for UI display)
    trace_log: Optional[List[str]]    # list of strings describing what each agent did
