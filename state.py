from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    query: str                        # original user question
    plan: Optional[List[str]]         # planner's list of sub-tasks
    rag_result: Optional[str]         # output from RAG agent
    sql_result: Optional[str]         # output from SQL agent
    chart_path: Optional[str]         # file path to generated chart
    sentiment_result: Optional[str]   # output from sentiment agent
    image_data: Optional[str]         # base64 image if user uploaded one
    eval_score: Optional[float]       # critic's quality score 0.0–1.0
    retry_count: int                  # how many reflection loops have run
    final_report: Optional[str]       # synthesized final answer
    sources: Optional[List[str]]      # citations used