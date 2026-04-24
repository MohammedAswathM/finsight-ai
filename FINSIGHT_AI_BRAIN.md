# FINSIGHT AI — THE COMPLETE PROJECT BRAIN
> **READ THIS FIRST.** This document is the single source of truth for FinSight AI. Every AI tool, every developer, every reviewer working on this project must read this document before touching any code. It contains the complete project idea, architecture, team split, implementation contracts, integration rules, and everything needed to build, review, and fix this system end-to-end.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [The Solution — What FinSight AI Does](#3-the-solution)
4. [Course Context — Two Courses, One Project](#4-course-context)
5. [Tech Stack — Complete List](#5-tech-stack)
6. [Repository Structure](#6-repository-structure)
7. [The Shared Contract — AgentState](#7-the-shared-contract)
8. [System Architecture — End to End Flow](#8-system-architecture)
9. [Member 1 — RAG Agent + Fraud Detection Model](#9-member-1)
10. [Member 2 — SQL + Chart Agent + FinBERT Model](#10-member-2)
11. [Member 3 — Orchestrator (YOU — READ CAREFULLY)](#11-member-3-orchestrator)
12. [Member 4 — UI + Sentiment Agent + MLOps Report](#12-member-4)
13. [Integration Rules — How Everything Merges](#13-integration-rules)
14. [The Three ML Models — AIML Infra Course](#14-the-three-ml-models)
15. [Demo Flow — Exactly What Happens](#15-demo-flow)
16. [What Good Output Looks Like](#16-what-good-output-looks-like)
17. [Common Failure Points and Fixes](#17-common-failure-points-and-fixes)
18. [Reviewing PRs — Checklist](#18-reviewing-prs)

---

## 1. PROJECT OVERVIEW

**Project Name:** FinSight AI — Autonomous Financial Research System

**One-line description:** A multi-agent AI system that takes a plain English financial question, deploys four specialized AI agents to research it from multiple sources, self-corrects the output using a reflection loop, and returns a grounded, cited, multi-source financial report in under 60 seconds.

**GitHub repo:** https://github.com/MohammedAswathM/finsight-ai

**Team size:** 4 members

**Deadline:** April 27, 2026

**Main branch:** `main` — only the orchestrator (Member 3) merges PRs here

**Presentation date:** April 27, 2026

---

## 2. THE PROBLEM WE ARE SOLVING

Financial research today requires an analyst to manually open multiple disconnected systems:

- SEC EDGAR (PDF filings) — 200+ page documents
- Finance terminals — price and volume data
- News sites — current headlines and sentiment
- Competitor reports — charts and images

This manual process takes 4–6 hours per company query, is error-prone, and cannot be delegated to a generic AI chatbot because those hallucinate financial facts (they make up revenue numbers, mix up fiscal quarters, cite non-existent reports).

**The two-layer problem:**
1. Data is siloed across incompatible formats and sources
2. AI assistants on financial topics are unreliable without grounding

---

## 3. THE SOLUTION

FinSight AI replaces the manual workflow with four specialized AI agents coordinated by a stateful orchestrator:

- **RAG Agent** — reads and retrieves from real SEC filings stored in a vector database
- **SQL + Chart Agent** — queries real stock price data and generates charts
- **Sentiment Agent** — fetches live news and classifies sentiment using a fine-tuned FinBERT model
- **Orchestrator** — plans, routes, evaluates, self-corrects, and synthesizes the final report

A critic/evaluator node reviews all agent outputs before the user sees anything. If quality is below threshold, the graph loops back and retries. The final output is a structured report with citations, a price chart, a sentiment score, and a 5-day price forecast — all grounded in real data or trained models.

**Key differentiator:** Every claim traces back to either a real document (RAG-grounded) or a real trained ML model (model-grounded). Nothing is hallucinated.

---

## 4. COURSE CONTEXT

This project serves **two courses simultaneously:**

### Course A — Foundations of Generative & Agentic AI (AIML Agentic)
Covers: RAG, vector databases, LangChain agents, LangGraph orchestration, multi-agent systems, evaluation.

| Lab Module | FinSight Feature |
|---|---|
| Lab 1 — RAG + prompt engineering | RAG agent with SEC filings |
| Lab 2 — Vector DBs + advanced retrievers | ChromaDB + ContextualCompression + ParentDocRetriever |
| Lab 3 — LangChain agents + tool calling | SQL agent, chart agent, sentiment agent |
| Lab 4 — LangGraph + agentic RAG | Full LangGraph StateGraph + reflection loop |
| Lab 5 — Multi-agent + multimodal | Critic agent + GPT-4V image upload |

### Course B — AIML Infrastructure Engineering
Covers: ML model training, deployment pipelines, MLOps, experiment tracking.

| Requirement | FinSight Implementation |
|---|---|
| Train ML models | Fraud detector (XGBoost + LightGBM), FinBERT fine-tune, Price forecaster |
| MLOps + experiment tracking | Local MLflow — tracks all 3 models |
| Model evaluation + comparison | XGBoost vs LightGBM, base vs fine-tuned FinBERT |
| Deployment-ready architecture | MLflow model registry + inference wrappers |

**Important:** No Azure, no cloud required. All models run locally. MLflow runs locally at `localhost:5000`. If asked about deployment, the answer is: "Our inference wrappers follow the MLflow serving interface and can be deployed to any endpoint (Azure ML, SageMaker, local FastAPI) by swapping a function call for an HTTP request."

---

## 5. TECH STACK

### Everything is free and open-source. No payment required anywhere.

| Category | Tool | Why |
|---|---|---|
| Orchestration | LangGraph | Stateful graph with cycle support — required for reflection loop |
| Agent framework | LangChain | Retriever components, tool abstractions, SQL agent |
| LLM (free) | Groq API — `llama-3.3-70b-versatile` | Fast, free, no credit card. Sign up: console.groq.com |
| Embeddings (free) | `sentence-transformers/all-MiniLM-L6-v2` | Local, no API key, runs offline |
| Vector DB | ChromaDB | Local, persistent, no server needed |
| SQL data | SQLite + yfinance | Free stock data, lightweight local DB |
| Charts | matplotlib | Standard, no cost |
| Sentiment model | FinBERT (ProsusAI/finbert) | HuggingFace, pre-trained on financial text |
| Fraud model | XGBoost + LightGBM | Scikit-learn compatible, fast, free |
| Experiment tracking | MLflow (local) | `mlflow ui` → localhost:5000 |
| Evaluation | RAGAS | Open source RAG evaluation |
| UI | Gradio | Free, local, streaming support |
| News | feedparser (RSS) | Free Yahoo Finance RSS, no API key |
| Vision (optional) | GPT-4V via OpenAI | Only if API key available — not required for demo |

### Python version: 3.11+

### Key packages (requirements.txt must include all of these):
```
langchain>=0.2.0
langchain-community
langchain-groq
langgraph>=0.1.0
chromadb
sentence-transformers
transformers
torch
datasets
accelerate
xgboost
lightgbm
imbalanced-learn
mlflow
scikit-learn
shap
yfinance
pandas
pandas-ta
matplotlib
gradio
ragas
feedparser
python-dotenv
sqlalchemy
```

---

## 6. REPOSITORY STRUCTURE

```
finsight-ai/                          ← root
├── state.py                          ← SHARED CONTRACT — never edit without telling everyone
├── config.py                         ← env var loading (GROQ_API_KEY, NEWSAPI_KEY)
├── requirements.txt                  ← all packages
├── .env.example                      ← template for .env file (never commit .env)
├── .gitignore                        ← includes .env, __pycache__, *.pkl, chroma_db/
│
├── agents/                           ← one file per agent
│   ├── base_agent.py                 ← Member 3 — shared LLM setup, utilities
│   ├── rag_agent.py                  ← Member 1
│   ├── sql_agent.py                  ← Member 2
│   ├── chart_agent.py                ← Member 2
│   └── sentiment_agent.py            ← Member 4
│
├── orchestrator/                     ← Member 3 OWNS this entire folder
│   ├── graph.py                      ← LangGraph StateGraph — the brain
│   ├── planner.py                    ← ReAct planner node
│   ├── evaluator.py                  ← critic/reflection node
│   └── synthesizer.py                ← final report generator node
│
├── retrieval/                        ← Member 1
│   ├── ingest.py                     ← PDF ingestion + chunking
│   ├── vectorstore.py                ← ChromaDB setup
│   └── retriever.py                  ← ContextualCompression + ParentDoc
│
├── data/                             ← Member 2
│   ├── db_setup.py                   ← yfinance → SQLite
│   └── fetch_prices.py               ← price fetching utilities
│
├── models/                           ← AIML Infra course — ML models
│   ├── train_fraud.py                ← Member 1 — XGBoost + LightGBM training
│   ├── fraud_detector.py             ← Member 1 — predict_fraud() wrapper
│   ├── train_finbert.py              ← Member 2 — FinBERT fine-tuning
│   ├── sentiment_model.py            ← Member 2 — predict_sentiment() wrapper
│   ├── feature_engineering.py        ← Member 3 — yfinance feature builder
│   ├── train_forecaster.py           ← Member 3 — LightGBM price forecaster
│   └── forecaster.py                 ← Member 3 — predict_trend() wrapper
│
├── ui/                               ← Member 4
│   ├── app.py                        ← Gradio interface
│   └── trace_panel.py                ← agent trace log display
│
├── evaluation/                       ← Member 4
│   ├── ragas_eval.py                 ← RAGAS metrics on 5 sample queries
│   ├── model_comparison.py           ← compare all 3 ML models
│   └── pipeline_benchmark.py         ← timing benchmark for each model call
│
└── outputs/                          ← generated at runtime, gitignored
    ├── chart.png
    └── report.md
```

### Branch rules:
- `main` — only Member 3 (orchestrator) pushes here directly
- `feature/rag` — Member 1's branch
- `feature/sql-chart` — Member 2's branch
- `feature/orchestrator` — Member 3's working branch (merges to main)
- `feature/ui-eval` — Member 4's branch
- Never push directly to main — always PR except Member 3

---

## 7. THE SHARED CONTRACT

**`state.py` is the most important file in this project.** Every agent reads from and writes to this object. No agent communicates with another agent directly. Everything flows through `AgentState`.

```python
# state.py — DO NOT MODIFY WITHOUT NOTIFYING ALL MEMBERS
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
```

### State rules — EVERY AGENT MUST FOLLOW THESE:
1. Every agent function signature is `def run(state: AgentState) -> AgentState`
2. Only write to the state keys your agent owns (see per-member sections)
3. Always append to `state["trace_log"]` — e.g. `state["trace_log"].append("RAG agent: retrieved 4 chunks from Apple 10-K")`
4. Never return `None` — always return the full state object
5. Wrap your agent in try/except — on failure, write an error message to your state key and return state

---

## 8. SYSTEM ARCHITECTURE

### Complete end-to-end flow for a single user query:

```
User types query in Gradio UI
        │
        ▼
app.py calls run_graph({"query": user_input, "image_data": img, "retry_count": 0, "trace_log": []})
        │
        ▼
LangGraph StateGraph (orchestrator/graph.py)
        │
        ▼
[NODE: planner]  ← orchestrator/planner.py
  - Reads query
  - Uses Groq LLM with ReAct prompt
  - Decides: which agents to call? (rag / sql+chart / sentiment / all)
  - Writes: state["plan"], state["agents_to_call"]
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
[NODE: rag_agent]                         [NODE: sql_agent]
  agents/rag_agent.py                       agents/sql_agent.py
  - Retrieves from ChromaDB                 - Queries SQLite with Groq SQL agent
  - ContextualCompression + ParentDoc       - Calls predict_fraud() on results
  - Writes: state["rag_result"]             - Writes: state["sql_result"]
            state["sources"]                          state["fraud_score"]
        │                                          │
        │                                   [NODE: chart_agent]
        │                                     agents/chart_agent.py
        │                                     - Generates matplotlib chart
        │                                     - Writes: state["chart_path"]
        │                                          │
        ├──────────────────────────────────────────┤
        ▼                                          ▼
[NODE: sentiment_agent]                   [NODE: forecast_agent]
  agents/sentiment_agent.py                orchestrator/graph.py (inline node)
  - Fetches RSS headlines                   - Calls predict_trend(ticker)
  - Calls predict_sentiment() [FinBERT]     - Writes: state["forecast"]
  - If image: calls GPT-4V extraction
  - Writes: state["sentiment_result"]
        │
        └──────────────┬───────────────────────────┘
                       ▼
              [NODE: evaluator]  ← orchestrator/evaluator.py
              - Reviews ALL state fields
              - Scores 0.0–1.0 on: completeness, consistency, confidence
              - Writes: state["eval_score"], state["eval_feedback"]
              - Decision:
                  if eval_score >= 0.7 → route to synthesizer
                  if eval_score < 0.7 AND retry_count < 2 → loop back to planner
                  if retry_count >= 2 → proceed with best available
                       │
              ┌────────┴────────┐
              ▼                 ▼
        [RETRY LOOP]    [NODE: synthesizer]  ← orchestrator/synthesizer.py
        retry_count++   - Combines all state fields
        back to planner - Writes final_report string
                        - Formats with sections: Filing Analysis,
                          Price Data, Sentiment, Forecast, Fraud Flag
                               │
                               ▼
                        Gradio UI receives result
                        - Displays: final_report text
                        - Displays: chart_path image
                        - Displays: trace_log panel
                        - Displays: fraud_score badge
                        - Displays: forecast badge
```

### LangGraph graph topology (graph.py implementation):

```python
from langgraph.graph import StateGraph, END
from state import AgentState

graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("planner", planner_node)
graph.add_node("rag", rag_node)
graph.add_node("sql", sql_node)
graph.add_node("chart", chart_node)
graph.add_node("sentiment", sentiment_node)
graph.add_node("forecast", forecast_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("synthesizer", synthesizer_node)

# Entry point
graph.set_entry_point("planner")

# Planner routes to agents
graph.add_edge("planner", "rag")
graph.add_edge("planner", "sql")
graph.add_edge("sql", "chart")
graph.add_edge("planner", "sentiment")
graph.add_edge("planner", "forecast")

# All agents converge at evaluator
graph.add_edge("rag", "evaluator")
graph.add_edge("chart", "evaluator")
graph.add_edge("sentiment", "evaluator")
graph.add_edge("forecast", "evaluator")

# Conditional routing from evaluator
graph.add_conditional_edges(
    "evaluator",
    route_after_eval,          # function defined in evaluator.py
    {
        "retry": "planner",    # loop back
        "proceed": "synthesizer"
    }
)

graph.add_edge("synthesizer", END)

app = graph.compile()

def run_graph(inputs: dict) -> AgentState:
    return app.invoke(inputs)
```

---

## 9. MEMBER 1 — RAG Agent + Fraud Detection Model

**Branch:** `feature/rag`
**Files owned:** `agents/rag_agent.py`, `retrieval/ingest.py`, `retrieval/vectorstore.py`, `retrieval/retriever.py`, `models/train_fraud.py`, `models/fraud_detector.py`
**State keys written:** `rag_result`, `sources`, `trace_log`

### Part A — RAG Agent

**Purpose:** Ingest SEC 10-K/10-Q PDF filings into ChromaDB and retrieve relevant passages when called.

**Embeddings — use HuggingFace (free, local, no API key):**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**Ingestion (retrieval/ingest.py):**
- Download 5–10 company 10-K filings from: `https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom&startdt=2024-01-01`
- Recommended companies: AAPL, MSFT, NVDA, TSLA, AMZN
- Use `PyPDFLoader` to load PDFs
- Chunk with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
- Store in ChromaDB at path `./chroma_db`

**Retriever setup (retrieval/retriever.py):**
```python
# Two-layer retrieval: ContextualCompression wraps ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ParentDocumentRetriever

# ParentDocumentRetriever: finds precise child chunk, returns parent context
# ContextualCompressionRetriever: strips irrelevant content before LLM sees it
```

**Required function signature:**
```python
from state import AgentState

def run(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        result = retriever.get_relevant_documents(query)
        state["rag_result"] = format_result(result)  # formatted string
        state["sources"] = [doc.metadata.get("source", "unknown") for doc in result]
        state["trace_log"] = state.get("trace_log", []) + [
            f"RAG agent: retrieved {len(result)} chunks from vector store"
        ]
    except Exception as e:
        state["rag_result"] = f"RAG retrieval failed: {str(e)}"
        state["sources"] = []
        state["trace_log"] = state.get("trace_log", []) + [f"RAG agent: ERROR — {str(e)}"]
    return state
```

**Standalone test:**
```bash
python agents/rag_agent.py
# Should print retrieved passages from Apple 10-K for query "What was Apple's revenue in 2024?"
```

### Part B — Fraud Detection Model (AIML Infra)

**Dataset:** Kaggle Credit Card Fraud — `creditcard.csv` (284,807 rows, 492 fraud)
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Models to train:** XGBoost AND LightGBM (both must be trained and compared)

**Training script must:**
1. Handle class imbalance with SMOTE
2. Log both models to local MLflow with metrics: AUC-ROC, F1, Precision, Recall
3. Save best model with `mlflow.sklearn.log_model()`

**Inference wrapper (models/fraud_detector.py):**
```python
def predict_fraud(transaction_features: dict) -> dict:
    """
    Input: dict with transaction features (Amount, V1-V28 for credit card dataset,
           or simplified: amount, merchant_category, hour_of_day, geo_distance)
    Output: {"fraud_probability": float, "is_fraud": bool, "risk_level": str}
    """
```

**How it connects:** The SQL agent (Member 2) will import and call `predict_fraud()` when it retrieves transaction records. Result goes into `state["fraud_score"]`.

---

## 10. MEMBER 2 — SQL + Chart Agent + FinBERT Model

**Branch:** `feature/sql-chart`
**Files owned:** `agents/sql_agent.py`, `agents/chart_agent.py`, `data/db_setup.py`, `data/fetch_prices.py`, `models/train_finbert.py`, `models/sentiment_model.py`
**State keys written:** `sql_result`, `chart_path`, `trace_log`

### Part A — SQL + Chart Agents

**Database setup (data/db_setup.py):**
```python
import yfinance as yf
import sqlite3
import pandas as pd

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# Table: prices (ticker, date, open, high, low, close, volume)
# Table: fundamentals (ticker, pe_ratio, market_cap, revenue, eps)

def setup_database():
    conn = sqlite3.connect("data/finsight.db")
    for ticker in TICKERS:
        df = yf.download(ticker, period="2y")
        df["ticker"] = ticker
        df.to_sql("prices", conn, if_exists="append", index=True)
    conn.close()
```

**SQL Agent (agents/sql_agent.py):**
```python
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

llm = ChatGroq(model="llama-3.3-70b-versatile")
db = SQLDatabase.from_uri("sqlite:///data/finsight.db")
sql_agent = create_sql_agent(llm=llm, db=db, verbose=True)
```

**Required signatures:**
```python
def run_sql(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        result = sql_agent.run(query)
        state["sql_result"] = result
        state["trace_log"] = state.get("trace_log", []) + [f"SQL agent: executed query, got result"]
        # Attempt fraud scoring if result contains transaction data
        try:
            from models.fraud_detector import predict_fraud
            # extract features from result and score
            # state["fraud_score"] = predict_fraud(features)
        except Exception:
            pass
    except Exception as e:
        state["sql_result"] = f"SQL query failed: {str(e)}"
    return state

def run_chart(state: AgentState) -> AgentState:
    data = state["sql_result"]
    try:
        # Use PythonREPLTool or write matplotlib code directly
        # Save chart to outputs/chart.png
        state["chart_path"] = "outputs/chart.png"
        state["trace_log"] = state.get("trace_log", []) + ["Chart agent: generated price chart"]
    except Exception as e:
        state["chart_path"] = None
        state["trace_log"] = state.get("trace_log", []) + [f"Chart agent: ERROR — {str(e)}"]
    return state
```

**Standalone test:**
```bash
python agents/sql_agent.py
# Query: "Show AAPL closing prices for last 6 months"
# Expected: tabular data string + chart.png saved
```

### Part B — FinBERT Fine-tuning (AIML Infra)

**Base model:** `ProsusAI/finbert` (HuggingFace, free, financial domain pre-trained)

**Dataset:** Financial PhraseBank
```python
from datasets import load_dataset
dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree")
# 4,840 sentences labeled: positive / neutral / negative
```

**Training (models/train_finbert.py):**
- Fine-tune for 3 epochs on Financial PhraseBank
- Log to local MLflow: accuracy, F1 macro
- Also test base FinBERT (no fine-tuning) as baseline comparison
- Save fine-tuned model: `model.save_pretrained("models/finbert-finetuned")`

**Inference wrapper (models/sentiment_model.py):**
```python
from transformers import pipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline(
            "text-classification",
            model="models/finbert-finetuned",
            tokenizer="ProsusAI/finbert"
        )
    return _pipeline

def predict_sentiment(text: str) -> dict:
    """
    Input: any financial text string (headline, sentence)
    Output: {"label": "positive"/"neutral"/"negative", "score": float, "summary": str}
    """
    result = get_pipeline()(text[:512])[0]
    return {
        "label": result["label"].lower(),
        "score": round(result["score"], 4),
        "summary": f"{result['label'].upper()} ({result['score']:.2f})"
    }
```

**How it connects:** Member 4's `sentiment_agent.py` imports and calls `predict_sentiment()` on each news headline. This replaces any GPT-4o calls and runs fully locally.

---

## 11. MEMBER 3 — ORCHESTRATOR (READ THIS WITH FULL ATTENTION)

**Branch:** `feature/orchestrator` (merges to `main`)
**Files owned:** `orchestrator/graph.py`, `orchestrator/planner.py`, `orchestrator/evaluator.py`, `orchestrator/synthesizer.py`, `agents/base_agent.py`, `models/feature_engineering.py`, `models/train_forecaster.py`, `models/forecaster.py`, `state.py`, `config.py`
**State keys written:** `plan`, `agents_to_call`, `eval_score`, `eval_feedback`, `final_report`, `forecast`, `trace_log`
**Role:** The brain. You own the entire LangGraph graph. Every other member's code plugs into your nodes. You are responsible for integration.

### Why this role is the most important:

You are the glue. When Members 1, 2, and 4 finish their features, they work in isolation. Your job is to wire them into a stateful graph that flows correctly, evaluates quality, self-corrects, and produces a final output. If your graph is correct, the whole system works. If your graph breaks, nothing works.

### Step 1 — config.py and base_agent.py (do this first)

**config.py:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")      # free at console.groq.com
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", None)   # optional — feedparser is fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)  # optional — for GPT-4V image feature

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set. Create .env file with your key.")
```

**agents/base_agent.py:**
```python
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Single LLM factory — all agents import from here."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=temperature
    )

def append_trace(state: dict, message: str) -> list:
    """Safe trace log append."""
    log = state.get("trace_log") or []
    log.append(message)
    return log
```

### Step 2 — orchestrator/planner.py

The planner is the first node in the graph. It reads the user query and decides which agents to invoke.

```python
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import get_llm, append_trace
from state import AgentState

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research planner. Given a user query, decide which
research agents are needed. Respond with a JSON object:
{{
  "plan": ["step1", "step2", ...],
  "agents_to_call": ["rag", "sql", "chart", "sentiment", "forecast"]
}}

Rules:
- Include "rag" if query involves company filings, annual reports, earnings, revenue, guidance
- Include "sql" if query involves stock prices, volume, historical data, trends
- Include "chart" if query involves trends, comparison, visual data (always pair with sql)
- Include "sentiment" if query involves news, market mood, analyst opinion
- Include "forecast" if query involves future outlook, prediction, price direction
- Include all agents for comprehensive analysis questions
- Respond ONLY with valid JSON. No explanation."""),
    ("human", "User query: {query}")
])

def planner_node(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = PLANNER_PROMPT | llm
    
    try:
        response = chain.invoke({"query": state["query"]})
        import json
        content = response.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        state["plan"] = parsed.get("plan", [])
        state["agents_to_call"] = parsed.get("agents_to_call", ["rag", "sql", "sentiment"])
    except Exception as e:
        # Fallback: call all agents
        state["plan"] = ["Retrieve filing data", "Query price data", "Check sentiment"]
        state["agents_to_call"] = ["rag", "sql", "chart", "sentiment", "forecast"]
    
    state["trace_log"] = append_trace(state, f"Planner: routing to {state['agents_to_call']}")
    return state
```

### Step 3 — orchestrator/evaluator.py

The evaluator is the most critical node. It reviews everything and decides whether to retry or proceed.

```python
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import get_llm, append_trace
from state import AgentState
import json

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research quality evaluator. Review the combined
research output and score it 0.0 to 1.0.

Score criteria:
- 1.0: All requested data is present, consistent, and well-sourced
- 0.7–0.9: Most data present, minor gaps
- 0.4–0.6: Significant data missing or inconsistent
- 0.0–0.3: Major failure, most data missing

Respond with JSON only:
{{
  "score": 0.0-1.0,
  "feedback": "what is weak or missing",
  "retry_agents": ["agent_name"] or []
}}"""),
    ("human", """Query: {query}

RAG Result: {rag_result}
SQL Result: {sql_result}
Sentiment Result: {sentiment_result}
Forecast: {forecast}
Fraud Score: {fraud_score}
Sources: {sources}

Evaluate the quality of this combined output.""")
])

def evaluator_node(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = EVALUATOR_PROMPT | llm
    
    try:
        response = chain.invoke({
            "query": state["query"],
            "rag_result": state.get("rag_result", "NOT RETRIEVED"),
            "sql_result": state.get("sql_result", "NOT RETRIEVED"),
            "sentiment_result": state.get("sentiment_result", "NOT RETRIEVED"),
            "forecast": str(state.get("forecast", "NOT COMPUTED")),
            "fraud_score": str(state.get("fraud_score", "NOT COMPUTED")),
            "sources": str(state.get("sources", []))
        })
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        state["eval_score"] = float(parsed.get("score", 0.5))
        state["eval_feedback"] = parsed.get("feedback", "")
    except Exception as e:
        state["eval_score"] = 0.6  # assume acceptable on parse failure
        state["eval_feedback"] = f"Evaluator parse error: {str(e)}"
    
    state["trace_log"] = append_trace(state,
        f"Evaluator: score={state['eval_score']:.2f} — {state['eval_feedback'][:80]}")
    return state

def route_after_eval(state: AgentState) -> str:
    """LangGraph routing function — called after evaluator node."""
    score = state.get("eval_score", 0.5)
    retry_count = state.get("retry_count", 0)
    
    if score >= 0.7 or retry_count >= 2:
        return "proceed"
    else:
        state["retry_count"] = retry_count + 1
        return "retry"
```

### Step 4 — orchestrator/synthesizer.py

The synthesizer combines all state fields into the final report string shown to the user.

```python
from agents.base_agent import get_llm, append_trace
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research report writer. Combine the provided
research data into a clear, structured report. Use these sections:

## 📄 Filing Analysis
[from rag_result — summarize key financial facts with source citations]

## 📈 Price & Market Data
[from sql_result — key price statistics, trends]

## 🔍 Fraud Risk Assessment
[from fraud_score — only include if fraud_score is present]

## 📰 News Sentiment
[from sentiment_result — overall sentiment and key themes]

## 🔮 5-Day Price Forecast
[from forecast — direction and confidence]

## ✅ Summary
[2–3 sentence overall conclusion]

Keep each section concise. Always cite sources from the sources list.
Respond with the formatted report only — no preamble."""),
    ("human", """Query: {query}

RAG: {rag_result}
Sources: {sources}
SQL: {sql_result}
Fraud: {fraud_score}
Sentiment: {sentiment_result}
Forecast: {forecast}""")
])

def synthesizer_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.2)
    chain = SYNTHESIZER_PROMPT | llm
    
    try:
        response = chain.invoke({
            "query": state["query"],
            "rag_result": state.get("rag_result", "No filing data retrieved."),
            "sources": ", ".join(state.get("sources", [])),
            "sql_result": state.get("sql_result", "No price data retrieved."),
            "fraud_score": str(state.get("fraud_score", "Not assessed")),
            "sentiment_result": state.get("sentiment_result", "No sentiment data."),
            "forecast": str(state.get("forecast", "No forecast available."))
        })
        state["final_report"] = response.content
    except Exception as e:
        state["final_report"] = f"Report synthesis failed: {str(e)}\n\nRaw data:\n{state}"
    
    state["trace_log"] = append_trace(state, "Synthesizer: final report generated")
    return state
```

### Step 5 — orchestrator/graph.py (the complete graph)

```python
from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator.planner import planner_node
from orchestrator.evaluator import evaluator_node, route_after_eval
from orchestrator.synthesizer import synthesizer_node

# Import agent run functions — these come from other members' branches
# Wire them defensively with fallback stubs until members push their code
try:
    from agents.rag_agent import run as rag_run
except ImportError:
    def rag_run(state): 
        state["rag_result"] = "RAG agent not yet implemented"
        return state

try:
    from agents.sql_agent import run_sql as sql_run
    from agents.chart_agent import run_chart as chart_run
except ImportError:
    def sql_run(state):
        state["sql_result"] = "SQL agent not yet implemented"
        return state
    def chart_run(state):
        state["chart_path"] = None
        return state

try:
    from agents.sentiment_agent import run as sentiment_run
except ImportError:
    def sentiment_run(state):
        state["sentiment_result"] = "Sentiment agent not yet implemented"
        return state

# Forecast node — Member 3 owns this
def forecast_node(state: AgentState) -> AgentState:
    try:
        from models.forecaster import predict_trend
        import re
        # Extract ticker from query
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META"]
        ticker = next((t for t in tickers if t in state["query"].upper()), "AAPL")
        state["forecast"] = predict_trend(ticker)
        state["trace_log"] = (state.get("trace_log") or []) + [
            f"Forecast agent: {ticker} → {state['forecast']['direction']} ({state['forecast']['confidence']})"
        ]
    except Exception as e:
        state["forecast"] = {"direction": "UNAVAILABLE", "confidence": 0.0, "error": str(e)}
        state["trace_log"] = (state.get("trace_log") or []) + [f"Forecast agent: ERROR — {str(e)}"]
    return state

# Build the graph
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("rag", rag_run)
    graph.add_node("sql", sql_run)
    graph.add_node("chart", chart_run)
    graph.add_node("sentiment", sentiment_run)
    graph.add_node("forecast", forecast_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("synthesizer", synthesizer_node)
    
    graph.set_entry_point("planner")
    
    # Planner → agents (parallel conceptually, sequential in LangGraph)
    graph.add_edge("planner", "rag")
    graph.add_edge("planner", "sql")
    graph.add_edge("sql", "chart")
    graph.add_edge("planner", "sentiment")
    graph.add_edge("planner", "forecast")
    
    # All agents → evaluator
    graph.add_edge("rag", "evaluator")
    graph.add_edge("chart", "evaluator")
    graph.add_edge("sentiment", "evaluator")
    graph.add_edge("forecast", "evaluator")
    
    # Evaluator conditional routing
    graph.add_conditional_edges(
        "evaluator",
        route_after_eval,
        {"retry": "planner", "proceed": "synthesizer"}
    )
    
    graph.add_edge("synthesizer", END)
    
    return graph

_compiled_graph = None

def run_graph(inputs: dict) -> AgentState:
    """Public entry point — called by Gradio UI."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    
    # Ensure required fields are initialized
    defaults = {
        "retry_count": 0,
        "trace_log": [],
        "plan": None,
        "agents_to_call": None,
        "rag_result": None,
        "sources": None,
        "sql_result": None,
        "chart_path": None,
        "sentiment_result": None,
        "image_data": None,
        "fraud_score": None,
        "forecast": None,
        "eval_score": None,
        "eval_feedback": None,
        "final_report": None
    }
    defaults.update(inputs)
    return _compiled_graph.invoke(defaults)
```

### Step 6 — Price Forecasting Model (AIML Infra — Member 3)

**Feature engineering (models/feature_engineering.py):**
```python
import pandas as pd
import yfinance as yf

def build_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    
    # Technical indicators
    df["returns_1d"] = df["Close"].pct_change(1)
    df["returns_5d"] = df["Close"].pct_change(5)
    df["returns_20d"] = df["Close"].pct_change(20)
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["vs_ma50_pct"] = (df["Close"] - df["ma_50"]) / df["ma_50"]
    df["volume_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    
    # Target: 5-day forward direction
    df["future_return"] = df["Close"].pct_change(5).shift(-5)
    df["target"] = df["future_return"].apply(
        lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0)
    )
    
    # Temporal
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    
    return df.dropna()
```

**Training (models/train_forecaster.py):**
```python
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_20d",
    "vs_ma50_pct", "volume_zscore", "rsi",
    "day_of_week", "month"
]

mlflow.set_experiment("stock-price-forecaster")

# Train on multiple tickers
# Log to MLflow
# Save model with mlflow.sklearn.log_model()
```

**Inference wrapper (models/forecaster.py):**
```python
def predict_trend(ticker: str, days_ahead: int = 5) -> dict:
    """
    Input: ticker string (e.g. "AAPL"), days_ahead int
    Output: {
        "ticker": "AAPL",
        "direction": "UP" / "DOWN" / "FLAT",
        "confidence": float 0–1,
        "days_ahead": int
    }
    Fails gracefully — returns direction "UNAVAILABLE" if model not trained yet.
    """
```

### Integration checklist — when merging other members' PRs:

When Member 1 pushes `feature/rag`:
- [ ] Verify `agents/rag_agent.py` has `def run(state: AgentState) -> AgentState`
- [ ] Verify it writes to `state["rag_result"]` and `state["sources"]`
- [ ] Verify it appends to `state["trace_log"]`
- [ ] Test: `from agents.rag_agent import run` — no import error
- [ ] Replace stub in `graph.py` with real import

When Member 2 pushes `feature/sql-chart`:
- [ ] Verify `sql_agent.py` has `def run_sql(state: AgentState) -> AgentState`
- [ ] Verify `chart_agent.py` has `def run_chart(state: AgentState) -> AgentState`
- [ ] Verify chart saves to `outputs/chart.png` (create `outputs/` folder if missing)
- [ ] Verify `models/sentiment_model.py` has `def predict_sentiment(text: str) -> dict`
- [ ] Test: `from models.sentiment_model import predict_sentiment` — no import error
- [ ] Replace stubs in graph.py

When Member 4 pushes `feature/ui-eval`:
- [ ] Verify `ui/app.py` imports `from orchestrator.graph import run_graph`
- [ ] Verify it calls `run_graph({"query": ..., "image_data": ..., "retry_count": 0, "trace_log": []})`
- [ ] Verify it renders `result["final_report"]`, `result["chart_path"]`, `result["trace_log"]`
- [ ] Verify `agents/sentiment_agent.py` has `def run(state: AgentState) -> AgentState`
- [ ] Verify it imports from `models.sentiment_model` not from OpenAI directly

---

## 12. MEMBER 4 — UI + Sentiment Agent + MLOps Report

**Branch:** `feature/ui-eval`
**Files owned:** `agents/sentiment_agent.py`, `ui/app.py`, `ui/trace_panel.py`, `evaluation/ragas_eval.py`, `evaluation/model_comparison.py`, `evaluation/pipeline_benchmark.py`
**State keys written:** `sentiment_result`, `trace_log`

### Part A — Sentiment Agent

**Uses local FinBERT — no GPT-4o, no API cost:**
```python
from state import AgentState
from models.sentiment_model import predict_sentiment
import feedparser

def fetch_headlines(query: str) -> list:
    """Free fallback: Yahoo Finance RSS. No API key needed."""
    # Extract company name or ticker from query
    ticker = extract_ticker(query)  # simple regex
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:10]]

def run(state: AgentState) -> AgentState:
    try:
        headlines = fetch_headlines(state["query"])
        if not headlines:
            headlines = ["No recent headlines found"]
        
        scores = [predict_sentiment(h) for h in headlines]
        labels = [s["label"] for s in scores]
        dominant = max(set(labels), key=labels.count)
        avg_score = sum(s["score"] for s in scores) / len(scores)
        
        state["sentiment_result"] = (
            f"{dominant.upper()} ({avg_score:.2f}): "
            f"Based on {len(headlines)} recent headlines. "
            f"Key themes: {'; '.join(headlines[:3])}"
        )
        state["trace_log"] = (state.get("trace_log") or []) + [
            f"Sentiment agent: analyzed {len(headlines)} headlines → {dominant.upper()}"
        ]
    except Exception as e:
        state["sentiment_result"] = f"Sentiment analysis failed: {str(e)}"
    return state
```

### Part B — Gradio UI (ui/app.py)

**Required structure:**
```python
import gradio as gr
from orchestrator.graph import run_graph

def run_query(user_query: str, image) -> tuple:
    """Returns: (report_text, chart_image, trace_log_text)"""
    import base64
    image_data = None
    if image is not None:
        # encode image to base64
        pass
    
    result = run_graph({
        "query": user_query,
        "image_data": image_data,
        "retry_count": 0,
        "trace_log": []
    })
    
    report = result.get("final_report", "No report generated.")
    chart = result.get("chart_path", None)
    trace = "\n".join(result.get("trace_log", []))
    fraud = result.get("fraud_score")
    forecast = result.get("forecast")
    
    # Add fraud + forecast to report if present
    if fraud:
        report += f"\n\n**Fraud Risk:** {fraud['risk_level']} ({fraud['fraud_probability']:.2%})"
    if forecast:
        report += f"\n\n**5-Day Forecast:** {forecast['direction']} (confidence: {forecast['confidence']:.2%})"
    
    return report, chart, trace

with gr.Blocks(title="FinSight AI") as demo:
    gr.Markdown("# FinSight AI — Autonomous Financial Research")
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Your financial question", lines=3)
            image_input = gr.Image(label="Upload chart/image (optional)", type="pil")
            submit_btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            report_output = gr.Markdown(label="Research Report")
            chart_output = gr.Image(label="Price Chart")
            trace_output = gr.Textbox(label="Agent Trace Log", lines=10, interactive=False)
    
    submit_btn.click(
        fn=run_query,
        inputs=[query_input, image_input],
        outputs=[report_output, chart_output, trace_output]
    )

if __name__ == "__main__":
    demo.launch()
```

### Part B — MLOps Report (AIML Infra)

Member 4 owns the AIML Infrastructure Engineering project report. Collect results from all 3 models and write:

1. **Model comparison table** — XGBoost vs LightGBM fraud detection metrics
2. **FinBERT comparison** — base vs fine-tuned accuracy on Financial PhraseBank test set
3. **Forecaster accuracy** — directional accuracy on 3 test tickers
4. **MLflow experiment screenshots** — run `mlflow ui`, take screenshots of all 3 experiments
5. **Pipeline benchmark** — timing of each model call
6. **RAGAS scores** — faithfulness, answer_relevance, context_precision for 5 queries

---

## 13. INTEGRATION RULES

### Rules that apply to EVERYONE — no exceptions:

**Rule 1: The state contract is sacred.**
Never add a new state field without updating `state.py` and telling everyone. Never write to a state key you don't own (see ownership table below).

| State Key | Owner |
|---|---|
| `query` | Read-only for everyone |
| `plan`, `agents_to_call` | Member 3 (planner) |
| `rag_result`, `sources` | Member 1 |
| `sql_result`, `chart_path` | Member 2 |
| `fraud_score` | Member 1 (model) called by Member 2 (SQL agent) |
| `sentiment_result` | Member 4 |
| `forecast` | Member 3 (forecast node) |
| `eval_score`, `eval_feedback` | Member 3 (evaluator) |
| `final_report` | Member 3 (synthesizer) |
| `trace_log` | Everyone appends — never overwrite, always append |
| `retry_count` | Member 3 (evaluator route function) |

**Rule 2: Every agent must fail gracefully.**
Every agent function must be wrapped in try/except. On failure, write an error message to your state key and return state. Never raise an exception that kills the graph.

**Rule 3: The function signature is non-negotiable.**
```python
def run(state: AgentState) -> AgentState:    # for single-action agents
def run_sql(state: AgentState) -> AgentState: # for sql agent
def run_chart(state: AgentState) -> AgentState: # for chart agent
```

**Rule 4: No circular imports.**
Agents never import from other agents. All shared utilities go in `agents/base_agent.py`. ML models go in `models/`. Agents import from `models/`.

**Rule 5: Outputs folder must exist.**
Any agent writing a file must ensure `outputs/` exists:
```python
import os
os.makedirs("outputs", exist_ok=True)
```

**Rule 6: No hard-coded API keys.**
All keys come from environment variables via `config.py`. Use `.env` file locally. Never commit `.env`.

**Rule 7: Use Groq for all LLM calls.**
```python
from agents.base_agent import get_llm
llm = get_llm()  # always — never instantiate LLM directly in agent files
```

**Rule 8: Test standalone before pushing.**
Every agent must be testable with `if __name__ == "__main__":` block that runs a sample query without the full graph.

---

## 14. THE THREE ML MODELS

These three models serve the AIML Infrastructure Engineering course. They run locally — no Azure, no cloud required.

### Model 1 — Fraud Detector (Member 1)
- **Type:** Binary classification (stacking ensemble)
- **Models:** XGBoost (primary) + LightGBM (comparison)
- **Dataset:** Kaggle Credit Card Fraud (creditcard.csv)
- **Imbalance handling:** SMOTE
- **Tracking:** Local MLflow experiment named `fraud-detection`
- **Interface:** `predict_fraud(features: dict) -> dict`
- **Called by:** SQL agent after retrieving transaction data
- **State key:** `fraud_score`

### Model 2 — FinBERT Sentiment Classifier (Member 2)
- **Type:** 3-class text classification (positive / neutral / negative)
- **Base model:** `ProsusAI/finbert` (HuggingFace)
- **Dataset:** Financial PhraseBank (`takala/financial_phrasebank`)
- **Fine-tuning:** 3 epochs, batch size 16, on CPU ~15–20 min
- **Tracking:** Local MLflow experiment named `finbert-sentiment`
- **Interface:** `predict_sentiment(text: str) -> dict`
- **Called by:** Sentiment agent (Member 4)
- **State key:** `sentiment_result`

### Model 3 — Stock Price Trend Forecaster (Member 3)
- **Type:** 3-class classification (UP / FLAT / DOWN)
- **Model:** LightGBM multi-class classifier
- **Dataset:** Built from yfinance data (same source as SQL agent)
- **Features:** RSI, MACD, rolling returns, volume z-score, MA deviation
- **Target:** 5-day forward direction (>2% = UP, <-2% = DOWN, else FLAT)
- **Tracking:** Local MLflow experiment named `stock-forecaster`
- **Interface:** `predict_trend(ticker: str) -> dict`
- **Called by:** Forecast node in LangGraph graph (Member 3)
- **State key:** `forecast`

### Viewing all MLflow experiments:
```bash
mlflow ui
# Opens at http://localhost:5000
# All 3 experiments visible side by side
# Take screenshots for AIML Infra course report
```

---

## 15. DEMO FLOW

### Demo Query 1 — Earnings Analysis (~45 sec)
```
Input: "Analyse Apple's Q4 2024 financial performance from their annual filing 
        and show me the stock price trend for the past 6 months"

Expected output:
- Filing Analysis section citing Apple 10-K with revenue figures
- Price chart (AAPL 6-month line chart saved as chart.png)
- Sentiment section based on recent Apple headlines
- 5-day forecast for AAPL
- Agent trace showing: planner → rag → sql → chart → sentiment → forecast → evaluator → synthesizer
```

### Demo Query 2 — Sentiment Analysis (~30 sec)
```
Input: "What is the current news sentiment around Nvidia and does it align 
        with their recent financial performance?"

Expected output:
- Filing analysis from NVDA 10-K (revenue, guidance)
- Sentiment score from FinBERT on recent NVDA headlines
- Comparison paragraph in final report
```

### Demo Query 3 — Multimodal (~20 sec, most dramatic)
```
Upload: Screenshot of a competitor revenue chart
Input: "Extract the data from this chart and compare it with what our 
        knowledge base says about this company"

Expected output:
- Image processed (GPT-4V if available, else skip gracefully)
- RAG retrieval comparing with filing data
- Comparison section in final report
```

### Pre-demo checklist:
- [ ] App running locally: `python ui/app.py`
- [ ] ChromaDB populated: `python retrieval/ingest.py`
- [ ] SQLite populated: `python data/db_setup.py`
- [ ] All 3 ML models trained and saved
- [ ] `.env` file has GROQ_API_KEY
- [ ] Pre-run queries 1 and 2 once to warm up — Groq has ~1s cold start
- [ ] `outputs/` folder exists and is writable

---

## 16. WHAT GOOD OUTPUT LOOKS LIKE

A correct final report for "Analyse Apple Q4 2024" should contain:

```markdown
## 📄 Filing Analysis
Apple Inc. reported total revenue of $94.9 billion in Q4 FY2024, a 6% year-over-year 
increase. Services revenue reached a record $24.2 billion... [Source: AAPL_10K_2024.pdf, p.34]

## 📈 Price & Market Data
AAPL traded between $165.00 and $198.50 over the past 6 months. The 20-day moving 
average shows an upward trend since October...

## 🔍 Fraud Risk Assessment
Transaction risk: LOW (fraud_probability: 0.03)

## 📰 News Sentiment
POSITIVE (0.78): 14 of 18 recent headlines are positive. Key themes: iPhone 16 demand, 
Vision Pro adoption, Services growth.

## 🔮 5-Day Price Forecast
Direction: UP | Confidence: 0.74 | Horizon: 5 days

## ✅ Summary
Apple demonstrates strong Q4 performance driven by Services and hardware. Sentiment is 
bullish and price momentum is positive. Risk indicators are low.
```

A correct trace log should look like:
```
Planner: routing to ['rag', 'sql', 'chart', 'sentiment', 'forecast']
RAG agent: retrieved 4 chunks from AAPL 10-K filing
SQL agent: executed query, got AAPL price data for 6 months
Chart agent: generated price chart → outputs/chart.png
Sentiment agent: analyzed 12 headlines → POSITIVE
Forecast agent: AAPL → UP (0.74)
Evaluator: score=0.85 — All key data present and consistent
Synthesizer: final report generated
```

---

## 17. COMMON FAILURE POINTS AND FIXES

| Problem | Likely Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'run' from 'agents.rag_agent'` | Member 1 hasn't pushed yet or named function differently | Use stub in graph.py until PR is merged |
| `KeyError: 'rag_result'` | State not initialized with all keys | Ensure `run_graph()` sets all defaults before invoke |
| `ChromaDB: collection not found` | `ingest.py` not run yet | Run `python retrieval/ingest.py` first |
| `sqlite3.OperationalError: no such table: prices` | `db_setup.py` not run yet | Run `python data/db_setup.py` first |
| Graph hangs / never reaches synthesizer | Conditional edge in evaluator returns unexpected value | Check `route_after_eval()` returns exactly `"retry"` or `"proceed"` |
| `Groq 429: rate limit` | Too many requests in quick succession | Add `time.sleep(1)` between agent calls, or use different Groq key |
| `sentiment_model.py not found` | Member 2 hasn't trained FinBERT yet | sentiment_agent.py must fall back gracefully — return placeholder |
| Chart not displaying in Gradio | `chart_path` is relative, Gradio needs absolute | Use `os.path.abspath("outputs/chart.png")` |
| LangGraph: multiple edges to same node | Graph topology error | Each node can only have one incoming trigger in sequential LangGraph |
| `mlflow: No active run` | Training script not using context manager | Always use `with mlflow.start_run():` |

---

## 18. REVIEWING PRS — CHECKLIST

When any member opens a PR to merge into their branch, or when Member 3 merges a feature branch to main, use this checklist:

### For every PR:
- [ ] Does the function signature match exactly what's in this document?
- [ ] Does it write only to state keys it owns?
- [ ] Does it append to `trace_log` (not overwrite)?
- [ ] Does it have try/except error handling?
- [ ] Does it have a standalone test (`if __name__ == "__main__":`)?
- [ ] No hardcoded API keys?
- [ ] No `import openai` without checking `OPENAI_API_KEY` first?
- [ ] `requirements.txt` updated if new packages added?

### For Member 1 PR (RAG):
- [ ] Uses `HuggingFaceEmbeddings`, not `OpenAIEmbeddings`
- [ ] ChromaDB path is `./chroma_db` (relative, gitignored)
- [ ] `format_result()` returns a readable string, not a list object
- [ ] `predict_fraud()` works on a sample dict input

### For Member 2 PR (SQL + Chart):
- [ ] `run_sql()` and `run_chart()` are separate functions
- [ ] Chart saves to `outputs/chart.png` with `os.makedirs("outputs", exist_ok=True)`
- [ ] `predict_sentiment()` handles text longer than 512 tokens (truncate to 512)
- [ ] FinBERT model saved to `models/finbert-finetuned/` (tracked in gitignore or LFS)

### For Member 4 PR (UI + Eval):
- [ ] `app.py` calls `run_graph()` correctly with all required initial keys
- [ ] Renders `final_report`, `chart_path`, `trace_log` separately
- [ ] Sentiment agent does NOT import openai or call GPT-4o
- [ ] RAGAS eval uses Groq as judge LLM, not OpenAI

---

*This document was last updated: April 2026*
*Maintained by: Member 3 (Orchestrator)*
*Repo: https://github.com/MohammedAswathM/finsight-ai*
