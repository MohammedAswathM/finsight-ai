# FinSight AI

Multi-agent financial research assistant. A LangGraph orchestrator coordinates six specialist agents (RAG over SEC 10-K filings, SQL over price data, sentiment over news headlines, fraud detection, price-trend forecasting, chart generation) and synthesizes their outputs into a single cited report.

Built as a college team project across two courses — **AIML Agentic** (the agent stack) and **AIML Infrastructure Engineering** (the ML models + MLOps).

## Stack

100% free / open-source. No paid services, no cloud costs.

- **LLM inference:** Groq free tier (`llama-3.3-70b-versatile`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (CPU)
- **Vector store:** ChromaDB (local persistent)
- **Relational store:** SQLite
- **Agent framework:** LangGraph + LangChain 0.2
- **MLOps:** MLflow file store
- **UI:** Chainlit
- **Evaluation:** RAGAS (LLM-as-judge)
- **ML models:** scikit-learn, LightGBM, XGBoost, FinBERT (transformers)

## Architecture

```
                   ┌─────────────┐
                   │   planner   │  Groq LLM decides which agents to involve
                   └──────┬──────┘
        ┌────────┬────────┼────────┬────────┐
        ▼        ▼        ▼        ▼        ▼
      ┌───┐   ┌───┐   ┌────────┐ ┌─────┐ ┌──────────┐
      │RAG│   │SQL│   │sentiment│ │fraud│ │forecaster│
      └─┬─┘   └─┬─┘   └────┬───┘ └──┬──┘ └────┬─────┘
        │       │ ┌──chart─┘        │         │
        │       │ │                 │         │
        ▼       ▼ ▼                 ▼         ▼
                ┌───────────┐
                │ evaluator │  scores answer; routes back to planner if too weak
                └─────┬─────┘
                ┌─────▼─────┐
                │synthesizer│  composes the final markdown report
                └───────────┘
```

State flows through a typed `AgentState` (see `state.py`); `trace_log` uses an additive reducer so parallel branches don't clobber each other.

## Repository layout

```
finsight-ai/
├── orchestrator/          ← graph, planner, evaluator, synthesizer (Member 3)
├── agents/                ← one file per specialist agent
├── retrieval/             ← ChromaDB vectorstore + ParentDocumentRetriever (Member 1)
├── data/                  ← SQLite setup + price fetchers (Member 2)
├── models/                ← trained ML wrappers + training scripts
│   ├── train_fraud.py        XGBoost + LightGBM (Member 1)
│   ├── train_finbert.py      FinBERT fine-tune (Member 2)
│   ├── train_forecaster.py   Price-direction LightGBM (Member 3)
│   └── train_volatility.py   Volatility-regime classifier (Member 3)
├── ui/                    ← Chainlit app (Member 4)
├── evaluation/            ← RAGAS, MLflow comparison, latency benchmark
├── tests/                 ← deliverable smoke tests
├── state.py               ← AgentState TypedDict (shared contract)
├── config.py              ← env loader (single source of truth for keys)
└── FINSIGHT_AI_BRAIN.md   ← internal design document
```

## First-time setup

Requires **Python 3.12** on Windows / macOS / Linux. The project is verified on Windows 11. (Python 3.13 has a SciPy build issue; 3.11 has a `pandas-ta` issue — 3.12 is the sweet spot.)

```bash
# 1. Clone and enter
git clone <repo-url>
cd finsight-ai

# 2. Create venv
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate # macOS/Linux

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env          # Windows
# or: cp .env.example .env      # macOS/Linux
```

Edit `.env` and set:

- `GROQ_API_KEY` — required, get one free at https://console.groq.com/keys
- `MLFLOW_TRACKING_URI=file:./mlruns` — already set in `.env.example`; do not leave blank on Windows (newer MLflow defaults to a SQLite URI that breaks on paths with spaces)
- `OPENAI_API_KEY` — optional, only used if you want the multimodal image-upload demo

## Build the data + model artifacts (one-time)

```bash
# SQLite price database (~30 s)
python -m data.db_setup

# SEC 10-K embeddings into ChromaDB (~5–10 min, ~90 MB download)
python -m retrieval.ingest

# Train the three ML models (each writes an MLflow run + a pickle)
python -m models.train_fraud           # needs data/creditcard.csv from Kaggle first
python -m models.train_forecaster
python -m models.train_volatility
python -m models.train_finbert         # ~10–25 min on CPU
```

After this, `chroma_db/`, `docstore/`, `outputs/finsight.db`, `mlruns/`, and `models/*.pkl|.joblib|finbert-finetuned/` all exist. **None of these are committed to the repo** (they're gitignored — too large or contain user paths).

## Running the system

```bash
# Chainlit web UI — the demo surface
chainlit run ui/app.py
```

Opens at http://localhost:8000. Type a financial question; the orchestrator runs, you see a structured report, an inline chart, and a per-node trace.

```bash
# Headless graph run (prints the same content to stdout)
python -m orchestrator.graph

# MLflow tracking UI (training metrics + artifacts)
mlflow ui --backend-store-uri file:./mlruns
```

## Testing

```bash
# Smoke-test all three model wrappers
python -m evaluation.pipeline_benchmark

# Member-2 deliverable suite (DB + SQL + Chart + Sentiment)
python -m tests.test_member2

# RAG agent standalone
python run_rag.py

# RAGAS evaluation across 5 canonical queries
python -m evaluation.ragas_eval

# MLflow run summary across all experiments
python -m evaluation.model_comparison
```

## Sample queries

Best-tested queries for the demo:

- `Analyze Microsoft 10-K cloud revenue and recent sentiment` — full agent fan-out
- `Show AAPL closing price trend and explain the outlook` — SQL + chart + forecaster
- `Summarize Amazon's risk factors from filings` — RAG-heavy, returns real legal-risk content
- `Compare Microsoft and Nvidia Q4 results` — multi-ticker, often triggers the reflection retry loop

Avoid Apple-10K-specific questions; the AAPL filing returned 404 from SEC EDGAR during ingest, so RAG has 4 of 5 companies indexed.

## Team

| Member | Branch | Owns |
|---|---|---|
| 1 | `feature/rag` | RAG agent, retrieval/, fraud detection model |
| 2 | `feature/sql-chart` | SQL + chart agents, data/, FinBERT fine-tune |
| 3 | `feature/orchestrator` → main | orchestrator/, forecaster, state.py, config.py |
| 4 | `feature/ui-eval` | Chainlit UI, sentiment agent, RAGAS eval |

The orchestrator is the only branch that merges to `main`. See `FINSIGHT_AI_BRAIN.md` for the full integration contract.

## Known limitations

- **Apple 10-K**: SEC EDGAR returned 404 for the linked accession during ingest. RAG correctly reports "no relevant info" for Apple-filing-specific queries rather than hallucinating.
- **Forecaster default ticker**: when the planner can't extract a ticker from the query, the forecaster defaults to AAPL. Reports for non-AAPL questions will note "data not available for X, but here's AAPL's forecast."
- **Free Groq daily quota**: ~100K tokens/day, ~14K requests/day. A heavy session of UI queries + RAGAS can exhaust this; quota resets at 00:00 UTC.
- **Trace duplicates**: the LangGraph join node fires once per super-step that has new inputs, so the trace can show two `Synthesizer: done` lines. The final state is correct.

## License

Course project — not for production use.
