"""Environment loader. Every module imports keys from here, never os.getenv directly."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
NEWSAPI_KEY: str | None = os.getenv("NEWSAPI_KEY") or None

# Leave blank to use MLflow's local file store (./mlruns) — recommended default.
# Set to http://127.0.0.1:5000 only if you're running `mlflow server` separately.
MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI") or None
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def require_groq() -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
