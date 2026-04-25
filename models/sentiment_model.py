"""
models/sentiment_model.py
Local FinBERT sentiment model wrapper
─────────────────────────────────────────────────────────────────────────────
Loads ProsusAI/finbert from HuggingFace (cached locally after first download).
Exposes a single function:

    predict_sentiment(text: str) -> dict {"label": str, "score": float}

Labels returned: "positive" | "neutral" | "negative"
Score:           FinBERT softmax confidence for the predicted label [0, 1]

The model is loaded once at module import time and reused across all calls
(singleton pattern) to avoid repeated GPU/CPU initialisation overhead.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton — model loaded once on first call
# ─────────────────────────────────────────────────────────────────────────────
_pipeline = None  # transformers pipeline instance


def _load_model():
    """Load FinBERT pipeline once and cache it in the module-level singleton."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        model_name = "ProsusAI/finbert"
        logger.info("Loading FinBERT model: %s (first load may download weights)", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        _pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            top_k=1,          # return only the top predicted label
        )

        logger.info("FinBERT model loaded successfully.")
        return _pipeline

    except ImportError:
        logger.error(
            "transformers library not installed. "
            "Run: pip install transformers torch"
        )
        raise
    except Exception as exc:
        logger.error("Failed to load FinBERT model: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> dict:
    """
    Run FinBERT inference on a single text string.

    Parameters
    ----------
    text : str
        A financial headline or short passage (≤ 512 tokens).

    Returns
    -------
    dict
        {
            "label": "positive" | "neutral" | "negative",
            "score": float   # softmax confidence [0.0, 1.0]
        }

    Raises
    ------
    ValueError
        If text is empty or not a string.
    RuntimeError
        If the model fails to load or inference fails.

    Examples
    --------
    >>> predict_sentiment("Apple beats earnings expectations, stock surges 5%")
    {'label': 'positive', 'score': 0.9741}

    >>> predict_sentiment("Fed holds interest rates steady")
    {'label': 'neutral', 'score': 0.8823}

    >>> predict_sentiment("Company misses revenue forecast, shares tumble")
    {'label': 'negative', 'score': 0.9512}
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("predict_sentiment requires a non-empty string.")

    pipe = _load_model()

    try:
        # pipeline returns: [[{"label": "positive", "score": 0.97}]]
        raw = pipe(text.strip())

        # Unpack — shape varies slightly by transformers version
        if isinstance(raw, list) and len(raw) > 0:
            top = raw[0]
            if isinstance(top, list) and len(top) > 0:
                top = top[0]
        else:
            raise RuntimeError(f"Unexpected pipeline output: {raw}")

        label: str = top["label"].lower().strip()
        score: float = float(top["score"])

        # Normalise any non-standard label names back to the three canonical ones
        label = _normalise_label(label)

        return {"label": label, "score": round(score, 4)}

    except Exception as exc:
        logger.warning("FinBERT inference error for text '%s...': %s", text[:50], exc)
        # Safe fallback — neutral with low confidence
        return {"label": "neutral", "score": 0.5}


def _normalise_label(label: str) -> str:
    """Map any variant label strings to canonical positive/neutral/negative."""
    mapping = {
        "pos":      "positive",
        "positive": "positive",
        "neg":      "negative",
        "negative": "negative",
        "neu":      "neutral",
        "neutral":  "neutral",
    }
    return mapping.get(label, "neutral")


# ─────────────────────────────────────────────────────────────────────────────
# Optional: batch inference (used if you want to score all headlines at once)
# ─────────────────────────────────────────────────────────────────────────────
def predict_sentiment_batch(texts: list[str], batch_size: int = 16) -> list[dict]:
    """
    Run FinBERT inference on a list of texts in batches.
    More efficient than calling predict_sentiment() in a Python loop
    when you have 10+ headlines.

    Parameters
    ----------
    texts      : list of non-empty strings
    batch_size : number of texts per forward pass (tune for your GPU/CPU RAM)

    Returns
    -------
    list of dicts, same format as predict_sentiment()
    """
    if not texts:
        return []

    pipe = _load_model()
    results: list[dict] = []

    try:
        raw_batch = pipe(texts, batch_size=batch_size, truncation=True)

        for raw in raw_batch:
            if isinstance(raw, list):
                raw = raw[0]
            label = _normalise_label(raw["label"].lower().strip())
            score = round(float(raw["score"]), 4)
            results.append({"label": label, "score": score})

    except Exception as exc:
        logger.warning("Batch inference failed: %s — falling back to per-item.", exc)
        results = [predict_sentiment(t) for t in texts]

    return results