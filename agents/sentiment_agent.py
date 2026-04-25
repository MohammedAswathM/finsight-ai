"""
agents/sentiment_agent.py
FinSight AI — Production Sentiment Agent (v2)
─────────────────────────────────────────────────────────────────────────────
Pipeline:
  state["query"]
    → extract_ticker()
    → fetch_headlines()       RSS via feedparser
    → deduplicate_headlines()
    → analyze_sentiment()     local FinBERT
    → aggregate_scores()      weighted score + distribution + confidence
    → detect_trend()
    → generate_summary()
    → state["sentiment_result"] + state["trace_log"]
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import feedparser

from models.sentiment_model import predict_sentiment
from state import AgentState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | sentiment_agent | %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
_FALLBACK_RSS_URLS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI",
]
_MAX_HEADLINES = 20
_DEDUP_THRESHOLD = 0.72

# Well-known ticker aliases so users can type natural queries
_TICKER_ALIASES: dict[str, str] = {
    "apple":     "AAPL",
    "microsoft": "MSFT",
    "google":    "GOOGL",
    "alphabet":  "GOOGL",
    "amazon":    "AMZN",
    "tesla":     "TSLA",
    "nvidia":    "NVDA",
    "meta":      "META",
    "facebook":  "META",
    "netflix":   "NFLX",
    "salesforce":"CRM",
    "sp500":     "^GSPC",
    "s&p":       "^GSPC",
    "dow":       "^DJI",
    "nasdaq":    "^IXIC",
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal data container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScoredHeadline:
    headline: str
    label: str          # positive | neutral | negative
    score: float        # FinBERT softmax confidence [0,1]
    signed_score: float = field(init=False)

    def __post_init__(self) -> None:
        sign = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(self.label, 0.0)
        self.signed_score = sign * self.score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Ticker extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_ticker(query: str) -> str:
    """
    Intelligently extract a stock ticker from a free-text query.

    Priority order:
      1. Explicit ticker in parentheses  → "Apple (AAPL) outlook"
      2. ALL-CAPS word 2-5 chars          → "What is TSLA doing?"
      3. Known company alias              → "nvidia sentiment"
      4. Fallback                         → "AAPL"
    """
    query_clean = query.strip()

    # 1. Parenthesised ticker
    m = re.search(r"\(([A-Z\^]{1,6})\)", query_clean)
    if m:
        return m.group(1)

    # 2. Standalone ALL-CAPS token (2–6 chars, may start with ^)
    tokens = re.findall(r"\b\^?[A-Z]{2,6}\b", query_clean)
    if tokens:
        return tokens[0]

    # 3. Company alias (case-insensitive)
    lower_query = query_clean.lower()
    for alias, ticker in _TICKER_ALIASES.items():
        if alias in lower_query:
            return ticker

    logger.warning("Could not extract ticker from query '%s' — defaulting to AAPL", query_clean)
    return "AAPL"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RSS news fetch
# ─────────────────────────────────────────────────────────────────────────────
def fetch_headlines(ticker: str) -> list[str]:
    """Fetch headlines from Yahoo Finance RSS via feedparser."""
    headlines: list[str] = []

    urls = [_RSS_URL.format(ticker=ticker)] + (
        _FALLBACK_RSS_URLS if ticker.startswith("^") else []
    )

    for url in urls:
        try:
            feed = feedparser.parse(url)
            items = [entry.title.strip() for entry in feed.entries if entry.get("title")]
            logger.info("Fetched %d headlines from %s", len(items), url)
            headlines.extend(items)
            if len(headlines) >= _MAX_HEADLINES:
                break
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", url, exc)

    # If still sparse, try fallback market feeds
    if len(headlines) < 5:
        for url in _FALLBACK_RSS_URLS:
            try:
                feed = feedparser.parse(url)
                items = [entry.title.strip() for entry in feed.entries if entry.get("title")]
                headlines.extend(items)
                logger.info("Supplemented with %d fallback headlines", len(items))
            except Exception as exc:
                logger.warning("Fallback RSS failed: %s", exc)

    return [h for h in headlines if len(h) > 10]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_headlines(headlines: list[str], threshold: float = _DEDUP_THRESHOLD) -> list[str]:
    """Remove near-duplicate headlines using Jaccard token similarity."""
    unique: list[str] = []
    for candidate in headlines:
        if not any(_jaccard(candidate, kept) >= threshold for kept in unique):
            unique.append(candidate)
    return unique[:_MAX_HEADLINES]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FinBERT inference
# ─────────────────────────────────────────────────────────────────────────────
def analyze_sentiment(headlines: list[str]) -> list[ScoredHeadline]:
    """Score each headline with local FinBERT; degrade gracefully on errors."""
    results: list[ScoredHeadline] = []
    for h in headlines:
        try:
            pred = predict_sentiment(h)
            if isinstance(pred, dict):
                label, score = pred.get("label", "neutral"), float(pred.get("score", 0.5))
            elif isinstance(pred, (list, tuple)) and len(pred) >= 2:
                label, score = str(pred[0]), float(pred[1])
            else:
                label, score = "neutral", 0.5

            label = label.lower().strip()
            score = max(0.0, min(1.0, score))
            results.append(ScoredHeadline(headline=h, label=label, score=score))
        except Exception as exc:
            logger.warning("Inference failed for '%s': %s", h[:50], exc)
            results.append(ScoredHeadline(headline=h, label="neutral", score=0.5))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_scores(scored: list[ScoredHeadline]) -> dict:
    """
    Compute:
      • weighted_score   — confidence-weighted signed mean  (−1 … +1)
      • confidence       — high / medium / low
      • distribution     — {positive, neutral, negative} as fractions
    """
    if not scored:
        return {
            "weighted_score": 0.0,
            "confidence": "low",
            "distribution": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        }

    n = len(scored)
    total_weight = sum(s.score for s in scored)

    weighted_score = (
        sum(s.signed_score * s.score for s in scored) / total_weight
        if total_weight > 0 else 0.0
    )

    counts: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for s in scored:
        counts[s.label] = counts.get(s.label, 0) + 1

    distribution = {k: round(v / n, 3) for k, v in counts.items()}
    mean_conf = total_weight / n

    if mean_conf >= 0.75:
        confidence = "high"
    elif mean_conf >= 0.55:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "weighted_score": round(weighted_score, 4),
        "confidence": confidence,
        "distribution": distribution,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Trend detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_trend(agg: dict) -> str:
    """Classify market trend from aggregated scores."""
    score = agg["weighted_score"]
    dist  = agg["distribution"]

    if score >= 0.25 and dist["positive"] >= 0.45:
        return "bullish"
    if score <= -0.25 and dist["negative"] >= 0.45:
        return "bearish"
    if abs(score) < 0.15 and dist["neutral"] >= 0.50:
        return "neutral"
    return "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Natural-language summary
# ─────────────────────────────────────────────────────────────────────────────
def _extract_topic(headline: str, max_words: int = 6) -> str:
    cleaned = re.sub(r"\([A-Z]{1,5}\)", "", headline)
    cleaned = re.sub(r"^[A-Za-z]+\s*[-–]\s*", "", cleaned)
    return " ".join(cleaned.split()[:max_words]).lower().rstrip(".,;:")


def _top_by_label(scored: list[ScoredHeadline], label: str, k: int = 2) -> list[str]:
    items = sorted([s for s in scored if s.label == label], key=lambda s: s.score, reverse=True)
    return [s.headline for s in items[:k]]


def generate_summary(ticker: str, agg: dict, trend: str, scored: list[ScoredHeadline]) -> str:
    """Build a concise analyst-style NL summary — no external LLM required."""
    score = agg["weighted_score"]
    confidence = agg["confidence"]
    dist = agg["distribution"]

    abs_score = abs(score)
    intensity = "strongly" if abs_score >= 0.60 else "moderately" if abs_score >= 0.35 else "slightly"
    trend_phrase = {
        "bullish": f"{intensity} bullish",
        "bearish": f"{intensity} bearish",
        "mixed":   "mixed",
        "neutral": "broadly neutral",
    }.get(trend, "uncertain")

    parts = [
        f"Market sentiment for {ticker.upper()} is {trend_phrase} "
        f"(confidence: {confidence}, score: {score:+.2f})."
    ]

    if dist["positive"] >= 0.30:
        pos = _top_by_label(scored, "positive")
        if pos:
            parts.append(
                f"Positive signals ({int(dist['positive']*100)}% of coverage) "
                f"are driven by themes around {_extract_topic(pos[0])}."
            )

    if dist["negative"] >= 0.20:
        neg = _top_by_label(scored, "negative")
        if neg:
            parts.append(
                f"Bearish signals ({int(dist['negative']*100)}%) centre on "
                f"{_extract_topic(neg[0])}."
            )

    if dist["neutral"] >= 0.35:
        parts.append(
            f"{int(dist['neutral']*100)}% of headlines remain factual or wait-and-see, "
            f"signalling market uncertainty."
        )

    if trend == "mixed":
        parts.append(
            "Conflicting signals suggest monitoring upcoming catalysts before taking directional positions."
        )

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run(state: AgentState) -> AgentState:
    """
    FinSight Sentiment Agent entry point.

    Reads:   state["query"]
    Writes:  state["sentiment_result"], state["trace_log"]
    """
    trace: list[dict] = state.get("trace_log", [])

    def log_trace(agent: str, action: str, detail: str = "") -> None:
        entry = {
            "agent":  agent,
            "action": action,
            "detail": detail,
            "time":   round(time.time(), 3),
        }
        trace.append(entry)
        logger.info("[TRACE] %s | %s | %s", agent, action, detail)

    agent_name = "SentimentAgent"
    t_start = time.perf_counter()
    log_trace(agent_name, "start", f"query='{state.get('query', '')}'")

    # ── Extract ticker ───────────────────────────────────────────────────────
    query: str = state.get("query", "").strip()
    if not query:
        log_trace(agent_name, "error", "Empty query")
        state["sentiment_result"] = _empty_result("UNKNOWN", "No query provided.")
        state["trace_log"] = trace
        return state

    ticker = extract_ticker(query)
    log_trace(agent_name, "ticker_extracted", f"ticker={ticker}")

    # ── Fetch headlines ──────────────────────────────────────────────────────
    raw_headlines = fetch_headlines(ticker)
    log_trace(agent_name, "headlines_fetched", f"raw_count={len(raw_headlines)}")

    if not raw_headlines:
        log_trace(agent_name, "warning", "No headlines found")
        state["sentiment_result"] = _empty_result(ticker, "No headlines found.")
        state["trace_log"] = trace
        return state

    # ── Deduplicate ──────────────────────────────────────────────────────────
    headlines = deduplicate_headlines(raw_headlines)
    log_trace(agent_name, "deduplication_complete", f"unique_count={len(headlines)}")

    # ── FinBERT inference ────────────────────────────────────────────────────
    scored = analyze_sentiment(headlines)
    log_trace(agent_name, "sentiment_scored", f"scored={len(scored)} headlines")

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = aggregate_scores(scored)
    trend = detect_trend(agg)
    log_trace(agent_name, "aggregation_complete",
              f"score={agg['weighted_score']:+.3f} trend={trend} confidence={agg['confidence']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = generate_summary(ticker, agg, trend, scored)
    log_trace(agent_name, "summary_generated")

    # ── Top headlines (highest FinBERT confidence) ───────────────────────────
    top_headlines = [
        s.headline for s in sorted(scored, key=lambda s: s.score, reverse=True)[:5]
    ]

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    log_trace(agent_name, "complete", f"elapsed={elapsed_ms}ms")

    state["sentiment_result"] = {
        "score":          agg["weighted_score"],
        "trend":          trend,
        "confidence":     agg["confidence"],
        "distribution":   agg["distribution"],
        "summary":        summary,
        "top_headlines":  top_headlines,
        "headline_count": len(scored),
        "ticker":         ticker,
        "elapsed_ms":     elapsed_ms,
    }
    state["trace_log"] = trace
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _empty_result(ticker: str, reason: str) -> dict:
    return {
        "score":          0.0,
        "trend":          "neutral",
        "confidence":     "low",
        "distribution":   {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        "summary":        f"Sentiment analysis unavailable for {ticker}. {reason}",
        "top_headlines":  [],
        "headline_count": 0,
        "ticker":         ticker,
        "elapsed_ms":     0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Direct execution:  python agents/sentiment_agent.py "What is TSLA sentiment?"
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is Apple stock sentiment?"
    final_state = run({"query": query_arg, "trace_log": []})
    s = final_state["sentiment_result"]
    print(f"\n{'='*60}")
    print(f"  Ticker    : {s['ticker']}")
    print(f"  Trend     : {s['trend']}")
    print(f"  Score     : {s['score']:+.4f}")
    print(f"  Confidence: {s['confidence']}")
    print(f"  Headlines : {s['headline_count']}")
    print(f"\n  Distribution:")
    for k, v in s["distribution"].items():
        print(f"    {k:<10}: {v*100:.1f}%")
    print(f"\n  Summary:\n    {s['summary']}")
    print(f"\n  Top Headlines:")
    for i, h in enumerate(s["top_headlines"], 1):
        print(f"    {i}. {h}")
    print(f"{'='*60}\n")