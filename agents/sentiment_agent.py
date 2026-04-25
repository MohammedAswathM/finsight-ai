"""
agents/sentiment_agent.py
Production-grade Financial Sentiment Agent
─────────────────────────────────────────────────────────────────────────────
Multi-source news ingestion → FinBERT scoring → trend detection →
confidence aggregation → natural-language summary.

Constraints honoured
  • Uses local FinBERT:  from models.sentiment_model import predict_sentiment
  • No OpenAI / paid APIs
  • Entry-point signature: run(state: AgentState) -> AgentState
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from models.sentiment_model import predict_sentiment  # local FinBERT wrapper
from state import AgentState

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_UA = "Mozilla/5.0 (FinancialSentimentBot/2.0)"
_REQUEST_TIMEOUT = 8          # seconds per feed
_MAX_HEADLINES = 20           # hard cap to keep latency bounded
_DEDUP_SIM_THRESHOLD = 0.75   # Jaccard threshold for near-duplicate removal

# RSS feed templates — {ticker} is substituted at runtime.
_RSS_TEMPLATES: list[str] = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://finance.yahoo.com/rss/headline?s={ticker}",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
]

# Fallback: broader market / ticker-agnostic RSS sources used when
# ticker-specific feeds return fewer than 5 items.
_FALLBACK_RSS: list[str] = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC",   # S&P 500
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI",    # Dow Jones
]

# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HeadlineScore:
    headline: str
    label: str          # "positive" | "neutral" | "negative"
    score: float        # raw FinBERT confidence [0, 1]
    signed_score: float = field(init=False)

    def __post_init__(self) -> None:
        sign = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(
            self.label.lower(), 0.0
        )
        self.signed_score = sign * self.score


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEWS FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_rss(url: str) -> list[str]:
    """Fetch and parse an RSS feed; return list of item titles."""
    try:
        req = Request(url, headers={"User-Agent": _UA})
        with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        titles: list[str] = []
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
        return titles
    except (URLError, ET.ParseError, Exception) as exc:
        logger.warning("RSS fetch failed for %s: %s", url, exc)
        return []


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two headlines."""
    set_a = set(re.findall(r"\w+", a.lower()))
    set_b = set(re.findall(r"\w+", b.lower()))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate(headlines: list[str], threshold: float = _DEDUP_SIM_THRESHOLD) -> list[str]:
    """Remove near-duplicate headlines using pairwise Jaccard similarity."""
    unique: list[str] = []
    for candidate in headlines:
        is_dup = any(
            _jaccard_similarity(candidate, kept) >= threshold for kept in unique
        )
        if not is_dup:
            unique.append(candidate)
    return unique


def fetch_headlines(ticker: str) -> list[str]:
    """
    Fetch 10–15 deduplicated headlines for *ticker* from multiple RSS sources.

    Strategy
    ────────
    1. Try all ticker-specific feed templates in parallel (sequential here to
       avoid extra dependencies; fast enough given 8-s timeout).
    2. If fewer than 5 results, supplement with fallback market feeds.
    3. Deduplicate, then return up to _MAX_HEADLINES items.
    """
    raw: list[str] = []

    # Primary: ticker-specific
    for template in _RSS_TEMPLATES:
        url = template.format(ticker=ticker.upper())
        items = _fetch_rss(url)
        raw.extend(items)
        logger.info("Fetched %d headlines from %s", len(items), url)
        if len(raw) >= _MAX_HEADLINES:
            break

    # Secondary: fallback if sparse
    if len(raw) < 5:
        logger.info("Sparse ticker results; supplementing with fallback feeds.")
        for url in _FALLBACK_RSS:
            items = _fetch_rss(url)
            raw.extend(items)

    # Filter: keep only non-empty strings
    raw = [h for h in raw if h and len(h) > 10]

    # Deduplicate
    deduped = _deduplicate(raw)

    logger.info(
        "Headlines after dedup: %d (from %d raw)", len(deduped), len(raw)
    )
    return deduped[:_MAX_HEADLINES]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_sentiment(headlines: list[str]) -> list[HeadlineScore]:
    """
    Run each headline through the local FinBERT wrapper and return
    a list of HeadlineScore objects.

    predict_sentiment(text) is assumed to return either:
      • a dict  {"label": str, "score": float}
      • a tuple (label: str, score: float)
    """
    results: list[HeadlineScore] = []
    for headline in headlines:
        try:
            prediction = predict_sentiment(headline)

            # Normalise output format
            if isinstance(prediction, dict):
                label: str = prediction.get("label", "neutral")
                score: float = float(prediction.get("score", 0.5))
            elif isinstance(prediction, (list, tuple)) and len(prediction) >= 2:
                label, score = str(prediction[0]), float(prediction[1])
            else:
                logger.warning("Unexpected predict_sentiment output: %s", prediction)
                label, score = "neutral", 0.5

            label = label.lower().strip()
            score = max(0.0, min(1.0, score))  # clamp to [0,1]
            results.append(HeadlineScore(headline=headline, label=label, score=score))

        except Exception as exc:
            logger.warning("Sentiment inference failed for '%s': %s", headline[:60], exc)
            # Degrade gracefully: treat as neutral
            results.append(HeadlineScore(headline=headline, label="neutral", score=0.5))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCORE AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_scores(scored: list[HeadlineScore]) -> dict:
    """
    Aggregate a list of HeadlineScore objects into:
      • weighted_score   – weighted mean signed score (−1 … +1)
      • confidence       – "high" | "medium" | "low"
      • distribution     – {"positive": %, "neutral": %, "negative": %}
      • trend            – "bullish" | "bearish" | "mixed" | "neutral"

    Weighting: each headline is weighted by its FinBERT confidence score so
    high-confidence predictions dominate the aggregate.
    """
    if not scored:
        return {
            "weighted_score": 0.0,
            "confidence": "low",
            "distribution": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
            "trend": "neutral",
        }

    n = len(scored)
    total_weight = sum(s.score for s in scored)

    # Weighted mean signed score
    if total_weight > 0:
        weighted_score = sum(s.signed_score * s.score for s in scored) / total_weight
    else:
        weighted_score = 0.0

    # Distribution
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in scored:
        counts[s.label] = counts.get(s.label, 0) + 1

    distribution = {k: round(v / n, 3) for k, v in counts.items()}

    # Confidence: based on mean FinBERT score magnitude
    mean_confidence = total_weight / n
    if mean_confidence >= 0.75:
        confidence = "high"
    elif mean_confidence >= 0.55:
        confidence = "medium"
    else:
        confidence = "low"

    # Trend detection
    pos_frac = distribution["positive"]
    neg_frac = distribution["negative"]
    neu_frac = distribution["neutral"]

    if weighted_score >= 0.25 and pos_frac >= 0.45:
        trend = "bullish"
    elif weighted_score <= -0.25 and neg_frac >= 0.45:
        trend = "bearish"
    elif abs(weighted_score) < 0.15 and neu_frac >= 0.5:
        trend = "neutral"
    else:
        trend = "mixed"

    return {
        "weighted_score": round(weighted_score, 4),
        "confidence": confidence,
        "distribution": distribution,
        "trend": trend,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. NATURAL-LANGUAGE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def _top_headlines_by_label(
    scored: list[HeadlineScore], label: str, k: int = 2
) -> list[str]:
    """Return the k highest-confidence headlines for a given sentiment label."""
    filtered = [s for s in scored if s.label == label]
    filtered.sort(key=lambda s: s.score, reverse=True)
    return [s.headline for s in filtered[:k]]


def generate_summary(
    ticker: str,
    agg: dict,
    scored: list[HeadlineScore],
) -> str:
    """
    Produce a concise, analyst-style sentiment summary using rule-based
    template filling — no external LLM required.

    Example output:
      "Market sentiment for AAPL is moderately bullish, driven by strong
       earnings reports and analyst upgrades. Neutral macro commentary
       tempers the optimism slightly. A minority of bearish signals relate
       to supply chain concerns."
    """
    trend = agg["trend"]
    confidence = agg["confidence"]
    score = agg["weighted_score"]
    dist = agg["distribution"]

    # ── Intensity adverb ──
    abs_score = abs(score)
    if abs_score >= 0.60:
        intensity = "strongly"
    elif abs_score >= 0.35:
        intensity = "moderately"
    else:
        intensity = "slightly"

    # ── Trend phrase ──
    trend_phrase = {
        "bullish": f"{intensity} bullish",
        "bearish": f"{intensity} bearish",
        "mixed":   "mixed",
        "neutral": "broadly neutral",
    }.get(trend, "uncertain")

    # ── Opening sentence ──
    summary_parts = [
        f"Market sentiment for {ticker.upper()} is {trend_phrase} "
        f"(confidence: {confidence}, score: {score:+.2f})."
    ]

    # ── Positive drivers ──
    if dist["positive"] >= 0.30:
        pos_headlines = _top_headlines_by_label(scored, "positive")
        if pos_headlines:
            # Extract a brief keyword phrase from the highest-conf headline
            driver = _extract_topic(pos_headlines[0])
            pct = int(dist["positive"] * 100)
            summary_parts.append(
                f"Positive signals account for {pct}% of coverage, "
                f"with themes around {driver}."
            )

    # ── Negative concerns ──
    if dist["negative"] >= 0.20:
        neg_headlines = _top_headlines_by_label(scored, "negative")
        if neg_headlines:
            concern = _extract_topic(neg_headlines[0])
            pct = int(dist["negative"] * 100)
            summary_parts.append(
                f"Bearish signals ({pct}%) centre on {concern}."
            )

    # ── Neutral commentary ──
    if dist["neutral"] >= 0.30:
        pct = int(dist["neutral"] * 100)
        summary_parts.append(
            f"A significant portion ({pct}%) of headlines remain factual "
            f"or wait-and-see, indicating market uncertainty."
        )

    # ── Closing risk note ──
    if trend == "mixed":
        summary_parts.append(
            "Overall, the conflicting signals suggest investors should "
            "monitor upcoming catalysts closely before taking directional bets."
        )
    elif trend == "bullish" and confidence == "low":
        summary_parts.append(
            "The bullish lean lacks strong conviction; treat with caution "
            "until confirmed by higher-volume news flow."
        )

    return " ".join(summary_parts)


def _extract_topic(headline: str, max_words: int = 6) -> str:
    """
    Heuristic: strip ticker symbols and common filler, then return
    the first *max_words* meaningful words as a lower-case phrase.
    """
    # Remove parenthetical ticker references like (AAPL)
    cleaned = re.sub(r"\([A-Z]{1,5}\)", "", headline)
    # Remove leading date/source patterns like "Reuters -"
    cleaned = re.sub(r"^[A-Za-z]+\s*[-–]\s*", "", cleaned)
    words = cleaned.split()[:max_words]
    return " ".join(words).lower().rstrip(".,;:")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN AGENT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run(state: AgentState) -> AgentState:
    """
    Financial Sentiment Agent — production entry point.

    Reads
    ─────
    state["ticker"]  (str, required)

    Writes
    ──────
    state["sentiment_result"] = {
        "score":        float,      # weighted signed score (−1 … +1)
        "trend":        str,        # "bullish" | "bearish" | "mixed" | "neutral"
        "confidence":   str,        # "high" | "medium" | "low"
        "distribution": {
            "positive": float,
            "neutral":  float,
            "negative": float,
        },
        "summary":      str,
        "top_headlines": list[str],
        "headline_count": int,
    }
    """
    ticker: str = state.get("ticker", "").strip()
    if not ticker:
        logger.error("AgentState missing 'ticker' key — aborting sentiment run.")
        state["sentiment_result"] = _empty_result("UNKNOWN", "No ticker provided.")
        return state

    logger.info("▶ Sentiment agent starting for ticker: %s", ticker)
    t0 = time.time()

    # ── Step 1: Fetch headlines ──────────────────────────────────────────────
    headlines = fetch_headlines(ticker)
    if not headlines:
        logger.warning("No headlines found for %s.", ticker)
        state["sentiment_result"] = _empty_result(
            ticker, f"No news headlines could be retrieved for {ticker}."
        )
        return state

    # ── Step 2: Analyse sentiment ────────────────────────────────────────────
    scored: list[HeadlineScore] = analyze_sentiment(headlines)

    # ── Step 3: Aggregate ────────────────────────────────────────────────────
    agg = aggregate_scores(scored)

    # ── Step 4: Generate summary ─────────────────────────────────────────────
    summary = generate_summary(ticker, agg, scored)

    # ── Step 5: Select top headlines for output ──────────────────────────────
    # Return the 5 highest-confidence headlines (mixed sentiment labels)
    top_scored = sorted(scored, key=lambda s: s.score, reverse=True)[:5]
    top_headlines = [s.headline for s in top_scored]

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "✔ Sentiment analysis complete in %.2fs — trend=%s confidence=%s score=%+.3f",
        elapsed, agg["trend"], agg["confidence"], agg["weighted_score"],
    )

    state["sentiment_result"] = {
        "score":          agg["weighted_score"],
        "trend":          agg["trend"],
        "confidence":     agg["confidence"],
        "distribution":   agg["distribution"],
        "summary":        summary,
        "top_headlines":  top_headlines,
        "headline_count": len(scored),
    }
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _empty_result(ticker: str, reason: str) -> dict:
    """Return a safe, zeroed-out result when the agent cannot proceed."""
    return {
        "score":          0.0,
        "trend":          "neutral",
        "confidence":     "low",
        "distribution":   {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        "summary":        f"Sentiment analysis unavailable for {ticker}. {reason}",
        "top_headlines":  [],
        "headline_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run directly:  python agents/sentiment_agent.py
#                python agents/sentiment_agent.py TSLA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Accept ticker from command line, default to AAPL
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

    print(f"\nRunning sentiment analysis for: {ticker_arg}\n")

    result_state = run({"ticker": ticker_arg})
    s = result_state["sentiment_result"]

    print("=" * 60)
    print(f"  Ticker    : {ticker_arg}")
    print(f"  Trend     : {s['trend']}")
    print(f"  Score     : {s['score']:+.4f}")
    print(f"  Confidence: {s['confidence']}")
    print(f"  Headlines : {s['headline_count']}")
    print()
    print("  Distribution:")
    for label, pct in s["distribution"].items():
        print(f"    {label:<10}: {pct * 100:.1f}%")
    print()
    print("  Summary:")
    print(f"    {s['summary']}")
    print()
    print("  Top Headlines:")
    for i, h in enumerate(s["top_headlines"], 1):
        print(f"    {i}. {h}")
    print("=" * 60)