"""
FinSight AI — Chainlit UI
ui/app.py

Architecture:
  - run_pipeline() is the SINGLE integration point.
  - RIGHT NOW: calls sentiment_agent.run()
  - LATER: swap ONE line → `from orchestrator.graph import run_graph`
    and replace the body of run_pipeline() with: return await run_graph(query, image_path)

The UI layer never changes.
"""

import asyncio
import traceback
from pathlib import Path
from typing import Optional

import chainlit as cl

# ─────────────────────────────────────────────
# Pipeline integration point
# ─────────────────────────────────────────────
# CURRENT: only sentiment agent
from agents.sentiment_agent import run as run_sentiment

# FUTURE (one-line swap):
# from orchestrator.graph import run_graph


async def run_pipeline(query: str, image_path: Optional[str] = None) -> dict:
    """
    Central pipeline dispatcher.
    CURRENT: only sentiment agent
    """

    # ✅ correct call
    raw = await asyncio.to_thread(run_sentiment, {
        "query": query,
        "trace_log": []
    })

    # ✅ normalize properly
    return {
        "sentiment": {
            "label": raw.get("sentiment_result", {}).get("trend", "neutral"),
            "score": raw.get("sentiment_result", {}).get("score", 0),
            "summary": raw.get("sentiment_result", {}).get("summary", "")
        },
        "headlines": [
            {"title": h, "source": "Yahoo Finance", "url": ""}
            for h in raw.get("sentiment_result", {}).get("top_headlines", [])
        ],
        "rag": None,
        "sql": None,
        "forecast": None,
        "trace": [
            {"agent": "SentimentAgent", "status": "success", "detail": t}
            for t in raw.get("trace_log", [])
        ]
    }


def _normalize(raw: dict) -> dict:
    """
    Coerce whatever the agent returns into the standard result schema.
    Keeps UI code free of agent-specific quirks.
    """
    return {
        "sentiment":  raw.get("sentiment",  {}),
        "headlines":  raw.get("headlines",  []),
        "rag":        raw.get("rag",        None),
        "sql":        raw.get("sql",        None),
        "forecast":   raw.get("forecast",   None),
        "trace":      raw.get("trace",      []),
    }


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

SENTIMENT_EMOJI = {
    "positive": "🟢",
    "neutral":  "🟡",
    "negative": "🔴",
}

SCORE_BAR_WIDTH = 20  # chars for ASCII progress bar


def _score_bar(score: float) -> str:
    """Render a simple ASCII confidence bar."""
    filled = round(score * SCORE_BAR_WIDTH)
    return f"[{'█' * filled}{'░' * (SCORE_BAR_WIDTH - filled)}] {score:.0%}"


def _sentiment_block(sentiment: dict) -> str:
    """Format the sentiment section as clean Markdown."""
    if not sentiment:
        return "_No sentiment data available._"

    label = sentiment.get("label", "unknown").lower()
    score = float(sentiment.get("score", 0))
    summary = sentiment.get("summary", "")
    emoji = SENTIMENT_EMOJI.get(label, "⚪")

    lines = [
        f"## 📊 Sentiment Analysis",
        "",
        f"**Verdict:** {emoji} `{label.upper()}`",
        f"**Confidence:** {_score_bar(score)}",
    ]
    if summary:
        lines += ["", "---", "", f"> {summary}"]
    return "\n".join(lines)


def _headlines_block(headlines: list) -> str:
    """Format top headlines as a Markdown list."""
    if not headlines:
        return ""

    lines = ["## 🗞️ Key Headlines", ""]
    for i, h in enumerate(headlines[:5], 1):
        title  = h.get("title",  "Untitled")
        source = h.get("source", "Unknown")
        url    = h.get("url",    "")
        if url:
            lines.append(f"{i}. [{title}]({url}) — *{source}*")
        else:
            lines.append(f"{i}. **{title}** — *{source}*")
    return "\n".join(lines)


def _supplemental_block(result: dict) -> str:
    """
    Placeholder sections for future agents.
    Shown only when data is actually present — never breaks if missing.
    """
    parts = []

    if result.get("rag"):
        parts.append(f"## 🔍 RAG Insights\n\n{result['rag']}")

    if result.get("sql"):
        parts.append(f"## 🗄️ SQL Findings\n\n{result['sql']}")

    if result.get("forecast"):
        parts.append(f"## 📈 Forecast\n\n{result['forecast']}")

    return "\n\n---\n\n".join(parts)


async def _render_trace(trace: list) -> None:
    """Render execution trace as a collapsible Chainlit Step."""
    if not trace:
        return

    async with cl.Step(name="🔍 Execution Trace", type="tool") as step:
        lines = []
        for entry in trace:
            agent  = entry.get("agent",  "unknown")
            status = entry.get("status", "done")
            detail = entry.get("detail", "")
            icon   = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
            lines.append(f"{icon} **{agent}** — {detail}")
        step.output = "\n".join(lines)


async def _render_result(result: dict) -> None:
    """Compose and send all UI sections for a successful pipeline run."""

    # 1. Sentiment block (always shown)
    sentiment_md = _sentiment_block(result["sentiment"])
    await cl.Message(content=sentiment_md).send()

    # 2. Headlines (shown only if present)
    headlines_md = _headlines_block(result["headlines"])
    if headlines_md:
        await cl.Message(content=headlines_md).send()

    # 3. Supplemental agent outputs (future-proofed)
    supplemental = _supplemental_block(result)
    if supplemental:
        await cl.Message(content=supplemental).send()

    # 4. Execution trace (collapsible)
    await _render_trace(result.get("trace", []))


# ─────────────────────────────────────────────
# Chainlit lifecycle hooks
# ─────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Greet the user and set session metadata."""
    cl.user_session.set("ready", True)

    await cl.Message(
        content=(
            "## 👋 Welcome to **FinSight AI**\n\n"
            "I analyze financial sentiment, surface key headlines, and (soon) "
            "deliver RAG-powered insights, SQL analytics, and price forecasts.\n\n"
            "**Ask me anything about a stock, sector, or financial event.**\n"
            "_Optionally attach an image (chart, screenshot) for richer analysis._"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main message handler.

    Flow:
      1. Extract query + optional image path
      2. Show a 'thinking' indicator via cl.Step
      3. Call run_pipeline()
      4. Render structured results
      5. Handle and display errors gracefully
    """
    query = message.content.strip()
    if not query:
        await cl.Message(content="⚠️ Please enter a query before submitting.").send()
        return

    # Extract image path from attachments (first image wins)
    image_path: Optional[str] = None
    for element in message.elements:
        if hasattr(element, "path") and element.path:
            suffix = Path(element.path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                image_path = element.path
                break

    # ── Thinking indicator ──────────────────────────────────────────────────
    async with cl.Step(name="⚙️ FinSight is thinking…", type="run") as thinking:
        thinking.input = query
        try:
            result = await run_pipeline(query, image_path)
            thinking.output = "Pipeline completed successfully."
        except Exception as exc:
            thinking.output = f"Pipeline error: {exc}"
            error_detail = traceback.format_exc()
            await cl.Message(
                content=(
                    f"❌ **An error occurred while processing your query.**\n\n"
                    f"```\n{error_detail}\n```\n\n"
                    "Please try again or contact support."
                )
            ).send()
            return

    # ── Render structured output ────────────────────────────────────────────
    await _render_result(result)