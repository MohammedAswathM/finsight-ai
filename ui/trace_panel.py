"""
ui/trace_panel.py
FinSight AI — Trace Panel Formatter
─────────────────────────────────────────────────────────────────────────────
Converts the raw trace_log list from AgentState into a clean, human-readable
string suitable for display in a Gradio Textbox or Markdown component.

Each trace entry is expected to have the shape:
    {
        "agent":  str,
        "action": str,
        "detail": str,          # optional
        "time":   float,        # unix timestamp, optional
    }
"""

from __future__ import annotations

import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Emoji badges — makes the trace visually scannable at a glance
# ─────────────────────────────────────────────────────────────────────────────
_ACTION_ICONS: dict[str, str] = {
    "start":                   "🚀",
    "complete":                "✅",
    "error":                   "❌",
    "warning":                 "⚠️",
    "ticker_extracted":        "🔍",
    "headlines_fetched":       "📰",
    "deduplication_complete":  "🧹",
    "sentiment_scored":        "🧠",
    "aggregation_complete":    "📊",
    "summary_generated":       "📝",
    "sql_query":               "🗄️",
    "rag_retrieval":           "📚",
    "chart_generated":         "📈",
    "routing":                 "🔀",
    "orchestrator":            "🎯",
}

_AGENT_COLORS: dict[str, str] = {
    "SentimentAgent":    "💹",
    "SQLAgent":          "🗃️",
    "RAGAgent":          "📖",
    "ChartAgent":        "📉",
    "Orchestrator":      "🎯",
}


def _icon_for_action(action: str) -> str:
    action_lower = action.lower()
    for key, icon in _ACTION_ICONS.items():
        if key in action_lower:
            return icon
    return "•"


def _badge_for_agent(agent: str) -> str:
    for key, badge in _AGENT_COLORS.items():
        if key.lower() in agent.lower():
            return badge
    return "🤖"


def _format_timestamp(ts: Optional[float]) -> str:
    """Convert unix float to HH:MM:SS.mmm string."""
    if ts is None:
        return ""
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def format_trace(trace_log: list[dict]) -> str:
    """
    Convert a list of trace entries into a clean, readable string.

    Parameters
    ----------
    trace_log : list[dict]
        List of trace dicts from state["trace_log"].

    Returns
    -------
    str
        Formatted multi-line string ready to display in Gradio.
    """
    if not trace_log:
        return "No trace data available."

    lines: list[str] = []
    lines.append("─" * 60)
    lines.append("  FinSight AI — Agent Execution Trace")
    lines.append("─" * 60)

    prev_agent: Optional[str] = None

    for i, entry in enumerate(trace_log):
        agent  = entry.get("agent", "Unknown")
        action = entry.get("action", "")
        detail = entry.get("detail", "")
        ts     = entry.get("time")

        # Print agent header when it changes
        if agent != prev_agent:
            if prev_agent is not None:
                lines.append("")
            badge = _badge_for_agent(agent)
            lines.append(f"{badge}  [{agent}]")
            prev_agent = agent

        icon     = _icon_for_action(action)
        ts_str   = _format_timestamp(ts)
        ts_part  = f"  [{ts_str}]" if ts_str else ""

        # Build the step line
        action_display = action.replace("_", " ").title()
        step = f"  {i+1:>2}. {icon} {action_display}{ts_part}"
        lines.append(step)

        # Indent detail on next line if present
        if detail:
            lines.append(f"      └─ {detail}")

    lines.append("")
    lines.append("─" * 60)

    # Compute total elapsed if start/end timestamps exist
    try:
        times = [e["time"] for e in trace_log if "time" in e]
        if len(times) >= 2:
            elapsed_ms = round((times[-1] - times[0]) * 1000, 1)
            lines.append(f"  ⏱  Total elapsed: {elapsed_ms} ms")
            lines.append("─" * 60)
    except Exception:
        pass

    return "\n".join(lines)


def format_trace_markdown(trace_log: list[dict]) -> str:
    """
    Same as format_trace() but uses Markdown formatting for use
    in Gradio gr.Markdown components.
    """
    if not trace_log:
        return "_No trace data available._"

    parts: list[str] = []
    parts.append("### 🔎 Agent Execution Trace\n")
    parts.append("---")

    prev_agent: Optional[str] = None

    for i, entry in enumerate(trace_log):
        agent  = entry.get("agent", "Unknown")
        action = entry.get("action", "")
        detail = entry.get("detail", "")
        ts     = entry.get("time")

        if agent != prev_agent:
            badge = _badge_for_agent(agent)
            parts.append(f"\n**{badge} {agent}**")
            prev_agent = agent

        icon           = _icon_for_action(action)
        ts_str         = _format_timestamp(ts)
        ts_part        = f" `{ts_str}`" if ts_str else ""
        action_display = action.replace("_", " ").title()

        parts.append(f"{i+1}. {icon} **{action_display}**{ts_part}")
        if detail:
            parts.append(f"   > {detail}")

    # Total elapsed
    try:
        times = [e["time"] for e in trace_log if "time" in e]
        if len(times) >= 2:
            elapsed_ms = round((times[-1] - times[0]) * 1000, 1)
            parts.append(f"\n---\n⏱ **Total elapsed:** `{elapsed_ms} ms`")
    except Exception:
        pass

    return "\n".join(parts)


def summarise_trace(trace_log: list[dict]) -> dict:
    """
    Return a structured summary of the trace for programmatic use.

    Returns
    -------
    dict with keys: agents_involved, total_steps, errors, elapsed_ms
    """
    if not trace_log:
        return {"agents_involved": [], "total_steps": 0, "errors": 0, "elapsed_ms": 0}

    agents  = list(dict.fromkeys(e.get("agent", "") for e in trace_log))
    errors  = sum(1 for e in trace_log if "error" in e.get("action", "").lower())
    times   = [e["time"] for e in trace_log if "time" in e]
    elapsed = round((times[-1] - times[0]) * 1000, 1) if len(times) >= 2 else 0

    return {
        "agents_involved": agents,
        "total_steps":     len(trace_log),
        "errors":          errors,
        "elapsed_ms":      elapsed,
    }