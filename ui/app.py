"""
ui/app.py
FinSight AI — Gradio Frontend
─────────────────────────────────────────────────────────────────────────────
Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  FinSight AI  —  Financial Intelligence Platform        │
  ├─────────────────────────────────────────────────────────┤
  │  [Query Textbox]          [Image Upload (optional)]     │
  │                    [Analyse]                            │
  ├────────────────────┬────────────────────────────────────┤
  │  📊 Report         │  🔎 Agent Trace                    │
  │  (Markdown)        │  (Textbox / monospace)             │
  ├────────────────────┴────────────────────────────────────┤
  │  📈 Chart (if generated)                                │
  └─────────────────────────────────────────────────────────┘

Run:
    python ui/app.py
    python ui/app.py --share      # public Gradio link
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — ensure project root is importable
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.sentiment_agent import run as run_sentiment   
from ui.trace_panel import format_trace_markdown  # trace formatter


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline handler
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(query: str, uploaded_image: Optional[str]):
    """
    Run ONLY sentiment agent (no full pipeline)
    """

    if not query or not query.strip():
        return (
            "⚠️ Please enter a query.",
            "_No trace available._",
            None,
        )

    try:
        # Build state
        state = {
            "query": query.strip(),
            "trace_log": [],
        }

        # 🚀 Call ONLY sentiment agent
        final_state = run_sentiment(state)

        sentiment = final_state.get("sentiment_result", {})
        trace_log = final_state.get("trace_log", [])

        # ── Build report manually ─────────────────────
        report = f"""
# 📊 Sentiment Report

**Ticker:** {sentiment.get("ticker", "N/A")}

**Trend:** {sentiment.get("trend", "N/A")}
**Score:** {sentiment.get("score", 0)}
**Confidence:** {sentiment.get("confidence", "N/A")}
**Headlines Analysed:** {sentiment.get("headline_count", 0)}

---

## 📝 Summary
{sentiment.get("summary", "No summary available.")}

---

## 📰 Top Headlines
"""

        for i, h in enumerate(sentiment.get("top_headlines", []), 1):
            report += f"{i}. {h}\n"

        # simple trace
        trace = "\n".join([str(t) for t in trace_log])

        return report, trace, None

    except Exception as e:
        return (
            f"❌ Error:\n{str(e)}",
            "_Trace unavailable_",
            None,
        )

def _build_report(state: dict) -> str:
    """Assemble a readable Markdown report from the final agent state."""
    parts: list[str] = ["# 📊 FinSight AI — Analysis Report\n"]
    query = state.get("query", "")
    if query:
        parts.append(f"> **Query:** {query}\n")

    # ── Sentiment block ──────────────────────────────────────────────────────
    sentiment = state.get("sentiment_result")
    if sentiment:
        trend_emoji = {
            "bullish": "📈", "bearish": "📉", "mixed": "↕️", "neutral": "➡️"
        }.get(sentiment.get("trend", ""), "•")

        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
            sentiment.get("confidence", ""), "⚪"
        )

        dist = sentiment.get("distribution", {})
        ticker = sentiment.get("ticker", "N/A")

        parts.append(f"## {trend_emoji} Sentiment Analysis — `{ticker}`\n")
        parts.append(f"| Metric | Value |")
        parts.append(f"|--------|-------|")
        parts.append(f"| **Trend** | {trend_emoji} {sentiment.get('trend', 'N/A').title()} |")
        parts.append(f"| **Score** | `{sentiment.get('score', 0):+.4f}` |")
        parts.append(f"| **Confidence** | {conf_emoji} {sentiment.get('confidence', 'N/A').title()} |")
        parts.append(f"| **Headlines Analysed** | {sentiment.get('headline_count', 0)} |")
        parts.append(f"| **Processing Time** | {sentiment.get('elapsed_ms', 0)} ms |\n")

        parts.append("### Distribution")
        parts.append(f"- 🟢 Positive: **{dist.get('positive', 0)*100:.1f}%**")
        parts.append(f"- ⚪ Neutral:  **{dist.get('neutral', 0)*100:.1f}%**")
        parts.append(f"- 🔴 Negative: **{dist.get('negative', 0)*100:.1f}%**\n")

        summary = sentiment.get("summary", "")
        if summary:
            parts.append(f"### 📝 Summary\n{summary}\n")

        top = sentiment.get("top_headlines", [])
        if top:
            parts.append("### 📰 Top Headlines")
            for i, h in enumerate(top, 1):
                parts.append(f"{i}. {h}")
            parts.append("")

    # ── RAG / SQL / Chart blocks (extensible) ────────────────────────────────
    rag_answer = state.get("rag_answer") or state.get("answer")
    if rag_answer:
        parts.append(f"## 📚 Research Answer\n{rag_answer}\n")

    sql_result = state.get("sql_result") or state.get("sql_answer")
    if sql_result:
        parts.append(f"## 🗄️ Data Query Result\n```\n{sql_result}\n```\n")

    final_report = state.get("final_report") or state.get("report")
    if final_report and final_report not in (rag_answer, sql_result):
        parts.append(f"## 📋 Final Report\n{final_report}\n")

    if len(parts) <= 2:
        parts.append("_No structured output was produced by the pipeline._")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Example queries
# ─────────────────────────────────────────────────────────────────────────────
_EXAMPLES = [
    ["What is the current sentiment for Apple (AAPL)?", None],
    ["Is TSLA bullish or bearish right now?", None],
    ["Give me a sentiment analysis of NVDA", None],
    ["How is Microsoft stock performing sentiment-wise?", None],
    ["What is the market sentiment for Amazon?", None],
]


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    custom_css = """
    .finsight-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .report-panel { font-size: 0.95rem; }
    .trace-panel  { font-family: monospace; font-size: 0.82rem; }
    .status-bar   { font-size: 0.78rem; color: #6b7280; }
    """

    with gr.Blocks(
        title="FinSight AI",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="finsight-header">
            <h1>📈 FinSight AI</h1>
            <p style="color:#6b7280; margin:0;">
                Multi-Agent Financial Intelligence Platform
            </p>
        </div>
        """)

        # ── Input row ────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="💬 Financial Query",
                    placeholder="e.g. 'What is NVDA sentiment?'",
                    lines=3
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📎 Upload (Optional)")
                image_upload = gr.Image(
                    type="filepath",
                    height=120,
                    container=True
                )

        with gr.Row():
            analyse_btn = gr.Button(
                "🔍  Analyse",
                variant="primary",
                size="lg",
                scale=2,
            )
            clear_btn = gr.Button("🗑️  Clear", variant="secondary", scale=1)

        status_text = gr.Textbox(
            label="",
            value="Ready.",
            interactive=False,
            max_lines=1,
            elem_classes=["status-bar"],
        )

        # ── Examples ─────────────────────────────────────────────────────────
        gr.Examples(
            examples=_EXAMPLES,
            inputs=[query_box, image_upload],
            label="💡 Example Queries",
        )

        gr.Markdown("---")

        # ── Output row ───────────────────────────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                report_output = gr.Markdown(
                    label="📊 Analysis Report",
                    value="_Results will appear here after you submit a query._",
                    elem_classes=["report-panel"],
                )
            with gr.Column(scale=2):
                trace_output = gr.Markdown(
                    label="🔎 Agent Trace",
                    value="_Execution trace will appear here._",
                    elem_classes=["trace-panel"],
                )

        # ── Chart row ────────────────────────────────────────────────────────
        with gr.Row():
            chart_output = gr.Image(
                label="📈 Generated Chart",
                visible=False,
                height=400,
            )

        # ─────────────────────────────────────────────────────────────────────
        # Event wiring
        # ─────────────────────────────────────────────────────────────────────
        def on_analyse(query: str, image_path: Optional[str]):
            """Wrapper that updates the status bar and toggles chart visibility."""
            yield (
                gr.update(value="⏳ Running pipeline — please wait..."),
                gr.update(value="_Analysing..._"),
                gr.update(value="_Analysing..._"),
                gr.update(visible=False, value=None),
            )

            report, trace, chart = run_pipeline(query, image_path)

            yield (
                gr.update(value="✅ Analysis complete."),
                gr.update(value=report),
                gr.update(value=trace),
                gr.update(visible=chart is not None, value=chart),
            )

        analyse_btn.click(
            fn=on_analyse,
            inputs=[query_box, image_upload],
            outputs=[status_text, report_output, trace_output, chart_output],
        )

        query_box.submit(
            fn=on_analyse,
            inputs=[query_box, image_upload],
            outputs=[status_text, report_output, trace_output, chart_output],
        )

        def on_clear():
            return (
                "",
                None,
                "Ready.",
                "_Results will appear here after you submit a query._",
                "_Execution trace will appear here._",
                gr.update(visible=False, value=None),
            )

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[query_box, image_upload, status_text,
                     report_output, trace_output, chart_output],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight AI — Gradio UI")
    parser.add_argument("--share",  action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",   type=int, default=7860, help="Local port (default 7860)")
    parser.add_argument("--host",   type=str, default="0.0.0.0", help="Bind host")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )