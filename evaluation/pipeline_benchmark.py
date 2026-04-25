"""
evaluation/pipeline_benchmark.py
FinSight AI — Pipeline Latency Benchmarking
─────────────────────────────────────────────────────────────────────────────
Measures end-to-end pipeline latency and optional per-agent timing.

Output example:
  ┌─────────────────────────────────────────────────────────┐
  │  FinSight AI — Pipeline Benchmark Results               │
  ├──────────────────────┬──────────────┬───────────────────┤
  │  Query               │  Sentiment   │  Total            │
  ├──────────────────────┼──────────────┼───────────────────┤
  │  AAPL sentiment?     │  1,243 ms    │  1,891 ms         │
  │  TSLA bullish?       │  1,102 ms    │  1,654 ms         │
  ...

Usage:
    python evaluation/pipeline_benchmark.py
    python evaluation/pipeline_benchmark.py --runs 5
    python evaluation/pipeline_benchmark.py --output benchmark.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from orchestrator.graph import run_graph


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark queries
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_QUERIES = [
    "What is the current sentiment for Apple AAPL?",
    "Is Tesla TSLA bullish or bearish right now?",
    "Give me a sentiment analysis of NVDA",
    "How is Microsoft MSFT performing sentiment-wise?",
    "What is the market outlook for Amazon AMZN?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_agent_timings(trace_log: list[dict]) -> dict[str, float]:
    """
    Parse the trace_log to extract per-agent elapsed time in ms.

    Looks for pairs of ("start", "complete") entries per agent.
    """
    timings: dict[str, float] = {}
    agent_starts: dict[str, float] = {}

    for entry in trace_log:
        agent  = entry.get("agent", "Unknown")
        action = entry.get("action", "").lower()
        ts     = entry.get("time")

        if ts is None:
            continue

        if action == "start":
            agent_starts[agent] = ts
        elif action == "complete" and agent in agent_starts:
            elapsed_ms = (ts - agent_starts[agent]) * 1000
            timings[agent] = round(elapsed_ms, 1)

    return timings


def _run_single(query: str) -> dict:
    """Run one query through the full pipeline and return timing metadata."""
    initial_state = {"query": query, "trace_log": []}

    t0 = time.perf_counter()
    final_state = run_graph(initial_state)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    trace_log     = final_state.get("trace_log", [])
    agent_timings = _extract_agent_timings(trace_log)

    # Also grab elapsed_ms from sentiment_result if present
    sentiment = final_state.get("sentiment_result", {})
    if "elapsed_ms" in sentiment:
        agent_timings["SentimentAgent"] = sentiment["elapsed_ms"]

    return {
        "query":         query,
        "total_ms":      total_ms,
        "agent_timings": agent_timings,
        "success":       "error" not in str(final_state.get("sentiment_result", {})
                                             .get("summary", "")).lower(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(
    queries: list[str],
    runs_per_query: int = 1,
    warmup: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Run the full benchmark suite.

    Parameters
    ----------
    queries         : list of query strings to benchmark
    runs_per_query  : number of repeated runs per query (averaged)
    warmup          : run one silent warmup pass before timing (loads model)
    verbose         : print live progress

    Returns
    -------
    list of result dicts
    """
    results: list[dict] = []

    # Warmup run — loads FinBERT and warms JIT caches without affecting timing
    if warmup:
        print("🔥 Warming up pipeline (loading models)...")
        try:
            run_graph({"query": queries[0], "trace_log": []})
        except Exception:
            pass
        print("   Warmup complete.\n")

    for query in queries:
        query_short = query[:45] + "..." if len(query) > 45 else query
        if verbose:
            print(f"⏱  Benchmarking: '{query_short}'")

        run_times: list[float] = []
        all_agent_timings: list[dict] = []

        for run_idx in range(runs_per_query):
            try:
                r = _run_single(query)
                run_times.append(r["total_ms"])
                all_agent_timings.append(r["agent_timings"])
                if verbose:
                    print(f"     run {run_idx+1}/{runs_per_query}: {r['total_ms']:.0f} ms  "
                          f"{'✅' if r['success'] else '❌'}")
            except Exception as exc:
                print(f"     ⚠️  Run {run_idx+1} failed: {exc}")

        if not run_times:
            continue

        # Aggregate timings across runs
        avg_total   = round(statistics.mean(run_times), 1)
        min_total   = round(min(run_times), 1)
        max_total   = round(max(run_times), 1)

        # Per-agent average across runs
        avg_agents: dict[str, float] = {}
        all_agent_keys = {k for d in all_agent_timings for k in d}
        for key in all_agent_keys:
            vals = [d[key] for d in all_agent_timings if key in d]
            if vals:
                avg_agents[key] = round(statistics.mean(vals), 1)

        results.append({
            "query":         query,
            "runs":          len(run_times),
            "avg_total_ms":  avg_total,
            "min_total_ms":  min_total,
            "max_total_ms":  max_total,
            "agent_timings": avg_agents,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Result display
# ─────────────────────────────────────────────────────────────────────────────
def print_benchmark_table(results: list[dict]) -> None:
    """Render benchmark results as a clean ASCII table."""
    if not results:
        print("No benchmark results to display.")
        return

    # Collect all unique agent names that appeared
    all_agents = list(dict.fromkeys(
        k for r in results for k in r["agent_timings"]
    ))

    col_query  = 38
    col_agent  = 14
    col_total  = 13

    header_agents = "".join(f"  {a[:12]:<12}" for a in all_agents)
    sep_agents    = "".join("  " + "-" * 12       for _   in all_agents)

    print()
    print("=" * 75)
    print("  FinSight AI — Pipeline Benchmark Results")
    print("=" * 75)
    print(
        f"  {'Query':<{col_query}}"
        f"{'Total (ms)':>{col_total}}"
        f"{'Min (ms)':>{col_total}}"
        f"{'Max (ms)':>{col_total}}"
        + header_agents
    )
    print("  " + "-" * (col_query + col_total * 3) + sep_agents)

    for r in results:
        q_short  = r["query"][:col_query - 2]
        agents_s = "".join(
            f"  {str(r['agent_timings'].get(a, '—')):>12}"
            for a in all_agents
        )
        print(
            f"  {q_short:<{col_query}}"
            f"{r['avg_total_ms']:>{col_total}.0f}"
            f"{r['min_total_ms']:>{col_total}.0f}"
            f"{r['max_total_ms']:>{col_total}.0f}"
            + agents_s
        )

    # Summary row
    totals = [r["avg_total_ms"] for r in results]
    avg    = round(statistics.mean(totals), 1)
    print("  " + "-" * (col_query + col_total * 3) + sep_agents)
    print(
        f"  {'AVERAGE':<{col_query}}"
        f"{avg:>{col_total}.0f}"
    )
    print("=" * 75)

    # One-liner summary
    sentiment_times = [r["agent_timings"].get("SentimentAgent") for r in results]
    sentiment_times = [t for t in sentiment_times if t is not None]
    if sentiment_times:
        avg_sent = round(statistics.mean(sentiment_times), 1)
        print(f"\n  Sentiment: {avg_sent} ms avg  |  "
              f"Total pipeline: {avg} ms avg")
    else:
        print(f"\n  Total pipeline: {avg} ms avg")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight AI — Pipeline Benchmark")
    parser.add_argument("--runs",    type=int, default=1,    help="Runs per query (default 1)")
    parser.add_argument("--no-warmup", action="store_true",  help="Skip warmup run")
    parser.add_argument("--verbose", action="store_true",    help="Print per-run details")
    parser.add_argument("--output",  type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 65)
    print("  FinSight AI — Pipeline Benchmark")
    print("=" * 65)
    print(f"  Queries     : {len(BENCHMARK_QUERIES)}")
    print(f"  Runs/query  : {args.runs}")
    print(f"  Warmup      : {'disabled' if args.no_warmup else 'enabled'}")
    print("=" * 65 + "\n")

    results = run_benchmark(
        queries=BENCHMARK_QUERIES,
        runs_per_query=args.runs,
        warmup=not args.no_warmup,
        verbose=args.verbose,
    )

    print_benchmark_table(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_path}\n")