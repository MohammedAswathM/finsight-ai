"""
evaluation/ragas_eval.py
FinSight AI — RAGAS Evaluation Suite
─────────────────────────────────────────────────────────────────────────────
Evaluates the RAG pipeline using the RAGAS framework across three metrics:
  • faithfulness        — does the answer stay faithful to the context?
  • answer_relevancy    — is the answer relevant to the question?
  • context_precision   — is the retrieved context precise and necessary?

Dataset: 5 curated financial Q&A pairs with realistic retrieval contexts.

Usage:
    python evaluation/ragas_eval.py
    python evaluation/ragas_eval.py --verbose
    python evaluation/ragas_eval.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation dataset — 5 financial queries
# ─────────────────────────────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "question": "What drove Apple's revenue growth in the most recent quarter?",
        "answer": (
            "Apple's revenue growth was primarily driven by strong iPhone sales, "
            "particularly the iPhone 15 Pro lineup, alongside continued growth in "
            "Services revenue which reached a new all-time high. The company also "
            "saw expansion in its installed base of active devices."
        ),
        "contexts": [
            (
                "Apple Inc. reported quarterly revenue of $89.5 billion, up 2% year "
                "over year. iPhone revenue was $46.0 billion. Services revenue reached "
                "an all-time high of $20.4 billion, up 16% year over year. The company's "
                "active installed base of devices reached an all-time high across all "
                "major product categories."
            ),
            (
                "The iPhone 15 Pro and Pro Max models contributed significantly to ASP "
                "(average selling price) improvements. Tim Cook highlighted the strong "
                "performance in emerging markets including India and Southeast Asia."
            ),
        ],
        "ground_truth": (
            "Apple's revenue was driven by strong iPhone 15 Pro sales and record "
            "Services revenue of $20.4 billion, up 16% YoY."
        ),
    },
    {
        "question": "How does NVIDIA's data center revenue compare to its gaming revenue?",
        "answer": (
            "NVIDIA's data center segment has become significantly larger than its gaming "
            "segment. Data center revenue grew explosively due to AI and machine learning "
            "demand, particularly for H100 and A100 GPUs. Gaming revenue, while still "
            "substantial, represents a much smaller fraction of total revenue compared "
            "to prior years."
        ),
        "contexts": [
            (
                "NVIDIA reported data center revenue of $18.4 billion for the quarter, "
                "representing a 427% year-over-year increase. Gaming revenue was $2.9 "
                "billion, up 56% year-over-year. Data center now constitutes approximately "
                "83% of NVIDIA's total revenue."
            ),
            (
                "The surge in data center revenue is attributed to hyperscaler adoption of "
                "H100 GPUs for large language model training. Demand continues to outpace "
                "supply. NVIDIA CEO Jensen Huang called the shift a 'platform transition.'"
            ),
        ],
        "ground_truth": (
            "Data center revenue ($18.4B) dwarfs gaming revenue ($2.9B), accounting for "
            "~83% of NVIDIA's total quarterly revenue."
        ),
    },
    {
        "question": "What is the Federal Reserve's current stance on interest rates?",
        "answer": (
            "The Federal Reserve has maintained a cautious stance, keeping interest rates "
            "at elevated levels while signalling a data-dependent approach. Fed officials "
            "have indicated they need to see sustained progress on inflation before "
            "considering rate cuts. The benchmark federal funds rate remains in its "
            "current target range."
        ),
        "contexts": [
            (
                "The Federal Open Market Committee (FOMC) voted to hold the federal funds "
                "rate at its target range. Fed Chair Jerome Powell stated in the post-meeting "
                "press conference that the committee does not believe it is appropriate to "
                "reduce rates until there is greater confidence that inflation is moving "
                "sustainably toward the 2% target."
            ),
            (
                "Recent CPI data showed inflation at 3.2%, still above the Fed's 2% target. "
                "Several FOMC members have pushed back against market expectations of "
                "imminent rate cuts. The Fed's dot plot indicates fewer rate cuts anticipated "
                "for the year than markets previously priced in."
            ),
        ],
        "ground_truth": (
            "The Fed is holding rates steady and is data-dependent, requiring sustained "
            "progress toward its 2% inflation target before cutting rates."
        ),
    },
    {
        "question": "What are the key risks facing Tesla in the current market environment?",
        "answer": (
            "Tesla faces several significant risks including intensifying competition from "
            "both traditional automakers and Chinese EV manufacturers. Margin pressure from "
            "price cuts to sustain demand, declining deliveries growth, and concerns about "
            "CEO Elon Musk's focus on other ventures are also key concerns. Additionally, "
            "demand uncertainty for EVs amid high interest rates poses a challenge."
        ),
        "contexts": [
            (
                "Tesla's gross margin declined to 17.4%, down from 25% a year earlier, "
                "primarily driven by aggressive price reductions. Vehicle deliveries missed "
                "analyst expectations. Competition from BYD and other Chinese EV makers has "
                "intensified significantly in key markets including China and Europe."
            ),
            (
                "Analyst reports highlight concerns about Tesla's near-term demand trajectory "
                "and CEO Elon Musk's time allocation across Tesla, SpaceX, and X (formerly "
                "Twitter). Several institutional investors have expressed concerns about "
                "governance. Interest rate headwinds continue to dampen consumer demand for "
                "big-ticket purchases including EVs."
            ),
        ],
        "ground_truth": (
            "Key Tesla risks include margin compression, competitive pressure from Chinese "
            "EVs, delivery growth slowdown, and governance concerns around Musk."
        ),
    },
    {
        "question": "How has the S&P 500 performed year-to-date and what sectors led gains?",
        "answer": (
            "The S&P 500 has delivered strong year-to-date gains, driven primarily by the "
            "technology sector, particularly AI-related stocks. The magnificent seven mega-cap "
            "tech stocks contributed a disproportionate share of total index returns. "
            "Communication services and consumer discretionary sectors also outperformed, "
            "while utilities and real estate lagged due to interest rate sensitivity."
        ),
        "contexts": [
            (
                "The S&P 500 index gained approximately 24% year-to-date as of the latest "
                "reporting period. Technology was the best-performing sector with gains "
                "exceeding 50% YTD. The combined market capitalisation of Apple, Microsoft, "
                "Alphabet, Amazon, Nvidia, Meta, and Tesla grew substantially, concentrating "
                "index returns."
            ),
            (
                "Sector performance dispersion was elevated this year. Rate-sensitive sectors "
                "including utilities (-5%), real estate (-4%), and consumer staples (+2%) "
                "underperformed. Communication services (+55%) and information technology "
                "(+52%) were standout performers driven by AI enthusiasm and strong earnings."
            ),
        ],
        "ground_truth": (
            "S&P 500 gained ~24% YTD, led by technology (+52%) and communication services "
            "(+55%), while utilities and real estate lagged due to interest rate pressures."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluation runner
# ─────────────────────────────────────────────────────────────────────────────
def run_ragas_evaluation(verbose: bool = False) -> dict:
    """
    Execute the RAGAS evaluation pipeline and return a results dict.

    RAGAS expects a Dataset with columns:
      question, answer, contexts, ground_truth
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install ragas datasets")
        sys.exit(1)

    print("=" * 65)
    print("  FinSight AI — RAGAS Evaluation Suite")
    print("=" * 65)
    print(f"  Dataset size : {len(EVAL_DATASET)} queries")
    print(f"  Metrics      : faithfulness, answer_relevancy, context_precision")
    print("=" * 65)

    # Build HuggingFace Dataset
    hf_dataset = Dataset.from_dict({
        "question":    [d["question"]    for d in EVAL_DATASET],
        "answer":      [d["answer"]      for d in EVAL_DATASET],
        "contexts":    [d["contexts"]    for d in EVAL_DATASET],
        "ground_truth":[d["ground_truth"] for d in EVAL_DATASET],
    })

    if verbose:
        print("\n📋 Evaluation samples:")
        for i, d in enumerate(EVAL_DATASET, 1):
            print(f"  {i}. {d['question'][:80]}...")
        print()

    t0 = time.perf_counter()
    print("⏳ Running evaluation (this may take a few minutes)...")

    result = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"✅ Evaluation complete in {elapsed}s\n")

    return dict(result), elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Result display
# ─────────────────────────────────────────────────────────────────────────────
def print_results_table(results: dict, elapsed: float) -> None:
    """Print results in a clean table format."""
    metric_labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
    }

    def score_rating(score: float) -> str:
        if score >= 0.85:  return "🟢 Excellent"
        if score >= 0.70:  return "🟡 Good"
        if score >= 0.55:  return "🟠 Fair"
        return               "🔴 Poor"

    print("=" * 65)
    print("  RAGAS Evaluation Results — FinSight AI")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Score':>8}   {'Rating'}")
    print("  " + "-" * 55)

    for key, label in metric_labels.items():
        score = results.get(key, float("nan"))
        try:
            score_f = float(score)
            rating  = score_rating(score_f)
            print(f"  {label:<25} {score_f:>8.4f}   {rating}")
        except (TypeError, ValueError):
            print(f"  {label:<25} {'N/A':>8}   —")

    print("  " + "-" * 55)
    # Average of available numeric scores
    numeric = [float(v) for v in results.values() if _is_numeric(v)]
    if numeric:
        avg = sum(numeric) / len(numeric)
        print(f"  {'OVERALL AVERAGE':<25} {avg:>8.4f}   {score_rating(avg)}")

    print("=" * 65)
    print(f"  ⏱  Evaluation time: {elapsed}s")
    print("=" * 65)


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight AI — RAGAS Evaluation")
    parser.add_argument("--verbose", action="store_true",  help="Print sample details")
    parser.add_argument("--output",  type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    results, elapsed = run_ragas_evaluation(verbose=args.verbose)
    print_results_table(results, elapsed)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"results": results, "elapsed_seconds": elapsed}, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")