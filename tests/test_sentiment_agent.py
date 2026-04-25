from agents.sentiment_agent import run


# ─────────────────────────────
# 1. Basic functionality
# ─────────────────────────────
def test_sentiment_basic():
    state = {"query": "What is TSLA sentiment?", "trace_log": []}
    result = run(state)

    # structure
    assert "sentiment_result" in result
    assert "trace_log" in result

    # types
    assert isinstance(result["trace_log"], list)
    assert len(result["trace_log"]) > 0
    assert isinstance(result["trace_log"][0], str)

    s = result["sentiment_result"]

    # required fields
    assert "trend" in s
    assert "score" in s
    assert "summary" in s
    assert "confidence" in s
    assert "headline_count" in s


# ─────────────────────────────
# 2. Empty query (edge case)
# ─────────────────────────────
def test_empty_query():
    result = run({"query": "", "trace_log": []})

    s = result["sentiment_result"]

    assert s["trend"] == "neutral"
    assert s["headline_count"] == 0
    assert isinstance(result["trace_log"], list)


# ─────────────────────────────
# 3. Random / unknown query
# ─────────────────────────────
def test_random_query():
    result = run({"query": "asdfghjkl", "trace_log": []})

    assert "sentiment_result" in result
    assert isinstance(result["trace_log"], list)


# ─────────────────────────────
# 4. Output value ranges
# ─────────────────────────────
def test_score_range():
    result = run({"query": "AAPL sentiment", "trace_log": []})
    score = result["sentiment_result"]["score"]

    assert -1.0 <= score <= 1.0


# ─────────────────────────────
# 5. Trace log format consistency
# ─────────────────────────────
def test_trace_log_format():
    result = run({"query": "NVDA sentiment", "trace_log": []})

    for entry in result["trace_log"]:
        assert isinstance(entry, str)
        assert len(entry) > 0


# ─────────────────────────────
# 6. Headlines structure
# ─────────────────────────────
def test_headlines_exist():
    result = run({"query": "Amazon sentiment", "trace_log": []})

    s = result["sentiment_result"]

    assert isinstance(s["top_headlines"], list)