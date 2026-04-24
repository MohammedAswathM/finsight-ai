"""Shared LLM factory + tiny utilities used by every agent/node."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from langchain_groq import ChatGroq

from config import GROQ_MODEL, require_groq


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Single LLM factory. All nodes must use this — no direct ChatGroq() elsewhere."""
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=require_groq(),
        temperature=temperature,
    )


def append_trace(message: str) -> List[str]:
    """Return a one-element list so LangGraph's `operator.add` reducer appends it."""
    return [message]


def strip_code_fence(content: str) -> str:
    """LLMs often wrap JSON in ```json ... ``` — strip it safely."""
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.lstrip().lower().startswith("json"):
                content = content.split("\n", 1)[1] if "\n" in content else content[4:]
    return content.strip()


def safe_get(state: Dict[str, Any], key: str, default: str = "NOT AVAILABLE") -> str:
    val = state.get(key)
    if val is None or val == "":
        return default
    return str(val)
