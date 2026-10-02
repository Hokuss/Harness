"""
tools.py — general-purpose (non-repo) tools for the Harness agent.

Where functions.py is scoped to the repository (read/edit/grep/tree-sitter),
this module holds standalone utility tools that don't touch REPO_ROOT:
basic arithmetic and a web search. Same contract as functions.py:
  - JSON-serializable arguments
  - JSON-serializable dict return
  - a TOOLS registry + call_tool(name, arguments) dispatcher
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════
# TOOL: arithmetic
# ══════════════════════════════════════════════════════════════════════

def add(a: float, b: float) -> Dict[str, Any]:
    """Add two numbers."""
    return {"result": a + b}


def subtract(a: float, b: float) -> Dict[str, Any]:
    """Subtract b from a."""
    return {"result": a - b}


def multiply(a: float, b: float) -> Dict[str, Any]:
    """Multiply two numbers."""
    return {"result": a * b}


def divide(a: float, b: float) -> Dict[str, Any]:
    """Divide a by b. Returns an error instead of raising on division by zero."""
    if b == 0:
        return {"error": "division by zero"}
    return {"result": a / b}


# ══════════════════════════════════════════════════════════════════════
# TOOL: web_search
# ══════════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the public web (DuckDuckGo, no API key required) and return
    lightweight results: title, url, snippet. Use this for anything
    outside the repo — docs, error messages, library APIs, current events.
    """
    max_results = max(1, min(max_results, 20))
    try:
        from ddgs import DDGS
    except ImportError:
        return {"error": "ddgs package not installed (pip install ddgs)"}

    try:
        raw = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw
    ]
    return {"results": results, "count": len(results)}


# ══════════════════════════════════════════════════════════════════════
# TOOLS registry — schemas for OpenAI / Anthropic tool-use
# ══════════════════════════════════════════════════════════════════════

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "add",
        "description": "Add two numbers together.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "subtract",
        "description": "Subtract the second number from the first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers together.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "divide",
        "description": "Divide the first number by the second.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the public web for information outside this repository "
            "(documentation, library APIs, error messages, general knowledge). "
            "Returns title/url/snippet per result. No API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════
# Dispatch helper — call a tool by name from the LLM's tool-call payload
# ══════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "web_search": web_search,
}


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name. This is what your LLM loop calls."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
