"""
webapp.py — minimal web GUI for the Harness agent.

Serves a single-page chat UI and streams each agent step (assistant text,
tool call, tool result) as Server-Sent Events, so tool-call detail is
revealed progressively instead of dumped all at once.

Run with:  python -m src.webapp   (from the repo root)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from . import functions
from .harness import ALL_TOOLS, HarnessAgent, TARGET_DIR

app = Flask(__name__)

_AGENT: Optional[HarnessAgent] = None


def _get_agent() -> HarnessAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = HarnessAgent(target_dir=TARGET_DIR)
    return _AGENT


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    agent = _get_agent()
    return jsonify({
        "target_dir": agent.target_dir,
        "tree_stats": functions.tree_stats(),
        "tools": [{"name": t["name"], "description": t["description"]} for t in ALL_TOOLS],
        "message_count": len(agent.messages),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    global _AGENT
    _AGENT = HarnessAgent(target_dir=TARGET_DIR)
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    agent = _get_agent()

    def generate() -> Iterator[str]:
        try:
            for event in agent.run_stream(message):
                yield _sse(event)
        except Exception as exc:
            # Keep the stream well-formed even if something throws mid-loop —
            # the UI gets an error card instead of a hung/broken connection.
            yield _sse({"type": "error", "content": f"{type(exc).__name__}: {exc}"})
        yield _sse({"type": "done"})

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
