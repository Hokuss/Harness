"""
webapp.py — minimal web GUI for the Harness agent.

Serves a single-page chat UI and streams each agent step (assistant text,
tool call, tool result) as Server-Sent Events, so tool-call detail is
revealed progressively instead of dumped all at once.

Run with:  python -m src.webapp   (from the repo root)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from . import functions
from .harness import ALL_TOOLS, HarnessAgent, TARGET_DIR

app = Flask(__name__)

_AGENT: Optional[HarnessAgent] = None
_REPO_ROOT = Path(".").resolve()


def _get_agent() -> HarnessAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = HarnessAgent(target_dir=TARGET_DIR)
    return _AGENT


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _resolve_target_dir(raw: str) -> Path:
    """
    Resolve a user-supplied target_dir against the repo root, rejecting
    anything that escapes it — HarnessAgent.enforce_target_dir() and
    functions._safe_path() both assume the target lives inside REPO_ROOT
    (which is always the harness repo root, not the target itself), so a
    target outside it would silently break every path-scoped tool.
    """
    cleaned = raw.strip().strip("/\\") or TARGET_DIR
    candidate = (_REPO_ROOT / cleaned).resolve()
    candidate.relative_to(_REPO_ROOT)  # raises ValueError if it escapes
    return candidate


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


@app.route("/api/targets")
def targets():
    available = []
    for p in sorted(_REPO_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name in functions.EXCLUDE_DIRS:
            continue
        available.append(p.name)
    return jsonify({"current": _get_agent().target_dir, "available": available})


@app.route("/api/reset", methods=["POST"])
def reset():
    global _AGENT
    payload = request.get_json(silent=True) or {}
    raw_target = payload.get("target_dir") or TARGET_DIR

    try:
        candidate = _resolve_target_dir(raw_target)
    except ValueError:
        return jsonify({"error": f"target_dir must be inside the repo root: {raw_target}"}), 400
    if not candidate.is_dir():
        return jsonify({"error": f"not a directory: {raw_target}"}), 400

    target_dir = candidate.relative_to(_REPO_ROOT).as_posix()
    _AGENT = HarnessAgent(target_dir=target_dir)
    return jsonify({"ok": True, "target_dir": target_dir})


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
