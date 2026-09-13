"""
functions.py — Harness tools for the repo assistant.

Every public function here is designed to be called by an LLM via tool-use:
  - JSON-serializable arguments
  - JSON-serializable dict return
  - path-safe (nothing escapes REPO_ROOT)

The tree is loaded lazily from the cache produced by the scanner
(`repo_cache.get_repo_tree`). Adjust the import below if your caching
module is named differently.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(os.environ.get("REPO_ROOT", ".")).resolve()
MAX_READ_BYTES = 512 * 1024          # 512 KB per read
MAX_GREP_RESULTS = 200
MAX_SEARCH_RESULTS = 100
MAX_OUTPUT_BYTES = 64 * 1024         # truncate runaway program output
DEFAULT_TIMEOUT = 15                 # seconds

EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "target", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".tox", ".cache",
}

# Semantic node kinds the scanner knows about — used by search/outline.
CONSTRUCT_KINDS = {
    "function", "class", "struct", "enum", "impl", "trait",
    "module", "namespace", "template", "macro",
}


# ══════════════════════════════════════════════════════════════════════
# Lazy tree loading + node indexing
# ══════════════════════════════════════════════════════════════════════

try:
    from repo_cache import get_repo_tree  # noqa: F401
except ImportError:  # fallback if you defined the caching in a notebook
    def get_repo_tree(force_refresh: bool = False) -> Dict[str, Any]:
        raise RuntimeError(
            "get_repo_tree unavailable — import from your caching module "
            "or set functions.get_repo_tree = <callable>"
        )


_TREE: Optional[Dict[str, Any]] = None
_NODE_INDEX: Optional[Dict[str, Dict[str, Any]]] = None   # id -> {node, file, parent_id}


def _tree() -> Dict[str, Any]:
    global _TREE
    if _TREE is None:
        _TREE = get_repo_tree()
    return _TREE


def refresh_tree() -> Dict[str, Any]:
    """Force-reload the tree from disk (e.g. after edits)."""
    global _TREE, _NODE_INDEX
    _TREE = get_repo_tree(force_refresh=True)
    _NODE_INDEX = None
    return _TREE


def _node_id(file_path: str, node: Dict[str, Any]) -> str:
    return f"{file_path}::{node['start_byte']}-{node['end_byte']}"


def _index() -> Dict[str, Dict[str, Any]]:
    """Build a flat id -> {node, file, parent_id} map over the whole tree."""
    global _NODE_INDEX
    if _NODE_INDEX is not None:
        return _NODE_INDEX

    idx: Dict[str, Dict[str, Any]] = {}
    for f in _tree()["files"]:
        if f.get("error") or not f.get("root_node"):
            continue
        path = f["path"]

        def walk(node: Dict[str, Any], parent_id: Optional[str]) -> None:
            nid = _node_id(path, node)
            idx[nid] = {"node": node, "file": path, "parent_id": parent_id}
            for c in node.get("children", []):
                walk(c, nid)

        walk(f["root_node"], None)
    _NODE_INDEX = idx
    return idx


def _summarize(node: Dict[str, Any], nid: str, file_path: str) -> Dict[str, Any]:
    return {
        "node_id": nid,
        "file": file_path,
        "kind": node["type"],
        "raw_type": node["raw_type"],
        "name": node.get("name"),
        "start_line": node["start_point"][0],
        "end_line": node["end_point"][0],
    }


# ══════════════════════════════════════════════════════════════════════
# Path safety
# ══════════════════════════════════════════════════════════════════════

def _safe_path(rel_or_abs: str) -> Path:
    """Resolve a path and confirm it lives inside REPO_ROOT."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        raise PermissionError(f"path escapes REPO_ROOT: {rel_or_abs}")
    return p


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


# ══════════════════════════════════════════════════════════════════════
# TOOL: search_symbols
# ══════════════════════════════════════════════════════════════════════

def search_symbols(
    query: str,
    kinds: Optional[List[str]] = None,
    file_pattern: Optional[str] = None,
    regex: bool = False,
    case_sensitive: bool = True,
    max_results: int = MAX_SEARCH_RESULTS,
) -> Dict[str, Any]:
    """
    Search for named constructs (functions, classes, structs, enums, impls,
    traits, modules, namespaces, templates, macros) and variables.

    Args:
        query: name or pattern to look for.
        kinds: restrict to these semantic kinds. e.g. ["function","class"].
               If None, all construct kinds + "variable" are searched.
        file_pattern: substring or glob matched against the file path.
        regex: treat `query` as a regex.
        case_sensitive: match case-sensitively.
        max_results: cap number of results.

    Returns:
        {"results": [...], "count": N, "truncated": bool}
    """
    kinds = set(kinds) if kinds else (CONSTRUCT_KINDS | {"variable"})
    include_constructs = bool(kinds & CONSTRUCT_KINDS)
    include_vars = "variable" in kinds

    try:
        matcher = re.compile(query if regex else re.escape(query),
                            0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        return {"error": f"bad regex: {exc}", "results": [], "count": 0}

    def path_matches(p: str) -> bool:
        if not file_pattern:
            return True
        if any(ch in file_pattern for ch in "*?["):
            import fnmatch
            return fnmatch.fnmatch(p, file_pattern)
        return file_pattern in p

    results: List[Dict[str, Any]] = []

    for f in _tree()["files"]:
        if not path_matches(f["path"]):
            continue
        if f.get("error") or not f.get("root_node"):
            continue

        def walk(node: Dict[str, Any]) -> None:
            if len(results) >= max_results:
                return
            nid = _node_id(f["path"], node)
            ntype = node["type"]

            if include_constructs and ntype in kinds:
                name = node.get("name") or ""
                if name and matcher.search(name):
                    results.append(_summarize(node, nid, f["path"]))

            # variable-like: name field of an assignment's LHS
            if include_vars and ntype == "assignment":
                # Heuristic: the first child with a `name` field
                for c in node.get("children", []):
                    cname = c.get("name") or ""
                    if cname and matcher.search(cname):
                        results.append({
                            "node_id": nid,
                            "file": f["path"],
                            "kind": "variable",
                            "raw_type": c["raw_type"],
                            "name": cname,
                            "start_line": c["start_point"][0],
                            "end_line": c["end_point"][0],
                        })
                        break

            for c in node.get("children", []):
                walk(c)

        walk(f["root_node"])

    return {
        "results": results,
        "count": len(results),
        "truncated": len(results) >= max_results,
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: get_adjacent_nodes
# ══════════════════════════════════════════════════════════════════════

def get_adjacent_nodes(node_id: str) -> Dict[str, Any]:
    """
    Given a node_id (from search_symbols, get_file_outline, etc.), return:
      - the node itself,
      - its parent,
      - its siblings (other children of the parent),
      - its direct children.

    Useful for the LLM to "zoom out" from a match and understand context.
    """
    idx = _index()
    entry = idx.get(node_id)
    if entry is None:
        return {"error": f"unknown node_id: {node_id}"}

    node = entry["node"]
    parent_id = entry["parent_id"]

    parent = idx.get(parent_id) if parent_id else None

    siblings: List[Dict[str, Any]] = []
    if parent:
        for c in parent["node"].get("children", []):
            cid = _node_id(parent["file"], c)
            if cid != node_id:
                siblings.append(_summarize(c, cid, parent["file"]))

    children = [
        _summarize(c, _node_id(entry["file"], c), entry["file"])
        for c in node.get("children", [])
    ]

    return {
        "node": _summarize(node, node_id, entry["file"]),
        "parent": (_summarize(parent["node"], parent_id, parent["file"])
                   if parent else None),
        "siblings": siblings,
        "children": children,
        "file": entry["file"],
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: read_file
# ══════════════════════════════════════════════════════════════════════

def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_bytes: int = MAX_READ_BYTES,
) -> Dict[str, Any]:
    """
    Read a file (or a slice of it). Lines are 1-based and inclusive.

    Returns {"path", "content", "start_line", "end_line", "total_lines",
             "truncated"} or {"error": ...}.
    """
    try:
        p = _safe_path(path)
    except PermissionError as exc:
        return {"error": str(exc)}

    if not p.is_file():
        return {"error": f"not a file: {path}"}

    try:
        raw = p.read_bytes()
    except OSError as exc:
        return {"error": str(exc)}

    truncated = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    total = len(lines)
    s = (start_line - 1) if start_line else 0
    e = end_line if end_line else total
    s = max(0, min(s, total))
    e = max(s, min(e, total))
    sliced = lines[s:e]

    return {
        "path": _rel(p),
        "content": "\n".join(sliced),
        "start_line": s + 1,
        "end_line": e,
        "total_lines": total,
        "truncated": truncated,
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: edit_file
# ══════════════════════════════════════════════════════════════════════

def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """
    Replace `old_string` with `new_string` in a repo file.

    Behaves like a strict str_replace:
      - `old_string` must appear (unless it's "" and you want to append),
      - by default must appear exactly once (safety),
      - set `replace_all=True` to replace every occurrence.

    After a successful edit, invalidates the in-memory tree (call
    refresh_tree() to rescan on the next need).
    """
    try:
        p = _safe_path(path)
    except PermissionError as exc:
        return {"error": str(exc)}

    if not p.is_file():
        return {"error": f"not a file: {path}"}

    original = p.read_text(encoding="utf-8")
    occurrences = original.count(old_string)

    if old_string == "":
        # explicit append mode
        new_content = original + new_string
    elif occurrences == 0:
        return {"error": "old_string not found", "occurrences": 0}
    elif occurrences > 1 and not replace_all:
        return {
            "error": f"old_string appears {occurrences} times; "
                     f"set replace_all=True or provide more context",
            "occurrences": occurrences,
        }
    else:
        new_content = original.replace(old_string, new_string,
                                       -1 if replace_all else 1)

    # Backup then write
    backup = p.with_suffix(p.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    p.write_text(new_content, encoding="utf-8")

    global _TREE, _NODE_INDEX
    _TREE = None
    _NODE_INDEX = None

    return {
        "path": _rel(p),
        "occurrences_replaced": occurrences if old_string else 0,
        "bytes_before": len(original),
        "bytes_after": len(new_content),
        "backup": _rel(backup),
    }


def write_file(path: str, content: str) -> Dict[str, Any]:
    """
    Overwrite (or create) a file with the given content.
    Use edit_file for surgical changes; use this for whole-file writes.
    """
    try:
        p = _safe_path(path)
    except PermissionError as exc:
        return {"error": str(exc)}

    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    if existed:
        backup = p.with_suffix(p.suffix + ".bak")
        backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(content, encoding="utf-8")

    global _TREE, _NODE_INDEX
    _TREE = None
    _NODE_INDEX = None

    return {"path": _rel(p), "existed": existed, "bytes": len(content)}


# ══════════════════════════════════════════════════════════════════════
# TOOL: run_python
# ══════════════════════════════════════════════════════════════════════

def run_python(
    file_path: Optional[str] = None,
    code: Optional[str] = None,
    args: Optional[List[str]] = None,
    stdin: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Python code. Either `file_path` (a repo file) or `code` (inline).

    Args (if `code` is used) are appended to the command line.
    Returns exit code, stdout, stderr, duration.
    """
    if not file_path and not code:
        return {"error": "provide file_path or code"}

    args = args or []
    workdir = _safe_path(cwd) if cwd else REPO_ROOT

    if file_path:
        try:
            target = _safe_path(file_path)
        except PermissionError as exc:
            return {"error": str(exc)}
        if not target.is_file():
            return {"error": f"not a file: {file_path}"}
        cmd = [sys.executable, str(target), *args]
    else:
        cmd = [sys.executable, "-c", code, *args]

    return _run(cmd, cwd=workdir, stdin=stdin, timeout=timeout)


# ══════════════════════════════════════════════════════════════════════
# TOOL: run_cpp
# ══════════════════════════════════════════════════════════════════════

def run_cpp(
    file_path: Optional[str] = None,
    code: Optional[str] = None,
    args: Optional[List[str]] = None,
    stdin: Optional[str] = None,
    std: str = "c++17",
    extra_flags: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    compiler: str = "g++",
) -> Dict[str, Any]:
    """
    Compile & run a C++ program. Either `file_path` (repo file) or `code`.

    The binary is built into a temporary directory and executed there.
    Returns compile_stdout/compile_stderr if compilation fails, otherwise
    exit_code/stdout/stderr of the run.
    """
    if not file_path and not code:
        return {"error": "provide file_path or code"}

    extra_flags = extra_flags or []
    args = args or []

    with tempfile.TemporaryDirectory(prefix="harness_cpp_") as tmp:
        tmp_path = Path(tmp)

        if file_path:
            try:
                src = _safe_path(file_path)
            except PermissionError as exc:
                return {"error": str(exc)}
            if not src.is_file():
                return {"error": f"not a file: {file_path}"}
        else:
            src = tmp_path / "main.cpp"
            src.write_text(code, encoding="utf-8")

        binary = tmp_path / "prog"
        compile_cmd = [
            compiler, f"-std={std}", "-O0", "-Wall",
            str(src), "-o", str(binary), *extra_flags,
        ]

        compiled = _run(compile_cmd, cwd=REPO_ROOT, timeout=timeout)
        if compiled["exit_code"] != 0:
            return {
                "stage": "compile",
                "compile_stdout": compiled["stdout"],
                "compile_stderr": compiled["stderr"],
                "compile_exit_code": compiled["exit_code"],
            }

        run_result = _run([str(binary), *args], cwd=tmp_path,
                          stdin=stdin, timeout=timeout)
        run_result["stage"] = "run"
        run_result["compile_stderr"] = compiled["stderr"]
        return run_result


# ══════════════════════════════════════════════════════════════════════
# Internal subprocess runner
# ══════════════════════════════════════════════════════════════════════

def _run(
    cmd: List[str],
    cwd: Path,
    stdin: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=stdin.encode() if stdin else None,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"error": f"command not found: {exc}",
                "exit_code": None, "stdout": "", "stderr": ""}
    except subprocess.TimeoutExpired as exc:
        return {
            "error": f"timeout after {timeout}s",
            "exit_code": None,
            "stdout": (exc.stdout or b"").decode("utf-8", "replace")[:MAX_OUTPUT_BYTES],
            "stderr": (exc.stderr or b"").decode("utf-8", "replace")[:MAX_OUTPUT_BYTES],
            "duration_s": time.time() - t0,
        }

    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.decode("utf-8", "replace")[:MAX_OUTPUT_BYTES],
        "stderr": proc.stderr.decode("utf-8", "replace")[:MAX_OUTPUT_BYTES],
        "duration_s": round(time.time() - t0, 3),
        "cmd": cmd,
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: list_files
# ══════════════════════════════════════════════════════════════════════

def list_files(
    directory: str = ".",
    pattern: str = "*",
    recursive: bool = True,
    max_results: int = 500,
) -> Dict[str, Any]:
    """
    List files under `directory` matching a glob `pattern`.

    Set recursive=False for a flat listing.
    """
    try:
        base = _safe_path(directory)
    except PermissionError as exc:
        return {"error": str(exc)}

    if not base.is_dir():
        return {"error": f"not a directory: {directory}"}

    results: List[str] = []
    iterator: Iterable[Path] = base.rglob(pattern) if recursive else base.glob(pattern)

    for p in iterator:
        if len(results) >= max_results:
            break
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.is_file():
            results.append(_rel(p))

    return {"results": sorted(results), "count": len(results),
            "truncated": len(results) >= max_results}


# ══════════════════════════════════════════════════════════════════════
# TOOL: grep
# ══════════════════════════════════════════════════════════════════════

def grep(
    pattern: str,
    file_glob: str = "*",
    case_sensitive: bool = True,
    max_results: int = MAX_GREP_RESULTS,
    context_lines: int = 0,
) -> Dict[str, Any]:
    """
    Text search across the repo. Returns matches with file + line number.

    `pattern` is a regex. `file_glob` restricts paths (e.g. "*.py").
    """
    try:
        rx = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        return {"error": f"bad regex: {exc}"}

    results: List[Dict[str, Any]] = []

    for f in _tree()["files"]:
        if len(results) >= max_results:
            break
        if not Path(f["path"]).match(file_glob):
            continue
        try:
            p = _safe_path(f["path"])
        except PermissionError:
            continue
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            if rx.search(line):
                if context_lines:
                    lo = max(0, i - context_lines)
                    hi = min(len(lines), i + context_lines + 1)
                    ctx = lines[lo:hi]
                else:
                    ctx = [line]
                results.append({
                    "file": f["path"],
                    "line": i + 1,
                    "match": line,
                    "context": ctx,
                })
                if len(results) >= max_results:
                    break

    return {"results": results, "count": len(results),
            "truncated": len(results) >= max_results}


# ══════════════════════════════════════════════════════════════════════
# TOOL: find_references
# ══════════════════════════════════════════════════════════════════════

def find_references(
    symbol: str,
    max_results: int = MAX_GREP_RESULTS,
) -> Dict[str, Any]:
    """
    Find all textual references to an identifier across the repo.
    Word-boundary regex — ignores substrings like `foo` inside `foobar`.
    """
    return grep(rf"\b{re.escape(symbol)}\b",
                case_sensitive=True, max_results=max_results)


# ══════════════════════════════════════════════════════════════════════
# TOOL: get_file_outline
# ══════════════════════════════════════════════════════════════════════

def get_file_outline(path: str) -> Dict[str, Any]:
    """
    Return the top-level (and one level deep) structure of a file:
    functions, classes, methods, imports.

    Cheaper than dumping the whole tree; ideal for orienting an LLM.
    """
    idx = _index()
    target_file = None
    for f in _tree()["files"]:
        if f["path"] == path or Path(f["path"]).name == Path(path).name:
            target_file = f
            break

    if target_file is None:
        return {"error": f"file not in scanned tree: {path}"}
    if target_file.get("error"):
        return {"error": target_file["error"]}

    outline: List[Dict[str, Any]] = []

    def add_entry(node: Dict[str, Any], depth: int) -> None:
        nid = _node_id(target_file["path"], node)
        outline.append({
            **_summarize(node, nid, target_file["path"]),
            "depth": depth,
        })
        if depth == 0:  # one more level for methods
            for c in node.get("children", []):
                if c["type"] in CONSTRUCT_KINDS:
                    add_entry(c, depth + 1)

    for child in target_file["root_node"].get("children", []):
        if child["type"] in CONSTRUCT_KINDS or child["type"] == "import":
            add_entry(child, 0)

    return {
        "path": target_file["path"],
        "language": target_file["language"],
        "outline": outline,
        "count": len(outline),
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: get_node
# ══════════════════════════════════════════════════════════════════════

def get_node(node_id: str) -> Dict[str, Any]:
    """Return the raw node dict for a given node_id."""
    entry = _index().get(node_id)
    if entry is None:
        return {"error": f"unknown node_id: {node_id}"}
    return {
        "node_id": node_id,
        "file": entry["file"],
        "node": entry["node"],
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: tree_stats
# ══════════════════════════════════════════════════════════════════════

def tree_stats() -> Dict[str, Any]:
    """Return the scanner's aggregate statistics for the current tree."""
    t = _tree()
    return {
        "root_path": t["root_path"],
        "stats": t["stats"],
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL: git_status / git_diff
# ══════════════════════════════════════════════════════════════════════

def git_status() -> Dict[str, Any]:
    """Short git status of the repo."""
    return _run(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT)


def git_diff(path: Optional[str] = None, staged: bool = False) -> Dict[str, Any]:
    """Unified diff. Pass `path` to restrict to one file."""
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    if path:
        try:
            cmd.append(str(_safe_path(path)))
        except PermissionError as exc:
            return {"error": str(exc)}
    return _run(cmd, cwd=REPO_ROOT)


# ══════════════════════════════════════════════════════════════════════
# TOOL: run_shell  (use sparingly — the LLM decides)
# ══════════════════════════════════════════════════════════════════════

def run_shell(command: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Run a shell command in the repo. Useful for build tools, tests, linters.
    Not sandboxed — the LLM is expected to use this responsibly.
    """
    return _run(["bash", "-lc", command], cwd=REPO_ROOT, timeout=timeout)


# ══════════════════════════════════════════════════════════════════════
# TOOLS registry — schemas for OpenAI / Anthropic tool-use
# ══════════════════════════════════════════════════════════════════════

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_symbols",
        "description": (
            "Search the repo for functions, classes, structs, enums, impls, "
            "traits, modules, namespaces, templates, macros, or variables by name. "
            "Use this FIRST when you don't know where something is defined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kinds": {"type": "array", "items": {"type": "string"}},
                "file_pattern": {"type": "string"},
                "regex": {"type": "boolean", "default": False},
                "case_sensitive": {"type": "boolean", "default": True},
                "max_results": {"type": "integer", "default": 100},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_adjacent_nodes",
        "description": (
            "Given a node_id, return its parent, siblings, and children — "
            "used to walk the syntax tree outward from a match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "get_node",
        "description": "Return the raw syntax-tree node for a given node_id.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file (or a line range). Lines are 1-based, inclusive. "
            "Always read before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file. Requires old_string to appear "
            "uniquely (set replace_all=True to allow multiple). Creates a .bak."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files under a directory matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
                "recursive": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "grep",
        "description": "Regex text search across the repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "file_glob": {"type": "string", "default": "*"},
                "case_sensitive": {"type": "boolean", "default": True},
                "context_lines": {"type": "integer", "default": 0},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "find_references",
        "description": "Find word-boundary references to an identifier.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_file_outline",
        "description": (
            "Return the structural outline of a file (functions, classes, "
            "methods, imports). Cheaper than reading the whole tree."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "tree_stats",
        "description": "Aggregate stats for the scanned repository.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_python",
        "description": "Run a Python file or inline snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "code": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "stdin": {"type": "string"},
                "timeout": {"type": "integer", "default": 15},
            },
        },
    },
    {
        "name": "run_cpp",
        "description": "Compile and run a C++ file or inline snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "code": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "stdin": {"type": "string"},
                "std": {"type": "string", "default": "c++17"},
                "extra_flags": {"type": "array", "items": {"type": "string"}},
                "timeout": {"type": "integer", "default": 15},
            },
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command in the repo (build, test, lint).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 15},
            },
            "required": ["command"],
        },
    },
    {
        "name": "git_status",
        "description": "Short git status of the repo.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_diff",
        "description": "Unified diff (optionally staged, optionally for one path).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "staged": {"type": "boolean", "default": False},
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════
# Dispatch helper — call a tool by name from the LLM's tool-call payload
# ══════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "search_symbols": search_symbols,
    "get_adjacent_nodes": get_adjacent_nodes,
    "get_node": get_node,
    "read_file": read_file,
    "edit_file": edit_file,
    "write_file": write_file,
    "list_files": list_files,
    "grep": grep,
    "find_references": find_references,
    "get_file_outline": get_file_outline,
    "tree_stats": tree_stats,
    "run_python": run_python,
    "run_cpp": run_cpp,
    "run_shell": run_shell,
    "git_status": git_status,
    "git_diff": git_diff,
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