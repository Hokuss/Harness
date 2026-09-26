"""
repository_tree_sitter_scanner.py
A comprehensive Tree-sitter scanner for C++, Rust, and Python repositories.

Produces a full syntax tree with internal operation details.
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# ── Tree-sitter imports ──────────────────────────────────────────────
from tree_sitter import Language, Parser, Node, Tree

# Language grammars (install via pip)
import tree_sitter_python as tspython
import tree_sitter_cpp as tscpp
import tree_sitter_rust as tsrust


# ══════════════════════════════════════════════════════════════════════
# 1. LANGUAGE REGISTRY
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LanguageConfig:
    name: str
    extensions: Set[str]
    grammar: Language
    node_types: Dict[str, Any]


def build_language_configs() -> Dict[str, LanguageConfig]:
    """Return a registry mapping language names to their configs."""

    return {
        "python": LanguageConfig(
            name="python",
            extensions={".py", ".pyi"},
            grammar=Language(tspython.language()),
            node_types={
                # Top-level constructs
                "module": "module",
                "function": ["function_definition"],
                "class": ["class_definition"],
                "decorated": ["decorated_definition"],
                "import": ["import_statement", "import_from_statement"],
                # Internal operations
                "call": ["call"],
                "assignment": ["assignment"],
                "return": ["return_statement"],
                "if": ["if_statement"],
                "for": ["for_statement"],
                "while": ["while_statement"],
                "try": ["try_statement"],
                "with": ["with_statement"],
                "lambda": ["lambda"],
                "await": ["await"],
                "yield": ["yield"],
                # Field names
                "name_field": "name",
                "params_field": "parameters",
                "body_field": "body",
            },
        ),
        "cpp": LanguageConfig(
            name="cpp",
            extensions={".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h", ".c"},
            grammar=Language(tscpp.language()),
            node_types={
                # Top-level constructs
                "translation_unit": "translation_unit",
                "function": ["function_definition"],
                "class": ["class_specifier"],
                "struct": ["struct_specifier"],
                "namespace": ["namespace_definition"],
                "enum": ["enum_specifier"],
                "template": ["template_declaration"],
                # Internal operations
                "call": ["call_expression"],
                "assignment": ["assignment_expression"],
                "return": ["return_statement"],
                "if": ["if_statement"],
                "for": ["for_statement"],
                "while": ["while_statement"],
                "switch": ["switch_statement"],
                "try": ["try_statement"],
                # Field names
                "name_field": "declarator",
                "params_field": "parameters",
                "body_field": "body",
            },
        ),
        "rust": LanguageConfig(
            name="rust",
            extensions={".rs"},
            grammar=Language(tsrust.language()),
            node_types={
                # Top-level constructs
                "source_file": "source_file",
                "function": ["function_item"],
                "struct": ["struct_item"],
                "enum": ["enum_item"],
                "impl": ["impl_item"],
                "trait": ["trait_item"],
                "module": ["mod_item"],
                "macro": ["macro_definition"],
                # Internal operations
                "call": ["call_expression"],
                "assignment": ["assignment_expression"],
                "return": ["return_expression"],
                "if": ["if_expression"],
                "for": ["for_expression"],
                "while": ["while_expression"],
                "match": ["match_expression"],
                "closure": ["closure_expression"],
                "await": ["await_expression"],
                # Field names
                "name_field": "name",
                "params_field": "parameters",
                "body_field": "body",
            },
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# 2. REPOSITORY WALKER
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SourceFile:
    path: Path
    language: str
    source_bytes: bytes


def walk_repository(
    root: Path,
    configs: Dict[str, LanguageConfig],
    exclude_dirs: Optional[Set[str]] = None,
) -> Generator[SourceFile, None, None]:
    """
    Walk a repository and yield SourceFile objects for every
    recognised source file.

    Internal operations:
      - Skips hidden directories, virtual environments, and build artefacts.
      - Reads file bytes lazily.
    """
    if exclude_dirs is None:
        exclude_dirs = {
            ".git", ".venv", "venv", "__pycache__",
            "node_modules", "target", "build", "dist",
            ".mypy_cache", ".pytest_cache", ".tox",
        }

    # Build an extension → language lookup
    ext_map: Dict[str, str] = {}
    for lang_name, cfg in configs.items():
        for ext in cfg.extensions:
            ext_map[ext] = lang_name

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            if ext not in ext_map:
                continue

            language = ext_map[ext]
            try:
                source_bytes = filepath.read_bytes()
            except (OSError, PermissionError) as exc:
                print(f"[WARN] Cannot read {filepath}: {exc}", file=sys.stderr)
                continue

            yield SourceFile(
                path=filepath,
                language=language,
                source_bytes=source_bytes,
            )


# ══════════════════════════════════════════════════════════════════════
# 3. TREE-SITTER PARSER LAYER
# ══════════════════════════════════════════════════════════════════════

class TreeSitterParser:
    """
    Wraps Tree-sitter Parser instances per language.
    Caches parsers to avoid repeated construction.
    """

    def __init__(self, configs: Dict[str, LanguageConfig]):
        self._configs = configs
        self._parsers: Dict[str, Parser] = {}

    def get_parser(self, language: str) -> Parser:
        """Return a cached Parser for the given language."""
        if language not in self._parsers:
            cfg = self._configs[language]
            parser = Parser(cfg.grammar)
            self._parsers[language] = parser
        return self._parsers[language]

    def parse_file(self, source_file: SourceFile) -> Tree:
        """
        Parse a SourceFile and return its Tree-sitter Tree.

        Internal operations:
          - Selects the correct grammar.
          - Encodes source as UTF-8 bytes.
          - Returns the CST root.
        """
        parser = self.get_parser(source_file.language)
        tree = parser.parse(source_file.source_bytes)
        return tree


# ══════════════════════════════════════════════════════════════════════
# 4. TREE BUILDER & NODE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SyntaxNode:
    """A normalised syntax node extracted from the CST."""
    node_type: str            # e.g. "function", "class", "call"
    raw_type: str             # original Tree-sitter node type
    name: Optional[str]       # identifier name, if available
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    start_byte: int
    end_byte: int
    children: List["SyntaxNode"] = field(default_factory=list)
    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Recursive serialisation to a plain dictionary."""
        return {
            "type": self.node_type,
            "raw_type": self.raw_type,
            "name": self.name,
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class TreeBuilder:
    """
    Walks a Tree-sitter CST and builds a normalised SyntaxNode tree.
    """

    def __init__(self, config: LanguageConfig):
        self.config = config
        self.nt = config.node_types

        # Build reverse lookup: raw type → semantic category
        self._type_map: Dict[str, str] = {}
        for category, raw_types in self.nt.items():
            if category.endswith("_field"):
                continue
            if isinstance(raw_types, str):
                raw_types = [raw_types]
            for rt in raw_types:
                self._type_map[rt] = category

    # ── Public API ───────────────────────────────────────────────────

    def build(self, tree: Tree) -> SyntaxNode:
        """Build a SyntaxNode tree from a Tree-sitter Tree."""
        root = tree.root_node
        return self._convert_node(root, depth=0)

    # ── Internal conversion ──────────────────────────────────────────

    def _convert_node(self, node: Node, depth: int = 0) -> SyntaxNode:
        """
        Recursively convert a Tree-sitter Node into a SyntaxNode.

        Internal operations:
          - Determines semantic category via reverse type map.
          - Extracts identifier name from the 'name' field.
          - Recurses into named children only (skips punctuation tokens
            unless they carry semantic meaning).
        """
        raw_type = node.type
        category = self._type_map.get(raw_type, raw_type)

        # Extract name field if present
        name: Optional[str] = None
        name_field = self.nt.get("name_field")
        if name_field:
            name_node = node.child_by_field_name(name_field)
            if name_node and name_node.text is not None:
                name = name_node.text.decode("utf-8", errors="replace")

        # For Rust/C++ declarators, try harder to find a name
        if name is None and category in ("function", "class", "struct", "enum"):
            name = self._extract_name_fallback(node)

        # Collect metadata
        meta: Dict[str, Any] = {}
        if depth == 0:
            meta["is_root"] = True

        # Recurse into children (only named children for efficiency)
        children: List[SyntaxNode] = []
        for child in node.children:
            if child.is_named:
                child_sn = self._convert_node(child, depth + 1)
                # Only keep nodes that are either categorised or
                # have children themselves (structural nodes)
                if child_sn.node_type != child_sn.raw_type or child_sn.children:
                    children.append(child_sn)

        return SyntaxNode(
            node_type=category,
            raw_type=raw_type,
            name=name,
            start_point=(node.start_point.row, node.start_point.column),
            end_point=(node.end_point.row, node.end_point.column),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            children=children,
            metadata=meta,
        )

    def _extract_name_fallback(self, node: Node) -> Optional[str]:
        """
        Try to find an identifier name when the standard name field
        is absent (common in C++ declarators and Rust impl blocks).
        """
        # BFS through children looking for first identifier
        queue: List[Node] = [node]
        while queue:
            current = queue.pop(0)
            for child in current.children:
                if child.type == "identifier" and child.text:
                    return child.text.decode("utf-8", errors="replace")
                if child.is_named:
                    queue.append(child)
        return None


# ══════════════════════════════════════════════════════════════════════
# 5. HIGH-LEVEL SCANNER
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FileResult:
    path: str
    language: str
    root_node: Optional[SyntaxNode] = None
    error: Optional[str] = None


@dataclass
class ScanResult:
    root_path: str
    files: List[FileResult] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_path": self.root_path,
            "stats": self.stats,
            "files": [
                {
                    "path": f.path,
                    "language": f.language,
                    "error": f.error,
                    "root_node": f.root_node.to_dict() if f.root_node else None,
                }
                for f in self.files
            ],
        }


class RepositoryScanner:
    """
    High-level orchestrator:
      1. Walks the repository.
      2. Parses each source file.
      3. Builds a normalised syntax tree.
      4. Aggregates results and statistics.
      5. Handles fingerprinting and caching.
    """

    def __init__(self, exclude_dirs: Optional[Set[str]] = None):
        self.configs = build_language_configs()
        self.parser = TreeSitterParser(self.configs)
        
        # Define default exclusions, ensuring .cache is ignored
        if exclude_dirs is None:
            self.exclude_dirs = {
                ".git", ".venv", "venv", "__pycache__",
                "node_modules", "target", "build", "dist",
                ".mypy_cache", ".pytest_cache", ".tox", ".cache"
            }
        else:
            self.exclude_dirs = exclude_dirs

    def _hash_file_fast(self, path: Path) -> str:
        """Cheap fingerprint: path + size + mtime."""
        st = path.stat()
        h = hashlib.blake2b(digest_size=16)
        h.update(str(path).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
        return h.hexdigest()

    def _compute_repo_fingerprint(self, root: Path) -> Tuple[str, int]:
        """Return (aggregate_hash, file_count) based on tracked language extensions."""
        # Dynamically build tracked extensions from language configs
        tracked_exts = {
            ext for cfg in self.configs.values() for ext in cfg.extensions
        }

        entries = []
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() in tracked_exts:
                    rel = p.relative_to(root).as_posix()
                    entries.append(f"{rel}:{self._hash_file_fast(p)}")

        entries.sort()
        agg = hashlib.blake2b(digest_size=16)
        for e in entries:
            agg.update(e.encode())
            agg.update(b"\n")
        return agg.hexdigest(), len(entries)

    def scan(self, root_path: str | Path) -> ScanResult:
        """
        Scan a repository and return a ScanResult.
        """
        root = Path(root_path).resolve()
        result = ScanResult(root_path=str(root))

        lang_counts: Dict[str, int] = {}
        total_nodes = 0

        for source_file in walk_repository(root, self.configs, self.exclude_dirs):
            lang = source_file.language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            file_result = FileResult(
                path=str(source_file.path.relative_to(root)),
                language=lang,
            )

            try:
                tree = self.parser.parse_file(source_file)
                config = self.configs[lang]
                builder = TreeBuilder(config)
                file_result.root_node = builder.build(tree)
                total_nodes += self._count_nodes(file_result.root_node)

            except Exception as exc:
                file_result.error = f"{type(exc).__name__}: {exc}"

            result.files.append(file_result)

        result.stats = {
            "total_files": len(result.files),
            "files_by_language": lang_counts,
            "total_nodes": total_nodes,
            "errors": sum(1 for f in result.files if f.error),
        }

        return result

    def get_repo_tree(self, root_path: str | Path, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Return the full repo tree as a dict, managing the JSON cache internally.
        """
        root = Path(root_path).resolve()
        cache_dir = root / ".cache"
        cache_file = cache_dir / "repo_tree.json"
        cache_meta = cache_dir / "repo_tree.meta.json"

        t0 = time.time()
        fingerprint, file_count = self._compute_repo_fingerprint(root)

        # ── Attempt Cache Load ───────────────────────────────────────────
        if not force_refresh:
            if cache_file.exists() and cache_meta.exists():
                try:
                    meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                    if meta.get("fingerprint") == fingerprint:
                        tree = json.loads(cache_file.read_text(encoding="utf-8"))
                        print(f"[cache] HIT  ({file_count} files, loaded in {time.time() - t0:.3f}s)")
                        return tree
                    else:
                        print("[cache] MISS (repo changed since last scan)")
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[cache] unreadable, will rebuild ({exc})")
            else:
                print("[cache] MISS (no cache on disk)")
        else:
            print("[cache] forced refresh")

        # ── Build fresh ──────────────────────────────────────────────────
        result = self.scan(root)
        tree = result.to_dict()

        # ── Save cache ───────────────────────────────────────────────────
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "fingerprint": fingerprint,
            "file_count": file_count,
            "created_at": time.time(),
            "scanner_version": 1,
        }
        
        tmp_tree = cache_file.with_suffix(".json.tmp")
        tmp_meta = cache_meta.with_suffix(".json.tmp")
        tmp_tree.write_text(json.dumps(tree, indent=2), encoding="utf-8")
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp_tree.replace(cache_file)
        tmp_meta.replace(cache_meta)
        
        print(f"[cache] wrote {cache_file} ({cache_file.stat().st_size / 1024:.1f} KB)")
        print(f"[cache] scan + write done in {time.time() - t0:.3f}s "
              f"({result.stats.get('total_files', 0)} files, "
              f"{result.stats.get('total_nodes', 0)} nodes)")

        return tree

    @staticmethod
    def _count_nodes(node: SyntaxNode) -> int:
        """Recursively count nodes in a SyntaxNode tree."""
        count = 1
        for child in node.children:
            count += RepositoryScanner._count_nodes(child)
        return count


# ══════════════════════════════════════════════════════════════════════
# 6. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    import json

    arg_parser = argparse.ArgumentParser(
        description="Tree-sitter repository scanner for C++, Rust, and Python."
    )
    arg_parser.add_argument(
        "root",
        help="Path to the repository root.",
    )
    arg_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write JSON output to this file (default: stdout).",
    )
    arg_parser.add_argument(
        "--language", "-l",
        choices=["python", "cpp", "rust", "all"],
        default="all",
        help="Restrict scanning to a specific language.",
    )
    args = arg_parser.parse_args()

    scanner = RepositoryScanner()
    result = scanner.scan(args.root)

    # Filter by language if requested
    if args.language != "all":
        result.files = [f for f in result.files if f.language == args.language]

    output = json.dumps(result.to_dict(), indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()