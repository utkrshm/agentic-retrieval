"""Split source files into retrieval units (functions, methods, class headers, module code).

One parser framework (tree-sitter) for every language; per language there is only a
small "classify this node" function. Supported: Python and JavaScript.

A unit is the thing we embed and return: an exact file:line span, a name, a signature,
the source text, and a hash of its token stream with comments and whitespace removed
(``ast_hash``: used only to group "same code" across versions, never as a cache key).

Rules:
- Functions and methods are units. Comments directly above a definition (JSDoc, ``#``)
  are part of its span. Functions nested inside a function stay inside it, unless the
  outer function is larger than ``max_chars``; then it is split into its nested
  functions plus groups of consecutive statements.
- A class contributes a header unit (class line, docstring, attributes before the
  first method) unless that header is just the bare declaration line, plus one unit
  per method.
- Remaining top-level statements are grouped into "module" units, so scripts and
  registration code (``RED.nodes.registerType(...)``) are searchable too.
- A single statement larger than ``max_chars`` is cut into line windows.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

import tree_sitter_javascript as _tsjs
import tree_sitter_python as _tspy
import tree_sitter_typescript as _tsts
from tree_sitter import Language, Node, Parser

# jina-code was trained on sequences of 512 tokens (arXiv 2508.21290), about 2,000-2,500 chars of code.
MAX_CHARS = 2500
MAX_FILE_BYTES = 1_000_000
EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
}
# Directories that hold generated or vendored code. ``node_modules`` is only skipped when
# walking a directory that is not a git checkout: in a checkout, tracked files are
# intentional (Node-RED keeps its real source in packages/node_modules/@node-red/).
SKIP_DIRS = {
    ".git", "dist", "build", "vendor", ".venv", "venv", "__pycache__", "coverage", ".next",
    ".cache", "bower_components", ".tox", "site-packages", "generated", "__generated__",
}
SKIP_DIRS_UNTRACKED = SKIP_DIRS | {"node_modules"}

_BLOCK_TYPES = {"block", "statement_block", "class_body"}
_JS_FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
_JS_FUNC_DECLS = {"function_declaration", "generator_function_declaration"}


@dataclass(frozen=True)
class Unit:
    language: str
    path: str  # repo-relative, posix separators
    kind: str  # function | method | class | module
    name: str
    qualname: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    signature: str
    text: str  # exact source lines start_line..end_line
    ast_hash: str  # token hash without comments/whitespace; groups "same code" across versions
    part: int = 0  # >0 when this is one piece of an oversized definition
    group: bool = False  # True for a run of statements between definitions (not a definition itself)


@dataclass
class _Def:
    node: Node  # outermost node covering the definition (incl. decorators / export / declaration)
    kind: str  # function | method | class
    name: str
    qualname: str
    body: Node | None
    members: list[Node] = field(default_factory=list)  # class members


def detect_language(path: str | Path) -> str | None:
    return EXTENSIONS.get(Path(path).suffix.lower())


@cache
def _parser(language: str) -> Parser:
    if language == "python":
        return Parser(Language(_tspy.language()))
    if language == "javascript":
        return Parser(Language(_tsjs.language()))
    if language == "typescript":
        return Parser(Language(_tsts.language_typescript()))
    if language == "tsx":
        return Parser(Language(_tsts.language_tsx()))
    raise ValueError(f"unsupported language: {language}")


def _tokens_hash(nodes: list[Node], end_byte: int | None = None) -> str:
    """Hash of leaf tokens (type + text), ignoring comments and whitespace."""
    h = hashlib.sha256()
    stack = list(reversed(nodes))
    while stack:
        n = stack.pop()
        if n.type == "comment":
            continue
        if end_byte is not None and n.start_byte >= end_byte:
            continue
        if n.child_count == 0:
            h.update(n.type.encode())
            h.update(b"\x00")
            h.update(n.text or b"")
            h.update(b"\x01")
        else:
            stack.extend(reversed(n.children))
    return h.hexdigest()[:16]


def _text_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).encode()).hexdigest()[:16]


def _collapse(s: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", s).strip()[:limit]


class _Extractor:
    def __init__(self, path: str, data: bytes, language: str, max_chars: int, min_chars: int = 0) -> None:
        self.path = path
        self.data = data
        self.language = language
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.lines = data.decode("utf-8", errors="replace").split("\n")

    # ---- spans and text -------------------------------------------------
    def _span(self, node: Node) -> tuple[int, int]:
        start = node.start_point.row + 1
        end_row = node.end_point.row
        if node.end_point.column == 0 and end_row > node.start_point.row:
            end_row -= 1
        return start, end_row + 1

    def _text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1 : end])

    def _node_text(self, node: Node | None) -> str:
        return node.text.decode("utf-8", errors="replace") if node is not None and node.text else ""

    def _signature(self, d: _Def) -> str:
        if d.body is None:
            return _collapse(self._node_text(d.node))
        raw = self.data[d.node.start_byte : d.body.start_byte].decode("utf-8", errors="replace")
        return _collapse(raw)

    def _start_with_comments(self, node: Node, leading: list[Node]) -> int:
        start = self._span(node)[0]
        return min(start, self._span(leading[0])[0]) if leading else start

    # ---- classification (per language) ----------------------------------
    def _classify(self, node: Node, parent_qual: str, in_class: bool) -> _Def | None:
        if self.language == "python":
            return self._classify_python(node, parent_qual, in_class)
        return self._classify_js(node, parent_qual, in_class)

    def _qual(self, parent_qual: str, name: str) -> str:
        return f"{parent_qual}.{name}" if parent_qual else name

    def _classify_python(self, node: Node, parent_qual: str, in_class: bool) -> _Def | None:
        target = node
        if node.type == "decorated_definition":
            target = node.child_by_field_name("definition") or node
        if target.type == "function_definition":
            name = self._node_text(target.child_by_field_name("name"))
            kind = "method" if in_class else "function"
            return _Def(node, kind, name, self._qual(parent_qual, name), target.child_by_field_name("body"))
        if target.type == "class_definition":
            name = self._node_text(target.child_by_field_name("name"))
            body = target.child_by_field_name("body")
            members = list(body.named_children) if body is not None else []
            return _Def(node, "class", name, self._qual(parent_qual, name), body, members)
        return None

    def _classify_js(self, node: Node, parent_qual: str, in_class: bool) -> _Def | None:
        t = node.type
        if t == "export_statement":
            inner = node.child_by_field_name("declaration") or node.child_by_field_name("value")
            if inner is None:
                return None
            d = self._classify_js(inner, parent_qual, in_class)
            if d is None and inner.type in _JS_FUNC_VALUES:
                body = inner.child_by_field_name("body")
                name = self._node_text(inner.child_by_field_name("name")) or "default"
                return _Def(node, "function", name, self._qual(parent_qual, name), body)
            if d is not None:
                d.node = node
            return d
        if t in _JS_FUNC_DECLS:
            name = self._node_text(node.child_by_field_name("name"))
            return _Def(node, "function", name, self._qual(parent_qual, name), node.child_by_field_name("body"))
        if t in {"class_declaration", "abstract_class_declaration"}:
            name = self._node_text(node.child_by_field_name("name"))
            body = node.child_by_field_name("body")
            members = list(body.named_children) if body is not None else []
            return _Def(node, "class", name, self._qual(parent_qual, name), body, members)
        if t == "method_definition":
            name = self._node_text(node.child_by_field_name("name"))
            return _Def(node, "method", name, self._qual(parent_qual, name), node.child_by_field_name("body"))
        if t in {"field_definition", "public_field_definition"} and in_class:
            value = node.child_by_field_name("value")
            if value is not None and value.type in _JS_FUNC_VALUES:
                name = self._node_text(node.child_by_field_name("property") or node.child_by_field_name("name"))
                return _Def(node, "method", name, self._qual(parent_qual, name), value.child_by_field_name("body"))
            return None
        if t in {"lexical_declaration", "variable_declaration"}:
            decls = [c for c in node.named_children if c.type == "variable_declarator"]
            if len(decls) == 1:
                value = decls[0].child_by_field_name("value")
                if value is not None and value.type in _JS_FUNC_VALUES:
                    name = self._node_text(decls[0].child_by_field_name("name"))
                    return _Def(node, "function", name, self._qual(parent_qual, name), value.child_by_field_name("body"))
            return None
        if t == "expression_statement" and node.named_child_count == 1:
            expr = node.named_children[0]
            if expr.type == "assignment_expression":
                left, right = expr.child_by_field_name("left"), expr.child_by_field_name("right")
                if right is not None and right.type in _JS_FUNC_VALUES and left is not None:
                    left_text = re.sub(r"\s+", "", self._node_text(left))
                    name = left_text.split(".")[-1]
                    kind = "method" if ".prototype." in left_text else "function"
                    return _Def(node, kind, name, self._qual(parent_qual, left_text), right.child_by_field_name("body"))
            return None
        return None

    # ---- emission -------------------------------------------------------
    def _unit(self, nodes: list[Node], start: int, end: int, kind: str, name: str, qual: str,
              sig: str, part: int, end_byte: int | None = None, group: bool = False) -> Unit:
        text = self._text(start, end)
        h = _tokens_hash(nodes, end_byte) if nodes else _text_hash(text)
        return Unit(self.language, self.path, kind, name, qual, start, end, sig, text, h, part, group)

    def _windows(self, start: int, end: int, kind: str, name: str, qual: str, sig: str,
                 part0: int) -> list[Unit]:
        out, i, part = [], start, part0
        while i <= end:
            j, size = i, 0
            while j <= end and (size + len(self.lines[j - 1]) + 1 <= self.max_chars or j == i):
                size += len(self.lines[j - 1]) + 1
                j += 1
            seg_end = j - 1
            if self._text(i, seg_end).strip():
                out.append(self._unit([], i, seg_end, kind, name, qual, sig, part))
            part += 1
            i = j
        return out

    def _atoms(self, node: Node) -> list[Node] | None:
        """Pieces of an oversized node, found by descending the syntax tree until each is small.

        Returns None when the node has no named children (nothing to descend into).
        """
        kids = list(node.named_children)
        if not kids:
            return None
        atoms: list[Node] = []
        for k in kids:
            s, e = self._span(k)
            if len(self._text(s, e)) > self.max_chars:
                sub = self._atoms(k)
                atoms.extend(sub if sub is not None else [k])
            else:
                atoms.append(k)
        return atoms

    def _flush(self, pending: list[Node], kind: str, name: str, qual: str, sig: str,
               part: int) -> tuple[list[Unit], int]:
        """Group consecutive uncovered statements into units of at most max_chars."""
        out: list[Unit] = []
        group: list[Node] = []

        def emit() -> None:
            nonlocal part
            if group and not all(n.type == "comment" for n in group):
                s, _ = self._span(group[0])
                _, e = self._span(group[-1])
                out.append(self._unit(list(group), s, e, kind, name, qual, sig, part, group=True))
                part += 1
            group.clear()

        for n in pending:
            s, e = self._span(n)
            if len(self._text(s, e)) > self.max_chars:
                emit()
                atoms = self._atoms(n)
                if atoms is not None:
                    # descend into the syntax tree (wrapper functions, IIFEs, big blocks)
                    out.extend(self._scope(atoms, qual if qual != name else "", False, kind, name, sig))
                else:  # an indivisible leaf (huge string or regex): cut by lines
                    for u in self._windows(s, e, kind, name, qual, sig, part):
                        out.append(u)
                        part = u.part + 1
                continue
            if group:
                gs, _ = self._span(group[0])
                if len(self._text(gs, e)) > self.max_chars:
                    emit()
            group.append(n)
        emit()
        return out, part

    def _emit_def(self, d: _Def, leading: list[Node]) -> list[Unit]:
        start = self._start_with_comments(d.node, leading)
        end = self._span(d.node)[1]
        sig = self._signature(d)
        if d.kind == "class":
            return self._emit_class(d, start, end, sig)
        if len(self._text(start, end)) <= self.max_chars:
            return [self._unit([d.node], start, end, d.kind, d.name, d.qualname, sig, 0)]
        # oversized function: split into nested definitions + statement groups
        if d.body is not None and d.body.type in _BLOCK_TYPES:
            return self._scope(list(d.body.named_children), d.qualname, False, d.kind, d.name, sig)
        return self._windows(start, end, d.kind, d.name, d.qualname, sig, 0)

    def _emit_class(self, d: _Def, start: int, end: int, sig: str) -> list[Unit]:
        out: list[Unit] = []
        method_nodes = [m for m in d.members if self._classify(m, d.qualname, True) is not None]
        if method_nodes:
            header_end = self._span(method_nodes[0])[0] - 1
            end_byte = method_nodes[0].start_byte
        else:
            header_end, end_byte = end, None
        header_end = max(header_end, start)
        while header_end > start and not self.lines[header_end - 1].strip():
            header_end -= 1
        non_blank = sum(1 for ln in self.lines[start - 1 : header_end] if ln.strip())
        if method_nodes and non_blank <= 1:
            pass  # bare "class X:" line: the methods already carry the class name in their qualname
        elif len(self._text(start, header_end)) <= self.max_chars:
            out.append(self._unit([d.node], start, header_end, "class", d.name, d.qualname, sig, 0, end_byte))
        else:
            out.extend(self._windows(start, header_end, "class", d.name, d.qualname, sig, 0))
        out.extend(self._scope(d.members, d.qualname, True, "class", d.name, sig))
        return out

    def _scope(self, nodes: list[Node], parent_qual: str, in_class: bool, wrap_kind: str,
               wrap_name: str, wrap_sig: str) -> list[Unit]:
        """Units for a sequence of sibling statements (module, class body or function body)."""
        defs = [self._classify(n, parent_qual, in_class) for n in nodes]
        # comments directly above a definition belong to it (JSDoc, "# note")
        leading: dict[int, list[Node]] = {}
        consumed: set[int] = set()
        for i, d in enumerate(defs):
            if d is None:
                continue
            j, chain = i - 1, []
            nxt_row = nodes[i].start_point.row
            while j >= 0 and nodes[j].type == "comment" and nxt_row - nodes[j].end_point.row <= 1:
                chain.append(nodes[j])
                nxt_row = nodes[j].start_point.row
                j -= 1
            chain.reverse()
            leading[i] = chain
            consumed.update(c.start_byte for c in chain)

        out: list[Unit] = []
        pending: list[Node] = []
        part = 0
        for i, n in enumerate(nodes):
            if n.start_byte in consumed:
                continue
            d = defs[i]
            if d is None:
                if not in_class:
                    pending.append(n)
                continue
            flushed, part = self._flush(pending, wrap_kind, wrap_name, parent_qual or wrap_name, wrap_sig, part)
            out.extend(flushed)
            pending = []
            out.extend(self._emit_def(d, leading.get(i, [])))
        flushed, part = self._flush(pending, wrap_kind, wrap_name, parent_qual or wrap_name, wrap_sig, part)
        out.extend(flushed)
        return out

    def _merge_tiny(self, units: list[Unit]) -> list[Unit]:
        """Fold statement groups with under min_chars non-blank characters into a neighbour.

        A tiny group (``var x;``, ``})();``) carries almost no meaning, but its unit text still starts
        with the file path and qualified name, so it can match a query on the name alone. It is merged
        into the previous unit, else the next, only when the two are contiguous (nothing but blank
        lines between them) and the merged text stays within max_chars. The neighbour keeps its kind,
        name and part; only its span grows.
        """
        if self.min_chars <= 0:
            return units
        out: list[Unit] = []
        i = 0
        while i < len(units):
            u = units[i]
            tiny = u.group and sum(1 for c in u.text if not c.isspace()) < self.min_chars
            if not tiny:
                out.append(u)
                i += 1
                continue
            prev = out[-1] if out else None
            nxt = units[i + 1] if i + 1 < len(units) else None
            if prev is not None and self._contiguous(prev.end_line, u.start_line) \
                    and len(self._text(prev.start_line, u.end_line)) <= self.max_chars:
                text = self._text(prev.start_line, u.end_line)
                out[-1] = replace(prev, end_line=u.end_line, text=text, ast_hash=_text_hash(text))
            elif nxt is not None and self._contiguous(u.end_line, nxt.start_line) \
                    and len(self._text(u.start_line, nxt.end_line)) <= self.max_chars:
                text = self._text(u.start_line, nxt.end_line)
                units[i + 1] = replace(nxt, start_line=u.start_line, text=text, ast_hash=_text_hash(text))
            else:
                out.append(u)
            i += 1
        return out

    def _contiguous(self, end_line: int, start_line: int) -> bool:
        return start_line > end_line and all(not ln.strip() for ln in self.lines[end_line : start_line - 1])

    def run(self, root: Node) -> list[Unit]:
        units = self._scope(list(root.named_children), "", False, "module", "", "")
        units.sort(key=lambda u: (u.start_line, u.end_line, u.part))
        return self._merge_tiny(units)


def chunk_source(path: str, source: str | bytes, language: str | None = None,
                 max_chars: int = MAX_CHARS, min_chars: int = 0) -> list[Unit]:
    lang = language or detect_language(path)
    if lang is None:
        raise ValueError(f"cannot detect language for {path!r}")
    data = source.encode("utf-8") if isinstance(source, str) else source
    tree = _parser(lang).parse(data)
    return _Extractor(path, data, lang, max_chars, min_chars).run(tree.root_node)


def chunk_file(path: Path, root: Path, max_chars: int = MAX_CHARS,
               min_chars: int = 0) -> tuple[list[Unit], bool]:
    """Chunk one file. Returns (units, parse_had_errors)."""
    data = path.read_bytes()
    lang = detect_language(path)
    if lang is None:
        return [], False
    tree = _parser(lang).parse(data)
    rel = path.relative_to(root).as_posix()
    return _Extractor(rel, data, lang, max_chars, min_chars).run(tree.root_node), tree.root_node.has_error


def _looks_minified(path: Path) -> bool:
    if path.name.endswith((".min.js", ".bundle.js")):
        return True
    with path.open("rb") as f:
        head = f.read(4096)
    lines = head.split(b"\n")
    return len(head) > 2000 and max(len(x) for x in lines) > 1000


def _git_tracked(root: Path) -> list[Path] | None:
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / p for p in out.decode("utf-8", errors="replace").split("\0") if p]


def _wanted(p: Path, root: Path, exts: set[str], skip: set[str]) -> bool:
    if p.suffix.lower() not in exts or not p.is_file() or p.is_symlink():
        return False
    if p.name.endswith((".d.ts", ".d.mts", ".d.cts")):  # type declarations: no implementation to find
        return False
    if any(part in skip for part in p.relative_to(root).parts[:-1]):
        return False
    return p.stat().st_size <= MAX_FILE_BYTES and not _looks_minified(p)


def iter_source_files(root: Path, extensions: set[str] | None = None) -> Iterator[Path]:
    """Source files to index. In a git checkout: the tracked files. Otherwise: a directory walk."""
    exts = extensions or set(EXTENSIONS)
    tracked = _git_tracked(root)
    if tracked is not None:
        for p in sorted(tracked):
            if _wanted(p, root, exts, SKIP_DIRS):
                yield p
        return
    stack = [root]
    while stack:
        d = stack.pop()
        for p in sorted(d.iterdir()):
            if p.is_dir():
                if p.name not in SKIP_DIRS_UNTRACKED and not p.is_symlink():
                    stack.append(p)
            elif _wanted(p, root, exts, SKIP_DIRS_UNTRACKED):
                yield p


def embed_text(u: Unit, signature_header: bool = False) -> str:
    """Text that gets embedded for a unit: where it is, what it is, then the code.

    With ``signature_header`` a statement group cut out of a larger definition also carries that
    definition's signature, which is otherwise only visible in the first piece.
    """
    head = f"{u.path}\n{u.kind} {u.qualname}".strip()
    if signature_header and u.group and u.signature:
        head += f"\n{u.signature}"
    return f"{head}\n{u.text}"
