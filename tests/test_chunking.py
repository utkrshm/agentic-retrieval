from pathlib import Path

import pytest

from coderet.chunking import (
    Unit,
    chunk_file,
    chunk_source,
    detect_language,
    embed_text,
    iter_source_files,
)

PY = '''\
"""Module doc."""
import os

CONST = 1


def top(a, b=2):
    """Add numbers."""
    def inner(x):
        return x + 1
    return inner(a) + b


class Greeter(Base):
    """Greets."""
    greeting = "hi"

    def __init__(self, name):
        self.name = name

    @staticmethod
    def make():
        return Greeter("x")


if __name__ == "__main__":
    print(top(1))
'''

JS = '''\
/**
 * Adds two numbers.
 */
function add(a, b) {
  return a + b;
}

const mul = (a, b) => a * b;

class Counter extends Base {
  constructor(start) {
    super();
    this.n = start;
  }

  inc() {
    this.n += 1;
  }
}

module.exports = function (RED) {
  RED.nodes.registerType("x", add);
};

Foo.prototype.bar = function (x) {
  return x;
};

export default function handler(req) {
  return req;
}

RED.start();
'''


def _by_qual(units: list[Unit]) -> dict[str, Unit]:
    return {u.qualname: u for u in units}


def _check_invariants(source: str, units: list[Unit]) -> None:
    lines = source.split("\n")
    assert units, "expected at least one unit"
    for u in units:
        assert 1 <= u.start_line <= u.end_line <= len(lines)
        assert u.text == "\n".join(lines[u.start_line - 1 : u.end_line])
        assert u.text.strip()
        assert u.ast_hash
    keys = [(u.start_line, u.end_line, u.part) for u in units]
    assert keys == sorted(keys)


def test_detect_language():
    assert detect_language("a/b.py") == "python"
    assert detect_language("a/b.mjs") == "javascript"
    assert detect_language("a/b.jsx") == "javascript"
    assert detect_language("a/b.rs") is None


def test_python_units():
    units = chunk_source("m.py", PY)
    _check_invariants(PY, units)
    got = {(u.kind, u.qualname) for u in units}
    assert ("function", "top") in got
    assert ("class", "Greeter") in got
    assert ("method", "Greeter.__init__") in got
    assert ("method", "Greeter.make") in got
    assert not any(u.qualname == "top.inner" for u in units), "nested function stays inside its parent"
    by = _by_qual(units)
    assert '"""Add numbers."""' in by["top"].text
    assert '"""Greets."""' in by["Greeter"].text
    assert by["Greeter"].end_line < by["Greeter.__init__"].start_line, "class header stops before first method"
    assert not by["Greeter"].text.endswith("\n"), "no trailing blank line in header"
    assert "@staticmethod" in by["Greeter.make"].text, "decorators belong to the method"
    assert by["top"].signature.startswith("def top(a, b=2)")
    modules = [u for u in units if u.kind == "module"]
    assert any("import os" in u.text for u in modules)
    assert any("__main__" in u.text for u in modules)


def test_javascript_units():
    units = chunk_source("m.js", JS)
    _check_invariants(JS, units)
    by = _by_qual(units)
    assert by["add"].kind == "function"
    assert by["add"].start_line == 1 and by["add"].text.startswith("/**"), "JSDoc is part of the unit span"
    assert by["mul"].kind == "function"
    assert "Counter" not in by, "bare class declaration header is dropped"
    assert by["Counter.constructor"].kind == "method"
    assert by["Counter.inc"].kind == "method"
    assert by["module.exports"].kind == "function"
    assert by["Foo.prototype.bar"].kind == "method"
    assert by["handler"].kind == "function" and by["handler"].text.startswith("export default")
    assert any(u.kind == "module" and "RED.start()" in u.text for u in units)


def test_oversized_function_is_split_and_nested_defs_surface():
    src = (
        "def big():\n"
        "    a = 1\n"
        "    def helper():\n"
        "        return 2\n"
        "    b = 3\n"
        "    c = a + b\n"
        "    return c\n"
    )
    units = chunk_source("m.py", src, max_chars=60)
    _check_invariants(src, units)
    assert any(u.qualname == "big.helper" for u in units)
    assert all(len(u.text) <= 60 or u.text.count("\n") == 0 for u in units)


def test_oversized_wrapper_expression_is_split_into_its_functions():
    """Old-style JS: one giant `X = (function () { ... })();` must not become blind line windows."""
    src = (
        "var RED = (function () {\n"
        "  function alpha(a) {\n    return a + 1;\n  }\n"
        "  function beta(b) {\n    return b * 2;\n  }\n"
        "  var shared = 42;\n"
        "  return { alpha: alpha, beta: beta };\n"
        "})();\n"
    )
    units = chunk_source("m.js", src, max_chars=90)
    _check_invariants(src, units)
    quals = {u.qualname for u in units}
    assert {"alpha", "beta"} <= quals
    assert all(len(u.text) <= 90 for u in units)
    joined = "\n".join(u.text for u in units)
    assert "var shared = 42;" in joined and "return { alpha" in joined


def test_oversized_python_block_is_split_by_syntax():
    src = "if True:\n" + "".join(f"    x{i} = {i}\n" for i in range(40))
    units = chunk_source("m.py", src, max_chars=120)
    _check_invariants(src, units)
    assert len(units) > 3
    assert all(u.text.count("\n") >= 0 and len(u.text) <= 120 for u in units)


def test_single_huge_statement_is_cut_into_windows():
    body = ",\n".join(f"  k{i}: {i}" for i in range(200))
    src = "module.exports = {\n" + body + "\n};\n"
    units = chunk_source("m.js", src, max_chars=300)
    _check_invariants(src, units)
    assert len(units) > 3
    assert all(len(u.text) <= 330 for u in units)


def test_ast_hash_ignores_comments_and_whitespace_but_not_renames():
    a = chunk_source("a.py", "def f(x):\n    return x + 1\n")[0]
    b = chunk_source("b.py", "def f(x):\n    # note\n    return   x+1\n")[0]
    c = chunk_source("c.py", "def f(y):\n    return y + 1\n")[0]
    assert a.ast_hash == b.ast_hash
    assert a.ast_hash != c.ast_hash
    ja = chunk_source("a.js", "function f(x) { return x + 1; }\n")[0]
    jb = chunk_source("b.js", "function f(x) {\n  /* c */ return x+1; }\n")[0]
    assert ja.ast_hash == jb.ast_hash


def test_broken_source_does_not_raise():
    units = chunk_source("x.py", "def f(:\n    pass\n\nclass A:\n    def g(self): return 1\n")
    assert isinstance(units, list)
    assert chunk_source("y.js", "function (( {{ \n") is not None


def test_empty_and_comment_only_files():
    assert chunk_source("e.py", "") == []
    assert chunk_source("c.js", "// just a license header\n// more\n") == []


def test_embed_text_has_location_name_and_code():
    units = chunk_source("src/m.js", JS)
    by = _by_qual(units)
    text = embed_text(by["add"])
    assert text.startswith("src/m.js\nfunction add\n/**")
    assert "Adds two numbers." in text and "return a + b;" in text
    py = _by_qual(chunk_source("m.py", PY))["top"]
    assert embed_text(py).count("Add numbers.") == 1


def test_bare_class_declaration_header_is_dropped_but_methods_keep_the_class_name():
    src = "class A:\n    def f(self):\n        return 1\n"
    units = chunk_source("m.py", src)
    assert [(u.kind, u.qualname) for u in units] == [("method", "A.f")]
    js = chunk_source("m.js", "class A {\n  f() { return 1; }\n}\n")
    assert [(u.kind, u.qualname) for u in js] == [("method", "A.f")]


def test_class_without_methods_is_kept_whole():
    units = chunk_source("m.py", "class Cfg:\n    a = 1\n    b = 2\n")
    assert [(u.kind, u.qualname) for u in units] == [("class", "Cfg")]
    assert "b = 2" in units[0].text


def test_inline_callbacks_and_object_methods_are_not_lost():
    src = (
        "app.get('/x', (req, res) => {\n  res.send('x');\n});\n\n"
        "const handlers = {\n  open() { return 1; },\n  close: () => 2,\n};\n"
    )
    units = chunk_source("m.js", src)
    _check_invariants(src, units)
    joined = "\n".join(u.text for u in units)
    assert "res.send('x')" in joined and "open()" in joined and "close:" in joined


def test_iter_source_files_skips_vendor_minified_and_unknown(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.js").write_text("function a() {}\n")
    (tmp_path / "src" / "b.py").write_text("def b():\n    pass\n")
    (tmp_path / "src" / "notes.txt").write_text("x")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "i.js").write_text("function i() {}\n")
    (tmp_path / "src" / "app.min.js").write_text("function m(){}\n")
    (tmp_path / "src" / "wide.js").write_text("var x=" + "1+" * 3000 + "1;\n")
    found = sorted(p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path))
    assert found == ["src/a.js", "src/b.py"]


def test_git_checkout_uses_tracked_files_even_under_node_modules(tmp_path: Path):
    import subprocess

    (tmp_path / "packages" / "node_modules" / "@x").mkdir(parents=True)
    (tmp_path / "packages" / "node_modules" / "@x" / "a.js").write_text("function a() {}\n")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "o.js").write_text("function o() {}\n")
    (tmp_path / "untracked.js").write_text("function u() {}\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "packages", "dist"], cwd=tmp_path, check=True)
    found = sorted(p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path))
    assert found == ["packages/node_modules/@x/a.js"], "tracked source kept, dist skipped, untracked ignored"


def test_chunk_file_reports_relative_path_and_errors(tmp_path: Path):
    f = tmp_path / "pkg" / "m.py"
    f.parent.mkdir()
    f.write_text("def ok():\n    return 1\n")
    units, had_error = chunk_file(f, tmp_path)
    assert not had_error and units[0].path == "pkg/m.py"
    f.write_text("def bad(:\n")
    _, had_error = chunk_file(f, tmp_path)
    assert had_error


@pytest.mark.parametrize("max_chars", [80, 200, 4000])
def test_invariants_hold_at_any_size_limit(max_chars: int):
    for name, src in (("m.py", PY), ("m.js", JS)):
        _check_invariants(src, chunk_source(name, src, max_chars=max_chars))


# ---- TypeScript and tiny-fragment merge ---------------------------------------------------

TS_SRC = """\
import { x } from "./x";

export interface Opts { a: number }

export abstract class Repo {
  abstract load(id: string): Promise<void>;
  save(id: string): void {
    console.log(id);
  }
}

export const run = async (n: number): Promise<number> => {
  return n + 1;
};
"""


def test_typescript_units_and_tsx():
    units = chunk_source("a/repo.ts", TS_SRC)
    names = {u.qualname for u in units}
    assert {"Repo.save", "run"} <= names
    assert all(1 <= u.start_line <= u.end_line for u in units)
    tsx = chunk_source("a/view.tsx", "export const View = () => <div>hi</div>;\n")
    assert [u.qualname for u in tsx] == ["View"]


MERGE_SRC = "var a;\n\nfunction f() {\n  return 1;\n}\n\nvar b;\n\nfunction g() {\n  return 2;\n}\n\nregister(g);\n"


def test_tiny_groups_merge_into_neighbour_only_when_contiguous():
    plain = chunk_source("m.js", MERGE_SRC)
    merged = chunk_source("m.js", MERGE_SRC, min_chars=20)
    assert len(merged) < len(plain)
    # no unit text is lost and each unit is exactly the lines of its span
    lines = MERGE_SRC.split("\n")
    for u in merged:
        assert u.text == "\n".join(lines[u.start_line - 1 : u.end_line])
    covered = {ln for u in merged for ln in range(u.start_line, u.end_line + 1)}
    assert {ln for ln, s in enumerate(lines, 1) if s.strip()} <= covered
    assert "f" in {u.qualname for u in merged} and "g" in {u.qualname for u in merged}


def test_merge_respects_max_chars_and_is_off_by_default():
    assert chunk_source("m.js", MERGE_SRC) == chunk_source("m.js", MERGE_SRC, min_chars=0)
    capped = chunk_source("m.js", MERGE_SRC, max_chars=30, min_chars=20)
    assert all(len(u.text) <= 30 or u.start_line == u.end_line for u in capped)



def test_declaration_files_are_not_indexed(tmp_path):
    (tmp_path / "types.d.ts").write_text("declare function f(): void;\n")
    (tmp_path / "real.ts").write_text("export function f(): void {}\n")
    assert [p.name for p in iter_source_files(tmp_path)] == ["real.ts"]
