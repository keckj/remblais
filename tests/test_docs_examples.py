"""Runs the ```python blocks of the documentation pages (docs/**/*.md).

The blocks of one page run in order, in a single namespace, on the CPU: the
line ``device = "cuda"`` that opens the GPU examples is replaced by
``device = "cpu"``, where remblais computes with the dense PyTorch backend
and the kernox CPU backend. The examples themselves are left unchanged, so they
read exactly as a user would type them.

A block preceded by the comment ``<!-- skip-test -->`` is not run (GeomLoss
imports, GPU-only timing code). The home page is README.md, which has no
code example.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")

DOCS = Path(__file__).resolve().parents[1] / "docs"
_BLOCK = re.compile(r"(<!-- skip-test -->\s*\n)?```python\n(.*?)```", flags=re.S)
_DEVICE = re.compile(r'^(\s*)device = "(?:cuda|xpu)"', flags=re.M)


def _pages():
    return sorted(p for p in DOCS.rglob("*.md") if p.name != "index.md")


def _blocks(page):
    return [(m.group(2), bool(m.group(1))) for m in _BLOCK.finditer(page.read_text())]


def test_docs_have_examples():
    runnable = [p for p in _pages() if any(not skip for _, skip in _blocks(p))]
    assert len(runnable) >= 8, runnable


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_page_examples_run(page, monkeypatch, tmp_path):
    blocks = [code for code, skip in _blocks(page) if not skip]
    if not blocks:
        pytest.skip("no runnable example on this page")
    monkeypatch.setenv("KERNOX_CACHE_DIR", str(tmp_path))  # compiled plans stay sandboxed
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "__docs__"}
    for k, code in enumerate(blocks):
        code = _DEVICE.sub(r'\1device = "cpu"', code)
        exec(compile(code, f"{page.name} (block {k + 1})", "exec"), ns)  # noqa: S102 - repository examples
