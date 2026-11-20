"""CHANGELOG.md documents the current version: the release workflow publishes
that section as the GitHub release notes and refuses to release without it."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location("release_notes", ROOT / "tools" / "release_notes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_changelog_documents_the_current_version():
    tool = _tool()
    version = re.search(r'__version__ = "(.+)"', (ROOT / "remblais" / "__init__.py").read_text())[1]
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = tool.notes(text, f"v{version}", "keckj/remblais")
    assert notes.strip().endswith("...v" + version), notes[-200:]
    versions = [v for v, _ in tool.sections(text)]
    assert versions == sorted(versions, key=lambda v: tuple(map(int, v.split("."))), reverse=True)
