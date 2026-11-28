"""Rewrite the relative links of README.md into absolute URLs, for PyPI.

PyPI renders the README without the repository around it, so relative links
and images break there. The release workflow runs this script on a clean
checkout before building the distributions:

- ``docs/x.md`` and ``docs/guides/x.md`` point to the documentation site;
- images (``src``, ``srcset``, Markdown images) point to the raw file on
  GitHub, at the release tag;
- anchors of the README point to the README on GitHub;
- files at the repository root (licenses) point to GitHub.

Usage: ``python tools/pypi_readme.py [REF]`` (default ``main``).
"""

import re
import sys
from pathlib import Path

SITE = "https://keckj.github.io/remblais/"
REPO = "keckj/remblais"


def rewrite(text: str, ref: str = "main") -> str:
    raw = f"https://raw.githubusercontent.com/{REPO}/{ref}/"
    blob = f"https://github.com/{REPO}/blob/{ref}/"
    # Images of HTML tags (<img src>, <source srcset>) and of Markdown
    text = re.sub(r'\b(src|srcset)="(?!https?:)([^"]+)"', lambda m: f'{m.group(1)}="{raw}{m.group(2)}"', text)
    text = re.sub(r"(!\[[^\]]*\])\((?!https?:)([^)]+)\)", lambda m: f"{m.group(1)}({raw}{m.group(2)})", text)
    text = re.sub(r"\]\(docs/([^)#]+?)\.md(#[^)]*)?\)",
                  lambda m: f"]({SITE}{m.group(1)}/{m.group(2) or ''})", text)
    # PyPI gives no anchors to the headings: point to the README on GitHub
    text = re.sub(r"\]\((#[^)]+)\)", lambda m: f"]({blob}README.md{m.group(1)})", text)
    text = re.sub(r"\]\((?!https?:|#|mailto:)([^)]+)\)", lambda m: f"]({blob}{m.group(1)})", text)
    return text


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "README.md"
    path.write_text(rewrite(path.read_text(), sys.argv[1] if len(sys.argv) > 1 else "main"))
