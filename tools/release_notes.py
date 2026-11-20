"""Release notes of a version, taken from CHANGELOG.md.

    python tools/release_notes.py v2.1.0 keckj/remblais > notes.md

Prints the body of the ``## 2.1.0 (date)`` section of CHANGELOG.md, followed
by a link comparing the tag with the previous version of the changelog.
Fails (exit status 1) when the section is missing or empty, so that a release
cannot be published without its changelog. Hard-wrapped lines are joined:
GitHub renders each line break of a release body as a break.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADING = re.compile(r"^## (\d+\.\d+\.\d+)\b.*$", re.M)


def sections(text: str) -> list[tuple[str, str]]:
    """(version, body) for each ``## X.Y.Z`` section, newest first."""
    marks = list(HEADING.finditer(text))
    return [(m[1], text[m.end():marks[k + 1].start() if k + 1 < len(marks) else len(text)].strip())
            for k, m in enumerate(marks)]


BLOCK = re.compile(r"^\s*([-*+] |\d+\. |#|\||```|>)")


def unwrap(text: str) -> str:
    """Join the continuation lines of paragraphs and list items."""
    out: list[str] = []
    fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append(line)
            continue
        joinable = (not fence and out and line.strip() and not BLOCK.match(line)
                    and out[-1].strip() and not out[-1].lstrip().startswith(("#", "|", "```")))
        if joinable:
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def notes(changelog: str, tag: str, repo: str) -> str:
    version = tag.removeprefix("v")
    secs = sections(changelog)
    versions = [v for v, _ in secs]
    if version not in versions:
        raise SystemExit(f"CHANGELOG.md has no section '## {version}': add one before tagging {tag}")
    k = versions.index(version)
    body = secs[k][1]
    if not body:
        raise SystemExit(f"CHANGELOG.md: the section of {version} is empty")
    if k + 1 < len(secs):
        link = f"https://github.com/{repo}/compare/v{versions[k + 1]}...{tag}"
    else:
        link = f"https://github.com/{repo}/commits/{tag}"
    return f"{unwrap(body)}\n\n**Full changelog**: {link}\n"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parent.parent
    sys.stdout.write(notes((root / "CHANGELOG.md").read_text(encoding="utf-8"), sys.argv[1], sys.argv[2]))
