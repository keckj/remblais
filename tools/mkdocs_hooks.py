"""MkDocs hooks for the remblais site.

- The home page (docs/index.md) is README.md, with its links rewritten:
  ``docs/x.md`` becomes ``x.md`` and ``docs/images/`` becomes ``images/``.
- Links to files outside docs/ (licenses, sources) point to GitHub.
"""

import re
from pathlib import Path

BLOB = "https://github.com/keckj/remblais/blob/main/"
TREE = "https://github.com/keckj/remblais/tree/main/"
_ROOT_FILES = ("LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY.md")


def on_page_markdown(markdown, page, config, **kwargs):
    if page.file.src_uri == "index.md":
        markdown = (Path(config.config_file_path).parent / "README.md").read_text()
        markdown = re.sub(r"\((?:\./)?docs/", "(", markdown).replace('src="docs/', 'src="')
        for name in _ROOT_FILES:
            markdown = markdown.replace(f"({name})", f"({BLOB}{name})")
    # Links from docs/ to the sources: ../remblais/... -> GitHub
    markdown = re.sub(r"\(\.\./(remblais/[^)]*/)\)", lambda m: f"({TREE}{m.group(1)})", markdown)
    markdown = re.sub(r"\(\.\./(remblais/[^)]*)\)", lambda m: f"({BLOB}{m.group(1)})", markdown)
    return markdown
