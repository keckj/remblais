"""MkDocs hooks for the remblais site.

- The home page (docs/index.md) is README.md, with its links rewritten:
  ``docs/x.md`` becomes ``x.md`` and ``docs/images/`` becomes ``images/``; the
  light and dark logos of the README follow the color scheme of the site.
- Links to files outside docs/ (licenses, sources) point to GitHub.
- A figure of tools/make_figures.py alone on its line (``![...](images/x.svg)``)
  is inlined, so that its colours follow the light/dark toggle of the site.
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
        # the logo follows the light or dark scheme of the site
        markdown = re.sub(
            r"<picture>.*?srcset=\"docs/([^\"]+)\".*?src=\"(?:docs/)?([^\"]+)\"([^>]*)>\s*</picture>",
            lambda m: f'<img alt="remblais" src="{m.group(2)}#only-light"{m.group(3)}><img alt="remblais" src="{m.group(1)}#only-dark"{m.group(3)}>',
            markdown, flags=re.S)
        for name in _ROOT_FILES:
            markdown = markdown.replace(f"({name})", f"({BLOB}{name})")
    # Links from docs/ to the sources: ../remblais/... -> GitHub
    markdown = re.sub(r"\(\.\./(remblais/[^)]*/)\)", lambda m: f"({TREE}{m.group(1)})", markdown)
    markdown = re.sub(r"\(\.\./(remblais/[^)]*)\)", lambda m: f"({BLOB}{m.group(1)})", markdown)
    return _inline_figures(markdown, config.docs_dir)


_FIGURE = re.compile(r"^!\[[^\]]*\]\((?:\.\./)*images/([\w-]+\.svg)\)[ \t]*$", re.M)


def _inline_figures(markdown, docs_dir):
    def inline(m):
        svg = (Path(docs_dir) / "images" / m.group(1)).read_text().strip()
        if 'class="kxfig"' not in svg:
            return m.group(0)
        return f'<figure class="kxfig-wrap">\n{svg}\n</figure>'
    return _FIGURE.sub(inline, markdown)
