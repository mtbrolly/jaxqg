"""Verify relative Markdown links point to files in the source tree."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / "README.md"]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def test_local_markdown_links_resolve():
    missing = []
    for source in MARKDOWN:
        for target in LINK.findall(source.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (source.parent / target).exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken local links:\n" + "\n".join(missing)
