#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert common LaTeX Portuguese accent escapes to UTF-8 in .tex narrative text.

Usage:
  python methods_comparison/scripts/latex_pt_utf8_convert.py FILE.tex ...

Safe for methods_comparison reports with inputenc utf8 + babel brazil.
Does not modify \\includegraphics paths or \\label{...} content (no accents there).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Longer patterns first
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\\c{c}\\~oes", "ções"),
    (r"\\c{c}\\~ao", "ção"),
    (r"\\c{c}\\~a", "çã"),
    (r"\\c{c}", "ç"),
    (r"\\~oes", "ões"),
    (r"\\~ao", "ão"),
    (r"\\~a", "ã"),
    (r"\\~o", "õ"),
    (r"\\\^\{e\}", "ê"),
    (r"\\\^\{o\}", "ô"),
    (r"\\\^\{a\}", "â"),
    (r"\\\^e", "ê"),
    (r"\\\^o", "ô"),
    (r"\\\^a", "â"),
    (r"\\'e", "é"),
    (r"\\'a", "á"),
    (r"\\'i", "í"),
    (r"\\'o", "ó"),
    (r"\\'u", "ú"),
    (r"\\`a", "à"),
    (r"\\`o", "ò"),
)


def convert_tex(content: str) -> str:
    """Apply accent replacements to full file content."""
    out = content
    for pattern, repl in REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: latex_pt_utf8_convert.py FILE.tex [FILE2.tex ...]", file=sys.stderr)
        return 1
    for arg in argv[1:]:
        path = Path(arg)
        text = path.read_text(encoding="utf-8")
        converted = convert_tex(text)
        if converted != text:
            path.write_text(converted, encoding="utf-8")
            print("OK converted {}".format(path))
        else:
            print("SKIP no changes {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
