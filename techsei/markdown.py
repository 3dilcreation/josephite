"""A small, dependency-free Markdown -> HTML renderer.

It supports the subset of Markdown that Tech SEI generates: headings,
paragraphs, blockquotes, ordered/unordered lists, bold/italic/code spans,
and links. This keeps the project lightweight (no extra Markdown library)
while producing clean, safe HTML.
"""

from __future__ import annotations

import html
import re


def _inline(text: str) -> str:
    """Render inline Markdown after HTML-escaping the raw text."""
    text = html.escape(text, quote=False)
    # Inline code (process first so its contents are not further formatted).
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Links [label](url)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" rel="noopener" target="_blank">\1</a>',
        text,
    )
    # Bold then italic.
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def render(md: str) -> str:
    """Convert a Markdown string to an HTML fragment."""
    lines = md.splitlines()
    html_parts: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Headings
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            html_parts.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        # Blockquote (consume consecutive '>' lines)
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            html_parts.append(f"<blockquote><p>{_inline(' '.join(quote_lines))}</p></blockquote>")
            continue

        # Unordered list
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            lis = "".join(f"<li>{_inline(it)}</li>" for it in items)
            html_parts.append(f"<ul>{lis}</ul>")
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            lis = "".join(f"<li>{_inline(it)}</li>" for it in items)
            html_parts.append(f"<ol>{lis}</ol>")
            continue

        # Paragraph (consume until blank line or block element)
        para_lines = []
        while i < n and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|>|[-*]\s|\d+\.\s)", lines[i].strip()
        ):
            para_lines.append(lines[i].strip())
            i += 1
        html_parts.append(f"<p>{_inline(' '.join(para_lines))}</p>")

    return "\n".join(html_parts)
