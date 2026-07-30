"""Article generation for Tech SEI.

Turns a raw :class:`~techsei.sources.Article` into a finished blog ``Post``.

Two strategies are supported:

* **ai** — calls the Claude API to write an original, well-structured article
  that contextualises the source story. Requires ``ANTHROPIC_API_KEY``.
* **extractive** — builds a clean post from the feed summary with no network
  dependency beyond what was already fetched. Always available.
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime

from .config import GenerationConfig
from .sources import Article


@dataclass
class Post:
    title: str
    slug: str
    category: str
    source: str
    source_url: str
    published: datetime
    body_md: str
    summary: str
    generated_by: str  # "ai" or "extractive"


def slugify(text: str, uid: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    base = base[:60].strip("-") or "post"
    return f"{base}-{uid}"


SYSTEM_PROMPT = (
    "You are a sharp, accurate technology journalist writing for 'Tech SEI', a "
    "blog covering tech news, innovations, and future research. Write clear, "
    "engaging articles in Markdown. Be factual and grounded in the provided "
    "source material; never invent quotes, statistics, or specifics that are "
    "not supported by the source. Add useful context and explain why the story "
    "matters. Do not include a top-level H1 title (it is added separately)."
)


def _build_user_prompt(article: Article) -> str:
    authors = ", ".join(article.authors) if article.authors else "unknown"
    return textwrap.dedent(
        f"""\
        Write a ~400-600 word blog article based on this source.

        Category: {article.category}
        Headline: {article.title}
        Source: {article.source}
        Authors: {authors}
        Original URL: {article.link}

        Source summary / abstract:
        \"\"\"
        {article.summary or "(no summary available)"}
        \"\"\"

        Requirements:
        - Start with a compelling 1-2 sentence hook.
        - Use 2-4 short sections with `##` subheadings.
        - Explain the significance and likely impact.
        - End with a short "Why it matters" takeaway.
        - Markdown only, no H1.
        """
    )


def _generate_ai(article: Article, cfg: GenerationConfig) -> str | None:
    """Return AI-written Markdown body, or None if the API is unavailable."""
    try:
        import anthropic
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(article)}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        body = "\n".join(parts).strip()
        return body or None
    except Exception as exc:  # API/network errors should fall back, not crash
        print(f"  ! AI generation failed ({exc}); using extractive fallback")
        return None


def _generate_extractive(article: Article) -> str:
    """Build a clean Markdown body from the source summary without an LLM."""
    summary = article.summary.strip()
    authors = ", ".join(article.authors) if article.authors else ""

    if summary:
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        lead = " ".join(sentences[:2]).strip()
        rest = " ".join(sentences[2:]).strip()
    else:
        lead = f"New from {article.source}: {article.title}."
        rest = ""

    parts = [f"> {lead}" if lead else "", ""]
    parts.append("## Overview")
    parts.append(summary or "Full details are available at the original source.")
    if rest:
        parts.append("")
        parts.append("## Details")
        parts.append(rest)
    parts.append("")
    parts.append("## Why it matters")
    parts.append(
        f"This story sits in the **{article.category}** space and is worth "
        "watching as the field evolves. Read the full report at the source "
        "link below for complete context."
    )
    if authors:
        parts.append("")
        parts.append(f"*Reported by {authors} via {article.source}.*")
    return "\n".join(p for p in parts).strip()


def generate_post(article: Article, cfg: GenerationConfig) -> Post:
    """Produce a finished :class:`Post` from a raw article."""
    mode = cfg.resolve_mode()
    body = None
    generated_by = "extractive"

    if mode == "ai":
        body = _generate_ai(article, cfg)
        if body is not None:
            generated_by = "ai"

    if body is None:
        body = _generate_extractive(article)

    # A short plain-text summary for index/meta usage.
    summary = re.sub(r"[#>*`_]", "", article.summary or article.title)
    summary = re.sub(r"\s+", " ", summary).strip()[:200]

    return Post(
        title=article.title,
        slug=slugify(article.title, article.uid),
        category=article.category,
        source=article.source,
        source_url=article.link,
        published=article.published,
        body_md=body,
        summary=summary,
        generated_by=generated_by,
    )
