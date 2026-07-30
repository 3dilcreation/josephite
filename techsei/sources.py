"""Content sources for Tech SEI.

Fetches candidate stories from RSS/Atom feeds and from the arXiv API. Every
source is normalised into an :class:`Article` so downstream code does not care
where a story came from.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote
from xml.etree import ElementTree

import feedparser
import requests
from dateutil import parser as date_parser

USER_AGENT = "TechSEI-BlogBot/0.1 (+https://github.com/3dilcreation/josephite)"
REQUEST_TIMEOUT = 20


@dataclass
class Article:
    """A raw candidate story pulled from a source."""

    title: str
    link: str
    summary: str
    source: str
    category: str
    published: datetime
    authors: list[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        """Stable identifier used for de-duplication and slugging."""
        return hashlib.sha1(self.link.encode("utf-8")).hexdigest()[:12]


def _clean_html(text: str) -> str:
    """Strip tags/entities from feed summaries so we get readable text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(value: str | None, fallback_struct=None) -> datetime:
    if fallback_struct is not None:
        try:
            return datetime.fromtimestamp(time.mktime(fallback_struct), tz=timezone.utc)
        except (OverflowError, ValueError, TypeError):
            pass
    if value:
        try:
            dt = date_parser.parse(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, OverflowError, TypeError):
            pass
    return datetime.now(timezone.utc)


def fetch_rss(url: str, category: str, limit: int = 15) -> list[Article]:
    """Fetch and normalise entries from a single RSS/Atom feed."""
    articles: list[Article] = []
    try:
        feed = feedparser.parse(url, agent=USER_AGENT)
    except Exception as exc:  # feedparser is broad; never let one feed kill a run
        print(f"  ! failed to parse feed {url}: {exc}")
        return articles

    source_title = feed.feed.get("title", url) if hasattr(feed, "feed") else url
    for entry in feed.entries[:limit]:
        summary = _clean_html(entry.get("summary") or entry.get("description") or "")
        published = _parse_date(
            entry.get("published") or entry.get("updated"),
            entry.get("published_parsed") or entry.get("updated_parsed"),
        )
        authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]
        link = entry.get("link", "")
        if not link or not entry.get("title"):
            continue
        articles.append(
            Article(
                title=_clean_html(entry.get("title", "")),
                link=link,
                summary=summary,
                source=source_title,
                category=category,
                published=published,
                authors=authors,
            )
        )
    return articles


def fetch_arxiv(arxiv_category: str, category: str, limit: int = 10) -> list[Article]:
    """Fetch recent papers from the arXiv API for a subject category."""
    articles: list[Article] = []
    query = (
        "https://export.arxiv.org/api/query?"
        f"search_query=cat:{quote(arxiv_category)}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={limit}"
    )
    try:
        resp = requests.get(
            query, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! failed to query arXiv {arxiv_category}: {exc}")
        return articles

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"  ! could not parse arXiv response for {arxiv_category}: {exc}")
        return articles

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        summary = _clean_html(
            entry.findtext("atom:summary", default="", namespaces=ns) or ""
        )
        published = _parse_date(
            entry.findtext("atom:published", default=None, namespaces=ns)
        )
        authors = [
            a.findtext("atom:name", default="", namespaces=ns).strip()
            for a in entry.findall("atom:author", ns)
        ]
        if not title or not link:
            continue
        articles.append(
            Article(
                title=re.sub(r"\s+", " ", title),
                link=link,
                summary=summary,
                source=f"arXiv ({arxiv_category})",
                category=category,
                published=published,
                authors=[a for a in authors if a],
            )
        )
    return articles


def collect(categories) -> list[Article]:
    """Pull articles from every configured source across all categories.

    Returns a de-duplicated list sorted newest-first.
    """
    seen: set[str] = set()
    collected: list[Article] = []

    for category in categories:
        print(f"Collecting '{category.name}'...")
        for feed in category.feeds:
            if feed.startswith("arxiv:"):
                items = fetch_arxiv(feed.split(":", 1)[1], category.name)
            else:
                items = fetch_rss(feed, category.name)
            print(f"  - {feed}: {len(items)} items")
            for art in items:
                if art.uid in seen:
                    continue
                seen.add(art.uid)
                collected.append(art)

    collected.sort(key=lambda a: a.published, reverse=True)
    return collected
