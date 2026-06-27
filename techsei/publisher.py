"""Static site publisher for Tech SEI.

Takes generated :class:`~techsei.generator.Post` objects plus the site config
and writes a complete static website: a home page, per-category pages, and an
individual page per article. Also writes the Markdown source for each post and
an RSS feed for the whole site.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import __version__
from .config import Config
from .generator import Post
from .markdown import render as render_md

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["dateformat"] = lambda d: d.strftime("%B %d, %Y")
    return env


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def publish(posts: list[Post], config: Config) -> Path:
    """Render the full static site and return the output directory."""
    out = Path(config.site.output_dir)
    if out.exists():
        # Preserve previously generated posts; only refresh index/assets here.
        pass
    out.mkdir(parents=True, exist_ok=True)

    env = _env()
    post_tmpl = env.get_template("post.html")
    index_tmpl = env.get_template("index.html")
    category_tmpl = env.get_template("category.html")

    # Copy static assets (CSS).
    assets_src = TEMPLATE_DIR / "static"
    if assets_src.exists():
        shutil.copytree(assets_src, out / "static", dirs_exist_ok=True)

    # Map category display names to their slugs for cross-linking.
    slug_by_name = {c.name: c.slug for c in config.categories}

    rendered = []
    for post in posts:
        body_html = render_md(post.body_md)
        html = post_tmpl.render(
            site=config.site,
            post=post,
            body=body_html,
            category_slug=slug_by_name.get(post.category, ""),
            version=__version__,
        )
        _write(out / "posts" / f"{post.slug}.html", html)
        # Markdown source alongside the HTML for archival / reuse.
        front = (
            f"---\ntitle: {post.title}\ncategory: {post.category}\n"
            f"source: {post.source_url}\ndate: {post.published.isoformat()}\n"
            f"generated_by: {post.generated_by}\n---\n\n"
        )
        _write(out / "posts" / f"{post.slug}.md", front + f"# {post.title}\n\n" + post.body_md + "\n")
        rendered.append(post)

    # Sort newest first for listings.
    rendered.sort(key=lambda p: p.published, reverse=True)

    # Per-category pages.
    for category in config.categories:
        cat_posts = [p for p in rendered if p.category == category.name]
        html = category_tmpl.render(
            site=config.site,
            category=category,
            posts=cat_posts,
            version=__version__,
        )
        _write(out / "category" / f"{category.slug}.html", html)

    # Home page.
    index_html = index_tmpl.render(
        site=config.site,
        categories=config.categories,
        posts=rendered,
        generated_at=datetime.now(timezone.utc),
        version=__version__,
    )
    _write(out / "index.html", index_html)

    # RSS feed.
    _write(out / "feed.xml", _build_rss(rendered, config))

    return out


def _build_rss(posts: list[Post], config: Config) -> str:
    items = []
    for p in posts[:30]:
        link = f"{config.site.base_url.rstrip('/')}/posts/{p.slug}.html"
        items.append(
            "<item>"
            f"<title>{xml_escape(p.title)}</title>"
            f"<link>{xml_escape(link)}</link>"
            f"<guid isPermaLink=\"false\">{p.slug}</guid>"
            f"<category>{xml_escape(p.category)}</category>"
            f"<pubDate>{format_datetime(p.published)}</pubDate>"
            f"<description>{xml_escape(p.summary)}</description>"
            "</item>"
        )
    now = format_datetime(datetime.now(timezone.utc))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"><channel>'
        f"<title>{xml_escape(config.site.title)}</title>"
        f"<link>{xml_escape(config.site.base_url)}</link>"
        f"<description>{xml_escape(config.site.tagline)}</description>"
        f"<lastBuildDate>{now}</lastBuildDate>"
        f"<generator>Tech SEI {__version__}</generator>"
        + "".join(items)
        + "</channel></rss>"
    )
