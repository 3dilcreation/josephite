"""High-level orchestration for Tech SEI: collect -> generate -> publish."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .generator import generate_post
from .publisher import publish
from .sources import collect


def build(config_path: str | Path = "feeds.yaml", limit: int | None = None) -> Path:
    """Run the full blog build and return the output directory."""
    config = Config.load(config_path)
    limit = limit or config.site.posts_per_run

    print(f"\n=== {config.site.title}: building blog ===")
    print(f"Generation mode: {config.generation.resolve_mode()}\n")

    articles = collect(config.categories)
    print(f"\nCollected {len(articles)} unique candidate stories.")

    selected = articles[:limit]
    print(f"Generating {len(selected)} posts...\n")

    posts = []
    for idx, article in enumerate(selected, 1):
        print(f"  [{idx}/{len(selected)}] {article.title[:70]}")
        posts.append(generate_post(article, config.generation))

    out = publish(posts, config)
    print(f"\nDone. Site written to: {out.resolve()}")
    print(f"Open {out.resolve()}/index.html in a browser.\n")
    return out
