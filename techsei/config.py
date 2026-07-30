"""Configuration loading for Tech SEI.

Reads ``feeds.yaml`` into lightweight dataclasses so the rest of the codebase
works with typed objects instead of raw dictionaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Category:
    name: str
    slug: str
    description: str
    feeds: list[str] = field(default_factory=list)


@dataclass
class SiteConfig:
    title: str = "Tech SEI"
    tagline: str = ""
    base_url: str = "/"
    author: str = "Tech SEI"
    posts_per_run: int = 8
    output_dir: str = "site"


@dataclass
class GenerationConfig:
    mode: str = "auto"  # ai | extractive | auto
    model: str = "claude-opus-4-8"
    max_tokens: int = 1200

    def resolve_mode(self) -> str:
        """Turn ``auto`` into a concrete mode based on the environment."""
        if self.mode != "auto":
            return self.mode
        return "ai" if os.environ.get("ANTHROPIC_API_KEY") else "extractive"


@dataclass
class Config:
    site: SiteConfig
    generation: GenerationConfig
    categories: list[Category]

    @classmethod
    def load(cls, path: str | Path = "feeds.yaml") -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. "
                "Copy feeds.yaml from the repo root or pass --config."
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        site = SiteConfig(**(raw.get("site") or {}))
        generation = GenerationConfig(**(raw.get("generation") or {}))
        categories = [
            Category(
                name=c["name"],
                slug=c["slug"],
                description=c.get("description", ""),
                feeds=c.get("feeds", []),
            )
            for c in (raw.get("categories") or [])
        ]
        if not categories:
            raise ValueError("No categories defined in config.")
        return cls(site=site, generation=generation, categories=categories)
