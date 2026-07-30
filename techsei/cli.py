"""Command-line interface for Tech SEI.

Usage examples
--------------
    python -m techsei build                 # build with defaults from feeds.yaml
    python -m techsei build --limit 4       # generate at most 4 posts
    python -m techsei build --config my.yaml
    python -m techsei serve                 # build, then preview locally
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

from .config import Config
from .pipeline import build


def _cmd_build(args: argparse.Namespace) -> int:
    build(config_path=args.config, limit=args.limit)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    out = build(config_path=args.config, limit=args.limit)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(out)
    )
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Serving {out} at http://localhost:{args.port} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    print(f"{config.site.title} — configured sources\n")
    for cat in config.categories:
        print(f"[{cat.name}] ({cat.slug}) — {len(cat.feeds)} feeds")
        for feed in cat.feeds:
            print(f"   - {feed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="techsei", description="Automated tech blog builder."
    )
    parser.add_argument(
        "--config", default="feeds.yaml", help="Path to config (default: feeds.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Collect, generate, and publish the blog.")
    p_build.add_argument("--limit", type=int, default=None, help="Max posts to generate.")
    p_build.set_defaults(func=_cmd_build)

    p_serve = sub.add_parser("serve", help="Build then preview the site locally.")
    p_serve.add_argument("--limit", type=int, default=None)
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_sources = sub.add_parser("sources", help="List configured content sources.")
    p_sources.set_defaults(func=_cmd_sources)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
