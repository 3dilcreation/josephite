# Tech SEI — Automated Tech Blog Builder

Tech SEI builds a tech blog **automatically**. It pulls current tech news, the
latest innovations, and future research from configurable sources, turns each
story into a blog article, and publishes a clean static website — ready to host
on GitHub Pages or anywhere else.

It runs on a schedule (daily by default) so the blog keeps itself up to date
with zero manual effort.

> _Looking for the original linear-regression demo? See
> [`docs/ml-demo.md`](docs/ml-demo.md)._

---

## Features

- **Multi-source aggregation** — RSS/Atom feeds (Ars Technica, The Verge,
  TechCrunch, Wired, MIT Tech Review, IEEE Spectrum, Hacker News…) plus the
  **arXiv API** for fresh research papers.
- **Three content sections out of the box** — *Tech News*, *Latest
  Innovations*, and *Future Research*. Fully configurable in `feeds.yaml`.
- **AI-written articles (optional)** — with an `ANTHROPIC_API_KEY`, posts are
  written as original, well-structured articles by the Claude API. Without a
  key, it falls back to clean **extractive summaries** — so it always works.
- **Static site generation** — responsive dark-themed HTML, per-category
  pages, individual article pages, Markdown source for every post, and an RSS
  feed for the whole blog.
- **Zero-touch automation** — a GitHub Actions workflow rebuilds and deploys
  the blog daily.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) enable AI-written articles
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Build the blog
python -m techsei build

# 4. Preview it locally (builds, then serves at http://localhost:8000)
python -m techsei serve
```

The generated site is written to `site/` (open `site/index.html`).

### CLI commands

| Command | Description |
| --- | --- |
| `python -m techsei build` | Collect sources, generate posts, publish the site. |
| `python -m techsei build --limit 4` | Cap the number of posts generated. |
| `python -m techsei serve --port 8000` | Build, then preview locally. |
| `python -m techsei sources` | List the configured content sources. |
| `python -m techsei build --config my.yaml` | Use a custom config file. |

---

## Configuration

Everything lives in [`feeds.yaml`](feeds.yaml):

- **`site`** — title, tagline, author, output directory, posts-per-run.
- **`generation`** — `ai`, `extractive`, or `auto` (uses AI when a key is
  present); plus the Claude model and token budget.
- **`categories`** — each becomes a blog section. Add any RSS/Atom URL, or use
  `arxiv:<category>` (e.g. `arxiv:cs.AI`) to pull research papers.

```yaml
categories:
  - name: "Future Research"
    slug: "research"
    description: "Papers and ideas shaping the next decade."
    feeds:
      - arxiv:cs.AI
      - arxiv:cs.LG
```

---

## Automated publishing (GitHub Pages)

The workflow at [`.github/workflows/build-blog.yml`](.github/workflows/build-blog.yml)
rebuilds the blog **daily at 06:00 UTC** (and on demand) and deploys it to
GitHub Pages.

To enable it:

1. Repo **Settings → Pages → Source: GitHub Actions**.
2. _(Optional)_ add an `ANTHROPIC_API_KEY` repository secret to turn on
   AI-written articles.
3. That's it — the blog publishes itself on schedule. Trigger a manual run any
   time from the **Actions** tab ("Build Tech SEI Blog" → *Run workflow*).

---

## How it works

```
feeds.yaml ─▶ sources.py ─▶ generator.py ─▶ publisher.py ─▶ site/
            (RSS + arXiv)   (AI / summary)   (HTML+RSS+MD)
```

| Module | Responsibility |
| --- | --- |
| `techsei/config.py` | Load and validate `feeds.yaml`. |
| `techsei/sources.py` | Fetch & normalise stories from RSS and arXiv. |
| `techsei/generator.py` | Turn a story into a finished post (AI or extractive). |
| `techsei/markdown.py` | Dependency-free Markdown → HTML rendering. |
| `techsei/publisher.py` | Render the static site, RSS feed, and Markdown archive. |
| `techsei/pipeline.py` | Orchestrates collect → generate → publish. |
| `techsei/cli.py` | Command-line interface. |

---

## Notes

- The engine never crashes on a single bad/unreachable feed — it logs and moves
  on, so a flaky source won't break a build.
- AI generation is grounded in the fetched source material and falls back to an
  extractive summary if the API is unavailable.
- `site/` is git-ignored; the published output is produced by CI.
