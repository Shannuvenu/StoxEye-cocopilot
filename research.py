from __future__ import annotations

import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import requests

# Optional deps (we'll guard against missing ones)
try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None  # fallback to no RSS if not available

try:
    import wikipedia  # type: ignore
except Exception:
    wikipedia = None

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None


# ---------------------------------
# Constants
# ---------------------------------
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------------------------
# Helpers
# ---------------------------------
def _filter_recent(entries, days: int = 1):
    """Return only entries newer than 'days' (UTC)."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    fresh = []
    for e in entries or []:
        try:
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime(*e.published_parsed[:6])
                if pub >= cutoff:
                    fresh.append(e)
            else:
                # keep if no publish date
                fresh.append(e)
        except Exception:
            fresh.append(e)
    return fresh


def _entry_source(entry) -> str:
    """Try to get a nice source name from a feedparser entry."""
    # Google News usually sets e.source.title
    try:
        if hasattr(entry, "source") and entry.source:
            # feedparser gives .title on source object
            title = getattr(entry.source, "title", "") or ""
            if isinstance(title, (list, tuple)):
                title = title[0] if title else ""
            return str(title).strip()
    except Exception:
        pass
    # fallback: try 'author' or feed title inside entry
    for k in ("author", "source", "feedburner_origlink"):
        v = getattr(entry, k, "") or ""
        if v:
            return str(v).strip()
    return ""


def _google_rss(query: str, limit: int = 6, days: int = 1) -> List[Dict]:
    """Fetch Google News RSS for query with freshness."""
    if not feedparser:
        return []

    q = requests.utils.quote(f"{query} when:{days}d")  # n-day freshness
    url = GOOGLE_NEWS_RSS.format(query=q)
    feed = feedparser.parse(url)
    entries = _filter_recent(getattr(feed, "entries", []), days=days)
    out: List[Dict] = []
    for e in entries[:limit]:
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        published = getattr(e, "published", "") or ""
        src = _entry_source(e)
        if title and link:
            out.append(
                {
                    "title": title.strip(),
                    "url": link.strip(),      # normalized key 'url'
                    "published": published,
                    "source": src,
                }
            )
    return out


def google_news(symbol_or_query: str, limit: int = 6, days: int = 1) -> List[Dict]:
    return _google_rss(symbol_or_query, limit=limit, days=days)


def wiki_summary(query: str, sentences: int = 3) -> Optional[str]:
    """Short company/term summary. Returns None if not available."""
    if not wikipedia:
        return None
    try:
        wikipedia.set_lang("en")
        hits = wikipedia.search(query, results=1)
        if not hits:
            return None
        page = wikipedia.page(hits[0], auto_suggest=False, redirect=True)
        return wikipedia.summary(page.title, sentences=sentences)
    except Exception:
        return None


def infer_company_query(symbol: str) -> str:
    # Tweak this if you want different heuristics
    return f"{symbol} stock India"


def corporate_announcements(symbol: str, limit: int = 5, days: int = 7) -> List[Dict]:
    """Approximate 'corporate announcements' by scoped queries via Google News."""
    queries = [
        f"site:nseindia.com {symbol} corporate announcement",
        f"site:bseindia.com {symbol} announcement",
        f"{symbol} corporate announcement India",
    ]
    items: List[Dict] = []
    seen = set()
    for q in queries:
        for it in _google_rss(q, limit=limit, days=days):
            key = (it.get("title", ""), it.get("url", ""))
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "source": it.get("source", ""),
                    "published": it.get("published", ""),
                }
            )
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break
    return items[:limit]


def _strip_html(text: str) -> str:
    """Very small HTML stripper for fallback."""
    import re as _re

    text = _re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = _re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = _re.sub(r"(?s)<.*?>", " ", text)
    text = _re.sub(r"\s+", " ", text)
    return text.strip()


def _download_text(url: str) -> str:
    """Download article text using trafilatura if available; fallback to plain HTML strip."""
    if not url:
        return ""
    try:
        if trafilatura:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            if downloaded:
                extracted = trafilatura.extract(
                    downloaded, include_links=False, include_comments=False
                )
                if extracted:
                    return extracted.strip()
        # Fallback if trafilatura not present or failed
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=10)
        if resp.ok:
            return _strip_html(resp.text)
    except Exception:
        pass
    return ""


def collect_article_snippets(
    items: List[Dict], max_articles: int = 5, max_chars: int = 1600
) -> List[Dict]:
    """Return articles with source/title/snippet/url - ALWAYS including 'url'."""
    out: List[Dict] = []
    for it in items[:max_articles]:
        url = it.get("url") or it.get("link") or ""
        if not url:
            continue
        try:
            text = _download_text(url)
            snippet = (
                text[:max_chars].strip()
                if text
                else f"(No article body parsed for {it.get('source') or 'source'})"
            )
            out.append(
                {
                    "source": it.get("source", "") or "",
                    "title": it.get("title", "") or "",
                    "snippet": snippet,
                    "url": url,  # ensure url present
                }
            )
        except Exception:
            out.append(
                {
                    "source": it.get("source", "") or "",
                    "title": it.get("title", "") or "",
                    "snippet": "(Parsing failed)",
                    "url": url,
                }
            )
    return out


# ---------------------------------
# Main Entry
# ---------------------------------
def research(
    symbol_or_query: str,
    *,
    days: int = 1,
    prefer_symbol: str | None = None,
    max_articles: int = 5,
    max_chars: int = 1600,
) -> Dict:
    """
    Use full user query + optional symbol query; fetch fresh news, filings, and article snippets.

    Returns dict with keys:
      - query_full, query_symbol, profile
      - news:   list of {title, url, source, published}
      - filings: list of {title, url, source, published}
      - articles: list of {source, title, snippet, url}   <-- used by app for links/summaries
      - timestamp
    """
    q_full = symbol_or_query
    q_symbol = infer_company_query(prefer_symbol) if prefer_symbol else None

    # Collect fresh Google News for both queries (dedupe by URL)
    items: List[Dict] = []
    seen = set()
    for q in [q_full, q_symbol]:
        if not q:
            continue
        for it in google_news(q, limit=8, days=days):
            key = it.get("url", "")
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(it)

    # Optional info chunks
    profile = wiki_summary(prefer_symbol or symbol_or_query, sentences=2)
    filings = corporate_announcements(
        prefer_symbol or symbol_or_query, limit=5, days=max(3, min(days, 7))
    )

    # Download article bodies (snippets)
    article_snips = collect_article_snippets(
        items, max_articles=max_articles, max_chars=max_chars
    )

    return {
        "query_full": q_full,
        "query_symbol": q_symbol,
        "news": items[:8],
        "filings": filings[:5],
        "articles": article_snips,  # used by app; every item has url
        "profile": profile,
        "timestamp": int(time.time()),
    }
