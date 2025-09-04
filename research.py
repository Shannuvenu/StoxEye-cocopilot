from __future__ import annotations
import time, requests, feedparser
from typing import List, Dict, Optional
import wikipedia
from datetime import datetime, timedelta
import trafilatura

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

def _filter_recent(entries, days: int = 1):
    cutoff = datetime.utcnow() - timedelta(days=days)
    fresh = []
    for e in entries:
        try:
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime(*e.published_parsed[:6])
                if pub >= cutoff:
                    fresh.append(e)
            else:
                fresh.append(e)
        except Exception:
            fresh.append(e)
    return fresh

def _google_rss(query: str, limit: int = 6, days: int = 1) -> List[Dict]:
    q = requests.utils.quote(f"{query} when:{days}d")  # freshness
    url = GOOGLE_NEWS_RSS.format(query=q)
    feed = feedparser.parse(url)
    entries = _filter_recent(feed.entries, days=days)
    out = []
    for e in entries[:limit]:
        out.append({
            "title": e.title,
            "link": e.link,
            "published": getattr(e, "published", ""),
            "source": getattr(getattr(e, "source", {}), "title", ""),
        })
    return out

def google_news(symbol_or_query: str, limit: int = 6, days: int = 1) -> List[Dict]:
    return _google_rss(symbol_or_query, limit=limit, days=days)

def wiki_summary(query: str, sentences: int = 3) -> Optional[str]:
    try:
        wikipedia.set_lang("en")
        page_title = wikipedia.search(query, results=1)
        if not page_title:
            return None
        page = wikipedia.page(page_title[0], auto_suggest=False, redirect=True)
        return wikipedia.summary(page.title, sentences=sentences)
    except Exception:
        return None

def infer_company_query(symbol: str) -> str:
    return f"{symbol} stock India"

def corporate_announcements(symbol: str, limit: int = 5, days: int = 7) -> List[Dict]:
    queries = [
        f"site:nseindia.com {symbol} corporate announcement",
        f"site:bseindia.com {symbol} announcement",
        f"{symbol} corporate announcement India",
    ]
    items: List[Dict] = []
    seen = set()
    for q in queries:
        for it in _google_rss(q, limit=limit, days=days):
            key = (it.get("title",""), it.get("link",""))
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
    return items[:limit]

# -------- article bodies → snippets (stable: trafilatura) --------
def collect_article_snippets(items: List[Dict], max_articles: int = 5, max_chars: int = 1600) -> List[Dict]:
    out: List[Dict] = []
    for it in items[:max_articles]:
        url = it.get("link")
        if not url:
            continue
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            text = trafilatura.extract(downloaded, include_links=False, include_comments=False) if downloaded else ""
            text = (text or "").strip()
            snippet = text[:max_chars] if text else f"(No article body parsed for {it.get('source')})"
            out.append({"source": it.get("source",""), "title": it.get("title",""), "snippet": snippet})
        except Exception:
            out.append({"source": it.get("source",""), "title": it.get("title",""), "snippet": "(Parsing failed)"})
    return out

def research(
    symbol_or_query: str,
    *,
    days: int = 1,
    prefer_symbol: str | None = None,
    max_articles: int = 5,
    max_chars: int = 1600
) -> Dict:
    """Use full user query + optional symbol query; fetch fresh news, filings, and article snippets."""
    q_full = symbol_or_query
    q_symbol = infer_company_query(prefer_symbol) if prefer_symbol else None

    items = []
    seen = set()
    for q in [q_full, q_symbol]:
        if not q:
            continue
        for it in google_news(q, limit=6, days=days):
            key = it.get("link","")
            if key in seen:
                continue
            seen.add(key)
            items.append(it)

    profile = wiki_summary(prefer_symbol or symbol_or_query, sentences=2)
    filings = corporate_announcements(prefer_symbol or symbol_or_query, limit=5, days=max(3, min(days, 7)))
    article_snips = collect_article_snippets(items, max_articles=max_articles, max_chars=max_chars)

    return {
        "query_full": q_full,
        "query_symbol": q_symbol,
        "news": items[:6],
        "filings": filings[:5],
        "articles": article_snips,
        "profile": profile,
        "timestamp": int(time.time()),
    }
