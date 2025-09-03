from __future__ import annotations
import time, requests, feedparser
from typing import List, Dict, Optional
import wikipedia

# Google News RSS (works well without auth)
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

def _google_rss(query: str, limit: int = 6) -> List[Dict]:
    q = requests.utils.quote(query)
    url = GOOGLE_NEWS_RSS.format(query=q)
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:limit]:
        out.append({
            "title": e.title,
            "link": e.link,
            "published": getattr(e, "published", ""),
            "source": getattr(getattr(e, "source", {}), "title", ""),
        })
    return out

def google_news(symbol_or_query: str, limit: int = 6) -> List[Dict]:
    return _google_rss(symbol_or_query, limit=limit)

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

# ---- “Filings” best-effort scan via news/RSS ----
# We query Google News constrained to NSE/BSE domains, which often surface
# corporate announcements / notices. (True NSE JSON API usually blocks without cookies.)
def corporate_announcements(symbol: str, limit: int = 5) -> List[Dict]:
    queries = [
        f"site:nseindia.com {symbol} corporate announcement",
        f"site:bseindia.com {symbol} announcement",
        f"{symbol} corporate announcement India",
    ]
    items: List[Dict] = []
    seen = set()
    for q in queries:
        for it in _google_rss(q, limit=limit):
            key = (it.get("title",""), it.get("link",""))
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
    return items[:limit]

def research(symbol_or_query: str) -> Dict:
    q = infer_company_query(symbol_or_query)
    news = google_news(q, limit=6)
    profile = wiki_summary(symbol_or_query, sentences=2)
    filings = corporate_announcements(symbol_or_query, limit=5)
    return {
        "query": q,
        "news": news,
        "filings": filings,   # <— NEW
        "profile": profile,
        "timestamp": int(time.time()),
    }
