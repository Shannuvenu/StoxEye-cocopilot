from __future__ import annotations
from typing import List, Dict

def build_citation_block(news_items: List[Dict]) -> str:
    if not news_items:
        return "_No fresh news sources found._"
    lines = []
    for i, n in enumerate(news_items, 1):
        title = n.get("title", "Untitled")
        link = n.get("link", "#")
        pub = n.get("published", "")
        src = n.get("source", "")
        lines.append(f"{i}. [{title}]({link}) — {src} {('• ' + pub) if pub else ''}")
    return "\n".join(lines)

def build_titled_block(title: str, items: List[Dict]) -> str:
    body = build_citation_block(items)
    return f"**{title}**\n{body}" if items else f"**{title}**\n_No items found._"
